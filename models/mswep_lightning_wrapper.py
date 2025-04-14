import torch
import lightning as L
from torch.nn import functional as F
from torchmetrics import MeanAbsoluteError, MeanSquaredError
import numpy as np
import multiprocessing
from tqdm import tqdm
from models.mswep_evaluation import calculate_crps_idr
from data.mswep_data_module_2 import TargetLogScaler  # Import TargetLogScaler

class UNetLightningModule(L.LightningModule):
    """
    Lightning Module wrapper for any UNet model.
    
    This wrapper handles the UNet model for MSWEP precipitation forecasting.
    The model takes 3 input channels representing precipitation at days t-3, t-2, and t-1,
    and predicts precipitation at day t.
    
    It also integrates a postprocessing step using IDR (EasyUQ) to convert deterministic
    forecasts into calibrated probabilistic forecasts.
    """
    def __init__(self, model, learning_rate=0.0001, loss_type='mse', intensity_weights=None, focal_gamma=2.0,
                 optimizer_type='adam', lr_scheduler_type='cosineannealinglr', 
                 use_regional_focus=True, region_weight=1.0, outside_weight=0.2, log_offset=0.01,
                 weight_decay=1e-3):
        """
        Initialize the UNetLightningModule.
        
        Args:
            model: UNet model instance
            learning_rate: Learning rate for the optimizer (default 0.001)
            loss_type: Type of loss function ('mse', 'weighted_mse', 'huber', 'focal_mse')
            intensity_weights: Optional dictionary with intensity ranges and weights
            focal_gamma: Gamma parameter for focal loss
            optimizer_type: Type of optimizer to use ('adam' or 'adamw'; default 'adam')
            lr_scheduler_type: Type of learning rate scheduler ('cosineannealinglr' or 'cosineannealingwarmrestarts'; default 'cosineannealinglr')
            use_regional_focus: Whether to apply regional weighting to the loss calculation
            region_weight: Weight for target region (Germany)
            outside_weight: Weight for areas outside target region
            log_offset: Offset to use in log transform of precipitation data (default 0.01)
            weight_decay: Weight decay for optimizer (L2 regularization)
        """
        super().__init__()
        
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        
        # Create the target scaler for log transformation
        self.target_scaler = TargetLogScaler(offset=log_offset)
        
        self.loss_type = loss_type
        self.loss_fn = self.configure_loss_function(
            loss_type=loss_type,
            intensity_weights=intensity_weights,
            focal_gamma=focal_gamma
        )
        
        self.mae_metric = MeanAbsoluteError()
        self.rmse_metric = MeanSquaredError()
        
        # Store best validation metrics observed so far
        self.best_val_loss = float('inf')
        self.best_val_mae = None
        self.best_val_rmse = None
        
        # Lists to store predictions and targets across epochs for CRPS calculation
        self.training_step_preds = []
        self.training_step_targets = []
        self.validation_step_preds = []
        self.validation_step_targets = []
    
    def forward(self, x: torch.Tensor, apply_easyuq: bool = False) -> torch.Tensor:
        """
        Forward pass through the UNet model.
        
        Args:
            x: Input tensor of shape [batch, channels, H, W]
            apply_easyuq: Whether to apply EasyUQ (IDR) postprocessing to the output
            
        Returns:
            Tensor of shape [batch, 1, H, W] for precipitation
        """
        # Get model output, passing the EasyUQ flag to the underlying model 
        out = self.model(x, apply_easyuq=apply_easyuq)
        
        # Ensure output has channel dimension
        if out.dim() == 3:  # [batch, H, W]
            out = out.unsqueeze(1)  # Add channel dim -> [batch, 1, H, W]
        
        return out
    
    def training_step(self, batch, batch_idx: int):
        """Training step with gradient clipping and mixed-precision stability."""
        x, y_original, y_for_transform = batch
        
        # Apply log transform to target
        y_scaled = self.target_scaler.transform(y_for_transform)
        
        y_hat = self(x)
        
        # Fix dimension mismatch - squeeze the channel dimension
        y_hat = y_hat.squeeze(1)  # Convert from [B,1,H,W] to [B,H,W]
        
        # Handle potential NaNs in both prediction and transformed target
        mask = ~(torch.isnan(y_scaled) | torch.isnan(y_hat))
        
        if mask.sum() == 0:
            # If all values are NaN, return zero loss but log this issue
            self.log("all_nan_batch", 1.0, prog_bar=True, sync_dist=True)
            # Return small non-zero loss to avoid NaN gradients
            return torch.tensor(0.01, device=self.device, requires_grad=True)
        
        # Compute loss only on non-NaN values
        y_hat_masked = torch.where(mask, y_hat, torch.zeros_like(y_hat))
        y_scaled_masked = torch.where(mask, y_scaled, torch.zeros_like(y_scaled))
        
        # Calculate element-wise loss (squared error for MSE)
        # Use the masked versions to avoid NaNs in calculation where possible
        elementwise_loss = F.mse_loss(y_hat_masked, y_scaled_masked, reduction='none') 

        # Apply regional weighting conditionally
        if self.hparams.use_regional_focus:
            # Check if inner model has the mask attribute
            if hasattr(self.model, 'germany_mask') and self.model.germany_mask is not None:
                weight_mask = self.model.germany_mask.to(elementwise_loss.device)

                # Apply weights to element-wise loss
                # Make sure weight_mask has compatible dimensions for broadcasting (e.g., HxW)
                weighted_loss_elements = elementwise_loss * weight_mask 

                # Calculate final loss: average weighted loss ONLY over valid (non-NaN target) pixels
                # We use the original 'mask' here which identifies valid target pixels
                loss = torch.sum(weighted_loss_elements[mask]) / mask.sum().clamp(min=1) # Safe division

            else:
                print("Warning: use_regional_focus=True but self.model.germany_mask not found. Using unweighted loss.")
                # Fallback to unweighted loss if mask is missing
                loss = torch.mean(elementwise_loss[mask]) # Average only over valid pixels
        else:
            # Standard unweighted loss: average element-wise loss ONLY over valid (non-NaN target) pixels
            loss = torch.mean(elementwise_loss[mask]) # Average only over valid pixels
        
        # Log metrics with distributed training support
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        
        with torch.no_grad():
            # Also log percentage of NaN values for monitoring
            nan_pct = 100 * (~mask).float().mean()
            self.log("nan_percentage", nan_pct, prog_bar=True, sync_dist=True)
            
            # Calculate MAE and RMSE only for Germany region
            if mask.sum() > 0 and self.hparams.use_regional_focus and hasattr(self.model, 'germany_mask'):
                # Get the Germany mask
                weight_mask = self.model.germany_mask.to(mask.device)
                # Get the region_weight value - this will be the exact weight assigned to Germany
                region_weight = self.hparams.region_weight if hasattr(self.hparams, 'region_weight') else 1.0
                # Germany is defined as areas with weight equal to region_weight
                germany_region = (weight_mask == region_weight).to(mask.device)
                
                # Combine with the valid data mask (non-NaN values)
                germany_valid_mask = mask & germany_region.expand_as(mask)
                
                if germany_valid_mask.sum() > 0:
                    # Calculate metrics only for valid points in Germany
                    germany_mae = torch.abs(y_hat_masked[germany_valid_mask] - y_scaled_masked[germany_valid_mask]).mean()
                    germany_mse = ((y_hat_masked[germany_valid_mask] - y_scaled_masked[germany_valid_mask]) ** 2).mean()
                    germany_rmse = torch.sqrt(germany_mse)
                    
                    self.log("train_mae", germany_mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                    self.log("train_rmse", germany_rmse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                else:
                    # If no valid points in Germany, log zero metrics
                    self.log("train_mae", torch.tensor(0.0, device=self.device), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                    self.log("train_rmse", torch.tensor(0.0, device=self.device), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            else:
                # Fallback to calculating metrics on all valid points if Germany mask is unavailable
                mae = self.mae_metric(y_hat_masked[mask], y_scaled_masked[mask])
                mse = self.rmse_metric(y_hat_masked[mask], y_scaled_masked[mask])
                rmse = torch.sqrt(mse)
                self.log("train_mae", mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                self.log("train_rmse", rmse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            
            # Apply inverse transform to get predictions back to original scale
            y_hat_rescaled = self.target_scaler.inverse_transform(y_hat.detach())
        
        # Apply gradient clipping to prevent exploding gradients
        # Important for precipitation data with occasional extremes
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Store predictions and targets for potential later CRPS postprocessing
        self.training_step_preds.append(y_hat_rescaled)
        self.training_step_targets.append(y_original.detach())
        
        return loss
    
    def validation_step(self, batch, batch_idx: int):
        x, y_original, y_for_transform = batch
        
        # Apply log transform to target
        y_scaled = self.target_scaler.transform(y_for_transform)
        
        # Forward pass with memory optimization
        with torch.amp.autocast('cuda', enabled=self.trainer.precision != 32):
            y_hat = self(x)
        
        # Fix dimension mismatch
        y_hat = y_hat.squeeze(1)
        
        # Handle NaNs properly
        mask = ~(torch.isnan(y_scaled) | torch.isnan(y_hat))
        
        if mask.sum() == 0:
            # If all values are NaN, log a warning and assign a poor performance score.
            self.log("val_all_nan", 1.0, prog_bar=True, sync_dist=True)
            # Return high fallback loss to avoid misleading the early stopping callback
            fallback_loss = torch.tensor(100.0, device=self.device, requires_grad=True)
            self.log("val_loss", fallback_loss, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("val_mae", torch.tensor(100.0, device=self.device), prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("val_rmse", torch.tensor(100.0, device=self.device), prog_bar=True, on_epoch=True, sync_dist=True)
            return {"val_loss": fallback_loss}
        
        # Compute masked validation loss
        y_hat_masked = torch.where(mask, y_hat, torch.zeros_like(y_hat))
        y_scaled_masked = torch.where(mask, y_scaled, torch.zeros_like(y_scaled))
        
        # Calculate element-wise loss
        elementwise_loss = F.mse_loss(y_hat_masked, y_scaled_masked, reduction='none')
        
        # Apply regional weighting conditionally (same as in training_step)
        if self.hparams.use_regional_focus:
            if hasattr(self.model, 'germany_mask') and self.model.germany_mask is not None:
                weight_mask = self.model.germany_mask.to(elementwise_loss.device)
                weighted_loss_elements = elementwise_loss * weight_mask
                val_loss = torch.sum(weighted_loss_elements[mask]) / mask.sum().clamp(min=1)
            else:
                # Fallback to unweighted loss if mask is missing
                val_loss = torch.mean(elementwise_loss[mask])
        else:
            # Standard unweighted loss
            val_loss = torch.mean(elementwise_loss[mask])
        
        # Log validation metrics with distributed training support
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        
        with torch.no_grad():
            # Calculate metrics only for Germany region
            if mask.sum() > 0 and self.hparams.use_regional_focus and hasattr(self.model, 'germany_mask'):
                # Get the Germany mask
                weight_mask = self.model.germany_mask.to(mask.device)
                # Get the region_weight value - this will be the exact weight assigned to Germany
                region_weight = self.hparams.region_weight if hasattr(self.hparams, 'region_weight') else 1.0
                # Germany is defined as areas with weight equal to region_weight
                germany_region = (weight_mask == region_weight).to(mask.device)
                
                # Combine with the valid data mask (non-NaN values)
                germany_valid_mask = mask & germany_region.expand_as(mask)
                
                if germany_valid_mask.sum() > 0:
                    # Calculate metrics only for valid points in Germany
                    germany_mae = torch.abs(y_hat_masked[germany_valid_mask] - y_scaled_masked[germany_valid_mask]).mean()
                    germany_mse = ((y_hat_masked[germany_valid_mask] - y_scaled_masked[germany_valid_mask]) ** 2).mean()
                    germany_rmse = torch.sqrt(germany_mse)
                    
                    self.log("val_mae", germany_mae, prog_bar=True, on_epoch=True, sync_dist=True)
                    self.log("val_rmse", germany_rmse, prog_bar=True, on_epoch=True, sync_dist=True)
                else:
                    # If no valid points in Germany, log zero metrics
                    self.log("val_mae", torch.tensor(0.0, device=self.device), prog_bar=True, on_epoch=True, sync_dist=True)
                    self.log("val_rmse", torch.tensor(0.0, device=self.device), prog_bar=True, on_epoch=True, sync_dist=True)
            else:
                # Fallback to calculating metrics on all valid points if Germany mask is unavailable
                val_mae = self.mae_metric(y_hat_masked[mask], y_scaled_masked[mask])
                mse = self.rmse_metric(y_hat_masked[mask], y_scaled_masked[mask])
                val_rmse = torch.sqrt(mse)
                self.log("val_mae", val_mae, prog_bar=True, on_epoch=True, sync_dist=True)
                self.log("val_rmse", val_rmse, prog_bar=True, on_epoch=True, sync_dist=True)
            
            # Apply inverse transform to get predictions back to original scale
            y_hat_rescaled = self.target_scaler.inverse_transform(y_hat.detach())
        
        val_loss = val_loss.detach()
        
        # Store predictions and targets for CRPS postprocessing later
        self.validation_step_preds.append(y_hat_rescaled)
        self.validation_step_targets.append(y_original.detach())
        
        # Help garbage collection
        del y_hat, y_hat_masked, y_scaled_masked, mask
        
        return {"val_loss": val_loss}
    
    def test_step(self, batch, batch_idx: int):
        """
        Test step - applies EasyUQ postprocessing for consistent probabilistic evaluation.
        """
        x, y_original, y_for_transform = batch
        
        # Apply log transform to target
        y_scaled = self.target_scaler.transform(y_for_transform)
        
        # Always apply EasyUQ in test mode for proper probabilistic evaluation
        y_hat = self(x, apply_easyuq=True)
        y_hat = y_hat.squeeze(1)  # Fix dimension mismatch
        
        with torch.no_grad():
            print(f"LOG SCALE - Min: {y_hat.min().item():.4f}, Max: {y_hat.max().item():.4f}, Mean: {y_hat.mean().item():.4f}")
        
        # Handle NaNs for metric calculation
        mask = ~(torch.isnan(y_scaled) | torch.isnan(y_hat))
        
        if not mask.any():
            # If all values are NaN, log nominal values
            self.log("test_all_nan", 1.0, sync_dist=True)
            self.log("test_mae", torch.tensor(1.0, device=self.device), sync_dist=True)
            self.log("test_rmse", torch.tensor(1.0, device=self.device), sync_dist=True)
            return
        
        # Compute masked test metrics
        y_hat_masked = torch.where(mask, y_hat, torch.zeros_like(y_hat))
        y_scaled_masked = torch.where(mask, y_scaled, torch.zeros_like(y_scaled))
        
        # Calculate element-wise loss
        elementwise_loss = F.mse_loss(y_hat_masked, y_scaled_masked, reduction='none')
        
        # Apply regional weighting conditionally (same as in other steps)
        if self.hparams.use_regional_focus:
            if hasattr(self.model, 'germany_mask') and self.model.germany_mask is not None:
                weight_mask = self.model.germany_mask.to(elementwise_loss.device)
                weighted_loss_elements = elementwise_loss * weight_mask
                test_loss = torch.sum(weighted_loss_elements[mask]) / mask.sum().clamp(min=1)
            else:
                # Fallback to unweighted loss if mask is missing
                test_loss = torch.mean(elementwise_loss[mask])
        else:
            # Standard unweighted loss
            test_loss = torch.mean(elementwise_loss[mask])
            
        self.log("test_loss", test_loss, sync_dist=True)
        
        with torch.no_grad():
            # Calculate metrics only for Germany region
            if mask.sum() > 0 and self.hparams.use_regional_focus and hasattr(self.model, 'germany_mask'):
                # Get the Germany mask
                weight_mask = self.model.germany_mask.to(mask.device)
                # Get the region_weight value - this will be the exact weight assigned to Germany
                region_weight = self.hparams.region_weight if hasattr(self.hparams, 'region_weight') else 1.0
                # Germany is defined as areas with weight equal to region_weight
                germany_region = (weight_mask == region_weight).to(mask.device)
                
                # Combine with the valid data mask (non-NaN values)
                germany_valid_mask = mask & germany_region.expand_as(mask)
                
                if germany_valid_mask.sum() > 0:
                    # Calculate metrics only for valid points in Germany
                    germany_mae = torch.abs(y_hat_masked[germany_valid_mask] - y_scaled_masked[germany_valid_mask]).mean()
                    germany_mse = ((y_hat_masked[germany_valid_mask] - y_scaled_masked[germany_valid_mask]) ** 2).mean()
                    germany_rmse = torch.sqrt(germany_mse)
                    
                    self.log("test_mae", germany_mae, sync_dist=True)
                    self.log("test_rmse", germany_rmse, sync_dist=True)
                    
                    print(f"Germany-only metrics - MAE: {germany_mae.item():.4f}, RMSE: {germany_rmse.item():.4f}")
                else:
                    # If no valid points in Germany, log zero metrics
                    self.log("test_mae", torch.tensor(0.0, device=self.device), sync_dist=True)
                    self.log("test_rmse", torch.tensor(0.0, device=self.device), sync_dist=True)
            else:
                # Fallback to calculating metrics on all valid points if Germany mask is unavailable
                test_mae = self.mae_metric(y_hat_masked[mask], y_scaled_masked[mask])
                mse = self.rmse_metric(y_hat_masked[mask], y_scaled_masked[mask])
                test_rmse = torch.sqrt(mse)
                self.log("test_mae", test_mae, sync_dist=True)
                self.log("test_rmse", test_rmse, sync_dist=True)
            
            # Apply inverse transform to get predictions back to original scale
            y_hat_rescaled = self.target_scaler.inverse_transform(y_hat.detach())
            
            # Debug info about original-scale predictions
            print(f"ORIG SCALE - Min: {y_hat_rescaled.min().item():.4f}, Max: {y_hat_rescaled.max().item():.4f}, Mean: {y_hat_rescaled.mean().item():.4f}")
            
            # Ensure predictions are positive and have a reasonable minimum value for visualization
            y_hat_rescaled = torch.clamp(y_hat_rescaled, min=0.0)  # Ensure no negative values
            
            # Ensure there's some signal for visualization
            if y_hat_rescaled.max().item() < 0.1:  
                print("WARNING: Maximum prediction value is very small. Scaling up for visibility.")
                # If predictions are too small to visualize, apply a small scaling
                y_hat_rescaled = y_hat_rescaled * 10.0
                print(f"After scaling - Min: {y_hat_rescaled.min().item():.4f}, Max: {y_hat_rescaled.max().item():.4f}")
        
        # Store sample inputs/outputs for visualization if this is the first batch
        if batch_idx == 0:
            self.sample_inputs = x.detach().cpu().numpy()
            self.sample_targets = y_original.detach().cpu().numpy()
            self.sample_predictions = y_hat_rescaled.detach().cpu().numpy()
    
    def on_validation_epoch_end(self):
        """
        Collects validation predictions and targets, and tracks the best metrics.
        EasyUQ/CRPS calculation is deferred.
        """
        # Access metrics logged during the validation epoch
        current_val_loss = self.trainer.callback_metrics.get('val_loss')
        current_val_mae = self.trainer.callback_metrics.get('val_mae')
        current_val_rmse = self.trainer.callback_metrics.get('val_rmse')

        # Check if val_loss improved
        if current_val_loss is not None and current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss # Update best loss tracked internally
            # Store the corresponding MAE and RMSE from this best epoch
            self.best_val_mae = current_val_mae
            self.best_val_rmse = current_val_rmse
            print(f"\n[Module Best Metrics] New best val_loss: {self.best_val_loss:.4f} -> val_mae: {self.best_val_mae:.4f}, val_rmse: {self.best_val_rmse:.4f}\n")

        # Predictions and targets are already collected in validation_step
        # These will be accessed and saved in the main training script.
        pass

    # Add this method to clear lists at the start of each validation epoch
    def on_validation_epoch_start(self):
        """Clear validation prediction and target lists."""
        if hasattr(self, 'validation_step_preds'):
            self.validation_step_preds.clear()
        if hasattr(self, 'validation_step_targets'):
            self.validation_step_targets.clear()
        print("Cleared validation_step_preds and validation_step_targets for new epoch.") # Optional: for confirmation

    def configure_optimizers(self):
        """Configure optimizers with learning rate scheduling and regularization."""
        weight_decay = self.hparams.weight_decay if hasattr(self.hparams, 'weight_decay') else 1e-3
        
        if self.hparams.optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=weight_decay
            )
        elif self.hparams.optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.hparams.optimizer_type}")
        
        if self.hparams.lr_scheduler_type.lower() == "cosineannealingwarmrestarts":
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, 
                    T_0=10,  # Restart every 10 epochs
                    T_mult=1, 
                    eta_min=1e-6
                ),
                'interval': 'epoch',
                'frequency': 1,
                'name': 'cosine_lr'
            }
        elif self.hparams.lr_scheduler_type.lower() == "reducelronplateau":
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.5,  # Multiply LR by this factor when plateauing
                    patience=5,   # Number of epochs with no improvement after which LR will be reduced
                    min_lr=1e-6,  # Lower bound on the learning rate
                    verbose=True
                ),
                'monitor': 'val_loss',  # Metric to monitor
                'interval': 'epoch',
                'frequency': 1,
                'name': 'plateau_lr'
            }
        elif self.hparams.lr_scheduler_type.lower() == "cosineannealinglr":
            scheduler_instance = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if hasattr(self.trainer, 'max_epochs') else 
                       (self.hparams.max_epochs if hasattr(self.hparams, 'max_epochs') else 10),
                eta_min=1e-6
            )
            scheduler = {"scheduler": scheduler_instance, "interval": "epoch"}
        else:
            raise ValueError(f"Unknown lr scheduler type: {self.hparams.lr_scheduler_type}")
        
        return [optimizer], [scheduler]

    def configure_loss_function(self, loss_type='mse', intensity_weights=None, focal_gamma=2.0):
        """
        Configure the loss function for precipitation forecasting.
        
        Args:
            loss_type: Type of loss function ('mse', 'weighted_mse', 'huber', 'focal_mse', 'asymmetric_mse')
            intensity_weights: Optional weights for different precipitation intensities
            focal_gamma: Gamma parameter for focal loss (higher = more focus on hard examples)
            
        Returns:
            The configured loss function
        """
        self.loss_type = loss_type
        
        def mse_loss(pred, target):
            return F.mse_loss(pred, target)
        
        def weighted_mse_loss(pred, target):
            # Create masks for different precipitation intensities
            if intensity_weights is None:
                # Default weights give higher importance to heavier precipitation
                weights = {
                    0.0: 1.0,    # No precipitation
                    0.1: 2.0,    # Light precipitation (>0.1mm)
                    1.0: 5.0,    # Moderate precipitation (>1mm)
                    5.0: 10.0,   # Heavy precipitation (>5mm)
                    20.0: 20.0   # Extreme precipitation (>20mm)
                }
            else:
                weights = intensity_weights
            
            # Initialize weight tensor with base weight
            weight_mask = torch.ones_like(target) * weights[0.0]
            
            # Apply weights based on precipitation intensity
            thresholds = sorted(weights.keys())
            for i in range(1, len(thresholds)):
                threshold = thresholds[i]
                mask = target >= threshold
                weight_mask[mask] = weights[threshold]
                
            # Calculate weighted MSE
            squared_error = (pred - target) ** 2
            return torch.mean(weight_mask * squared_error)
        
        def huber_loss(pred, target, delta=1.0):
            # Huber loss: MSE for small errors, MAE for large errors
            abs_error = torch.abs(pred - target)
            quadratic = torch.min(abs_error, torch.tensor(delta, device=abs_error.device))
            linear = abs_error - quadratic
            return torch.mean(0.5 * quadratic ** 2 + delta * linear)
        
        def focal_mse_loss(pred, target, gamma=focal_gamma):
            # Focal MSE: gives more weight to examples with higher error
            squared_error = (pred - target) ** 2
            error_weight = (squared_error + 1e-8) ** (gamma / 2.0)
            return torch.mean(error_weight * squared_error)
        
        def asymmetric_mse_loss(pred, target, beta=2.0):
            """
            Asymmetric MSE loss for precipitation: penalizes underestimation more than overestimation.
            The beta parameter controls the degree of asymmetry (higher = more penalty for underestimation).
            
            This is scientifically motivated by the fact that missing extreme precipitation events 
            (underestimation) is typically more problematic than false alarms (overestimation).
            """
            diff = pred - target
            # Create two masks: one for underestimation, one for overestimation
            under_mask = diff < 0
            over_mask = diff >= 0
            
            # Apply different weights based on direction of error
            squared_error = diff ** 2
            asymmetric_error = torch.zeros_like(squared_error)
            asymmetric_error[under_mask] = beta * squared_error[under_mask]  # Higher weight for underestimation
            asymmetric_error[over_mask] = squared_error[over_mask]           # Normal weight for overestimation
            
            return torch.mean(asymmetric_error)
        
        def log_cosh_loss(pred, target):
            """
            Log-cosh loss: a smoothed version of Huber loss that's differentiable everywhere.
            Good for precipitation as it's less sensitive to outliers than MSE but still differentiable.
            """
            diff = pred - target
            return torch.mean(torch.log(torch.cosh(diff)))
        
        # Assign the selected loss function
        if loss_type == 'mse':
            loss_fn = mse_loss
        elif loss_type == 'weighted_mse':
            loss_fn = weighted_mse_loss
        elif loss_type == 'huber':
            loss_fn = huber_loss
        elif loss_type == 'focal_mse':
            loss_fn = focal_mse_loss
        elif loss_type == 'asymmetric_mse':
            loss_fn = asymmetric_mse_loss
        elif loss_type == 'log_cosh':
            loss_fn = log_cosh_loss
        else:
            print(f"Warning: Unknown loss type '{loss_type}'. Defaulting to MSE.")
            loss_fn = mse_loss
            
        self.loss_fn = loss_fn
        return loss_fn 
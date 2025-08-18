import torch
import lightning as L
from torch.nn import functional as F
from torchmetrics import MeanAbsoluteError, MeanSquaredError
import numpy as np
import multiprocessing
from tqdm import tqdm
from models.mswep_evaluation import calculate_crps_idr
from data.mswep_data_module_2 import TargetLogScaler  # Import TargetLogScaler
# Note: MSWEPUNet import would be: from models.mswep_unet import MSWEPUNet

class UNetLightningModule(L.LightningModule):
    """
    Lightning Module wrapper for any UNet model.
    
    This wrapper handles the UNet model for MSWEP precipitation forecasting.
    The model supports dynamic input channels:
    - Default: 5 channels (3 precipitation lags + 2 seasonality features)
    - With ERA5: 5 base channels + 3*M ERA5 channels (M = number of ERA5 variables)
    
    Example usage:
        # Without ERA5
        model = MSWEPUNet(in_channels=5)
        lightning_module = UNetLightningModule(model, ...)
        
        # With ERA5 (e.g., 6 variables)
        model = MSWEPUNet(in_channels=23)  # 5 + 3*6 = 23
        lightning_module = UNetLightningModule(model, ...)
    
    It also integrates a postprocessing step using IDR (EasyUQ) to convert deterministic
    forecasts into calibrated probabilistic forecasts.
    """
    def __init__(self, model, learning_rate=0.0001, loss_type='mse', intensity_weights=None, focal_gamma=2.0,
                 optimizer_type='adam', lr_scheduler_type='cosineannealinglr', 
                 use_regional_focus=True, region_weight=1.0, outside_weight=0.2, 
                 target_scaler=None, weight_decay=1e-3):
        """
        Initialize the UNetLightningModule.
        
        Args:
            model: UNet model instance
            learning_rate: Learning rate for the optimizer (default 0.001)
            loss_type: Type of loss function ('mse', 'weighted_mse', 'huber', 'focal_mse')
            intensity_weights: Optional dictionary with intensity ranges and weights
            focal_gamma: Gamma parameter for focal loss
            optimizer_type: Type of optimizer to use ('adam' or 'adamw'; default 'adam')
            lr_scheduler_type: Type of learning rate scheduler ('cosineannealinglr', 'cosineannealingwarmrestarts', 'reducelronplateau', 'doubledescent', 'constant'; default 'doubledescent')
            use_regional_focus: Whether to apply regional weighting to the loss calculation
            region_weight: Weight for target region (Germany)
            outside_weight: Weight for areas outside target region
            target_scaler: TargetLogScaler instance from DataModule (replaces log_offset parameter)
            weight_decay: Weight decay for optimizer (L2 regularization)
        """
        super().__init__()
        
        self.save_hyperparameters(ignore=['model', 'target_scaler'])
        
        self.model = model
        
        # Use the target scaler from DataModule instead of creating a new one
        self.target_scaler = target_scaler
        if self.target_scaler is not None:
            print(f"UNetLightningModule: Using provided TargetLogScaler with offset={self.target_scaler.offset}")
        else:
            print("UNetLightningModule: No target scaler provided - working with original scale only")
        
        self.loss_type = loss_type
        self.loss_fn = self.configure_loss_function(
            loss_type=loss_type,
            intensity_weights=intensity_weights,
            focal_gamma=focal_gamma
        )
        
        self.mae_metric = MeanAbsoluteError()
        self.rmse_metric = MeanSquaredError()
        
        # Store best validation metrics observed so far (both scales)
        self.best_val_loss = float('inf')
        self.best_val_mae_log = None      # MAE in log scale (for training monitoring)
        self.best_val_rmse_log = None     # RMSE in log scale
        self.best_val_mae_mm = None       # MAE in original scale (for real-world interpretation)
        self.best_val_rmse_mm = None      # RMSE in original scale
        
        # Lists to store validation predictions and targets for CRPS calculation
        # No longer accumulating training predictions - we'll collect those after training
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
        x, y_original, y_transformed = batch
        
        # y_transformed is already log-transformed (if enabled) from DataModule
        # y_original is always in original scale for evaluation
        
        y_hat = self(x)
        
        # Fix dimension mismatch - squeeze the channel dimension
        y_hat = y_hat.squeeze(1)  # Convert from [B,1,H,W] to [B,H,W]
        
        # Handle potential NaNs in both prediction and transformed target
        mask = ~(torch.isnan(y_transformed) | torch.isnan(y_hat))
        
        if mask.sum() == 0:
            # If all values are NaN, return zero loss but log this issue
            self.log("all_nan_batch", 1.0, prog_bar=True, sync_dist=True)
            # Return small non-zero loss to avoid NaN gradients
            return torch.tensor(0.01, device=self.device, requires_grad=True)
        
        # Compute loss only on non-NaN values using transformed targets
        y_hat_masked = torch.where(mask, y_hat, torch.zeros_like(y_hat))
        y_transformed_masked = torch.where(mask, y_transformed, torch.zeros_like(y_transformed))
        
        # Calculate element-wise loss based on loss type
        # Use the masked versions to avoid NaNs in calculation where possible
        if self.loss_type == 'mae':
            elementwise_loss = F.l1_loss(y_hat_masked, y_transformed_masked, reduction='none')
        else:
            # For MSE and MSE-based losses (weighted_mse, focal_mse, etc.)
            elementwise_loss = F.mse_loss(y_hat_masked, y_transformed_masked, reduction='none')

        # Apply regional weighting conditionally
        if self.hparams.use_regional_focus:
            # Check if inner model has the mask attribute
            if hasattr(self.model, 'spatial_weight_mask') and self.model.spatial_weight_mask is not None:
                weight_mask = self.model.spatial_weight_mask.to(elementwise_loss.device)

                # Apply weights to element-wise loss
                # Make sure weight_mask has compatible dimensions for broadcasting (e.g., HxW)
                weighted_loss_elements = elementwise_loss * weight_mask 

                # Calculate final loss: average weighted loss ONLY over valid (non-NaN target) pixels
                # We use the original 'mask' here which identifies valid target pixels
                loss = torch.sum(weighted_loss_elements[mask]) / mask.sum().clamp(min=1) # Safe division

            else:
                print("Warning: use_regional_focus=True but self.model.spatial_weight_mask not found. Using unweighted loss.")
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
            
            # Calculate metrics for Germany region in BOTH scales
            if mask.sum() > 0 and self.hparams.use_regional_focus and hasattr(self.model, 'germany_mask'):
                # Get the Germany mask
                germany_mask = self.model.germany_mask.to(mask.device)
                
                # Combine with the valid data mask (non-NaN values)
                germany_valid_mask = mask & germany_mask.expand_as(mask)
                
                if germany_valid_mask.sum() > 0:
                    # Calculate LOG SCALE metrics (used for loss computation)
                    germany_mae_log = torch.abs(y_hat_masked[germany_valid_mask] - y_transformed_masked[germany_valid_mask]).mean()
                    germany_mse_log = ((y_hat_masked[germany_valid_mask] - y_transformed_masked[germany_valid_mask]) ** 2).mean()
                    germany_rmse_log = torch.sqrt(germany_mse_log)
                    
                    self.log("train_mae_log", germany_mae_log, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                    self.log("train_rmse_log", germany_rmse_log, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                    
                    # Calculate ORIGINAL SCALE metrics (for real-world interpretation)
                    if self.target_scaler is not None:
                        y_hat_orig = self.target_scaler.inverse_transform(y_hat_masked[germany_valid_mask])
                    else:
                        y_hat_orig = y_hat_masked[germany_valid_mask]
                    
                    y_original_masked = y_original[germany_valid_mask]
                    germany_mae_mm = torch.abs(y_hat_orig - y_original_masked).mean()
                    germany_mse_mm = ((y_hat_orig - y_original_masked) ** 2).mean()
                    germany_rmse_mm = torch.sqrt(germany_mse_mm)
                    
                    self.log("train_mae_mm", germany_mae_mm, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                    self.log("train_rmse_mm", germany_rmse_mm, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                    
                else:
                    # If no valid points in Germany, log zero metrics
                    for metric_suffix in ["_log", "_mm"]:
                        self.log(f"train_mae{metric_suffix}", torch.tensor(0.0, device=self.device), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                        self.log(f"train_rmse{metric_suffix}", torch.tensor(0.0, device=self.device), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            else:
                # Fallback to calculating metrics on all valid points if Germany mask is unavailable
                # LOG SCALE metrics
                mae_log = self.mae_metric(y_hat_masked[mask], y_transformed_masked[mask])
                mse_log = self.rmse_metric(y_hat_masked[mask], y_transformed_masked[mask])
                rmse_log = torch.sqrt(mse_log)
                self.log("train_mae_log", mae_log, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                self.log("train_rmse_log", rmse_log, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                
                # ORIGINAL SCALE metrics
                if self.target_scaler is not None:
                    y_hat_orig_all = self.target_scaler.inverse_transform(y_hat_masked[mask])
                else:
                    y_hat_orig_all = y_hat_masked[mask]
                
                mae_mm = torch.abs(y_hat_orig_all - y_original[mask]).mean()
                mse_mm = ((y_hat_orig_all - y_original[mask]) ** 2).mean()
                rmse_mm = torch.sqrt(mse_mm)
                self.log("train_mae_mm", mae_mm, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                self.log("train_rmse_mm", rmse_mm, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        # Apply gradient clipping to prevent exploding gradients
        # Important for precipitation data with occasional extremes
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        return loss
    
    def validation_step(self, batch, batch_idx: int):
        x, y_original, y_transformed = batch
        
        # y_transformed is already log-transformed (if enabled) from DataModule
        # y_original is always in original scale for evaluation
        
        # Forward pass with memory optimization
        with torch.amp.autocast('cuda', enabled=self.trainer.precision != 32):
            y_hat = self(x)
        
        # Fix dimension mismatch
        y_hat = y_hat.squeeze(1)
        
        # Handle NaNs properly
        mask = ~(torch.isnan(y_transformed) | torch.isnan(y_hat))
        
        if mask.sum() == 0:
            # If all values are NaN, log a warning and assign a poor performance score.
            self.log("val_all_nan", 1.0, prog_bar=True, sync_dist=True)
            # Return high fallback loss to avoid misleading the early stopping callback
            fallback_loss = torch.tensor(100.0, device=self.device, requires_grad=True)
            self.log("val_loss", fallback_loss, prog_bar=True, on_epoch=True, sync_dist=True)
            for metric_suffix in ["_log", "_mm"]:
                self.log(f"val_mae{metric_suffix}", torch.tensor(100.0, device=self.device), prog_bar=True, on_epoch=True, sync_dist=True)
                self.log(f"val_rmse{metric_suffix}", torch.tensor(100.0, device=self.device), prog_bar=True, on_epoch=True, sync_dist=True)
            return {"val_loss": fallback_loss}
        
        # Compute masked validation loss using transformed targets
        y_hat_masked = torch.where(mask, y_hat, torch.zeros_like(y_hat))
        y_transformed_masked = torch.where(mask, y_transformed, torch.zeros_like(y_transformed))
        
        # Calculate element-wise loss based on loss type
        if self.loss_type == 'mae':
            elementwise_loss = F.l1_loss(y_hat_masked, y_transformed_masked, reduction='none')
        else:
            # For MSE and MSE-based losses
            elementwise_loss = F.mse_loss(y_hat_masked, y_transformed_masked, reduction='none')
        
        # Apply regional weighting conditionally (same as in training_step)
        if self.hparams.use_regional_focus:
            if hasattr(self.model, 'spatial_weight_mask') and self.model.spatial_weight_mask is not None:
                weight_mask = self.model.spatial_weight_mask.to(elementwise_loss.device)
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
            # Calculate metrics for Germany region in BOTH scales
            if mask.sum() > 0 and self.hparams.use_regional_focus and hasattr(self.model, 'germany_mask'):
                # Get the Germany mask
                germany_mask = self.model.germany_mask.to(mask.device)
                
                # Combine with the valid data mask (non-NaN values)
                germany_valid_mask = mask & germany_mask.expand_as(mask)
                
                if germany_valid_mask.sum() > 0:
                    # Calculate LOG SCALE metrics (consistent with loss)
                    germany_mae_log = torch.abs(y_hat_masked[germany_valid_mask] - y_transformed_masked[germany_valid_mask]).mean()
                    germany_mse_log = ((y_hat_masked[germany_valid_mask] - y_transformed_masked[germany_valid_mask]) ** 2).mean()
                    germany_rmse_log = torch.sqrt(germany_mse_log)
                    
                    self.log("val_mae_log", germany_mae_log, prog_bar=True, on_epoch=True, sync_dist=True)
                    self.log("val_rmse_log", germany_rmse_log, prog_bar=True, on_epoch=True, sync_dist=True)
                    
                    # Calculate ORIGINAL SCALE metrics (for real-world interpretation)
                    if self.target_scaler is not None:
                        y_hat_orig = self.target_scaler.inverse_transform(y_hat_masked[germany_valid_mask])
                    else:
                        y_hat_orig = y_hat_masked[germany_valid_mask]
                    
                    y_original_masked = y_original[germany_valid_mask]
                    germany_mae_mm = torch.abs(y_hat_orig - y_original_masked).mean()
                    germany_mse_mm = ((y_hat_orig - y_original_masked) ** 2).mean()
                    germany_rmse_mm = torch.sqrt(germany_mse_mm)
                    
                    self.log("val_mae_mm", germany_mae_mm, prog_bar=True, on_epoch=True, sync_dist=True)
                    self.log("val_rmse_mm", germany_rmse_mm, prog_bar=True, on_epoch=True, sync_dist=True)
                    
                else:
                    # If no valid points in Germany, log zero metrics
                    for metric_suffix in ["_log", "_mm"]:
                        self.log(f"val_mae{metric_suffix}", torch.tensor(0.0, device=self.device), prog_bar=True, on_epoch=True, sync_dist=True)
                        self.log(f"val_rmse{metric_suffix}", torch.tensor(0.0, device=self.device), prog_bar=True, on_epoch=True, sync_dist=True)
            else:
                # Fallback to calculating metrics on all valid points if Germany mask is unavailable
                # LOG SCALE metrics
                val_mae_log = self.mae_metric(y_hat_masked[mask], y_transformed_masked[mask])
                mse_log = self.rmse_metric(y_hat_masked[mask], y_transformed_masked[mask])
                val_rmse_log = torch.sqrt(mse_log)
                self.log("val_mae_log", val_mae_log, prog_bar=True, on_epoch=True, sync_dist=True)
                self.log("val_rmse_log", val_rmse_log, prog_bar=True, on_epoch=True, sync_dist=True)
                
                # ORIGINAL SCALE metrics
                if self.target_scaler is not None:
                    y_hat_orig_all = self.target_scaler.inverse_transform(y_hat_masked[mask])
                else:
                    y_hat_orig_all = y_hat_masked[mask]
                
                val_mae_mm = torch.abs(y_hat_orig_all - y_original[mask]).mean()
                mse_mm = ((y_hat_orig_all - y_original[mask]) ** 2).mean()
                val_rmse_mm = torch.sqrt(mse_mm)
                self.log("val_mae_mm", val_mae_mm, prog_bar=True, on_epoch=True, sync_dist=True)
                self.log("val_rmse_mm", val_rmse_mm, prog_bar=True, on_epoch=True, sync_dist=True)
            
            # Apply inverse transform to get predictions back to original scale for saving
            if self.target_scaler is not None:
                y_hat_rescaled = self.target_scaler.inverse_transform(y_hat.detach())
            else:
                y_hat_rescaled = y_hat.detach()
        
        val_loss = val_loss.detach()
        
        # Store predictions and targets for CRPS postprocessing later (always in original scale)
        self.validation_step_preds.append(y_hat_rescaled)
        self.validation_step_targets.append(y_original.detach())
        
        # Help garbage collection
        del y_hat, y_hat_masked, y_transformed_masked, mask
        
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
        
        # Calculate element-wise loss based on loss type
        if self.loss_type == 'mae':
            elementwise_loss = F.l1_loss(y_hat_masked, y_scaled_masked, reduction='none')
        else:
            # For MSE and MSE-based losses
            elementwise_loss = F.mse_loss(y_hat_masked, y_scaled_masked, reduction='none')
        
        # Apply regional weighting conditionally (same as in other steps)
        if self.hparams.use_regional_focus:
            if hasattr(self.model, 'spatial_weight_mask') and self.model.spatial_weight_mask is not None:
                weight_mask = self.model.spatial_weight_mask.to(elementwise_loss.device)
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
                # Get the Germany mask (boolean)
                germany_mask = self.model.germany_mask.to(mask.device)
                
                # Combine with the valid data mask (non-NaN values)
                germany_valid_mask = mask & germany_mask.expand_as(mask)
                
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
        """Track best validation metrics at the end of each validation epoch."""
        # Get current validation metrics from logs
        logs = self.trainer.callback_metrics
        current_val_loss = logs.get("val_loss", float('inf'))
        
        # Track metrics in both scales
        current_val_mae_log = logs.get("val_mae_log", float('inf'))
        current_val_rmse_log = logs.get("val_rmse_log", float('inf'))
        current_val_mae_mm = logs.get("val_mae_mm", float('inf'))
        current_val_rmse_mm = logs.get("val_rmse_mm", float('inf'))
        
        # Update best metrics when validation loss improves
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            # Store the corresponding metrics from this best epoch (both scales)
            self.best_val_mae_log = current_val_mae_log
            self.best_val_rmse_log = current_val_rmse_log
            self.best_val_mae_mm = current_val_mae_mm
            self.best_val_rmse_mm = current_val_rmse_mm
            
            print(f"\n[Module Best Metrics] New best val_loss: {self.best_val_loss:.4f}")
            print(f"  -> Log scale: MAE={self.best_val_mae_log:.4f}, RMSE={self.best_val_rmse_log:.4f}")
            print(f"  -> Original scale (mm): MAE={self.best_val_mae_mm:.4f}, RMSE={self.best_val_rmse_mm:.4f}\n")

        # Predictions and targets are already collected in validation_step
        # No need to process them here again

    # Add this method to clear lists at the start of each validation epoch
    def on_validation_epoch_start(self):
        """Clear validation prediction and target lists."""
        if hasattr(self, 'validation_step_preds'):
            self.validation_step_preds.clear()
        if hasattr(self, 'validation_step_targets'):
            self.validation_step_targets.clear()
        print("Cleared validation_step_preds and validation_step_targets for new epoch.") # Optional: for confirmation

    def configure_optimizers(self):
        """Configure optimizers with learning rate scheduling optimized for double descent."""
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
        
        # Get max epochs for scheduler configuration
        max_epochs = 200  # Default for double descent
        if hasattr(self.trainer, 'max_epochs') and self.trainer.max_epochs:
            max_epochs = self.trainer.max_epochs
        elif hasattr(self.hparams, 'max_epochs') and self.hparams.max_epochs:
            max_epochs = self.hparams.max_epochs
        
        if self.hparams.lr_scheduler_type.lower() == "cosineannealingwarmrestarts":
            # Optimized for double descent: fewer, longer cycles
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, 
                    T_0=50,  # Longer initial cycle (50 epochs) for double descent
                    T_mult=2,  # Double the cycle length each restart
                    eta_min=1e-7  # Lower minimum for better exploration
                ),
                'interval': 'epoch',
                'frequency': 1,
                'name': 'cosine_warmrestart_dd'
            }
        elif self.hparams.lr_scheduler_type.lower() == "reducelronplateau":
            # Very conservative for double descent - avoid premature LR reduction
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.7,  # Less aggressive reduction for double descent
                    patience=50,   # Very high patience for double descent (was 30)
                    min_lr=1e-7,  # Lower minimum for continued learning
                    threshold=1e-4,  # More stringent improvement threshold
                    verbose=True
                ),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
                'name': 'plateau_dd'
            }
        elif self.hparams.lr_scheduler_type.lower() == "cosineannealinglr":
            # Standard cosine annealing optimized for double descent
            scheduler_instance = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
                eta_min=1e-7  # Lower minimum for better late-stage learning
            )
            scheduler = {"scheduler": scheduler_instance, "interval": "epoch", "name": "cosine_dd"}
        elif self.hparams.lr_scheduler_type.lower() == "doubledescent":
            # Custom scheduler designed specifically for double descent
            def double_descent_lambda(epoch):
                """
                Custom learning rate schedule for double descent:
                - Phase 1 (0-60): Gradual decay to encourage first descent
                - Phase 2 (60-120): Maintain moderate LR through overfitting valley
                - Phase 3 (120-200): Gradual decay to enable second descent
                """
                if epoch < 60:
                    # First descent phase: moderate decay
                    return 0.5 * (1 + torch.cos(torch.tensor(epoch / 60 * 3.14159))).item()
                elif epoch < 120:
                    # Overfitting valley: maintain higher LR for exploration
                    return 0.3 + 0.2 * (1 + torch.cos(torch.tensor((epoch - 60) / 60 * 3.14159))).item()
                else:
                    # Second descent phase: gentle decay
                    progress = (epoch - 120) / 80
                    return 0.3 * (1 + torch.cos(torch.tensor(progress * 3.14159))).item()
            
            scheduler_instance = torch.optim.lr_scheduler.LambdaLR(optimizer, double_descent_lambda)
            scheduler = {"scheduler": scheduler_instance, "interval": "epoch", "name": "double_descent"}
        elif self.hparams.lr_scheduler_type.lower() == "constant":
            # Constant learning rate - sometimes optimal for double descent
            scheduler = {"scheduler": torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0), 
                        "interval": "epoch", "name": "constant"}
        else:
            raise ValueError(f"Unknown lr scheduler type: {self.hparams.lr_scheduler_type}")
        
        return [optimizer], [scheduler]

    def configure_loss_function(self, loss_type='mse', intensity_weights=None, focal_gamma=2.0):
        """
        Configure the loss function for precipitation forecasting.
        
        Args:
            loss_type: Type of loss function ('mse', 'mae', 'weighted_mse', 'huber', 'focal_mse', 'asymmetric_mse')
            intensity_weights: Optional weights for different precipitation intensities
            focal_gamma: Gamma parameter for focal loss (higher = more focus on hard examples)
            
        Returns:
            The configured loss function
        """
        self.loss_type = loss_type
        
        def mse_loss(pred, target):
            return F.mse_loss(pred, target)
        
        def mae_loss(pred, target):
            """
            Mean Absolute Error loss.
            MAE is computed in log space (same as MSE) when log transform is enabled.
            This provides a more balanced treatment of relative errors across precipitation ranges.
            """
            return F.l1_loss(pred, target)
        
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
        elif loss_type == 'mae':
            loss_fn = mae_loss
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
#!/usr/bin/env python
"""
Train MSWEP UNet models using a time-series approach.
Predictors are based on MSWEP precipitation at days t-3, t-2, t-1,
and the target is the current day t precipitation.
Grid size is 41 x 121.
Example usage:
   python precipitation/training/mswep_unet_training.py --data_dir /path/to/mswep/data
"""
import os
import sys
import argparse
import torch
import numpy as np
import json
import traceback
import time
import distutils.util
from pathlib import Path
import multiprocessing as mp
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import lightning as L
import xarray as xr
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, ModelSummary, RichModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from datetime import datetime, timedelta

# Set matplotlib backend to 'Agg' (non-interactive) to prevent Tkinter threading issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torchmetrics import MeanAbsoluteError, MeanSquaredError
import torch.nn.functional as F

# Add the project root directory to Python path to enable imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


# Enable tensor cores for better performance
torch.set_float32_matmul_precision('high')

from data.mswep_data_module_2 import MSWEPDataModule, TargetLogScaler
from models.mswep_lightning_wrapper import UNetLightningModule
from models.mswep_unet import MSWEPUNet

# Check if isodisreg is available for EasyUQ
try:
    from isodisreg import idr
    ISODISREG_AVAILABLE = True
    
except ImportError:
    ISODISREG_AVAILABLE = False
    print("Warning: isodisreg module not available. EasyUQ postprocessing will be skipped.")

# Timer callback to monitor training performance
class TimerCallback(L.Callback):
    """
    Lightning callback to track timing metrics during training.
    
    This callback measures and logs:
    - Time per step (training batch)
    - Time per epoch
    - Total training time
    - Time per validation step
    - Total validation time per epoch
    - GPU memory usage and utilization (if available)
    
    All timing metrics are automatically logged to TensorBoard.
    """
    def __init__(self):
        super().__init__()
        self.train_start_time = None
        self.train_epoch_start_time = None
        self.train_batch_start_time = None
        self.val_start_time = None
        self.val_batch_start_time = None
        self.train_batch_times = []
        self.val_batch_times = []
        self.epoch_times = []
        self.has_logged_gpu_info = False
        
    def on_train_start(self, trainer, pl_module):
        self.train_start_time = time.time()
        print(f"[Timer] Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if torch.cuda.is_available() and not self.has_logged_gpu_info:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                
                print(f"[GPU] Device: {gpu_name}")
                print(f"[GPU] Initial memory allocated: {gpu_mem_allocated:.2f} GB")
                print(f"[GPU] Initial memory reserved: {gpu_mem_reserved:.2f} GB")
                
                trainer.logger.experiment.add_text(
                    "gpu_info",
                    f"Device: {gpu_name}<br>"
                    f"CUDA: {torch.version.cuda}<br>"
                    f"Tensor Cores: {'Enabled' if torch.get_float32_matmul_precision() != 'highest' else 'Disabled'}<br>"
                    f"cuDNN Benchmark: {'Enabled' if torch.backends.cudnn.benchmark else 'Disabled'}<br>",
                    0
                )
                self.has_logged_gpu_info = True
            except Exception as e:
                print(f"[GPU] Error logging GPU info: {str(e)}")
        
    def on_train_epoch_start(self, trainer, pl_module):
        self.train_epoch_start_time = time.time()
        self.train_batch_times = []
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.train_batch_start_time = time.time()
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        batch_time = time.time() - self.train_batch_start_time
        self.train_batch_times.append(batch_time)
        
        trainer.logger.experiment.add_scalar(
            "time/step_ms", 
            batch_time * 1000, 
            trainer.global_step
        )
        
        if torch.cuda.is_available() and batch_idx % 50 == 0:
            try:
                gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                
                trainer.logger.experiment.add_scalar(
                    "gpu/memory_allocated_gb", 
                    gpu_mem_allocated,
                    trainer.global_step
                )
                trainer.logger.experiment.add_scalar(
                    "gpu/memory_reserved_gb", 
                    gpu_mem_reserved,
                    trainer.global_step
                )
            except Exception as e:
                pass  # Silently handle GPU monitoring errors
        
        if batch_idx % 50 == 0:
            avg_time = sum(self.train_batch_times[-20:]) / min(len(self.train_batch_times), 20)
            # Get batch size from the actual batch, not dataloader (which is safer)
            batch_size = batch[0].shape[0] if isinstance(batch, (list, tuple)) else batch.shape[0]
            samples_per_sec = batch_size / avg_time
            
            gpu_info = ""
            if torch.cuda.is_available():
                try:
                    gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_info = f" | GPU Mem: {gpu_mem_allocated:.2f} GB"
                except:
                    pass
                    
            print(f"[Timer] Batch {batch_idx}: {batch_time*1000:.2f}ms | Avg(20): {avg_time*1000:.2f}ms | {samples_per_sec:.1f} samples/sec{gpu_info}")
            
    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.train_epoch_start_time
        self.epoch_times.append(epoch_time)
        
        avg_batch_time = sum(self.train_batch_times) / len(self.train_batch_times) if self.train_batch_times else 0
        
        trainer.logger.experiment.add_scalar(
            "time/epoch_sec", 
            epoch_time,
            trainer.current_epoch
        )
        trainer.logger.experiment.add_scalar(
            "time/avg_step_ms", 
            avg_batch_time * 1000, 
            trainer.current_epoch
        )
        
        elapsed = time.time() - self.train_start_time
        estimated_total = elapsed / (trainer.current_epoch + 1) * trainer.max_epochs
        remaining = estimated_total - elapsed
        
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        remaining_str = str(timedelta(seconds=int(remaining)))
        
    def on_validation_start(self, trainer, pl_module):
        self.val_start_time = time.time()
        self.val_batch_times = []
    
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.val_batch_start_time = time.time()
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        val_batch_time = time.time() - self.val_batch_start_time
        self.val_batch_times.append(val_batch_time)

    def on_validation_end(self, trainer, pl_module):
        val_time = time.time() - self.val_start_time
        avg_val_batch_time = sum(self.val_batch_times) / len(self.val_batch_times) if self.val_batch_times else 0
        
        trainer.logger.experiment.add_scalar(
            "time/validation_sec", 
            val_time,
            trainer.current_epoch
        )
        trainer.logger.experiment.add_scalar(
            "time/avg_val_step_ms", 
            avg_val_batch_time * 1000, 
            trainer.current_epoch
        )
        
    def on_train_end(self, trainer, pl_module):
        total_time = time.time() - self.train_start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        trainer.logger.experiment.add_scalar(
            "time/total_training_hours", 
            total_time / 3600, 
            0
        )
        
        print(f"[Timer] Total training time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        
        if torch.cuda.is_available():
            try:
                gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                print(f"[GPU] Final memory allocated: {gpu_mem_allocated:.2f} GB")
                print(f"[GPU] Final memory reserved: {gpu_mem_reserved:.2f} GB")
            except Exception as e:
                pass
        
        # Save detailed timing data to JSON
        timing_data = {
            "total_seconds": total_time,
            "epoch_seconds": self.epoch_times,
            "avg_batch_ms": sum(self.train_batch_times) / len(self.train_batch_times) * 1000 if self.train_batch_times else 0,
            "avg_val_batch_ms": sum(self.val_batch_times) / len(self.val_batch_times) * 1000 if self.val_batch_times else 0,
        }
        
        # Try to save timing data to output directory
        try:
            if hasattr(trainer, "default_root_dir") and trainer.default_root_dir:
                timing_file = os.path.join(trainer.default_root_dir, "training_timing.json")
                with open(timing_file, "w") as f:
                    json.dump(timing_data, f, indent=2)
                print(f"[Timer] Timing data saved to {timing_file}")
        except Exception as e:
            print(f"[Timer] Error saving timing data: {str(e)}")

# Callback to track metrics from the epoch with the best validation loss
class BestEpochMetricsCallback(L.Callback):
    """Tracks MAE and RMSE from the epoch with the best validation loss."""
    def __init__(self, monitor='val_loss', mode='min'):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_mae = None
        self.best_rmse = None

    def on_validation_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        current_score = logs.get(self.monitor)

        if current_score is None:
            return # Metric not logged yet

        # Check if current score is better than best score
        if self.mode == 'min':
            is_better = current_score < self.best_score
        else: # mode == 'max'
            is_better = current_score > self.best_score

        if is_better:
            self.best_score = current_score
            self.best_mae = logs.get('val_mae')
            self.best_rmse = logs.get('val_rmse')
            print(f"\n[Best Metrics] New best val_loss: {self.best_score:.4f} -> val_mae: {self.best_mae:.4f}, val_rmse: {self.best_rmse:.4f}\n")

class GradientAndWeightMonitorCallback(L.Callback):
    """
    Lightning callback to monitor gradient norms and weight updates during training.
    Logs total and max norms for gradients and weight updates.
    """
    def __init__(self, log_every_n_steps=50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.prev_weights = {}
        self.layer_names_to_log = [] # Store a few key layers for detailed logging

    def on_train_start(self, trainer, pl_module):
        # Store initial weights and identify a few layers to log specifically
        self.layer_names_to_log = []
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                self.prev_weights[name] = param.data.clone()
                # Log weights/grads for first few conv layers in encoder/decoder
                if ('contr_' in name or 'expand_' in name) and 'weight' in name and len(self.layer_names_to_log) < 5:
                     self.layer_names_to_log.append(name)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Log gradients after optimizer step, before zeroing (or check if grads exist)
        # Note: This hook might execute *after* optimizer.step() and zero_grad() in some Lightning versions.
        # A safer place might be on_after_backward if needing grads before optimizer step.
        # However, for logging norms *after* the step, this hook is fine if grads aren't zeroed yet,
        # or we can calculate norms based on `param.grad` if it persists until this hook.
        # Let's assume grads are available here for simplicity based on original code placement.

        if (trainer.global_step + 1) % self.log_every_n_steps != 0:
             # Use global_step for consistent logging frequency across epochs
             return

        global_step = trainer.global_step

        # --- Log Gradient Norms ---
        total_grad_norm = 0.0
        max_grad_norm = 0.0
        max_grad_layer = "N/A"
        num_params_with_grad = 0

        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                try:
                    grad_norm = torch.linalg.norm(param.grad.detach()).item()
                    if np.isfinite(grad_norm):
                        total_grad_norm += grad_norm ** 2
                        num_params_with_grad += 1
                        if grad_norm > max_grad_norm:
                            max_grad_norm = grad_norm
                            max_grad_layer = name
                        # Log detailed info for key layers
                        if name in self.layer_names_to_log:
                            trainer.logger.experiment.add_scalar(f"grad_norm_layer/{name}", grad_norm, global_step)
                    else:
                         print(f"Warning: Non-finite gradient norm for {name}")
                except Exception as e:
                     print(f"Warning: Could not calculate grad norm for {name}: {e}")


        if num_params_with_grad > 0:
             total_grad_norm = total_grad_norm ** 0.5
             trainer.logger.experiment.add_scalar("grad_norm/total", total_grad_norm, global_step)
             trainer.logger.experiment.add_scalar("grad_norm/max", max_grad_norm, global_step)
        else:
             # Handle case where no gradients were found (e.g., validation epoch end)
             pass


        # --- Log Weight Updates ---
        total_weight_update = 0.0
        max_weight_update = 0.0
        max_update_layer = "N/A"
        num_params_updated = 0

        for name, param in pl_module.named_parameters():
            if param.requires_grad and name in self.prev_weights:
                try:
                    update_tensor = param.data - self.prev_weights[name]
                    update_norm = torch.linalg.norm(update_tensor).item()

                    if np.isfinite(update_norm):
                         total_weight_update += update_norm ** 2
                         num_params_updated +=1
                         if update_norm > max_weight_update:
                             max_weight_update = update_norm
                             max_update_layer = name
                         # Log detailed info for key layers
                         if name in self.layer_names_to_log:
                             trainer.logger.experiment.add_scalar(f"weight_update_norm/{name}", update_norm, global_step)
                    else:
                         print(f"Warning: Non-finite weight update norm for {name}")

                    # Store current weights for next comparison *after* calculating update
                    self.prev_weights[name] = param.data.clone()
                except Exception as e:
                     print(f"Warning: Could not calculate weight update for {name}: {e}")

        if num_params_updated > 0:
             total_weight_update = total_weight_update ** 0.5
             trainer.logger.experiment.add_scalar("weight_update/total", total_weight_update, global_step)
             trainer.logger.experiment.add_scalar("weight_update/max", max_weight_update, global_step)


class EpochMetricsTracker(L.Callback):
    """Tracks detailed metrics using the validation dataloader at the end of each validation epoch."""

    def __init__(self, val_dataloader_fn):
        super().__init__()
        # Pass a function that returns the val dataloader
        # This avoids issues with accessing trainer.datamodule early
        self.val_dataloader_fn = val_dataloader_fn
        self.epoch_metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        # Ensure model is in eval mode and move to correct device
        pl_module.eval()
        device = pl_module.device
        val_dataloader = self.val_dataloader_fn() # Get dataloader

        val_preds_list, val_targets_list = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                # Assuming batch structure: x, y_original, y_transformed
                # We need y_original for evaluation against predictions
                x, y_original, _ = batch
                x = x.to(device)
                # Use model's forward pass - ensure correct arguments if needed
                # Pass apply_easyuq=True if available and desired for this evaluation
                if hasattr(pl_module, 'forward') and 'apply_easyuq' in pl_module.forward.__code__.co_varnames:
                     y_hat = pl_module(x, apply_easyuq=True)
                else:
                     y_hat = pl_module(x) # Fallback if apply_easyuq not implemented/needed here

                if y_hat.dim() == 4 and y_hat.shape[1] == 1:
                    y_hat = y_hat.squeeze(1)
                
                # IMPORTANT: Apply inverse transform to convert predictions from log scale to original scale
                # This is crucial for proper metric calculation against original-scale targets
                if hasattr(pl_module, 'target_scaler'):
                    y_hat = pl_module.target_scaler.inverse_transform(y_hat)
                    print(f"Applied inverse transform to predictions. Min: {y_hat.min().item():.4f}, Max: {y_hat.max().item():.4f}")
                else:
                    print("Warning: No target_scaler found on model. Metrics may be incorrect if scales don't match.")

                val_preds_list.append(y_hat.cpu())
                val_targets_list.append(y_original.cpu()) # Compare against original targets

        if not val_preds_list:
            print("Warning: No validation predictions collected for EpochMetricsTracker.")
            return

        val_preds = torch.cat(val_preds_list, dim=0)
        val_targets = torch.cat(val_targets_list, dim=0)

        # Calculate metrics
        metrics = {}
        current_epoch = trainer.current_epoch
        metrics['epoch'] = current_epoch

        # Handle potential NaNs in targets or predictions before calculating metrics
        valid_mask = torch.isfinite(val_targets) & torch.isfinite(val_preds)
        if valid_mask.sum() == 0:
             print(f"Epoch {current_epoch}: No valid data points for detailed metrics.")
             return

        val_preds_valid = val_preds[valid_mask]
        val_targets_valid = val_targets[valid_mask]

        metrics['mae'] = torch.abs(val_preds_valid - val_targets_valid).mean().item()
        metrics['rmse'] = torch.sqrt(F.mse_loss(val_preds_valid, val_targets_valid)).item()

        # Metrics by intensity bins on original target scale
        bins = [0, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, float('inf')]
        bin_names = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" if bins[i+1] != float('inf') else f">{bins[i]:.1f}" for i in range(len(bins)-1)]

        for i, bin_name in enumerate(bin_names):
            if bins[i+1] == float('inf'):
                bin_mask = (val_targets_valid >= bins[i])
            else:
                bin_mask = (val_targets_valid >= bins[i]) & (val_targets_valid < bins[i+1])

            if bin_mask.sum() > 0:
                 mae_bin = torch.abs(val_preds_valid[bin_mask] - val_targets_valid[bin_mask]).mean().item()
                 rmse_bin = torch.sqrt(F.mse_loss(val_preds_valid[bin_mask], val_targets_valid[bin_mask])).item()
                 metrics[f'mae_{bin_name}'] = mae_bin
                 metrics[f'rmse_{bin_name}'] = rmse_bin
                 # Log to tensorboard
                 trainer.logger.experiment.add_scalar(f"val_intensity_mae/{bin_name}", mae_bin, current_epoch)
                 trainer.logger.experiment.add_scalar(f"val_intensity_rmse/{bin_name}", rmse_bin, current_epoch)

        self.epoch_metrics.append(metrics)

        # Print summary for the epoch
        print(f"\n--- Epoch {current_epoch} Detailed Validation Metrics ---")
        print(f"  Overall MAE : {metrics.get('mae', 'N/A'):.4f}")
        print(f"  Overall RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        for bin_name in bin_names:
             if f'mae_{bin_name}' in metrics:
                 print(f"  MAE ({bin_name} mm): {metrics[f'mae_{bin_name}']:.4f}")
        print("-------------------------------------------\n")


class DeterministicPerformanceTracker(L.Callback):
    """
    Tracks deterministic performance metrics (MAE, RMSE, Bias, CSI) using thresholds.
    Note: Evaluation functionality was removed as per user request in previous steps,
          so this callback currently does nothing on epoch end.
          Kept definition here for completeness if needed.
    """
    def __init__(self, val_dataloader_fn, thresholds=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
                 eval_every_n_epochs=5):
        super().__init__()
        self.val_dataloader_fn = val_dataloader_fn # Function to get dataloader
        self.thresholds = thresholds
        self.eval_every_n_epochs = eval_every_n_epochs
        self.metrics_history = [] # Store history if needed

    def on_validation_epoch_end(self, trainer, pl_module):
        # Evaluation functionality previously removed based on user priority.
        # If deterministic evaluation is needed later, implement logic here.
        # current_epoch = trainer.current_epoch
        # if (current_epoch + 1) % self.eval_every_n_epochs == 0:
        #     print(f"Epoch {current_epoch}: DeterministicPerformanceTracker evaluation skipped.")
        pass # Does nothing currently

    def get_metrics_history(self):
        # Returns stored history if evaluation logic were implemented
        return self.metrics_history

# New precipitation-specific metrics
class CriticalSuccessIndex(torch.nn.Module):
    def __init__(self, threshold=0.2):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, preds, target):
        # Convert to binary using threshold
        preds_binary = (preds >= self.threshold).float()
        target_binary = (target >= self.threshold).float()
        
        # Calculate hits, false alarms, misses
        hits = torch.sum(preds_binary * target_binary)
        false_alarms = torch.sum(preds_binary * (1 - target_binary))
        misses = torch.sum((1 - preds_binary) * target_binary)
        
        # Calculate CSI: hits / (hits + misses + false_alarms)
        csi = hits / (hits + misses + false_alarms + 1e-8)  # add small epsilon to avoid division by zero
        return csi

class SpatialCorrelation(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, preds, target):
        # Calculate spatial correlation coefficient across the batch
        # Remove mean from each sample
        preds_centered = preds - torch.mean(preds, dim=(-2, -1), keepdim=True)
        target_centered = target - torch.mean(target, dim=(-2, -1), keepdim=True)
        
        # Calculate correlation
        numerator = torch.sum(preds_centered * target_centered, dim=(-2, -1))
        denominator = torch.sqrt(torch.sum(preds_centered**2, dim=(-2, -1)) * 
                                torch.sum(target_centered**2, dim=(-2, -1)) + 1e-8)
        correlation = numerator / denominator
        
        # Average across batch
        return torch.mean(correlation)

def generate_prediction_plots(inputs, targets, predictions, threshold=1.0, save_path=None):
    """
    Generate plots comparing inputs, targets, and predictions.
    
    Args:
        inputs: Input tensor of shape [batch_size, channels, height, width]
        targets: Target tensor of shape [batch_size, height, width]
        predictions: Prediction tensor of shape [batch_size, height, width]
        threshold: Precipitation threshold for binary evaluation
        save_path: Path to save the plot
    """
    try:
        import numpy as np
        import matplotlib.colors as colors
        
        # Handle empty inputs/targets/predictions
        if inputs is None or targets is None or predictions is None:
            print("Warning: Cannot generate plots - inputs, targets, or predictions is None")
            return
            
        # Ensure batch dimension exists
        if inputs.ndim < 4:
            print(f"Warning: Inputs have unexpected shape {inputs.shape}, adding batch dimension")
            inputs = inputs.unsqueeze(0)
        if targets.ndim < 3:
            print(f"Warning: Targets have unexpected shape {targets.shape}, adding batch dimension")
            targets = targets.unsqueeze(0)
        if predictions.ndim < 3:
            print(f"Warning: Predictions have unexpected shape {predictions.shape}, adding batch dimension")
            predictions = predictions.unsqueeze(0)
            
        # Ensure we have at least one sample in the batch
        if inputs.shape[0] == 0 or targets.shape[0] == 0 or predictions.shape[0] == 0:
            print("Warning: Cannot generate plots - empty batch")
            return
            
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Get the first sample from the batch
        batch_idx = 0
        
        # Handle input visualization - select one channel or combine them
        # Option 1: Visualize t-3 precipitation (channel 0)
        t_minus_3_precip = inputs[batch_idx, 0].cpu().numpy()
        # Create a non-linear norm for better visualization of precipitation
        norm_input = colors.PowerNorm(gamma=0.5)
        im0 = axes[0, 0].imshow(t_minus_3_precip, cmap='Blues', norm=norm_input)
        axes[0, 0].set_title('Input: t-3 Day Precip (Power Scale)')
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # Option 2: Visualize t-2 precipitation (channel 1)
        t_minus_2_precip = inputs[batch_idx, 1].cpu().numpy()
        im1 = axes[0, 1].imshow(t_minus_2_precip, cmap='Blues', norm=norm_input)
        axes[0, 1].set_title('Input: t-2 Day Precip (Power Scale)')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Option 3: Visualize t-1 precipitation (channel 2)
        t_minus_1_precip = inputs[batch_idx, 2].cpu().numpy()
        im2 = axes[0, 2].imshow(t_minus_1_precip, cmap='Blues', norm=norm_input)
        axes[0, 2].set_title('Input: t-1 Day Precip (Power Scale)')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # Define Germany's grid boundaries - same as in _initialize_germany_mask()
        lat_min, lat_max = 17, 25  # Germany latitude range
        lon_min, lon_max = 75, 85  # Germany longitude range
        
        # Target visualization - Full region
        target = targets[batch_idx].cpu().numpy()
        
        # Create a non-linear norm for the target and prediction plots
        # This will compress high values and expand low values for better visualization
        norm = colors.PowerNorm(gamma=0.5)
        
        im3 = axes[1, 0].imshow(target, cmap='Blues', norm=norm)
        axes[1, 0].set_title('Target: Current Day (t) Precip (Power Scale)')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Add indicator box for Germany region on the target plot
        axes[1, 0].add_patch(plt.Rectangle((lon_min, lat_min), 
                                         lon_max-lon_min+1, 
                                         lat_max-lat_min+1, 
                                         fill=False, edgecolor='red', linewidth=2))
        
        # Prediction visualization
        pred = predictions[batch_idx].cpu().numpy().squeeze()  # Apply squeeze to remove leading singleton dimension
        
        # Check prediction values distribution
        pred_mean = np.mean(pred)
        target_mean = np.mean(target)
        
        # Print statistics to debug prediction values
        print(f"Plot statistics - Target: max={target.max():.4f}, mean={target_mean:.4f}")
        print(f"Plot statistics - Prediction: max={pred.max():.4f}, mean={pred_mean:.4f}")
        
        # Use the same norm for the prediction plot to ensure consistency
        im4 = axes[1, 1].imshow(pred, cmap='Blues', norm=norm)
        axes[1, 1].set_title('Prediction: Current Day (t) Precip (Power Scale)')
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # Add indicator box for Germany region on the prediction plot
        axes[1, 1].add_patch(plt.Rectangle((lon_min, lat_min), 
                                         lon_max-lon_min+1, 
                                         lat_max-lat_min+1, 
                                         fill=False, edgecolor='red', linewidth=2))
        
        # Difference map (prediction - target)
        diff = pred - target  # pred is now 2D (after squeeze), target should also be 2D
        im5 = axes[1, 2].imshow(diff, cmap='RdBu_r', vmin=-10, vmax=10)
        axes[1, 2].set_title('Difference (Pred - Target)')
        plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        # Add indicator box for Germany region on the difference plot
        axes[1, 2].add_patch(plt.Rectangle((lon_min, lat_min), 
                                         lon_max-lon_min+1, 
                                         lat_max-lat_min+1, 
                                         fill=False, edgecolor='red', linewidth=2))
        
        # Add overall title
        plt.suptitle(f'Precipitation Forecast Evaluation (threshold = {threshold} mm)', fontsize=16)
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    except Exception as e:
        print(f"Error generating prediction plots: {str(e)}")
        import traceback
        traceback.print_exc()

def convert_to_json_serializable(obj):
    """
    Convert objects to JSON serializable types.
    
    Args:
        obj: The object to convert
        
    Returns:
        JSON serializable version of the object
    """
    import numpy as np
    import torch
    
    if isinstance(obj, (np.ndarray, np.number)):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        return {key: convert_to_json_serializable(value) for key, value in obj.__dict__.items()
                if not key.startswith('_')}
    elif np.isnan(obj):
        return "NaN"
    elif np.isinf(obj):
        return "Infinity" if obj > 0 else "-Infinity"
    else:
        try:
            # Check if object is JSON serializable
            import json
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)

def train_unet_model(data_module, model, output_dir, log_dir, epochs, accelerator='auto', 
                   devices=None, fold=0, precipitation_threshold=0.2, generate_plots=True,
                   patience=30, loss_type='mse', intensity_weights=None, focal_gamma=2.0, 
                   track_epochs=False, evaluate_raw=False, save_predictions=True):
    """
    Train a UNet model for MSWEP precipitation forecasting following the principles
    from the training guide to avoid data leakage.
    
    Key principles implemented:
    - Strict temporal separation of training and validation data
    - Use of expanding/growing training sets (training on all data up to year Y-1 for predicting year Y)
    - Prevention of "future" information leakage into the model
    - Deterministic training with log-transformed MSE loss
    - Saving predictions for offline post-processing with calibration methods
    - EasyUQ/IDR probabilistic calibration for validation and test forecasts
    
    Args:
        data_module: Lightning DataModule with proper temporal splits
        model: Lightning module containing UNet model
        output_dir: Directory to save model and results
        log_dir: Directory to save training logs
        epochs: Maximum number of epochs for training
        accelerator: 'auto', 'gpu', 'cpu', etc.
        devices: Number of devices to use
        fold: Current fold number
        precipitation_threshold: Threshold for Brier score calculation
        generate_plots: Whether to generate prediction plots
        patience: Early stopping patience in epochs
        loss_type: Type of loss function to use (prefer 'mse' with log-scaling as per the paper)
        intensity_weights: Weights for different precipitation intensities
        focal_gamma: Gamma parameter for focal loss
        track_epochs: Whether to track detailed metrics for each epoch
        evaluate_raw: If True, evaluate raw UNet outputs with intensity-stratified metrics
        save_predictions: If True, save raw predictions and targets for offline calibration (recommended)
    
    Returns:
        Dictionary of results and best model path
    """
    import os
    import torch
    import json
    import numpy as np
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    # Use the output directory directly (no need to create a new timestamped directory here)
    fold_dir = output_dir
    os.makedirs(fold_dir, exist_ok=True)
    
    # No need to create fold_log_dir here, TensorBoardLogger handles it based on save_dir and version
    
    # 2. Configure loss function - ensure we use log-scaling for MSE as in the paper
    # If the model already has a log scaler (e.g., TargetLogScaler), we keep it
    # Otherwise, we configure the loss function as needed
    if hasattr(model, 'loss_type'):
        print(f"Setting loss function to: {loss_type}")
        model.loss_type = loss_type
        model.loss_fn = model.configure_loss_function(
            loss_type=loss_type,
            intensity_weights=intensity_weights,
            focal_gamma=focal_gamma
        )
    
    # 3. Setup callbacks 
    # Early stopping callback based on validation loss
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=patience,
        verbose=True,
        mode="min"
    )
    
    # Model checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        dirpath=fold_dir,
        filename='best_model',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    # Learning rate monitor to track learning rate changes
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Gradient and weight monitoring callback
    grad_monitor = GradientAndWeightMonitorCallback(log_every_n_steps=50)
    
    # Define new histogram gradient logging callback
    class HistGradientLoggingCallback(L.Callback):
        def __init__(self, log_every_n_steps=100):
            super().__init__()
            self.log_every_n_steps = log_every_n_steps
            self.step_count = 0
            
        def on_after_backward(self, trainer, pl_module):
            # Only log every n steps to reduce overhead
            self.step_count += 1
            if self.step_count % self.log_every_n_steps != 0:
                return
                
            global_step = trainer.global_step
            for name, param in pl_module.named_parameters():
                if param.grad is not None and param.grad.numel() > 0 and not torch.all(param.grad == 0.0):
                    try:
                        # Convert to CPU to avoid GPU memory issues
                        grad_cpu = param.grad.detach().cpu()
                        # Check if the tensor has valid values for histogram
                        if not torch.isnan(grad_cpu).any() and not torch.isinf(grad_cpu).any():
                            trainer.logger.experiment.add_histogram(
                                tag=f"grads/{name}",
                                values=grad_cpu,
                                global_step=global_step
                            )
                    except Exception as e:
                        # Silently handle errors in histogram logging
                        pass

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            # Only log weights occasionally to save overhead
            if batch_idx % self.log_every_n_steps != 0:
                return
                
            global_step = trainer.global_step
            for name, param in pl_module.named_parameters():
                if param.numel() > 0 and not torch.all(param == 0.0):
                    try:
                        # Convert to CPU to avoid GPU memory issues
                        param_cpu = param.detach().cpu()
                        # Check if the tensor has valid values for histogram
                        if not torch.isnan(param_cpu).any() and not torch.isinf(param_cpu).any():
                            trainer.logger.experiment.add_histogram(
                                tag=f"weights/{name}",
                                values=param_cpu,
                                global_step=global_step
                            )
                    except Exception as e:
                        # Silently handle errors in histogram logging
                        pass

    hist_grad_logging = HistGradientLoggingCallback(log_every_n_steps=100)
    
    # Compose callbacks list with additional monitoring
    callbacks = [checkpoint_callback, lr_monitor, ModelSummary(max_depth=-1), hist_grad_logging, grad_monitor]
    
    # Add timer callback to track and log training time
    timer_callback = TimerCallback()
    callbacks.append(timer_callback)
    
    if track_epochs:
        print("Adding detailed metric tracking per epoch")
        callbacks.append(EpochMetricsTracker(lambda: data_module.val_dataloader()))
    
    # Add deterministic performance tracker
    perf_tracker = DeterministicPerformanceTracker(
        lambda: data_module.val_dataloader(),
        thresholds=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
        eval_every_n_epochs=5
    )
    callbacks.append(perf_tracker)
    
    # Add early stopping if requested
    if patience > 0:
        callbacks.append(early_stop_callback)
    
    # Add the custom callback to track best epoch metrics
    best_metrics_callback = BestEpochMetricsCallback(monitor='val_loss', mode='min')
    callbacks.append(best_metrics_callback)
    
    # 4. Setup logger - either TensorBoard or CSV
    # Extract run name from the fold_dir path to keep consistent naming between output dir and logs
    # run_name = os.path.basename(os.path.dirname(fold_dir)) # No longer needed for logger version

    # The log_dir is now the run-specific log directory (e.g., <log_dir>/run_<timestamp>)
    # We want the structure: log_dir/foldX/events...
    # So, set save_dir=log_dir, name=None, and version=f"fold{fold}"
    logger = TensorBoardLogger(
        save_dir=log_dir,           # Use the run-specific log dir (e.g., .../<log_dir>/run_XXXX)
        name=None,                  # Don't create an extra subdirectory
        version=f"fold{fold}",      # Create the fold subdirectory (e.g., .../<log_dir>/run_XXXX/fold0)
        default_hp_metric=False
    )
    
    # 5. Configure the trainer with all our callbacks
    trainer = L.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=10,
        deterministic='warn',
        default_root_dir=fold_dir,
        precision="32-true" # Use 16-bit mixed precision to leverage Tensor Cores
    )
    
    # Log hyperparameters to TensorBoard
    hparams = {
        "optimizer_type": "Adam",           # or "AdamW"
        "lr_scheduler": "CosineAnnealingLR",  # or "ReduceLROnPlateau"
        "loss_function": model.loss_type       # as configured
    }
    logger.log_hyperparams(hparams)
    
    # 6. Train the model
    print(f"Starting training for {epochs} epochs")
    trainer.fit(model, datamodule=data_module)

    # --- Capture best metrics from the trained model instance --- 
    # Retrieve before potentially loading checkpoint, which creates a new instance
    final_best_mae = model.best_val_mae.item() if hasattr(model, 'best_val_mae') and model.best_val_mae is not None else None
    final_best_rmse = model.best_val_rmse.item() if hasattr(model, 'best_val_rmse') and model.best_val_rmse is not None else None
    # -------------------------------------------------------------

    # Save predictions BEFORE loading the best checkpoint (otherwise lists will be empty)
    if save_predictions:
        # Use the provided output directory directly
        actual_save_dir = fold_dir
        print(f"Saving predictions and targets for offline calibration to: {actual_save_dir}")
        
        try:
            # Access collected data from the model - already in original scale with inverse transform applied
            # Validation data
            if hasattr(model, 'validation_step_preds') and model.validation_step_preds:
                # Predictions are already in original scale and CPU from the LightningModule
                val_preds_all = torch.cat(model.validation_step_preds).cpu().numpy()
                np.save(os.path.join(actual_save_dir, "val_preds.npy"), val_preds_all)
                print(f"  Saved val_preds shape: {val_preds_all.shape} (original scale)")
                model.validation_step_preds.clear()  # Clear memory
            else:
                print("Warning: No validation_step_preds found or list is empty.")

            if hasattr(model, 'validation_step_targets') and model.validation_step_targets:
                # Targets are already in original scale
                val_targets_all = torch.cat(model.validation_step_targets).cpu().numpy()
                np.save(os.path.join(actual_save_dir, "val_targets.npy"), val_targets_all)
                print(f"  Saved val_targets shape: {val_targets_all.shape} (original scale)")
                model.validation_step_targets.clear()  # Clear memory
            else:
                print("Warning: No validation_step_targets found or list is empty.")

            # Save validation timestamps
            if hasattr(data_module, 'val_times') and data_module.val_times is not None:
                val_times_all = data_module.val_times # Access the stored timestamps
                np.save(os.path.join(actual_save_dir, "val_times.npy"), val_times_all)
                print(f"  Saved val_times shape: {val_times_all.shape}")
            else:
                print("Warning: Could not find val_times on data_module to save.")

            # Training data (needed for EasyUQ fitting)
            if hasattr(model, 'training_step_preds') and model.training_step_preds:
                 # Code to save train_preds_all.npy was missing, adding it now:
                 train_preds_all = torch.cat(model.training_step_preds).cpu().numpy()
                 np.save(os.path.join(actual_save_dir, "train_preds_all.npy"), train_preds_all)
                 print(f"  Saved ALL train_preds shape: {train_preds_all.shape} (original scale)")
                 model.training_step_preds.clear()
            else:
                print("Warning: No training_step_preds found or list is empty.")

            if hasattr(model, 'training_step_targets') and model.training_step_targets:
                # Targets are already in original scale
                train_targets_all = torch.cat(model.training_step_targets).cpu().numpy()
                np.save(os.path.join(actual_save_dir, "train_targets_all.npy"), train_targets_all)
                print(f"  Saved ALL train_targets shape: {train_targets_all.shape} (original scale)")
                model.training_step_targets.clear()
            else:
                print("Warning: No training_step_targets found or list is empty.")

            # Save model's Germany mask if available (for later CRPS calculation)
            if hasattr(model, 'model') and hasattr(model.model, 'germany_mask') and model.model.germany_mask is not None:
                germany_mask = model.model.germany_mask.cpu().numpy()
                np.save(os.path.join(actual_save_dir, "germany_mask.npy"), germany_mask)
                print(f"  Saved germany_mask shape: {germany_mask.shape}")
            else:
                print("Warning: Could not find germany_mask on model to save.")

        except Exception as e:
            print(f"Error saving predictions/targets: {e}")
            traceback.print_exc()

    # 8. Load the best checkpoint
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model path: {best_model_path}")
    loaded_model = model # Start with the model as it finished training
    if best_model_path and os.path.exists(best_model_path):
        print(f"Attempting to load best model from: {best_model_path}")
        try:
            # Create a fresh UNet model instance with the CORRECT number of input channels
            from models.mswep_unet import MSWEPUNet
            unet_model_for_load = MSWEPUNet(in_channels=5) # Use 5 channels

            # Load the checkpoint state into the Lightning Module wrapper
            # Pass the correctly initialized inner model
            # Also pass other necessary hyperparameters if they might have changed
            # or if loading requires them (e.g., learning_rate, loss_type etc. might
            # be needed by the wrapper's __init__ even when loading state)
            # Assuming 'model' arg is sufficient for structure, and others are in hparams:
            loaded_model = UNetLightningModule.load_from_checkpoint(
                best_model_path,
                model=unet_model_for_load
                # Add any other essential args here if load_from_checkpoint needs them
                # e.g., learning_rate=model.hparams.learning_rate (or args.lr) if needed
            )
            loaded_model.eval()
            print("Loaded best model checkpoint successfully.")
            model = loaded_model # Use the loaded model going forward if successful
        except FileNotFoundError:
             print(f"Warning: Best model path reported but not found ({best_model_path}). Using model state from end of training.")
        except RuntimeError as re:
             print(f"RuntimeError loading best model checkpoint: {re}. Check model architecture compatibility.")
             print("Continuing with model state from end of training.")
            # traceback.print_exc() # Uncomment for detailed stack trace
        except Exception as e:
            print(f"Generic error loading best model checkpoint: {str(e)}")
            traceback.print_exc() # Print full traceback for unexpected errors
            print("Continuing with model state from end of training.")
    else:
        print("Best model path not found or not valid. Using model state from end of training.")
    

    # 9. Evaluate raw model performance if requested
    results = {}
    results['best_model_path'] = best_model_path
    results['fold'] = fold
    
    # Get the best validation loss from the checkpoint callback (most reliable source)
    best_val_loss = checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score is not None else float('inf')
    results['best_val_loss'] = best_val_loss

    # Get the corresponding MAE and RMSE tracked within the LightningModule
    # Use the values captured *before* checkpoint loading
    results['best_val_mae'] = final_best_mae 
    results['best_val_rmse'] = final_best_rmse

    # Prepare values for printing, handling None
    loss_str = f"{results['best_val_loss']:.4f}" if results['best_val_loss'] is not None else "N/A"
    mae_str = f"{results['best_val_mae']:.4f}" if results['best_val_mae'] is not None else "N/A"
    rmse_str = f"{results['best_val_rmse']:.4f}" if results['best_val_rmse'] is not None else "N/A"
    print(f"Retrieved best metrics: Loss={loss_str}, MAE={mae_str}, RMSE={rmse_str}")
    
    # Get metrics history if tracked
    if track_epochs and hasattr(perf_tracker, 'get_metrics_history'):
        metrics_history = perf_tracker.get_metrics_history()
        results['metrics_history'] = metrics_history
    
    # 12. Generate prediction plots if requested
    if generate_plots:
        try:
            # Get a small batch of validation data
            val_iter = iter(data_module.val_dataloader())
            batch = next(val_iter)
            # Unpack all three items, but we primarily need x and y_original for plotting
            x, y_original, _ = batch
            
            # Move to the same device as model
            device = next(model.parameters()).device
            x = x.to(device)
            # We don't need to move y_original or y_transformed to device for plotting
            
            # Generate predictions with EasyUQ/IDR enabled
            with torch.no_grad():
                y_hat = model(x, apply_easyuq=True)
                # Apply inverse transform to get predictions in original scale for plotting
                if hasattr(model, 'target_scaler'):
                    y_hat_orig_scale = model.target_scaler.inverse_transform(y_hat)
                    print(f"Applied inverse transform for plotting. Original scale - "
                          f"Min: {y_hat_orig_scale.min().item():.4f}, Max: {y_hat_orig_scale.max().item():.4f}")
                else:
                    y_hat_orig_scale = y_hat
                    print("Warning: model has no target_scaler attribute. Plotting raw predictions.")
            
            # Generate plots using the original target data and transformed predictions
            generate_prediction_plots(
                x.cpu(),
                y_original.cpu(), # Use the original target tensor for plotting
                y_hat_orig_scale.cpu(), # Use inverse-transformed predictions
                threshold=precipitation_threshold,
                save_path=os.path.join(actual_save_dir, f"prediction_examples_eu_fold{fold}.png")
            )
            print(f"Saved EasyUQ prediction example plots to {actual_save_dir}")
        except Exception as e:
            print(f"Error generating prediction plots: {str(e)}")
    
    # Save the full results dictionary
    try:
        # Add timing information to results
        if hasattr(timer_callback, 'train_batch_times') and timer_callback.train_batch_times:
            results['timing'] = {
                'avg_batch_ms': sum(timer_callback.train_batch_times) / len(timer_callback.train_batch_times) * 1000,
                'avg_epoch_sec': sum(timer_callback.epoch_times) / len(timer_callback.epoch_times) if timer_callback.epoch_times else 0,
                'total_time_sec': time.time() - timer_callback.train_start_time if timer_callback.train_start_time else 0
            }
        
        with open(os.path.join(actual_save_dir, f"training_results_fold{fold}.json"), "w") as f:
            # Ensure the results dict includes the best metrics before saving
            json.dump(results, f, indent=2, default=convert_to_json_serializable)
            print(f"Saved final results including best epoch metrics to {os.path.join(actual_save_dir, f'training_results_fold{fold}.json')}")
            
        # Also save a dedicated timing report
        if hasattr(timer_callback, 'train_start_time') and timer_callback.train_start_time:
            timing_report = {
                'model_type': type(model).__name__,
                'batch_size': data_module.batch_size,
                'num_workers': data_module.num_workers,
                'precision': trainer.precision,
                'avg_batch_time_ms': sum(timer_callback.train_batch_times) / len(timer_callback.train_batch_times) * 1000 if timer_callback.train_batch_times else 0,
                'avg_val_batch_time_ms': sum(timer_callback.val_batch_times) / len(timer_callback.val_batch_times) * 1000 if timer_callback.val_batch_times else 0,
                'epoch_times_sec': timer_callback.epoch_times,
                'total_time_sec': time.time() - timer_callback.train_start_time,
                'gpu_type': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(os.path.join(actual_save_dir, f"timing_report_fold{fold}.json"), "w") as f:
                json.dump(timing_report, f, indent=2, default=convert_to_json_serializable)
            
    except Exception as e:
        print(f"Error saving results: {str(e)}")
    
    print(f"Training completed. Results saved to {actual_save_dir}")
    return results, best_model_path

def train_fold(fold, args, fold_output_dir=None, run_log_dir=None):
    """
    Train a model for a specific fold, using a time-series based approach.
    For each fold, we use a different portion of the continuous time series split
    as defined by scikit-learn's TimeSeriesSplit.
    
    Args:
        fold: The fold number
        args: Command line arguments
        fold_output_dir: If provided, use this specific directory for output instead of creating a new one
        run_log_dir: The log directory specific to this run (e.g., <log_dir>/run_<timestamp>)
    """
    print(f"=== Starting training for fold {fold} using time-series split approach ===")
    
    # Ensure fold_output_dir exists
    if fold_output_dir is None:
        # This path should generally not be taken if called from main
        from datetime import datetime
        run_timestamp_fallback = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir_local = os.path.join(args.output_dir, f"run_{run_timestamp_fallback}") 
        fold_output_dir = os.path.join(run_output_dir_local, f"fold{fold}")
    os.makedirs(fold_output_dir, exist_ok=True)
    
    # Ensure run_log_dir exists
    log_dir_for_fold = run_log_dir
    if log_dir_for_fold is None:
        # Fallback if not provided (should be provided from main)
        print("Warning: run_log_dir not provided to train_fold. Defaulting log path.")
        run_timestamp_fallback = datetime.now().strftime("%Y%m%d_%H%M%S") # Use another timestamp or derive from output
        log_dir_for_fold = os.path.join(args.log_dir or './logs', f"run_{run_timestamp_fallback}")
    os.makedirs(log_dir_for_fold, exist_ok=True) # Ensure it exists
    
    # Use the command line arguments for regional focus settings
    current_use_regional_focus = args.use_regional_focus
    # Always keep Germany weight as 1.0
    current_region_weight = 1.0
    # Use the command line argument for outside_weight
    current_outside_weight = args.outside_weight
    
    print(f"Fold {fold}: Regional focus settings:")
    print(f"  - Use regional focus: {current_use_regional_focus}")
    print(f"  - Germany region weight: {current_region_weight} (fixed)")
    print(f"  - Outside region weight: {current_outside_weight} (from command line)")
    
    # Create the data module for this fold using TimeSeriesSplit
    data_module = MSWEPDataModule(
        data_dir=args.data_dir,
        test_data_dir=args.test_data_dir if hasattr(args, 'test_data_dir') else None,
        batch_size=HARDCODED_BATCH_SIZE,
        num_workers=HARDCODED_NUM_WORKERS,
        fold=fold,
        target_source=args.target_source if hasattr(args, 'target_source') else 'mswep'
    )
    
    # Set up the data module before using it
    data_module.setup(stage='fit')
    
    # Print fold information for clarity
    print(f"Fold {fold} using year-based expanding window approach")
    print(f"  - Target source: {args.target_source if hasattr(args, 'target_source') else 'mswep'}")
    print(f"  - Regional focus: {current_use_regional_focus}")
    print(f"  - Training samples: {len(data_module.train_dataset)}")
    print(f"  - Validation samples: {len(data_module.val_dataset)}")
    
    # Initialize or load the UNet model sequentially.
    if fold == 0:
         print("Using standard UNet architecture for initial fold")
         from models.mswep_unet import MSWEPUNet
         unet_model = MSWEPUNet(
             in_channels=5,  # 5 channels (3 lags + 2 seasonality features)
             dropout=args.dropout if hasattr(args, 'dropout') else 0.2,
             # Pass regional focus parameters to the UNet model
             use_regional_focus=current_use_regional_focus,
             region_weight=current_region_weight,
             outside_weight=current_outside_weight
         )
         from models.mswep_lightning_wrapper import UNetLightningModule
         model = UNetLightningModule(
             model=unet_model, 
             learning_rate=args.lr, 
             loss_type=args.loss_type,
             optimizer_type=args.optimizer_type if hasattr(args, 'optimizer_type') else 'adam',
             lr_scheduler_type=args.lr_scheduler_type if hasattr(args, 'lr_scheduler_type') else 'cosineannealinglr',
             use_regional_focus=current_use_regional_focus,
             region_weight=current_region_weight,
             outside_weight=current_outside_weight,
             log_offset=args.log_offset if hasattr(args, 'log_offset') else 0.00001,
             weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 1e-3
         )
    else:
        # Get the current run directory (parent of the fold output directory)
        current_run_dir = os.path.dirname(fold_output_dir)
        
        # Check for previous fold checkpoint in the same run
        prev_fold_dir = os.path.join(current_run_dir, f"fold{fold-1}")
        print(f"Looking for previous fold checkpoint in: {prev_fold_dir}")
        
        # Look for checkpoint file
        prev_ckpt_path = None
        if os.path.exists(prev_fold_dir):
            # Find any file that starts with best_model and ends with .ckpt
            potential_ckpts = [f for f in os.listdir(prev_fold_dir) 
                             if f.startswith('best_model') and f.endswith('.ckpt')]
             
            if potential_ckpts:
                # Sort by modification time (most recent first) if multiple matches
                potential_ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(prev_fold_dir, x)), 
                                    reverse=True)
                prev_ckpt_path = os.path.join(prev_fold_dir, potential_ckpts[0])
                print(f"Found checkpoint: {prev_ckpt_path}")
            else:
                print(f"No checkpoint files found in {prev_fold_dir}")
        else:
            print(f"Previous fold directory not found: {prev_fold_dir}")
            
        # Check if we found a checkpoint
        if prev_ckpt_path and os.path.exists(prev_ckpt_path):
            print(f"Loading model from previous fold's checkpoint: {prev_ckpt_path}")
            try:
                # Create a fresh UNet model for proper reconstruction
                from models.mswep_unet import MSWEPUNet
                # Use 5 channels (3 lags + 2 seasonality features)
                inner_unet_model = MSWEPUNet(
                    in_channels=5,
                    # Pass regional focus parameters to the UNet model
                    use_regional_focus=current_use_regional_focus,
                    region_weight=current_region_weight,
                    outside_weight=current_outside_weight
                )
                
                # Import modules needed for optimizer/scheduler reset
                import types
                import torch
                from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
                
                # Load checkpoint with explicit model parameter and current fold's configuration
                from models.mswep_lightning_wrapper import UNetLightningModule
                model = UNetLightningModule.load_from_checkpoint(
                    checkpoint_path=prev_ckpt_path,
                    model=inner_unet_model,
                    # Pass current fold's configuration to override saved values
                    use_regional_focus=current_use_regional_focus,
                    region_weight=current_region_weight,
                    outside_weight=current_outside_weight,
                    # Reset learning rate for new fold
                    learning_rate=args.lr
                )
                
                # IMPORTANT: Override the configure_optimizers method to reset the optimizer and scheduler state
                print(f"Resetting optimizer and scheduler state for fold {fold}")
                
                # Store the necessary parameters to recreate optimizer/scheduler
                optimizer_type = args.optimizer_type if hasattr(args, 'optimizer_type') else 'adam'
                lr_scheduler_type = args.lr_scheduler_type if hasattr(args, 'lr_scheduler_type') else 'cosineannealinglr'
                max_epochs = args.epochs
                
                # Define a new configure_optimizers method that creates fresh optimizer/scheduler
                def new_configure_optimizers(self):
                    print(f"Creating fresh optimizer ({optimizer_type}) with learning rate: {args.lr}")
                    
                    # Create optimizer based on type
                    if optimizer_type.lower() == 'adam':
                        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
                    elif optimizer_type.lower() == 'adamw':
                        optimizer = torch.optim.AdamW(self.parameters(), lr=args.lr, weight_decay=0.001)
                    else:
                        # Default to Adam if unknown
                        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
                    
                    # Create scheduler based on type
                    print(f"Creating fresh scheduler ({lr_scheduler_type}) for {max_epochs} epochs")
                    if lr_scheduler_type.lower() == 'cosineannealinglr':
                        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
                    elif lr_scheduler_type.lower() == 'reducelronplateau':
                        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True)
                    elif lr_scheduler_type.lower() == 'cosineannealingwarmrestarts':
                        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
                    else:
                        # Default to CosineAnnealingLR if unknown
                        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
                    
                    # Return the optimizer and scheduler configuration
                    if lr_scheduler_type.lower() == 'reducelronplateau':
                        return {
                            'optimizer': optimizer,
                            'lr_scheduler': {
                                'scheduler': scheduler,
                                'monitor': 'val_loss',
                                'interval': 'epoch',
                                'frequency': 1
                            }
                        }
                    else:
                        return {
                            'optimizer': optimizer,
                            'lr_scheduler': scheduler
                        }
                
                # Bind the new method to the model instance
                model.configure_optimizers = types.MethodType(new_configure_optimizers, model)
                
                # Update the model's hparams to ensure consistency
                if hasattr(model, 'hparams'):
                    model.hparams.learning_rate = args.lr
                    if hasattr(model.hparams, 'optimizer_type'):
                        model.hparams.optimizer_type = optimizer_type
                    if hasattr(model.hparams, 'lr_scheduler_type'):
                        model.hparams.lr_scheduler_type = lr_scheduler_type
                
                print(f"Successfully loaded checkpoint from fold {fold-1} with reset optimizer/scheduler")
            except Exception as e:
                print(f"Error loading previous checkpoint: {e}")
                traceback.print_exc()
                print("ERROR: Could not load previous fold's checkpoint. Stopping training.")
                sys.exit(1)  # Exit with error code
        else:
            print(f"ERROR: No checkpoint found for fold {fold-1} in {prev_fold_dir}")
            print("Stopping training to avoid wasting resources. Please ensure previous fold completed successfully.")
            sys.exit(1)  # Exit with error code
    
    # Print regional focus confirmation
    if current_use_regional_focus:
        print(f"\n--- Regional Focus Configuration for Fold {fold} ---")
        print(f"Regional Focus Enabled: {current_use_regional_focus}")
        print(f"Region Weight: {current_region_weight}")
        print(f"Outside Weight: {current_outside_weight}")
        print(f"Target Source: {args.target_source if hasattr(args, 'target_source') else 'mswep'}")
        print("-" * 20)
    
    # Proceed with training using the time-series split approach
    trained_model_results, best_model_path = train_unet_model(
        data_module=data_module,
        model=model,
        output_dir=fold_output_dir,
        log_dir=log_dir_for_fold, # Pass the run-specific log directory (e.g., <log_dir>/run_<timestamp>)
        epochs=args.epochs,
        accelerator=HARDCODED_ACCELERATOR,
        devices=HARDCODED_DEVICES,
        fold=fold,
        precipitation_threshold=args.threshold,
        generate_plots=True,
        patience=HARDCODED_PATIENCE,
        track_epochs=HARDCODED_TRACK_EPOCHS,
        evaluate_raw=False,
        save_predictions=True  # Always save predictions for offline calibration
    )
    
    # Add fold explicitly to ensure it's included
    trained_model_results['fold'] = fold
    
    if best_model_path:
        print(f"Using best model path: {best_model_path} (type: {type(best_model_path)})")
    else:
        print("No best model path returned, possibly due to training error or early termination")
        
    print(f"Fold {fold} training complete. Predictions saved for offline calibration.")
    
    return trained_model_results

def calculate_final_crps(output_dir, num_folds, skip_crps=False):
    """Calculate the final CRPS across all folds"""
    if skip_crps:
        print("Skipping CRPS calculation as requested.")
        return
    
    print(f"Calculating final CRPS across {num_folds} folds")
    import numpy as np
    import multiprocessing
    from tqdm import tqdm
    
    # --- OPTIMIZATION: Constants for recent data selection ---
    N_RECENT_SAMPLES = 1000  # Example: Use last 1000 training samples for IDR fitting
    # ------------------------------------------------------
    
    try:
        from models.mswep_evaluation import calculate_crps_idr, run_evaluation
        
        # Find the most recent run directory by sorting based on timestamp
        run_dirs = [d for d in os.listdir(output_dir) if d.startswith("run_")]
        if not run_dirs:
            print("No run directories found. Ensure the output directory is correct.")
            return
            
        # Sort by run timestamp (newest first)
        run_dirs.sort(reverse=True)
        latest_run_dir = os.path.join(output_dir, run_dirs[0])
        print(f"Calculating CRPS for most recent run: {latest_run_dir}")
        
        # Process each fold
        for fold in range(num_folds):
            # Use the fold directory in the latest run
            fold_dir = os.path.join(latest_run_dir, f"fold{fold}")
            
            print(f"\nProcessing fold {fold}...")
            
            # Check if directory exists
            if not os.path.exists(fold_dir):
                print(f"  Error: Could not find fold directory: {fold_dir}")
                continue
                
            # Check for necessary files
            if not os.path.exists(os.path.join(fold_dir, "val_preds.npy")):
                print(f"  Error: Could not find prediction files in {fold_dir}")
                continue
            
            # Load saved data
            try:
                val_preds = np.load(os.path.join(fold_dir, "val_preds.npy"))
                val_targets = np.load(os.path.join(fold_dir, "val_targets.npy"))
                train_preds = np.load(os.path.join(fold_dir, "train_preds_all.npy"))
                train_targets = np.load(os.path.join(fold_dir, "train_targets_all.npy"))
                region_mask = np.load(os.path.join(fold_dir, "germany_mask.npy"))
                
                print(f"  Loaded data shapes:")
                print(f"    Val preds: {val_preds.shape}")
                print(f"    Val targets: {val_targets.shape}")
                print(f"    Train preds (all): {train_preds.shape}")
                print(f"    Train targets (all): {train_targets.shape}")
                print(f"    Region mask: {region_mask.shape}")
                
                # Add scale verification to ensure both predictions and targets are in original scale (mm)
                print(f"  Verifying data scales (all values should be in mm):")
                print(f"    Val preds - Min: {val_preds.min():.4f}, Max: {val_preds.max():.4f}, Mean: {val_preds.mean():.4f}")
                print(f"    Val targets - Min: {val_targets.min():.4f}, Max: {val_targets.max():.4f}, Mean: {val_targets.mean():.4f}")
                print(f"    Train preds - Min: {train_preds.min():.4f}, Max: {train_preds.max():.4f}, Mean: {train_preds.mean():.4f}")
                print(f"    Train targets - Min: {train_targets.min():.4f}, Max: {train_targets.max():.4f}, Mean: {train_targets.mean():.4f}")
                
                # --- OPTIMIZATION: Select most recent training samples for IDR fitting ---
                grid_lat, grid_lon = region_mask.shape
                print(f"  Optimization: Using most recent {N_RECENT_SAMPLES} training samples for IDR fitting.")
                train_preds_recent = train_preds[-N_RECENT_SAMPLES:, ...]
                train_targets_recent = train_targets[-N_RECENT_SAMPLES:, ...]
                print(f"    Train preds (recent): {train_preds_recent.shape}")
                # ------------------------------------------------------
                
                # Set up pool for parallel processing
                # Use a smaller number of processes to avoid excessive memory usage
                num_processes = min(4, multiprocessing.cpu_count())
                pool = multiprocessing.Pool(processes=num_processes)
                
                # Create arguments list for each grid cell
                args_list = [
                    (val_preds, val_targets, 
                     train_preds_recent, train_targets_recent,  # <-- Use optimized recent data
                     region_mask, i, j)
                    for i in range(grid_lat) for j in range(grid_lon)
                ]
                
                # Set up progress bar
                pbar = tqdm(total=len(args_list), desc=f"Calculating CRPS for fold {fold}")
                def update_pbar(*args):
                    pbar.update()
                
                # Submit jobs to the process pool
                crps_results = []
                for args_item in args_list:
                    result = pool.apply_async(calculate_crps_idr, args=args_item, callback=update_pbar)
                    crps_results.append(result)
                
                pool.close()
                pool.join()
                pbar.close()
                
                crps_list = [res.get() for res in crps_results if res.get() is not None]
                if crps_list:
                    mean_masked_crps = np.mean(crps_list)
                    print(f"  Mean masked CRPS for fold {fold}: {mean_masked_crps:.6f}")
                    
                    # Save the result to a file
                    with open(os.path.join(fold_dir, "final_crps.txt"), "w") as f:
                        f.write(f"Mean masked CRPS: {mean_masked_crps:.6f}\n")
                    
                    # Save detailed CRPS results if needed
                    np.save(os.path.join(fold_dir, "crps_results.npy"), np.array(crps_list))
                else:
                    print(f"  No valid grid cells found for CRPS calculation in fold {fold}")
            
            except FileNotFoundError as e:
                print(f"  Error: Missing data file in fold {fold}: {e}")
            except Exception as e:
                print(f"  Error processing fold {fold}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n=== Final CRPS calculation complete ===")
        
        # Run extended evaluation to generate all plots and metrics
        print("\n=== Running extended evaluation on all folds ===")
        for fold in range(num_folds):
            print(f"\nRunning full evaluation for fold {fold}...")
            try:
                # Run the extended evaluation functions from mswep_evaluation.py
                # Pass a smaller batch size to avoid memory issues
                evaluation_results = run_evaluation(latest_run_dir, fold, batch_size=50)
                if evaluation_results:
                    print(f"Extended evaluation complete for fold {fold}")
                else:
                    print(f"Warning: Extended evaluation returned no results for fold {fold}")
            except Exception as e:
                print(f"Error during extended evaluation for fold {fold}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n=== All evaluations complete ===")
    
    except ImportError:
        print("Error: Could not import calculate_crps_idr. CRPS calculation skipped.")
        return
    except Exception as e:
        print(f"Error in final CRPS calculation: {e}")
        import traceback
        traceback.print_exc()
        return

# Execute main function
if __name__ == "__main__": 

    def main():
        # Enable GPU performance optimizations
        # These settings should help maximize throughput on Tensor Core GPUs like the RTX 3060 Ti
        torch.set_float32_matmul_precision('high')  # Enable tensor cores
        torch.backends.cudnn.benchmark = True       # Enable cudnn autotuner
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
        
        # Define hardcoded values
        global HARDCODED_BATCH_SIZE, HARDCODED_NUM_WORKERS, HARDCODED_ACCELERATOR, HARDCODED_DEVICES
        global HARDCODED_APPLY_LOG, HARDCODED_LOG_OFFSET, HARDCODED_PATIENCE, HARDCODED_TRACK_EPOCHS
        global HARDCODED_EVAL_INTENSITY, HARDCODED_ENABLE_EASYUQ
        
        HARDCODED_BATCH_SIZE = 32
        HARDCODED_NUM_WORKERS = 14
        HARDCODED_ACCELERATOR = 'gpu'
        HARDCODED_DEVICES = [0]  # Assuming single GPU setup
        HARDCODED_APPLY_LOG = True
        HARDCODED_LOG_OFFSET = 0.01
        HARDCODED_PATIENCE = 10
        HARDCODED_TRACK_EPOCHS = True
        HARDCODED_EVAL_INTENSITY = True
        HARDCODED_ENABLE_EASYUQ = True
        
        # Log GPU information and optimization settings
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"PyTorch CUDA: {torch.version.cuda}")
            print(f"Tensor Cores enabled: {torch.get_float32_matmul_precision() != 'highest'}")
            print(f"CuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        
        parser = argparse.ArgumentParser(description='Train a UNet model for MSWEP precipitation forecasting.')
        parser.add_argument('--data_dir', type=str, help='Directory containing the MSWEP data files')
        parser.add_argument('--test_data_dir', type=str, help='Optional directory for test data files', default=None)
        parser.add_argument('--output_dir', type=str, help='Directory to save results')
        parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save training logs (defaults to ./logs)')
        parser.add_argument('--epochs', type=int, default=10, help='Maximum number of epochs for training')
        parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
        parser.add_argument('--threshold', type=float, default=0.2, help='Precipitation threshold for evaluation')
        parser.add_argument('--folds', type=int, default=3, help='Number of folds to use')
        parser.add_argument('--loss_type', type=str, default='mse', 
                            choices=['mse', 'weighted_mse', 'huber', 'focal_mse', 'asymmetric_mse', 'log_cosh'],
                            help='Type of loss function to use')
        parser.add_argument('--optimizer_type', type=str, default='adam', choices=['adam', 'adamw'],
                            help='Type of optimizer to use (adam or adamw)')
        parser.add_argument('--lr_scheduler_type', type=str, default='cosineannealinglr',
                            choices=['cosineannealinglr', 'cosineannealingwarmrestarts', 'reducelronplateau'],
                            help='Type of learning rate scheduler to use (cosineannealinglr, cosineannealingwarmrestarts, or reducelronplateau)')
        parser.add_argument('--target_source', 
                        type=str, 
                        default='mswep', 
                        choices=['mswep', 'hyras'], 
                        help='Target dataset source (mswep or hyras - assumes hyras is pre-regridded)')
        # Add shared parameters that need to be consistent
        parser.add_argument('--use_regional_focus', type=lambda x: bool(distutils.util.strtobool(x)), default=True,
                        help='Whether to apply regional weighting to the loss calculation (True/False)')
        parser.add_argument('--region_weight', type=float, default=1.0,
                        help='Weight for target region (Germany) - fixed at 1.0 and not configurable')
        parser.add_argument('--outside_weight', type=float, default=0.2,
                        help='Weight for areas outside Germany (range: 0.0-1.0, default: 0.2)')
        parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer (L2 regularization)')
        parser.add_argument('--log_offset', type=float, default=0.01,
                        help='Offset to use in log transform of precipitation data')
        parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate for UNet model')
        parser.add_argument('--apply_transform', type=bool, default=True,
                        help='Whether to apply random transformations (e.g., Gaussian blur)')
        parser.add_argument('--transform_probability', type=float, default=0.5,
                        help='Probability of applying transformations')
        parser.add_argument('--skip_crps', type=bool, default=False,
                        help='Skip CRPS calculation to save time during hyperparameter search')
        
        args = parser.parse_args()
        
        # Create a timestamped run directory for output
        from datetime import datetime
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(args.output_dir, f"run_{run_timestamp}")
        os.makedirs(run_output_dir, exist_ok=True)
        print(f"Creating unique run output directory: {run_output_dir}")
        
        # Create a timestamped directory for logs within the specified log_dir
        run_log_dir = os.path.join(args.log_dir, f"run_{run_timestamp}")
        os.makedirs(run_log_dir, exist_ok=True)
        print(f"Log directory for this run: {run_log_dir}")
        
        print(f"Starting training with {args.folds} folds.")
        for fold in range(args.folds):
            print(f"\n=== Training fold {fold} ===")
            fold_output_dir = os.path.join(run_output_dir, f"fold{fold}")
            # Pass the run-specific log directory to train_fold
            train_fold(fold, args, fold_output_dir, run_log_dir)
        
        # After all folds are complete, calculate final CRPS with the run directory
        calculate_final_crps(args.output_dir, args.folds, args.skip_crps)

    main()
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
import warnings
# Suppress the specific deprecation warning from PyTorch about float32_matmul_precision
warnings.filterwarnings('ignore', message='.*float32_matmul_precision.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*This API is going to be deprecated.*', category=UserWarning)
import torch
# Set float32 matmul precision immediately after importing torch to prevent warnings
# This must be done before importing any other PyTorch modules
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')  # Use tensor cores for better performance
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
import pandas as pd

# Set matplotlib backend to 'Agg' (non-interactive) to prevent Tkinter threading issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as MatplotlibColors # Renamed to avoid conflict

from torchmetrics import MeanAbsoluteError, MeanSquaredError
import torch.nn.functional as F

# Add the project root directory to Python path to enable imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from data.mswep_data_module_2 import MSWEPDataModule, TargetLogScaler
from models.mswep_lightning_wrapper import UNetLightningModule
from models.mswep_unet import MSWEPUNet

# Import evaluation utilities from mswep_evaluation
try:
    from models.mswep_evaluation import (
        apply_easyuq_per_cell,
        calculate_brier_score_for_cell,
        brier_score_decomposition,      # Added for comprehensive evaluation
        calculate_seeps_scores,         # Added for comprehensive evaluation
        numpy_json_encoder,
        plot_det_vs_prob_comparison,
        plot_seasonal_metrics,
        plot_quantile_map,              # Added for comprehensive evaluation visualization
        plot_sample,                    # For replacing generate_prediction_plots
        plot_seasonal_samples,          # For the last fold seasonal plots
        DEFAULT_LATITUDES, DEFAULT_LONGITUDES, XINC, YINC,
        GERMANY_PLOT_LON_MIN, GERMANY_PLOT_LON_MAX, GERMANY_PLOT_LAT_MIN, GERMANY_PLOT_LAT_MAX,
        GERMANY_BOX_GRID_LON_INDICES, GERMANY_BOX_GRID_LAT_INDICES # Needed by plot_sample
    )
except ImportError as e:
    print(f"Could not import all required functions from models.mswep_evaluation: {e}")
    print("Ensure mswep_evaluation.py is in the correct path and has all necessary functions.")
    # Define dummy functions or pass if some are critical for script to run partially.
    def plot_sample(*args, **kwargs): print("Warning: plot_sample (from mswep_evaluation) not available.")
    def plot_quantile_map(*args, **kwargs): print("Warning: plot_quantile_map (from mswep_evaluation) not available.")

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
                    f"TF32: Automatically optimized for modern GPUs<br>"
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
    """Tracks MAE and RMSE from the epoch with the best validation loss (both scales)."""
    def __init__(self, monitor='val_loss', mode='min'):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_mae_log = None
        self.best_rmse_log = None
        self.best_mae_mm = None
        self.best_rmse_mm = None

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
            self.best_mae_log = logs.get('val_mae_log')
            self.best_rmse_log = logs.get('val_rmse_log')
            self.best_mae_mm = logs.get('val_mae_mm')
            self.best_rmse_mm = logs.get('val_rmse_mm')
            print(f"\n[Best Metrics] New best val_loss: {self.best_score:.4f}")
            print(f"  -> Log scale: MAE={self.best_mae_log:.4f}, RMSE={self.best_rmse_log:.4f}")
            print(f"  -> Original scale (mm): MAE={self.best_mae_mm:.4f}, RMSE={self.best_rmse_mm:.4f}\n")

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
                if hasattr(pl_module, 'target_scaler') and pl_module.target_scaler is not None:
                    y_hat = pl_module.target_scaler.inverse_transform(y_hat)
                    print(f"Applied inverse transform to predictions. Min: {y_hat.min().item():.4f}, Max: {y_hat.max().item():.4f}")
                else:
                    print("Warning: No target_scaler found on model. Metrics may be incorrect if scales don't match.")
                    # If we're in log scale but no scaler is available, try to detect and warn
                    if y_hat.max().item() < 10 and y_hat.min().item() < 0:  # Likely in log scale
                        print("WARNING: Predictions appear to be in log scale but no scaler available to transform back!")
                
                # Calculate metrics
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

def verify_temporal_alignment(preds_path, targets_path, times_path=None, sample_cells=5):
    """
    Verify that predictions and targets are temporally aligned.
    
    This function checks:
    1. Shape consistency
    2. Temporal autocorrelation patterns
    3. Value distributions
    
    Args:
        preds_path: Path to predictions numpy file
        targets_path: Path to targets numpy file
        times_path: Optional path to timestamps file
        sample_cells: Number of random cells to check for temporal patterns
        
    Returns:
        dict: Verification results
    """
    print("\n" + "="*70)
    print("VERIFYING TEMPORAL ALIGNMENT")
    print("="*70)
    
    # Load data
    preds = np.load(preds_path)
    targets = np.load(targets_path)
    
    results = {
        'shapes_match': preds.shape == targets.shape,
        'preds_shape': preds.shape,
        'targets_shape': targets.shape
    }
    
    if not results['shapes_match']:
        print(f"❌ SHAPE MISMATCH: preds {preds.shape} != targets {targets.shape}")
        return results
    
    print(f"✓ Shapes match: {preds.shape}")
    
    # Check temporal autocorrelation for random cells
    if len(preds.shape) == 3:  # (time, lat, lon)
        n_time, n_lat, n_lon = preds.shape
        
        # Sample random cells
        np.random.seed(42)
        sample_lats = np.random.randint(0, n_lat, sample_cells)
        sample_lons = np.random.randint(0, n_lon, sample_cells)
        
        autocorr_preds = []
        autocorr_targets = []
        
        for lat, lon in zip(sample_lats, sample_lons):
            # Get time series for this cell
            pred_series = preds[:, lat, lon]
            target_series = targets[:, lat, lon]
            
            # Calculate lag-1 autocorrelation
            if len(pred_series) > 1:
                pred_ac = np.corrcoef(pred_series[:-1], pred_series[1:])[0, 1]
                target_ac = np.corrcoef(target_series[:-1], target_series[1:])[0, 1]
                
                autocorr_preds.append(pred_ac)
                autocorr_targets.append(target_ac)
        
        results['mean_autocorr_preds'] = np.mean(autocorr_preds)
        results['mean_autocorr_targets'] = np.mean(autocorr_targets)
        
        print(f"✓ Mean lag-1 autocorrelation - Preds: {results['mean_autocorr_preds']:.3f}, "
              f"Targets: {results['mean_autocorr_targets']:.3f}")
        
        # Check if autocorrelations are similar (they should be for aligned data)
        autocorr_diff = abs(results['mean_autocorr_preds'] - results['mean_autocorr_targets'])
        if autocorr_diff > 0.2:
            print(f"⚠️  Large autocorrelation difference ({autocorr_diff:.3f}) may indicate misalignment")
        
    # Check value distributions
    results['preds_mean'] = np.mean(preds)
    results['targets_mean'] = np.mean(targets)
    results['preds_std'] = np.std(preds)
    results['targets_std'] = np.std(targets)
    
    print(f"✓ Value statistics - Preds: mean={results['preds_mean']:.3f}, std={results['preds_std']:.3f}")
    print(f"                    Targets: mean={results['targets_mean']:.3f}, std={results['targets_std']:.3f}")
    
    # Load and check timestamps if provided
    if times_path and os.path.exists(times_path):
        times = np.load(times_path, allow_pickle=True)
        results['n_timestamps'] = len(times)
        results['timestamps_match_samples'] = len(times) == preds.shape[0]
        
        if results['timestamps_match_samples']:
            print(f"✓ Timestamps match samples: {len(times)} timestamps")
        else:
            print(f"❌ Timestamp mismatch: {len(times)} timestamps vs {preds.shape[0]} samples")
    
    print("="*70 + "\n")
    
    return results

def train_unet_model(data_module, model, output_dir, log_dir, epochs, accelerator='auto', 
                   devices=None, fold=0, precipitation_threshold=0.2, generate_plots=True,
                   patience=30, loss_type='mse', intensity_weights=None, focal_gamma=2.0, 
                   track_epochs=False, evaluate_raw=False, save_predictions=True,
                   args_all=None, is_last_fold=False):
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
        args_all: All command line arguments passed to the script
        is_last_fold: Whether this is the last fold being trained
    
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
    from torch.utils.data import DataLoader  # Add missing import
    from tqdm import tqdm  # Add tqdm import for progress bar
    
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
    final_best_mae_log = model.best_val_mae_log.item() if hasattr(model, 'best_val_mae_log') and model.best_val_mae_log is not None else None
    final_best_rmse_log = model.best_val_rmse_log.item() if hasattr(model, 'best_val_rmse_log') and model.best_val_rmse_log is not None else None
    final_best_mae_mm = model.best_val_mae_mm.item() if hasattr(model, 'best_val_mae_mm') and model.best_val_mae_mm is not None else None
    final_best_rmse_mm = model.best_val_rmse_mm.item() if hasattr(model, 'best_val_rmse_mm') and model.best_val_rmse_mm is not None else None
    # -------------------------------------------------------------

    # Save predictions BEFORE loading the best checkpoint (except for final fold)
    if save_predictions:
        # Use the provided output directory directly
        actual_save_dir = fold_dir
        
        if is_last_fold:
            print(f"\nFinal fold detected - will collect aligned predictions AFTER loading best checkpoint")
            print("This ensures temporal alignment for accurate IDR fitting")
            
            # Still save timestamps and mask for final fold
            try:
                # Save validation timestamps
                if hasattr(data_module, 'val_times') and data_module.val_times is not None:
                    val_times_all = data_module.val_times
                    np.save(os.path.join(actual_save_dir, "val_times.npy"), val_times_all)
                    print(f"  Saved val_times shape: {val_times_all.shape}")
                
                # Save model's Germany mask if available (for later CRPS calculation)
                if hasattr(model, 'model') and hasattr(model.model, 'germany_mask') and model.model.germany_mask is not None:
                    germany_mask_np = model.model.germany_mask.cpu().numpy()
                    save_path_mask = os.path.join(actual_save_dir, "germany_mask.npy")
                    np.save(save_path_mask, germany_mask_np)
                    print(f"  Saved germany_mask to {save_path_mask}, Shape: {germany_mask_np.shape}, Number of True values: {np.sum(germany_mask_np)}")
            except Exception as e:
                print(f"Error saving timestamps/mask: {e}")
                traceback.print_exc()
        else:
            # For non-final folds, save accumulated predictions (with warning about potential misalignment)
            print(f"Saving predictions and targets for fold {fold} (non-final fold)")
            print("WARNING: These may have temporal misalignment issues")
            
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
                    val_times_all = data_module.val_times
                    np.save(os.path.join(actual_save_dir, "val_times.npy"), val_times_all)
                    print(f"  Saved val_times shape: {val_times_all.shape}")
                else:
                    print("Warning: Could not find val_times on data_module to save.")

                # Save model's Germany mask if available (for later CRPS calculation)
                if hasattr(model, 'model') and hasattr(model.model, 'germany_mask') and model.model.germany_mask is not None:
                    germany_mask_np = model.model.germany_mask.cpu().numpy()
                    save_path_mask = os.path.join(actual_save_dir, "germany_mask.npy")
                    np.save(save_path_mask, germany_mask_np)
                    print(f"  Saved germany_mask to {save_path_mask}, Shape: {germany_mask_np.shape}, Number of True values: {np.sum(germany_mask_np)}")
                else:
                    print("Warning: Could not find germany_mask on model to save during training.")

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
            # Get the correct number of input channels from args if available
            total_channels = args_all.total_input_channels if (args_all and hasattr(args_all, 'total_input_channels')) else 5
            print(f"Loading best checkpoint: Creating UNet with {total_channels} input channels")
            unet_model_for_load = MSWEPUNet(in_channels=total_channels)

            # Load the checkpoint state into the Lightning Module wrapper
            # Pass the correctly initialized inner model
            # Also pass other necessary hyperparameters if they might have changed
            # or if loading requires them (e.g., learning_rate, loss_type etc. might
            # be needed by the wrapper's __init__ even when loading state)
            # Assuming 'model' arg is sufficient for structure, and others are in hparams:
            loaded_model = UNetLightningModule.load_from_checkpoint(
                best_model_path,
                model=unet_model_for_load,
                target_scaler=data_module.target_scaler if hasattr(data_module, 'target_scaler') else None
                # Add any other essential args here if load_from_checkpoint needs them
                # e.g., learning_rate=model.hparams.learning_rate (or args.lr) if needed
            )
            
            # CRITICAL FIX: Ensure target_scaler is properly set after loading
            # Since target_scaler is not saved in hyperparameters, we need to set it explicitly
            if hasattr(data_module, 'target_scaler') and data_module.target_scaler is not None:
                loaded_model.target_scaler = data_module.target_scaler
                print(f"Restored target_scaler to loaded model with offset={data_module.target_scaler.offset}")
            else:
                print("WARNING: No target_scaler found in data_module to restore to loaded model")
            
            loaded_model.eval()
            print("Loaded best model checkpoint successfully.")
            model = loaded_model # Use the loaded model going forward if successful
            
            # =================================================================
            # NEW CODE: Run inference with best model on training set to collect 
            # properly aligned predictions and targets for IDR fitting
            # ONLY ON THE FINAL FOLD for maximum accuracy
            # =================================================================
            if save_predictions and is_last_fold:
                print("\n" + "="*70)
                print("FINAL FOLD DETECTED: Collecting temporally aligned data for IDR fitting")
                print("This ensures the most accurate probabilistic calibration")
                print("="*70 + "\n")
                
                try:
                    # Create a DataLoader for the training set
                    train_loader = DataLoader(
                        data_module.train_dataset,
                        batch_size=data_module.batch_size,
                        shuffle=False,  # CRITICAL: Keep the original order for temporal alignment
                        num_workers=data_module.num_workers,
                        pin_memory=True
                    )
                    
                    # Collect predictions and targets
                    all_train_preds = []
                    all_train_targets = []
                    
                    # Put model in evaluation mode and on correct device
                    model.eval()
                    device = next(model.parameters()).device
                    
                    with torch.no_grad():
                        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Collecting aligned training data (FINAL FOLD)")):
                            # Get inputs and targets from batch (unpack all three elements)
                            x, y_original, _ = batch
                            x = x.to(device)
                            
                            # Forward pass
                            y_hat = model(x)
                            
                            # Squeeze prediction if necessary
                            if y_hat.dim() == 4 and y_hat.shape[1] == 1:
                                y_hat = y_hat.squeeze(1)
                            
                            # Apply inverse transform to get predictions back to original scale
                            if hasattr(model, 'target_scaler') and model.target_scaler is not None:
                                y_hat_rescaled = model.target_scaler.inverse_transform(y_hat.detach())
                            else:
                                # No target scaler - predictions are already in original scale
                                print("WARNING: No target_scaler found on model during aligned data collection")
                                y_hat_rescaled = y_hat.detach()
                            
                            # Store predictions and targets
                            all_train_preds.append(y_hat_rescaled.cpu())
                            all_train_targets.append(y_original.cpu())
                            
                            # Print progress occasionally
                            if batch_idx % 50 == 0:
                                print(f"  Processed {batch_idx} batches. Current shapes: "
                                      f"preds={all_train_preds[-1].shape}, targets={all_train_targets[-1].shape}")
                    
                    # Concatenate all predictions and targets
                    train_preds_all = torch.cat(all_train_preds, dim=0).numpy()
                    train_targets_all = torch.cat(all_train_targets, dim=0).numpy()
                    
                    # Verify shapes match (CRITICAL for IDR fitting)
                    assert train_preds_all.shape == train_targets_all.shape, (
                        f"CRITICAL ERROR: Shape mismatch between predictions {train_preds_all.shape} "
                        f"and targets {train_targets_all.shape}. This would break IDR fitting!"
                    )
                    
                    # Save the aligned arrays
                    np.save(os.path.join(actual_save_dir, "train_preds_all.npy"), train_preds_all)
                    np.save(os.path.join(actual_save_dir, "train_targets_all.npy"), train_targets_all)
                    print(f"\n✓ Saved ALIGNED train_preds_all shape: {train_preds_all.shape}")
                    print(f"✓ Saved ALIGNED train_targets_all shape: {train_targets_all.shape}")
                    print("✓ These arrays are perfectly aligned for IDR fitting\n")
                    
                    # Also collect aligned validation data with the best model
                    print("Collecting aligned validation data with best model...")
                    val_loader = data_module.val_dataloader()
                    all_val_preds_aligned = []
                    all_val_targets_aligned = []
                    
                    with torch.no_grad():
                        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Collecting aligned validation data")):
                            x, y_original, _ = batch
                            x = x.to(device)
                            
                            y_hat = model(x)
                            if y_hat.dim() == 4 and y_hat.shape[1] == 1:
                                y_hat = y_hat.squeeze(1)
                            
                            if hasattr(model, 'target_scaler') and model.target_scaler is not None:
                                y_hat_rescaled = model.target_scaler.inverse_transform(y_hat.detach())
                            else:
                                # No target scaler - predictions are already in original scale
                                y_hat_rescaled = y_hat.detach()
                            
                            all_val_preds_aligned.append(y_hat_rescaled.cpu())
                            all_val_targets_aligned.append(y_original.cpu())
                    
                    # Save aligned validation data (overwriting the accumulated data)
                    val_preds_aligned = torch.cat(all_val_preds_aligned, dim=0).numpy()
                    val_targets_aligned = torch.cat(all_val_targets_aligned, dim=0).numpy()
                    
                    np.save(os.path.join(actual_save_dir, "val_preds.npy"), val_preds_aligned)
                    np.save(os.path.join(actual_save_dir, "val_targets.npy"), val_targets_aligned)
                    print(f"✓ Saved ALIGNED val_preds shape: {val_preds_aligned.shape} (replaced accumulated data)")
                    print(f"✓ Saved ALIGNED val_targets shape: {val_targets_aligned.shape}")
                    
                    # Verify temporal alignment
                    print("\nVerifying temporal alignment of saved data...")
                    train_verification = verify_temporal_alignment(
                        os.path.join(actual_save_dir, "train_preds_all.npy"),
                        os.path.join(actual_save_dir, "train_targets_all.npy")
                    )
                    val_verification = verify_temporal_alignment(
                        os.path.join(actual_save_dir, "val_preds.npy"),
                        os.path.join(actual_save_dir, "val_targets.npy"),
                        os.path.join(actual_save_dir, "val_times.npy")
                    )
                    
                    # Save verification results
                    verification_results = {
                        'train': train_verification,
                        'validation': val_verification,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    with open(os.path.join(actual_save_dir, "temporal_alignment_verification.json"), 'w') as f:
                        json.dump(verification_results, f, indent=2, default=convert_to_json_serializable)
                    print(f"✓ Saved temporal alignment verification to: {os.path.join(actual_save_dir, 'temporal_alignment_verification.json')}")
                    
                    # Free memory
                    del all_train_preds, all_train_targets, train_preds_all, train_targets_all
                    del all_val_preds_aligned, all_val_targets_aligned, val_preds_aligned, val_targets_aligned
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"ERROR during aligned data collection: {str(e)}")
                    traceback.print_exc()
                    print("Falling back to accumulated predictions from training...")
                    
                    # Fall back to accumulated predictions if aligned collection fails
                    if hasattr(model, 'validation_step_preds') and model.validation_step_preds:
                        print("Saving accumulated validation predictions (may have temporal misalignment)")
                        val_preds_all = torch.cat(model.validation_step_preds).cpu().numpy()
                        np.save(os.path.join(actual_save_dir, "val_preds.npy"), val_preds_all)
                        model.validation_step_preds.clear()
                        
                    if hasattr(model, 'validation_step_targets') and model.validation_step_targets:
                        val_targets_all = torch.cat(model.validation_step_targets).cpu().numpy()
                        np.save(os.path.join(actual_save_dir, "val_targets.npy"), val_targets_all)
                        model.validation_step_targets.clear()
                    
            elif save_predictions and not is_last_fold:
                print(f"\nFold {fold} is not the final fold. Skipping aligned data collection for IDR.")
                print("IDR fitting will use data collected during training (may have temporal misalignment).")
                
                # For non-final folds, just save what we have (with the understanding it may be misaligned)
                # This maintains backward compatibility but with a warning
                if hasattr(model, 'validation_step_preds') and model.validation_step_preds:
                    print("WARNING: Using accumulated validation predictions which may have temporal misalignment!")
                    val_preds_all = torch.cat(model.validation_step_preds).cpu().numpy()
                    np.save(os.path.join(actual_save_dir, "val_preds.npy"), val_preds_all)
                    
                if hasattr(model, 'validation_step_targets') and model.validation_step_targets:
                    val_targets_all = torch.cat(model.validation_step_targets).cpu().numpy()
                    np.save(os.path.join(actual_save_dir, "val_targets.npy"), val_targets_all)
                
            # =================================================================
            # END NEW CODE
            # =================================================================
            
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
    
    # NEW: Ensure we save predictions even if best model loading failed
    if save_predictions and is_last_fold and hasattr(model, 'validation_step_preds'):
        # Check if we already saved predictions during the try block
        val_preds_path = os.path.join(actual_save_dir, "val_preds.npy")
        if not os.path.exists(val_preds_path):
            print("\nBest model loading failed but attempting to save predictions from current model state...")
            if model.validation_step_preds:
                try:
                    val_preds_all = torch.cat(model.validation_step_preds).cpu().numpy()
                    np.save(val_preds_path, val_preds_all)
                    print(f"Saved validation predictions shape: {val_preds_all.shape}")
                    model.validation_step_preds.clear()
                except Exception as e:
                    print(f"Error saving validation predictions: {e}")
            
            if model.validation_step_targets:
                try:
                    val_targets_all = torch.cat(model.validation_step_targets).cpu().numpy()
                    np.save(os.path.join(actual_save_dir, "val_targets.npy"), val_targets_all)
                    print(f"Saved validation targets shape: {val_targets_all.shape}")
                    model.validation_step_targets.clear()
                except Exception as e:
                    print(f"Error saving validation targets: {e}")

    # 9. Evaluate raw model performance if requested
    results = {}
    results['best_model_path'] = best_model_path
    results['fold'] = fold
    
    # Get the best validation loss from the checkpoint callback (most reliable source)
    best_val_loss = checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score is not None else float('inf')
    results['best_val_loss'] = best_val_loss

    # Get the corresponding MAE and RMSE tracked within the LightningModule
    # Use the values captured *before* checkpoint loading
    results['best_val_mae'] = final_best_mae_log
    results['best_val_rmse'] = final_best_rmse_log
    results['best_val_mae_mm'] = final_best_mae_mm
    results['best_val_rmse_mm'] = final_best_rmse_mm

    # Prepare values for printing, handling None
    loss_str = f"{results['best_val_loss']:.4f}" if results['best_val_loss'] is not None else "N/A"
    mae_str = f"{results['best_val_mae']:.4f}" if results['best_val_mae'] is not None else "N/A"
    rmse_str = f"{results['best_val_rmse']:.4f}" if results['best_val_rmse'] is not None else "N/A"
    mae_mm_str = f"{results['best_val_mae_mm']:.4f}" if results['best_val_mae_mm'] is not None else "N/A"
    rmse_mm_str = f"{results['best_val_rmse_mm']:.4f}" if results['best_val_rmse_mm'] is not None else "N/A"
    print(f"Retrieved best metrics: Loss={loss_str}, MAE={mae_str}, RMSE={rmse_str}, MAE_mm={mae_mm_str}, RMSE_mm={rmse_mm_str}")
    
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
                if hasattr(model, 'target_scaler') and model.target_scaler is not None:
                    y_hat_orig_scale = model.target_scaler.inverse_transform(y_hat)
                    print(f"Applied inverse transform for plotting. Original scale - "
                          f"Min: {y_hat_orig_scale.min().item():.4f}, Max: {y_hat_orig_scale.max().item():.4f}")
                else:
                    y_hat_orig_scale = y_hat
                    print("Warning: model has no target_scaler attribute or it's None. Plotting raw predictions.")
                    # If predictions are in log scale but no scaler, try to detect and transform
                    if y_hat.max().item() < 10 and y_hat.min().item() < 0:
                        print("Predictions appear to be in log scale. Attempting manual inverse transform with offset 0.1")
                        y_hat_orig_scale = torch.exp(y_hat) - 0.1
                        y_hat_orig_scale = torch.clamp(y_hat_orig_scale, min=0.0)
                        print(f"After manual transform - Min: {y_hat_orig_scale.min().item():.4f}, Max: {y_hat_orig_scale.max().item():.4f}")

            # Prepare inputs, target, and prediction for plotting
            # Ensure they are on CPU and get the first sample from the batch
            input_for_plot = x.cpu()[0]
            target_for_plot = y_original.cpu()[0]
            prediction_for_plot = y_hat_orig_scale.cpu()[0]

            # Squeeze the prediction if it has a leading singleton dimension
            if prediction_for_plot.ndim == 3 and prediction_for_plot.shape[0] == 1:
                prediction_for_plot = prediction_for_plot.squeeze(0)
            
            # Also ensure target is 2D, just in case (though less likely to be an issue here)
            if target_for_plot.ndim == 3 and target_for_plot.shape[0] == 1:
                target_for_plot = target_for_plot.squeeze(0)

            plot_sample(
                inputs=input_for_plot,
                target=target_for_plot,
                prediction=prediction_for_plot,
                title=f"Precipitation Forecast (Fold {fold})",
                save_path=os.path.join(actual_save_dir, f"prediction_examples_eu_fold{fold}.png"),
                lats=DEFAULT_LATITUDES,
                lons=DEFAULT_LONGITUDES
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
        
        # Add ERA5 configuration to results
        if args_all:
            results['era5_config'] = {
                'variables': args_all.era5_variables_list if hasattr(args_all, 'era5_variables_list') else None,
                'pressure_levels': args_all.era5_pressure_levels_list if hasattr(args_all, 'era5_pressure_levels_list') else None,
                'data_dir': args_all.era5_data_dir if hasattr(args_all, 'era5_data_dir') else None,
                'total_input_channels': args_all.total_input_channels if hasattr(args_all, 'total_input_channels') else 5
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
        from datetime import datetime
        run_timestamp_fallback = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir_local = os.path.join(args.output_dir, f"run_{run_timestamp_fallback}") 
        fold_output_dir = os.path.join(run_output_dir_local, f"fold{fold}")
    os.makedirs(fold_output_dir, exist_ok=True)
    
    log_dir_for_fold = run_log_dir
    if log_dir_for_fold is None:
        print("Warning: run_log_dir not provided to train_fold. Defaulting log path.")
        run_timestamp_fallback = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir_for_fold = os.path.join(args.log_dir or './logs', f"run_{run_timestamp_fallback}")
    os.makedirs(log_dir_for_fold, exist_ok=True)
    
    current_use_regional_focus = args.use_regional_focus
    current_region_weight = 1.0 
    current_outside_weight = args.outside_weight
    
    print(f"Fold {fold}: Regional focus settings:")
    print(f"  - Use regional focus: {current_use_regional_focus}")
    print(f"  - Germany region weight: {current_region_weight} (fixed)")
    print(f"  - Outside region weight: {current_outside_weight} (from command line)")
    
    data_module = MSWEPDataModule(
        data_dir=args.data_dir,
        test_data_dir=args.test_data_dir if hasattr(args, 'test_data_dir') else None,
        batch_size=HARDCODED_BATCH_SIZE, # Global or from args
        num_workers=HARDCODED_NUM_WORKERS, # Global or from args
        fold=fold,
        num_total_folds=args.folds, # Pass total folds to data module
        target_source=args.target_source if hasattr(args, 'target_source') else 'mswep',
        apply_log_transform=True,  # Enable log transform in data module
        log_offset=args.log_offset if hasattr(args, 'log_offset') else 0.1,  # Use 0.1 as default
        era5_variables=args.era5_variables_list if hasattr(args, 'era5_variables_list') else None,
        era5_pressure_levels=args.era5_pressure_levels_list if hasattr(args, 'era5_pressure_levels_list') else None,
        era5_data_dir=args.era5_data_dir if hasattr(args, 'era5_data_dir') else None,
        era5_single_level_variables=args.era5_single_level_variables_list if hasattr(args, 'era5_single_level_variables_list') else None
    )
    
    data_module.setup(stage='fit')
    
    print(f"Fold {fold} using year-based expanding window approach")
    print(f"  - Target source: {args.target_source if hasattr(args, 'target_source') else 'mswep'}")
    print(f"  - Regional focus: {current_use_regional_focus}")
    print(f"  - Training samples: {len(data_module.train_dataset) if data_module.train_dataset else 'N/A'}") # type: ignore
    print(f"  - Validation samples: {len(data_module.val_dataset) if data_module.val_dataset else 'N/A'}") # type: ignore
    
    model_to_train = None # Initialize to None
    if fold == 0:
         print("Using standard UNet architecture for initial fold")
         from models.mswep_unet import MSWEPUNet
         # Get total input channels from args
         total_channels = args.total_input_channels if hasattr(args, 'total_input_channels') else 5
         print(f"Creating UNet with {total_channels} input channels")
         unet_model = MSWEPUNet(
             in_channels=total_channels, 
             dropout=args.dropout if hasattr(args, 'dropout') else 0.2,
             use_regional_focus=current_use_regional_focus,
             region_weight=current_region_weight,
             outside_weight=current_outside_weight
         )
         from models.mswep_lightning_wrapper import UNetLightningModule
         model_to_train = UNetLightningModule(
             model=unet_model, 
             learning_rate=args.lr, 
             loss_type=args.loss_type,
             optimizer_type=args.optimizer_type if hasattr(args, 'optimizer_type') else 'adam',
             lr_scheduler_type=args.lr_scheduler_type if hasattr(args, 'lr_scheduler_type') else 'cosineannealinglr',
             use_regional_focus=current_use_regional_focus,
             region_weight=current_region_weight,
             outside_weight=current_outside_weight,
             target_scaler=data_module.target_scaler,  # Pass scaler from data module
             weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 1e-3
         )
    else:
        current_run_dir = os.path.dirname(fold_output_dir)
        prev_fold_dir = os.path.join(current_run_dir, f"fold{fold-1}")
        print(f"Looking for previous fold checkpoint in: {prev_fold_dir}")
        
        prev_ckpt_path = None
        if os.path.exists(prev_fold_dir):
            potential_ckpts = [f for f in os.listdir(prev_fold_dir) 
                             if f.startswith('best_model') and f.endswith('.ckpt')]
            if potential_ckpts:
                potential_ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(prev_fold_dir, x)), 
                                    reverse=True)
                prev_ckpt_path = os.path.join(prev_fold_dir, potential_ckpts[0])
                print(f"Found checkpoint: {prev_ckpt_path}")
            else:
                print(f"No checkpoint files found in {prev_fold_dir}")
        else:
            print(f"Previous fold directory not found: {prev_fold_dir}")
            
        if prev_ckpt_path and os.path.exists(prev_ckpt_path):
            print(f"Loading model from previous fold's checkpoint: {prev_ckpt_path}")
            try:
                from models.mswep_unet import MSWEPUNet
                # Get total input channels from args
                total_channels = args.total_input_channels if hasattr(args, 'total_input_channels') else 5
                print(f"Creating inner UNet with {total_channels} input channels for checkpoint loading")
                inner_unet_model = MSWEPUNet(
                    in_channels=total_channels,
                    use_regional_focus=current_use_regional_focus,
                    region_weight=current_region_weight,
                    outside_weight=current_outside_weight
                )
                
                import types
                import torch
                from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
                from models.mswep_lightning_wrapper import UNetLightningModule
                
                model_to_train = UNetLightningModule.load_from_checkpoint(
                    checkpoint_path=prev_ckpt_path,
                    model=inner_unet_model,
                    use_regional_focus=current_use_regional_focus,
                    region_weight=current_region_weight,
                    outside_weight=current_outside_weight,
                    learning_rate=args.lr,
                    target_scaler=data_module.target_scaler  # Pass scaler from data module
                )
                
                optimizer_type = args.optimizer_type if hasattr(args, 'optimizer_type') else 'adam'
                lr_scheduler_type = args.lr_scheduler_type if hasattr(args, 'lr_scheduler_type') else 'cosineannealinglr'
                max_epochs = args.epochs
                
                def new_configure_optimizers(self_model): # Changed self to self_model to avoid conflict
                    print(f"Creating fresh optimizer ({optimizer_type}) with learning rate: {args.lr}")
                    if optimizer_type.lower() == 'adam':
                        optimizer = torch.optim.Adam(self_model.parameters(), lr=args.lr)
                    elif optimizer_type.lower() == 'adamw':
                        optimizer = torch.optim.AdamW(self_model.parameters(), lr=args.lr, weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 1e-4) # Use configured weight_decay
                    else:
                        optimizer = torch.optim.Adam(self_model.parameters(), lr=args.lr)
                    
                    print(f"Creating fresh scheduler ({lr_scheduler_type}) for {max_epochs} epochs")
                    if lr_scheduler_type.lower() == 'cosineannealinglr':
                        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
                    elif lr_scheduler_type.lower() == 'reducelronplateau':
                        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=50, min_lr=1e-7, threshold=1e-4, verbose=True) # Optimized for double descent
                    elif lr_scheduler_type.lower() == 'cosineannealingwarmrestarts':
                        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10) # Consider T_0 configurable
                    else:
                        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
                    
                    if lr_scheduler_type.lower() == 'reducelronplateau':
                        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss', 'interval': 'epoch', 'frequency': 1}}
                    else:
                        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
                
                model_to_train.configure_optimizers = types.MethodType(new_configure_optimizers, model_to_train)
                
                if hasattr(model_to_train, 'hparams'):
                    model_to_train.hparams.learning_rate = args.lr # type: ignore
                    if hasattr(model_to_train.hparams, 'optimizer_type'): # type: ignore
                        model_to_train.hparams.optimizer_type = optimizer_type # type: ignore
                    if hasattr(model_to_train.hparams, 'lr_scheduler_type'): # type: ignore
                        model_to_train.hparams.lr_scheduler_type = lr_scheduler_type # type: ignore
                
                print(f"Successfully loaded checkpoint from fold {fold-1} with reset optimizer/scheduler")
            except Exception as e:
                print(f"Error loading previous checkpoint: {e}")
                traceback.print_exc()
                print("ERROR: Could not load previous fold's checkpoint. Stopping training.")
                sys.exit(1)
        else:
            print(f"ERROR: No checkpoint found for fold {fold-1} in {prev_fold_dir}")
            print("Stopping training. Ensure previous fold completed successfully.")
            sys.exit(1)
    
    if current_use_regional_focus:
        print(f"\n--- Regional Focus Configuration for Fold {fold} ---")
        print(f"Regional Focus Enabled: {current_use_regional_focus}")
        print(f"Region Weight: {current_region_weight}")
        print(f"Outside Weight: {current_outside_weight}")
        print(f"Target Source: {args.target_source if hasattr(args, 'target_source') else 'mswep'}")
        print("-" * 20)
    
    is_last_fold_to_train = (fold == args.folds - 1)
    print(f"Is this the last fold to train? {is_last_fold_to_train}")

    # Ensure model_to_train is assigned
    if model_to_train is None:
        print("ERROR: model_to_train was not initialized. This should not happen.")
        sys.exit(1)

    trained_model_results, best_model_path = train_unet_model(
        data_module=data_module,
        model=model_to_train, # Use the correctly initialized or loaded model
        output_dir=fold_output_dir,
        log_dir=log_dir_for_fold,
        epochs=args.epochs,
        accelerator=HARDCODED_ACCELERATOR, # Global or from args
        devices=HARDCODED_DEVICES, # Global or from args
        fold=fold,
        precipitation_threshold=args.threshold,
        generate_plots=True, # This controls the end-of-fold quick plot
        patience=HARDCODED_PATIENCE, # Global or from args
        track_epochs=HARDCODED_TRACK_EPOCHS, # Global or from args
        evaluate_raw=False,
        save_predictions=True,
        args_all=args, # Pass all args for access to args.folds, etc.
        is_last_fold=is_last_fold_to_train # Pass the new flag
    )
    
    trained_model_results['fold'] = fold # type: ignore
    
    if best_model_path:
        print(f"Using best model path: {best_model_path} (type: {type(best_model_path)})")
    else:
        print("No best model path returned.")
        
    print(f"Fold {fold} training complete.")
    print(f"\nNote: Per-fold IDR/CRPS evaluation removed. Final comprehensive evaluation will be performed after all folds complete.")
    
    return trained_model_results

def calculate_final_crps(output_dir, num_folds, skip_crps=False, args=None):
    if skip_crps:
        print("WARNING: Skipping FINAL comprehensive evaluation as requested.")
        return

    print(f"\n=== RUNNING FINAL COMPREHENSIVE EVALUATION ===")
    try:
        import numpy as np
        import pandas as pd
        import os
        import json
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        import shutil
        # Assuming mswep_evaluation.py is in python path or same directory
        from models.mswep_evaluation import (
            apply_easyuq_per_cell,
            calculate_brier_score_for_cell, # Keep if used elsewhere, but BSD handles BS
            brier_score_decomposition,
            calculate_seeps_scores,
            numpy_json_encoder,
            plot_det_vs_prob_comparison,
            plot_seasonal_metrics,
            plot_quantile_map,
            plot_seasonal_samples, # For the last fold seasonal plots
            DEFAULT_LATITUDES, DEFAULT_LONGITUDES, XINC, YINC,
            # GERMANY_PLOT_LON_MIN, GERMANY_PLOT_LON_MAX, GERMANY_PLOT_LAT_MIN, GERMANY_PLOT_LAT_MAX, # Defined in plot_sample
            # GERMANY_BOX_GRID_LON_INDICES, GERMANY_BOX_GRID_LAT_INDICES # Defined in plot_sample
        )


        run_dirs = [d for d in os.listdir(output_dir) if d.startswith("run_") and os.path.isdir(os.path.join(output_dir, d))]
        if not run_dirs:
            print(f"No run directories found in {output_dir}. Ensure the output directory is correct.")
            return

        run_dirs.sort(reverse=True)
        latest_run_dir = os.path.join(output_dir, run_dirs[0])
        print(f"Performing comprehensive evaluation for latest run: {latest_run_dir}")

        final_eval_dir = os.path.join(latest_run_dir, "final_evaluation")
        os.makedirs(final_eval_dir, exist_ok=True)


        print("\nStep 1: Collecting data from all folds...")
        all_train_preds_list, all_train_targets_list, all_val_preds_list, all_val_targets_list = [], [], [], []
        combined_val_times_list = []
        val_samples_per_fold = [] # To track number of validation samples for each fold

        germany_mask = None
        loaded_mask_path = "Not attempted"

        for fold_idx in range(num_folds):
            fold_data_dir = os.path.join(latest_run_dir, f"fold{fold_idx}")
            if not os.path.exists(fold_data_dir):
                print(f"Warning: Fold directory {fold_data_dir} not found. Skipping data loading for this fold.")
                val_samples_per_fold.append(0)
                continue

            current_fold_val_samples = 0
            for name, arr_list_ref in [("val_preds", all_val_preds_list), ("val_targets", all_val_targets_list),
                                   ("train_preds_all", all_train_preds_list), ("train_targets_all", all_train_targets_list)]:
                path = os.path.join(fold_data_dir, f"{name}.npy")
                if os.path.exists(path):
                    try:
                        data = np.load(path)
                        arr_list_ref.append(data)
                        print(f"  Loaded {name} from fold {fold_idx}: shape {data.shape}")
                        if name == "val_targets": # Get sample count from one of the val arrays
                            current_fold_val_samples = data.shape[0]
                    except Exception as e:
                        print(f"  Error loading {path}: {e}")
                else:
                    print(f"  Warning: {name}.npy not found for fold {fold_idx}")
            val_samples_per_fold.append(current_fold_val_samples)

            val_times_path = os.path.join(fold_data_dir, "val_times.npy")
            if os.path.exists(val_times_path):
                try:
                    val_times_data = np.load(val_times_path, allow_pickle=True)
                    combined_val_times_list.append(val_times_data)
                    print(f"  Loaded val_times from fold {fold_idx}: shape {val_times_data.shape if hasattr(val_times_data, 'shape') else 'N/A'}")
                except Exception as e:
                    print(f"  Error loading {val_times_path}: {e}")
            else:
                print(f"  Warning: val_times.npy not found for fold {fold_idx}")

            if germany_mask is None: # Try to load mask from any fold, assume it's consistent
                mask_path_current_fold = os.path.join(fold_data_dir, "germany_mask.npy")
                if os.path.exists(mask_path_current_fold):
                    try:
                        current_fold_mask = np.load(mask_path_current_fold)
                        # Basic check for a 2D boolean array (adjust shape if needed)
                        if current_fold_mask.ndim == 2 and current_fold_mask.dtype == np.bool_:
                            germany_mask = current_fold_mask
                            loaded_mask_path = mask_path_current_fold
                            print(f"  Loaded germany_mask from {mask_path_current_fold}, Sum: {np.sum(germany_mask)}")
                        else:
                            print(f"  Warning: germany_mask at {mask_path_current_fold} has unexpected shape/type: {current_fold_mask.shape}/{current_fold_mask.dtype}.")
                    except Exception as e:
                        print(f"  Error loading germany_mask from {mask_path_current_fold}: {e}")
                else:
                    print(f"  germany_mask.npy not found in {fold_data_dir}")
        
        combined_val_preds = np.concatenate(all_val_preds_list, axis=0) if all_val_preds_list else np.array([])
        combined_val_targets = np.concatenate(all_val_targets_list, axis=0) if all_val_targets_list else np.array([])
        combined_train_preds = np.concatenate(all_train_preds_list, axis=0) if all_train_preds_list else np.array([])
        combined_train_targets = np.concatenate(all_train_targets_list, axis=0) if all_train_targets_list else np.array([])
        
        combined_val_times = None
        if combined_val_times_list:
            valid_times_arrays = [arr for arr in combined_val_times_list if isinstance(arr, np.ndarray) and arr.size > 0]
            if valid_times_arrays:
                try:
                    combined_val_times_np = np.concatenate(valid_times_arrays)
                    combined_val_times = pd.to_datetime(combined_val_times_np)
                except Exception as e:
                    print(f"Error concatenating or converting validation times: {e}")
            else: # No valid time arrays found
                 print("Warning: No valid validation time arrays found after loading all folds.")
        else: # List was empty
             print("Warning: combined_val_times_list is empty. No validation times loaded.")


        print(f"Combined training data shape: {combined_train_preds.shape if combined_train_preds.size > 0 else 'Empty'}")
        print(f"Combined validation data shape: {combined_val_preds.shape if combined_val_preds.size > 0 else 'Empty'}")
        
        if germany_mask is None:
            if combined_val_targets.ndim == 3 and combined_val_targets.size > 0:
                print(f"Warning: Germany mask not found in any fold directories (last attempt info: {loaded_mask_path}). Creating default (all True). Evaluation on entire domain.")
                germany_mask = np.ones((combined_val_targets.shape[1], combined_val_targets.shape[2]), dtype=bool)
            else:
                print("Error: Germany mask not found AND cannot infer dimensions from combined_val_targets. Aborting comprehensive evaluation.")
                return
        else:
            print(f"Using Germany mask loaded from {loaded_mask_path} for comprehensive evaluation. Shape: {germany_mask.shape}, Num True cells: {np.sum(germany_mask)}")

        np.save(os.path.join(final_eval_dir, "germany_mask_used_for_final_eval.npy"), germany_mask)
        print(f"Saved the active Germany mask to {os.path.join(final_eval_dir, 'germany_mask_used_for_final_eval.npy')}")

        if combined_val_preds.size == 0 or combined_train_preds.size == 0 :
            print("Error: Insufficient data collected (validation or training preds are empty). Cannot perform evaluation.")
            print(f"  Combined validation predictions shape: {combined_val_preds.shape if combined_val_preds.size > 0 else 'EMPTY'}")
            print(f"  Combined training predictions shape: {combined_train_preds.shape if combined_train_preds.size > 0 else 'EMPTY'}")
            print(f"  Combined validation targets shape: {combined_val_targets.shape if combined_val_targets.size > 0 else 'EMPTY'}")
            print(f"  Combined training targets shape: {combined_train_targets.shape if combined_train_targets.size > 0 else 'EMPTY'}")
            
            # Try to proceed with validation data only if available
            if combined_val_preds.size > 0 and combined_val_targets.size > 0:
                print("\nAttempting to proceed with validation data only (skipping IDR fitting which requires training data)...")
                # Save what we have
                np.save(os.path.join(final_eval_dir, "combined_val_preds.npy"), combined_val_preds)
                np.save(os.path.join(final_eval_dir, "combined_val_targets.npy"), combined_val_targets)
                if combined_val_times is not None:
                    np.save(os.path.join(final_eval_dir, "combined_val_times.npy"), combined_val_times.values)
                
                # Calculate deterministic metrics only
                print("\nCalculating deterministic metrics for available validation data...")
                overall_mae = np.mean(np.abs(combined_val_preds - combined_val_targets))
                overall_rmse = np.sqrt(np.mean((combined_val_preds - combined_val_targets)**2))
                print(f"Overall MAE: {overall_mae:.4f}, Overall RMSE: {overall_rmse:.4f}")
                
                limited_results = {
                    'evaluation_type': 'limited_deterministic_only',
                    'overall_mae': float(overall_mae),
                    'overall_rmse': float(overall_rmse),
                    'note': 'IDR/probabilistic metrics skipped due to missing training data'
                }
                
                with open(os.path.join(final_eval_dir, "limited_evaluation_summary.json"), "w") as f:
                    json.dump(limited_results, f, indent=4)
                print(f"Saved limited evaluation results to {final_eval_dir}/limited_evaluation_summary.json")
            return
        
        np.save(os.path.join(final_eval_dir, "combined_val_preds.npy"), combined_val_preds)
        np.save(os.path.join(final_eval_dir, "combined_val_targets.npy"), combined_val_targets)
        np.save(os.path.join(final_eval_dir, "combined_train_preds.npy"), combined_train_preds)
        np.save(os.path.join(final_eval_dir, "combined_train_targets.npy"), combined_train_targets)
        if combined_val_times is not None:
             np.save(os.path.join(final_eval_dir, "combined_val_times.npy"), combined_val_times.values)


        print("\nStep 2: Calculating deterministic metrics for Germany region...")
        bins = [0, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, float('inf')]
        bin_names = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" if bins[i+1] != float('inf') else f">{bins[i]:.1f}" for i in range(len(bins)-1)]
        
        det_metrics = {'mae': {}, 'rmse': {}, 'bias': {}}
        # Initialize intensity bin MAE storage within det_metrics
        for bn_mae_init in bin_names: det_metrics[f'mae_{bn_mae_init}'] = {}


        all_germany_cell_mae, all_germany_cell_rmse = [], []
        
        # Initialize seasonal aggregation dictionary. CRPS and BS sums will be populated in Step 3.
        seasons_list = ['DJF', 'MAM', 'JJA', 'SON']
        seasonal_metrics_agg = None
        timestamp_seasons_val = None

        if combined_val_times is not None:
            season_map = {m: s_name for s_list_inner, s_name in zip([[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], seasons_list) for m in s_list_inner}
            timestamp_seasons_val = np.array([season_map.get(m.month, 'N/A') for m in combined_val_times]) # Use .month
            # Initialize with separate spatial and temporal tracking to fix aggregation issue
            seasonal_metrics_agg = {
                s: {
                    'mae_cell_seasonal_means': [],  # Store cell-wise seasonal MAE for proper spatial averaging
                    'crps_values': [],  # Store individual CRPS values for proper aggregation
                    'bs_values': [],    # Store individual BS values for proper aggregation
                    'sample_count': 0   # Count of temporal samples per season
                }
                for s in seasons_list
            }
            print(f"Initialized seasonal metrics tracking for: {seasons_list}")
        else:
            print("Warning: combined_val_times is None. Seasonal metrics cannot be calculated.")

        num_val_samples, grid_lat, grid_lon = combined_val_targets.shape
        num_germany_cells_total = np.sum(germany_mask)
        print(f"Grid: {grid_lat}x{grid_lon}, Val samples: {num_val_samples}, Germany cells: {num_germany_cells_total}")
        
        for r_idx in range(grid_lat):
            for c_idx in range(grid_lon):
                if germany_mask[r_idx, c_idx]:
                    vp_cell = combined_val_preds[:, r_idx, c_idx]
                    vt_cell = combined_val_targets[:, r_idx, c_idx]
                    errs = vp_cell - vt_cell
                    abs_errs = np.abs(errs)
                    
                    cell_mae_val = np.mean(abs_errs)
                    cell_rmse_val = np.sqrt(np.mean(np.square(errs)))
                    
                    det_metrics['mae'][(r_idx, c_idx)] = cell_mae_val
                    det_metrics['rmse'][(r_idx, c_idx)] = cell_rmse_val
                    det_metrics['bias'][(r_idx, c_idx)] = np.mean(errs)
                    all_germany_cell_mae.append(cell_mae_val)
                    all_germany_cell_rmse.append(cell_rmse_val)
                    
                    for i, bn_iter_mae in enumerate(bin_names):
                        b_mask = (vt_cell >= bins[i]) & (vt_cell < bins[i+1] if bins[i+1] != float('inf') else True)
                        if np.any(b_mask):
                            det_metrics[f'mae_{bn_iter_mae}'][(r_idx, c_idx)] = np.mean(abs_errs[b_mask])
                    
                    if seasonal_metrics_agg and timestamp_seasons_val is not None:
                        for s_name_mae in seasons_list: 
                            s_mask_mae = (timestamp_seasons_val == s_name_mae) # Mask for all samples in this season
                            if np.any(s_mask_mae): # If there are any samples for this season
                                # Accumulate sum of absolute errors and count for samples in this season AT THIS CELL
                                seasonal_metrics_agg[s_name_mae]['mae_cell_seasonal_means'].append(np.mean(abs_errs[s_mask_mae]))
                                seasonal_metrics_agg[s_name_mae]['sample_count'] += np.sum(s_mask_mae)
        
        overall_mae_germany = np.mean(all_germany_cell_mae) if all_germany_cell_mae else np.nan
        overall_rmse_germany = np.mean(all_germany_cell_rmse) if all_germany_cell_rmse else np.nan
        print(f"\n--- Deterministic Metrics (Germany) --- MAE: {overall_mae_germany:.4f}, RMSE: {overall_rmse_germany:.4f}")
        
        intensity_bin_mae_final = {}
        print("\nIntensity-specific MAE (Germany):")
        for bn_print_mae in bin_names:
            # Correctly access the pre-calculated per-cell MAEs for this bin
            vals_mae = [v for v in det_metrics.get(f'mae_{bn_print_mae}', {}).values() if np.isfinite(v)]
            avg_mae_bin = np.mean(vals_mae) if vals_mae else np.nan
            intensity_bin_mae_final[bn_print_mae] = avg_mae_bin
            print(f"  {bn_print_mae} mm: MAE = {avg_mae_bin:.4f}")
        np.save(os.path.join(final_eval_dir, "deterministic_metrics.npy"), det_metrics)
        
        print("\nStep 2.5: Calculating SEEPS scores (Germany only)...")
        mean_seeps_germany_final = np.nan
        try:
            seeps_results_all = calculate_seeps_scores(
                consolidated_train_targets_all_cells=combined_train_targets,
                consolidated_val_preds_all_cells=combined_val_preds,
                consolidated_val_targets_all_cells=combined_val_targets,
                germany_mask=germany_mask, dry_threshold_mm=0.2
            )
            mean_seeps_germany_final = seeps_results_all.get('mean_seeps', np.nan)
            if np.isfinite(mean_seeps_germany_final):
                print(f"\n--- SEEPS Score (Germany) --- Mean: {mean_seeps_germany_final:.4f}")
            else:
                print("Warning: SEEPS calculation failed or returned invalid.")
            with open(os.path.join(final_eval_dir, "seeps_results.json"), "w") as f:
                json.dump({k: (v if k != 'cell_climatologies' else 'summary_omitted') for k,v in seeps_results_all.items()}, f, indent=4, default=numpy_json_encoder)
        except Exception as e_seeps:
            print(f"Error calculating SEEPS: {e_seeps}\n{traceback.format_exc()}")


        print("\nStep 3: Applying IDR & calculating probabilistic metrics (Germany)...")
        cell_metrics_final = {'crps': {}, 'bs': {}}
        for bn_prob_init in bin_names: cell_metrics_final[f'crps_{bn_prob_init}'] = {}

        all_germany_cell_mean_crps_final, all_germany_cell_mean_bs_final = [], []
        all_forecast_probs_germany_bsd_accum, all_binary_obs_germany_bsd_accum = [], []
        brier_thresh_bsd = 0.2 

        val_preds_p50_final = np.full((num_val_samples, grid_lat, grid_lon), np.nan)
        
        valid_germany_indices = [(r, c) for r in range(grid_lat) for c in range(grid_lon) if germany_mask[r, c]]
        print(f"Processing IDR for all {len(valid_germany_indices)} valid Germany cells...")

        for r_cell, c_cell in tqdm(valid_germany_indices, desc="Applying IDR and Calculating Metrics"):
            train_p_cell = combined_train_preds[:, r_cell, c_cell]
            train_t_cell = combined_train_targets[:, r_cell, c_cell]
            val_p_cell_loop = combined_val_preds[:, r_cell, c_cell]
            val_t_cell_loop = combined_val_targets[:, r_cell, c_cell]

            pred_dist = apply_easyuq_per_cell(train_p_cell, train_t_cell, eval_preds_cell=val_p_cell_loop)

            if pred_dist:
                # CRPS Calculation and Seasonal Accumulation
                try: 
                    y_true_crps_calc = val_t_cell_loop.astype(float).flatten()
                    crps_vals_arr = np.array(pred_dist.crps(y_true_crps_calc), dtype=float)
                    valid_crps_mask = np.isfinite(crps_vals_arr)
                    
                    if np.any(valid_crps_mask):
                        mean_crps_this_cell = np.mean(crps_vals_arr[valid_crps_mask])
                        if np.isfinite(mean_crps_this_cell):
                            all_germany_cell_mean_crps_final.append(mean_crps_this_cell)
                            cell_metrics_final['crps'][(r_cell, c_cell)] = mean_crps_this_cell
                        
                        for i_crps_bin, bn_crps_iter in enumerate(bin_names): 
                            b_mask_crps_iter = (val_t_cell_loop >= bins[i_crps_bin]) & \
                                           (val_t_cell_loop < bins[i_crps_bin+1] if bins[i_crps_bin+1]!=float('inf') else True)
                            if np.any(b_mask_crps_iter & valid_crps_mask):
                                cell_metrics_final[f'crps_{bn_crps_iter}'][(r_cell, c_cell)] = np.mean(crps_vals_arr[b_mask_crps_iter & valid_crps_mask])
                        
                        # *** MODIFIED: Accumulate CRPS for seasonal metrics ***
                        if seasonal_metrics_agg and timestamp_seasons_val is not None:
                            # Ensure we only use CRPS values corresponding to valid_crps_mask
                            crps_values_for_season = crps_vals_arr[valid_crps_mask]
                            seasons_for_valid_crps = timestamp_seasons_val[valid_crps_mask]
                            for s_iter_crps in seasons_list: 
                                s_mask_for_crps = (seasons_for_valid_crps == s_iter_crps)
                                if np.any(s_mask_for_crps):
                                    seasonal_metrics_agg[s_iter_crps]['crps_values'].append(np.sum(crps_values_for_season[s_mask_for_crps]))
                                    seasonal_metrics_agg[s_iter_crps]['sample_count'] += np.sum(s_mask_for_crps) # Use prob_count for CRPS/BS
                except Exception as e_crps_cell_calc: print(f"CRPS Error cell ({r_cell},{c_cell}): {e_crps_cell_calc}")

                # Brier Score Calculation and Seasonal Accumulation
                try: 
                    if hasattr(pred_dist, 'cdf'):
                        prob_le_bsd_calc = pred_dist.cdf(thresholds=np.array([brier_thresh_bsd]))
                        if isinstance(prob_le_bsd_calc, np.ndarray) and prob_le_bsd_calc.ndim > 1 : prob_le_bsd_calc = prob_le_bsd_calc.flatten()

                        pred_prob_ex_bsd_calc = np.clip(1.0 - prob_le_bsd_calc, 0.0, 1.0)
                        bin_out_bsd_calc = (val_t_cell_loop > brier_thresh_bsd).astype(float)

                        if pred_prob_ex_bsd_calc.shape == bin_out_bsd_calc.shape:
                            all_forecast_probs_germany_bsd_accum.extend(pred_prob_ex_bsd_calc)
                            all_binary_obs_germany_bsd_accum.extend(bin_out_bsd_calc)
                            
                            bs_arr_calc = (pred_prob_ex_bsd_calc - bin_out_bsd_calc) ** 2
                            valid_bs_mask = np.isfinite(bs_arr_calc)
                            if np.any(valid_bs_mask):
                                mean_bs_this_cell = np.mean(bs_arr_calc[valid_bs_mask])
                                if np.isfinite(mean_bs_this_cell):
                                    all_germany_cell_mean_bs_final.append(mean_bs_this_cell)
                                    cell_metrics_final['bs'][(r_cell, c_cell)] = mean_bs_this_cell
                                
                                # *** MODIFIED: Accumulate BS for seasonal metrics ***
                                if seasonal_metrics_agg and timestamp_seasons_val is not None:
                                    bs_values_for_season = bs_arr_calc[valid_bs_mask]
                                    seasons_for_valid_bs = timestamp_seasons_val[valid_bs_mask]
                                    for s_iter_bs_calc in seasons_list:
                                        s_mask_for_bs = (seasons_for_valid_bs == s_iter_bs_calc)
                                        if np.any(s_mask_for_bs):
                                            seasonal_metrics_agg[s_iter_bs_calc]['bs_values'].append(np.sum(bs_values_for_season[s_mask_for_bs]))
                                            # prob_count is already incremented by CRPS part for the same samples
                        else:
                            print(f"Warning BSD shapes: Pred {pred_prob_ex_bsd_calc.shape}, Obs {bin_out_bsd_calc.shape} for cell ({r_cell},{c_cell})")
                except Exception as e_bs_cell_calc: print(f"BS Error cell ({r_cell},{c_cell}): {e_bs_cell_calc}")

                # P50 (Median) Prediction Extraction
                try: 
                    p50_cell_calc = pred_dist.qpred(quantiles=np.array([0.5]))
                    if p50_cell_calc is not None and len(p50_cell_calc) == num_val_samples: 
                        val_preds_p50_final[:, r_cell, c_cell] = p50_cell_calc.flatten()
                except Exception as e_p50_cell_calc: print(f"P50 Error cell ({r_cell},{c_cell}): {e_p50_cell_calc}")
                del pred_dist # Free memory
        
        np.save(os.path.join(final_eval_dir, "cell_metrics.npy"), cell_metrics_final)
        np.save(os.path.join(final_eval_dir, "val_preds_p50.npy"), val_preds_p50_final) # This is the combined P50
        print(f"Saved cell_metrics and combined val_preds_p50 to {final_eval_dir}")
        
        # *** ADDED: Save val_preds_p50_last_fold_specific.npy ***
        if num_folds > 0 and val_samples_per_fold:
            last_fold_idx = num_folds - 1
            last_fold_output_dir = os.path.join(latest_run_dir, f"fold{last_fold_idx}")
            os.makedirs(last_fold_output_dir, exist_ok=True) # Ensure dir exists

            # Determine start and end index for the last fold's validation data in the combined array
            samples_before_last_fold = sum(val_samples_per_fold[:last_fold_idx])
            num_samples_last_fold = val_samples_per_fold[last_fold_idx]

            if num_samples_last_fold > 0:
                start_idx_last_fold = samples_before_last_fold
                end_idx_last_fold = samples_before_last_fold + num_samples_last_fold
                
                # Ensure indices are within bounds of val_preds_p50_final
                if end_idx_last_fold <= val_preds_p50_final.shape[0]:
                    last_fold_p50_data = val_preds_p50_final[start_idx_last_fold:end_idx_last_fold, :, :]
                    last_fold_p50_save_path = os.path.join(last_fold_output_dir, "val_preds_p50_last_fold_specific.npy")
                    np.save(last_fold_p50_save_path, last_fold_p50_data)
                    print(f"Saved P50 predictions for last fold ({last_fold_idx}) to: {last_fold_p50_save_path} with shape {last_fold_p50_data.shape}")
                else:
                    print(f"Warning: Calculated end index {end_idx_last_fold} for last fold's P50 data is out of bounds for combined P50 array shape {val_preds_p50_final.shape[0]}. Cannot save last fold specific P50.")
            else:
                print(f"Warning: Last fold ({last_fold_idx}) has no validation samples. Cannot save last fold specific P50.")
        else:
            print("Warning: Cannot determine last fold specific P50 data due to num_folds or val_samples_per_fold issues.")

        import gc; gc.collect()
        print("Finished IDR processing for all cells.")
        
        # --- Final Metrics Summary Calculation ---
        results_summary_final = {
            'evaluation_region': 'Germany',
            'num_valid_cells_for_prob_metrics': len(all_germany_cell_mean_crps_final)
        }
        results_summary_final['overall_mean_crps'] = np.mean(all_germany_cell_mean_crps_final) if all_germany_cell_mean_crps_final else np.nan
        results_summary_final['overall_mean_bs_0.2mm'] = np.mean(all_germany_cell_mean_bs_final) if all_germany_cell_mean_bs_final else np.nan
        
        if all_forecast_probs_germany_bsd_accum and all_binary_obs_germany_bsd_accum:
            bsd_mbs, bsd_rel, bsd_res, bsd_unc = brier_score_decomposition(
                np.array(all_forecast_probs_germany_bsd_accum), np.array(all_binary_obs_germany_bsd_accum)
            )
            results_summary_final['brier_score_decomposition_0.2mm'] = {
                'mean_bs': bsd_mbs, 'miscalibration': bsd_rel, 'discrimination': bsd_res, 'uncertainty': bsd_unc
            }
        else:
            results_summary_final['brier_score_decomposition_0.2mm'] = None
            print("Warning: Insufficient data for Brier Score Decomposition.")

        results_summary_final['deterministic_mae'] = overall_mae_germany
        results_summary_final['deterministic_rmse'] = overall_rmse_germany
        results_summary_final['deterministic_seeps'] = mean_seeps_germany_final

        intensity_bin_crps_final_summary = {}
        for bn_final_crps_sum_iter in bin_names:
            vals_crps_sum_iter = [v for v in cell_metrics_final.get(f'crps_{bn_final_crps_sum_iter}', {}).values() if np.isfinite(v)]
            intensity_bin_crps_final_summary[bn_final_crps_sum_iter] = np.mean(vals_crps_sum_iter) if vals_crps_sum_iter else np.nan
        
        results_summary_final['intensity_bin_metrics'] = {
            'deterministic_mae': intensity_bin_mae_final,
            'probabilistic_crps': intensity_bin_crps_final_summary
        }

        # *** MODIFIED: Final seasonal metrics calculation (MAE was already cell-wise, CRPS/BS now accumulated per cell-sample) ***
        seasonal_results_final_summary = {}
        if seasonal_metrics_agg:
            for season_sum_iter in seasons_list:
                metrics_sum_iter = seasonal_metrics_agg[season_sum_iter]
                # For MAE, mae_count is total number of cell-samples contributing to this season
                mean_mae_s = np.mean(metrics_sum_iter['mae_cell_seasonal_means']) if metrics_sum_iter['mae_cell_seasonal_means'] else np.nan
                # For CRPS/BS, prob_count is total number of cell-samples with valid probabilistic scores in this season
                mean_crps_s = np.sum(metrics_sum_iter['crps_values']) / metrics_sum_iter['sample_count'] if metrics_sum_iter['sample_count'] > 0 else np.nan
                mean_bs_s = np.sum(metrics_sum_iter['bs_values']) / metrics_sum_iter['sample_count'] if metrics_sum_iter['sample_count'] > 0 else np.nan
                
                seasonal_results_final_summary[season_sum_iter] = {
                    'mean_crps': mean_crps_s,
                    'mean_bs_0.2mm': mean_bs_s,
                    'mean_mae': mean_mae_s, # Spatially averaged MAE for this season
                    'num_samples_prob': metrics_sum_iter['sample_count'], # Number of samples contributing to CRPS/BS
                    'num_samples_det': len(metrics_sum_iter['mae_cell_seasonal_means']) # Number of samples contributing to MAE
                }
        results_summary_final['seasonal_metrics'] = seasonal_results_final_summary

        # Global color norms calculation removed - using default visualization scaling

        # --- Step 5: Generating visualization plots ---
        print("\nStep 5: Generating visualization plots...")
        
        det_vs_prob_save_path = os.path.join(final_eval_dir, "det_vs_prob_comparison_comprehensive.png")
        if results_summary_final and 'intensity_bin_metrics' in results_summary_final:
            plot_det_vs_prob_comparison(results_summary_final, det_vs_prob_save_path, "Comprehensive_Combined_Val")
        else:
            print("Warning: Skipping det_vs_prob_comparison plot due to incomplete results_summary_final.")

        if results_summary_final and results_summary_final.get('seasonal_metrics') and (args and hasattr(args, 'seasonal_plots') and args.seasonal_plots):
             seasonal_crps_save_path = os.path.join(final_eval_dir, "seasonal_crps_comprehensive_combined_val.png")
             seasonal_bs_save_path = os.path.join(final_eval_dir, "seasonal_bs_comprehensive_combined_val.png")
             plot_seasonal_metrics(results_summary_final, seasonal_crps_save_path, seasonal_bs_save_path, "Comprehensive_Combined_Val")
        
        plot_time_idx = args.plot_time_index if args and hasattr(args, 'plot_time_index') else 0
        lats_plot_q, lons_plot_q = DEFAULT_LATITUDES, DEFAULT_LONGITUDES # Use defaults from evaluation script
        # Adjust if data grid differs from defaults (e.g. if not MSWEP standard)
        if combined_val_targets.shape[1] != len(DEFAULT_LATITUDES) or combined_val_targets.shape[2] != len(DEFAULT_LONGITUDES):
            print("Warning: Grid dimensions of combined_val_targets differ from DEFAULT_LATITUDES/LONGITUDES. Adjusting for plot_quantile_map.")
            lats_plot_q = np.linspace(DEFAULT_LATITUDES[0], DEFAULT_LATITUDES[-1], combined_val_targets.shape[1])
            lons_plot_q = np.linspace(DEFAULT_LONGITUDES[0], DEFAULT_LONGITUDES[-1], combined_val_targets.shape[2])

        if os.path.exists(os.path.join(final_eval_dir, "val_preds_p50.npy")) and \
           os.path.exists(os.path.join(final_eval_dir, "combined_val_targets.npy")): # Make sure combined_val_targets.npy exists
            plot_quantile_map(
                fold_dir=final_eval_dir, # Plotting combined data
                fold_num="Comprehensive_Combined_Val", 
                time_index=plot_time_idx,
                lats=lats_plot_q, lons=lons_plot_q
            )
        else:
            missing_files_log = []
            if not os.path.exists(os.path.join(final_eval_dir, "val_preds_p50.npy")):
                missing_files_log.append("val_preds_p50.npy")
            if not os.path.exists(os.path.join(final_eval_dir, "combined_val_targets.npy")):
                missing_files_log.append("combined_val_targets.npy")
            print(f"Warning: Skipping quantile map for combined data. Missing {' and '.join(missing_files_log)} in {final_eval_dir}")


        # Seasonal Sample Plots from LAST FOLD's validation data
        if args and hasattr(args, 'seasonal_plots') and args.seasonal_plots and num_folds > 0:
            print("\nGenerating seasonal sample plots from LAST FOLD's validation data...")
            last_fold_idx_plot = num_folds - 1 # Corrected variable name
            last_fold_dir_plot = os.path.join(latest_run_dir, f"fold{last_fold_idx_plot}")
            
            last_fold_val_preds_path_plot = os.path.join(last_fold_dir_plot, "val_preds.npy") # Deterministic
            last_fold_val_targets_path_plot = os.path.join(last_fold_dir_plot, "val_targets.npy")
            last_fold_val_times_path_plot = os.path.join(last_fold_dir_plot, "val_times.npy")
            # This is the file we now aim to create correctly
            last_fold_p50_preds_path_plot = os.path.join(last_fold_dir_plot, "val_preds_p50_last_fold_specific.npy")

            if all(os.path.exists(p) for p in [last_fold_val_preds_path_plot, 
                                               last_fold_val_targets_path_plot, 
                                               last_fold_val_times_path_plot, 
                                               last_fold_p50_preds_path_plot]): # Check for the newly created file
                
                # Adjust lat/lon for these plots if necessary (similar to quantile_map)
                # Assuming for now that lat/lon is consistent across folds for plotting.
                # If not, you might need to load fold-specific lat/lon if saved, or use data dimensions.
                
                plot_seasonal_samples(
                    fold_dir=last_fold_dir_plot, 
                    fold_num=f"LastFold_Val_{last_fold_idx_plot}",
                    num_samples_per_season=args.num_seasonal_plot_samples if hasattr(args, 'num_seasonal_plot_samples') else 3,
                    lats=lats_plot_q, # Use the potentially adjusted lats/lons
                    lons=lons_plot_q,
                    val_preds_fname="val_preds.npy", 
                    val_targets_fname="val_targets.npy",
                    val_times_fname="val_times.npy",
                    p50_preds_fname="val_preds_p50_last_fold_specific.npy" # Point to the specific file
                )
                print(f"Seasonal sample plots for last fold's validation data potentially saved in: {last_fold_dir_plot}/seasonal_sample_plots")
            else:
                print(f"Skipping seasonal sample plots for last fold ({last_fold_idx_plot}). Missing one or more required files in {last_fold_dir_plot}:")
                print(f"  Deterministic Preds: {last_fold_val_preds_path_plot} (Exists: {os.path.exists(last_fold_val_preds_path_plot)})")
                print(f"  Targets: {last_fold_val_targets_path_plot} (Exists: {os.path.exists(last_fold_val_targets_path_plot)})")
                print(f"  Times: {last_fold_val_times_path_plot} (Exists: {os.path.exists(last_fold_val_times_path_plot)})")
                print(f"  P50 Preds (Specific): {last_fold_p50_preds_path_plot} (Exists: {os.path.exists(last_fold_p50_preds_path_plot)})")
        
        # Save final summary results
        summary_save_path = os.path.join(final_eval_dir, "final_evaluation_summary.json")
        with open(summary_save_path, "w") as f_sum:
            json.dump(results_summary_final, f_sum, indent=4, default=numpy_json_encoder)
        print(f"Saved final comprehensive evaluation summary to: {summary_save_path}")
            
        print("\n=== FINAL COMPREHENSIVE EVALUATION COMPLETE ===")
            
    except ImportError as e_imp_final:
        print(f"ImportError in comprehensive_evaluation: {e_imp_final}. Ensure all dependencies are met.")
        traceback.print_exc()
    except Exception as e_comp_final:
        print(f"Error in comprehensive_evaluation: {e_comp_final}")
        traceback.print_exc()
# Execute main function
if __name__ == "__main__":

    def main():
        # Declare globals first
        global HARDCODED_BATCH_SIZE, HARDCODED_NUM_WORKERS, HARDCODED_ACCELERATOR, HARDCODED_DEVICES
        global HARDCODED_APPLY_LOG, HARDCODED_LOG_OFFSET, HARDCODED_PATIENCE, HARDCODED_TRACK_EPOCHS
        global HARDCODED_EVAL_INTENSITY, HARDCODED_ENABLE_EASYUQ

        # Enable GPU performance optimizations
        # These settings help maximize throughput on modern GPUs
        # Note: PyTorch automatically handles TF32 precision on RTX 5070/Blackwell GPUs
        torch.backends.cudnn.benchmark = True       # Enable cudnn autotuner
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
        
        # Define hardcoded values
        HARDCODED_BATCH_SIZE = 32
        HARDCODED_NUM_WORKERS = max(1, os.cpu_count() // 2 if os.cpu_count() else 1) # Safer default for num_workers
        HARDCODED_ACCELERATOR = 'gpu' if torch.cuda.is_available() else 'cpu' # Auto-detect GPU
        HARDCODED_DEVICES = [0] if HARDCODED_ACCELERATOR == 'gpu' else 1 # Use 1 CPU process if no GPU
        HARDCODED_APPLY_LOG = True
        HARDCODED_LOG_OFFSET = 0.1  # Changed from 0.01 to 0.1
        HARDCODED_PATIENCE = 180 # Very high patience for double descent (200 epochs)
        HARDCODED_TRACK_EPOCHS = True
        HARDCODED_EVAL_INTENSITY = True
        HARDCODED_ENABLE_EASYUQ = ISODISREG_AVAILABLE # Set based on import
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("No GPU detected, using CPU.")
        
        parser = argparse.ArgumentParser(description='Train a UNet model for MSWEP precipitation forecasting.')
        parser.add_argument('--data_dir', type=str, required=True, help='Directory containing MSWEP data')
        parser.add_argument('--test_data_dir', type=str, default=None, help='Optional directory for test data')
        parser.add_argument('--output_dir', type=str, required=True, help='Base directory to save results')
        parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs')
        parser.add_argument('--epochs', type=int, default=200, help='Max epochs for double descent') # Double descent default
        parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate') # Common default
        parser.add_argument('--threshold', type=float, default=0.2, help='Precip threshold for eval')
        parser.add_argument('--folds', type=int, default=5, help='Number of folds') # Common for CV
        parser.add_argument('--loss_type', type=str, default='mse', 
                            choices=['mse', 'mae', 'weighted_mse', 'huber', 'focal_mse', 'asymmetric_mse', 'log_cosh'],
                            help='Loss function type')
        parser.add_argument('--optimizer_type', type=str, default='adamw', choices=['adam', 'adamw'], help='Optimizer') # AdamW often better
        parser.add_argument('--lr_scheduler_type', type=str, default='doubledescent',
                            choices=['cosineannealinglr', 'cosineannealingwarmrestarts', 'reducelronplateau', 'doubledescent', 'constant'],
                            help='LR scheduler type (doubledescent is optimized for 200-epoch training)')
        parser.add_argument('--target_source', type=str, default='mswep', choices=['mswep', 'hyras'], help='Target dataset')
        parser.add_argument('--use_regional_focus', type=lambda x: bool(distutils.util.strtobool(x)), default=True, help='Apply regional weighting')
        # region_weight is fixed at 1.0, not an arg
        parser.add_argument('--outside_weight', type=float, default=0.2, help='Weight for areas outside Germany (0.0-1.0)')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 regularization)')
        parser.add_argument('--log_offset', type=float, default=0.1, help='Offset for log transform')  # Changed from 0.01 to 0.1
        parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for UNet')
        parser.add_argument('--skip_crps', action='store_true', help='Skip final CRPS calculation (for quick debug runs)') # Changed to action
        parser.add_argument('--seasonal_plots', action='store_true', help='Generate seasonal plots')
        parser.add_argument('--skip_seasonal_plots', action='store_true', help='Skip seasonal plots')
        parser.add_argument('--batch_size', type=int, default=HARDCODED_BATCH_SIZE, help='Batch size for training and final eval processing')
        parser.add_argument('--plot_time_index', type=int, default=0, help='Time index for P50 maps in final eval (default: 0)')
        parser.add_argument('--num_seasonal_plot_samples', type=int, default=3, help='Number of sample days per season for final seasonal plots (default: 3)')
        
        # ERA5 configuration arguments
        parser.add_argument('--era5_variables', type=str, default=None, 
                            help='Comma-separated list of ERA5 variables (e.g., "u,v,q,t"). If None, no ERA5 data is used.')
        parser.add_argument('--era5_pressure_levels', type=str, default=None,
                            help='Comma-separated list of pressure levels (e.g., "300,500,700,850"). Required if era5_variables is set.')
        parser.add_argument('--era5_data_dir', type=str, default=None,
                            help='Path to ERA5 predictors directory. Required if era5_variables is set.')
        parser.add_argument('--era5_single_level_variables', type=str, default=None,
                            help='Comma-separated list of ERA5 single-level variables (e.g., "msl,t2m,tcwv,sp"). These are loaded from individual files.')

        args = parser.parse_args()
        
        # Update batch size from CLI args
        HARDCODED_BATCH_SIZE = args.batch_size
        
        # Process ERA5 arguments
        era5_variables = []
        era5_pressure_levels = []
        era5_single_level_variables = []
        
        if args.era5_variables:
            era5_variables = [v.strip() for v in args.era5_variables.split(',')]
            
            if not args.era5_pressure_levels:
                print("ERROR: --era5_pressure_levels is required when --era5_variables is specified")
                sys.exit(1)
            if not args.era5_data_dir:
                print("ERROR: --era5_data_dir is required when --era5_variables is specified")
                sys.exit(1)
                
            era5_pressure_levels = [int(p.strip()) for p in args.era5_pressure_levels.split(',')]
            
            print(f"\nERA5 Configuration:")
            print(f"  Variables: {era5_variables}")
            print(f"  Pressure levels: {era5_pressure_levels}")
            print(f"  Data directory: {args.era5_data_dir}")
            
        # Process single-level ERA5 variables
        if args.era5_single_level_variables:
            if not args.era5_data_dir:
                print("ERROR: --era5_data_dir is required when --era5_single_level_variables is specified")
                sys.exit(1)
                
            era5_single_level_variables = [v.strip() for v in args.era5_single_level_variables.split(',')]
            print(f"  Single-level variables: {era5_single_level_variables}")
            
        # Calculate total input channels
        base_channels = 5  # 3 precip lags + 2 seasonality
        num_era5_pressure_features = len(era5_variables) * len(era5_pressure_levels)
        num_era5_single_level_features = len(era5_single_level_variables)
        total_era5_features = num_era5_pressure_features + num_era5_single_level_features
        total_input_channels = base_channels + (3 * total_era5_features)  # 3 lags for all ERA5 features
        
        print(f"\nInput channel calculation:")
        print(f"  Base channels (precip + seasonality): {base_channels}")
        if num_era5_pressure_features > 0:
            print(f"  ERA5 pressure-level features: {num_era5_pressure_features} ({len(era5_variables)} vars × {len(era5_pressure_levels)} levels)")
        if num_era5_single_level_features > 0:
            print(f"  ERA5 single-level features: {num_era5_single_level_features}")
        if total_era5_features > 0:
            print(f"  ERA5 channels (3 lags × {total_era5_features} features): {3 * total_era5_features}")
        print(f"  Total input channels: {total_input_channels}")
        
        # Store processed ERA5 config in args for use in train_fold
        args.era5_variables_list = era5_variables if era5_variables else None
        args.era5_pressure_levels_list = era5_pressure_levels if era5_pressure_levels else None
        args.era5_single_level_variables_list = era5_single_level_variables if era5_single_level_variables else None
        args.total_input_channels = total_input_channels

        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a more descriptive run directory name when using ERA5
        if era5_variables or era5_single_level_variables:
            # Create a compact representation of ERA5 config
            run_parts = [f"run_{run_timestamp}"]
            
            if era5_variables:
                vars_str = "_".join(era5_variables)
                levels_str = "_".join(str(p) for p in era5_pressure_levels[:2])  # First 2 levels for brevity
                if len(era5_pressure_levels) > 2:
                    levels_str += "+"  # Indicate more levels
                run_parts.append(f"era5_{vars_str}_{levels_str}")
                
            if era5_single_level_variables:
                sl_vars_str = "_".join(era5_single_level_variables[:3])  # First 3 for brevity
                if len(era5_single_level_variables) > 3:
                    sl_vars_str += "+"  # Indicate more variables
                run_parts.append(f"sl_{sl_vars_str}")
                
            run_name = "_".join(run_parts)
        else:
            run_name = f"run_{run_timestamp}"
            
        run_output_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(run_output_dir, exist_ok=True)
        print(f"Run output directory: {run_output_dir}")
        
        run_log_dir = os.path.join(args.log_dir, f"run_{run_timestamp}")
        os.makedirs(run_log_dir, exist_ok=True)
        print(f"Log directory for this run: {run_log_dir}")
        
        # Save run configuration
        run_config = {
            'run_timestamp': run_timestamp,
            'run_name': run_name,
            'era5_config': {
                'variables': era5_variables,
                'pressure_levels': era5_pressure_levels,
                'single_level_variables': era5_single_level_variables,
                'data_dir': args.era5_data_dir,
                'total_input_channels': total_input_channels
            },
            'training_config': {
                'folds': args.folds,
                'epochs': args.epochs,
                'batch_size': HARDCODED_BATCH_SIZE,
                'learning_rate': args.lr,
                'loss_type': args.loss_type,
                'optimizer': args.optimizer_type,
                'lr_scheduler': args.lr_scheduler_type,
                'target_source': args.target_source,
                'use_regional_focus': args.use_regional_focus,
                'outside_weight': args.outside_weight,
                'weight_decay': args.weight_decay,
                'log_offset': args.log_offset,
                'dropout': args.dropout
            },
            'data_paths': {
                'mswep_data_dir': args.data_dir,
                'test_data_dir': args.test_data_dir,
                'output_dir': run_output_dir,
                'log_dir': run_log_dir
            }
        }
        
        with open(os.path.join(run_output_dir, "run_configuration.json"), 'w') as f:
            json.dump(run_config, f, indent=2)
        print(f"Saved run configuration to: {os.path.join(run_output_dir, 'run_configuration.json')}")
        
        print(f"Starting training with {args.folds} folds.")
        all_fold_results = []
        for fold_num_main in range(args.folds): # Renamed
            print(f"\n=== Training fold {fold_num_main} ===")
            current_fold_output_dir = os.path.join(run_output_dir, f"fold{fold_num_main}") # Renamed
            fold_results_single = train_fold(fold_num_main, args, current_fold_output_dir, run_log_dir) # Renamed
            if fold_results_single:
                all_fold_results.append(fold_results_single)
        
        if args.skip_crps:
            print("\nWARNING: Skipping final comprehensive CRPS calculation as per --skip_crps flag.")
        else:
            if not ISODISREG_AVAILABLE:
                print("\nWARNING: isodisreg module not available. Skipping final comprehensive CRPS calculation.")
            else:
                print("\n" + "="*60 + "\nAll folds complete. Starting final comprehensive evaluation.\n" + "="*60)
                # Pass args to calculate_final_crps
                calculate_final_crps(args.output_dir, args.folds, skip_crps=False, args=args) 
                print("\nFinal comprehensive evaluation complete.")
        
        print(f"\nTraining and evaluation complete. Results saved to: {run_output_dir}")

    main()
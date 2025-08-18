#!/usr/bin/env python
"""
Test script to demonstrate using MAE loss function for MSWEP precipitation forecasting.

This script shows how to configure the model to use MAE instead of MSE as the loss function.
"""

import torch
import numpy as np
from models.mswep_unet import MSWEPUNet
from models.mswep_lightning_wrapper import UNetLightningModule
from data.mswep_data_module_2 import TargetLogScaler

def test_mae_loss():
    """Test MAE loss function implementation."""
    
    # Create a simple UNet model
    model = MSWEPUNet(in_channels=5)
    
    # Create target scaler with log transform
    target_scaler = TargetLogScaler(offset=0.1)
    
    # Create Lightning module with MAE loss
    lightning_module = UNetLightningModule(
        model=model,
        learning_rate=1e-4,
        loss_type='mae',  # Using MAE instead of MSE
        use_regional_focus=True,
        target_scaler=target_scaler
    )
    
    # Create dummy batch data
    batch_size = 4
    height, width = 41, 121
    channels = 5
    
    # Dummy input (5 channels: 3 precip lags + 2 seasonality)
    x = torch.randn(batch_size, channels, height, width)
    
    # Dummy targets in original scale (mm)
    y_original = torch.abs(torch.randn(batch_size, height, width)) * 10  # 0-10mm range
    
    # Apply log transform to create transformed targets
    y_transformed = target_scaler.transform(y_original)
    
    # Create batch tuple as expected by the model
    batch = (x, y_original, y_transformed)
    
    # Test training step
    print("Testing training step with MAE loss...")
    loss = lightning_module.training_step(batch, batch_idx=0)
    print(f"Training loss (MAE in log space): {loss.item():.4f}")
    
    # Test that the loss is reasonable
    assert loss.item() > 0, "Loss should be positive"
    assert loss.item() < 10, "Loss seems too high"
    
    # Compare with MSE loss
    lightning_module_mse = UNetLightningModule(
        model=model,
        learning_rate=1e-4,
        loss_type='mse',  # Using MSE for comparison
        use_regional_focus=True,
        target_scaler=target_scaler
    )
    
    loss_mse = lightning_module_mse.training_step(batch, batch_idx=0)
    print(f"Training loss (MSE in log space): {loss_mse.item():.4f}")
    
    # MAE is typically smaller than MSE for the same errors
    print(f"\nMAE/MSE ratio: {loss.item() / loss_mse.item():.3f}")
    print("(MAE is typically smaller than MSE for the same prediction errors)")
    
    # Test validation step
    print("\nTesting validation step with MAE loss...")
    val_output = lightning_module.validation_step(batch, batch_idx=0)
    val_loss = val_output['val_loss']
    print(f"Validation loss (MAE): {val_loss.item():.4f}")
    
    print("\nMAE loss function test completed successfully!")
    
    # Print configuration summary
    print("\n--- Configuration Summary ---")
    print(f"Loss function: {lightning_module.loss_type}")
    print(f"Log transform offset: {target_scaler.offset}")
    print(f"Regional focus: {lightning_module.hparams.use_regional_focus}")
    print("\nNote: MAE is computed in log space when log transform is enabled.")
    print("This provides balanced treatment of relative errors across precipitation ranges.")

if __name__ == "__main__":
    test_mae_loss() 
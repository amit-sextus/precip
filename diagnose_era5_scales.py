#!/usr/bin/env python
"""
Diagnostic script to check ERA5 data scales and identify why NaN values occur.
"""

import os
import sys
import numpy as np
import xarray as xr
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from data.mswep_data_module_2 import MSWEPDataModule

def check_data_scales():
    """Check the scales of MSWEP and ERA5 data."""
    print("="*70)
    print("ERA5 DATA SCALE DIAGNOSTIC")
    print("="*70)
    
    # Create data module with ERA5
    data_module = MSWEPDataModule(
        data_dir="data/MSWEP_daily",
        batch_size=32,
        num_workers=0,
        fold=0,
        num_total_folds=1,
        apply_log_transform=True,
        log_offset=0.1,
        era5_variables=['u', 'v'],
        era5_pressure_levels=[500, 850],
        era5_data_dir="precip_data/era5_precipitation_project/predictors",
        era5_single_level_variables=['msl', 'tcwv']
    )
    
    data_module.setup(stage='fit')
    
    # Get a batch of data
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    x, y_original, y_transformed = batch
    
    print(f"\nBatch shapes:")
    print(f"  Input (x): {x.shape}")
    print(f"  Target original: {y_original.shape}")
    print(f"  Target transformed: {y_transformed.shape}")
    
    # Analyze each channel
    print(f"\nChannel-wise statistics (Input tensor):")
    print(f"{'Channel':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'NaN%':>10}")
    print("-"*70)
    
    channel_names = [
        "Precip t-3", "Precip t-2", "Precip t-1", 
        "Sin(DOY)", "Cos(DOY)",
        # ERA5 variables for each lag
        "u500 (t-4)", "v500 (t-4)", "u850 (t-4)", "v850 (t-4)", "msl (t-4)", "tcwv (t-4)",
        "u500 (t-3)", "v500 (t-3)", "u850 (t-3)", "v850 (t-3)", "msl (t-3)", "tcwv (t-3)",
        "u500 (t-2)", "v500 (t-2)", "u850 (t-2)", "v850 (t-2)", "msl (t-2)", "tcwv (t-2)",
    ]
    
    for i in range(x.shape[1]):
        channel_data = x[:, i, :, :].numpy()
        nan_pct = 100 * np.isnan(channel_data).sum() / channel_data.size
        
        # Handle all-NaN case
        if nan_pct == 100:
            print(f"{channel_names[i]:<20} {'ALL NaN':>10} {'ALL NaN':>10} {'ALL NaN':>10} {'ALL NaN':>10} {nan_pct:>10.1f}")
        else:
            valid_data = channel_data[~np.isnan(channel_data)]
            if len(valid_data) > 0:
                print(f"{channel_names[i]:<20} {np.min(valid_data):>10.3f} {np.max(valid_data):>10.3f} "
                      f"{np.mean(valid_data):>10.3f} {np.std(valid_data):>10.3f} {nan_pct:>10.1f}")
            else:
                print(f"{channel_names[i]:<20} {'No valid':>10} {'No valid':>10} {'No valid':>10} {'No valid':>10} {nan_pct:>10.1f}")
    
    # Check target statistics
    print(f"\nTarget statistics:")
    print(f"  Original scale - Min: {y_original.min():.3f}, Max: {y_original.max():.3f}, "
          f"Mean: {y_original.mean():.3f}, Std: {y_original.std():.3f}")
    print(f"  Log scale - Min: {y_transformed.min():.3f}, Max: {y_transformed.max():.3f}, "
          f"Mean: {y_transformed.mean():.3f}, Std: {y_transformed.std():.3f}")
    
    # Check for extreme values
    print(f"\n⚠️  Extreme value check:")
    for i in range(x.shape[1]):
        channel_data = x[:, i, :, :].numpy()
        valid_data = channel_data[~np.isnan(channel_data)]
        if len(valid_data) > 0:
            if np.abs(valid_data).max() > 1000:
                print(f"  Channel {i} ({channel_names[i]}): Contains values > 1000!")
            if np.abs(valid_data).max() > 10000:
                print(f"  Channel {i} ({channel_names[i]}): Contains EXTREME values > 10000! Max: {np.abs(valid_data).max():.1f}")
    
    # Load raw ERA5 data to check original scales
    print("\n" + "="*70)
    print("CHECKING RAW ERA5 DATA SCALES")
    print("="*70)
    
    era5_dir = "precip_data/era5_precipitation_project/predictors/regridded"
    
    # Check single-level variables
    for var in ['msl', 'tcwv']:
        file_path = os.path.join(era5_dir, f"era5_{var}_regrid.nc")
        if os.path.exists(file_path):
            ds = xr.open_dataset(file_path)
            data = ds[var].values
            print(f"\n{var}:")
            print(f"  Shape: {data.shape}")
            print(f"  Min: {np.nanmin(data):.3f}, Max: {np.nanmax(data):.3f}")
            print(f"  Mean: {np.nanmean(data):.3f}, Std: {np.nanstd(data):.3f}")
            
            # Check expected units
            if var == 'msl':
                print(f"  Expected units: Pa (typical range 95000-105000)")
                if np.nanmean(data) > 50000:
                    print(f"  ✓ Appears to be in Pa")
                else:
                    print(f"  ⚠️  Values seem too low for Pa")
            elif var == 'tcwv':
                print(f"  Expected units: kg/m² (typical range 0-70)")
                if np.nanmax(data) < 100:
                    print(f"  ✓ Appears to be in kg/m²")
                else:
                    print(f"  ⚠️  Values seem too high for kg/m²")
            
            ds.close()
    
    # Check pressure-level variables
    for level in [500, 850]:
        file_path = os.path.join(era5_dir, f"era5_uvq{level}_2010_regrid.nc")
        if os.path.exists(file_path):
            ds = xr.open_dataset(file_path)
            print(f"\nPressure level {level}hPa:")
            for var in ['u', 'v']:
                if var in ds:
                    data = ds[var].values
                    print(f"  {var}: Min={np.nanmin(data):.3f}, Max={np.nanmax(data):.3f}, "
                          f"Mean={np.nanmean(data):.3f}, Std={np.nanstd(data):.3f}")
            ds.close()
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    print("1. Check if ERA5 variables need normalization/scaling")
    print("2. Mean sea level pressure (msl) is likely in Pa - consider scaling to hPa")
    print("3. Consider standardizing all ERA5 inputs to have zero mean and unit variance")
    print("4. Check if any channels have extreme outliers that could cause numerical instability")
    print("="*70)

if __name__ == "__main__":
    check_data_scales() 
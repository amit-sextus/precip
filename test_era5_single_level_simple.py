#!/usr/bin/env python
"""
Simple test to verify single-level ERA5 integration works correctly.
"""

import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from data.mswep_data_module_2 import MSWEPDataModule

def create_minimal_test_data():
    """Create minimal test data for MSWEP and ERA5."""
    print("Creating minimal test data...")
    
    # Setup directories
    os.makedirs("test_data/MSWEP_daily", exist_ok=True)
    os.makedirs("test_data/ERA5_predictors/regridded", exist_ok=True)
    
    # Grid
    lats = np.linspace(47.0, 55.0, 41)
    lons = np.linspace(5.0, 15.0, 121)
    
    # Create MSWEP data for 2007-2011 (to support multiple folds)
    for year in range(2007, 2012):
        for month in range(1, 13):
            times = pd.date_range(f"{year}-{month:02d}-01", periods=pd.Timestamp(f"{year}-{month:02d}-01").days_in_month, freq='D')
            precip = np.random.exponential(2.0, (len(times), 41, 121)).astype(np.float32)
            
            ds = xr.Dataset(
                {"precip": (("time", "latitude", "longitude"), precip)},
                coords={"time": times, "latitude": lats, "longitude": lons}
            )
            ds.to_netcdf(f"test_data/MSWEP_daily/mswep_daily_{year}{month:02d}.nc")
    
    # Create ERA5 single-level data (full time period in one file)
    all_times = pd.date_range("2007-01-01", "2011-12-31", freq='D')
    
    # Mean sea level pressure
    msl_data = np.random.normal(101325, 1000, (len(all_times), 41, 121)).astype(np.float32)
    ds_msl = xr.Dataset(
        {"msl": (("time", "latitude", "longitude"), msl_data)},
        coords={"time": all_times, "latitude": lats, "longitude": lons}
    )
    ds_msl.to_netcdf("test_data/ERA5_predictors/regridded/era5_msl_regrid.nc")
    
    # Total column water vapour
    tcwv_data = np.random.uniform(10, 50, (len(all_times), 41, 121)).astype(np.float32)
    ds_tcwv = xr.Dataset(
        {"tcwv": (("time", "latitude", "longitude"), tcwv_data)},
        coords={"time": all_times, "latitude": lats, "longitude": lons}
    )
    ds_tcwv.to_netcdf("test_data/ERA5_predictors/regridded/era5_tcwv_regrid.nc")
    
    print("Test data created successfully!")

def test_single_level_loading():
    """Test loading single-level ERA5 variables."""
    print("\n" + "="*50)
    print("Testing single-level ERA5 loading...")
    print("="*50)
    
    # Create data module with only single-level variables
    dm = MSWEPDataModule(
        data_dir="test_data/MSWEP_daily",
        batch_size=4,
        num_workers=0,
        fold=0,
        era5_single_level_variables=["msl", "tcwv"],
        era5_data_dir="test_data/ERA5_predictors"
    )
    
    # Setup data
    dm.setup(stage='fit')
    
    # Check if datasets were created
    assert dm.train_dataset is not None, "Train dataset not created"
    assert dm.val_dataset is not None, "Validation dataset not created"
    
    print(f"✓ Train dataset size: {len(dm.train_dataset)}")
    print(f"✓ Val dataset size: {len(dm.val_dataset)}")
    
    # Get a batch and check shapes
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    inputs, targets_orig, targets_trans = batch
    
    print(f"\nBatch shapes:")
    print(f"  Inputs: {inputs.shape}")
    print(f"  Targets (original): {targets_orig.shape}")
    print(f"  Targets (transformed): {targets_trans.shape}")
    
    # Expected channels: 5 base + 3*2 single-level = 11
    expected_channels = 5 + 3 * 2
    assert inputs.shape[1] == expected_channels, f"Expected {expected_channels} channels, got {inputs.shape[1]}"
    print(f"✓ Channel count correct: {inputs.shape[1]} channels")
    
    # Check for NaN values
    has_nan = torch.isnan(inputs).any()
    if has_nan:
        print("✗ WARNING: NaN values detected in inputs!")
        nan_count = torch.isnan(inputs).sum().item()
        total_values = inputs.numel()
        print(f"  NaN count: {nan_count}/{total_values} ({100*nan_count/total_values:.2f}%)")
        return False
    else:
        print("✓ No NaN values in inputs")
    
    # Test forward pass through a simple model
    print("\nTesting forward pass...")
    from models.mswep_unet import MSWEPUNet
    model = MSWEPUNet(in_channels=expected_channels)
    
    with torch.no_grad():
        output = model(inputs)
    
    print(f"  Model output shape: {output.shape}")
    
    output_has_nan = torch.isnan(output).any()
    if output_has_nan:
        print("✗ WARNING: NaN values in model output!")
        return False
    else:
        print("✓ No NaN values in model output")
    
    print("\n✓ Single-level ERA5 integration test PASSED!")
    return True

def test_mixed_variables():
    """Test loading both pressure-level and single-level variables."""
    print("\n" + "="*50)
    print("Testing mixed ERA5 variables...")
    print("="*50)
    
    # First create pressure-level data
    lats = np.linspace(47.0, 55.0, 41)
    lons = np.linspace(5.0, 15.0, 121)
    
    # Create pressure-level data for 2007-2011
    for year in range(2007, 2012):
        for pressure in [500]:
            times = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq='D')
            u_data = np.random.normal(5, 10, (len(times), 41, 121)).astype(np.float32)
            v_data = np.random.normal(0, 8, (len(times), 41, 121)).astype(np.float32)
            q_data = np.random.uniform(0.001, 0.015, (len(times), 41, 121)).astype(np.float32)
            
            ds = xr.Dataset(
                {
                    "u": (("time", "latitude", "longitude"), u_data),
                    "v": (("time", "latitude", "longitude"), v_data),
                    "q": (("time", "latitude", "longitude"), q_data)
                },
                coords={"time": times, "latitude": lats, "longitude": lons}
            )
            ds.to_netcdf(f"test_data/ERA5_predictors/regridded/era5_uvq{pressure}_{year}_regrid.nc")
    
    # Create data module with mixed variables
    dm = MSWEPDataModule(
        data_dir="test_data/MSWEP_daily",
        batch_size=4,
        num_workers=0,
        fold=0,
        era5_variables=["u", "v"],
        era5_pressure_levels=[500],
        era5_single_level_variables=["msl", "tcwv"],
        era5_data_dir="test_data/ERA5_predictors"
    )
    
    dm.setup(stage='fit')
    
    # Get a batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    inputs, _, _ = batch
    
    # Expected channels: 5 base + 3*(2 pressure + 2 single) = 17
    expected_channels = 5 + 3 * (2 + 2)
    print(f"Input shape: {inputs.shape}")
    print(f"Expected channels: {expected_channels}")
    
    assert inputs.shape[1] == expected_channels, f"Expected {expected_channels} channels, got {inputs.shape[1]}"
    
    has_nan = torch.isnan(inputs).any()
    if has_nan:
        print("✗ WARNING: NaN values detected in mixed inputs!")
        return False
    else:
        print("✓ No NaN values in mixed inputs")
        print("✓ Mixed variables test PASSED!")
        return True

def cleanup():
    """Clean up test data."""
    import shutil
    if os.path.exists("test_data"):
        shutil.rmtree("test_data")
        print("\nCleaned up test data.")

if __name__ == "__main__":
    try:
        # Create test data
        create_minimal_test_data()
        
        # Run tests
        test1_passed = test_single_level_loading()
        test2_passed = test_mixed_variables()
        
        # Cleanup
        cleanup()
        
        # Summary
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        print(f"Single-level loading: {'PASSED' if test1_passed else 'FAILED'}")
        print(f"Mixed variables: {'PASSED' if test2_passed else 'FAILED'}")
        
        if test1_passed and test2_passed:
            print("\nAll tests PASSED! ✓")
            sys.exit(0)
        else:
            print("\nSome tests FAILED! ✗")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        cleanup()
        sys.exit(1) 
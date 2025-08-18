#!/usr/bin/env python3
"""
Test script to verify the complete ERA5 integration pipeline.
This runs a minimal training with ERA5 data to ensure all components work together.
"""

import os
import sys
import subprocess
import shutil
import tempfile
import numpy as np
import xarray as xr
import pandas as pd

# Add project root to path
sys.path.append('.')

def create_minimal_test_data(base_dir):
    """Create minimal test data for MSWEP and ERA5."""
    print("Creating minimal test data...")
    
    # Create directories
    mswep_dir = os.path.join(base_dir, "MSWEP_daily")
    era5_dir = os.path.join(base_dir, "ERA5_predictors")
    os.makedirs(mswep_dir, exist_ok=True)
    os.makedirs(era5_dir, exist_ok=True)
    
    # Grid specifications (small for testing)
    lats = np.arange(30, 30 + 41*1.0, 1.0)
    lons = np.arange(-70, -70 + 121*0.5, 0.5)
    
    # Create minimal MSWEP data for 2007-2010 (required for fold 0)
    years = [2007, 2008, 2009, 2010]
    
    for year in years:
        times = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        
        # Create MSWEP monthly files
        for month in range(1, 13):
            month_mask = times.month == month
            month_times = times[month_mask]
            n_days_month = len(month_times)
            
            precip_data = np.random.exponential(1.0, size=(n_days_month, len(lats), len(lons)))
            
            ds_mswep = xr.Dataset(
                {
                    "precip": (["time", "lat", "lon"], precip_data)
                },
                coords={
                    "time": month_times,
                    "lat": lats,
                    "lon": lons
                }
            )
            
            filename = f"mswep_daily_{year}{month:02d}.nc"
            filepath = os.path.join(mswep_dir, filename)
            ds_mswep.to_netcdf(filepath)
            print(f"  Created {filename}")
    
    # Create ERA5 data for two variables and two pressure levels
    era5_vars = ['u', 'v']  # Wind components
    pressure_levels = [500, 850]
    
    # Create data for full time range 2007-2010
    all_times = pd.date_range("2007-01-01", "2010-12-31", freq="D")
    n_days_total = len(all_times)
    
    for var in era5_vars:
        for level in pressure_levels:
            # Create some correlated data
            base_pattern = np.sin(np.linspace(0, 4*np.pi, n_days_total))
            era5_data = np.zeros((n_days_total, len(lats), len(lons)))
            
            for i in range(n_days_total):
                spatial_pattern = np.random.randn(len(lats), len(lons))
                era5_data[i] = base_pattern[i] * 10 + spatial_pattern * 2
            
            ds_era5 = xr.Dataset(
                {
                    var: (["time", "lat", "lon"], era5_data)
                },
                coords={
                    "time": all_times,
                    "lat": lats,
                    "lon": lons
                }
            )
            
            # Use the expected filename pattern
            filename = f"era5_{var}{level}_train_2007_2019.nc"
            filepath = os.path.join(era5_dir, filename)
            ds_era5.to_netcdf(filepath)
            print(f"  Created {filename}")
    
    return mswep_dir, era5_dir


def run_minimal_training(mswep_dir, era5_dir, output_dir):
    """Run minimal training with ERA5 data."""
    print("\nRunning minimal training pipeline...")
    
    cmd = [
        "python", "models/mswep_unet_training.py",
        "--data_dir", mswep_dir,
        "--output_dir", output_dir,
        "--log_dir", os.path.join(output_dir, "logs"),
        "--epochs", "2",  # Minimal epochs
        "--folds", "1",   # Single fold
        "--batch_size", "4",  # Small batch
        "--lr", "0.001",
        "--era5_variables", "u,v",
        "--era5_pressure_levels", "500,850",
        "--era5_data_dir", era5_dir,
        "--skip_crps"  # Skip final evaluation for speed
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("\nTraining completed successfully!")
        print("\nStdout:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code {e.returncode}")
        print("\nStderr:")
        print(e.stderr)
        print("\nStdout:")
        print(e.stdout)
        return False


def verify_outputs(output_dir):
    """Verify that expected outputs were created."""
    print("\nVerifying outputs...")
    
    # Find the run directory
    run_dirs = [d for d in os.listdir(output_dir) if d.startswith("run_") and os.path.isdir(os.path.join(output_dir, d))]
    if not run_dirs:
        print("ERROR: No run directory found!")
        return False
    
    run_dir = os.path.join(output_dir, run_dirs[0])
    print(f"Found run directory: {run_dir}")
    
    # Check for expected files
    expected_files = [
        "run_configuration.json",
        "fold0/best_model.ckpt",
        "fold0/training_results_fold0.json"
    ]
    
    all_found = True
    for file_path in expected_files:
        full_path = os.path.join(run_dir, file_path)
        if os.path.exists(full_path):
            print(f"  ✓ Found: {file_path}")
        else:
            print(f"  ✗ Missing: {file_path}")
            all_found = False
    
    # Check run configuration
    import json
    config_path = os.path.join(run_dir, "run_configuration.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("\nRun configuration:")
        print(f"  ERA5 variables: {config['era5_config']['variables']}")
        print(f"  ERA5 pressure levels: {config['era5_config']['pressure_levels']}")
        print(f"  Total input channels: {config['era5_config']['total_input_channels']}")
        
        # Verify channel calculation
        expected_channels = 5 + 3 * 2 * 2  # 5 base + 3 lags × 2 vars × 2 levels = 17
        if config['era5_config']['total_input_channels'] == expected_channels:
            print(f"  ✓ Channel calculation correct: {expected_channels}")
        else:
            print(f"  ✗ Channel calculation incorrect: expected {expected_channels}, got {config['era5_config']['total_input_channels']}")
            all_found = False
    
    return all_found


def main():
    """Main test function."""
    print("="*70)
    print("ERA5 Integration Pipeline Test")
    print("="*70)
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nUsing temporary directory: {temp_dir}")
        
        # Create test data
        mswep_dir, era5_dir = create_minimal_test_data(temp_dir)
        
        # Create output directory
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Run training
        success = run_minimal_training(mswep_dir, era5_dir, output_dir)
        
        if success:
            # Verify outputs
            outputs_ok = verify_outputs(output_dir)
            
            if outputs_ok:
                print("\n" + "="*70)
                print("✓ ERA5 INTEGRATION TEST PASSED!")
                print("="*70)
                return 0
            else:
                print("\n" + "="*70)
                print("✗ Output verification failed!")
                print("="*70)
                return 1
        else:
            print("\n" + "="*70)
            print("✗ Training failed!")
            print("="*70)
            return 1


if __name__ == "__main__":
    sys.exit(main()) 
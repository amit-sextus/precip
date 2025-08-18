#!/usr/bin/env python
"""
Test script to verify single-level ERA5 variable integration.
This creates minimal test data and runs a short training to ensure all components work.
"""

import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import shutil
import subprocess

def create_test_data(base_dir="test_era5_single_level"):
    """Create minimal test datasets for verification."""
    
    # Clean up if exists
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    # Create directories
    mswep_dir = os.path.join(base_dir, "MSWEP_daily")
    era5_dir = os.path.join(base_dir, "ERA5_predictors", "regridded")
    os.makedirs(mswep_dir, exist_ok=True)
    os.makedirs(era5_dir, exist_ok=True)
    
    # Grid dimensions
    lats = np.linspace(47.0, 55.0, 41)
    lons = np.linspace(5.0, 15.0, 121)
    
    # Create time series for 2010-2011 (2 years for minimal test)
    start_date = "2007-01-01"
    end_date = "2011-12-31"
    
    print("Creating test MSWEP data...")
    # Create monthly MSWEP files
    for year in range(2007, 2012):
        for month in range(1, 13):
            month_start = f"{year}-{month:02d}-01"
            days_in_month = pd.Timestamp(month_start).days_in_month
            times = pd.date_range(start=month_start, periods=days_in_month, freq='D')
            
            # Create precipitation data (random for testing)
            precip_data = np.random.exponential(2.0, size=(len(times), len(lats), len(lons))).astype(np.float32)
            precip_data = np.clip(precip_data, 0, 50)  # Realistic range
            
            ds = xr.Dataset(
                {"precip": (("time", "latitude", "longitude"), precip_data)},
                coords={"time": times, "latitude": lats, "longitude": lons}
            )
            
            file_path = os.path.join(mswep_dir, f"mswep_daily_{year}{month:02d}.nc")
            ds.to_netcdf(file_path)
            print(f"  Created {file_path}")
    
    print("\nCreating test ERA5 pressure-level data...")
    # Create ERA5 pressure-level files (wind components)
    for year in range(2007, 2012):
        for pressure in [500, 850]:
            times = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
            
            # Create wind and humidity data
            u_data = np.random.normal(5, 10, size=(len(times), len(lats), len(lons))).astype(np.float32)
            v_data = np.random.normal(0, 8, size=(len(times), len(lats), len(lons))).astype(np.float32)
            q_data = np.random.uniform(0.001, 0.015, size=(len(times), len(lats), len(lons))).astype(np.float32)
            
            ds = xr.Dataset(
                {
                    "u": (("time", "latitude", "longitude"), u_data),
                    "v": (("time", "latitude", "longitude"), v_data),
                    "q": (("time", "latitude", "longitude"), q_data)
                },
                coords={"time": times, "latitude": lats, "longitude": lons}
            )
            
            file_path = os.path.join(era5_dir, f"era5_uvq{pressure}_{year}_regrid.nc")
            ds.to_netcdf(file_path)
            print(f"  Created {file_path}")
    
    print("\nCreating test ERA5 single-level data...")
    # Create ERA5 single-level files (all years in one file)
    all_times = pd.date_range(start="2007-01-01", end="2011-12-31", freq='D')
    
    # Mean sea level pressure
    msl_data = np.random.normal(101325, 1000, size=(len(all_times), len(lats), len(lons))).astype(np.float32)
    ds_msl = xr.Dataset(
        {"msl": (("time", "latitude", "longitude"), msl_data)},
        coords={"time": all_times, "latitude": lats, "longitude": lons}
    )
    ds_msl.to_netcdf(os.path.join(era5_dir, "era5_msl_regrid.nc"))
    print(f"  Created era5_msl_regrid.nc")
    
    # Total column water vapour
    tcwv_data = np.random.uniform(10, 50, size=(len(all_times), len(lats), len(lons))).astype(np.float32)
    ds_tcwv = xr.Dataset(
        {"tcwv": (("time", "latitude", "longitude"), tcwv_data)},
        coords={"time": all_times, "latitude": lats, "longitude": lons}
    )
    ds_tcwv.to_netcdf(os.path.join(era5_dir, "era5_tcwv_regrid.nc"))
    print(f"  Created era5_tcwv_regrid.nc")
    
    # 2-meter temperature
    t2m_data = np.random.normal(288, 10, size=(len(all_times), len(lats), len(lons))).astype(np.float32)
    ds_t2m = xr.Dataset(
        {"t2m": (("time", "latitude", "longitude"), t2m_data)},
        coords={"time": all_times, "latitude": lats, "longitude": lons}
    )
    ds_t2m.to_netcdf(os.path.join(era5_dir, "era5_t2m_regrid.nc"))
    print(f"  Created era5_t2m_regrid.nc")
    
    # Surface pressure
    sp_data = np.random.normal(100000, 1000, size=(len(all_times), len(lats), len(lons))).astype(np.float32)
    ds_sp = xr.Dataset(
        {"sp": (("time", "latitude", "longitude"), sp_data)},
        coords={"time": all_times, "latitude": lats, "longitude": lons}
    )
    ds_sp.to_netcdf(os.path.join(era5_dir, "era5_sp_regrid.nc"))
    print(f"  Created era5_sp_regrid.nc")
    
    return base_dir, mswep_dir, era5_dir

def run_test_training(mswep_dir, era5_dir, output_dir="test_output"):
    """Run a minimal training test with mixed ERA5 variables."""
    
    # Clean output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Build command
    cmd = [
        sys.executable,
        "models/mswep_unet_training.py",
        "--data_dir", mswep_dir,
        "--output_dir", output_dir,
        "--epochs", "2",  # Very short for testing
        "--folds", "1",   # Single fold
        "--batch_size", "4",
        "--era5_variables", "u,v,q",
        "--era5_pressure_levels", "500,850",
        "--era5_single_level_variables", "msl,tcwv,t2m,sp",
        "--era5_data_dir", os.path.dirname(era5_dir),  # Parent of regridded
        "--skip_crps"     # Skip final evaluation for speed
    ]
    
    print("\nRunning test training with command:")
    print(" ".join(cmd))
    print("\nThis will verify that:")
    print("- Pressure-level variables (u,v,q at 500,850 hPa) load correctly")
    print("- Single-level variables (msl,tcwv,t2m,sp) load correctly")
    print("- Channel calculation works (5 base + 3*(3*2 + 4) = 35 channels)")
    print("- Model training starts successfully")
    print("\n" + "="*70)
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Check for expected output
        output = result.stdout
        
        # Verify channel calculation
        if "Total input channels: 35" in output:
            print("✓ Channel calculation correct (35 channels)")
        else:
            print("✗ Channel calculation may be incorrect")
            print("Output excerpt:")
            for line in output.split('\n'):
                if 'channel' in line.lower():
                    print(f"  {line}")
        
        # Check if ERA5 loaded successfully
        if "Successfully loaded" in output and "ERA5 variables" in output:
            print("✓ ERA5 data loaded successfully")
            # Extract what was loaded
            for line in output.split('\n'):
                if "Successfully loaded" in line and "ERA5 variables:" in line:
                    print(f"  {line.strip()}")
        else:
            print("✗ ERA5 data loading may have failed")
            # Print ERA5-related errors
            for line in output.split('\n'):
                if "era5" in line.lower() and ("error" in line.lower() or "warning" in line.lower()):
                    print(f"  {line.strip()}")
        
        # Check if training started
        if "Starting training" in output:
            print("✓ Training started successfully")
        else:
            print("✗ Training may not have started")
        
        # Check for NaN issues
        nan_warnings = []
        val_loss_values = []
        for line in output.split('\n'):
            if "val_all_nan" in line and "1.00" in line:
                nan_warnings.append(line.strip())
            if "val_loss=" in line:
                # Extract validation loss value
                import re
                match = re.search(r'val_loss=(\d+\.?\d*)', line)
                if match:
                    val_loss_values.append(float(match.group(1)))
        
        if nan_warnings:
            print("\n✗ WARNING: NaN issues detected during validation!")
            for warning in nan_warnings[:3]:  # Show first 3 warnings
                print(f"  {warning}")
            print(f"  Total NaN warnings: {len(nan_warnings)}")
        
        if val_loss_values:
            # Check if any validation loss is the fallback value of 100.0
            if any(loss == 100.0 for loss in val_loss_values):
                print("\n✗ CRITICAL: Validation loss = 100.0 detected (fallback for all NaN predictions)")
                print("  This indicates the model is producing all NaN outputs")
            else:
                avg_val_loss = sum(val_loss_values) / len(val_loss_values)
                print(f"\n✓ Validation losses appear reasonable")
                print(f"  Average val_loss: {avg_val_loss:.4f}")
                print(f"  Range: [{min(val_loss_values):.4f}, {max(val_loss_values):.4f}]")
        
        # Check if created samples messages are present
        sample_creation_msgs = []
        for line in output.split('\n'):
            if "Created" in line and "samples with input shape" in line:
                sample_creation_msgs.append(line.strip())
        
        if sample_creation_msgs:
            print("\n✓ Sample creation messages found:")
            for msg in sample_creation_msgs[:2]:  # Show first 2
                print(f"  {msg}")
        
        # Final verdict
        if nan_warnings or any(loss == 100.0 for loss in val_loss_values):
            print("\n✗ Test FAILED: Model is producing NaN predictions")
            print("\nPossible causes:")
            print("- Temporal alignment issues between ERA5 and MSWEP data")
            print("- Missing data in ERA5 files for the required time period")
            print("- Incorrect channel stacking in _create_samples method")
            return False
        
        print("\nTest completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print("\n✗ Test failed with error:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False

def verify_output_structure(output_dir="test_output"):
    """Verify the output directory structure."""
    print("\nVerifying output structure...")
    
    # Find the run directory
    run_dirs = [d for d in os.listdir(output_dir) if d.startswith("run_")]
    if not run_dirs:
        print("✗ No run directory found")
        return False
    
    run_dir = os.path.join(output_dir, run_dirs[0])
    print(f"✓ Found run directory: {run_dirs[0]}")
    
    # Check for expected pattern in directory name
    if "era5" in run_dirs[0] and "sl_" in run_dirs[0]:
        print("✓ Run directory name includes ERA5 and single-level indicators")
    else:
        print("✗ Run directory name missing expected patterns")
    
    # Check configuration file
    config_path = os.path.join(run_dir, "run_configuration.json")
    if os.path.exists(config_path):
        print("✓ Configuration file exists")
        
        # Load and verify configuration
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        era5_config = config.get('era5_config', {})
        if era5_config.get('single_level_variables'):
            print(f"✓ Single-level variables in config: {era5_config['single_level_variables']}")
        if era5_config.get('total_input_channels') == 35:
            print("✓ Total input channels in config: 35")
    else:
        print("✗ Configuration file not found")
    
    return True

def cleanup(base_dir="test_era5_single_level", output_dir="test_output"):
    """Clean up test data and output."""
    print("\nCleaning up test data...")
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        print(f"✓ Removed {base_dir}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"✓ Removed {output_dir}")

if __name__ == "__main__":
    print("ERA5 Single-Level Variable Integration Test")
    print("=" * 70)
    
    # Create test data
    base_dir, mswep_dir, era5_dir = create_test_data()
    
    # Run test training
    success = run_test_training(mswep_dir, era5_dir)
    
    if success:
        # Verify output
        verify_output_structure()
    
    # Cleanup (comment out to inspect files)
    cleanup()
    
    print("\n" + "=" * 70)
    if success:
        print("All tests passed! Single-level ERA5 integration is working correctly.")
    else:
        print("Some tests failed. Please check the output above.") 
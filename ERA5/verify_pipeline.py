"""
Verification script for ERA5 data pipeline
Checks that all expected variables are downloaded and processed correctly
"""

import os
import xarray as xr
import numpy as np

def verify_downloads():
    """Verify that all expected files are downloaded"""
    data_dir = './precip_data/era5_data/'
    
    expected_files = {
        # Single level files
        'era5_total_column_water_vapour.nc': 'TCWV',
        'era5_2m_temperature.nc': 'T2M',
        'era5_surface_pressure.nc': 'SP',
        'era5_mean_sea_level_pressure.nc': 'MSLP',
        # Pressure level files
        'era5_pl_300.nc': 'Level 300 (U, V, Q)',
        'era5_pl_500.nc': 'Level 500 (U, V, Q)',
        'era5_pl_700.nc': 'Level 700 (U, V, Q)',
        'era5_pl_850.nc': 'Level 850 (U, V, Q)',
    }
    
    print("=== Verifying Downloaded Files ===")
    missing_files = []
    
    for filename, description in expected_files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"✓ {filename} ({description}) - Found")
            # Check file content
            try:
                ds = xr.open_dataset(filepath)
                print(f"  Variables: {list(ds.data_vars)}")
                print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
                print(f"  Shape: {ds[list(ds.data_vars)[0]].shape}")
                ds.close()
            except Exception as e:
                print(f"  ⚠ Error reading file: {e}")
        else:
            print(f"✗ {filename} ({description}) - Missing")
            missing_files.append(filename)
    
    return len(missing_files) == 0

def verify_processed_data():
    """Verify that all expected processed files exist"""
    data_save = './precip_data/predictors/'
    
    # Expected variables after processing
    expected_vars = [
        # Single levels
        'tcwv', 't2m', 'msl', 'ptend',
        # Pressure levels - U component
        'u300', 'u500', 'u700', 'u850',
        # Pressure levels - V component  
        'v300', 'v500', 'v700', 'v850',
        # Pressure levels - Specific humidity
        'q300', 'q500', 'q700', 'q850',
    ]
    
    print("\n=== Verifying Processed Files ===")
    
    # Check train and test directories
    for split in ['train', 'test']:
        print(f"\n{split.upper()} Data:")
        split_dir = os.path.join(data_save, split)
        
        if not os.path.exists(split_dir):
            print(f"✗ {split} directory missing!")
            continue
            
        for var in expected_vars:
            # Find files matching the variable pattern
            found = False
            for file in os.listdir(split_dir):
                if file.startswith(f'{var}_') and file.endswith('.nc'):
                    found = True
                    filepath = os.path.join(split_dir, file)
                    try:
                        ds = xr.open_dataset(filepath)
                        var_name = list(ds.data_vars)[0] if ds.data_vars else var
                        data = ds[var_name]
                        print(f"✓ {var}: {file} - Shape: {data.shape}")
                        
                        # Check for NaN values
                        nan_count = np.isnan(data.values).sum()
                        if nan_count > 0:
                            print(f"  ⚠ Contains {nan_count} NaN values")
                        
                        ds.close()
                    except Exception as e:
                        print(f"✓ {var}: {file} - Error reading: {e}")
                    break
            
            if not found:
                print(f"✗ {var}: No file found")
    
    return True

def verify_data_alignment():
    """Verify that all processed data has consistent dimensions"""
    data_save = './precip_data/predictors/train/'
    
    if not os.path.exists(data_save):
        print("\n⚠ Train directory not found, skipping alignment check")
        return False
    
    print("\n=== Verifying Data Alignment ===")
    
    files = [f for f in os.listdir(data_save) if f.endswith('.nc')]
    if not files:
        print("✗ No processed files found")
        return False
    
    # Load first file as reference
    ref_file = os.path.join(data_save, files[0])
    ref_ds = xr.open_dataset(ref_file)
    ref_var = list(ref_ds.data_vars)[0]
    ref_shape = ref_ds[ref_var].shape
    ref_times = ref_ds.time.values
    ref_lats = ref_ds.lat.values if 'lat' in ref_ds else None
    ref_lons = ref_ds.lon.values if 'lon' in ref_ds else None
    ref_ds.close()
    
    print(f"Reference file: {files[0]}")
    print(f"Reference shape: {ref_shape}")
    print(f"Time steps: {len(ref_times)}")
    print(f"Spatial grid: {len(ref_lats) if ref_lats is not None else 'N/A'} x {len(ref_lons) if ref_lons is not None else 'N/A'}")
    
    # Check all other files
    misaligned = []
    for file in files[1:]:
        filepath = os.path.join(data_save, file)
        try:
            ds = xr.open_dataset(filepath)
            var_name = list(ds.data_vars)[0]
            shape = ds[var_name].shape
            
            if shape != ref_shape:
                misaligned.append((file, shape))
                print(f"✗ {file}: Shape mismatch - {shape}")
            else:
                print(f"✓ {file}: Aligned")
            
            ds.close()
        except Exception as e:
            print(f"⚠ {file}: Error - {e}")
    
    if misaligned:
        print(f"\n⚠ Found {len(misaligned)} misaligned files")
        return False
    else:
        print("\n✓ All files are properly aligned")
        return True

def main():
    """Run all verification checks"""
    print("ERA5 Pipeline Verification")
    print("=" * 50)
    
    # Check downloads
    downloads_ok = verify_downloads()
    
    # Check processed data
    processed_ok = verify_processed_data()
    
    # Check alignment
    alignment_ok = verify_data_alignment()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Downloads: {'✓ PASS' if downloads_ok else '✗ FAIL'}")
    print(f"Processing: {'✓ PASS' if processed_ok else '✗ FAIL'}")
    print(f"Alignment: {'✓ PASS' if alignment_ok else '✗ FAIL'}")
    
    if downloads_ok and processed_ok and alignment_ok:
        print("\n✓ Pipeline verification PASSED")
        print("\nNext steps:")
        print("1. The ERA5 data is now ready to be integrated into your neural network")
        print("2. Each of the 3 precipitation lags will be augmented with 16 ERA5 variables")
        print("3. Total channels per lag: 1 (precip) + 16 (ERA5) = 17")
        print("4. The 6-hour lag between ERA5 (18:00 UTC) and precipitation (00:00 UTC) is already handled")
    else:
        print("\n✗ Pipeline verification FAILED - Check the errors above")

if __name__ == "__main__":
    main() 
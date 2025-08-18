import numpy as np
import xarray as xr
import pandas as pd
import os
import time
import glob
from helpers import con_grid, regrid_bilinear, get_pressure_tendency, get_short_name, handle_expver

# Match data_dir with updated ERA5_download.py
data_dir = '../precip_data/era5_precipitation_project/'
# Save processed predictors in a subdirectory of the project folder
data_save = '../precip_data/era5_precipitation_project/predictors/'
regrid_dir = os.path.join(data_save, 'regridded')
temp_dir = os.path.join(data_save, 'temp')

os.makedirs(os.path.join(data_save, 'train'), exist_ok=True)
os.makedirs(os.path.join(data_save, 'test'), exist_ok=True)
os.makedirs(regrid_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

# Define target grid file (make sure this file exists)
target_grid_file = 'targetgrid_germany.txt'

train_start_date = '2007-01-01T18:00:00'
train_end_date = '2019-12-31T18:00:00'
test_start_date = '2020-01-01T18:00:00'
test_end_date = '2020-12-31T18:00:00'

train_time_index = pd.date_range(start=train_start_date, end=train_end_date, freq='D')
test_time_index = pd.date_range(start=test_start_date, end=test_end_date, freq='D')

# Define all variables to process based on new download configuration
variables_to_process = {
    # Single Levels
    'total_column_water_vapour': {'regrid': 'conservative'},
    '2m_temperature': {'regrid': 'bilinear'},
    'surface_pressure': {'regrid': 'bilinear', 'process': 'pressure_tendency'},
    'mean_sea_level_pressure': {'regrid': 'bilinear'},
}

# Add pressure level variables for all 4 levels
pressure_levels = ['300', '500', '700', '850']
pressure_vars = ['u_component_of_wind', 'v_component_of_wind', 'specific_humidity']

for level in pressure_levels:
    for var in pressure_vars:
        key = f"{var}_{level}"
        variables_to_process[key] = {
            'regrid': 'bilinear',
            'level': int(level),
            'var': var.split('_')[0]  # Extract 'u', 'v', or 'specific' (will map to 'q')
        }

processed_data = {}

# Function to get input file paths - returns list of files for pressure levels
def get_input_paths(var_info, var_key):
    if 'level' in var_info:
        level = var_info['level']
        # Find all yearly files for this pressure level
        pattern = os.path.join(data_dir, f"era5_pl_{level}_*.nc")
        yearly_files = sorted(glob.glob(pattern))
        
        if yearly_files:
            print(f"Found {len(yearly_files)} yearly files for level {level}")
            return yearly_files
        else:
            print(f"Warning: No files found for pressure level {level}")
            return []
    else:
        # For single-level variables, return as single-item list for consistency
        single_file = os.path.join(data_dir, f"era5_{var_key}.nc")
        if os.path.exists(single_file):
            return [single_file]
        else:
            print(f"Warning: File not found: {single_file}")
            return []

print("--- Starting Regridding ---")
regridded_files = {}

for key, info in variables_to_process.items():
    input_files = get_input_paths(info, key)
    if not input_files:
        print(f"Warning: No input files found for {key}, skipping.")
        continue
    
    level = info.get('level')
    short_name = get_short_name(key, level)
    
    # For pressure levels, process each yearly file
    if 'level' in info:
        regridded_yearly_files = []
        for input_file in input_files:
            # Extract year from filename
            year = os.path.basename(input_file).split('_')[-1].replace('.nc', '')
            output_file = os.path.join(regrid_dir, f"era5_{short_name}_{year}_regrid.nc")
            regridded_yearly_files.append(output_file)
            
            if os.path.exists(output_file):
                print(f"Regridded file already exists, skipping: {output_file}")
                continue
            
            print(f"Regridding {key} year {year} from {os.path.basename(input_file)}...")
            try:
                if info['regrid'] == 'conservative':
                    con_grid(input_file, output_file, target_grid_file)
                elif info['regrid'] == 'bilinear':
                    regrid_bilinear(input_file, output_file, target_grid_file)
                else:
                    print(f"Warning: Unknown regridding method '{info['regrid']}' for {key}")
            except Exception as e:
                print(f"ERROR during regridding for {key} year {year}: {e}")
        
        regridded_files[key] = regridded_yearly_files
    else:
        # For single-level variables, process normally
        output_file = os.path.join(regrid_dir, f"era5_{short_name}_regrid.nc")
        regridded_files[key] = [output_file]
        
        if os.path.exists(output_file):
            print(f"Regridded file already exists, skipping: {output_file}")
            continue
        
        print(f"Regridding {key} from {os.path.basename(input_files[0])}...")
        try:
            if info['regrid'] == 'conservative':
                con_grid(input_files[0], output_file, target_grid_file)
            elif info['regrid'] == 'bilinear':
                regrid_bilinear(input_files[0], output_file, target_grid_file)
            else:
                print(f"Warning: Unknown regridding method '{info['regrid']}' for {key}")
        except Exception as e:
            print(f"ERROR during regridding for {key}: {e}")

print("--- Regridding Complete ---")

# --- Processing Step ---
print("\n--- Starting Processing ---")
processed_vars = {}

# Process individual variables
for key, info in variables_to_process.items():
    regrid_file_paths = regridded_files.get(key, [])
    if not regrid_file_paths or not any(os.path.exists(f) for f in regrid_file_paths):
        print(f"Warning: No regridded files found for {key}, skipping processing.")
        continue

    # Skip surface pressure for direct processing (will be used for pressure tendency)
    if info.get('process') == 'pressure_tendency' and key == 'surface_pressure':
        continue

    print(f"Processing {key}...")
    
    # For pressure levels, open multiple files and concatenate
    if 'level' in info and len(regrid_file_paths) > 1:
        # Filter existing files
        existing_files = [f for f in regrid_file_paths if os.path.exists(f)]
        if not existing_files:
            print(f"Warning: No existing regridded files for {key}, skipping.")
            continue
            
        # Use open_mfdataset to open all yearly files
        ds = xr.open_mfdataset(existing_files, combine='by_coords', engine='netcdf4')
    else:
        # For single files
        if not os.path.exists(regrid_file_paths[0]):
            print(f"Warning: Regridded file not found for {key}, skipping.")
            continue
        ds = xr.open_dataset(regrid_file_paths[0])

    # Standardize coordinate names
    if 'valid_time' in ds.dims: ds = ds.rename({'valid_time': 'time'})
    if 'valid_time' in ds.coords: ds = ds.rename({'valid_time': 'time'})
    if 'longitude' in ds.dims: ds = ds.rename({'longitude': 'lon'})
    if 'longitude' in ds.coords: ds = ds.rename({'longitude': 'lon'})
    if 'latitude' in ds.dims: ds = ds.rename({'latitude': 'lat'})
    if 'latitude' in ds.coords: ds = ds.rename({'latitude': 'lat'})

    if 'level' in info:
        # For pressure level variables, map the correct variable name
        if info['var'] == 'specific':
            var_name_in_file = 'q'
        else:
            var_name_in_file = info['var']
    else:
        # For single-level files, get the first (and usually only) variable
        var_name_in_file = list(ds.data_vars)[0]

    if var_name_in_file not in ds:
        print(f"Warning: Variable '{var_name_in_file}' not found in {key} files, skipping.")
        ds.close()
        continue

    data_var = ds[var_name_in_file]

    # Handle expver dimension if present
    data_var = handle_expver(data_var)

    # Squeeze dimension if only one level exists
    if 'level' in data_var.dims and data_var.dims['level'] == 1:
        data_var = data_var.squeeze('level', drop=True)
    if 'plev' in data_var.dims and data_var.dims['plev'] == 1:
        data_var = data_var.squeeze('plev', drop=True)

    level = info.get('level')
    short_name = get_short_name(key, level)
    processed_vars[short_name] = data_var.sortby('time')
    ds.close()

print("\nCalculating derived variables...")
# Pressure Tendency
sp_regridded_files = regridded_files.get('surface_pressure', [])
if sp_regridded_files and os.path.exists(sp_regridded_files[0]):
    ptend_da = get_pressure_tendency(sp_regridded_files[0])
    if ptend_da is not None:
        processed_vars['ptend'] = ptend_da.sortby('time')
else:
    print("Warning: Cannot calculate pressure tendency, regridded surface_pressure file missing.")

print("\n--- Aligning Timesteps and Saving ---")

# Align all processed variables to a common time index
full_time_index = pd.date_range(start=train_start_date, end=test_end_date, freq='D')

aligned_processed_vars = {}
common_lat = None
common_lon = None

for name, da in processed_vars.items():
    print(f"Aligning {name}...")
    if common_lat is None and 'lat' in da.coords:
        common_lat = da['lat']
    if common_lon is None and 'lon' in da.coords:
        common_lon = da['lon']

    try:
        coords_to_align = {'time': full_time_index}
        if 'lat' in da.coords: coords_to_align['lat'] = common_lat
        if 'lon' in da.coords: coords_to_align['lon'] = common_lon

        # Use method='nearest' with tolerance to align time, allowing some flexibility
        aligned_da = da.reindex(**coords_to_align, method='nearest', tolerance=pd.Timedelta('12h'), fill_value=np.nan)
        aligned_processed_vars[name] = aligned_da
        print(f"Aligned {name} shape: {aligned_da.shape}")
    except Exception as e:
        print(f"ERROR aligning {name}: {e}. Original shape: {da.shape}")
        print(f"Coords: {da.coords}")

print("\nSplitting into Train/Test and Saving...")
for name, data_var in aligned_processed_vars.items():
    print(f"Saving {name}...")
    data_train = data_var.sel(time=train_time_index)
    data_test = data_var.sel(time=test_time_index)

    if data_train.time.size == 0:
         print(f"Warning: No training data found for {name} in period {train_time_index[0]} to {train_time_index[-1]}")
    else:
         train_filename = os.path.join(data_save, 'train', f'{name}_train_{train_time_index[0].year}_{train_time_index[-1].year}.nc')
         data_train.to_netcdf(train_filename)
         print(f"Saved Train: {train_filename}")

    if data_test.time.size == 0:
         print(f"Warning: No testing data found for {name} in period {test_time_index[0]} to {test_time_index[-1]}")
    else:
         test_filename = os.path.join(data_save, 'test', f'{name}_test_{test_time_index[0].year}.nc')
         data_test.to_netcdf(test_filename)
         print(f"Saved Test: {test_filename}")

print("\n--- Preprocessing Complete ---")
print("\nProcessed variables summary:")
print("- Single levels: TCWV, T2M, MSLP, Pressure Tendency")
print("- Pressure levels (300, 500, 700, 850 hPa): U, V, Q")
print(f"\nTotal variables: {len(aligned_processed_vars)}")
print(f"\nData saved to: {data_save}")
import numpy as np
import xarray as xr
import pandas as pd
import os
from helpers import con_grid, regrid_bilinear, get_shear, get_pressure_tendency, get_stream, get_short_name

# Match data_dir with era5_download.py
data_dir = './precip_data/era5_data/'
# Adjust this path to save final predictor files
data_save = './precip_data/predictors/'
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

# Based on downloaded files from era5_download.py
variables_to_process = {
    # Single Levels
    'total_column_water_vapour': {'regrid': 'conservative'},
    'convective_available_potential_energy': {'regrid': 'bilinear'},
    'convective_inhibition': {'regrid': 'bilinear'},
    '2m_temperature': {'regrid': 'bilinear'},
    '2m_dewpoint_temperature': {'regrid': 'bilinear'},
    'surface_pressure': {'regrid': 'bilinear', 'process': 'pressure_tendency'},
    'mean_sea_level_pressure': {'regrid': 'bilinear'},
    # Pressure Levels
    'geopotential_500': {'regrid': 'bilinear', 'level': 500, 'var': 'z'},
    'temperature_850': {'regrid': 'bilinear', 'level': 850, 'var': 't'},
    'temperature_500': {'regrid': 'bilinear', 'level': 500, 'var': 't'},
    'specific_humidity_850': {'regrid': 'bilinear', 'level': 850, 'var': 'q'},
    'specific_humidity_700': {'regrid': 'bilinear', 'level': 700, 'var': 'q'},
    'u_component_of_wind_850': {'regrid': 'bilinear', 'level': 850, 'var': 'u'},
    'v_component_of_wind_850': {'regrid': 'bilinear', 'level': 850, 'var': 'v'},
    'u_component_of_wind_500': {'regrid': 'bilinear', 'level': 500, 'var': 'u'},
    'v_component_of_wind_500': {'regrid': 'bilinear', 'level': 500, 'var': 'v'},
    'vertical_velocity_700': {'regrid': 'bilinear', 'level': 700, 'var': 'w'},
    'vertical_velocity_500': {'regrid': 'bilinear', 'level': 500, 'var': 'w'},
    'relative_humidity_700': {'regrid': 'bilinear', 'level': 700, 'var': 'r'},
    'relative_humidity_500': {'regrid': 'bilinear', 'level': 500, 'var': 'r'},
    'divergence_850': {'regrid': 'bilinear', 'level': 850, 'var': 'd'},
    'divergence_700': {'regrid': 'bilinear', 'level': 700, 'var': 'd'},
    'divergence_500': {'regrid': 'bilinear', 'level': 500, 'var': 'd'},
    # Variables for Derived Calculations
    'vorticity_700': {'regrid': 'bilinear', 'level': 700, 'var': 'vo', 'process': 'stream'},
    'u_component_of_wind_925': {'regrid': 'bilinear', 'level': 925, 'var': 'u', 'process': 'shear'},
    'v_component_of_wind_925': {'regrid': 'bilinear', 'level': 925, 'var': 'v', 'process': 'shear'},
    'u_component_of_wind_600': {'regrid': 'bilinear', 'level': 600, 'var': 'u', 'process': 'shear'},
    'v_component_of_wind_600': {'regrid': 'bilinear', 'level': 600, 'var': 'v', 'process': 'shear'},
}

processed_data = {}

print("--- Starting Regridding ---")
regridded_files = {}

# Function to get input file path
def get_input_path(var_info, var_key):
    if 'level' in var_info:
        return os.path.join(data_dir, f"era5_pl_{var_info['level']}.nc")
    else:
        return os.path.join(data_dir, f"era5_{var_key}.nc")

for key, info in variables_to_process.items():
    input_file = get_input_path(info, key)
    level = info.get('level')
    short_name = get_short_name(key, level)
    output_file = os.path.join(regrid_dir, f"era5_{short_name}_regrid.nc")
    regridded_files[key] = output_file

    if not input_file or not os.path.exists(input_file):
        print(f"Warning: Input file not found, skipping regridding: {input_file}")
        continue

    if os.path.exists(output_file):
        print(f"Regridded file already exists, skipping: {output_file}")
        continue

    print(f"Regridding {key} from {os.path.basename(input_file)} to {output_file}...")
    try:
        if info['regrid'] == 'conservative':
            con_grid(input_file, output_file, target_grid_file)
        elif info['regrid'] == 'bilinear':
            regrid_bilinear(input_file, output_file, target_grid_file)
        else:
            print(f"Warning: Unknown regridding method '{info['regrid']}' for {key}")
    except Exception as e:
         print(f"ERROR during regridding for {key} from {input_file}: {e}")

print("--- Regridding Complete ---")


# --- Processing Step ---
print("\n--- Starting Processing ---")
processed_vars = {}

# Process individual variables
for key, info in variables_to_process.items():
    regrid_file_path = regridded_files.get(key)
    if not regrid_file_path or not os.path.exists(regrid_file_path):
        print(f"Warning: Regridded file not found for {key}, skipping processing.")
        continue

    # Skip components used only for derived variables
    if info.get('process') == 'shear' and key not in ['u_component_of_wind_925', 'v_component_of_wind_925','u_component_of_wind_600','v_component_of_wind_600']: continue
    if info.get('process') == 'stream' and key != 'vorticity_700': continue
    if info.get('process') == 'pressure_tendency' and key != 'surface_pressure': continue

    print(f"Processing {key}...")
    ds = xr.open_dataset(regrid_file_path)

    if 'valid_time' in ds.dims: ds = ds.rename({'valid_time': 'time'})
    if 'valid_time' in ds.coords: ds = ds.rename({'valid_time': 'time'})
    if 'longitude' in ds.dims: ds = ds.rename({'longitude': 'lon'})
    if 'longitude' in ds.coords: ds = ds.rename({'longitude': 'lon'})
    if 'latitude' in ds.dims: ds = ds.rename({'latitude': 'lat'})
    if 'latitude' in ds.coords: ds = ds.rename({'latitude': 'lat'})

    if 'level' in info:
        var_name_in_file = info['var']
    else:
         # Assumes single variable in single-level files
         var_name_in_file = list(ds.data_vars)[0]

    if var_name_in_file not in ds:
         print(f"Warning: Variable '{var_name_in_file}' not found in {regrid_file_path}, skipping.")
         ds.close()
         continue

    data_var = ds[var_name_in_file]

    # Squeeze dimension if only one level exists
    if 'level' in data_var.dims and data_var.dims['level']==1:
       data_var = data_var.squeeze('level', drop=True)
    if 'plev' in data_var.dims and data_var.dims['plev']==1:
       data_var = data_var.squeeze('plev', drop=True)

    # Handle potential fill values or NaNs (example: fill CIN NaNs with 0)
    if key == 'convective_inhibition':
        data_var = data_var.fillna(0)

    level = info.get('level')
    short_name = get_short_name(key, level)
    processed_vars[short_name] = data_var.sortby('time')
    ds.close()


print("Calculating derived variables...")
# Pressure Tendency
sp_regridded_file = regridded_files.get('surface_pressure')
if sp_regridded_file and os.path.exists(sp_regridded_file):
    ptend_da = get_pressure_tendency(sp_regridded_file)
    if ptend_da is not None:
        processed_vars['ptend'] = ptend_da.sortby('time')
else:
    print("Warning: Cannot calculate pressure tendency, regridded surface_pressure file missing.")

# Shear
u925_file = regridded_files.get('u_component_of_wind_925')
v925_file = regridded_files.get('v_component_of_wind_925')
u600_file = regridded_files.get('u_component_of_wind_600')
v600_file = regridded_files.get('v_component_of_wind_600')
if all(f and os.path.exists(f) for f in [u925_file, v925_file, u600_file, v600_file]):
    shear_da = get_shear(u925_file, v925_file, u600_file, v600_file)
    if shear_da is not None:
        processed_vars['shear925_600'] = shear_da.sortby('time')
else:
    print("Warning: Cannot calculate shear, required regridded wind files missing.")


# Stream Function
vo700_file = regridded_files.get('vorticity_700')
if vo700_file and os.path.exists(vo700_file):
    # Pass the dedicated temp_dir for CDO temporary files
    stream_da = get_stream(vo700_file, temp_dir, target_grid_file)
    if stream_da is not None:
        processed_vars['stream700'] = stream_da.sortby('time')
else:
     print("Warning: Cannot calculate stream function, regridded vorticity_700 file missing.")


print("\n--- Aligning Timesteps and Saving ---")

# Align all processed variables to a common time index.
# This might be memory intensive. Alternative: align to train/test indices directly.
# Let's align to the full 2007-2020 daily index first.
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


print("Splitting into Train/Test and Saving...")
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

print("--- Preprocessing Complete ---")
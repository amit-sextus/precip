import numpy as np
import xarray as xr
import pandas as pd
from cdo import Cdo
import os
import subprocess

# Initialize CDO
cdo = Cdo()

def regrid_bilinear(input_file, output_file, target_grid_file):
    """Regrids using bilinear interpolation."""
    try:
        cdo.remapbil(target_grid_file, input=input_file, output=output_file, options='-P 8') # Use 8 parallel processes if helpful
        print(f"Bilinear regridding successful: {output_file}")
    except Exception as e:
        print(f"ERROR during bilinear regridding for {input_file}: {e}")


def con_grid(input_file, output_file, target_grid_file):
    """Regrids using conservative remapping."""
    try:
        cdo.remapcon(target_grid_file, input=input_file, output=output_file, options='-P 8')
        print(f"Conservative regridding successful: {output_file}")
    except Exception as e:
        print(f"ERROR during conservative regridding for {input_file}: {e}")

def get_shear(u925_regridded_file, v925_regridded_file, u600_regridded_file, v600_regridded_file):
    """Calculates wind shear between 925 hPa and 600 hPa from regridded files."""
    try:
        u925_ds = xr.open_dataset(u925_regridded_file)
        if 'valid_time' in u925_ds.dims: u925_ds = u925_ds.rename({'valid_time': 'time'})
        if 'valid_time' in u925_ds.coords: u925_ds = u925_ds.rename({'valid_time': 'time'})
        v925_ds = xr.open_dataset(v925_regridded_file)
        if 'valid_time' in v925_ds.dims: v925_ds = v925_ds.rename({'valid_time': 'time'})
        if 'valid_time' in v925_ds.coords: v925_ds = v925_ds.rename({'valid_time': 'time'})
        u600_ds = xr.open_dataset(u600_regridded_file)
        if 'valid_time' in u600_ds.dims: u600_ds = u600_ds.rename({'valid_time': 'time'})
        if 'valid_time' in u600_ds.coords: u600_ds = u600_ds.rename({'valid_time': 'time'})
        v600_ds = xr.open_dataset(v600_regridded_file)
        if 'valid_time' in v600_ds.dims: v600_ds = v600_ds.rename({'valid_time': 'time'})
        if 'valid_time' in v600_ds.coords: v600_ds = v600_ds.rename({'valid_time': 'time'})

        # Select the variable (ensure names match ERA5 download: 'u', 'v')
        u925 = u925_ds['u'].squeeze()
        v925 = v925_ds['v'].squeeze()
        u600 = u600_ds['u'].squeeze()
        v600 = v600_ds['v'].squeeze()

        # Ensure time coordinates match before calculation
        u600, v600 = xr.align(u600, v600, join='inner')
        u925, v925 = xr.align(u925, v925, join='inner')
        u925, u600 = xr.align(u925, u600, join='inner')
        v925, v600 = xr.align(v925, v600, join='inner')

        z_diff_hpa = 925 - 600 # Difference in hPa

        # Approximate height difference (using standard atmosphere, very rough estimate)
        # A better approach might use geopotential height difference if available
        # Standard atmosphere approx: dz/dp = -RT/gp => dz ~ -(RT/gp) * dp
        # Assume T ~ 273K, R=287, g=9.81, p_mid ~ 762 hPa = 76200 Pa
        # dz ~ -(287*273 / (9.81*76200)) * (60000-92500) Pa ~ 3400 m
        # For simplicity, using pressure difference directly might be acceptable
        # if used as a relative index, but it's not physically shear (unit: m/s / m).
        # Let's stick to pressure difference denominator for consistency with original logic
   
        z = z_diff_hpa

        diff_u = u925 - u600
        diff_v = v925 - v600
        diff_u2 = diff_u ** 2
        diff_v2 = diff_v ** 2
        diff_mag = (diff_u2 + diff_v2)**(0.5)
        shear = diff_mag / z # Units: m/s / hPa

        # Create a new DataArray for shear
        shear_da = xr.DataArray(
            data=shear.values,
            dims=u925.dims,
            coords=u925.coords,
            name="shear925_600"
        )
        print("Shear calculation successful.")
        return shear_da
    except Exception as e:
        print(f"ERROR calculating shear: {e}")
        return None


def get_pressure_tendency(sp_regridded_file):
    """Calculates pressure tendency from regridded surface pressure file."""
    try:
        data = xr.open_dataset(sp_regridded_file)
        if 'valid_time' in data.dims:
            data = data.rename({'valid_time': 'time'})
        if 'valid_time' in data.coords:
            data = data.rename({'valid_time': 'time'})

        sp_var_name = list(data.data_vars)[0]
        sp_data = data[sp_var_name]

        sp_data = sp_data.sortby('time')
        pressure_tendency_da = sp_data.diff("time", label="upper")
        # The difference is assigned to the later timestamp. First time step is lost.
        pressure_tendency_da = pressure_tendency_da.rename('pressure_tendency')

        # Keep as Pa/24h assuming daily (00Z) input
        print("Pressure tendency calculation successful.")
        return pressure_tendency_da
    except Exception as e:
        print(f"ERROR calculating pressure tendency: {e}")
        return None


def get_stream(vo700_regridded_file, temp_dir_base, target_grid_file):
    """Calculates stream function using CDO from regridded vorticity 700hPa file.

    Args:
        vo700_regridded_file: Path to the regridded vorticity file.
        temp_dir_base: Path to the base directory where temporary files will be created.
        target_grid_file: Path to the target grid definition file.
    """
    print("Calculating stream function...")
    original_cwd = os.path.abspath(os.getcwd())

    vo_in_abs = os.path.abspath(vo700_regridded_file)
    target_grid_file_abs = os.path.abspath(target_grid_file)
    temp_dir_abs = os.path.abspath(temp_dir_base)

    if not os.path.exists(vo_in_abs):
         print(f"ERROR: Input vorticity file not found at {vo_in_abs}")
         return None
    if not os.path.exists(target_grid_file_abs):
         print(f"ERROR: Target grid file not found at {target_grid_file_abs}")
         return None
    if not os.path.exists(temp_dir_abs):
        print(f"ERROR: Base temporary directory not found at {temp_dir_abs}")
        return None
    if not os.path.isdir(temp_dir_abs):
        print(f"ERROR: Provided temporary path is not a directory: {temp_dir_abs}")
        return None

    # Define intermediate filenames (these will be created within the temp_dir_abs)
    vo_svo = os.path.join(temp_dir_abs, "era5_vorticity_700_svo.nc")
    zero_div = os.path.join(temp_dir_abs, "era5_zerodiv.nc")
    svosd = os.path.join(temp_dir_abs, "era5_svosd.nc")
    stream_out = os.path.join(temp_dir_abs, "era5_stream_final.nc")
    t511grid = "t511grid" # Intermediate grid for CDO processing steps

    vo_filled_temp_file = os.path.join(temp_dir_abs, "era5_vorticity_700_filled.nc")
    try:
        with xr.open_dataset(vo_in_abs) as ds_vo:
            vo_var_name = list(ds_vo.data_vars)[0]
            ds_vo[vo_var_name] = ds_vo[vo_var_name].fillna(0.0) # Fill NaNs with 0
            ds_vo.to_netcdf(vo_filled_temp_file)
        print(f"Filled NaNs in vorticity, saved to {vo_filled_temp_file}")
        input_for_cdo = vo_filled_temp_file
    except Exception as e:
        print(f"ERROR filling NaNs in vorticity file {vo_in_abs}: {e}")
        os.chdir(original_cwd)
        return None

    # Define the CDO commands using absolute paths
    commands = [
        f"cdo -b 32 setname,svo {input_for_cdo} {vo_svo}",
        f"cdo -L -b 32 chname,svo,sd -mulc,0 {vo_svo} {zero_div}",
        f"cdo -merge {vo_svo} {zero_div} {svosd}",
        f"cdo -L -b 32 remapbil,{target_grid_file_abs} -selvar,stream -sp2gp -dv2ps -gp2sp -remapbil,{t511grid} {svosd} {stream_out}"
    ]

    success = True
    stream_da = None
    try:
        for cmd in commands:
            print(f"Running command: {cmd}")
            subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print("Command completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {cmd}")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
        success = False
    except FileNotFoundError as e:
        print(f"Error: CDO command not found. Is CDO installed and in your PATH? Command: {cmd}")
        print(e)
        success = False
    finally:
        os.chdir(original_cwd)

        result_path = stream_out

        if success and os.path.exists(result_path):
            print("Stream function calculation successful.")
            try:
                with xr.open_dataset(result_path) as data:
                    if 'valid_time' in data.dims:
                        data = data.rename({'valid_time': 'time'})
                    if 'valid_time' in data.coords:
                        data = data.rename({'valid_time': 'time'})

                    if 'stream' in data:
                         stream_da_temp = data['stream']
                         if 'plev' in stream_da_temp.coords:
                             stream_da_temp = stream_da_temp.drop_vars('plev', errors='ignore')
                         if 'level' in stream_da_temp.coords:
                             stream_da_temp = stream_da_temp.drop_vars('level', errors='ignore')
                         stream_da = stream_da_temp.rename('stream700')
                    else:
                        print(f"ERROR: 'stream' variable not found in output file: {result_path}")
                        success = False
            except Exception as e:
                print(f"Error opening or processing stream function result file {result_path}: {e}")
                success = False
        else:
             if not success:
                 pass # Error already printed
             else:
                 print(f"Stream function calculation failed: Output file not found at {result_path}")
                 success = False

        print(f"Cleaning up temporary files in: {temp_dir_abs}")
        for f in [vo_filled_temp_file, vo_svo, zero_div, svosd, stream_out]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError as e:
                    print(f"Warning: Could not remove temp file {f}: {e}")
        if os.path.exists(temp_dir_abs):
            try:
                os.rmdir(temp_dir_abs)
            except OSError as e:
                print(f"Warning: Could not remove temp directory {temp_dir_abs}: {e}. Check for leftover files.")

    return stream_da


# --- Variable Name Mapping ---
def get_short_name(variable_key, level=None):
    """Maps ERA5 variable names/keys (potentially including level) and levels to shorter names for filenames/use."""
    mapping = {
        'total_column_water_vapour': 'tcwv',
        'convective_available_potential_energy': 'cape',
        'convective_inhibition': 'cin',
        '2m_temperature': 't2m',
        '2m_dewpoint_temperature': 'd2m',
        'surface_pressure': 'sp',
        'mean_sea_level_pressure': 'msl',
        # Pressure level base names
        'geopotential': 'z',
        'temperature': 't',
        'specific_humidity': 'q',
        'relative_humidity': 'r',
        'u_component_of_wind': 'u',
        'v_component_of_wind': 'v',
        'vertical_velocity': 'w',
        'vorticity': 'vo',
        'divergence': 'd',
        # Derived variables from this script (already unique)
        # These should ideally be handled before calling this function if passed as keys,
        # but we include them here for completeness or if the function is called directly.
        'shear925_600': 'shear925_600',
        'pressure_tendency': 'ptend',
        'stream700': 'stream700',
    }
    pressure_level_base_vars = ['z', 't', 'q', 'r', 'u', 'v', 'w', 'vo', 'd']

    base_era5_name = variable_key
    extracted_level = level
    parts = variable_key.split('_')
    # Heuristic check if the last part looks like a level number
    if len(parts) > 1 and parts[-1].isdigit() and parts[-2] not in ['of', 'column', 'available', 'level', 'm']:
        base_era5_name = '_'.join(parts[:-1])
        if extracted_level is None:
            extracted_level = int(parts[-1])

    base_short_name = mapping.get(base_era5_name)

    if base_short_name:
        if extracted_level is not None and base_short_name in pressure_level_base_vars:
            return f"{base_short_name}{extracted_level}"
        else:
            return base_short_name
    else:
        # Fallback for unmapped or already-short names
        if variable_key in ['shear925_600', 'pressure_tendency', 'stream700']:
             return variable_key
        print(f"Warning: No short name mapping found for base name '{base_era5_name}' derived from key '{variable_key}' (Level: {extracted_level}). Using original key format.")
        return f"{variable_key}_{level}" if level else variable_key
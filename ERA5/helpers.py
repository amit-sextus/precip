import numpy as np
import xarray as xr
import pandas as pd
import os

# Initialize CDO with explicit path to avoid initialization issues
try:
    from cdo import Cdo
    # Try to initialize CDO with explicit path to the system executable
    cdo = Cdo(cdo='/usr/bin/cdo')
    print("CDO initialized successfully with explicit path.")
except Exception as e:
    print(f"Error initializing CDO: {e}")
    print("Falling back to xarray-based regridding methods.")
    cdo = None

def regrid_bilinear(input_file, output_file, target_grid_file):
    """Regrids using bilinear interpolation."""
    # Add file existence check
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found for regridding: {input_file}")
    
    # Try opening with xarray to catch HDF errors early
    try:
        test_ds = xr.open_dataset(input_file)
        test_ds.close()
    except Exception as e:
        raise RuntimeError(f"Cannot open {input_file} as NetCDF: {e}")
    
    # Check if this is a pressure level file
    is_pressure_level = 'pl_' in os.path.basename(input_file)
    
    if cdo is None or is_pressure_level:
        if is_pressure_level:
            print("Using xarray-based regridding for pressure level file")
        else:
            print("CDO not available, using xarray-based regridding")
        _regrid_xarray(input_file, output_file, target_grid_file, method='bilinear')
        return
        
    try:
        cdo.remapbil(target_grid_file, input=input_file, output=output_file, options='-P 8')
        print(f"Bilinear regridding successful: {output_file}")
    except Exception as e:
        print(f"ERROR during bilinear regridding for {input_file}: {e}")
        print("Falling back to xarray-based regridding")
        _regrid_xarray(input_file, output_file, target_grid_file, method='bilinear')


def con_grid(input_file, output_file, target_grid_file):
    """Regrids using conservative remapping."""
    # Add file existence check
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found for regridding: {input_file}")
    
    # Try opening with xarray to catch HDF errors early
    try:
        test_ds = xr.open_dataset(input_file)
        test_ds.close()
    except Exception as e:
        raise RuntimeError(f"Cannot open {input_file} as NetCDF: {e}")
    
    # Check if this is a pressure level file
    is_pressure_level = 'pl_' in os.path.basename(input_file)
    
    if cdo is None or is_pressure_level:
        if is_pressure_level:
            print("Using xarray-based regridding for pressure level file")
        else:
            print("CDO not available, using xarray-based regridding")
        _regrid_xarray(input_file, output_file, target_grid_file, method='conservative')
        return
        
    try:
        cdo.remapcon(target_grid_file, input=input_file, output=output_file, options='-P 8')
        print(f"Conservative regridding successful: {output_file}")
    except Exception as e:
        print(f"ERROR during conservative regridding for {input_file}: {e}")
        print("Falling back to xarray-based regridding")
        _regrid_xarray(input_file, output_file, target_grid_file, method='conservative')


def _regrid_xarray(input_file, output_file, target_grid_file, method='bilinear'):
    """Fallback regridding using xarray and scipy interpolation."""
    try:
        import scipy.interpolate
        
        # Read input data
        ds = xr.open_dataset(input_file)
        
        # Read target grid
        target_grid = _read_target_grid(target_grid_file)
        
        # Perform interpolation for each variable
        regridded_vars = {}
        for var_name in ds.data_vars:
            var = ds[var_name]
            
            # Get coordinates
            if 'longitude' in var.dims:
                lon_dim = 'longitude'
            elif 'lon' in var.dims:
                lon_dim = 'lon'
            else:
                raise ValueError(f"No longitude coordinate found in {var_name}")
                
            if 'latitude' in var.dims:
                lat_dim = 'latitude'
            elif 'lat' in var.dims:
                lat_dim = 'lat'
            else:
                raise ValueError(f"No latitude coordinate found in {var_name}")
            
            # Interpolate variable
            regridded_var = var.interp(
                {lon_dim: target_grid['lon'], lat_dim: target_grid['lat']},
                method='linear' if method == 'bilinear' else 'nearest'
            )
            
            # Rename coordinates to standard names
            if lon_dim != 'lon':
                regridded_var = regridded_var.rename({lon_dim: 'lon'})
            if lat_dim != 'lat':
                regridded_var = regridded_var.rename({lat_dim: 'lat'})
                
            regridded_vars[var_name] = regridded_var
        
        # Create new dataset
        regridded_ds = xr.Dataset(regridded_vars)
        
        # Copy attributes
        regridded_ds.attrs = ds.attrs
        for var_name in regridded_vars:
            regridded_ds[var_name].attrs = ds[var_name].attrs
        
        # Save to file
        regridded_ds.to_netcdf(output_file)
        print(f"Xarray regridding successful: {output_file}")
        
        # Close datasets
        ds.close()
        regridded_ds.close()
        
    except Exception as e:
        print(f"ERROR during xarray regridding for {input_file}: {e}")
        raise


def _read_target_grid(target_grid_file):
    """Read target grid file and return lon/lat arrays."""
    try:
        # Try to read as a simple text file with grid description
        with open(target_grid_file, 'r') as f:
            lines = f.readlines()
        
        # Parse grid description (assuming CDO grid format)
        grid_info = {}
        for line in lines:
            line = line.strip()
            if line.startswith('xsize'):
                grid_info['nlon'] = int(line.split('=')[1])
            elif line.startswith('ysize'):
                grid_info['nlat'] = int(line.split('=')[1])
            elif line.startswith('xfirst'):
                grid_info['lon_first'] = float(line.split('=')[1])
            elif line.startswith('yfirst'):
                grid_info['lat_first'] = float(line.split('=')[1])
            elif line.startswith('xinc'):
                grid_info['lon_inc'] = float(line.split('=')[1])
            elif line.startswith('yinc'):
                grid_info['lat_inc'] = float(line.split('=')[1])
        
        # Generate coordinate arrays
        lons = np.linspace(
            grid_info['lon_first'],
            grid_info['lon_first'] + (grid_info['nlon'] - 1) * grid_info['lon_inc'],
            grid_info['nlon']
        )
        lats = np.linspace(
            grid_info['lat_first'],
            grid_info['lat_first'] + (grid_info['nlat'] - 1) * grid_info['lat_inc'],
            grid_info['nlat']
        )
        
        return {'lon': lons, 'lat': lats}
        
    except Exception as e:
        print(f"Error reading target grid file {target_grid_file}: {e}")
        # Fallback to a default Germany grid
        print("Using default Germany grid")
        lons = np.arange(5.0, 16.0, 0.25)  # 5°E to 15°E, 0.25° resolution
        lats = np.arange(47.0, 56.0, 0.25)  # 47°N to 55°N, 0.25° resolution
        return {'lon': lons, 'lat': lats}


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

        # Keep as Pa/24h assuming daily (18:00 UTC) input
        print("Pressure tendency calculation successful.")
        return pressure_tendency_da
    except Exception as e:
        print(f"ERROR calculating pressure tendency: {e}")
        return None


def handle_expver(ds):
    """Handles the 'expver' dimension in ERA5 data by prioritizing ERA5 over ERA5T."""
    if 'expver' in ds.dims:
        if ds.expver.size > 1:
            # Prioritize ERA5 (expver=1) over ERA5T (expver=5)
            ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
        else:
            ds = ds.squeeze('expver', drop=True)
    return ds


# --- Variable Name Mapping ---
def get_short_name(variable_key, level=None):
    """Maps ERA5 variable names/keys to shorter names for filenames/use."""
    mapping = {
        'total_column_water_vapour': 'tcwv',
        '2m_temperature': 't2m',
        'surface_pressure': 'sp',
        'mean_sea_level_pressure': 'msl',
        # Pressure level base names
        'specific_humidity': 'q',
        'u_component_of_wind': 'u',
        'v_component_of_wind': 'v',
        # Derived variables
        'pressure_tendency': 'ptend',
    }
    pressure_level_base_vars = ['q', 'u', 'v']

    base_era5_name = variable_key
    extracted_level = level
    parts = variable_key.split('_')
    # Heuristic check if the last part looks like a level number
    if len(parts) > 1 and parts[-1].isdigit() and parts[-2] not in ['of', 'column', 'level', 'm']:
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
        # Fallback for unmapped names
        if variable_key == 'pressure_tendency':
             return variable_key
        print(f"Warning: No short name mapping found for '{base_era5_name}' (Level: {extracted_level}). Using original key.")
        return f"{variable_key}_{level}" if level else variable_key
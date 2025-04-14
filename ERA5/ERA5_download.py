# ERA5 Download Script Modified Based on Research

# Add your API key to the ~/.cdsapirc file on your computer in the following format:
# url: https://cds.climate.copernicus.eu/api/v2
# key: <your_uid>:<your_api_key>

import cdsapi
import os

# Configuration
data_dir = './precip_data/era5_data/'
os.makedirs(data_dir, exist_ok=True)

# Time Period
years = [str(y) for y in range(2007, 2021)]
months = [f"{m:02d}" for m in range(1, 13)]
days = [f"{d:02d}" for d in range(1, 32)]
# Download data for 18:00 UTC daily
times = ['18:00'] # Kept the 18:00 change

# Target Area [North, West, South, East] covering Germany and surroundings for 1-deg grid
area = [70, -70, 30, 50]

# Variables identified as relevant + dependencies for helpers.py
single_level_vars = [
    'total_column_water_vapour',                # tcwv
    'convective_available_potential_energy',    # cape
    'convective_inhibition',                    # cin
    '2m_temperature',                           # t2m
    '2m_dewpoint_temperature',                 # d2m
    'surface_pressure',                         # sp (for pressure_tendency)
    'mean_sea_level_pressure',                  # msl
]
# single_level_vars = ['2m_temperature'] # Test with single variable

# Group pressure level variables for efficient download
pressure_level_requests = {
    '500': ['geopotential', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity', 'relative_humidity', 'divergence'],
    '700': ['specific_humidity', 'vertical_velocity', 'relative_humidity', 'vorticity', 'divergence'], # vorticity for stream function
    '850': ['temperature', 'specific_humidity', 'u_component_of_wind', 'v_component_of_wind', 'divergence'],
    '600': ['u_component_of_wind', 'v_component_of_wind'], # For shear calc
    '925': ['u_component_of_wind', 'v_component_of_wind'], # For shear calc
}

# --- CDS API Client ---
c = cdsapi.Client()

# --- Download Single Level Variables ---
print("--- Downloading Single Level Variables ---")
for variable in single_level_vars:
    output_file = os.path.join(data_dir, f'era5_{variable}.nc')
    print(f"Requesting: {variable}...")
    try:
        c.retrieve(
            'reanalysis-era5-single-levels', # Original dataset
            {
                'product_type': 'reanalysis',
                'variable': variable,
                'year': years,
                'month': months,
                'day': days,
                'time': times,
                'area': area,
                'grid': '1.0/1.0', # Original grid
                'format': 'netcdf', # Original format
            },
            output_file
        )
        print(f"Completed: {variable}")
    except Exception as e:
        print(f"ERROR downloading {variable}: {e}")
print("--- Single Level Download Complete ---")

# --- Download Pressure Level Variables ---
print("\n--- Downloading Pressure Level Variables ---")
for level, variables in pressure_level_requests.items():
    output_file = os.path.join(data_dir, f'era5_pl_{level}.nc')
    print(f"Requesting: Level {level} Vars: {', '.join(variables)}...")
    try:
        c.retrieve(
            'reanalysis-era5-pressure-levels', # Original dataset
            {
                'product_type': 'reanalysis',
                'variable': variables,
                'pressure_level': level,
                'year': years,
                'month': months,
                'day': days,
                'time': times,
                'area': area,
                'grid': '1.0/1.0', # Original grid
                'format': 'netcdf', # Original format
            },
            output_file
        )
        print(f"Completed: Level {level}")
    except Exception as e:
        print(f"ERROR downloading level {level} variables: {e}")
print("--- Pressure Level Download Complete ---")
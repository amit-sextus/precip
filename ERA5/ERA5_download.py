# ERA5 Download Script Modified Based on Research

# Add your API key to the ~/.cdsapirc file on your computer in the following format:
# url: https://cds.climate.copernicus.eu/api/v2
# key: <your_uid>:<your_api_key>

import cdsapi
import os
import time

# Configuration - NEW DIRECTORY to avoid mixing with existing data
data_dir = './precip_data/era5_precipitation_project/'
os.makedirs(data_dir, exist_ok=True)

# Time Period
years = [str(y) for y in range(2007, 2021)]
months = [f"{m:02d}" for m in range(1, 13)]
days = [f"{d:02d}" for d in range(1, 32)]
# Download data for 18:00 UTC daily (6 hours before precipitation at 00:00 UTC)
times = ['18:00']

# Target Area [North, West, South, East] covering Germany and surroundings for 1-deg grid
area = [70, -70, 30, 50]

# Variables for supporting precipitation prediction
single_level_vars = [
    'total_column_water_vapour',    # tcwv
    '2m_temperature',                # t2m
    'surface_pressure',              # sp (for pressure_tendency calculation)
    'mean_sea_level_pressure',       # msl
]

# Pressure level variables - u/v winds and specific humidity at 4 levels
pressure_levels = ['300', '500', '700', '850']
pressure_level_vars = ['u_component_of_wind', 'v_component_of_wind', 'specific_humidity']

# --- CDS API Client ---
c = cdsapi.Client()

# --- Download Single Level Variables ---
print("--- Downloading Single Level Variables ---")
for variable in single_level_vars:
    output_file = os.path.join(data_dir, f'era5_{variable}.nc')
    
    # Skip if file already exists
    if os.path.exists(output_file):
        print(f"File already exists, skipping: {output_file}")
        continue
        
    print(f"Requesting: {variable}...")
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': variable,
                'year': years,
                'month': months,
                'day': days,
                'time': times,
                'area': area,
                'grid': '1.0/1.0',
                'format': 'netcdf',
            },
            output_file
        )
        print(f"Completed: {variable}")
    except Exception as e:
        print(f"ERROR downloading {variable}: {e}")
        # Add a small delay before next request
        time.sleep(5)
print("--- Single Level Download Complete ---")

# --- Download Pressure Level Variables ---
print("\n--- Downloading Pressure Level Variables ---")
# Split downloads by year to avoid exceeding API cost limits
for level in pressure_levels:
    for year in years:
        output_file = os.path.join(data_dir, f'era5_pl_{level}_{year}.nc')
        
        # Skip if file already exists
        if os.path.exists(output_file):
            print(f"File already exists, skipping: {output_file}")
            continue
            
        print(f"Requesting: Level {level}, Year {year} - Variables: {', '.join(pressure_level_vars)}...")
        try:
            c.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': pressure_level_vars,
                    'pressure_level': level,
                    'year': year,
                    'month': months,
                    'day': days,
                    'time': times,
                    'area': area,
                    'grid': '1.0/1.0',
                    'format': 'netcdf',
                },
                output_file
            )
            print(f"Completed: Level {level}, Year {year}")
            # Add delay between requests to avoid rate limiting
            time.sleep(10)
        except Exception as e:
            print(f"ERROR downloading level {level} year {year}: {e}")
            # Add longer delay on error
            time.sleep(30)
            
print("--- Pressure Level Download Complete ---")

print("\nAll downloads complete!")
print(f"Data saved to: {data_dir}")
print("\nDownloaded variables summary:")
print("- Single levels: TCWV, T2M, SP (for pressure tendency), MSLP")
print("- Pressure levels (300, 500, 700, 850 hPa): U, V, Q")
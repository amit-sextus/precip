import os
import glob

# Configuration from updated scripts
data_dir = './precip_data/era5_precipitation_project/'
predictors_dir = './precip_data/era5_precipitation_project/predictors/'
target_grid_file = 'ERA5/targetgrid_germany.txt'

print("ERA5 Precipitation Project Setup Verification")
print("=" * 60)

# Check directories
print("\n1. Directory Structure:")
print(f"   Raw data directory: {data_dir}")
print(f"   Exists: {os.path.exists(data_dir)}")
print(f"   Predictors directory: {predictors_dir}")
print(f"   Exists: {os.path.exists(predictors_dir)}")

# Check target grid file
print(f"\n2. Target Grid File:")
print(f"   File: {target_grid_file}")
print(f"   Exists: {os.path.exists(target_grid_file)}")
if not os.path.exists(target_grid_file):
    print("   WARNING: Target grid file not found! Please ensure it exists before running preprocessing.")

# Check for existing data
print("\n3. Existing Data Files:")
if os.path.exists(data_dir):
    existing_files = glob.glob(os.path.join(data_dir, "*.nc"))
    if existing_files:
        print(f"   Found {len(existing_files)} NetCDF files:")
        for f in sorted(existing_files)[:10]:  # Show first 10
            print(f"     - {os.path.basename(f)}")
        if len(existing_files) > 10:
            print(f"     ... and {len(existing_files) - 10} more files")
    else:
        print("   No existing data files found.")

# Download configuration summary
print("\n4. Download Configuration:")
print("   Time Period: 2007-2021 (14 years)")
print("   Time: 18:00 UTC daily (6 hours before precipitation)")
print("   Area: [70°N, -70°W, 30°S, 50°E] (Germany + surroundings)")
print("   Grid: 1.0° x 1.0°")

print("\n5. Variables to Download:")
print("   Single Levels:")
print("     - total_column_water_vapour (TCWV)")
print("     - 2m_temperature (T2M)")
print("     - surface_pressure (SP) -> for pressure tendency")
print("     - mean_sea_level_pressure (MSLP)")
print("\n   Pressure Levels (300, 500, 700, 850 hPa):")
print("     - u_component_of_wind (U)")
print("     - v_component_of_wind (V)")
print("     - specific_humidity (Q)")

print("\n6. Download Strategy:")
print("   - Single level variables: Downloaded as complete time series")
print("   - Pressure level variables: Downloaded year-by-year to avoid API limits")
print("   - Total expected files after download:")
print("     * 4 single level files")
print("     * 56 yearly pressure level files (4 levels × 14 years)")
print("     * 4 merged pressure level files (after merging)")

print("\n7. Processing Pipeline:")
print("   1. Download raw ERA5 data to:", data_dir)
print("   2. Regrid data to target grid")
print("   3. Calculate pressure tendency from surface pressure")
print("   4. Align all variables to common time/space grid")
print("   5. Split into train (2007-2019) and test (2020) sets")
print("   6. Save processed predictors to:", predictors_dir)

print("\n8. Expected Output:")
print("   Total predictor variables: 16")
print("   - 4 single level variables (TCWV, T2M, MSLP, Pressure Tendency)")
print("   - 12 pressure level variables (U, V, Q at 4 levels each)")

print("\nNotes:")
print("- Downloads may take several hours due to data volume and API rate limits")
print("- Ensure you have sufficient disk space (~50-100 GB)")
print("- The download script includes automatic retries and delays")
print("- Existing files will be skipped to allow resuming interrupted downloads")
print("\n" + "=" * 60) 
#!/bin/bash

# Activate the dedicated conda environment for this task
eval "$(conda shell.bash hook)"
conda activate hyras_process

# Directory where HYRAS raw files are stored
INDIR="data/HYRAS_raw"

# Directory where regridded HYRAS files will be written
OUTDIR="data/HYRAS_regridded"

# Path to the target grid definition file
GRIDFILE="pre_process/target_grid.txt"

# Path to the source grid definition file
SOURCE_GRID="pre_process/hyras_laea_grid.txt"

# Make sure the output directory exists
mkdir -p "${OUTDIR}"

# Function to process a single year
process_year() {
    year=$1
    echo "Processing year ${year}..."
    
    # Construct input filename
    file_in="${INDIR}/pr_hyras_1_${year}_v6-0_de.nc"
    
    # Check if the file exists; if not, skip
    if [ ! -f "${file_in}" ]; then
        echo "WARNING: File not found: ${file_in} ... skipping."
        return
    fi
    
    echo "Processing ${file_in} ..."
    
    # Final output filename
    outfile="${OUTDIR}/hyras_regridded_${year}.nc"
    
    # Temporary file for debugging
    temp_with_grid="temp_hyras_with_grid_${year}.nc"
    
    # Apply the source grid definition and then remap
    echo "Applying source grid definition and performing conservative remapping..."
    cdo remapcon,${GRIDFILE} -setgrid,${SOURCE_GRID} ${file_in} ${outfile}
    
    if [ $? -ne 0 ]; then
        echo "ERROR: CDO remapcon with setgrid failed!"
        echo "Saving the intermediate file with grid definition for debugging..."
        cdo -setgrid,${SOURCE_GRID} ${file_in} ${temp_with_grid}
        
        echo "Falling back to bilinear interpolation (remapbil)..."
        cdo remapbil,${GRIDFILE} -setgrid,${SOURCE_GRID} ${file_in} ${outfile}
        
        if [ $? -ne 0 ]; then
            echo "ERROR: CDO remapbil also failed!"
            echo "Please check the grid definition file."
            return
        fi
        echo "WARNING: Used less accurate bilinear interpolation instead of conservative remapping."
    fi
    
    # Clean up temporary files
    rm -f ${temp_with_grid} 2>/dev/null
    
    echo "Finished writing: ${outfile}"
    echo "-----------------------------"
}

# Process all years
for year in {2007..2020}; do
    process_year ${year}
done

echo "All done!" 
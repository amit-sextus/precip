#!/bin/bash

# Directory where your monthly MSWEP raw files are stored:
INDIR="/mnt/z/data/MSWEP_raw"

# Directory where final daily 1-degree files will be written:
OUTDIR="/mnt/z/data/MSWEP_daily"

# Path to the target grid definition file:
GRIDFILE="target_grid.txt"

# Make sure the output directory exists
mkdir -p "${OUTDIR}"

# Loop over each year and month
for year in {2007..2020}; do
  for month in {01..12}; do
    
    # Construct input filename
    file_in="${INDIR}/amit_mswep_v2.8_3hourly_0.1deg_${year}-${month}.nc"
    
    # Check if the file exists; if not, skip
    if [ ! -f "${file_in}" ]; then
      echo "WARNING: File not found: ${file_in} ... skipping."
      continue
    fi

    echo "Processing ${file_in} ..."

    # 1) Regrid to 1-degree resolution using CDO
    #    You can use 'remapbil' (bilinear) or 'remapcon' (conservative).
    #    For precipitation, conservative might be more physically meaningful, but
    #    bilinear is often simpler. Change 'remapbil' to 'remapcon' if desired.
    cdo remapcon,"${GRIDFILE}" "${file_in}" tmp_regrid.nc

    # 2) Sum 3-hour intervals to daily totals (00:00 to 00:00).
    cdo daysum tmp_regrid.nc tmp_daily.nc

    # 3) Rename (or move) final output to mswep_daily_YYYYMM.nc
    outfile="${OUTDIR}/mswep_daily_${year}${month}.nc"
    mv tmp_daily.nc "${outfile}"

    # 4) Clean up intermediate file
    rm -f tmp_regrid.nc

    echo "Finished writing: ${outfile}"
    echo "-----------------------------"
    
  done
done

echo "All done!"

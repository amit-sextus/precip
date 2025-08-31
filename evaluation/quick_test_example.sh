#!/bin/bash
# Quick test example - runs minimal experiments for testing the pipeline
# This uses reduced epochs/folds and only a subset of configurations

echo "Running quick test of experiment pipeline..."
echo "This will run reduced experiments to verify the system works"
echo ""

# --- Configuration ---
set -e  # Exit on error

# Activate environment (commented out - already activated in terminal)
# conda activate precip_2.0

# Base directories (same as main script)
BASE_DATA_DIR="data/MSWEP_daily"
ERA5_DATA_DIR="precip_data/era5_precipitation_project/predictors"
BASE_OUTPUT_DIR="./test_experiment_runs"
EVAL_DIR="evaluation"
MANIFEST_FILE="${EVAL_DIR}/test_experiment_manifest.yaml"

# Training parameters - REDUCED FOR TESTING
EPOCHS=5  # Reduced from 20
FOLDS=2   # Reduced from 10
BATCH_SIZE=32
OPTIMIZER="adamw"
LR=0.00001  # Same as main script
LR_SCHEDULER="cosineannealinglr"
LOSS_TYPE="mae"
DROPOUT=0.2
WEIGHT_DECAY=1e-4

# Test experiment groups - reduced set
declare -a GROUP_A_WEIGHTS=(0.2 0.5)  # Only 2 weights instead of 3
declare -a GROUP_B_ERA5=("none" "wind_only")  # Skip "all" for speed

# Create output directories
mkdir -p "${BASE_OUTPUT_DIR}"
mkdir -p "${EVAL_DIR}/test_figures"
mkdir -p "${EVAL_DIR}/test_tables"

# Initialize manifest file
echo "Creating test manifest file: ${MANIFEST_FILE}"
cat > "${MANIFEST_FILE}" << EOF
# Test manifest for quick verification
# Generated on: $(date)

runs:
EOF

# Function to run a single experiment (same as main script)
run_experiment() {
    local run_id=$1
    local outside_weight=$2
    local era5_group=$3
    local output_dir="${BASE_OUTPUT_DIR}/${run_id}"
    
    echo ""
    echo "========================================================================"
    echo "Starting test experiment: ${run_id}"
    echo "Outside weight: ${outside_weight}"
    echo "ERA5 group: ${era5_group}"
    echo "========================================================================"
    
    # Build base command
    local cmd="python models/mswep_unet_training.py"
    cmd="${cmd} --data_dir ${BASE_DATA_DIR}"
    cmd="${cmd} --output_dir ${output_dir}"
    cmd="${cmd} --epochs ${EPOCHS}"
    cmd="${cmd} --folds ${FOLDS}"
    cmd="${cmd} --batch_size ${BATCH_SIZE}"
    cmd="${cmd} --optimizer_type ${OPTIMIZER}"
    cmd="${cmd} --lr ${LR}"
    cmd="${cmd} --lr_scheduler_type ${LR_SCHEDULER}"
    cmd="${cmd} --loss_type ${LOSS_TYPE}"
    cmd="${cmd} --dropout ${DROPOUT}"
    cmd="${cmd} --weight_decay ${WEIGHT_DECAY}"
    cmd="${cmd} --use_regional_focus true"
    cmd="${cmd} --outside_weight ${outside_weight}"
    # Skip seasonal plots for speed in test
    # cmd="${cmd} --seasonal_plots"
    
    # Add ERA5 configuration based on group
    if [ "${era5_group}" != "none" ]; then
        cmd="${cmd} --era5_data_dir ${ERA5_DATA_DIR}"
        
        if [ "${era5_group}" == "wind_only" ]; then
            # Wind components only - reduced levels for test
            cmd="${cmd} --era5_variables u,v"
            cmd="${cmd} --era5_pressure_levels 300,850"  # Only 2 levels for speed
        fi
    fi
    
    # Log command
    echo "Command: ${cmd}"
    echo ""
    
    # Run the experiment
    eval "${cmd}"
    
    # Check if successful
    if [ $? -eq 0 ]; then
        echo "Test experiment ${run_id} completed successfully"
        
        # Find the actual run directory created by the training script
        local actual_run_dir=$(ls -td "${output_dir}/run_"* 2>/dev/null | head -n 1)
        
        if [ -n "${actual_run_dir}" ]; then
            # Add to manifest - use the last fold for evaluation
            local eval_fold_dir="${actual_run_dir}/fold$((FOLDS-1))"
            
            cat >> "${MANIFEST_FILE}" << EOF
  - run_id: "${run_id}"
    path: "${eval_fold_dir}"
    outside_weight: ${outside_weight}
    era5_group: "${era5_group}"
    note: "Test run: Group ${4}"
EOF
        else
            echo "Warning: Could not find run directory for ${run_id}"
        fi
    else
        echo "Error: Test experiment ${run_id} failed"
        return 1
    fi
}

# --- Main Test Execution ---

echo "Starting quick test suite"
echo "Output directory: ${BASE_OUTPUT_DIR}"
echo ""

# Group A: Regional weighting test
echo "=========================================="
echo "TEST GROUP A: Regional Weighting"
echo "=========================================="

for weight in "${GROUP_A_WEIGHTS[@]}"; do
    run_id="test_group_a_weight_${weight}"
    run_experiment "${run_id}" "${weight}" "none" "A"
done

# Quick evaluation of Group A to find best weight
echo ""
echo "Running test evaluation to verify CRPS/BS computation fix..."

# Run batch evaluation on test runs
python -m evaluation.batch_evaluate --manifest "${MANIFEST_FILE}" --output "${EVAL_DIR}/test_aggregated_metrics.csv"

# Check results
if [ -f "${EVAL_DIR}/test_aggregated_metrics.csv" ]; then
    echo ""
    echo "===== TEST EVALUATION COMPLETE ====="
    echo ""
    echo "CRPS/BS Fix Verification:"
    echo "------------------------"
    
    # Check if we have valid CRPS values (not NaN)
    python -c "
import pandas as pd
import sys

df = pd.read_csv('${EVAL_DIR}/test_aggregated_metrics.csv')
print(f'Total runs evaluated: {len(df)}')
print('')

# Check for NaN values in key metrics
has_valid_crps = df['mean_crps'].notna().any()
has_valid_bs = df['mean_bs'].notna().any()

if has_valid_crps and has_valid_bs:
    print('✅ CRPS computation: WORKING')
    print('✅ Brier Score computation: WORKING')
    print('✅ Fix verified successfully!')
    print('')
    
    # Show summary statistics
    print('Metric Summary:')
    print(f'  Mean CRPS range: {df[\"mean_crps\"].min():.4f} - {df[\"mean_crps\"].max():.4f}')
    print(f'  Mean BS range: {df[\"mean_bs\"].min():.4f} - {df[\"mean_bs\"].max():.4f}')
    print(f'  CRPSS range: {df[\"crpss_mpc\"].min():.4f} - {df[\"crpss_mpc\"].max():.4f}')
    
    # Check decomposition
    crps_check = abs(df['mean_crps'] - (df['crps_mcb'] - df['crps_dsc'] + df['crps_unc'])).max()
    bs_check = abs(df['mean_bs'] - (df['bs_mcb'] - df['bs_dsc'] + df['bs_unc'])).max()
    
    print('')
    print('Decomposition Checks:')
    print(f'  CRPS identity error: {crps_check:.2e} (should be < 1e-6)')
    print(f'  BS identity error: {bs_check:.2e} (should be < 1e-6)')
    
    if crps_check < 1e-6 and bs_check < 1e-6:
        print('✅ CORP decomposition: VERIFIED')
    else:
        print('❌ CORP decomposition: FAILED')
        
else:
    print('❌ CRPS computation: FAILED (NaN values)')
    print('❌ Brier Score computation: FAILED (NaN values)')
    print('❌ Fix not working - investigation needed')
    sys.exit(1)
"
    
    echo ""
    echo "Quick results overview:"
    echo "----------------------"
    
    # Show first few rows
    head -n 5 "${EVAL_DIR}/test_aggregated_metrics.csv" | cut -d, -f1-8 | column -t -s,
    
    echo ""
    echo "Generated files:"
    echo "- Test metrics: ${EVAL_DIR}/test_aggregated_metrics.csv"
    echo "- Test manifest: ${MANIFEST_FILE}"
    echo "- Experiment outputs: ${BASE_OUTPUT_DIR}/"
    
    # Show best configuration
    echo ""
    echo "Best test configuration:"
    python -c "
import pandas as pd
df = pd.read_csv('${EVAL_DIR}/test_aggregated_metrics.csv')
df['mean_crps'] = pd.to_numeric(df['mean_crps'], errors='coerce')
if not df.empty and df['mean_crps'].notna().any():
    best_idx = df['mean_crps'].idxmin()
    row = df.loc[best_idx]
    print(f'  Run ID: {row[\"run_id\"]}')
    print(f'  Outside weight: {row.get(\"outside_weight\", \"N/A\")}')
    print(f'  ERA5 group: {row.get(\"era5_group\", \"N/A\")}')
    print(f'  Mean CRPS: {row[\"mean_crps\"]:.4f}')
    print(f'  Mean BS: {row[\"mean_bs\"]:.4f}')
    print(f'  CRPSS: {row[\"crpss_mpc\"]:.4f}')
"
else
    echo "Test failed - check logs for errors"
    echo ""
    echo "Common issues:"
    echo "1. Data not found at expected paths"
    echo "2. Missing dependencies (PyYAML, isodisreg, etc.)"
    echo "3. Insufficient memory/disk space"
    echo "4. CRPS/BS computation still broken"
    echo ""
    echo "Check error output above for details"
fi

echo ""
echo "====================================="
echo "To run full experiments after fix verification, use:"
echo "  ./evaluation/run_experiments_and_evaluate.sh"
echo "====================================="

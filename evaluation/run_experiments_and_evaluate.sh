#!/bin/bash
# Comprehensive experiment runner and evaluator for MSWEP precipitation forecasting
# This script runs all configurations from Groups A and B, then evaluates them

# --- Configuration ---
set -e  # Exit on error

# Activate conda environment (commented out - already activated in terminal)
# echo "Activating conda environment..."
# conda activate precip_2.0

# Base directories
BASE_DATA_DIR="data/MSWEP_daily"
ERA5_DATA_DIR="precip_data/era5_precipitation_project/predictors"
BASE_OUTPUT_DIR="./experiment_runs"
EVAL_DIR="evaluation"
MANIFEST_FILE="${EVAL_DIR}/experiment_manifest.yaml"

# Training parameters
EPOCHS=20
FOLDS=10
BATCH_SIZE=32
OPTIMIZER="adamw"
LR=0.00001
LR_SCHEDULER="cosineannealinglr"
LOSS_TYPE="mae"
DROPOUT=0.2
WEIGHT_DECAY=1e-4

# Experiment groups
declare -a GROUP_A_WEIGHTS=(0.2 0.5 0.7)
declare -a GROUP_B_ERA5=("none" "wind_only" "all")

# For quick testing, uncomment these:
# declare -a GROUP_A_WEIGHTS=(0.2 0.5)
# declare -a GROUP_B_ERA5=("none" "wind_only")
# EPOCHS=5
# FOLDS=2

# Create output directories
mkdir -p "${BASE_OUTPUT_DIR}"
mkdir -p "${EVAL_DIR}/figures"
mkdir -p "${EVAL_DIR}/tables"

# Initialize manifest file
echo "Creating manifest file: ${MANIFEST_FILE}"
cat > "${MANIFEST_FILE}" << EOF
# Automatically generated manifest for experiment runs
# Generated on: $(date)

runs:
EOF

# Function to run a single experiment
run_experiment() {
    local run_id=$1
    local outside_weight=$2
    local era5_group=$3
    local output_dir="${BASE_OUTPUT_DIR}/${run_id}"
    
    echo ""
    echo "========================================================================"
    echo "Starting experiment: ${run_id}"
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
    cmd="${cmd} --seasonal_plots"
    
    # Add ERA5 configuration based on group
    if [ "${era5_group}" != "none" ]; then
        cmd="${cmd} --era5_data_dir ${ERA5_DATA_DIR}"
        
        if [ "${era5_group}" == "wind_only" ]; then
            # Wind components only
            cmd="${cmd} --era5_variables u,v"
            cmd="${cmd} --era5_pressure_levels 300,500,700,850"
        elif [ "${era5_group}" == "all" ]; then
            # Comprehensive configuration
            cmd="${cmd} --era5_variables u,v,q,mslp,t2m,tcwv"
            cmd="${cmd} --era5_pressure_levels 300,500,700,850"
            # Add any additional parameters for comprehensive config
        fi
    fi
    
    # Log command
    echo "Command: ${cmd}"
    echo ""
    
    # Run the experiment
    eval "${cmd}"
    
    # Check if successful
    if [ $? -eq 0 ]; then
        echo "Experiment ${run_id} completed successfully"
        
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
    note: "Auto-generated: Group ${4}"
EOF
        else
            echo "Warning: Could not find run directory for ${run_id}"
        fi
    else
        echo "Error: Experiment ${run_id} failed"
        return 1
    fi
}

# Function to determine best outside weight from Group A results
find_best_outside_weight() {
    # This will be determined after Group A runs complete
    # For now, we'll use a default or analyze the preliminary results
    echo "0.5"  # Default, will be updated based on results
}

# --- Main Execution ---

echo "Starting comprehensive experiment suite"
echo "Output directory: ${BASE_OUTPUT_DIR}"
echo ""

# Group A: Regional weighting sweep (rainfall + seasonality only)
echo "=========================================="
echo "GROUP A: Regional Weighting Sweep"
echo "=========================================="

for weight in "${GROUP_A_WEIGHTS[@]}"; do
    run_id="group_a_weight_${weight}"
    run_experiment "${run_id}" "${weight}" "none" "A"
done

# Quick evaluation of Group A to find best weight
echo ""
echo "Running preliminary evaluation to find best outside weight..."

# Create temporary manifest for Group A only
TEMP_MANIFEST="${EVAL_DIR}/temp_group_a_manifest.yaml"
head -n 4 "${MANIFEST_FILE}" > "${TEMP_MANIFEST}"
grep -A 5 "group_a_" "${MANIFEST_FILE}" >> "${TEMP_MANIFEST}"

# Run batch evaluation on Group A
python -m evaluation.batch_evaluate --manifest "${TEMP_MANIFEST}" --output "${EVAL_DIR}/group_a_metrics.csv"

# Find best outside weight from Group A results
if [ -f "${EVAL_DIR}/group_a_metrics.csv" ]; then
    BEST_WEIGHT=$(python -c "
import pandas as pd
df = pd.read_csv('${EVAL_DIR}/group_a_metrics.csv')
df['mean_crps'] = pd.to_numeric(df['mean_crps'], errors='coerce')
best_idx = df['mean_crps'].idxmin()
print(df.loc[best_idx, 'outside_weight'])
")
    echo "Best outside weight from Group A: ${BEST_WEIGHT}"
else
    BEST_WEIGHT="0.5"
    echo "Warning: Could not determine best weight, using default: ${BEST_WEIGHT}"
fi

# Group B: ERA5 feature groups (at best outside weight)
echo ""
echo "=========================================="
echo "GROUP B: ERA5 Feature Groups (weight=${BEST_WEIGHT})"
echo "=========================================="

for era5_group in "${GROUP_B_ERA5[@]}"; do
    run_id="group_b_${era5_group}_w${BEST_WEIGHT}"
    run_experiment "${run_id}" "${BEST_WEIGHT}" "${era5_group}" "B"
done

# --- Comprehensive Evaluation ---
echo ""
echo "=========================================="
echo "COMPREHENSIVE EVALUATION"
echo "=========================================="

# Run full evaluation pipeline
echo "Running complete evaluation pipeline..."
python -m evaluation.make_results --manifest "${MANIFEST_FILE}"

# Run acceptance checks
echo ""
echo "Running acceptance checks..."
python -m evaluation.acceptance_check --csv "${EVAL_DIR}/aggregated_metrics.csv"

# Generate LaTeX tables
echo ""
echo "Generating LaTeX tables..."

# Regional weighting table
python -m evaluation.export_latex \
    --csv "${EVAL_DIR}/aggregated_metrics.csv" \
    --out "${EVAL_DIR}/tables/regional_weighting.tex" \
    --select 'era5_group=="none"' \
    --caption "Regional weighting sweep results (rainfall and seasonality only)"

# ERA5 comparison table
python -m evaluation.export_latex \
    --csv "${EVAL_DIR}/aggregated_metrics.csv" \
    --out "${EVAL_DIR}/tables/era5_comparison.tex" \
    --select "outside_weight==${BEST_WEIGHT}" \
    --caption "ERA5 feature group comparison at optimal outside weight (${BEST_WEIGHT})"

# Find overall best configuration
BEST_RUN_ID=$(python -c "
import pandas as pd
df = pd.read_csv('${EVAL_DIR}/aggregated_metrics.csv')
df['mean_crps'] = pd.to_numeric(df['mean_crps'], errors='coerce')
best_idx = df['mean_crps'].idxmin()
print(df.loc[best_idx, 'run_id'])
")

# Best model tables
python -m evaluation.export_latex \
    --csv "${EVAL_DIR}/aggregated_metrics.csv" \
    --out "${EVAL_DIR}/tables/best_model_seasonal.tex" \
    --select "run_id=='${BEST_RUN_ID}'" \
    --caption "Seasonal performance of best configuration"

python -m evaluation.export_latex \
    --csv "${EVAL_DIR}/aggregated_metrics.csv" \
    --out "${EVAL_DIR}/tables/best_model_decomposition.tex" \
    --table-type decomposition \
    --run-id "${BEST_RUN_ID}"

# --- Final Summary ---
echo ""
echo "========================================================================"
echo "EXPERIMENT SUITE COMPLETE"
echo "========================================================================"
echo ""
echo "Results summary:"
echo "  - Experiment outputs: ${BASE_OUTPUT_DIR}/"
echo "  - Aggregated metrics: ${EVAL_DIR}/aggregated_metrics.csv"
echo "  - Figures: ${EVAL_DIR}/figures/"
echo "  - LaTeX tables: ${EVAL_DIR}/tables/"
echo ""
echo "Best configuration: ${BEST_RUN_ID}"
echo "Best outside weight: ${BEST_WEIGHT}"
echo ""
echo "Key figures generated:"
ls -1 "${EVAL_DIR}/figures/"*.png 2>/dev/null | head -5
echo ""
echo "To view results:"
echo "  - Overall summary: cat ${EVAL_DIR}/aggregated_metrics.csv | column -t -s,"
echo "  - Best configs: grep -E '${BEST_RUN_ID}|run_id' ${EVAL_DIR}/aggregated_metrics.csv"
echo ""

# Cleanup temporary files
rm -f "${TEMP_MANIFEST}"

echo "Pipeline completed successfully!"

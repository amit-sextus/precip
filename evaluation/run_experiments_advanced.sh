#!/bin/bash
# Advanced experiment runner with configuration file support
# Supports resuming failed runs and parallel execution

# --- Configuration ---
set -e  # Exit on error

# Default configuration file
CONFIG_FILE="${1:-evaluation/experiment_config.yaml}"

# Parse command line options
RESUME=false
PARALLEL=false
GROUP=""
SKIP_EVALUATION=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --group)
            GROUP="$2"
            shift 2
            ;;
        --skip-evaluation)
            SKIP_EVALUATION=true
            shift
            ;;
        --help)
            cat << EOF
Usage: $0 [OPTIONS]

Options:
    --config FILE       Configuration file (default: evaluation/experiment_config.yaml)
    --resume           Resume from previous runs (skip completed experiments)
    --parallel         Run experiments in parallel (requires GNU parallel)
    --group A|B        Run only Group A or B experiments
    --skip-evaluation  Skip the evaluation phase
    --help            Show this help message

Examples:
    # Run all experiments
    $0

    # Run only Group A with custom config
    $0 --config myconfig.yaml --group A

    # Resume failed runs in parallel
    $0 --resume --parallel
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check for required tools
if [ "$PARALLEL" = true ] && ! command -v parallel &> /dev/null; then
    echo "Error: GNU parallel not found. Install with: sudo apt-get install parallel"
    exit 1
fi

if ! command -v yq &> /dev/null; then
    echo "Error: yq not found. Install with: pip install yq"
    exit 1
fi

# Activate conda environment (commented out - already activated in terminal)
# echo "Activating conda environment..."
# conda activate precip_2.0

# Parse configuration file
echo "Loading configuration from: ${CONFIG_FILE}"

# Extract values from YAML using yq
MSWEP_DIR=$(yq -r '.data.mswep_dir' "$CONFIG_FILE")
ERA5_DIR=$(yq -r '.data.era5_dir' "$CONFIG_FILE")
BASE_OUTPUT_DIR=$(yq -r '.data.output_base' "$CONFIG_FILE")

EPOCHS=$(yq -r '.training.epochs' "$CONFIG_FILE")
FOLDS=$(yq -r '.training.folds' "$CONFIG_FILE")
BATCH_SIZE=$(yq -r '.training.batch_size' "$CONFIG_FILE")
OPTIMIZER=$(yq -r '.training.optimizer' "$CONFIG_FILE")
LR=$(yq -r '.training.learning_rate' "$CONFIG_FILE")
LR_SCHEDULER=$(yq -r '.training.lr_scheduler' "$CONFIG_FILE")
LOSS_TYPE=$(yq -r '.training.loss_type' "$CONFIG_FILE")
DROPOUT=$(yq -r '.training.dropout' "$CONFIG_FILE")
WEIGHT_DECAY=$(yq -r '.training.weight_decay' "$CONFIG_FILE")
SEASONAL_PLOTS=$(yq -r '.training.seasonal_plots' "$CONFIG_FILE")
SKIP_CRPS=$(yq -r '.training.skip_crps' "$CONFIG_FILE")

# Check for test mode
if yq -e '.test_mode' "$CONFIG_FILE" > /dev/null 2>&1; then
    echo "Test mode detected - using reduced parameters"
    EPOCHS=$(yq -r '.test_mode.epochs // .training.epochs' "$CONFIG_FILE")
    FOLDS=$(yq -r '.test_mode.folds // .training.folds' "$CONFIG_FILE")
fi

# Create directories
EVAL_DIR="evaluation"
MANIFEST_FILE="${EVAL_DIR}/experiment_manifest.yaml"
RUN_LOG="${BASE_OUTPUT_DIR}/run_log.txt"

mkdir -p "${BASE_OUTPUT_DIR}"
mkdir -p "${EVAL_DIR}/figures"
mkdir -p "${EVAL_DIR}/tables"

# Initialize or update manifest
if [ "$RESUME" = false ] || [ ! -f "$MANIFEST_FILE" ]; then
    cat > "${MANIFEST_FILE}" << EOF
# Automatically generated manifest for experiment runs
# Generated on: $(date)
# Configuration: ${CONFIG_FILE}

runs:
EOF
fi

# Function to check if experiment already completed
is_completed() {
    local run_id=$1
    if [ -f "$RUN_LOG" ] && grep -q "^${run_id}:COMPLETED" "$RUN_LOG"; then
        return 0
    fi
    return 1
}

# Function to mark experiment as completed
mark_completed() {
    local run_id=$1
    echo "${run_id}:COMPLETED:$(date)" >> "$RUN_LOG"
}

# Function to run a single experiment
run_experiment() {
    local run_id=$1
    local outside_weight=$2
    local era5_group=$3
    local group_label=$4
    local output_dir="${BASE_OUTPUT_DIR}/${run_id}"
    
    # Check if already completed
    if [ "$RESUME" = true ] && is_completed "$run_id"; then
        echo "Skipping completed experiment: ${run_id}"
        return 0
    fi
    
    echo ""
    echo "========================================================================"
    echo "Starting experiment: ${run_id}"
    echo "Outside weight: ${outside_weight}"
    echo "ERA5 group: ${era5_group}"
    echo "========================================================================"
    
    # Build base command
    local cmd="python models/mswep_unet_training.py"
    cmd="${cmd} --data_dir ${MSWEP_DIR}"
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
    
    # Add optional flags
    if [ "$SEASONAL_PLOTS" = "true" ]; then
        cmd="${cmd} --seasonal_plots"
    fi
    
    if [ "$SKIP_CRPS" = "true" ]; then
        cmd="${cmd} --skip_crps true"
    fi
    
    # Add ERA5 configuration based on group
    if [ "${era5_group}" != "none" ]; then
        # Check if ERA5 data directory exists
        if [ ! -d "${ERA5_DIR}" ]; then
            echo "Warning: ERA5 data directory not found: ${ERA5_DIR}"
            echo "Skipping ERA5 features for this run"
        else
            cmd="${cmd} --era5_data_dir ${ERA5_DIR}"
            
            # Get ERA5 variables from config
            local era5_vars=$(yq -r ".experiments.group_b.configurations.${era5_group}.era5_variables | join(\",\")" "$CONFIG_FILE")
            local era5_levels=$(yq -r ".experiments.group_b.configurations.${era5_group}.era5_pressure_levels | join(\",\")" "$CONFIG_FILE")
            
            if [ "$era5_vars" != "null" ] && [ "$era5_vars" != "" ]; then
                cmd="${cmd} --era5_variables ${era5_vars}"
            fi
            
            if [ "$era5_levels" != "null" ] && [ "$era5_levels" != "" ]; then
                cmd="${cmd} --era5_pressure_levels ${era5_levels}"
            fi
        fi
    fi
    
    # Log command
    echo "Command: ${cmd}"
    echo ""
    
    # Create a log file for this specific run
    local run_log_file="${output_dir}/run.log"
    mkdir -p "${output_dir}"
    
    # Run the experiment
    if eval "${cmd}" 2>&1 | tee "${run_log_file}"; then
        echo "Experiment ${run_id} completed successfully"
        mark_completed "$run_id"
        
        # Find the actual run directory created by the training script
        local actual_run_dir=$(ls -td "${output_dir}/run_"* 2>/dev/null | head -n 1)
        
        if [ -n "${actual_run_dir}" ]; then
            # Add to manifest - use the last fold for evaluation
            local eval_fold_dir="${actual_run_dir}/fold$((FOLDS-1))"
            
            # Check if entry already exists in manifest
            if ! grep -q "run_id: \"${run_id}\"" "$MANIFEST_FILE"; then
                cat >> "${MANIFEST_FILE}" << EOF
  - run_id: "${run_id}"
    path: "${eval_fold_dir}"
    outside_weight: ${outside_weight}
    era5_group: "${era5_group}"
    note: "Auto-generated: Group ${group_label}"
EOF
            fi
        else
            echo "Warning: Could not find run directory for ${run_id}"
        fi
    else
        echo "Error: Experiment ${run_id} failed"
        echo "${run_id}:FAILED:$(date)" >> "$RUN_LOG"
        return 1
    fi
}

# Export function for parallel execution
export -f run_experiment is_completed mark_completed
export CONFIG_FILE MSWEP_DIR ERA5_DIR BASE_OUTPUT_DIR EPOCHS FOLDS BATCH_SIZE
export OPTIMIZER LR LR_SCHEDULER LOSS_TYPE DROPOUT WEIGHT_DECAY
export SEASONAL_PLOTS SKIP_CRPS RUN_LOG MANIFEST_FILE RESUME

# --- Main Execution ---

echo "Starting comprehensive experiment suite"
echo "Configuration: ${CONFIG_FILE}"
echo "Output directory: ${BASE_OUTPUT_DIR}"
echo "Resume mode: ${RESUME}"
echo "Parallel mode: ${PARALLEL}"
echo ""

# Build experiment list
EXPERIMENTS=()

# Group A experiments
if [ -z "$GROUP" ] || [ "$GROUP" = "A" ]; then
    # Get weights from config
    GROUP_A_WEIGHTS=($(yq -r '.experiments.group_a.outside_weights[]' "$CONFIG_FILE"))
    
    echo "Group A weights: ${GROUP_A_WEIGHTS[@]}"
    
    for weight in "${GROUP_A_WEIGHTS[@]}"; do
        EXPERIMENTS+=("group_a_weight_${weight}:${weight}:none:A")
    done
fi

# Group B experiments (need to determine best weight first)
if [ -z "$GROUP" ] || [ "$GROUP" = "B" ]; then
    # If Group A was run, evaluate to find best weight
    if [ -z "$GROUP" ] || [ "$GROUP" = "A" ]; then
        echo ""
        echo "Waiting for Group A to complete before starting Group B..."
        
        # Wait for Group A experiments if running in parallel
        if [ "$PARALLEL" = true ]; then
            wait
        fi
        
        # Quick evaluation of Group A
        if [ -f "$MANIFEST_FILE" ]; then
            echo "Running preliminary evaluation to find best outside weight..."
            
            # Create temporary manifest for Group A only
            TEMP_MANIFEST="${EVAL_DIR}/temp_group_a_manifest.yaml"
            head -n 5 "$MANIFEST_FILE" > "$TEMP_MANIFEST"
            grep -A 5 "group_a_" "$MANIFEST_FILE" >> "$TEMP_MANIFEST" || true
            
            if [ -s "$TEMP_MANIFEST" ]; then
                python -m evaluation.batch_evaluate --manifest "$TEMP_MANIFEST" --output "${EVAL_DIR}/group_a_metrics.csv" --overwrite
                
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
                
                rm -f "$TEMP_MANIFEST"
            else
                BEST_WEIGHT="0.5"
                echo "No Group A results found, using default weight: ${BEST_WEIGHT}"
            fi
        else
            BEST_WEIGHT="0.5"
            echo "No manifest found, using default weight: ${BEST_WEIGHT}"
        fi
    else
        # If only running Group B, need to specify best weight
        BEST_WEIGHT="${BEST_WEIGHT:-0.5}"
        echo "Using specified best weight for Group B: ${BEST_WEIGHT}"
    fi
    
    # Get ERA5 configurations
    ERA5_CONFIGS=($(yq -r '.experiments.group_b.configurations | keys[]' "$CONFIG_FILE"))
    
    echo "Group B ERA5 configurations: ${ERA5_CONFIGS[@]}"
    
    for era5_group in "${ERA5_CONFIGS[@]}"; do
        EXPERIMENTS+=("group_b_${era5_group}_w${BEST_WEIGHT}:${BEST_WEIGHT}:${era5_group}:B")
    done
fi

# Run experiments
if [ "$PARALLEL" = true ]; then
    echo "Running ${#EXPERIMENTS[@]} experiments in parallel..."
    printf '%s\n' "${EXPERIMENTS[@]}" | parallel -j 4 --colsep ':' run_experiment {1} {2} {3} {4}
else
    echo "Running ${#EXPERIMENTS[@]} experiments sequentially..."
    for exp in "${EXPERIMENTS[@]}"; do
        IFS=':' read -r run_id weight era5_group group_label <<< "$exp"
        run_experiment "$run_id" "$weight" "$era5_group" "$group_label"
    done
fi

# --- Evaluation Phase ---
if [ "$SKIP_EVALUATION" = false ]; then
    echo ""
    echo "=========================================="
    echo "COMPREHENSIVE EVALUATION"
    echo "=========================================="
    
    # Check if any experiments were run
    if [ ! -f "$MANIFEST_FILE" ] || [ $(grep -c "run_id:" "$MANIFEST_FILE") -eq 0 ]; then
        echo "No experiments found to evaluate"
        exit 0
    fi
    
    # Run full evaluation pipeline
    echo "Running complete evaluation pipeline..."
    python -m evaluation.make_results --manifest "${MANIFEST_FILE}"
    
    # Generate summary report
    if [ -f "${EVAL_DIR}/aggregated_metrics.csv" ]; then
        echo ""
        echo "=========================================="
        echo "EXPERIMENT SUMMARY"
        echo "=========================================="
        
        # Show failed runs if any
        if [ -f "$RUN_LOG" ]; then
            FAILED_RUNS=$(grep ":FAILED:" "$RUN_LOG" | cut -d: -f1)
            if [ -n "$FAILED_RUNS" ]; then
                echo ""
                echo "Failed experiments:"
                echo "$FAILED_RUNS"
            fi
        fi
        
        # Best configuration
        BEST_CONFIG=$(python -c "
import pandas as pd
df = pd.read_csv('${EVAL_DIR}/aggregated_metrics.csv')
df['mean_crps'] = pd.to_numeric(df['mean_crps'], errors='coerce')
best_idx = df['mean_crps'].idxmin()
row = df.loc[best_idx]
print(f\"Run ID: {row['run_id']}\")
print(f\"Outside weight: {row.get('outside_weight', 'N/A')}\")
print(f\"ERA5 group: {row.get('era5_group', 'N/A')}\")
print(f\"Mean CRPS: {row['mean_crps']:.4f}\")
print(f\"CRPSS vs MPC: {row.get('crpss_mpc', 'N/A'):.4f}\")
")
        
        echo ""
        echo "Best configuration:"
        echo "$BEST_CONFIG"
        
        echo ""
        echo "All results saved to:"
        echo "  - Metrics: ${EVAL_DIR}/aggregated_metrics.csv"
        echo "  - Figures: ${EVAL_DIR}/figures/"
        echo "  - Tables: ${EVAL_DIR}/tables/"
    fi
else
    echo "Skipping evaluation phase as requested"
fi

echo ""
echo "Pipeline completed!"

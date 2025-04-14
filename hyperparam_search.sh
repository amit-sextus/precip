#!/bin/bash

# --- Configuration ---
PYTHON_SCRIPT="models/mswep_unet_training.py"
SKIP_CRPS=true                                 # Set to false if you want to calculate CRPS (time consuming)

# Check for jq dependency
if ! command -v jq &> /dev/null
then
    echo "ERROR: jq is not installed."
    echo "Please install jq (e.g., sudo apt-get update && sudo apt-get install jq)"
    exit 1
fi

# Check for correct data directory
BASE_DATA_DIR="/home/batman/precipitation/data"

# Add a verification step for data directory
echo "Verifying data directory: ${BASE_DATA_DIR}"
if [ ! -d "${BASE_DATA_DIR}" ]; then
    echo "ERROR: Data directory does not exist: ${BASE_DATA_DIR}"
    echo "Please update the BASE_DATA_DIR variable with the correct path to your MSWEP data"
    exit 1
fi

# Check for NetCDF files that might be MSWEP data
NETCDF_COUNT=$(find "${BASE_DATA_DIR}" -name "*.nc" | wc -l)
echo "Found ${NETCDF_COUNT} NetCDF (.nc) files in data directory"

if [ "${NETCDF_COUNT}" -eq 0 ]; then
    echo "ERROR: No NetCDF (.nc) files found in data directory"
    echo "Make sure your data is in the correct location or update BASE_DATA_DIR"
    echo ""
    echo "You can provide the correct data directory as an argument:"
    echo "  ./hyperparam_search.sh /path/to/mswep/data"
    exit 1
fi

# Allow overriding data directory from command line
if [ $# -ge 1 ]; then
    if [ -d "$1" ]; then
        echo "Using data directory from command line argument: $1"
        BASE_DATA_DIR="$1"
    else
        echo "ERROR: Provided data directory does not exist: $1"
        exit 1
    fi
fi

BASE_OUTPUT_DIR="./hyperparam_search_results"
BASE_LOG_DIR="./hyperparam_search_logs"

# Fixed parameters for all evaluations
NUM_EPOCHS=20
NUM_FOLDS=2
TARGET_SOURCE="mswep"
THRESHOLD=0.2

# Parameters to Vary
echo "Running focused hyperparameter search to address plateau and overfitting issues"
declare -a lrs=(0.001 0.0003 0.00005)
declare -a loss_types=('mse' 'log_cosh' 'asymmetric_mse')
declare -a optimizer_types=('adamw')
declare -a scheduler_types=('cosineannealinglr' 'reducelronplateau')
declare -a weight_decays=(1e-4 1e-3)
declare -a dropouts=(0.2 0.4)
declare -a transform_probs=(0.5)
declare -a regional_focus=(True)
declare -a log_offsets=(0.01)
declare -a outside_weights=(0.01 0.1 0.2 0.5 1)

# --- For quick testing, uncomment these lines instead ---
# declare -a lrs=(0.0001)
# declare -a loss_types=('mse' 'log_cosh')
# declare -a optimizer_types=('adamw')
# declare -a scheduler_types=('cosineannealinglr')
# declare -a weight_decays=(1e-3)
# declare -a dropouts=(0.2 0.3)
# declare -a transform_probs=(0.7)
# declare -a regional_focus=(True)
# declare -a log_offsets=(0.01)
# declare -a outside_weights=(0.2)

RESULTS_FILE="${BASE_OUTPUT_DIR}/hyperparam_results_summary.txt"

run_counter=0

mkdir -p $BASE_OUTPUT_DIR
mkdir -p $BASE_LOG_DIR

# Updated header for desired metrics and new parameter
echo "Run ID, Learning Rate, Loss Type, Optimizer, Scheduler, Weight Decay, Dropout, Transform Prob, Regional Focus, Outside Weight, Log Offset, Min Val Loss, Min Val MAE, Min Val RMSE" > $RESULTS_FILE

for lr in "${lrs[@]}"; do
  for loss in "${loss_types[@]}"; do
    for opt in "${optimizer_types[@]}"; do
      for sched in "${scheduler_types[@]}"; do
        for wd in "${weight_decays[@]}"; do
          for dr in "${dropouts[@]}"; do
            for tp in "${transform_probs[@]}"; do
              for rf in "${regional_focus[@]}"; do
                for lo in "${log_offsets[@]}"; do
                  for ow in "${outside_weights[@]}"; do

                    run_counter=$((run_counter + 1))

                    timestamp=$(date +"%Y%m%d_%H%M%S")
                    run_id="run${run_counter}_${timestamp}_lr${lr}_${loss}_${opt}_${sched}_wd${wd}_dr${dr}_tp${tp}_rf${rf}_ow${ow}_lo${lo}"
                    output_dir="${BASE_OUTPUT_DIR}/${run_id}"
                    log_dir="${BASE_LOG_DIR}/${run_id}"

                    mkdir -p $output_dir
                    mkdir -p $log_dir

                    echo "######################################################################"
                    echo "Starting Run ${run_counter}: ${run_id}"
                    echo "######################################################################"

                    command="python ${PYTHON_SCRIPT} \
                      --data_dir ${BASE_DATA_DIR} \
                      --output_dir ${output_dir} \
                      --log_dir ${log_dir} \
                      --epochs ${NUM_EPOCHS} \
                      --folds ${NUM_FOLDS} \
                      --lr ${lr} \
                      --loss_type ${loss} \
                      --optimizer_type ${opt} \
                      --lr_scheduler_type ${sched} \
                      --weight_decay ${wd} \
                      --dropout ${dr} \
                      --transform_probability ${tp} \
                      --target_source ${TARGET_SOURCE} \
                      --use_regional_focus ${rf} \
                      --threshold ${THRESHOLD} \
                      --log_offset ${lo} \
                      --skip_crps ${SKIP_CRPS}"

                    # Add outside_weight only if regional focus is True
                    if [ "$rf" = "True" ]; then
                      command="${command} --outside_weight ${ow}"
                    fi

                    echo "$command"
                    echo "----------------------------------------------------------------------"

                    eval $command
                    
                    exit_status=$?
                    if [ $exit_status -ne 0 ]; then
                        echo "ERROR: Command failed with exit status $exit_status"
                        echo "Check the error message above and fix data issues before continuing"
                    fi

                    # Extract and record results
                    echo "DEBUG: Searching for run_* directory inside ${output_dir}"
                    latest_run_dir=$(ls -td "${output_dir}/run_"* 2>/dev/null | head -n 1)
                    
                    if [ -z "$latest_run_dir" ]; then
                       echo "Error: Could not find a 'run_*' directory inside ${output_dir}"
                       echo "DEBUG: Listing contents of ${output_dir}:"
                       ls -l "${output_dir}"
                    else
                       echo "DEBUG: Found latest run directory: ${latest_run_dir}"
                    fi

                    min_val_loss="N/A"
                    min_val_mae="N/A"
                    min_val_rmse="N/A"

                    if [ -n "$latest_run_dir" ] && [ -d "$latest_run_dir" ]; then
                        fold0_dir="${latest_run_dir}/fold0"

                        if [ -d "$fold0_dir" ]; then
                            json_file="${fold0_dir}/training_results_fold0.json"
                            echo "Attempting to read metrics from: ${json_file}"

                            if [ -f "$json_file" ]; then
                                echo "Reading metrics using jq from: $json_file"
                                best_val_loss=$(jq -r '.best_val_loss // "N/A"' "$json_file")
                                best_val_mae=$(jq -r '.best_val_mae // "N/A"' "$json_file")
                                best_val_rmse=$(jq -r '.best_val_rmse // "N/A"' "$json_file")

                                if [[ "$best_val_loss" == "N/A" ]]; then
                                    echo "Warning: 'best_val_loss' key not found or null in $json_file"
                                fi
                                if [[ "$best_val_mae" == "N/A" ]]; then
                                    echo "Warning: 'best_val_mae' key not found or null in $json_file"
                                fi
                                if [[ "$best_val_rmse" == "N/A" ]]; then
                                    echo "Warning: 'best_val_rmse' key not found or null in $json_file"
                                fi
                                min_val_loss=$best_val_loss
                                min_val_mae=$best_val_mae
                                min_val_rmse=$best_val_rmse
                            else
                                echo "ERROR: JSON file not found at expected path: $json_file"
                            fi
                        else
                            echo "Warning: fold0 directory not found within run directory: ${fold0_dir}"
                        fi
                    else
                        echo "Warning: No run directory (like run_*) found within ${output_dir} or ${latest_run_dir} is not a directory."
                    fi

                    # Record results with the potentially extracted metrics
                    echo "${run_id}, ${lr}, ${loss}, ${opt}, ${sched}, ${wd}, ${dr}, ${tp}, ${rf}, ${ow}, ${lo}, ${min_val_loss}, ${min_val_mae}, ${min_val_rmse}" >> $RESULTS_FILE

                    # --- Cleanup: Remove run output directory --- 
                    echo "Cleaning up run output directory: ${output_dir}"
                    if [ -d "$output_dir" ]; then 
                        rm -rf "$output_dir"
                        echo "Removed directory: ${output_dir}"
                    else
                        echo "Warning: Output directory not found for cleanup: ${output_dir}"
                    fi

                    echo "----------------------------------------------------------------------"
                    echo "Finished Run ${run_counter}: ${run_id}"
                    echo "Results in: ${output_dir}"
                    echo "Logs in: ${log_dir}"
                    echo "######################################################################"
                    echo ""
                    sleep 2 # Small pause between runs

                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "All hyperparameter search runs completed."
echo "Results summary saved to: ${RESULTS_FILE}"

# Sort results by validation loss (best to worst)
echo "Top 5 best configurations by Min Val Loss:"
grep -v ", N/A," $RESULTS_FILE | grep -v "Run ID" | sort -t, -k12 -n | head -n 5

# Optionally, show top results by other metrics
echo ""
echo "Top 5 best configurations by Min Val MAE (if available):"
grep -v ", N/A," $RESULTS_FILE | grep -v "Run ID" | sort -t, -k13 -n | head -n 5

echo ""
echo "Top 5 best configurations by Min Val RMSE (if available):"
grep -v ", N/A," $RESULTS_FILE | grep -v "Run ID" | sort -t, -k14 -n | head -n 5

echo ""
echo "To analyze all results in detail, examine: ${RESULTS_FILE}"
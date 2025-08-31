#!/bin/bash
# Example script to run the complete evaluation pipeline
# This demonstrates the full workflow from manifest to publication-ready outputs

# Ensure we're in the correct conda environment (commented out - already activated)
# echo "Activating conda environment..."
# conda activate precip_2.0

# Set base directory
EVAL_DIR="evaluation"
MANIFEST="${EVAL_DIR}/run_manifest.yaml"  # Your actual manifest
OUTPUT_DIR="${EVAL_DIR}"

# Step 1: Generate all results and figures
echo "=================="
echo "Step 1: Running complete evaluation and generating figures..."
echo "=================="
python -m evaluation.make_results --manifest "${MANIFEST}"

# Check if successful
if [ $? -ne 0 ]; then
    echo "Error: make_results failed"
    exit 1
fi

# Step 2: Run acceptance checks
echo ""
echo "=================="
echo "Step 2: Running acceptance checks..."
echo "=================="
python -m evaluation.acceptance_check --csv "${OUTPUT_DIR}/aggregated_metrics.csv"

# Continue even if checks fail (for demonstration)

# Step 3: Generate LaTeX tables
echo ""
echo "=================="
echo "Step 3: Generating LaTeX tables..."
echo "=================="

# Create tables directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}/tables"

# Regional weighting table (era5_group="none")
echo "Creating regional weighting table..."
python -m evaluation.export_latex \
    --csv "${OUTPUT_DIR}/aggregated_metrics.csv" \
    --out "${OUTPUT_DIR}/tables/regional_weighting.tex" \
    --select 'era5_group=="none"' \
    --caption "Regional weighting sweep results (rainfall and seasonality only)"

# ERA5 comparison table (best outside weight)
echo "Creating ERA5 comparison table..."
python -m evaluation.export_latex \
    --csv "${OUTPUT_DIR}/aggregated_metrics.csv" \
    --out "${OUTPUT_DIR}/tables/era5_comparison.tex" \
    --caption "ERA5 feature group comparison at optimal outside weight"

# Find best run ID (lowest CRPS) for detailed tables
BEST_RUN_ID=$(python -c "
import pandas as pd
df = pd.read_csv('${OUTPUT_DIR}/aggregated_metrics.csv')
df['mean_crps'] = pd.to_numeric(df['mean_crps'], errors='coerce')
best_idx = df['mean_crps'].idxmin()
print(df.loc[best_idx, 'run_id'])
")

echo "Best run identified: ${BEST_RUN_ID}"

# Best model seasonal summary
echo "Creating best model seasonal summary..."
python -m evaluation.export_latex \
    --csv "${OUTPUT_DIR}/aggregated_metrics.csv" \
    --out "${OUTPUT_DIR}/tables/best_model_seasonal.tex" \
    --select "run_id=='${BEST_RUN_ID}'" \
    --caption "Seasonal performance of best configuration"

# Decomposition table for best model
echo "Creating decomposition table..."
python -m evaluation.export_latex \
    --csv "${OUTPUT_DIR}/aggregated_metrics.csv" \
    --out "${OUTPUT_DIR}/tables/best_model_decomposition.tex" \
    --table-type decomposition \
    --run-id "${BEST_RUN_ID}"

# Step 4: Summary
echo ""
echo "=================="
echo "Pipeline Complete!"
echo "=================="
echo ""
echo "Generated outputs:"
echo "  - Metrics: ${OUTPUT_DIR}/aggregated_metrics.csv"
echo "  - Figures: ${OUTPUT_DIR}/figures/"
echo "  - Tables: ${OUTPUT_DIR}/tables/"
echo ""
echo "Key figures:"
ls -la "${OUTPUT_DIR}/figures/"*.png 2>/dev/null | awk '{print "    - " $NF}'
echo ""
echo "LaTeX tables:"
ls -la "${OUTPUT_DIR}/tables/"*.tex 2>/dev/null | awk '{print "    - " $NF}'
echo ""
echo "To use in LaTeX document:"
echo "  \usepackage{booktabs}"
echo "  \input{${OUTPUT_DIR}/tables/regional_weighting.tex}"
echo ""

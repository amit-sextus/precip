#!/bin/bash
# Quick test example - runs minimal experiments for testing the pipeline
# This uses reduced epochs/folds and only a subset of configurations

echo "Running quick test of experiment pipeline..."
echo "This will run reduced experiments to verify the system works"
echo ""

# Activate environment (commented out - already activated in terminal)
# conda activate precip_2.0

# Create test configuration
cat > evaluation/test_config.yaml << EOF
# Test configuration - minimal settings for quick verification

data:
  mswep_dir: "data/MSWEP_daily"
  era5_dir: "precip_data/era5_precipitation_project/predictors"
  output_base: "./test_experiment_runs"

training:
  epochs: 5            # Reduced from 20
  folds: 2             # Reduced from 10
  batch_size: 32
  optimizer: "adamw"
  learning_rate: 0.0001
  lr_scheduler: "cosineannealinglr"
  loss_type: "mae"
  dropout: 0.2
  weight_decay: 1e-4
  seasonal_plots: false  # Skip for speed
  skip_crps: true        # Skip CRPS calculation for speed

experiments:
  group_a:
    outside_weights: [0.2, 0.5]  # Only 2 weights instead of 3
  
  group_b:
    configurations:
      none:
        era5_variables: null
        era5_pressure_levels: null
        
      wind_only:
        era5_variables: ["u", "v"]
        era5_pressure_levels: [300, 850]  # Only 2 levels instead of 4

evaluation:
  crps_decomp_sample_days: 10  # Reduced sampling
  tolerance: 1e-6
  figure_dpi: 150  # Lower DPI for speed
EOF

echo "Created test configuration file"
echo ""

# Run the experiment pipeline with test config
echo "Starting experiment runs..."
./evaluation/run_experiments_advanced.sh --config evaluation/test_config.yaml

# Check results
if [ -f "evaluation/aggregated_metrics.csv" ]; then
    echo ""
    echo "Test completed successfully!"
    echo ""
    echo "Quick summary:"
    echo "--------------"
    
    # Show header and first few results
    head -n 5 evaluation/aggregated_metrics.csv | column -t -s,
    
    echo ""
    echo "Generated files:"
    echo "- Metrics: evaluation/aggregated_metrics.csv"
    echo "- Figures: evaluation/figures/"
    echo "- Experiment outputs: test_experiment_runs/"
    
    # Count generated figures
    NUM_FIGS=$(ls evaluation/figures/*.png 2>/dev/null | wc -l)
    echo ""
    echo "Generated ${NUM_FIGS} figures"
    
    # Show best configuration
    echo ""
    echo "Best configuration from test:"
    python -c "
import pandas as pd
df = pd.read_csv('evaluation/aggregated_metrics.csv')
df['mean_crps'] = pd.to_numeric(df['mean_crps'], errors='coerce')
if not df.empty and df['mean_crps'].notna().any():
    best_idx = df['mean_crps'].idxmin()
    row = df.loc[best_idx]
    print(f'  Run ID: {row[\"run_id\"]}')
    print(f'  Outside weight: {row.get(\"outside_weight\", \"N/A\")}')
    print(f'  ERA5 group: {row.get(\"era5_group\", \"N/A\")}')
    print(f'  Mean CRPS: {row[\"mean_crps\"]:.4f}')
"
else
    echo "Test failed - check logs for errors"
    echo ""
    echo "Common issues:"
    echo "1. Data not found at expected paths"
    echo "2. Missing dependencies (PyYAML, isodisreg, etc.)"
    echo "3. Insufficient memory/disk space"
    echo ""
    echo "Check logs in test_experiment_runs/ for details"
fi

echo ""
echo "To run full experiments, use:"
echo "  ./evaluation/run_experiments_and_evaluate.sh"

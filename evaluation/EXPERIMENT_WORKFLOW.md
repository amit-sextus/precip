# Complete Experiment and Evaluation Workflow

This document describes the end-to-end workflow for running precipitation forecasting experiments and generating publication-ready results.

## Overview

The workflow consists of three main phases:
1. **Experiment Execution**: Running model training with different configurations
2. **Batch Evaluation**: Computing metrics for all runs
3. **Results Generation**: Creating figures and tables for publication

## Quick Start

### Option 1: Run Everything (Recommended)

```bash
# Activate environment
conda activate precip_2.0

# Run all experiments and evaluation
./evaluation/run_experiments_and_evaluate.sh
```

This single command will:
- Run all Group A experiments (regional weighting sweep)
- Automatically determine the best outside_weight
- Run all Group B experiments (ERA5 feature groups)
- Generate all evaluation metrics
- Create publication figures
- Export LaTeX tables

### Option 2: Quick Test

To verify the system works with minimal resources:

```bash
./evaluation/quick_test_example.sh
```

This runs a reduced version (5 epochs, 2 folds) for testing.

## Detailed Workflow

### Phase 1: Experiment Groups

#### Group A: Regional Weighting Sweep
- **Purpose**: Find optimal outside_weight parameter
- **Configurations**: outside_weight ∈ {0.2, 0.5, 0.7}
- **Features**: Rainfall + seasonality only (no ERA5)

#### Group B: ERA5 Feature Comparison
- **Purpose**: Evaluate impact of atmospheric predictors
- **Configurations** (at best outside_weight):
  - `none`: No ERA5 features
  - `wind_only`: u,v at 300/500/700/850 hPa
  - `all`: Comprehensive (wind + humidity + surface)

### Phase 2: Training Commands

Example training command structure:
```bash
python models/mswep_unet_training.py \
    --data_dir data/MSWEP_daily \
    --output_dir experiment_runs/run_name \
    --epochs 20 \
    --folds 10 \
    --optimizer_type adamw \
    --use_regional_focus true \
    --outside_weight 0.5 \
    --era5_variables u,v \
    --era5_pressure_levels 300,500,700,850 \
    --era5_data_dir precip_data/era5_precipitation_project/predictors
```

### Phase 3: Evaluation Pipeline

The evaluation automatically:
1. Parses experiment manifest
2. Computes metrics for each run:
   - Daily CRPS time series
   - Seasonal aggregations
   - CORP decompositions
   - Skill scores vs MPC baseline
3. Performs acceptance tests
4. Generates consolidated CSV

### Phase 4: Results Generation

Output artifacts include:

#### Figures (300 DPI PNG)
- **Fig A**: Outside weight sweep (CRPS & BS vs weight)
- **Fig B**: ERA5 comparison (CRPS & CRPSS by group)
- **Fig C**: Seasonal CRPS bars (best config)
- **Fig D**: Seasonal Brier Score bars (best config)
- **Fig E**: MCB-DSC scatter plots

#### LaTeX Tables (booktabs format)
- Regional weighting results
- ERA5 comparison results
- Best model seasonal summary
- CORP decomposition details

## Advanced Usage

### Custom Configuration

Edit `evaluation/experiment_config.yaml`:

```yaml
training:
  epochs: 20      # Adjust for longer/shorter runs
  folds: 10       # Number of cross-validation folds
  
experiments:
  group_a:
    outside_weights: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  
  group_b:
    configurations:
      custom:
        era5_variables: ["u", "v", "t", "q"]
        era5_pressure_levels: [500, 850]
```

### Parallel Execution

Run experiments in parallel (requires GNU parallel):

```bash
./evaluation/run_experiments_advanced.sh --parallel --config my_config.yaml
```

### Resume Failed Runs

Continue from where you left off:

```bash
./evaluation/run_experiments_advanced.sh --resume
```

## Directory Structure

```
precipitation/
├── evaluation/
│   ├── run_experiments_and_evaluate.sh    # Main runner
│   ├── run_experiments_advanced.sh        # Advanced runner
│   ├── experiment_config.yaml             # Configuration
│   ├── quick_test_example.sh              # Test script
│   ├── make_results.py                    # Results generator
│   ├── batch_evaluate.py                  # Batch evaluator
│   ├── export_latex.py                    # LaTeX exporter
│   ├── acceptance_check.py                # Validation
│   ├── aggregated_metrics.csv             # [Generated] All metrics
│   ├── experiment_manifest.yaml           # [Generated] Run registry
│   ├── figures/                           # [Generated] All plots
│   └── tables/                            # [Generated] LaTeX tables
├── experiment_runs/                       # [Generated] Training outputs
├── models/
│   └── mswep_unet_training.py           # Training script
└── data/
    ├── MSWEP_daily/                      # Precipitation data
    └── era5_precipitation_project/       # ERA5 predictors
```

## Troubleshooting

### Common Issues

1. **Missing data**: Ensure MSWEP and ERA5 data paths are correct
2. **Memory errors**: Reduce batch_size or use fewer folds
3. **Import errors**: Install dependencies: `pip install -r requirements.txt`
4. **Parallel errors**: Install GNU parallel: `sudo apt-get install parallel`

### Checking Results

```bash
# View metrics summary
cat evaluation/aggregated_metrics.csv | column -t -s,

# Check acceptance tests
python -m evaluation.acceptance_check --csv evaluation/aggregated_metrics.csv

# View best configuration
grep -E 'run_id|mean_crps' evaluation/aggregated_metrics.csv | sort -t, -k2 -n
```

## Expected Outputs

After successful completion:

1. **Metrics CSV**: Complete comparison of all configurations
2. **Figures**: Publication-ready plots in `evaluation/figures/`
3. **Tables**: LaTeX tables in `evaluation/tables/`
4. **Run logs**: Detailed logs in `experiment_runs/`

## Citation

If using this workflow, please cite:
- MSWEP precipitation dataset
- ERA5 reanalysis data
- EasyUQ/isodisreg package

## Support

For issues or questions:
1. Check run logs in `experiment_runs/run_log.txt`
2. Verify data paths in configuration
3. Run acceptance checks for validation

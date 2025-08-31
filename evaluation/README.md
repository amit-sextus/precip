# Batch Evaluation System for MSWEP Precipitation Forecasting

This module provides a batch evaluation framework for running systematic evaluations across multiple model runs with different hyperparameters.

## Overview

The batch evaluation system:
- Parses a YAML manifest mapping run directories to metadata
- Evaluates each run using the Germany/coslat setup with MPC baseline
- Performs acceptance checks on metrics consistency
- Collects all outputs into a consolidated CSV for analysis

## Setup

1. Ensure your conda environment is activated:
   ```bash
   conda activate precip_2.0
   ```

2. Create/edit the manifest file `evaluation/run_manifest.yaml`:
   ```yaml
   runs:
     - run_id: "run_20250707_110229_fold9"
       path: "/home/batman/precipitation/test_output/run_20250707_110229/fold9"
       outside_weight: 0.2
       era5_group: "none"  # one of: none | wind_only | all
       note: "baseline: rainfall+seasonality only"
     - run_id: "run_20250708_084012_fold9" 
       path: "/home/batman/precipitation/test_output/run_20250708_084012/fold9"
       outside_weight: 0.5
       era5_group: "none"
   ```

## Usage

Basic evaluation:
```bash
python -m evaluation.batch_evaluate --manifest evaluation/run_manifest.yaml
```

With options:
```bash
# Force recomputation even if results exist
python -m evaluation.batch_evaluate --manifest evaluation/run_manifest.yaml --overwrite

# Generate seasonal bar plots for each run
python -m evaluation.batch_evaluate --manifest evaluation/run_manifest.yaml --export-per-run

# Specify custom output location
python -m evaluation.batch_evaluate --manifest evaluation/run_manifest.yaml --output results/metrics.csv
```

## Output

The system generates:

1. **Per-run outputs** (in each run directory):
   - `evaluation_metrics_germany.json` - Detailed metrics for the run
   - Publication-ready plots (if --export-per-run):
     - `seasonal_crps_{run_id}.png` - Seasonal CRPS bar chart with neutral styling
     - `seasonal_bs_{run_id}.png` - Seasonal Brier Score bar chart  
     - `bs_mcb_dsc_{run_id}.png` - MCB-DSC scatter plot for Brier Score decomposition
     - `crps_mcb_dsc_{run_id}.png` - MCB-DSC scatter plot for CRPS decomposition (if available)

2. **Consolidated output**:
   - `evaluation/aggregated_metrics.csv` (or custom path) with columns:
     - Run metadata: run_id, path, outside_weight, era5_group, year_eval
     - Overall metrics: mean_crps, mean_bs, crpss_mpc
     - Seasonal metrics: crps_{season}, bs_{season}, crpss_{season} for each season
     - Decomposition: CORP components for both CRPS and BS
     - Intensity bins: MAE and CRPS for precipitation intensity categories
     - Fixed metadata: mask="Germany", spatial_weight="coslat"

3. **Summary plots** (in output directory):
   - `outside_weight_sweep.png` - Shows impact of outside_weight on CRPS and BS (rainfall+seasonality only)
   - `era5_comparison.png` - Compares ERA5 feature groups (none, wind_only, all) at best outside_weight

## Plotting Features

All plots follow publication-ready standards:
- **Color scheme**: Neutral gray (#4c4c4c) with black edges for main bars
- **Title format**: All titles include "(Germany; cos(lat) area-weighted)"
- **Value labels**: Numeric values displayed on all bars (format: {value:.3f})
- **Grid**: Dotted light gray on y-axis only
- **MCB-DSC plots**: Include diagonal isopleths and special markers for MPC baseline and best run

## Acceptance Tests

The system performs several acceptance tests for each run:

1. **Overall consistency**: Verifies that overall mean CRPS equals the time-weighted average of seasonal means (tolerance: 1e-6)
2. **BS decomposition**: Checks BS ≈ MCB - DSC + UNC (tolerance: 1e-6)  
3. **CRPS decomposition**: Checks CRPS ≈ MCB - DSC + UNC (tolerance: 1e-6)
4. **Value sanity**: Ensures CRPS ≥ 0, 0 ≤ BS ≤ 1, CRPSS < 1

## Key Features

- **MPC Baseline**: Uses Monthly Probabilistic Climatology from training years only
- **CRPSS Calculation**: Skill score computed as (S_base - S_fcst) / S_base
- **Area Weighting**: Uses cos(latitude) weights normalized over Germany mask
- **Daily Aggregation**: Computes per-cell metrics daily, then spatially aggregates, then time-averages

## Troubleshooting

If you encounter import errors:
```bash
# Ensure all dependencies are installed
pip install PyYAML pandas numpy matplotlib tqdm isodisreg
```

If metrics files are missing:
- Check that the run directories contain required files:
  - `val_preds.npy`, `val_targets.npy` 
  - `train_preds_all.npy`, `train_targets_all.npy`
  - `val_times.npy`
  - `germany_mask.npy`

## Complete Results Pipeline

### 1. Generate All Results (One Command)

To reproduce all results section artifacts:

```bash
python -m evaluation.make_results --manifest evaluation/run_manifest.yaml
```

This will:
- Run batch evaluation on all runs
- Generate `aggregated_metrics.csv`
- Create all publication figures in `evaluation/figures/`:
  - Fig A: Outside weight sweep (CRPS & BS)
  - Fig B: ERA5 group comparison (CRPS & CRPSS)
  - Fig C: Seasonal CRPS (best config)
  - Fig D: Seasonal BS (best config)
  - Fig E: MCB-DSC scatter plots

### 2. Export LaTeX Tables

Generate publication-ready booktabs tables:

```bash
# Regional weighting table
python -m evaluation.export_latex --csv evaluation/aggregated_metrics.csv \
    --out evaluation/tables/regional_weighting.tex \
    --select 'era5_group=="none"'

# Best model seasonal summary
python -m evaluation.export_latex --csv evaluation/aggregated_metrics.csv \
    --out evaluation/tables/best_model.tex \
    --select 'run_id=="<BEST_RUN_ID>"'

# Decomposition table for specific run
python -m evaluation.export_latex --csv evaluation/aggregated_metrics.csv \
    --out evaluation/tables/decomposition.tex \
    --table-type decomposition \
    --run-id "<RUN_ID>"
```

### 3. Run Acceptance Checks

Verify all metrics satisfy required invariants:

```bash
python -m evaluation.acceptance_check --csv evaluation/aggregated_metrics.csv
```

This checks:
- Overall metrics = time-weighted seasonal averages (tolerance: 1e-6)
- CORP decomposition identities (BS/CRPS = MCB - DSC + UNC)
- All runs use Germany mask and cos(lat) weighting
- Value validity (CRPS ≥ 0, 0 ≤ BS ≤ 1, CRPSS < 1)

## Example Manifest

See `evaluation/examples/run_manifest_examples.yaml` for a complete example with:
- Group A: Regional weighting sweep (outside_weight ∈ {0.2, 0.5, 0.7})
- Group B: ERA5 feature groups (none, wind_only, all)

Key definitions:
- `wind_only`: u,v at 300/500/700/850 hPa with standard temporal lags
- `all`: Full comprehensive configuration (wind + humidity + surface variables)

## Automated Experiment Runner

### Basic Usage

Run all experiments from Groups A and B automatically:

```bash
./evaluation/run_experiments_and_evaluate.sh
```

This script will:
1. Run Group A experiments (regional weighting sweep)
2. Evaluate Group A to find the best outside_weight
3. Run Group B experiments (ERA5 groups) using the best weight
4. Generate all evaluation outputs automatically

### Advanced Usage

For more control, use the advanced script with configuration file:

```bash
# Run with custom configuration
./evaluation/run_experiments_advanced.sh --config evaluation/experiment_config.yaml

# Resume from previous runs (skip completed experiments)
./evaluation/run_experiments_advanced.sh --resume

# Run experiments in parallel (requires GNU parallel)
./evaluation/run_experiments_advanced.sh --parallel

# Run only specific group
./evaluation/run_experiments_advanced.sh --group A
./evaluation/run_experiments_advanced.sh --group B

# Skip evaluation phase (just run experiments)
./evaluation/run_experiments_advanced.sh --skip-evaluation
```

### Configuration

Edit `evaluation/experiment_config.yaml` to customize:
- Training parameters (epochs, batch size, optimizer, etc.)
- Group A outside weights
- Group B ERA5 configurations
- Data directories
- Test mode settings for quick runs

### Expected Directory Structure

```
precipitation/
├── data/MSWEP_daily/           # MSWEP precipitation data
├── precip_data/era5_precipitation_project/predictors/  # ERA5 data
├── models/mswep_unet_training.py
├── evaluation/
│   ├── run_experiments_and_evaluate.sh
│   ├── run_experiments_advanced.sh
│   ├── experiment_config.yaml
│   ├── make_results.py
│   ├── batch_evaluate.py
│   ├── export_latex.py
│   └── acceptance_check.py
```

## Notes

- The system reuses existing metrics if found (unless --overwrite is used)
- Intensity bin metrics are currently placeholders in the batch system
- CRPS decomposition is computed on a diagnostic subset (30 days) for efficiency
- All figures are saved at 300 DPI for publication quality
- LaTeX tables use booktabs formatting with proper decimal alignment
- Experiment runs are logged in `experiment_runs/run_log.txt` for tracking

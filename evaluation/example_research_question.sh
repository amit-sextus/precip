#!/bin/bash
# Example: Investigating the impact of wind predictors on summer precipitation
# This script demonstrates how to use the system for a specific research question

echo "Research Question: How do wind predictors affect summer precipitation forecasting?"
echo "============================================================================"
echo ""

# Activate environment (commented out - already activated in terminal)
# conda activate precip_2.0

# Create custom configuration for this research
cat > evaluation/wind_impact_config.yaml << EOF
# Configuration for wind impact study
data:
  mswep_dir: "data/MSWEP_daily"
  era5_dir: "precip_data/era5_precipitation_project/predictors"
  output_base: "./wind_impact_study"

training:
  epochs: 20
  folds: 10
  batch_size: 32
  optimizer: "adamw"
  learning_rate: 0.0001
  lr_scheduler: "cosineannealinglr"
  loss_type: "mae"
  dropout: 0.2
  weight_decay: 1e-4
  seasonal_plots: true
  skip_crps: false

experiments:
  # First, find optimal weight without ERA5
  group_a:
    outside_weights: [0.3, 0.5, 0.7]
  
  # Then test different wind configurations
  group_b:
    configurations:
      no_era5:
        era5_variables: null
        era5_pressure_levels: null
        
      wind_surface:
        # Only surface/near-surface winds
        era5_variables: ["u", "v"]
        era5_pressure_levels: [850]
        
      wind_mid:
        # Mid-troposphere winds
        era5_variables: ["u", "v"]
        era5_pressure_levels: [500, 700]
        
      wind_upper:
        # Upper troposphere winds (jet stream level)
        era5_variables: ["u", "v"]
        era5_pressure_levels: [300]
        
      wind_all:
        # All levels
        era5_variables: ["u", "v"]
        era5_pressure_levels: [300, 500, 700, 850]

evaluation:
  crps_decomp_sample_days: 30
  tolerance: 1e-6
  figure_dpi: 300
EOF

echo "Step 1: Running experiments with different wind configurations..."
./evaluation/run_experiments_advanced.sh --config evaluation/wind_impact_config.yaml

# Custom analysis focusing on summer (JJA) performance
echo ""
echo "Step 2: Analyzing summer-specific performance..."

python << EOF
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
df = pd.read_csv('evaluation/aggregated_metrics.csv')

# Convert numeric columns
for col in ['mean_crps', 'crps_jja', 'bs_jja', 'crpss_mpc', 'crpss_jja']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter for wind study configurations
wind_configs = ['no_era5', 'wind_surface', 'wind_mid', 'wind_upper', 'wind_all']
study_df = df[df['era5_group'].isin(wind_configs)]

if len(study_df) > 0:
    # Create custom analysis plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Summer CRPS by configuration
    configs = study_df['era5_group'].values
    summer_crps = study_df['crps_jja'].values
    
    x = np.arange(len(configs))
    bars = ax1.bar(x, summer_crps, color='#4c4c4c', edgecolor='black')
    
    # Add value labels
    for i, (config, val) in enumerate(zip(configs, summer_crps)):
        if np.isfinite(val):
            ax1.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace('_', ' ').title() for c in configs], rotation=45, ha='right')
    ax1.set_ylabel('Summer (JJA) CRPS')
    ax1.set_title('Impact of Wind Predictors on Summer Precipitation CRPS')
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    
    # Plot 2: Improvement over no ERA5
    baseline_crps = study_df[study_df['era5_group'] == 'no_era5']['crps_jja'].values[0]
    improvements = 100 * (baseline_crps - summer_crps) / baseline_crps
    
    colors = ['red' if imp < 0 else '#228B22' for imp in improvements]
    bars2 = ax2.bar(x, improvements, color=colors, edgecolor='black')
    
    # Add percentage labels
    for i, imp in enumerate(improvements):
        if np.isfinite(imp):
            ax2.text(i, imp + 0.5 if imp > 0 else imp - 0.5, 
                    f'{imp:+.1f}%', ha='center', 
                    va='bottom' if imp > 0 else 'top')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace('_', ' ').title() for c in configs], rotation=45, ha='right')
    ax2.set_ylabel('Improvement over No ERA5 (%)')
    ax2.set_title('Summer CRPS Improvement with Wind Predictors')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('evaluation/figures/wind_impact_summer_analysis.png', dpi=300)
    plt.close()
    
    # Print summary
    print("\nSummer (JJA) Performance Summary:")
    print("================================")
    for _, row in study_df.iterrows():
        config = row['era5_group']
        crps = row['crps_jja']
        skill = row.get('crpss_jja', np.nan)
        print(f"{config:15s}: CRPS={crps:.4f}, Skill Score={skill:.4f}")
    
    # Best configuration for summer
    best_idx = study_df['crps_jja'].idxmin()
    best_config = study_df.loc[best_idx]
    print(f"\nBest configuration for summer: {best_config['era5_group']}")
    print(f"Summer CRPS: {best_config['crps_jja']:.4f}")
    print(f"Improvement: {improvements[list(configs).index(best_config['era5_group'])]:.1f}%")

else:
    print("No results found for wind study configurations")
EOF

# Generate specific LaTeX table for this study
echo ""
echo "Step 3: Creating publication table..."

python -m evaluation.export_latex \
    --csv evaluation/aggregated_metrics.csv \
    --out evaluation/tables/wind_impact_study.tex \
    --select 'era5_group in ["no_era5", "wind_surface", "wind_mid", "wind_upper", "wind_all"]' \
    --caption "Impact of wind predictors at different pressure levels on summer precipitation forecasting"

echo ""
echo "Research Study Complete!"
echo "======================="
echo ""
echo "Key outputs:"
echo "  - Custom analysis plot: evaluation/figures/wind_impact_summer_analysis.png"
echo "  - LaTeX table: evaluation/tables/wind_impact_study.tex"
echo "  - Full metrics: evaluation/aggregated_metrics.csv"
echo ""
echo "Findings summary saved above. Use these results to answer:"
echo "1. Which pressure levels provide the most value for summer precipitation?"
echo "2. Is there a benefit to including all levels vs. selective levels?"
echo "3. How does the improvement vary by season (check other seasonal columns)?"

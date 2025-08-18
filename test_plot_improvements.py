#!/usr/bin/env python3
"""
Test script to regenerate a seasonal sample plot with the improved styling.
This will help verify that the lat/lon intervals are larger and borders look cleaner.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mswep_evaluation import plot_seasonal_samples
import numpy as np

# Set up paths
base_dir = "test_output/run_20250706_170705_era5_u_v_300_500+_sl_msl_tcwv_t2m+/fold1"

# Test with a specific time index
print("Regenerating seasonal sample plots with improved styling...")
print("- Larger lat/lon intervals (10° for focused, 30°x20° for full)")
print("- Cleaner coastlines and borders")
print("- Consistent gridlines across all panels")

try:
    # Generate plots for just one sample to test
    plot_seasonal_samples(
        fold_dir=base_dir,
        fold_num="1",
        num_samples_per_season=1,  # Just one sample per season for testing
        time_indices=[53]  # Specific index from the example
    )
    print("\nPlot regeneration complete. Check the seasonal_sample_plots directory.")
except Exception as e:
    print(f"Error generating plot: {e}")
    import traceback
    traceback.print_exc() 
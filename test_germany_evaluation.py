#!/usr/bin/env python
"""
Test script to verify Germany-specific evaluation functionality.
This script tests that the regional focus on Germany is properly implemented
in the evaluation workflow.
"""

import numpy as np
import os
import tempfile
import shutil

def create_test_data(output_dir, num_folds=2, grid_lat=41, grid_lon=121, num_samples=100):
    """Create synthetic test data with Germany mask for testing."""
    
    # Create run directory
    run_dir = os.path.join(output_dir, "run_test")
    os.makedirs(run_dir, exist_ok=True)
    
    # Create Germany mask (simulating the region)
    germany_mask = np.zeros((grid_lat, grid_lon), dtype=bool)
    # Define Germany's grid boundaries (same as in mswep_unet.py)
    lat_min, lat_max = 17, 25  # Germany latitude range
    lon_min, lon_max = 75, 85  # Germany longitude range
    germany_mask[lat_min:lat_max+1, lon_min:lon_max+1] = True
    
    print(f"Created Germany mask with {np.sum(germany_mask)} cells marked as Germany")
    print(f"Germany region: lat[{lat_min}:{lat_max}], lon[{lon_min}:{lon_max}]")
    
    for fold in range(num_folds):
        fold_dir = os.path.join(run_dir, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Create synthetic predictions and targets
        # Add some spatial structure to make it more realistic
        np.random.seed(42 + fold)
        
        # Training data
        train_preds = np.random.gamma(2, 2, (num_samples * 3, grid_lat, grid_lon))
        train_targets = train_preds + np.random.normal(0, 0.5, train_preds.shape)
        train_targets = np.maximum(train_targets, 0)  # Ensure non-negative
        
        # Validation data (smaller set)
        val_preds = np.random.gamma(2, 2, (num_samples, grid_lat, grid_lon))
        val_targets = val_preds + np.random.normal(0, 0.5, val_preds.shape)
        val_targets = np.maximum(val_targets, 0)  # Ensure non-negative
        
        # Make Germany region have higher values to test regional differences
        train_preds[:, germany_mask] *= 1.5
        train_targets[:, germany_mask] *= 1.5
        val_preds[:, germany_mask] *= 1.5
        val_targets[:, germany_mask] *= 1.5
        
        # Save data
        np.save(os.path.join(fold_dir, "train_preds_all.npy"), train_preds)
        np.save(os.path.join(fold_dir, "train_targets_all.npy"), train_targets)
        np.save(os.path.join(fold_dir, "val_preds.npy"), val_preds)
        np.save(os.path.join(fold_dir, "val_targets.npy"), val_targets)
        np.save(os.path.join(fold_dir, "germany_mask.npy"), germany_mask)
        
        # Create synthetic timestamps
        import pandas as pd
        val_times = pd.date_range(start='2019-01-01', periods=num_samples, freq='D')
        np.save(os.path.join(fold_dir, "val_times.npy"), val_times.values)
        
        print(f"Created test data for fold {fold}")
    
    return run_dir, germany_mask

def test_calculate_final_crps():
    """Test the calculate_final_crps function with Germany mask."""
    # Import the function
    try:
        from models.mswep_unet_training import calculate_final_crps
    except ImportError:
        print("Error: Cannot import calculate_final_crps. Make sure the module path is correct.")
        return False
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nTesting in temporary directory: {temp_dir}")
        
        # Create test data
        run_dir, germany_mask = create_test_data(temp_dir, num_folds=2)
        
        # Run the evaluation
        print("\nRunning calculate_final_crps...")
        try:
            calculate_final_crps(temp_dir, num_folds=2, skip_crps=False, args=None)
            
            # Check that results were created
            final_eval_dir = os.path.join(run_dir, "final_evaluation")
            if not os.path.exists(final_eval_dir):
                print("Error: final_evaluation directory was not created")
                return False
            
            # Check for expected output files
            expected_files = [
                "combined_val_preds.npy",
                "combined_val_targets.npy", 
                "combined_train_preds.npy",
                "combined_train_targets.npy",
                "germany_mask.npy",  # Should be germany_mask not just mask
                "deterministic_metrics.npy",
                "cell_metrics.npy",
                "comprehensive_results.json"
            ]
            
            for file in expected_files:
                filepath = os.path.join(final_eval_dir, file)
                if not os.path.exists(filepath):
                    print(f"Error: Expected file not found: {file}")
                    return False
                else:
                    print(f"✓ Found: {file}")
            
            # Load and check the results
            import json
            with open(os.path.join(final_eval_dir, "comprehensive_results.json"), 'r') as f:
                results = json.load(f)
            
            # Check that results contain expected keys
            expected_keys = [
                'overall_mean_crps',
                'overall_mean_bs_0.2mm',
                'deterministic_mae',
                'deterministic_rmse',
                'num_valid_cells',
                'evaluation_region',
                'intensity_bin_metrics',
                'seasonal_metrics'
            ]
            
            for key in expected_keys:
                if key not in results:
                    print(f"Error: Expected key '{key}' not found in results")
                    return False
                else:
                    print(f"✓ Found key: {key}")
            
            # Verify that evaluation_region is set to 'Germany'
            if results.get('evaluation_region') != 'Germany':
                print(f"Error: evaluation_region should be 'Germany' but got '{results.get('evaluation_region')}'")
                return False
            else:
                print("✓ evaluation_region correctly set to 'Germany'")
            
            # Verify num_valid_cells matches Germany mask
            expected_cells = np.sum(germany_mask)
            if results['num_valid_cells'] != expected_cells:
                print(f"Error: num_valid_cells ({results['num_valid_cells']}) doesn't match Germany mask cells ({expected_cells})")
                return False
            else:
                print(f"✓ num_valid_cells correctly matches Germany region: {expected_cells}")
            
            print("\n✅ All tests passed!")
            return True
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_mask_handling():
    """Test that masks are properly handled in the evaluation workflow."""
    print("\n=== Testing Mask Handling ===")
    
    # Create a simple mask
    grid_lat, grid_lon = 41, 121
    germany_mask = np.zeros((grid_lat, grid_lon), dtype=bool)
    
    # Define Germany's grid boundaries
    lat_min, lat_max = 17, 25
    lon_min, lon_max = 75, 85
    germany_mask[lat_min:lat_max+1, lon_min:lon_max+1] = True
    
    total_cells = grid_lat * grid_lon
    germany_cells = np.sum(germany_mask)
    outside_cells = total_cells - germany_cells
    
    print(f"Total grid cells: {total_cells}")
    print(f"Germany cells: {germany_cells}")
    print(f"Outside Germany cells: {outside_cells}")
    print(f"Germany percentage: {100 * germany_cells / total_cells:.1f}%")
    
    # Verify mask dimensions match expected grid
    assert germany_mask.shape == (grid_lat, grid_lon), f"Mask shape mismatch: {germany_mask.shape}"
    print("✓ Mask dimensions correct")
    
    # Verify Germany region is correctly defined
    assert germany_cells == (lat_max - lat_min + 1) * (lon_max - lon_min + 1), "Germany cell count mismatch"
    print("✓ Germany region correctly defined")
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("Testing Germany-Specific Evaluation Implementation")
    print("="*60)
    
    # Test 1: Mask handling
    if not test_mask_handling():
        print("\n❌ Mask handling test failed")
        exit(1)
    
    # Test 2: Full evaluation workflow
    if not test_calculate_final_crps():
        print("\n❌ calculate_final_crps test failed")
        exit(1)
    
    print("\n" + "="*60)
    print("✅ All tests completed successfully!")
    print("The Germany-specific evaluation is working correctly.")
    print("="*60) 
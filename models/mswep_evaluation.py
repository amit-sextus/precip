import numpy as np
import pandas as pd
import os
import xarray as xr
import json
import matplotlib.pyplot as plt
import matplotlib.colors as colors  # Add colors import for PowerNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from isodisreg import idr
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
import argparse
import calendar

# Import verification helpers for area weighting and diagnostics
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.verification_helpers import (
    coslat_weights, spatial_weighted_mean, month_to_season,
    skill_score, corp_bs_decomposition, corp_crps_decomposition_from_cdf,
    build_mpc_climatology, crps_sample_distribution, mpc_pop
)

# Assumed global grid definition (should match the data being evaluated)
# These are based on the prompt's suggested arrays and typical MSWEP-like grids.
# Users should verify these match their specific data dimensions and extents.
EXPECTED_GRID_LON_DIM = 121  # Example: Number of longitude points
EXPECTED_GRID_LAT_DIM = 41   # Example: Number of latitude points
DEFAULT_LONGITUDES = np.arange(-70, -70 + EXPECTED_GRID_LON_DIM * 1, 1)  # Lon from -70 to 50
DEFAULT_LATITUDES = np.arange(30, 30 + EXPECTED_GRID_LAT_DIM * 1, 1)    # Lat from 30 to 70
XINC = 1.0 # longitude resolution
YINC = 1.0 # latitude resolution


# Germany bounding box for plotting (approximate, in degrees)
GERMANY_PLOT_LON_MIN = 5.0
GERMANY_PLOT_LON_MAX = 15.5
GERMANY_PLOT_LAT_MIN = 47.0
GERMANY_PLOT_LAT_MAX = 55.5

# Germany bounding box (grid indices, assuming they match DEFAULT_LONGITUDES/LATITUDES mapping)
# These were lat_min, lat_max = 17, 25 and lon_min, lon_max = 75, 85 in plot_sample
# latitudes[17] = 30+17 = 47
# latitudes[25] = 30+25 = 55
# longitudes[75] = -70+75 = 5
# longitudes[85] = -70+85 = 15
GERMANY_BOX_GRID_LON_INDICES = (75, 85) # Corresponds to ~5E to 15E
GERMANY_BOX_GRID_LAT_INDICES = (17, 25) # Corresponds to ~47N to 55N


def calculate_crps_idr(val_preds, val_target, train_preds, train_target, mask, lat, lon):
    """
    Calculate the CRPS for a given grid cell using IDR (EasyUQ).

    Parameters
    ----------
    val_preds : np.ndarray
        Array of validation predictions with shape [n_val, grid_lat, grid_lon].
    val_target : np.ndarray
        Array of validation observations with shape [n_val, grid_lat, grid_lon].
    train_preds : np.ndarray
        Array of training predictions with shape [n_train, grid_lat, grid_lon].
    train_target : np.ndarray
        Array of training observations with shape [n_train, grid_lat, grid_lon].
    mask : np.ndarray
        Binary mask of shape [grid_lat, grid_lon] indicating valid grid points.
    lat : int
        The latitude index of the grid cell.
    lon : int
        The longitude index of the grid cell.

    Returns
    -------
    crps_per_grid : float or None
        The averaged CRPS for the grid cell if the mask is True; otherwise, None.
    """
    if mask[lat, lon]:
        # Create an IDR model at this grid cell based on training data
        idr_per_grid = idr(y=train_target[:, lat, lon],
                           X=pd.DataFrame(train_preds[:, lat, lon]))
        # Predict the distribution for validation predictions
        val_dist_pred = idr_per_grid.predict(pd.DataFrame(val_preds[:, lat, lon]))
        crps_per_grid = np.mean(val_dist_pred.crps(val_target[:, lat, lon]))
        return crps_per_grid
    return None

def get_mean_masked_crps(experiment_id):
    """
    Retrieve and aggregate TensorBoard scalar logs for masked CRPS.

    This function connects to TensorBoard using an experiment id (as used
    in the original check_logs.ipynb) and aggregates the 'val_metrics/masked_crps'
    across runs/experiments.

    Parameters
    ----------
    experiment_id : str
        The TensorBoard experiment id.

    Returns:
        mean_crps_vals : pd.DataFrame
            A DataFrame with the mean and standard deviation of CRPS values grouped by experiment.
    """
    # Import tensorboard API (make sure tensorboard is installed in your environment)
    import tensorboard as tb

    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    # Extract experiment name from run string (assuming run names are formatted appropriately)
    df['experiment'] = df['run'].str.split('/').str[0]
    # Select nonzero masked CRPS values
    crps_df = df[(df['tag'] == 'val_metrics/masked_crps') & (df['value'] != 0.0)]
    # Group by experiment and compute mean and standard deviation
    mean_crps_vals = crps_df.groupby('experiment').mean()
    mean_crps_vals['mean_crps'] = mean_crps_vals['value']
    mean_crps_vals['std_crps'] = crps_df.groupby('experiment').std()['value']
    return mean_crps_vals

def load_evaluation_data(base_output_dir: str, fold: int):
    """
    Loads predictions, targets, timestamps, and mask for a specific fold.
    Now includes stricter checks and avoids generating synthetic timestamps.

    Args:
        base_output_dir (str): The base directory containing fold subdirectories.
        fold (int): The fold number to load data for.

    Returns:
        tuple: (val_preds, val_targets, train_preds, train_targets, val_times, mask)
               Returns (None, None, None, None, None, None) if any essential file is missing or inconsistent.
    """
    fold_dir = os.path.join(base_output_dir, f"fold{fold}")
    print(f"Loading evaluation data from: {fold_dir}")

    def load_npy(filename):
        filepath = os.path.join(fold_dir, filename)
        if os.path.exists(filepath):
            try:
                return np.load(filepath)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                return None
        else:
            # Make file not found error more prominent
            print(f"ERROR: Required file not found - {filepath}")
            return None

    val_preds_file = "val_preds.npy"
    val_targets_file = "val_targets.npy"
    train_preds_file = "train_preds_all.npy"
    train_targets_file = "train_targets_all.npy"
    val_times_file = "val_times.npy"
    mask_file = "germany_mask.npy"

    val_preds = load_npy(val_preds_file)
    val_targets = load_npy(val_targets_file)
    train_preds = load_npy(train_preds_file)
    train_targets = load_npy(train_targets_file)
    val_times = load_npy(val_times_file)
    mask = load_npy(mask_file)
    
    # For mask loading
    mask_file = "germany_mask.npy" # This remains the same
    # Define the full path that load_npy will attempt to use, for logging purposes
    filepath_for_mask_logging = os.path.join(fold_dir, mask_file) 
    
    mask = load_npy(mask_file) # load_npy internally uses os.path.join(fold_dir, filename)

    # --- Strict Checks ---
    if val_preds is None or val_targets is None or train_preds is None or train_targets is None or val_times is None:
        print("Error: Essential data arrays (preds, targets, or times) could not be loaded. Aborting evaluation for this fold.")
        return None, None, None, None, None, None

    # 2. Check if mask was loaded (generate default if missing, but warn)
    # 2. Check if mask was loaded
    if mask is not None:
        # Use the correctly defined path for the print statement
        print(f"Loaded mask from {filepath_for_mask_logging}. Shape: {mask.shape}, Number of True values: {np.sum(mask)}") 
    else:
        # Use the correctly defined path for the warning
        print(f"Warning: Mask file '{filepath_for_mask_logging}' not found in fold directory.") 
        if val_targets is not None: # Ensure val_targets exists before accessing shape
            print("Creating default mask (all cells valid).")
            mask = np.ones((val_targets.shape[1], val_targets.shape[2]), dtype=bool)
            print(f"Created default mask with shape {mask.shape}, Number of True values: {np.sum(mask)}")
        else:
            print(f"ERROR: Mask file '{filepath_for_mask_logging}' not found AND val_targets is None. Cannot create default mask of correct shape.")
            # Consider returning None for mask or raising error if mask is absolutely critical

    # 3. Check consistency between validation data and timestamps
    num_val_samples_targets = val_targets.shape[0]
    num_val_samples_preds = val_preds.shape[0]
    num_timestamps = len(val_times)

    if not (num_val_samples_targets == num_val_samples_preds == num_timestamps):
        print(f"ERROR: Mismatch in validation data lengths!")
        print(f"  Targets: {num_val_samples_targets}, Predictions: {num_val_samples_preds}, Timestamps: {num_timestamps}")
        print("  Cannot proceed with evaluation. Check saved files.")
        return None, None, None, None, None, None

    # 4. (Optional but recommended) Check if validation length matches expected year length
    try:
        # Convert val_times to pandas DatetimeIndex for easier year extraction
        val_times_pd = pd.to_datetime(val_times)
        validation_year = val_times_pd.year[0]  # Assume first timestamp represents the year
        expected_days = 366 if calendar.isleap(validation_year) else 365
        # The number of SAMPLES should be expected_days - 3 due to lag
        expected_samples = expected_days - 3
        if num_timestamps != expected_samples:
            print(f"Warning: Loaded validation data ({num_timestamps} samples/timestamps) "
                  f"does not match expected samples ({expected_samples}) for year {validation_year}. "
                  f"Data might be incomplete or inconsistent with lag calculation.")
            # Continue, but be aware results might be based on partial year
    except Exception as e:
        print(f"Warning: Could not perform year length validation on timestamps: {e}")
        val_times_pd = pd.to_datetime(val_times)  # Still try to convert for return type consistency

    # --- End Strict Checks ---

    print(f"Loaded val_preds shape: {val_preds.shape}")
    print(f"Loaded val_targets shape: {val_targets.shape}")
    print(f"Loaded train_preds shape: {train_preds.shape}")
    print(f"Loaded train_targets shape: {train_targets.shape}")
    print(f"Loaded val_times shape: {val_times.shape} ({len(val_times_pd)} timestamps)")
    print(f"Loaded mask shape: {mask.shape}")
    
    # Verify grid dimensions against default lat/lon arrays if possible
    if val_targets.shape[1] != EXPECTED_GRID_LAT_DIM or val_targets.shape[2] != EXPECTED_GRID_LON_DIM:
        print(f"Warning: Loaded data grid dimensions ({val_targets.shape[1]}x{val_targets.shape[2]}) "
              f"do not match DEFAULT_LATITUDES/LONGITUDES dimensions ({EXPECTED_GRID_LAT_DIM}x{EXPECTED_GRID_LON_DIM}). "
              "Geographical plotting may be incorrect if these globals are not adjusted.")


    # Return the loaded (and validated) data
    return val_preds, val_targets, train_preds, train_targets, val_times_pd, mask  # Return pandas Timestamps

def apply_easyuq_per_cell(train_preds_cell: np.ndarray,
                          train_target_cell: np.ndarray,
                          eval_preds_cell: np.ndarray):
    """
    Applies EasyUQ/IDR calibration for a single grid cell.

    Fits the IDR model using training data and predicts distributions
    for evaluation data points.

    Args:
        train_preds_cell (np.ndarray): 1D array of deterministic training forecasts for the cell.
        train_target_cell (np.ndarray): 1D array of training observations for the cell.
        eval_preds_cell (np.ndarray): 1D array of deterministic evaluation forecasts for the cell.

    Returns:
        idrpredict object or None: The predicted distributions object from isodisreg.idr.predict(),
                                   or None if fitting/prediction fails.
    """
    if train_preds_cell is None or train_target_cell is None or eval_preds_cell is None:
        print("Warning: Missing input data for EasyUQ application.")
        return None
    if len(train_preds_cell) != len(train_target_cell):
        print(f"Warning: Training predictions ({len(train_preds_cell)}) and targets ({len(train_target_cell)}) length mismatch.")
        return None
    if len(train_preds_cell) == 0:
        print("Warning: Empty training data for EasyUQ application.")
        return None

    try:
        # Ensure inputs are pandas DataFrames/Series as expected by isodisreg
        train_preds_df = pd.DataFrame(train_preds_cell)
        eval_preds_df = pd.DataFrame(eval_preds_cell)

        # Fit the IDR model using training data
        # 'y' should be the observed target values
        # 'X' should be the corresponding deterministic forecasts
        idr_model = idr(y=train_target_cell, X=train_preds_df)

        # Predict distributions for the evaluation forecasts
        predicted_distributions = idr_model.predict(eval_preds_df)

        return predicted_distributions

    except Exception as e:
        print(f"Error during EasyUQ/IDR fitting or prediction: {e}")
        # Optionally add more specific error handling or logging
        # Consider cases like zero variance in inputs/outputs if they cause issues
        # traceback.print_exc() # Uncomment for detailed debugging
        return None


def create_lat_lon_2d(lats_1d, lons_1d):
    """
    Create 2D latitude and longitude arrays from 1D coordinate arrays.
    
    Parameters
    ----------
    lats_1d : np.ndarray
        1D array of latitude values.
    lons_1d : np.ndarray
        1D array of longitude values.
        
    Returns
    -------
    lat_2d, lon_2d : tuple of np.ndarray
        2D arrays where lat_2d[i,j] is the latitude at grid point (i,j).
    """
    lon_2d, lat_2d = np.meshgrid(lons_1d, lats_1d)
    return lat_2d, lon_2d


def compute_idr_crps_timeseries(idr_models_by_cell, det_preds_test, obs_test, germany_mask, lat_2d):
    """
    Compute daily Germany-mean CRPS time series using existing IDR workflow.
    
    This function uses the IDR objects' own .crps() method to compute CRPS values,
    then applies area-weighted spatial averaging over Germany for each day.
    
    Parameters
    ----------
    idr_models_by_cell : dict
        Dictionary mapping (lat_idx, lon_idx) to fitted IDR models.
        These must be pre-fitted on training data only.
    det_preds_test : np.ndarray
        Array [T, Ny, Nx] of deterministic predictions for test period.
    obs_test : np.ndarray
        Array [T, Ny, Nx] of MSWEP observations for test period.
    germany_mask : np.ndarray
        Boolean array [Ny, Nx] where True indicates cells within Germany.
    lat_2d : np.ndarray
        2D array [Ny, Nx] of latitude values in degrees for area weighting.
        
    Returns
    -------
    np.ndarray
        Array of length T containing area-weighted Germany-mean CRPS for each day.
        Days where no valid CRPS could be computed will have NaN.
    """
    T, Ny, Nx = obs_test.shape
    daily_regional_crps = np.full(T, np.nan)
    
    # Pre-compute area weights for Germany
    area_weights = coslat_weights(lat_2d, germany_mask)
    
    # Pre-generate predictions and compute CRPS for all cells once
    crps_by_cell = {}
    for (i, j), idr_model in idr_models_by_cell.items():
        if idr_model is None or not germany_mask[i, j]:
            continue
        try:
            # Get all predictions for this cell
            det_preds_cell = det_preds_test[:, i, j]
            obs_cell = obs_test[:, i, j]
            
            # Skip if all data is NaN
            if np.all(np.isnan(det_preds_cell)) or np.all(np.isnan(obs_cell)):
                continue
            
            # Create predictions
            preds_df = pd.DataFrame(det_preds_cell)
            predicted_dist = idr_model.predict(preds_df)
            
            # Compute CRPS for all time points
            crps_values = predicted_dist.crps(obs_cell)
            if isinstance(crps_values, list):
                crps_values = np.array(crps_values)
            
            crps_by_cell[(i, j)] = crps_values
            
        except Exception as e:
            # Skip cells with errors
            crps_by_cell[(i, j)] = None
    
    # Now compute daily averages
    for t in range(T):
        # Initialize CRPS map for this day
        crps_map = np.full((Ny, Nx), np.nan)
        
        for i in range(Ny):
            for j in range(Nx):
                # Skip if not in Germany or no CRPS for this cell
                if not germany_mask[i, j] or (i, j) not in crps_by_cell:
                    continue
                
                crps_values = crps_by_cell[(i, j)]
                if crps_values is None or len(crps_values) <= t:
                    continue
                
                # Get CRPS for this time point
                crps_value = crps_values[t]
                if np.isfinite(crps_value):
                    crps_map[i, j] = crps_value
        
        # Compute area-weighted mean for this day
        if np.any(np.isfinite(crps_map)):
            daily_regional_crps[t] = spatial_weighted_mean(crps_map, area_weights)
    
    return daily_regional_crps


def compute_brier_score_timeseries(idr_models_by_cell, det_preds_test, obs_test, germany_mask, lat_2d, threshold=0.2):
    """
    Compute daily Germany-mean Brier Score time series using IDR predictions.
    
    This function extracts PoP from IDR CDFs and computes area-weighted Brier Scores.
    
    Parameters
    ----------
    idr_models_by_cell : dict
        Dictionary mapping (lat_idx, lon_idx) to fitted IDR models.
    det_preds_test : np.ndarray
        Array [T, Ny, Nx] of deterministic predictions for test period.
    obs_test : np.ndarray
        Array [T, Ny, Nx] of MSWEP observations for test period.
    germany_mask : np.ndarray
        Boolean array [Ny, Nx] where True indicates cells within Germany.
    lat_2d : np.ndarray
        2D array [Ny, Nx] of latitude values in degrees for area weighting.
    threshold : float
        Precipitation threshold in mm (default: 0.2).
        
    Returns
    -------
    np.ndarray, list, list
        daily_bs: Array of length T containing area-weighted Germany-mean BS for each day.
        all_probs: List of all probability forecasts (for decomposition).
        all_obs: List of all binary observations (for decomposition).
    """
    T, Ny, Nx = obs_test.shape
    daily_regional_bs = np.full(T, np.nan)
    all_probs_germany = []
    all_obs_germany = []
    
    # Pre-compute area weights for Germany
    area_weights = coslat_weights(lat_2d, germany_mask)
    
    # Pre-generate predictions and compute CDFs for all cells once
    cdf_by_cell = {}
    for (i, j), idr_model in idr_models_by_cell.items():
        if idr_model is None or not germany_mask[i, j]:
            continue
        try:
            # Get all predictions for this cell
            det_preds_cell = det_preds_test[:, i, j]
            
            # Skip if all data is NaN
            if np.all(np.isnan(det_preds_cell)):
                continue
            
            # Create predictions
            preds_df = pd.DataFrame(det_preds_cell)
            predicted_dist = idr_model.predict(preds_df)
            
            # Compute CDF at threshold for all time points
            cdf_values = predicted_dist.cdf(thresholds=np.array([threshold]))
            if cdf_values.ndim == 2:
                cdf_values = cdf_values[:, 0]  # Extract first column if 2D
            
            cdf_by_cell[(i, j)] = cdf_values
            
        except Exception as e:
            # Skip cells with errors
            cdf_by_cell[(i, j)] = None
    
    # Now compute daily Brier Scores
    for t in range(T):
        # Initialize BS map for this day
        bs_map = np.full((Ny, Nx), np.nan)
        
        for i in range(Ny):
            for j in range(Nx):
                # Skip if not in Germany or no CDF for this cell
                if not germany_mask[i, j] or (i, j) not in cdf_by_cell:
                    continue
                
                cdf_values = cdf_by_cell[(i, j)]
                if cdf_values is None or len(cdf_values) <= t:
                    continue
                
                try:
                    # Get observation
                    obs_scalar = obs_test[t, i, j]
                    if np.isnan(obs_scalar):
                        continue
                    
                    # Get CDF value at threshold for this time
                    prob_le_threshold = cdf_values[t]
                    
                    # Compute PoP = 1 - F(threshold)
                    pop = 1.0 - prob_le_threshold
                    pop = np.clip(pop, 0.0, 1.0)
                    
                    # Binary observation
                    binary_obs = float(obs_scalar > threshold)
                    
                    # Brier Score
                    bs_value = (pop - binary_obs) ** 2
                    
                    if np.isfinite(bs_value):
                        bs_map[i, j] = bs_value
                        all_probs_germany.append(pop)
                        all_obs_germany.append(binary_obs)
                        
                except Exception as e:
                    # Silently skip cells with errors
                    pass
        
        # Compute area-weighted mean for this day
        if np.any(np.isfinite(bs_map)):
            daily_regional_bs[t] = spatial_weighted_mean(bs_map, area_weights)
    
    return daily_regional_bs, all_probs_germany, all_obs_germany


def calculate_stratified_crps(predicted_distributions, val_target_cell: np.ndarray, bins=None):
    """
    Calculates CRPS for a cell, stratified by precipitation intensity.
    
    Args:
        predicted_distributions: The output object from apply_easyuq_per_cell.
                                 Expected to have a .crps() method.
        val_target_cell (np.ndarray): 1D array of validation observations for the cell.
        bins (list): List of precipitation thresholds defining intensity bins.
                     If None, default bins are used.
                     
    Returns:
        dict: Dictionary with bin names as keys and mean CRPS as values.
    """
    if predicted_distributions is None or val_target_cell is None:
        return None
        
    if not hasattr(predicted_distributions, 'crps'):
        return None
        
    # Define intensity bins if not provided
    if bins is None:
        bins = [0, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, float('inf')]
        
    bin_names = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" if bins[i+1] != float('inf') else f">{bins[i]:.1f}" 
                for i in range(len(bins)-1)]
    
    try:
        # Calculate CRPS for all samples
        crps_values = predicted_distributions.crps(val_target_cell)
        
        # Ensure crps_values is an array with the right shape
        if isinstance(crps_values, list):
            crps_values = np.array(crps_values)
            
        # Handle potential shape issues
        if len(crps_values) != len(val_target_cell):
            print(f"Warning: CRPS values length ({len(crps_values)}) doesn't match targets ({len(val_target_cell)})")
            return None
            
        # Calculate stratified CRPS
        results = {}
        for i, bin_name in enumerate(bin_names):
            bin_mask = (val_target_cell >= bins[i]) & (val_target_cell < bins[i+1] if bins[i+1] != float('inf') else True)
            if np.any(bin_mask):
                bin_crps = np.mean(crps_values[bin_mask])
                results[bin_name] = bin_crps
                
        return results
    except Exception as e:
        print(f"Error calculating stratified CRPS: {e}")
        return None

def calculate_brier_score_for_cell(predicted_distributions,
                                 val_target_cell: np.ndarray,
                                 threshold: float = 0.2):
    """
    Calculates the mean Brier Score for precipitation occurrence
    exceeding a threshold for a single grid cell.

    Args:
        predicted_distributions: The output object from apply_easyuq_per_cell.
                                 Expected to have a .cdf() method.
        val_target_cell (np.ndarray): 1D array of validation observations for the cell.
        threshold (float): The precipitation threshold defining the binary event
                         (e.g., 0.2 mm). Defaults to 0.2.

    Returns:
        float or None: The mean Brier Score for the grid cell, or None if calculation fails.
    """
    if predicted_distributions is None or val_target_cell is None:
        print("Warning: Missing predicted distributions or targets for Brier Score calculation.")
        return None
    if not hasattr(predicted_distributions, 'cdf'):
        print("Warning: predicted_distributions object does not have a .cdf() method.")
        return None
    if len(val_target_cell) == 0:
        print("Warning: Empty target data for Brier Score calculation.")
        return None

    try:
        # 1. Calculate predicted probabilities P(Y > threshold) = 1 - CDF(threshold)
        # The .cdf() method likely needs the threshold(s) as input.
        # Assuming .cdf() returns probabilities for P(Y <= threshold)
        prob_le_threshold = predicted_distributions.cdf(thresholds=np.array([threshold]))

        # Handle potential multi-dimensional output if cdf handles multiple thresholds
        if isinstance(prob_le_threshold, np.ndarray) and prob_le_threshold.ndim > 1:
            if prob_le_threshold.shape[1] == 1:
                prob_le_threshold = prob_le_threshold.flatten()
            else:
                print(f"Warning: Unexpected shape from .cdf() method: {prob_le_threshold.shape}. Cannot calculate Brier Score.")
                return None
        
        if len(prob_le_threshold) != len(val_target_cell):
             print(f"Warning: Length mismatch between CDF output ({len(prob_le_threshold)}) and targets ({len(val_target_cell)}). Cannot calculate Brier Score.")
             return None


        predicted_prob_exceed = 1.0 - prob_le_threshold

        # Ensure probabilities are valid (between 0 and 1)
        predicted_prob_exceed = np.clip(predicted_prob_exceed, 0.0, 1.0)

        # 2. Determine binary outcomes
        binary_outcomes = (val_target_cell > threshold).astype(float)

        # 3. Calculate Brier Score: mean((probability - outcome)^2)
        brier_scores = (predicted_prob_exceed - binary_outcomes) ** 2
        mean_bs_cell = np.mean(brier_scores)

        # Handle potential NaN/inf results
        if not np.isfinite(mean_bs_cell):
            print(f"Warning: Non-finite Brier Score value calculated: {mean_bs_cell}")
            return None

        return mean_bs_cell

    except Exception as e:
        print(f"Error during Brier Score calculation: {e}")
        return None

def brier(y, x):
    """Calculates element-wise Brier scores."""
    return (x - y) ** 2

def CEP_pav(x, y):
    """
    Calculates the Conditional Event Probability using Pool Adjacent Violators Algorithm (PAVA)
    via scikit-learn's IsotonicRegression. Returns calibrated probabilities.
    Args:
        x: 1D array of forecast probabilities.
        y: 1D array of binary outcomes (0 or 1).
    Returns:
        1D array of calibrated probabilities.
    """
    # Ensure y contains only 0s and 1s, handle potential NaNs if necessary
    y_clean = np.nan_to_num(y).astype(int)
    if not np.all(np.isin(y_clean, [0, 1])):
         # Handle cases where y is not strictly binary after cleaning, maybe log warning
         # For now, proceed, but IsotonicRegression might behave unexpectedly
         print("Warning: Input 'y' to CEP_pav is not strictly binary [0, 1].")

    # IsotonicRegression expects x to be the independent variable (forecast)
    # and y to be the dependent variable (outcome)
    # It finds the best non-decreasing fit for y given x.
    try:
         # Handle potential issues with constant inputs/outputs
         if len(np.unique(x)) < 2 or len(np.unique(y_clean)) < 2:
              # If x or y is constant, isotonic regression might return trivial results
              # Return the mean observation frequency as a simple calibrated forecast
              print("Warning: Constant input encountered in CEP_pav. Returning mean frequency.")
              return np.full_like(x, np.mean(y_clean))

         ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
         # Fit_transform learns the isotonic relationship and returns the transformed (calibrated) x values
         pav_x = ir.fit_transform(x, y_clean)
         return pav_x
    except ValueError as ve:
         print(f"Error during IsotonicRegression fitting: {ve}")
         # Fallback: return original probabilities or mean frequency? Let's return mean.
         return np.full_like(x, np.mean(y_clean))

def brier_score_decomposition(forecast_probs: np.ndarray, binary_obs: np.ndarray):
    """
    Calculates the CORP decomposition of the Brier Score.

    Args:
        forecast_probs (np.ndarray): 1D array of forecast probabilities (between 0 and 1).
        binary_obs (np.ndarray): 1D array of binary outcomes (0 or 1).

    Returns:
        tuple: (mean_bs, miscalibration, discrimination, uncertainty) or (None, None, None, None) if calculation fails.
    """
    if forecast_probs is None or binary_obs is None:
        print("Warning: Missing forecast probabilities or observations for BS decomposition.")
        return None, None, None, None
    if len(forecast_probs) != len(binary_obs):
        print(f"Warning: Length mismatch for BS decomposition: Probs ({len(forecast_probs)}), Obs ({len(binary_obs)})")
        return None, None, None, None
    if len(forecast_probs) == 0:
        print("Warning: Empty data for BS decomposition.")
        return None, None, None, None

    # Ensure inputs are numpy arrays
    x = np.asarray(forecast_probs)
    y = np.asarray(binary_obs).astype(int)  # Ensure binary 0/1

    # Clip forecast probabilities to avoid issues with logit/numerical stability if used elsewhere
    x = np.clip(x, 1e-8, 1.0 - 1e-8)

    if not np.all(np.isin(y, [0, 1])):
        print("Error: binary_obs must contain only 0s and 1s.")
        return None, None, None, None

    try:
        # Calculate Mean Brier Score (BS)
        mean_bs = np.mean(brier(y, x))

        # Calculate Uncertainty (UNC) - BS of climatology forecast (mean observation)
        mean_obs = np.mean(y)
        uncertainty = np.mean(brier(y, mean_obs))  # Uses scalar mean_obs, broadcasts correctly

        # Calculate Calibrated Forecast using PAVA (Isotonic Regression)
        # Note: CEP_pav handles sorting internally via IsotonicRegression
        pav_x = CEP_pav(x, y)  # Gets the calibrated probabilities

        # Calculate Score of Calibrated Forecast (Sc)
        Sc = np.mean(brier(y, pav_x))

        # Calculate Decomposition Components
        # MCB = BS - Sc (Should be >= 0)
        miscalibration = max(0.0, mean_bs - Sc)  # Ensure non-negative due to potential float precision
        # DSC = UNC - Sc (Should be >= 0)
        discrimination = max(0.0, uncertainty - Sc)  # Ensure non-negative

        # Verify decomposition: BS ~= MCB - DSC + UNC => BS - MCB + DSC ~= UNC
        if not np.isclose(mean_bs, miscalibration - discrimination + uncertainty):
            print(f"Warning: BS decomposition components do not sum correctly. "
                  f"BS={mean_bs:.4f}, MCB={miscalibration:.4f}, DSC={discrimination:.4f}, UNC={uncertainty:.4f}. "
                  f"Sum={miscalibration - discrimination + uncertainty:.4f}")
                   
        # Handle potential non-finite results
        if not all(np.isfinite([mean_bs, miscalibration, discrimination, uncertainty])):
            print("Warning: Non-finite value encountered in BS decomposition results.")
            return None, None, None, None

        return mean_bs, miscalibration, discrimination, uncertainty

    except Exception as e:
        print(f"Error during Brier Score decomposition: {e}")
        return None, None, None, None

def numpy_json_encoder(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
         # Handle potential NaN/Inf
         if np.isnan(obj): return 'NaN'
         if np.isinf(obj): return 'Infinity' if obj > 0 else '-Infinity'
         return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
         # Fallback to string representation for other types
         return str(obj)

# ============================================================================
# SEEPS (Stable Equitable Error in Probability Space) Implementation
# Based on Rodwell et al. (2010)
# ============================================================================

def calculate_cell_precipitation_climatology(observed_precip_series, dry_threshold_mm=0.2):
    """
    Calculates climatological parameters for a single grid cell.
    
    This function establishes a local precipitation climatology by determining
    the probability of dry days and thresholds for light and heavy precipitation
    categories based on the method described in Rodwell et al. (2010).
    
    Args:
        observed_precip_series (np.ndarray): 1D array of observed precipitation values 
                                           for one grid cell from training data.
        dry_threshold_mm (float): Precipitation amount (e.g., 0.2 mm) at or below which 
                                a day is considered 'dry'. Default: 0.2 mm.
    
    Returns:
        dict or None: Dictionary containing:
            - 'p1': Probability of dry days
            - 'p2': Probability of light precipitation days  
            - 'p3': Probability of heavy precipitation days
            - 'dry_light_threshold': Threshold between dry and light (same as dry_threshold_mm)
            - 'light_heavy_threshold': Threshold between light and heavy precipitation
            Returns None if climatology cannot be calculated (e.g., extreme p1 values).
    """
    if observed_precip_series is None or len(observed_precip_series) == 0:
        print("Warning: Empty or None precipitation series provided")
        return None
    
    # Remove any NaN values
    valid_data = observed_precip_series[~np.isnan(observed_precip_series)]
    if len(valid_data) == 0:
        print("Warning: All values are NaN in precipitation series")
        return None
    
    # Step 1: Determine dry days
    dry_days = valid_data <= dry_threshold_mm
    p1 = np.sum(dry_days) / len(valid_data)
    
    # Check if p1 is within reasonable range for SEEPS applicability
    # Paper suggests p1 should be in range [0.10, 0.85]
    if p1 < 0.10 or p1 > 0.85:
        print(f"Warning: p1={p1:.3f} is outside recommended range [0.10, 0.85]. "
              f"SEEPS may not be appropriate for this cell.")
        # Continue anyway but flag this in the results
    
    # Step 2: Get wet days (precipitation > dry_threshold)
    wet_days_data = valid_data[valid_data > dry_threshold_mm]
    
    if len(wet_days_data) == 0:
        # All days are dry
        print("Warning: No wet days found in climatology. Setting p2=p3=0.")
        return {
            'p1': 1.0,
            'p2': 0.0,
            'p3': 0.0,
            'dry_light_threshold': dry_threshold_mm,
            'light_heavy_threshold': np.inf # or dry_threshold_mm, paper implies it can be same if no light precip
        }
    
    # Step 3: Calculate p2 and p3 using EMPIRICAL PERCENTILES (FIXED IMPLEMENTATION)
    # The paper suggests using terciles of wet day amounts to define light vs heavy
    # This ensures the thresholds are based on actual local climatology
    
    # Calculate the 66.67th percentile of wet days to separate light from heavy
    # This naturally creates a 2:1 ratio in the wet day population
    percentile_67 = 66.67
    light_heavy_threshold = np.percentile(wet_days_data, percentile_67)
    
    # Ensure light_heavy_threshold is at least dry_light_threshold
    if light_heavy_threshold < dry_threshold_mm:
        light_heavy_threshold = dry_threshold_mm

    # Now calculate p2 and p3 based on actual data distribution
    light_days = wet_days_data[wet_days_data <= light_heavy_threshold]
    heavy_days = wet_days_data[wet_days_data > light_heavy_threshold]
    
    # Calculate probabilities relative to the entire dataset (not just wet days)
    p2 = len(light_days) / len(valid_data)
    p3 = len(heavy_days) / len(valid_data)
    
    # Verify that p1 + p2 + p3 = 1 (within floating point precision)
    total_prob = p1 + p2 + p3
    if not np.isclose(total_prob, 1.0, rtol=1e-10):
        print(f"Warning: Probabilities don't sum to 1.0: p1={p1:.6f}, p2={p2:.6f}, p3={p3:.6f}, sum={total_prob:.6f}")
        # Normalize to ensure they sum to 1
        p1 = p1 / total_prob
        p2 = p2 / total_prob  
        p3 = p3 / total_prob
        print(f"Normalized: p1={p1:.6f}, p2={p2:.6f}, p3={p3:.6f}")

    return {
        'p1': p1,
        'p2': p2,
        'p3': p3,
        'dry_light_threshold': dry_threshold_mm,
        'light_heavy_threshold': light_heavy_threshold
    }

def get_seeps_error_matrix(p1, p2, p3):
    """
    Computes the 3x3 SEEPS error matrix based on climatological probabilities.
    
    The matrix elements s_vf represent the error when the observed category is v
    and the forecast category is f, following Equation (15) from Rodwell et al. (2010).
    
    Args:
        p1 (float): Climatological probability of dry days
        p2 (float): Climatological probability of light precipitation days
        p3 (float): Climatological probability of heavy precipitation days
        
    Returns:
        np.ndarray: 3x3 error matrix where element [v-1, f-1] is the SEEPS score
                   for observed category v and forecast category f.
                   Categories: 1=dry, 2=light, 3=heavy
    """
    # Initialize error matrix
    s = np.zeros((3, 3))
    
    # Handle potential division by zero or very small probabilities which can lead to large scores
    # Clamping probabilities to a small epsilon to avoid Inf scores if they are exactly zero
    eps = 1e-9
    p1_eff = max(p1, eps)
    p3_eff = max(p3, eps)
    one_minus_p1_eff = max(1 - p1, eps)


    # Diagonal elements (perfect forecasts) have zero error
    s[0, 0] = 0  # s_11: Observed dry, forecast dry
    s[1, 1] = 0  # s_22: Observed light, forecast light  
    s[2, 2] = 0  # s_33: Observed heavy, forecast heavy
    
    # Off-diagonal elements from Equation (15) in Rodwell et al. (2010)
    # Note: Matrix uses 0-based indexing, but formulas use 1-based
    
    # Row 1: Observed dry
    s[0, 1] = 1 / (2 * one_minus_p1_eff)                    # s_12: Obs dry, fcst light
    s[0, 2] = 1 / (2 * one_minus_p1_eff) + 1 / (2 * p3_eff)    # s_13: Obs dry, fcst heavy
    
    # Row 2: Observed light  
    s[1, 0] = 1 / (2 * p1_eff)                          # s_21: Obs light, fcst dry
    s[1, 2] = 1 / (2 * p3_eff)                          # s_23: Obs light, fcst heavy
    
    # Row 3: Observed heavy
    # Original: s[2, 0] = 1 / (2 * p1) + p2 / (2 * p1 * p3)
    # Original: s[2, 1] = p2 / (2 * p1 * p3)
    # Factor out common term for s_31 and s_32: term = p2 / (2 * p1_eff * p3_eff)
    # However, Rodwell (2010) Eq 15 explicitly states s_31 = 1/(2p1) + 1/(2p3) * p2/(1-p1-p3) if p2 is derived.
    # Simpler: s_31 = 1/(2p1) + 1/(2p3) * (p2/p_wet) / (p3/p_wet) ??? No.
    # Let's stick to the direct formula as written in paper if possible.
    # The version in the code seems to be:
    # s[2,0] = 1/(2*p1) + X  where X = p2 / (2*p1*p3)
    # s[2,1] = X
    # This structure where s[2,0] = s[1,0] + s[2,1] is common in some SEEPS variations.
    # The paper's Eq 15 has:
    # s_31 = 1/(2p_1) + p_2/(2*p_1*p_3)
    # s_32 = p_2/(2*p_1*p_3)
    # This looks consistent with what's implemented.
    
    common_term_s3 = p2 / (2 * p1_eff * p3_eff) # p2 can be zero if p_wet is very small or p1 is high
    s[2, 0] = 1 / (2 * p1_eff) + common_term_s3
    s[2, 1] = common_term_s3
    
    if np.any(np.isinf(s)) or np.any(np.isnan(s)):
         print(f"Warning: Invalid values (Inf/NaN) in SEEPS error matrix with "
               f"p1={p1:.3f}, p2={p2:.3f}, p3={p3:.3f}. Clamped p1_eff={p1_eff:.3e}, p3_eff={p3_eff:.3e}")
         # Fallback for extreme cases
         s[np.isinf(s) | np.isnan(s)] = 1e6 # Assign a large error

    return s

def categorize_precipitation(precip_value, dry_light_threshold, light_heavy_threshold):
    """
    Maps a precipitation value to a category (dry, light, or heavy).
    
    Args:
        precip_value (float): The precipitation amount in mm
        dry_light_threshold (float): Threshold separating dry from light (e.g., 0.2 mm)
        light_heavy_threshold (float): Threshold separating light from heavy
        
    Returns:
        int: Category code (1=dry, 2=light, 3=heavy)
    """
    if precip_value <= dry_light_threshold:
        return 1  # Dry
    elif precip_value <= light_heavy_threshold: # light_heavy_threshold can be equal to dry_light_threshold
        return 2  # Light
    else:
        return 3  # Heavy

def calculate_seeps_scores(consolidated_train_targets_all_cells, 
                         consolidated_val_preds_all_cells,
                         consolidated_val_targets_all_cells, 
                         germany_mask,
                         dry_threshold_mm=0.2):
    """
    Calculate SEEPS scores for deterministic precipitation forecasts.
    
    This function implements the complete SEEPS evaluation workflow:
    1. Builds local climatologies for each grid cell using training data
    2. Categorizes forecasts and observations into dry/light/heavy
    3. Computes SEEPS scores using cell-specific error matrices
    4. Returns spatially averaged scores for the Germany region
    
    Args:
        consolidated_train_targets_all_cells (np.ndarray): Training observations 
                                                         [n_train, grid_lat, grid_lon]
        consolidated_val_preds_all_cells (np.ndarray): Validation forecasts 
                                                     [n_val, grid_lat, grid_lon]
        consolidated_val_targets_all_cells (np.ndarray): Validation observations
                                                       [n_val, grid_lat, grid_lon]
        germany_mask (np.ndarray): Boolean mask [grid_lat, grid_lon] for Germany region
        dry_threshold_mm (float): Threshold for dry days (default: 0.2 mm)
        
    Returns:
        dict: Dictionary containing:
            - 'mean_seeps': Mean SEEPS score over Germany region
            - 'num_valid_cells': Number of cells included in calculation
            - 'num_samples': Total number of forecast-observation pairs scored
            - 'cell_climatologies': Dict of climatology parameters per cell (optional)
    """
    print(f"\nCalculating SEEPS scores for Germany region...")
    print(f"Dry threshold: {dry_threshold_mm} mm")
    
    # Get grid dimensions
    _, grid_lat, grid_lon = consolidated_val_targets_all_cells.shape
    num_val_samples = consolidated_val_preds_all_cells.shape[0]
    
    # Initialize storage
    all_seeps_scores_germany = []
    valid_cells_count = 0
    skipped_cells_count = 0
    cell_climatologies = {}
    
    # Process each grid cell
    for lat in range(grid_lat):
        for lon in range(grid_lon):
            # Skip cells outside Germany
            if not germany_mask[lat, lon]:
                continue
                
            # Extract training observations for this cell
            train_obs_cell = consolidated_train_targets_all_cells[:, lat, lon]
            
            # Calculate climatology
            climatology_params = calculate_cell_precipitation_climatology(
                train_obs_cell, dry_threshold_mm
            )
            
            if climatology_params is None:
                print(f"Skipping cell ({lat}, {lon}) - climatology calculation failed")
                skipped_cells_count += 1
                continue
                
            # Store climatology for this cell
            cell_climatologies[(lat, lon)] = climatology_params
            
            # Get SEEPS error matrix for this cell
            error_matrix = get_seeps_error_matrix(
                climatology_params['p1'],
                climatology_params['p2'], 
                climatology_params['p3']
            )
            
            # Check if error matrix is valid (already handled inside get_seeps_error_matrix by replacing Inf with large num)
            # if np.any(np.isinf(error_matrix)) or np.any(np.isnan(error_matrix)):
            #     print(f"Skipping cell ({lat}, {lon}) - invalid error matrix")
            #     skipped_cells_count += 1
            #     continue
                
            # Extract validation data for this cell
            val_forecasts_cell = consolidated_val_preds_all_cells[:, lat, lon]
            val_observations_cell = consolidated_val_targets_all_cells[:, lat, lon]
            
            # Calculate SEEPS scores for all validation samples at this cell
            for i in range(num_val_samples):
                # Skip if either value is NaN
                if np.isnan(val_forecasts_cell[i]) or np.isnan(val_observations_cell[i]):
                    continue
                    
                # Categorize observation and forecast
                obs_category = categorize_precipitation(
                    val_observations_cell[i],
                    climatology_params['dry_light_threshold'],
                    climatology_params['light_heavy_threshold']
                )
                
                fcst_category = categorize_precipitation(
                    val_forecasts_cell[i],
                    climatology_params['dry_light_threshold'],
                    climatology_params['light_heavy_threshold']
                )
                
                # Get SEEPS score from error matrix (adjust for 0-based indexing)
                seeps_score = error_matrix[obs_category - 1, fcst_category - 1]
                
                if np.isfinite(seeps_score): # Should always be finite now
                    all_seeps_scores_germany.append(seeps_score)
            
            valid_cells_count += 1
    
    # Calculate mean SEEPS score
    if len(all_seeps_scores_germany) > 0:
        mean_seeps_germany = np.mean(all_seeps_scores_germany)
        print(f"\nSEEPS calculation complete:")
        print(f"  Valid cells processed: {valid_cells_count}")
        print(f"  Cells skipped: {skipped_cells_count}")
        print(f"  Total scores calculated: {len(all_seeps_scores_germany)}")
        print(f"  Mean SEEPS score (Germany): {mean_seeps_germany:.4f}")
    else:
        mean_seeps_germany = np.nan
        print("Warning: No valid SEEPS scores calculated")
    
    return {
        'mean_seeps': mean_seeps_germany,
        'num_valid_cells': valid_cells_count,
        'num_samples': len(all_seeps_scores_germany),
        'cell_climatologies': cell_climatologies # Can be large, consider option to not return
    }

def plot_seasonal_metrics(results_summary: dict, crps_save_path: str, bs_save_path: str, fold_label: str):
    """
    Generates and saves bar charts for seasonal CRPS and Brier Score.

    Args:
        results_summary (dict): Dictionary containing evaluation results,
                                including the 'seasonal_metrics' key.
        crps_save_path (str): Full path where the seasonal CRPS plot will be saved.
        bs_save_path (str): Full path where the seasonal Brier Score plot will be saved.
        fold_label (str): Label for the fold (e.g., "Fold 0", "Comprehensive").
    """
    seasonal_metrics = results_summary.get('seasonal_metrics')
    if not seasonal_metrics:
        print("Seasonal metrics not found in results. Skipping plotting.")
        return

    evaluation_region = results_summary.get('evaluation_region', 'Germany')
    area_weighting = results_summary.get('area_weighting', 'cos(latitude)')
    region_suffix = f' - {evaluation_region} Region' if evaluation_region else ''

    season_order = ['DJF', 'MAM', 'JJA', 'SON']
    seasons = [s for s in season_order if s in seasonal_metrics]

    mean_crps_seasonal = [seasonal_metrics[s].get('mean_crps', np.nan) for s in seasons]
    # Assuming Brier score key is 'mean_bs_0.2mm' or similar, adjust if needed
    mean_bs_seasonal = [seasonal_metrics[s].get('mean_bs_0.2mm', seasonal_metrics[s].get('mean_bs', np.nan)) for s in seasons]

    # Define single dark gray color
    bar_color = '#4c4c4c'

    # --- Plot Seasonal CRPS ---
    try:
        plt.figure(figsize=(8, 5))
        plt.bar(seasons, mean_crps_seasonal, color=bar_color)
        plt.ylabel("Mean CRPS")
        plt.xlabel("Season")
        plt.title(f"{fold_label}: Spatially Area-Weighted ({area_weighting}) Mean CRPS per Season{region_suffix}")
        for i, val in enumerate(mean_crps_seasonal):
             if np.isfinite(val):
                  plt.text(i, val + (max(filter(np.isfinite,mean_crps_seasonal), default=0)*0.01), f'{val:.3f}', ha='center', va='bottom')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        os.makedirs(os.path.dirname(crps_save_path), exist_ok=True) # Ensure directory exists
        plt.savefig(crps_save_path, dpi=150)
        plt.close()
        print(f"Saved seasonal CRPS plot to: {crps_save_path}")
    except Exception as e:
        print(f"Error generating seasonal CRPS plot: {e}")
        plt.close()

    # --- Plot Seasonal Brier Score ---
    try:
        plt.figure(figsize=(8, 5))
        plt.bar(seasons, mean_bs_seasonal, color=bar_color)
        plt.ylabel("Mean Brier Score (Threshold: 0.2 mm)")
        plt.xlabel("Season")
        plt.title(f"{fold_label}: Spatially Area-Weighted ({area_weighting}) Mean Brier Score per Season{region_suffix}")
        for i, val in enumerate(mean_bs_seasonal):
             if np.isfinite(val):
                  plt.text(i, val + (max(filter(np.isfinite,mean_bs_seasonal), default=0)*0.01), f'{val:.3f}', ha='center', va='bottom')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.figtext(0.5, 0.02, "Baseline: MPC (training years only)", ha='center', fontsize=8, style='italic')
        plt.tight_layout()
        os.makedirs(os.path.dirname(bs_save_path), exist_ok=True) # Ensure directory exists
        plt.savefig(bs_save_path, dpi=150)
        plt.close()
        print(f"Saved seasonal Brier Score plot to: {bs_save_path}")
    except Exception as e:
        print(f"Error generating seasonal Brier Score plot: {e}")
        plt.close()

def plot_quantile_map(fold_dir, fold_num, time_index=0,
                      lats=DEFAULT_LATITUDES, lons=DEFAULT_LONGITUDES):
    """
    Plots a map comparison of the 50th percentile predictions vs. targets for a specific time index.
    Uses PowerNorm for consistent visualization with the training script.
    Uses Cartopy for geographical plotting.

    Args:
        fold_dir (str): Directory containing the fold's evaluation results.
                        Expected to contain "val_preds_p50.npy" and "combined_val_targets.npy".
        fold_num (str): The fold label (e.g., "Fold 0", "Comprehensive").
        time_index (int): The time index to plot. Default is 0.
        lats (np.ndarray): Array of latitudes for the grid.
        lons (np.ndarray): Array of longitudes for the grid.
    """
    print(f"Plotting quantile map for {fold_num} at time index {time_index}...")

    p50_preds_path = os.path.join(fold_dir, "val_preds_p50.npy")
    targets_path = os.path.join(fold_dir, "combined_val_targets.npy") # Corrected filename

    try:
        print(f"Loading target data from {targets_path}")
        print(f"Loading p50 prediction data from {p50_preds_path}")
        val_targets = np.load(targets_path)
        val_preds_p50 = np.load(p50_preds_path)
    except FileNotFoundError as e:
        print(f"Error: Could not load required files for quantile map: {e}")
        return

    if time_index >= val_targets.shape[0]:
        print(f"Error: Time index {time_index} exceeds available time steps ({val_targets.shape[0]}).")
        return

    if val_targets.shape[1] != len(lats) or val_targets.shape[2] != len(lons):
        print(f"Warning: Data grid dimensions ({val_targets.shape[1]}x{val_targets.shape[2]}) "
              f"do not match provided lats/lons dimensions ({len(lats)}x{len(lons)}). "
              "Using data dimensions for plotting extent but this might be inaccurate.")
        lons_plot = np.arange(val_targets.shape[2])
        lats_plot = np.arange(val_targets.shape[1])
        x_inc_plot, y_inc_plot = 1.0, 1.0
        map_projection = ccrs.PlateCarree()
        germany_extent_plot = None
    else:
        lons_plot = lons
        lats_plot = lats
        x_inc_plot = XINC
        y_inc_plot = YINC
        map_projection = ccrs.PlateCarree()
        germany_extent_plot = [GERMANY_PLOT_LON_MIN - 7, GERMANY_PLOT_LON_MAX + 7,
                               GERMANY_PLOT_LAT_MIN - 5, GERMANY_PLOT_LAT_MAX + 5]


    target_map = val_targets[time_index, :, :]
    p50_map = val_preds_p50[time_index, :, :]

    fig = plt.figure(figsize=(16, 6))

    norm = colors.PowerNorm(gamma=0.5)
    plot_extent = [lons_plot[0] - 0.5 * x_inc_plot, lons_plot[-1] + 0.5 * x_inc_plot,
                   lats_plot[0] - 0.5 * y_inc_plot, lats_plot[-1] + 0.5 * y_inc_plot]

    # Calculate nice tick locations for the quantile map
    if germany_extent_plot:
        lon_min, lon_max = germany_extent_plot[0], germany_extent_plot[1]
        lat_min, lat_max = germany_extent_plot[2], germany_extent_plot[3]
        lon_interval = 5  # Use 5-degree intervals for Germany-focused view
        lat_interval = 5
    else:
        lon_min, lon_max = plot_extent[0], plot_extent[1]
        lat_min, lat_max = plot_extent[2], plot_extent[3]
        lon_interval = 20  # Larger intervals for cleaner plots
        lat_interval = 20
    
    lon_ticks = np.arange(np.floor(lon_min/lon_interval)*lon_interval, 
                          np.ceil(lon_max/lon_interval)*lon_interval + lon_interval, 
                          lon_interval)
    lat_ticks = np.arange(np.floor(lat_min/lat_interval)*lat_interval, 
                          np.ceil(lat_max/lat_interval)*lat_interval + lat_interval, 
                          lat_interval)
    
    lon_ticks = lon_ticks[(lon_ticks >= lon_min) & (lon_ticks <= lon_max)]
    lat_ticks = lat_ticks[(lat_ticks >= lat_min) & (lat_ticks <= lat_max)]

    ax1 = fig.add_subplot(1, 2, 1, projection=map_projection)
    im0 = ax1.imshow(target_map, extent=plot_extent, origin='lower',
                     transform=ccrs.PlateCarree(), cmap='Blues', norm=norm)
    ax1.coastlines(resolution='50m', linewidth=1.0, color='black', alpha=0.7)
    ax1.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.6, linewidth=0.6, edgecolor='black')
    if germany_extent_plot: ax1.set_extent(germany_extent_plot, crs=ccrs.PlateCarree())
    
    # Add gridlines with controlled labels
    gl1 = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, linestyle='--',
                        xlocs=lon_ticks, ylocs=lat_ticks)
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xlabel_style = {'size': 10, 'rotation': 0}
    gl1.ylabel_style = {'size': 10}
    
    ax1.set_title(f"Target Precipitation (Time Index {time_index}, Power Scale)")
    fig.colorbar(im0, ax=ax1, label="Precipitation (mm)", shrink=0.8)

    ax2 = fig.add_subplot(1, 2, 2, projection=map_projection)
    im1 = ax2.imshow(p50_map, extent=plot_extent, origin='lower',
                     transform=ccrs.PlateCarree(), cmap='Blues', norm=norm)
    ax2.coastlines(resolution='50m', linewidth=1.0, color='black', alpha=0.7)
    ax2.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.6, linewidth=0.6, edgecolor='black')
    if germany_extent_plot: ax2.set_extent(germany_extent_plot, crs=ccrs.PlateCarree())
    
    # Add gridlines with controlled labels
    gl2 = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, linestyle='--',
                        xlocs=lon_ticks, ylocs=lat_ticks)
    gl2.top_labels = False
    gl2.right_labels = False
    gl2.left_labels = False  # Don't repeat y-axis labels on second plot
    gl2.xlabel_style = {'size': 10, 'rotation': 0}
    gl2.ylabel_style = {'size': 10}
    
    ax2.set_title(f"P50 Prediction (Time Index {time_index}, Power Scale)")
    fig.colorbar(im1, ax=ax2, label="Precipitation (mm)", shrink=0.8)

    fig.suptitle(f"{fold_num}: Target vs P50 Prediction", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the plot directly. The renaming logic in mswep_unet_training.py can be simplified or removed
    # if fold_num is passed as "Comprehensive" directly.
    save_path = os.path.join(fold_dir, f"quantile_p50_map_fold{fold_num}_time{time_index}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure directory exists
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved quantile map comparison to: {save_path}")

def plot_single_cdf(fold_dir, fold_num, lat_idx, lon_idx, time_index=0):
    """
    Plots the predicted CDF for a single grid cell and time step.
    
    Args:
        fold_dir (str): Directory containing the fold's evaluation results.
        fold_num (int): The fold number (for labeling purposes).
        lat_idx (int): Latitude index of the target cell.
        lon_idx (int): Longitude index of the target cell.
        time_index (int): The time index to plot. Default is 0.
    """
    print(f"Plotting CDF for cell ({lat_idx}, {lon_idx}) at time index {time_index}...")
    
    # Load data for EasyUQ application
    val_preds_path = os.path.join(fold_dir, "val_preds.npy")
    val_targets_path = os.path.join(fold_dir, "val_targets.npy")
    train_preds_path = os.path.join(fold_dir, "train_preds_all.npy")
    train_targets_path = os.path.join(fold_dir, "train_targets_all.npy")
    mask_path = os.path.join(fold_dir, "germany_mask.npy")
    
    try:
        val_preds = np.load(val_preds_path)
        val_targets = np.load(val_targets_path)
        train_preds = np.load(train_preds_path)
        train_targets = np.load(train_targets_path)
        mask = np.load(mask_path)
    except FileNotFoundError as e:
        print(f"Error: Could not load required files: {e}")
        return
    
    # Check if the cell is valid according to the mask
    try:
        if not mask[lat_idx, lon_idx]:
            print(f"Error: Cell ({lat_idx}, {lon_idx}) is not a valid cell according to the mask.")
            return
    except IndexError:
        print(f"Error: Cell coordinates ({lat_idx}, {lon_idx}) are out of bounds for mask with shape {mask.shape}.")
        return
    
    # Check if time_index is valid
    if time_index >= val_targets.shape[0]:
        print(f"Error: Time index {time_index} exceeds available time steps ({val_targets.shape[0]}).")
        return
        
    # Get cell data
    train_preds_cell = train_preds[:, lat_idx, lon_idx]
    train_target_cell = train_targets[:, lat_idx, lon_idx]
    val_preds_cell = val_preds[:, lat_idx, lon_idx]
    val_target_cell = val_targets[:, lat_idx, lon_idx]
    
    # Get the actual observation for this time step
    target_obs = val_target_cell[time_index]
    
    # Apply EasyUQ to get predicted distribution
    predicted_distributions = apply_easyuq_per_cell(
        train_preds_cell, train_target_cell, eval_preds_cell=val_preds_cell
    )
    
    if predicted_distributions is None:
        print(f"Error: Failed to generate EasyUQ predictions for cell ({lat_idx}, {lon_idx}).")
        return
    
    # Access the prediction for the specific time index
    try:
        # Try to access the internal prediction data structure
        # The exact structure might depend on the isodisreg implementation
        # We need to get jump points and CDF values
        single_pred_data = predicted_distributions.predictions[time_index]
        points = single_pred_data.points  # Jump points in the CDF
        cdf_values_at_points = single_pred_data.ecdf  # CDF values at those points
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot the CDF as a step function
        # To make a proper step plot from points and ecdf values:
        # The ecdf values are P(X <= point).
        # So, for x < points[0], CDF is 0.
        # For points[i] <= x < points[i+1], CDF is cdf_values_at_points[i]
        plot_x = []
        plot_y = []
        
        # Start from a point slightly less than the first jump if points[0] > 0, or from 0
        # x_min_plot = points[0] - 0.1 * points[0] if points[0] > 0 else 0
        # if x_min_plot < 0: x_min_plot = 0
        
        # If first point is > 0, CDF is 0 up to that point
        if points[0] > 0:
            plot_x.extend([0, points[0]])
            plot_y.extend([0, 0])
        elif points[0] == 0: # If first jump is at 0
             plot_x.append(0)
             plot_y.append(0) # Start CDF at 0 for P(X <= 0) if points[0] is 0

        for i in range(len(points)):
            plot_x.append(points[i])
            plot_y.append(cdf_values_at_points[i])
            if i + 1 < len(points): # Add next segment start
                plot_x.append(points[i+1]) 
                plot_y.append(cdf_values_at_points[i]) # Keep previous CDF value until next jump
        
        # Extend to a max value if needed, assuming CDF reaches 1
        if plot_y[-1] < 1.0:
            plot_x.append(points[-1] + (points[-1] - points[0])*0.1 if len(points)>1 else points[-1]+1) # Extend a bit
            plot_y.append(plot_y[-1]) # Should be 1 already if properly formed

        plt.plot(plot_x, plot_y, color='blue', linewidth=2, label='Predicted CDF')
        
        # Add vertical line for the actual observation
        plt.axvline(target_obs, color='red', linestyle='--', 
                   label=f'Actual Obs: {target_obs:.2f} mm')
        
        # Add median line for reference
        p50_val_array = predicted_distributions.qpred(quantiles=np.array([0.5]))
        if p50_val_array is not None and len(p50_val_array) > time_index:
            p50 = p50_val_array[time_index]
            plt.axvline(p50, color='green', linestyle='-', 
                       label=f'Median (P50): {p50:.2f} mm')
        
        # Set limits and labels
        max_x_val = max(target_obs*1.5, np.max(points)*1.1 if len(points)>0 else 10, 10)
        plt.xlim(left=0, right=max_x_val) 
        plt.ylim(0, 1.05)  # Probability range with slight margin
        plt.xlabel('Precipitation (mm)')
        plt.ylabel('Cumulative Probability')
        plt.title(f'Predicted CDF - Fold {fold_num} - Cell ({lat_idx},{lon_idx}) - Time {time_index}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the figure
        save_path = os.path.join(fold_dir, f"single_cdf_plot_fold{fold_num}_cell{lat_idx}-{lon_idx}_time{time_index}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved CDF plot to: {save_path}")
        
    except (AttributeError, IndexError, TypeError) as e:
        print(f"Error accessing prediction data structure or plotting: {e}")
        print("The isodisreg implementation might have a different internal structure than expected, or data was None.")
        
        # Alternative approach: Plot CDF by evaluating at multiple threshold points
        try:
            print("Attempting alternative CDF plotting approach...")
            # Generate points to evaluate the CDF
            max_precip_eval = max(30, target_obs*2, np.max(val_target_cell)*1.1 if len(val_target_cell)>0 else 30)
            precipitation_values = np.linspace(0, max_precip_eval, 200)
            
            # Evaluate CDF at these points
            # The cdf method expects a 1D array of thresholds and returns a 2D array [n_samples, n_thresholds]
            # We want the CDF for a specific sample (time_index) across all these thresholds
            cdf_vals_for_sample = []
            for p_thresh in precipitation_values:
                cdf_at_p_for_all_samples = predicted_distributions.cdf(thresholds=np.array([p_thresh]))
                if cdf_at_p_for_all_samples is not None and len(cdf_at_p_for_all_samples) > time_index:
                     cdf_vals_for_sample.append(cdf_at_p_for_all_samples[time_index, 0])
                else: # Handle case where cdf eval fails or returns unexpected
                     cdf_vals_for_sample.append(np.nan)

            if any(np.isnan(cdf_vals_for_sample)):
                 print("Warning: NaN values encountered during alternative CDF calculation.")

            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(precipitation_values, cdf_vals_for_sample, color='blue', linewidth=2, label='Predicted CDF (evaluated)')
            
            plt.axvline(target_obs, color='red', linestyle='--', 
                       label=f'Actual Obs: {target_obs:.2f} mm')
            
            p50_val_array = predicted_distributions.qpred(quantiles=np.array([0.5]))
            if p50_val_array is not None and len(p50_val_array) > time_index:
                p50 = p50_val_array[time_index]
                plt.axvline(p50, color='green', linestyle='-', 
                           label=f'Median (P50): {p50:.2f} mm')
            
            plt.xlim(left=0, right=max_precip_eval)
            plt.ylim(0, 1.05)
            plt.xlabel('Precipitation (mm)')
            plt.ylabel('Cumulative Probability')
            plt.title(f'Predicted CDF (Alt) - Fold {fold_num} - Cell ({lat_idx},{lon_idx}) - Time {time_index}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            save_path = os.path.join(fold_dir, f"single_cdf_plot_fold{fold_num}_cell{lat_idx}-{lon_idx}_time{time_index}_alt.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved alternative CDF plot to: {save_path}")
            
        except Exception as e2:
            print(f"Alternative CDF plotting also failed: {e2}")
            plt.close() # Ensure plot is closed if error occurs
            return


def plot_seasonal_samples(fold_dir, fold_num, num_samples_per_season=5, time_indices=None,
                          lats=DEFAULT_LATITUDES, lons=DEFAULT_LONGITUDES,
                          val_preds_fname="val_preds.npy",
                          val_targets_fname="val_targets.npy",
                          val_times_fname="val_times.npy",
                          p50_preds_fname="val_preds_p50.npy"):
    """
    Generates plots of representative sample days from each meteorological season,
    for both full and focused domains, and for deterministic and P50 predictions.
    Uses Cartopy for geographical plotting.
    Adds fallback for loading p50 prediction file.
    
    Args:
        fold_dir (str): Directory containing the fold's evaluation results
        fold_num (str): The fold number or label (e.g., "0", "Comprehensive")
        num_samples_per_season (int): Number of sample days to select from each season
        time_indices (list, optional): Specific time indices to plot. If None, randomly selects.
        lats (np.ndarray): Array of latitudes for the grid.
        lons (np.ndarray): Array of longitudes for the grid.
        val_preds_fname (str): Filename for deterministic validation predictions.
        val_targets_fname (str): Filename for validation targets.
        val_times_fname (str): Filename for validation timestamps.
        p50_preds_fname (str): Filename for P50 (EasyUQ median) predictions.
    """
    print(f"Generating seasonal sample plots for {fold_num}...")
    
    val_preds_path = os.path.join(fold_dir, val_preds_fname)
    val_targets_path = os.path.join(fold_dir, val_targets_fname)
    val_times_path = os.path.join(fold_dir, val_times_fname)
    
    # Determine the actual path for p50 predictions with fallback
    p50_path_to_load_primary = os.path.join(fold_dir, p50_preds_fname)
    p50_path_to_load_fallback = os.path.join(fold_dir, "val_preds_p50.npy") # Standard generic name
    
    actual_p50_path_to_load = None
    loaded_p50_fname = p50_preds_fname

    if os.path.exists(p50_path_to_load_primary):
        actual_p50_path_to_load = p50_path_to_load_primary
        print(f"  Primary P50 file found: {actual_p50_path_to_load}")
    elif os.path.exists(p50_path_to_load_fallback):
        print(f"  Primary P50 file '{p50_preds_fname}' not found at '{p50_path_to_load_primary}'.")
        print(f"  Falling back to use generic P50 predictions: {p50_path_to_load_fallback}")
        actual_p50_path_to_load = p50_path_to_load_fallback
        loaded_p50_fname = "val_preds_p50.npy"
    else:
        print(f"  Error: Neither primary P50 file '{p50_preds_fname}' ({p50_path_to_load_primary}) "
              f"nor fallback P50 file '{os.path.basename(p50_path_to_load_fallback)}' ({p50_path_to_load_fallback}) found.")
        # Let the np.load fail below if actual_p50_path_to_load remains None or points to a non-existent file.
        # For safety, ensure it points to the primary path if both are missing, so the error message is about the requested file.
        actual_p50_path_to_load = p50_path_to_load_primary


    try:
        val_preds = np.load(val_preds_path)
        val_targets = np.load(val_targets_path)
        val_times_np = np.load(val_times_path, allow_pickle=True) # Allow pickle for datetime objects
        
        print(f"  Attempting to load P50 predictions from: {actual_p50_path_to_load}")
        val_preds_p50 = np.load(actual_p50_path_to_load)
        
        print(f"Loaded data shapes for seasonal plots:")
        print(f"  Val preds ('{val_preds_fname}'): {val_preds.shape}")
        print(f"  Val targets ('{val_targets_fname}'): {val_targets.shape}")
        print(f"  Val times ('{val_times_fname}'): {val_times_np.shape}")
        print(f"  Val p50 preds ('{loaded_p50_fname}'): {val_preds_p50.shape}")
        
        val_times = pd.to_datetime(val_times_np)
        
        season_map = {
            12: 'DJF', 1: 'DJF', 2: 'DJF', 
            3: 'MAM', 4: 'MAM', 5: 'MAM',  
            6: 'JJA', 7: 'JJA', 8: 'JJA',  
            9: 'SON', 10: 'SON', 11: 'SON' 
        }
        
        timestamp_months = val_times.month
        timestamp_seasons = np.array([season_map[m] for m in timestamp_months])
        
        base_seasonal_plots_dir = os.path.join(fold_dir, "seasonal_sample_plots")
        os.makedirs(base_seasonal_plots_dir, exist_ok=True)
        
        for season in ['DJF', 'MAM', 'JJA', 'SON']:
            print(f"Processing {season} season...")
            season_indices = np.where(timestamp_seasons == season)[0]
            
            if len(season_indices) == 0:
                print(f"  No days found for season {season}")
                continue
                
            print(f"  Found {len(season_indices)} days for season {season}")
            
            selected_samples_for_season = []
            if time_indices is not None:
                season_samples_requested = [i for i in time_indices if i in season_indices]
                if season_samples_requested:
                     selected_samples_for_season = np.array(season_samples_requested)
            
            if not selected_samples_for_season: 
                selected_samples_for_season = np.random.choice(season_indices, 
                                                               min(num_samples_per_season, len(season_indices)), 
                                                               replace=False)
            
            print(f"  Selected {len(selected_samples_for_season)} samples for {season}: {selected_samples_for_season}")
            
            input_channels = 3 
            
            for sample_idx in selected_samples_for_season:
                sample_date_str = val_times[sample_idx].strftime('%Y-%m-%d')
                print(f"  Processing sample for {sample_date_str} (index {sample_idx})")
                
                inputs_viz_sample = np.zeros((input_channels, val_targets.shape[1], val_targets.shape[2]))
                for i_channel in range(input_channels):
                    # FIX: Correct the indexing logic to properly show t-3, t-2, t-1
                    # We want: i_channel 0 -> t-3, i_channel 1 -> t-2, i_channel 2 -> t-1
                    lag_idx = sample_idx - (input_channels - i_channel)
                    if lag_idx >= 0:
                        inputs_viz_sample[i_channel] = val_targets[lag_idx].copy()
                
                target_sample = val_targets[sample_idx]
                det_pred_sample = val_preds[sample_idx] 
                p50_pred_sample = val_preds_p50[sample_idx]
                
                plot_types = {
                    "deterministic": det_pred_sample,
                    "p50_easyuq": p50_pred_sample
                }
                domain_views = {
                    "focused_ger": "focused",
                    "full_domain": "full"
                }

                for pred_key, pred_data in plot_types.items():
                    for domain_key, domain_val in domain_views.items():
                        plot_title = f"{pred_key.replace('_', ' ').title()} - {domain_key.replace('_', ' ').title()}\n{season} {sample_date_str} (Idx {sample_idx})"
                        
                        sample_day_plot_dir = os.path.join(base_seasonal_plots_dir, f"{season}_{sample_date_str}_idx{sample_idx}")
                        os.makedirs(sample_day_plot_dir, exist_ok=True)
                        
                        save_plot_path = os.path.join(sample_day_plot_dir, f"{pred_key}_{domain_key}.png")
                        
                        plot_sample(inputs_viz_sample, target_sample, pred_data, 
                                    title=plot_title,
                                    save_path=save_plot_path, 
                                    lats=lats, lons=lons,
                                    domain_view=domain_val)
        
        print(f"Finished generating seasonal sample plots for {fold_num}")
    
    except FileNotFoundError as e:
        print(f"Error: Could not load required files for seasonal plots: {e}")
        print(f"  Attempted Det Preds: {val_preds_path}")
        print(f"  Attempted Targets: {val_targets_path}")
        print(f"  Attempted Times: {val_times_path}")
        print(f"  Attempted P50 Preds (primary): {p50_path_to_load_primary}")
        print(f"  Attempted P50 Preds (fallback): {p50_path_to_load_fallback}")

    except Exception as e:
        print(f"Error generating seasonal sample plots: {e}")
        import traceback
        traceback.print_exc()
        
def plot_sample(inputs, target, prediction, title="Precipitation Forecast", save_path=None,
                lats=DEFAULT_LATITUDES, lons=DEFAULT_LONGITUDES, domain_view='focused'):
    """
    Generate a 6-panel plot for a single sample day showing inputs, target, and prediction.
    Uses Cartopy for geographical plotting with consistent scaling across all panels.

    Args:
        inputs: Input array with shape [channels, height, width] (e.g., for t-3, t-2, t-1)
        target: Target array with shape [height, width]
        prediction: Prediction array with shape [height, width]
        title: Plot title
        save_path: Path to save the plot
        lats (np.ndarray): Array of latitudes for the grid.
        lons (np.ndarray): Array of longitudes for the grid.
        domain_view (str): 'focused' for Germany-centric view, 'full' for the whole data extent.
    """
    try:
        # Increase figure size and DPI for maximum sharpness
        fig = plt.figure(figsize=(24, 14), dpi=150)  # Larger figure with explicit DPI
        
        # Set font sizes for better readability
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.titlesize': 16
        })

        # Create consistent scaling across all precipitation panels
        # Combine all precipitation data for scaling calculation
        all_precip_data = []
        for i in range(min(3, inputs.shape[0])):
            all_precip_data.append(inputs[i, :, :])
        all_precip_data.extend([target, prediction])
        
        # Calculate robust min/max using percentiles to handle outliers
        all_precip_values = np.concatenate([data.flatten() for data in all_precip_data])
        all_precip_values = all_precip_values[~np.isnan(all_precip_values)]
        
        if len(all_precip_values) > 0:
            # Use 2nd and 98th percentiles for more robust scaling
            vmin_precip = max(0, np.percentile(all_precip_values, 2))
            vmax_precip = np.percentile(all_precip_values, 98)
            
            # Ensure reasonable range
            if vmax_precip <= vmin_precip:
                vmax_precip = vmin_precip + 1.0
            
            # Round to nice values for cleaner colorbar
            if vmax_precip > 10:
                vmax_precip = np.ceil(vmax_precip / 5) * 5  # Round up to nearest 5
            elif vmax_precip > 1:
                vmax_precip = np.ceil(vmax_precip)  # Round up to nearest integer
            else:
                vmax_precip = np.ceil(vmax_precip * 10) / 10  # Round up to nearest 0.1
        else:
            vmin_precip, vmax_precip = 0, 10  # Fallback values
        
        # Create consistent normalization - using Normalize instead of PowerNorm for more predictable scaling
        # PowerNorm can create inconsistent colorbar ticks
        norm_consistent = colors.Normalize(vmin=vmin_precip, vmax=vmax_precip)
        
        # Define consistent colorbar levels for all precipitation panels
        n_levels = 11  # Number of colorbar levels
        precip_levels = np.linspace(vmin_precip, vmax_precip, n_levels)

        data_shape_lat, data_shape_lon = target.shape
        if data_shape_lat != len(lats) or data_shape_lon != len(lons):
            print(f"Warning in plot_sample: Data grid dimensions ({data_shape_lat}x{data_shape_lon}) "
                  f"do not match provided lats/lons dimensions ({len(lats)}x{len(lons)}). "
                  "Plotting might be incorrect.")

        # This is the full data extent
        full_plot_extent_cartopy = [lons[0] - 0.5*XINC, lons[-1] + 0.5*XINC,
                                    lats[0] - 0.5*YINC, lats[-1] + 0.5*YINC]

        # This is the focused view (Germany + context)
        focused_map_view_extent_cartopy = [GERMANY_PLOT_LON_MIN - 15, GERMANY_PLOT_LON_MAX + 15,
                                           GERMANY_PLOT_LAT_MIN - 10, GERMANY_PLOT_LAT_MAX + 10]
        
        current_map_view_extent = focused_map_view_extent_cartopy
        if domain_view == 'full':
            # For 'full' view, we want the plot to naturally fit the full_plot_extent_cartopy.
            # We can set a slightly larger extent or let cartopy auto-adjust based on imshow's extent.
            # Using full_plot_extent_cartopy directly for set_extent can work.
            current_map_view_extent = full_plot_extent_cartopy
        elif domain_view != 'focused':
            print(f"Warning: Unknown domain_view '{domain_view}'. Defaulting to 'focused'.")
            current_map_view_extent = focused_map_view_extent_cartopy


        # Calculate nice tick locations based on the current map extent
        # Use much larger intervals for cleaner appearance
        if domain_view == 'focused':
            lon_interval = 10  # Increased from 5
            lat_interval = 10  # Increased from 5
        else:
            lon_interval = 30  # Increased from 10
            lat_interval = 20  # Increased from 10
            
        # Calculate tick locations that fall within the current extent
        lon_min, lon_max = current_map_view_extent[0], current_map_view_extent[1]
        lat_min, lat_max = current_map_view_extent[2], current_map_view_extent[3]
        
        # Round to nice numbers and create tick arrays
        lon_ticks = np.arange(np.floor(lon_min/lon_interval)*lon_interval, 
                              np.ceil(lon_max/lon_interval)*lon_interval + lon_interval, 
                              lon_interval)
        lat_ticks = np.arange(np.floor(lat_min/lat_interval)*lat_interval, 
                              np.ceil(lat_max/lat_interval)*lat_interval + lat_interval, 
                              lat_interval)
        
        # Filter to only include ticks within the extent
        lon_ticks = lon_ticks[(lon_ticks >= lon_min) & (lon_ticks <= lon_max)]
        lat_ticks = lat_ticks[(lat_ticks >= lat_min) & (lat_ticks <= lat_max)]

        # Input visualization with consistent scaling
        num_input_channels_to_plot = min(3, inputs.shape[0])
        for i in range(num_input_channels_to_plot):
            ax = fig.add_subplot(2, 3, i + 1, projection=ccrs.PlateCarree())
            input_day = inputs[i, :, :]
            
            # Use contourf for smoother visualization with exact levels
            im = ax.contourf(lons, lats, input_day, levels=precip_levels, 
                             transform=ccrs.PlateCarree(), cmap='Blues', 
                             norm=norm_consistent, extend='max')
            
            # Add geographic features - prioritize data visibility
            ax.coastlines(resolution='50m', linewidth=1.0, color='black', alpha=0.7)
            ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.6, linewidth=0.6, edgecolor='black')
            # Don't add ocean/land shading - it interferes with precipitation data visibility
            ax.set_extent(current_map_view_extent, crs=ccrs.PlateCarree())
            
            # Grid lines with controlled labels
            # Top row: no x labels (to avoid crowding), y labels only on leftmost
            show_y_labels = (i == 0)  # Only show y labels on leftmost column
            
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, linestyle='--',
                              xlocs=lon_ticks, ylocs=lat_ticks)
            gl.top_labels = False
            gl.right_labels = False
            gl.bottom_labels = False  # No x labels on top row
            gl.left_labels = show_y_labels
            gl.xlabel_style = {'size': 10, 'rotation': 0}
            gl.ylabel_style = {'size': 10}
            
            # Correct the titles to match the fixed indexing (t-3, t-2, t-1)
            lag_days = num_input_channels_to_plot - i
            ax.set_title(f'Input: t-{lag_days} Day Precip', fontsize=14, weight='bold', pad=10)
            
            # Consistent colorbar with fixed ticks
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Precipitation (mm)",
                                ticks=precip_levels[::2])  # Show every other tick for clarity
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label("Precipitation (mm)", size=11)
            
            # Format colorbar tick labels
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}' if x < 10 else f'{x:.0f}'))

        ger_lon_coord_min = lons[GERMANY_BOX_GRID_LON_INDICES[0]]
        ger_lon_coord_max = lons[GERMANY_BOX_GRID_LON_INDICES[1]]
        ger_lat_coord_min = lats[GERMANY_BOX_GRID_LAT_INDICES[0]]
        ger_lat_coord_max = lats[GERMANY_BOX_GRID_LAT_INDICES[1]]

        rect_lons = [ger_lon_coord_min, ger_lon_coord_max, ger_lon_coord_max, ger_lon_coord_min, ger_lon_coord_min]
        rect_lats = [ger_lat_coord_min, ger_lat_coord_min, ger_lat_coord_max, ger_lat_coord_max, ger_lat_coord_min]

        # Target panel with consistent scaling
        ax_target = fig.add_subplot(2, 3, 4, projection=ccrs.PlateCarree())
        im_target = ax_target.contourf(lons, lats, target, levels=precip_levels,
                                       transform=ccrs.PlateCarree(), cmap='Blues',
                                       norm=norm_consistent, extend='max')
        
        # Add geographic features - prioritize data visibility
        ax_target.coastlines(resolution='50m', linewidth=1.0, color='black', alpha=0.7)
        ax_target.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.6, linewidth=0.6, edgecolor='black')
        # Don't add ocean/land shading - it interferes with precipitation data visibility
        
        # Add Germany bounding box with subtler styling
        ax_target.plot(rect_lons, rect_lats, transform=ccrs.PlateCarree(), 
                       color='darkred', linewidth=1.5, linestyle='--', alpha=0.7, label='Germany Region')
        
        ax_target.set_extent(current_map_view_extent, crs=ccrs.PlateCarree())
        
        # Grid lines with controlled labels - bottom row shows x labels, left column shows y labels
        gl_target = ax_target.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, linestyle='--',
                                        xlocs=lon_ticks, ylocs=lat_ticks)
        gl_target.top_labels = False
        gl_target.right_labels = False
        gl_target.bottom_labels = True  # Show x labels on bottom row
        gl_target.left_labels = True    # Show y labels on left column
        gl_target.xlabel_style = {'size': 10, 'rotation': 0}
        gl_target.ylabel_style = {'size': 10}
        
        ax_target.set_title('Target: Current Day (t)', fontsize=14, weight='bold', pad=10)
        
        # Consistent colorbar
        cbar_target = plt.colorbar(im_target, ax=ax_target, fraction=0.046, pad=0.04,
                                   ticks=precip_levels[::2])
        cbar_target.ax.tick_params(labelsize=10)
        cbar_target.set_label("Precipitation (mm)", size=11)
        cbar_target.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}' if x < 10 else f'{x:.0f}'))

        # Prediction panel with consistent scaling
        ax_pred = fig.add_subplot(2, 3, 5, projection=ccrs.PlateCarree())
        im_pred = ax_pred.contourf(lons, lats, prediction, levels=precip_levels,
                                   transform=ccrs.PlateCarree(), cmap='Blues',
                                   norm=norm_consistent, extend='max')
        
        # Add geographic features - prioritize data visibility
        ax_pred.coastlines(resolution='50m', linewidth=1.0, color='black', alpha=0.7)
        ax_pred.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.6, linewidth=0.6, edgecolor='black')
        # Don't add ocean/land shading - it interferes with precipitation data visibility
        
        # Add Germany bounding box with subtler styling
        ax_pred.plot(rect_lons, rect_lats, transform=ccrs.PlateCarree(), 
                     color='darkred', linewidth=1.5, linestyle='--', alpha=0.7)
        
        ax_pred.set_extent(current_map_view_extent, crs=ccrs.PlateCarree())
        
        # Grid lines with controlled labels - only show x labels on bottom row
        gl_pred = ax_pred.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, linestyle='--',
                                    xlocs=lon_ticks, ylocs=lat_ticks)
        gl_pred.top_labels = False
        gl_pred.right_labels = False
        gl_pred.bottom_labels = True  # Show x labels on bottom row
        gl_pred.left_labels = False   # No y labels in middle column
        gl_pred.xlabel_style = {'size': 10, 'rotation': 0}
        gl_pred.ylabel_style = {'size': 10}
        
        ax_pred.set_title('Prediction: Current Day (t)', fontsize=14, weight='bold', pad=10)
        
        # Consistent colorbar
        cbar_pred = plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04,
                                 ticks=precip_levels[::2])
        cbar_pred.ax.tick_params(labelsize=10)
        cbar_pred.set_label("Precipitation (mm)", size=11)
        cbar_pred.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}' if x < 10 else f'{x:.0f}'))

        # Difference panel with symmetric scaling
        ax_diff = fig.add_subplot(2, 3, 6, projection=ccrs.PlateCarree())
        diff = prediction - target
        
        # Calculate symmetric difference bounds
        max_abs_diff = np.nanpercentile(np.abs(diff), 98) if not np.isnan(diff).all() else 10
        if max_abs_diff < 0.1: 
            max_abs_diff = 0.1  # Minimum scale for visibility
        
        # Round to nice values
        if max_abs_diff > 10:
            max_abs_diff = np.ceil(max_abs_diff / 5) * 5
        elif max_abs_diff > 1:
            max_abs_diff = np.ceil(max_abs_diff)
        else:
            max_abs_diff = np.ceil(max_abs_diff * 10) / 10
            
        vmin_diff, vmax_diff = -max_abs_diff, max_abs_diff
        
        # Create levels for difference plot
        n_diff_levels = 21  # More levels for smoother gradient
        diff_levels = np.linspace(vmin_diff, vmax_diff, n_diff_levels)
        
        im_diff = ax_diff.contourf(lons, lats, diff, levels=diff_levels,
                                   transform=ccrs.PlateCarree(), cmap='RdBu_r',
                                   extend='both')
        
        # Add contour lines at zero for reference
        ax_diff.contour(lons, lats, diff, levels=[0], colors='black', 
                        linewidths=1, alpha=0.5, transform=ccrs.PlateCarree())
        
        # Add geographic features - prioritize data visibility
        ax_diff.coastlines(resolution='50m', linewidth=1.0, color='black', alpha=0.7)
        ax_diff.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.6, linewidth=0.6, edgecolor='black')
        # Don't add ocean/land shading - it interferes with precipitation data visibility
        
        # Add Germany bounding box with subtler styling
        ax_diff.plot(rect_lons, rect_lats, transform=ccrs.PlateCarree(), 
                     color='darkred', linewidth=1.5, linestyle='--', alpha=0.7)
        
        ax_diff.set_extent(current_map_view_extent, crs=ccrs.PlateCarree())
        
        # Grid lines with controlled labels - only show x labels on bottom row
        gl_diff = ax_diff.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, linestyle='--',
                                    xlocs=lon_ticks, ylocs=lat_ticks)
        gl_diff.top_labels = False
        gl_diff.right_labels = False
        gl_diff.bottom_labels = True  # Show x labels on bottom row
        gl_diff.left_labels = False   # No y labels in right column
        gl_diff.xlabel_style = {'size': 10, 'rotation': 0}
        gl_diff.ylabel_style = {'size': 10}
        
        ax_diff.set_title('Difference (Pred - Target)', fontsize=14, weight='bold', pad=10)
        
        # Difference colorbar with symmetric ticks
        cbar_diff = plt.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04,
                                 ticks=np.linspace(vmin_diff, vmax_diff, 7))
        cbar_diff.ax.tick_params(labelsize=10)
        cbar_diff.set_label("Difference (mm)", size=11)
        cbar_diff.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:+.1f}' if abs(x) < 10 else f'{x:+.0f}'))

        # Add overall title with better formatting
        plt.suptitle(title, fontsize=18, weight='bold', y=0.98)
        
        # Add a text box with statistics
        target_mean = np.nanmean(target)
        pred_mean = np.nanmean(prediction)
        diff_mean = np.nanmean(diff)
        diff_rmse = np.sqrt(np.nanmean(diff**2))
        
        stats_text = (f'Target Mean: {target_mean:.2f} mm  |  '
                      f'Pred Mean: {pred_mean:.2f} mm  |  '
                      f'Bias: {diff_mean:+.2f} mm  |  '
                      f'RMSE: {diff_rmse:.2f} mm')
        
        fig.text(0.5, 0.01, stats_text, ha='center', fontsize=11, 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))
        
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0.025, 1, 0.97])

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Save with maximum quality settings
            plt.savefig(save_path, 
                       dpi=300,  # High DPI for sharpness
                       bbox_inches='tight', 
                       facecolor='white',
                       edgecolor='none',
                       format='png',
                       pad_inches=0.1,  # Small padding around figure
                       metadata={'Software': 'MSWEP Evaluation System'})  # Add metadata
            plt.close()
            print(f"  Saved high-quality plot to: {save_path}")
        else:
            plt.show()
            plt.close()
        
        # Reset matplotlib parameters to defaults
        plt.rcParams.update(plt.rcParamsDefault)

    except Exception as e:
        print(f"Error generating plot sample: {e}")
        import traceback
        traceback.print_exc()
        if 'fig' in locals(): plt.close(fig)

def run_evaluation(base_output_dir: str, fold: int, batch_size: int = 500, n_recent_train_samples: int = 1000, generate_seasonal_plots: bool = True, is_final_evaluation: bool = False):
    """
    Runs the evaluation pipeline for a specific fold.
    Loads data, applies EasyUQ per cell (using recent train data), and stores results.

    WARNING: This function is designed for per-fold evaluation and should NOT be used
    in the refactored workflow. Use calculate_final_crps in mswep_unet_training.py
    for the comprehensive final evaluation instead.

    Args:
        base_output_dir (str): Base directory where fold results are saved.
        fold (int): The fold number to evaluate.
        batch_size (int): Number of grid cells to process in each batch (IGNORED if batching removed).
        n_recent_train_samples (int): Number of recent training samples for IDR fitting. (DEPRECATED - uses full aligned)
        generate_seasonal_plots (bool): Whether to generate seasonal plots.
        is_final_evaluation (bool): Whether this is the final evaluation on the complete dataset.
    """
    if not is_final_evaluation:
        print("\n" + "!"*70)
        print("! WARNING: Per-fold evaluation is deprecated in the refactored workflow!")
        print("! This function should only be called for debugging purposes.")
        print("! Use the comprehensive final evaluation instead.")
        print("!"*70 + "\n")

    evaluation_type = "FINAL COMPREHENSIVE" if is_final_evaluation else "Per-fold"
    print(f"\n=== Running {evaluation_type} Evaluation for Fold {fold} ===")

    # 1. Load Data
    data = load_evaluation_data(base_output_dir, fold)
    if data[0] is None:
        print(f"Skipping evaluation for fold {fold} due to data loading issues.")
        return None

    val_preds, val_targets, train_preds, train_targets, val_times, mask = data

    grid_lat_data, grid_lon_data = val_targets.shape[1], val_targets.shape[2]
    current_lats = DEFAULT_LATITUDES
    current_lons = DEFAULT_LONGITUDES
    if grid_lat_data != len(DEFAULT_LATITUDES) or grid_lon_data != len(DEFAULT_LONGITUDES):
        print(f"Data grid {grid_lat_data}x{grid_lon_data} differs from default {len(DEFAULT_LATITUDES)}x{len(DEFAULT_LONGITUDES)}. "
              "Attempting to create compatible coordinate arrays for plotting.")
        current_lats = np.linspace(DEFAULT_LATITUDES[0], DEFAULT_LATITUDES[-1], grid_lat_data)
        current_lons = np.linspace(DEFAULT_LONGITUDES[0], DEFAULT_LONGITUDES[-1], grid_lon_data)


    if train_preds is not None and train_targets is not None:
        assert train_preds.shape == train_targets.shape, (
            f"Shape mismatch between training predictions {train_preds.shape} and targets {train_targets.shape}. "
            f"This indicates misaligned data which will break IDR fitting."
        )
        print(f"Using full aligned training dataset with {len(train_preds)} samples for IDR fitting.")
        train_preds_recent = train_preds
        train_targets_recent = train_targets
    else:
        print("Error: Cannot proceed without training data for IDR.")
        return None

    if mask is None:
        print("Warning: Mask is None. Assuming all grid cells are valid.")
        if val_targets is not None:
             mask = np.ones(val_targets.shape[1:], dtype=bool)
        else:
             print("Error: Cannot determine grid size without targets or mask.")
             return None

    num_val_samples, grid_lat, grid_lon = val_targets.shape

    all_forecast_probs_germany_bsd = []
    all_binary_obs_germany_bsd = []
    brier_threshold = 0.2

    if val_times is not None:
         timestamps = val_times
         season_map = {
             12: 'DJF', 1: 'DJF', 2: 'DJF',
             3: 'MAM', 4: 'MAM', 5: 'MAM',
             6: 'JJA', 7: 'JJA', 8: 'JJA',
             9: 'SON', 10: 'SON', 11: 'SON'
         }
         seasons = ['DJF', 'MAM', 'JJA', 'SON']
         timestamp_months = timestamps.month
         timestamp_seasons = np.array([season_map[m] for m in timestamp_months])
         seasonal_aggregates = {
             season: {'crps_sum': 0.0, 'bs_sum': 0.0, 'count': 0,
                     'mae_sum': 0.0, 'mae_count': 0}
             for season in seasons
         }
         print(f"Initialized seasonal aggregates for: {seasons}")
    else:
         print("Warning: Validation timestamps not loaded, seasonal analysis will not be possible.")
         timestamps = None
         seasonal_aggregates = None
         seasons = []

    print("Calculating deterministic metrics on raw UNet predictions...")
    det_metrics = {'mae': {}, 'rmse': {}, 'bias': {}}
    bins = [0, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, float('inf')]
    bin_names = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" if bins[i+1] != float('inf') else f">{bins[i]:.1f}" for i in range(len(bins)-1)]
    for bin_name in bin_names:
        det_metrics[f'mae_{bin_name}'] = {}
    all_cell_mae = []
    all_cell_rmse = []

    for lat_idx_loop in range(grid_lat):
        for lon_idx_loop in range(grid_lon):
            if mask[lat_idx_loop, lon_idx_loop]:
                val_preds_cell = val_preds[:, lat_idx_loop, lon_idx_loop]
                val_target_cell = val_targets[:, lat_idx_loop, lon_idx_loop]
                errors = val_preds_cell - val_target_cell
                abs_errors = np.abs(errors)
                cell_mae = np.mean(abs_errors)
                cell_rmse = np.sqrt(np.mean(np.square(errors)))
                cell_bias = np.mean(errors)
                det_metrics['mae'][(lat_idx_loop, lon_idx_loop)] = cell_mae
                det_metrics['rmse'][(lat_idx_loop, lon_idx_loop)] = cell_rmse
                det_metrics['bias'][(lat_idx_loop, lon_idx_loop)] = cell_bias
                all_cell_mae.append(cell_mae)
                all_cell_rmse.append(cell_rmse)
                for i, bin_name in enumerate(bin_names):
                    bin_mask_strat = (val_target_cell >= bins[i]) & (val_target_cell < bins[i+1] if bins[i+1] != float('inf') else True)
                    if np.any(bin_mask_strat):
                        bin_mae = np.mean(abs_errors[bin_mask_strat])
                        det_metrics[f'mae_{bin_name}'][(lat_idx_loop, lon_idx_loop)] = bin_mae
                if seasonal_aggregates is not None and timestamps is not None:
                    for season in seasons:
                        season_mask_det = timestamp_seasons == season
                        if np.any(season_mask_det):
                            season_errors = abs_errors[season_mask_det]
                            seasonal_aggregates[season]['mae_sum'] += np.sum(season_errors)
                            seasonal_aggregates[season]['mae_count'] += len(season_errors)

    overall_mae = np.mean(all_cell_mae) if all_cell_mae else np.nan
    overall_rmse = np.mean(all_cell_rmse) if all_cell_rmse else np.nan
    print(f"\n--- Deterministic Metrics (Raw UNet Output) ---")
    print(f"Overall Mean Absolute Error (MAE): {overall_mae:.4f}")
    print(f"Overall Root Mean Squared Error (RMSE): {overall_rmse:.4f}")
    print("\nIntensity-specific MAE:")
    for bin_name in bin_names:
        bin_metrics_list = [v for v in det_metrics[f'mae_{bin_name}'].values() if np.isfinite(v)] # Renamed
        if bin_metrics_list: # Check if list is not empty
            bin_avg_mae = np.mean(bin_metrics_list)
            print(f"  Precipitation {bin_name} mm: MAE = {bin_avg_mae:.4f}")

    fold_dir = os.path.join(base_output_dir, f"fold{fold}")
    np.save(os.path.join(fold_dir, "deterministic_metrics.npy"), det_metrics)
    print(f"Saved deterministic metrics to: {os.path.join(fold_dir, 'deterministic_metrics.npy')}")

    val_preds_p50 = np.full((num_val_samples, grid_lat, grid_lon), np.nan)
    valid_cell_indices = [(lat_idx, lon_idx) for lat_idx in range(grid_lat) for lon_idx in range(grid_lon) if mask[lat_idx, lon_idx]]
    total_valid_cells = len(valid_cell_indices)
    print(f"Found {total_valid_cells} valid grid cells to process for IDR")

    all_cell_mean_crps = []
    all_cell_mean_bs = []
    processed_cells = 0

    metrics_path = os.path.join(fold_dir, "cell_metrics.npy")
    cell_metrics = {'crps': {}, 'bs': {}}
    for bin_name_iter in bin_names:
        cell_metrics[f'crps_{bin_name_iter}'] = {}

    # Create 2D lat/lon arrays for area weighting
    lat_2d, lon_2d = create_lat_lon_2d(current_lats, current_lons)
    
    # First, fit IDR models for all cells (needed for both old and new approaches)
    idr_models_by_cell = {}
    print(f"Fitting IDR models for all {total_valid_cells} valid cells...")
    
    for lat_cell, lon_cell in tqdm(valid_cell_indices, desc="Fitting IDR models"):
        train_preds_recent_cell = train_preds_recent[:, lat_cell, lon_cell]
        train_targets_recent_cell = train_targets_recent[:, lat_cell, lon_cell]
        
        # Fit IDR model for this cell
        try:
            train_preds_df = pd.DataFrame(train_preds_recent_cell)
            idr_model = idr(y=train_targets_recent_cell, X=train_preds_df)
            idr_models_by_cell[(lat_cell, lon_cell)] = idr_model
        except Exception as e:
            print(f"Warning: Failed to fit IDR for cell ({lat_cell}, {lon_cell}): {e}")
            idr_models_by_cell[(lat_cell, lon_cell)] = None
    
    # Compute daily CRPS time series using the new function
    print("Computing daily CRPS time series with area weighting...")
    daily_crps_series = compute_idr_crps_timeseries(
        idr_models_by_cell, val_preds, val_targets, mask, lat_2d
    )
    
    # Calculate overall mean CRPS (correct method)
    overall_mean_crps_new = np.nanmean(daily_crps_series)
    print(f"\nNew method - Overall mean CRPS (area-weighted): {overall_mean_crps_new:.4f}")
    
    # Continue with existing per-cell processing for other metrics
    print(f"\nProcessing all {total_valid_cells} valid cells for detailed metrics...")

    for lat_cell, lon_cell in tqdm(valid_cell_indices, desc="Processing Cells for Detailed Metrics"):
        val_preds_cell_loop = val_preds[:, lat_cell, lon_cell]
        val_target_cell_loop = val_targets[:, lat_cell, lon_cell]

        # Get pre-fitted IDR model
        if (lat_cell, lon_cell) not in idr_models_by_cell:
            continue
            
        idr_model = idr_models_by_cell[(lat_cell, lon_cell)]
        if idr_model is None:
            continue
            
        # Predict distributions for validation data
        try:
            eval_preds_df = pd.DataFrame(val_preds_cell_loop)
            predicted_distributions = idr_model.predict(eval_preds_df)
        except Exception as e:
            print(f"Warning: Failed to predict for cell ({lat_cell}, {lon_cell}): {e}")
            continue

        if predicted_distributions is not None:
            try:
                y_true_crps = val_target_cell_loop.astype(float).flatten()
                crps_result = predicted_distributions.crps(y_true_crps)
                crps_values = None
                if isinstance(crps_result, list): crps_values = np.array(crps_result, dtype=float)
                elif isinstance(crps_result, np.ndarray): crps_values = crps_result

                if crps_values is not None and crps_values.shape == y_true_crps.shape:
                    valid_crps_mask = np.isfinite(crps_values)
                    if np.any(valid_crps_mask):
                        mean_crps_cell = np.mean(crps_values[valid_crps_mask])
                        if np.isfinite(mean_crps_cell):
                            all_cell_mean_crps.append(mean_crps_cell)
                            cell_metrics['crps'][(lat_cell, lon_cell)] = mean_crps_cell
                            for i_bin, bin_name_crps in enumerate(bin_names):
                                bin_mask_crps = (val_target_cell_loop >= bins[i_bin]) & \
                                                (val_target_cell_loop < bins[i_bin+1] if bins[i_bin+1] != float('inf') else True)
                                if np.any(bin_mask_crps):
                                    valid_bin_mask_crps = bin_mask_crps & valid_crps_mask
                                    if np.any(valid_bin_mask_crps):
                                        bin_crps_val = np.mean(crps_values[valid_bin_mask_crps])
                                        cell_metrics[f'crps_{bin_name_crps}'][(lat_cell, lon_cell)] = bin_crps_val
                        # Skip seasonal aggregation here - will use daily time series instead
                else:
                    print(f"Warning: CRPS calculation failed or shape mismatch for cell ({lat_cell}, {lon_cell}).")
            except Exception as e:
                print(f"Error calculating CRPS for cell ({lat_cell}, {lon_cell}): {e}")

            try:
                prob_le_thresh_bsd = predicted_distributions.cdf(thresholds=np.array([brier_threshold]))
                if isinstance(prob_le_thresh_bsd, np.ndarray) and prob_le_thresh_bsd.ndim > 1 and prob_le_thresh_bsd.shape[1] == 1:
                    prob_le_thresh_bsd = prob_le_thresh_bsd.flatten()
                if prob_le_thresh_bsd is not None and len(prob_le_thresh_bsd) == len(val_target_cell_loop):
                    pred_prob_exceed_bsd = 1.0 - prob_le_thresh_bsd
                    pred_prob_exceed_bsd = np.clip(pred_prob_exceed_bsd, 0.0, 1.0)
                    binary_outcomes_bsd = (val_target_cell_loop > brier_threshold).astype(float)
                    all_forecast_probs_germany_bsd.extend(pred_prob_exceed_bsd)
                    all_binary_obs_germany_bsd.extend(binary_outcomes_bsd)
                    brier_scores_cell = (pred_prob_exceed_bsd - binary_outcomes_bsd) ** 2
                    valid_bs_mask_cell = np.isfinite(brier_scores_cell)
                    if np.any(valid_bs_mask_cell):
                        mean_bs_cell_val = np.mean(brier_scores_cell[valid_bs_mask_cell])
                        if np.isfinite(mean_bs_cell_val):
                            all_cell_mean_bs.append(mean_bs_cell_val)
                            cell_metrics['bs'][(lat_cell, lon_cell)] = mean_bs_cell_val
                        if seasonal_aggregates is not None and timestamps is not None:
                            valid_bs_values_cell = brier_scores_cell[valid_bs_mask_cell]
                            valid_seasons_bs = timestamp_seasons[valid_bs_mask_cell]
                            for season_iter_bs in seasons:
                                season_mask_bs_val = (valid_seasons_bs == season_iter_bs)
                                if np.any(season_mask_bs_val):
                                    seasonal_aggregates[season_iter_bs]['bs_sum'] += np.sum(valid_bs_values_cell[season_mask_bs_val])
                else:
                    print(f"Warning: CDF output shape mismatch for Brier Score at cell ({lat_cell}, {lon_cell}).")
            except Exception as e:
                print(f"Error calculating Brier Score components for cell ({lat_cell}, {lon_cell}): {e}")

            try:
                median_preds_cell = predicted_distributions.qpred(quantiles=np.array([0.5]))
                if median_preds_cell is not None and len(median_preds_cell) == num_val_samples:
                    val_preds_p50[:, lat_cell, lon_cell] = median_preds_cell.flatten()
            except Exception as e:
                print(f"Error extracting median predictions for cell ({lat_cell}, {lon_cell}): {e}")
            del predicted_distributions
        processed_cells += 1

    # Save metrics and p50 predictions once after all cells are processed
    try:
        np.save(metrics_path, cell_metrics)
        print(f"Saved final cell metrics after {processed_cells}/{total_valid_cells} cells to {metrics_path}")
        np.save(os.path.join(fold_dir, "val_preds_p50.npy"), val_preds_p50)
        print(f"Saved final 50th percentile predictions to: {os.path.join(fold_dir, 'val_preds_p50.npy')}")
    except Exception as e:
        print(f"Error saving final cell metrics or p50 predictions: {e}")

    import gc
    gc.collect()
    print(f"Memory freed after processing all cells.")
    # --- END OF BATCHING REMOVAL ---

    print(f"EasyUQ application complete. Processed {processed_cells} valid cells.")

    # Use the new area-weighted daily time series for overall mean CRPS
    overall_mean_crps = overall_mean_crps_new  # From compute_idr_crps_timeseries
    
    # Compute area-weighted Brier Score time series
    print("Computing daily Brier Score time series with area weighting...")
    daily_bs_series, all_probs_for_decomp, all_obs_for_decomp = compute_brier_score_timeseries(
        idr_models_by_cell, val_preds, val_targets, mask, lat_2d, threshold=brier_threshold
    )
    
    # Calculate overall mean BS from daily time series
    overall_mean_bs_02mm = np.nanmean(daily_bs_series)
    
    print(f"\n--- Overall Validation Metrics (Fold {fold}) ---")
    print(f"Spatially Averaged Mean CRPS (area-weighted): {overall_mean_crps:.4f}")
    print(f"Spatially Averaged Mean Brier Score (area-weighted, Thresh={brier_threshold:.1f}mm): {overall_mean_bs_02mm:.4f}")
    print(f"Number of valid cells used for averaging: {len(all_cell_mean_crps)}")
    
    # Calculate seasonal CRPS and BS from daily time series
    seasonal_crps_from_daily = {}
    seasonal_bs_from_daily = {}
    if timestamps is not None and len(daily_crps_series) > 0:
        for season in seasons:
            season_mask = np.array([month_to_season(t.month) == season for t in timestamps])
            # CRPS
            season_daily_crps = daily_crps_series[season_mask]
            if len(season_daily_crps) > 0:
                seasonal_crps_from_daily[season] = np.nanmean(season_daily_crps)
            else:
                seasonal_crps_from_daily[season] = np.nan
            # BS
            season_daily_bs = daily_bs_series[season_mask]
            if len(season_daily_bs) > 0:
                seasonal_bs_from_daily[season] = np.nanmean(season_daily_bs)
            else:
                seasonal_bs_from_daily[season] = np.nan
                
        print(f"\nSeasonal metrics from daily time series:")
        for season in seasons:
            print(f"  {season}: CRPS={seasonal_crps_from_daily.get(season, np.nan):.4f}, "
                  f"BS={seasonal_bs_from_daily.get(season, np.nan):.4f}")
    
    # Build MPC baseline from training data only
    print("\n--- Building MPC (Monthly Probabilistic Climatology) baseline ---")
    print("Using training years only to avoid data leakage...")
    
    # Generate training dates based on the shape of training data
    # Assuming daily data and that training ends just before validation starts
    n_train_samples = train_targets.shape[0]
    if val_times is not None and len(val_times) > 0:
        val_start = val_times[0]
        # Work backwards from validation start date
        train_end = val_start - pd.Timedelta(days=1)
        train_dates = pd.date_range(end=train_end, periods=n_train_samples, freq='D')
        train_years = sorted(set(train_dates.year))
        print(f"Training period: {train_dates[0].date()} to {train_dates[-1].date()}")
        print(f"Training years: {train_years[0]}-{train_years[-1]}")
        print(f"Validation year: {val_start.year}")
    else:
        # Fallback: create synthetic dates
        print("Warning: No validation timestamps available, using synthetic training dates")
        train_dates = pd.date_range(start='2000-01-01', periods=n_train_samples, freq='D')
    
    # Build MPC from training observations
    mpc_climatology = build_mpc_climatology(train_targets, train_dates, mask)
    
    # Compute MPC CRPS for validation period
    print("Computing MPC CRPS time series...")
    daily_crps_mpc = np.full(len(val_times), np.nan)
    
    for t in range(len(val_times)):
        # Initialize CRPS map for this day
        crps_map_mpc = np.full((grid_lat, grid_lon), np.nan)
        current_month = val_times[t].month
        
        for i in range(grid_lat):
            for j in range(grid_lon):
                if not mask[i, j]:
                    continue
                
                # Get climatology samples for this cell and month
                if current_month in mpc_climatology and len(mpc_climatology[current_month][i][j]) > 0:
                    clim_samples = np.array(mpc_climatology[current_month][i][j])
                    obs_value = val_targets[t, i, j]
                    
                    if np.isfinite(obs_value):
                        # Compute CRPS using sample distribution
                        crps_mpc = crps_sample_distribution(obs_value, clim_samples)
                        if np.isfinite(crps_mpc):
                            crps_map_mpc[i, j] = crps_mpc
        
        # Compute area-weighted mean for this day
        if np.any(np.isfinite(crps_map_mpc)):
            daily_crps_mpc[t] = spatial_weighted_mean(crps_map_mpc, coslat_weights(lat_2d, mask))
    
    # Calculate overall and seasonal MPC CRPS
    overall_mean_crps_mpc = np.nanmean(daily_crps_mpc)
    print(f"\nMPC baseline - Overall mean CRPS (area-weighted): {overall_mean_crps_mpc:.4f}")
    
    # Calculate CRPSS (skill score)
    crpss_overall = skill_score(overall_mean_crps, overall_mean_crps_mpc)
    print(f"Overall CRPSS (vs MPC): {crpss_overall:.4f}")
    
    # Seasonal MPC CRPS and skill scores
    seasonal_crps_mpc = {}
    seasonal_crpss = {}
    if timestamps is not None:
        for season in seasons:
            season_mask = np.array([month_to_season(t.month) == season for t in timestamps])
            season_daily_mpc = daily_crps_mpc[season_mask]
            if len(season_daily_mpc) > 0:
                seasonal_crps_mpc[season] = np.nanmean(season_daily_mpc)
                seasonal_crpss[season] = skill_score(
                    seasonal_crps_from_daily.get(season, np.nan),
                    seasonal_crps_mpc[season]
                )
        
        print("\nSeasonal CRPSS (vs MPC):")
        for season in seasons:
            print(f"  {season}: CRPS_model={seasonal_crps_from_daily.get(season, np.nan):.4f}, "
                  f"CRPS_MPC={seasonal_crps_mpc.get(season, np.nan):.4f}, "
                  f"CRPSS={seasonal_crpss.get(season, np.nan):.4f}")

    # CORP Brier Score Decomposition
    bsd_results = None
    if all_probs_for_decomp and all_obs_for_decomp:
        final_forecast_probs_bsd = np.array(all_probs_for_decomp)
        final_binary_obs_bsd = np.array(all_obs_for_decomp)
        print(f"\nCalculating CORP Brier Score Decomposition for Germany region (Thresh={brier_threshold:.1f}mm)...")
        print(f"  Total samples for BSD: {len(final_forecast_probs_bsd)}")
        
        bsd_decomp = corp_bs_decomposition(final_forecast_probs_bsd, final_binary_obs_bsd)
        if bsd_decomp is not None:
            print(f"  Mean Brier Score: {bsd_decomp['bs']:.4f}")
            print(f"  MCB (Miscalibration): {bsd_decomp['mcb']:.4f}")
            print(f"  DSC (Discrimination): {bsd_decomp['dsc']:.4f}")
            print(f"  UNC (Uncertainty): {bsd_decomp['unc']:.4f}")
            print(f"  Identity check (BS = MCB - DSC + UNC): {abs(bsd_decomp['bs'] - (bsd_decomp['mcb'] - bsd_decomp['dsc'] + bsd_decomp['unc'])) < 1e-6}")
            
            bsd_results = bsd_decomp
            
            # Check consistency with time series mean
            if np.isfinite(bsd_decomp['bs']) and np.isfinite(overall_mean_bs_02mm):
                if not np.isclose(bsd_decomp['bs'], overall_mean_bs_02mm, rtol=1e-3):
                    print(f"  Note: BSD Mean BS ({bsd_decomp['bs']:.4f}) differs from time series mean ({overall_mean_bs_02mm:.4f}).")
        else:
            print("  CORP Brier Score Decomposition failed.")
    else:
        print("\nNo data available for Brier Score Decomposition.")
    
    # CRPS CORP Decomposition (diagnostic only)
    print("\n--- CRPS CORP Decomposition (diagnostic) ---")
    print("Computing on a representative subset of days...")
    
    # Sample subset of days for CRPS decomposition
    n_days_sample = min(30, len(val_times))  # Sample up to 30 days
    sample_indices = np.random.choice(len(val_times), n_days_sample, replace=False)
    sample_indices.sort()
    
    # Define threshold grid
    thresholds = np.array([0.0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50])
    print(f"Using {len(thresholds)} thresholds: {thresholds}")
    print(f"Sampling {n_days_sample} days from validation period")
    
    # Collect CDF values and observations for sampled days
    all_cdf_values = []
    all_obs_values = []
    
    for t_idx in sample_indices:
        for i in range(grid_lat):
            for j in range(grid_lon):
                if not mask[i, j] or (i, j) not in idr_models_by_cell:
                    continue
                
                idr_pred = idr_models_by_cell[(i, j)]
                if idr_pred is None:
                    continue
                
                try:
                    obs_value = val_targets[t_idx, i, j]
                    if np.isnan(obs_value):
                        continue
                    
                    # Evaluate CDF at thresholds
                    cdf_values = idr_pred.cdf(thresholds=thresholds)
                    if isinstance(cdf_values, np.ndarray) and len(cdf_values) == len(thresholds):
                        all_cdf_values.append(cdf_values)
                        all_obs_values.append(obs_value)
                except:
                    pass
    
    if len(all_cdf_values) > 0:
        # Convert to arrays
        Fz = np.array(all_cdf_values)  # Shape: [n_samples, n_thresholds]
        y = np.array(all_obs_values)   # Shape: [n_samples]
        
        print(f"Collected {len(y)} valid samples for CRPS decomposition")
        
        # Compute CRPS decomposition
        crps_decomp = corp_crps_decomposition_from_cdf(Fz, y, thresholds)
        
        if crps_decomp is not None:
            print(f"\nCRPS CORP Decomposition Results:")
            print(f"  CRPS: {crps_decomp['crps']:.4f}")
            print(f"  MCB (Miscalibration): {crps_decomp['mcb']:.4f}")
            print(f"  DSC (Discrimination): {crps_decomp['dsc']:.4f}")
            print(f"  UNC (Uncertainty): {crps_decomp['unc']:.4f}")
            print(f"  Identity check (CRPS = MCB - DSC + UNC): {abs(crps_decomp['crps'] - (crps_decomp['mcb'] - crps_decomp['dsc'] + crps_decomp['unc'])) < 1e-6}")
            
            # Check consistency with overall mean CRPS
            # Note: This is a subset so exact match isn't expected
            print(f"\n  Subset CRPS ({crps_decomp['crps']:.4f}) vs Overall mean CRPS ({overall_mean_crps:.4f})")
            print(f"  (Difference expected due to sampling)")
            
            # Store decomposition results
            crps_decomp_results = crps_decomp
        else:
            print("CRPS decomposition failed")
            crps_decomp_results = None
    else:
        print("No valid samples collected for CRPS decomposition")
        crps_decomp_results = None

    intensity_bin_crps = {}
    for bin_name_calc in bin_names:
        bin_values = [v for v in cell_metrics.get(f'crps_{bin_name_calc}', {}).values() if np.isfinite(v)]
        intensity_bin_crps[bin_name_calc] = np.mean(bin_values) if bin_values else np.nan

    print("\n--- Deterministic vs Probabilistic Performance Comparison ---")
    print(f"{'Bin':<10} {'Det. MAE':<12} {'Prob. CRPS':<12} {'Improvement %':<14}")
    print("-" * 50)
    intensity_bin_mae = {}
    for bin_name_comp in bin_names:
        bin_values_mae = [v for v in det_metrics.get(f'mae_{bin_name_comp}', {}).values() if np.isfinite(v)]
        intensity_bin_mae[bin_name_comp] = np.mean(bin_values_mae) if bin_values_mae else np.nan
    if np.isfinite(overall_mae) and np.isfinite(overall_mean_crps):
        improvement = 100 * (overall_mae - overall_mean_crps) / overall_mae if overall_mae != 0 else 0
        print(f"{'Overall':<10} {overall_mae:<12.4f} {overall_mean_crps:<12.4f} {improvement:<14.2f}")
    else:
        print(f"{'Overall':<10} {overall_mae:<12.4f} {overall_mean_crps:<12.4f} {'N/A':<14}")
    for bin_name_print in bin_names:
        bin_mae_val = intensity_bin_mae.get(bin_name_print, np.nan)
        bin_crps_val_comp = intensity_bin_crps.get(bin_name_print, np.nan)
        if np.isfinite(bin_mae_val) and np.isfinite(bin_crps_val_comp):
            improvement_bin = 100 * (bin_mae_val - bin_crps_val_comp) / bin_mae_val if bin_mae_val != 0 else 0
            print(f"{bin_name_print:<10} {bin_mae_val:<12.4f} {bin_crps_val_comp:<12.4f} {improvement_bin:<14.2f}")
        else:
            print(f"{bin_name_print:<10} {bin_mae_val:<12.4f} {bin_crps_val_comp:<12.4f} {'N/A':<14}")
    print("-" * 50)
    print("Note: CRPS for probabilistic forecast should be compared to MAE for deterministic forecast.")
    print("      Improvement % shows benefit of IDR calibration, especially for extreme events.")

    results_summary = {
        'fold': fold,
        'overall_mean_crps': overall_mean_crps,
        'overall_mean_crps_mpc': overall_mean_crps_mpc,
        'overall_crpss': crpss_overall,
        'overall_mean_bs_0.2mm': overall_mean_bs_02mm,
        'brier_score_decomposition_0.2mm': bsd_results,
        'num_valid_cells': len(all_cell_mean_crps),
        'deterministic_mae': overall_mae,
        'deterministic_rmse': overall_rmse,
        'intensity_bin_metrics': {
            'deterministic_mae': intensity_bin_mae,
            'probabilistic_crps': intensity_bin_crps
        },
        'evaluation_region': 'Germany',
        'area_weighting': 'cos(latitude)',
        'mpc_baseline': {
            'overall_crps': overall_mean_crps_mpc,
            'seasonal_crps': seasonal_crps_mpc
        },
        'skill_scores': {
            'crpss_overall': crpss_overall,
            'crpss_seasonal': seasonal_crpss
        },
        'crps_decomposition': crps_decomp_results
    }

    if timestamps is None:
        print("Warning: Timestamps not found. Skipping seasonal analysis.")
        results_summary['seasonal_metrics'] = None
    else:
        seasonal_metrics_results = {}
        print("\nCalculating final seasonal metrics from daily time series...")
        for season_res in seasons:
            # Use the pre-calculated seasonal CRPS from daily time series
            mean_crps_s = seasonal_crps_from_daily.get(season_res, np.nan)
            
            # Use the pre-calculated seasonal BS from daily time series
            mean_bs_s = seasonal_bs_from_daily.get(season_res, np.nan)
            
            # Calculate seasonal MAE from aggregates if available
            if seasonal_aggregates is not None:
                season_data = seasonal_aggregates[season_res]
                mae_count = season_data.get('mae_count', 0)
                mae_sum = season_data.get('mae_sum', 0.0)
                mean_mae_s = mae_sum / mae_count if mae_count > 0 else np.nan
            else:
                mean_mae_s = np.nan
                mae_count = 0
            
            # Count number of days in this season
            season_mask = np.array([month_to_season(t.month) == season_res for t in timestamps])
            num_days = np.sum(season_mask)
            
            seasonal_metrics_results[season_res] = {
                'mean_crps': mean_crps_s,
                'mean_bs_0.2mm': mean_bs_s,
                'mean_mae': mean_mae_s,
                'num_days': num_days,
                'mae_samples': mae_count
            }
            if np.isfinite(mean_mae_s) and np.isfinite(mean_crps_s) and mean_mae_s != 0:
                improvement_s = 100 * (mean_mae_s - mean_crps_s) / mean_mae_s
                seasonal_metrics_results[season_res]['crps_vs_mae_improvement'] = improvement_s
                print(f"  {season_res}: Mean CRPS = {mean_crps_s:.4f}, Mean BS = {mean_bs_s:.4f}, Mean MAE = {mean_mae_s:.4f}, Improv. {improvement_s:.2f}% ({num_days} days)")
            else:
                print(f"  {season_res}: Mean CRPS = {mean_crps_s:.4f}, Mean BS = {mean_bs_s:.4f}, Mean MAE = {mean_mae_s:.4f} ({num_days} days)")
        results_summary['seasonal_metrics'] = seasonal_metrics_results

    print(f"\n--- Evaluation Summary for Fold {fold} ---") # Corrected f-string
    print(f"Overall Mean CRPS: {results_summary.get('overall_mean_crps', np.nan):.4f}")
    print(f"Overall Mean BS (Thresh={brier_threshold:.1f}mm): {results_summary.get('overall_mean_bs_0.2mm', np.nan):.4f}")
    if results_summary.get('brier_score_decomposition_0.2mm'):
        bsd_sum = results_summary['brier_score_decomposition_0.2mm']
        print(f"  BSD Mean BS: {bsd_sum.get('mean_bs', np.nan):.4f}")
        print(f"  BSD Miscalibration: {bsd_sum.get('miscalibration', np.nan):.4f}")
        print(f"  BSD Discrimination: {bsd_sum.get('discrimination', np.nan):.4f}")
        print(f"  BSD Uncertainty: {bsd_sum.get('uncertainty', np.nan):.4f}")
    print(f"Overall Mean MAE (Deterministic): {results_summary.get('deterministic_mae', np.nan):.4f}")
    print(f"Number of Valid Cells: {results_summary.get('num_valid_cells', 'N/A')}")
    print("\nSeasonal Metrics:")
    print(f"{'Season':<6} {'Mean CRPS':<12} {'Mean BS':<12} {'Mean MAE':<12} {'Improvement':<10}")
    print("-" * 55)
    seasonal_metrics_disp = results_summary.get('seasonal_metrics')
    if seasonal_metrics_disp:
        for season_disp, metrics_disp in seasonal_metrics_disp.items():
            crps_disp = metrics_disp.get('mean_crps', np.nan)
            bs_disp = metrics_disp.get('mean_bs_0.2mm', np.nan)
            mae_disp = metrics_disp.get('mean_mae', np.nan)
            imp_disp = metrics_disp.get('crps_vs_mae_improvement', np.nan)
            imp_str_disp = f"{imp_disp:.2f}%" if np.isfinite(imp_disp) else "N/A"
            print(f"{season_disp:<6} {crps_disp:<12.4f} {bs_disp:<12.4f} {mae_disp:<12.4f} {imp_str_disp:<10}")
    else:
        print("  (Seasonal metrics not available)")
    print("-" * 55)

    json_output_path = os.path.join(fold_dir, f"evaluation_summary_fold{fold}.json")
    try:
        with open(json_output_path, 'w') as f:
            json.dump(results_summary, f, indent=4, default=numpy_json_encoder)
        print(f"Evaluation summary saved to: {json_output_path}")
    except Exception as e:
        print(f"Error saving evaluation summary to JSON: {e}")
        try:
             with open(json_output_path.replace('.json', '_fallback.json'), 'w') as f_fallback:
                  json.dump(results_summary, f_fallback, indent=4, default=str)
             print(f"Fallback summary saved with string conversion.")
        except Exception as e_fallback:
              print(f"Error during fallback JSON saving: {e_fallback}")

    # Generate publication-ready plots
    print("\nGenerating publication-ready plots...")
    
    # Replace old plot_seasonal_metrics with new plot_seasonal_bars
    if results_summary.get('seasonal_metrics'):
        # Seasonal CRPS bars
        seasonal_crps_dict = {}
        seasonal_bs_dict = {}
        for season, metrics in results_summary['seasonal_metrics'].items():
            seasonal_crps_dict[season] = metrics.get('mean_crps', np.nan)
            seasonal_bs_dict[season] = metrics.get('mean_bs_0.2mm', np.nan)
        
        crps_outfile = os.path.join(fold_dir, f"seasonal_crps_fold{fold}.png")
        plot_seasonal_bars(seasonal_crps_dict, 'Mean CRPS', f'Fold {fold}: Seasonal CRPS', crps_outfile)
        
        bs_outfile = os.path.join(fold_dir, f"seasonal_bs_fold{fold}.png")
        plot_seasonal_bars(seasonal_bs_dict, 'Mean Brier Score (0.2 mm)', f'Fold {fold}: Seasonal Brier Score', bs_outfile)
    
    # Intensity bin comparison plot
    if results_summary.get('intensity_bin_metrics'):
        det_mae = results_summary['intensity_bin_metrics'].get('deterministic_mae', {})
        prob_crps = results_summary['intensity_bin_metrics'].get('probabilistic_crps', {})
        bins = ['0.0-0.1', '0.1-1.0', '1.0-5.0', '5.0-10.0', '10.0-20.0', '20.0-50.0', '>50.0']
        
        intensity_outfile = os.path.join(fold_dir, f"intensity_comparison_fold{fold}.png")
        plot_intensity_bars(bins, det_mae, prob_crps, intensity_outfile)
    
    # MCB-DSC scatter plots
    mcb_dsc_points = []
    
    # Add model point if decomposition exists
    if results_summary.get('brier_score_decomposition_0.2mm'):
        bsd = results_summary['brier_score_decomposition_0.2mm']
        mcb_dsc_points.append({
            'label': f'Fold {fold}',
            'MCB': bsd.get('mcb', np.nan),
            'DSC': bsd.get('dsc', np.nan),
            'UNC': bsd.get('unc', np.nan)
        })
    
    # Add MPC baseline point (would need to compute BS decomposition for MPC)
    # For now, just adding a placeholder - in production, compute actual MPC decomposition
    
    if mcb_dsc_points:
        # BS decomposition scatter
        bs_scatter_outfile = os.path.join(fold_dir, f"bs_mcb_dsc_fold{fold}.png")
        plot_mcb_dsc_scatter(mcb_dsc_points, 'Brier Score', bs_scatter_outfile)
        
        # CRPS decomposition scatter (if available)
        if results_summary.get('crps_decomposition'):
            crps_decomp = results_summary['crps_decomposition']
            crps_points = [{
                'label': f'Fold {fold}',
                'MCB': crps_decomp.get('mcb', np.nan),
                'DSC': crps_decomp.get('dsc', np.nan),
                'UNC': crps_decomp.get('unc', np.nan)
            }]
            crps_scatter_outfile = os.path.join(fold_dir, f"crps_mcb_dsc_fold{fold}.png")
            plot_mcb_dsc_scatter(crps_points, 'CRPS', crps_scatter_outfile)
    
    # Keep the old comparison plot for backward compatibility
    plot_det_vs_prob_comparison(results_summary, base_output_dir, fold)

    if generate_seasonal_plots:
        print("\nGenerating seasonal sample plots...")
        try:
            plot_seasonal_samples(fold_dir, fold, num_samples_per_season=5, lats=current_lats, lons=current_lons)
            print("Seasonal plots generation completed.")
        except Exception as e:
            print(f"Error generating seasonal plots: {e}")
            import traceback
            traceback.print_exc()

    # Write results to CSV
    csv_filename = os.path.join(fold_dir, f"evaluation_results_fold{fold}.csv")
    write_results_to_csv(results_summary, csv_filename)
    
    # Run acceptance tests
    tests_passed = run_acceptance_tests(results_summary, daily_crps_series, timestamps, seasonal_crps_from_daily)
    
    print("Finished evaluation.")
    return results_summary


def write_results_to_csv(results_summary, csv_filename):
    """
    Write evaluation results to CSV format suitable for paper tables.
    
    Parameters
    ----------
    results_summary : dict
        Dictionary containing all evaluation results.
    csv_filename : str
        Path to output CSV file.
    """
    import csv
    
    print(f"\nWriting results to CSV: {csv_filename}")
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow(['Metric', 'Value', 'Description'])
        writer.writerow([])  # Empty row
        
        # Overall metrics
        writer.writerow(['# Overall Metrics (Germany, area-weighted)', '', ''])
        writer.writerow(['overall_mean_crps', f"{results_summary.get('overall_mean_crps', np.nan):.4f}", 'Mean CRPS'])
        writer.writerow(['overall_mean_crps_mpc', f"{results_summary.get('overall_mean_crps_mpc', np.nan):.4f}", 'Mean CRPS (MPC baseline)'])
        writer.writerow(['overall_crpss', f"{results_summary.get('overall_crpss', np.nan):.4f}", 'CRPSS vs MPC'])
        writer.writerow(['overall_mean_bs_0.2mm', f"{results_summary.get('overall_mean_bs_0.2mm', np.nan):.4f}", 'Mean Brier Score (0.2mm)'])
        writer.writerow(['deterministic_mae', f"{results_summary.get('deterministic_mae', np.nan):.4f}", 'Deterministic MAE'])
        writer.writerow(['deterministic_rmse', f"{results_summary.get('deterministic_rmse', np.nan):.4f}", 'Deterministic RMSE'])
        writer.writerow([])
        
        # Brier Score decomposition
        if results_summary.get('brier_score_decomposition_0.2mm'):
            bsd = results_summary['brier_score_decomposition_0.2mm']
            writer.writerow(['# Brier Score CORP Decomposition (0.2mm)', '', ''])
            writer.writerow(['bs_mean', f"{bsd.get('bs', np.nan):.4f}", 'Mean Brier Score'])
            writer.writerow(['bs_mcb', f"{bsd.get('mcb', np.nan):.4f}", 'MCB (Miscalibration)'])
            writer.writerow(['bs_dsc', f"{bsd.get('dsc', np.nan):.4f}", 'DSC (Discrimination)'])
            writer.writerow(['bs_unc', f"{bsd.get('unc', np.nan):.4f}", 'UNC (Uncertainty)'])
            writer.writerow([])
        
        # CRPS decomposition
        if results_summary.get('crps_decomposition'):
            crps_d = results_summary['crps_decomposition']
            writer.writerow(['# CRPS CORP Decomposition (diagnostic subset)', '', ''])
            writer.writerow(['crps_subset', f"{crps_d.get('crps', np.nan):.4f}", 'CRPS (subset)'])
            writer.writerow(['crps_mcb', f"{crps_d.get('mcb', np.nan):.4f}", 'MCB (Miscalibration)'])
            writer.writerow(['crps_dsc', f"{crps_d.get('dsc', np.nan):.4f}", 'DSC (Discrimination)'])
            writer.writerow(['crps_unc', f"{crps_d.get('unc', np.nan):.4f}", 'UNC (Uncertainty)'])
            writer.writerow([])
        
        # Seasonal metrics
        if results_summary.get('seasonal_metrics'):
            writer.writerow(['# Seasonal Metrics', '', ''])
            seasons = ['DJF', 'MAM', 'JJA', 'SON']
            
            # CRPS by season
            writer.writerow(['## CRPS by season', '', ''])
            for season in seasons:
                if season in results_summary['seasonal_metrics']:
                    val = results_summary['seasonal_metrics'][season].get('mean_crps', np.nan)
                    writer.writerow([f'crps_{season}', f"{val:.4f}", f'{season} CRPS'])
            writer.writerow([])
            
            # BS by season
            writer.writerow(['## Brier Score by season (0.2mm)', '', ''])
            for season in seasons:
                if season in results_summary['seasonal_metrics']:
                    val = results_summary['seasonal_metrics'][season].get('mean_bs_0.2mm', np.nan)
                    writer.writerow([f'bs_{season}', f"{val:.4f}", f'{season} BS'])
            writer.writerow([])
            
            # CRPSS by season
            if results_summary.get('skill_scores', {}).get('crpss_seasonal'):
                writer.writerow(['## CRPSS by season (vs MPC)', '', ''])
                crpss_seasonal = results_summary['skill_scores']['crpss_seasonal']
                for season in seasons:
                    if season in crpss_seasonal:
                        val = crpss_seasonal[season]
                        writer.writerow([f'crpss_{season}', f"{val:.4f}", f'{season} CRPSS'])
                writer.writerow([])
        
        # Intensity bin metrics
        if results_summary.get('intensity_bin_metrics'):
            writer.writerow(['# Intensity Bin Metrics', '', ''])
            bins = ['0.0-0.1', '0.1-1.0', '1.0-5.0', '5.0-10.0', '10.0-20.0', '20.0-50.0', '>50.0']
            
            # MAE by bin
            writer.writerow(['## Deterministic MAE by precipitation intensity', '', ''])
            mae_dict = results_summary['intensity_bin_metrics'].get('deterministic_mae', {})
            for bin_name in bins:
                if bin_name in mae_dict:
                    val = mae_dict[bin_name]
                    writer.writerow([f'mae_{bin_name}mm', f"{val:.4f}", f'MAE for {bin_name} mm'])
            writer.writerow([])
            
            # CRPS by bin
            writer.writerow(['## Probabilistic CRPS by precipitation intensity', '', ''])
            crps_dict = results_summary['intensity_bin_metrics'].get('probabilistic_crps', {})
            for bin_name in bins:
                if bin_name in crps_dict:
                    val = crps_dict[bin_name]
                    writer.writerow([f'crps_{bin_name}mm', f"{val:.4f}", f'CRPS for {bin_name} mm'])
            writer.writerow([])
        
        # Metadata
        writer.writerow(['# Metadata', '', ''])
        writer.writerow(['fold', results_summary.get('fold', 'NA'), 'Fold number'])
        writer.writerow(['run_id', results_summary.get('run_id', ''), 'Run identifier'])
        writer.writerow(['outside_weight', results_summary.get('outside_weight', ''), 'Outside weight parameter'])
        writer.writerow(['era5_group', results_summary.get('era5_group', ''), 'ERA5 group'])
        writer.writerow(['year_eval', results_summary.get('year_eval', 'NA'), 'Evaluation year'])
        writer.writerow(['threshold_mm', '0.2', 'Threshold for PoP (mm)'])
        writer.writerow(['mask', 'Germany', 'Evaluation mask'])
        writer.writerow(['spatial_weight', 'coslat', 'Spatial weighting method'])
        writer.writerow(['evaluation_region', results_summary.get('evaluation_region', 'NA'), 'Evaluation region'])
        writer.writerow(['area_weighting', results_summary.get('area_weighting', 'NA'), 'Area weighting method'])
        writer.writerow(['num_valid_cells', results_summary.get('num_valid_cells', 'NA'), 'Number of valid cells'])
    
    print(f"CSV file written successfully: {csv_filename}")


def run_acceptance_tests(results_summary, daily_crps_series, timestamps, seasonal_crps_from_daily):
    """
    Run acceptance tests to verify the evaluation results are consistent.
    
    Tests:
    1. Seasonal parity (if comparing with old results)
    2. Overall consistency between daily mean and seasonal weighted mean
    3. Identity check for decompositions
    
    Parameters
    ----------
    results_summary : dict
        Complete evaluation results
    daily_crps_series : np.ndarray
        Daily CRPS time series
    timestamps : pandas.DatetimeIndex
        Timestamps for validation period
    seasonal_crps_from_daily : dict
        Seasonal CRPS values computed from daily series
    """
    print("\n" + "="*60)
    print("Running Acceptance Tests")
    print("="*60)
    
    all_tests_passed = True
    
    # Test 1: Overall consistency
    print("\nTest 1: Overall Consistency")
    print("-"*40)
    
    # Check that overall mean CRPS equals time-mean of daily series
    overall_from_daily = np.nanmean(daily_crps_series)
    overall_reported = results_summary.get('overall_mean_crps', np.nan)
    
    test1a_passed = np.abs(overall_from_daily - overall_reported) < 1e-6
    print(f"  1a. Overall mean CRPS consistency:")
    print(f"      From daily series: {overall_from_daily:.6f}")
    print(f"      Reported:          {overall_reported:.6f}")
    print(f"      Difference:        {np.abs(overall_from_daily - overall_reported):.2e}")
    print(f"      Status: {'PASSED' if test1a_passed else 'FAILED'}")
    all_tests_passed &= test1a_passed
    
    # Check seasonal weighted mean equals overall
    if timestamps is not None and seasonal_crps_from_daily:
        season_days = {}
        for t in timestamps:
            season = month_to_season(t.month)
            season_days[season] = season_days.get(season, 0) + 1
        
        total_days = sum(season_days.values())
        weighted_seasonal_mean = 0
        for season, n_days in season_days.items():
            if season in seasonal_crps_from_daily:
                weight = n_days / total_days
                weighted_seasonal_mean += weight * seasonal_crps_from_daily[season]
        
        test1b_passed = np.abs(weighted_seasonal_mean - overall_reported) < 1e-6
        print(f"\n  1b. Season-weighted mean consistency:")
        print(f"      Weighted seasonal: {weighted_seasonal_mean:.6f}")
        print(f"      Overall reported:  {overall_reported:.6f}")
        print(f"      Difference:        {np.abs(weighted_seasonal_mean - overall_reported):.2e}")
        print(f"      Status: {'PASSED' if test1b_passed else 'FAILED'}")
        all_tests_passed &= test1b_passed
    
    # Test 2: Decomposition identities
    print("\nTest 2: Decomposition Identities")
    print("-"*40)
    
    # Check BS decomposition
    if results_summary.get('brier_score_decomposition_0.2mm'):
        bsd = results_summary['brier_score_decomposition_0.2mm']
        bs = bsd.get('bs', np.nan)
        mcb = bsd.get('mcb', np.nan)
        dsc = bsd.get('dsc', np.nan)
        unc = bsd.get('unc', np.nan)
        
        identity = mcb - dsc + unc
        diff = np.abs(bs - identity)
        test2a_passed = diff < 1e-6
        
        print(f"  2a. Brier Score decomposition (BS = MCB - DSC + UNC):")
        print(f"      BS:                {bs:.6f}")
        print(f"      MCB - DSC + UNC:   {identity:.6f}")
        print(f"      Difference:        {diff:.2e}")
        print(f"      Status: {'PASSED' if test2a_passed else 'FAILED'}")
        all_tests_passed &= test2a_passed
    
    # Check CRPS decomposition
    if results_summary.get('crps_decomposition'):
        crps_d = results_summary['crps_decomposition']
        crps = crps_d.get('crps', np.nan)
        mcb = crps_d.get('mcb', np.nan)
        dsc = crps_d.get('dsc', np.nan)
        unc = crps_d.get('unc', np.nan)
        
        identity = mcb - dsc + unc
        diff = np.abs(crps - identity)
        test2b_passed = diff < 1e-6
        
        print(f"\n  2b. CRPS decomposition (CRPS = MCB - DSC + UNC):")
        print(f"      CRPS:              {crps:.6f}")
        print(f"      MCB - DSC + UNC:   {identity:.6f}")
        print(f"      Difference:        {diff:.2e}")
        print(f"      Status: {'PASSED' if test2b_passed else 'FAILED'}")
        all_tests_passed &= test2b_passed
    
    # Test 3: Value sanity checks
    print("\nTest 3: Value Sanity Checks")
    print("-"*40)
    
    # CRPS should be positive
    test3a_passed = results_summary.get('overall_mean_crps', -1) >= 0
    print(f"  3a. CRPS >= 0: {'PASSED' if test3a_passed else 'FAILED'}")
    all_tests_passed &= test3a_passed
    
    # BS should be between 0 and 1
    bs_val = results_summary.get('overall_mean_bs_0.2mm', -1)
    test3b_passed = 0 <= bs_val <= 1
    print(f"  3b. 0 <= BS <= 1: {'PASSED' if test3b_passed else 'FAILED'} (BS = {bs_val:.4f})")
    all_tests_passed &= test3b_passed
    
    # CRPSS should be less than 1 (perfect forecast would be 1)
    crpss = results_summary.get('overall_crpss', 2)
    test3c_passed = crpss < 1
    print(f"  3c. CRPSS < 1: {'PASSED' if test3c_passed else 'FAILED'} (CRPSS = {crpss:.4f})")
    all_tests_passed &= test3c_passed
    
    print("\n" + "="*60)
    print(f"Overall Test Result: {'ALL TESTS PASSED' if all_tests_passed else 'SOME TESTS FAILED'}")
    print("="*60 + "\n")
    
    return all_tests_passed


def plot_det_vs_prob_comparison(results_summary: dict, save_path: str, fold_label: str):
    """
    Generates and saves a bar chart comparing deterministic MAE vs probabilistic CRPS
    across different precipitation intensity bins.

    Args:
        results_summary (dict): Dictionary containing evaluation results,
                              including intensity bin metrics.
        save_path (str): Full path where the plot will be saved.
        fold_label (str): Label for the fold (e.g., "Fold 0", "Comprehensive").
    """
    if 'intensity_bin_metrics' not in results_summary:
        print("Intensity bin metrics not found in results. Skipping comparison plot.")
        return

    bin_metrics = results_summary['intensity_bin_metrics']
    det_mae = bin_metrics.get('deterministic_mae', {})
    prob_crps = bin_metrics.get('probabilistic_crps', {})

    if not det_mae or not prob_crps:
        print("Missing deterministic MAE or probabilistic CRPS data. Skipping comparison plot.")
        return

    evaluation_region = results_summary.get('evaluation_region', 'Germany')
    area_weighting = results_summary.get('area_weighting', 'cos(latitude)')
    region_suffix = f' - {evaluation_region} Region' if evaluation_region else ''

    bins_plot = [0, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, float('inf')]
    bin_names_plot = [f"{bins_plot[i]:.1f}-{bins_plot[i+1]:.1f}" if bins_plot[i+1] != float('inf') else f">{bins_plot[i]:.1f}"
                      for i in range(len(bins_plot)-1)]
    valid_bins = [b for b in bin_names_plot if b in det_mae and b in prob_crps]

    if not valid_bins:
        print("No matching bin names found in both metrics for plot. Skipping comparison plot.")
        return

    mae_values = [det_mae.get(b, np.nan) for b in valid_bins]
    crps_values = [prob_crps.get(b, np.nan) for b in valid_bins]

    improvements = []
    for mae, crps in zip(mae_values, crps_values):
        if np.isfinite(mae) and np.isfinite(crps) and mae != 0:
            improvements.append(100 * (mae - crps) / mae)
        else:
            improvements.append(np.nan)

    # Define single dark gray color
    bar_color_light = '#6c6c6c'  # Slightly lighter gray for MAE
    bar_color_dark = '#4c4c4c'   # Dark gray for CRPS

    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})

        x_plot = np.arange(len(valid_bins))
        width = 0.35

        rects1 = ax1.bar(x_plot - width/2, mae_values, width, label='Deterministic MAE', color=bar_color_light)
        rects2 = ax1.bar(x_plot + width/2, crps_values, width, label='Probabilistic CRPS', color=bar_color_dark)

        ax1.set_ylabel('Error Metric Value')
        ax1.set_title(f'{fold_label}: Det. MAE vs Prob. CRPS by Precipitation Intensity{region_suffix}\nSpatially Area-Weighted ({area_weighting})')
        ax1.set_xticks(x_plot)
        ax1.set_xticklabels(valid_bins, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        def autolabel(rects, ax_target):
            for rect in rects:
                height = rect.get_height()
                if np.isfinite(height):
                    ax_target.annotate(f'{height:.3f}',
                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=8)
        autolabel(rects1, ax1)
        autolabel(rects2, ax1)

        # Use dark gray for improvement bars
        rects_imp = ax2.bar(x_plot, improvements, width=0.6, color=bar_color_dark)
        ax2.set_ylabel('Improvement (%)')
        ax2.set_xticks(x_plot)
        ax_labels_imp = [vb.replace("inf", "∞") for vb in valid_bins]
        ax2.set_xticklabels(ax_labels_imp, rotation=45, ha="right")
        ax2.set_xlabel('Precipitation Intensity (mm)')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')

        for i, imp_val in enumerate(improvements):
            if np.isfinite(imp_val):
                color_imp = 'black'  # Use black text for all values
                va_imp = 'bottom' if imp_val >=0 else 'top'
                offset = 3 if imp_val >=0 else -3
                ax2.text(x_plot[i], imp_val + offset if va_imp == 'bottom' else imp_val + offset,
                        f'{imp_val:.1f}%', ha='center', va=va_imp,
                        fontsize=9, weight='bold', color=color_imp)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure directory exists
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved deterministic vs probabilistic comparison plot to: {save_path}")
    except Exception as e:
        print(f"Error generating comparison plot: {e}")
        plt.close()


def plot_seasonal_bars(values_dict: dict, ylabel: str, title: str, outfile: str):
    """
    Generate publication-ready seasonal bar plot with neutral styling.
    
    Args:
        values_dict: Dictionary with seasons as keys ("DJF", "MAM", "JJA", "SON") and values
        ylabel: Y-axis label
        title: Plot title (will append Germany/cos(lat) info)
        outfile: Output file path
    """
    try:
        # Extract seasons and values in order
        seasons = ['DJF', 'MAM', 'JJA', 'SON']
        values = [values_dict.get(s, np.nan) for s in seasons]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot bars with specified styling
        bars = ax.bar(seasons, values, color='#4c4c4c', edgecolor='black', linewidth=1.0)
        
        # Add value labels on top of bars
        for i, (season, val) in enumerate(zip(seasons, values)):
            if np.isfinite(val):
                ax.text(i, val + max(values) * 0.01, f'{val:.3f}', 
                       ha='center', va='bottom', fontsize=10)
        
        # Styling
        ax.set_xlabel('Season', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{title} (Germany; cos(lat) area-weighted)', fontsize=14)
        
        # Grid - dotted light gray on y-axis only
        ax.grid(axis='y', linestyle=':', color='lightgray', alpha=0.7)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved seasonal bar plot to: {outfile}")
        
    except Exception as e:
        print(f"Error generating seasonal bar plot: {e}")
        if 'fig' in locals():
            plt.close(fig)


def plot_intensity_bars(bins: list, det_mae: dict, prob_crps: dict, outfile: str):
    """
    Generate intensity bin comparison plot with two panels.
    
    Args:
        bins: List of bin labels (e.g., ["0.0-0.1", "0.1-1.0", ...])
        det_mae: Dictionary of MAE values by bin
        prob_crps: Dictionary of CRPS values by bin
        outfile: Output file path
    """
    try:
        # Extract values in order
        mae_values = [det_mae.get(b, np.nan) for b in bins]
        crps_values = [prob_crps.get(b, np.nan) for b in bins]
        
        # Calculate improvement percentages
        improvements = []
        for mae, crps in zip(mae_values, crps_values):
            if np.isfinite(mae) and np.isfinite(crps) and mae > 0:
                improvements.append(100 * (mae - crps) / mae)
            else:
                improvements.append(np.nan)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), 
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        # Top panel: MAE vs CRPS comparison
        x = np.arange(len(bins))
        width = 0.35
        
        # Use muted blue and rose colors
        bars1 = ax1.bar(x - width/2, mae_values, width, 
                        label='Deterministic MAE', color='#5B7C99', edgecolor='black')
        bars2 = ax1.bar(x + width/2, crps_values, width, 
                        label='Probabilistic CRPS', color='#CC8899', edgecolor='black')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar, val in zip(bars, mae_values if bars == bars1 else crps_values):
                if np.isfinite(val):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2, height + max(mae_values + crps_values) * 0.01,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax1.set_ylabel('Error Metric Value', fontsize=12)
        ax1.set_title('Intensity-Stratified Performance (Germany; cos(lat) area-weighted)', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(bins, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', linestyle=':', color='lightgray', alpha=0.7)
        
        # Bottom panel: Improvement percentage
        bars_imp = ax2.bar(x, improvements, color='#228B22', edgecolor='black')
        
        # Add percentage labels
        for bar, imp in zip(bars_imp, improvements):
            if np.isfinite(imp):
                ax2.text(bar.get_x() + bar.get_width()/2, 
                        imp + 1 if imp >= 0 else imp - 1,
                        f'{imp:.1f}%', ha='center', 
                        va='bottom' if imp >= 0 else 'top',
                        fontsize=9, weight='bold')
        
        ax2.set_ylabel('Improvement (%)', fontsize=12)
        ax2.set_xlabel('Precipitation Intensity (mm)', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(bins, rotation=45, ha='right')
        ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax2.grid(axis='y', linestyle=':', color='lightgray', alpha=0.7)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved intensity bars plot to: {outfile}")
        
    except Exception as e:
        print(f"Error generating intensity bars plot: {e}")
        if 'fig' in locals():
            plt.close(fig)


def plot_mcb_dsc_scatter(points: list, score_name: str, outfile: str):
    """
    Generate MCB-DSC scatter plot with isopleths.
    
    Args:
        points: List of dicts with keys "label", "MCB", "DSC", and optionally "UNC"
        score_name: Name of the score (e.g., "CRPS" or "Brier Score")
        outfile: Output file path
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Extract MCB and DSC values
        mcb_values = [p['MCB'] for p in points]
        dsc_values = [p['DSC'] for p in points]
        
        # Determine plot limits
        mcb_range = [0, max(mcb_values) * 1.2]
        dsc_range = [0, max(dsc_values) * 1.2]
        
        # Draw isopleths of constant score (S = MCB - DSC + UNC)
        # For visualization, we'll draw lines where MCB - DSC = constant
        # This assumes UNC is relatively constant across models
        isopleth_values = np.arange(-0.5, 1.0, 0.1)
        for iso_val in isopleth_values:
            # Line equation: DSC = MCB - iso_val
            mcb_line = np.linspace(max(0, iso_val), mcb_range[1], 100)
            dsc_line = mcb_line - iso_val
            
            # Only plot where DSC >= 0
            valid = dsc_line >= 0
            if np.any(valid):
                ax.plot(mcb_line[valid], dsc_line[valid], 
                       color='lightgray', linewidth=0.5, alpha=0.5, zorder=1)
        
        # Plot points
        colors = []
        markers = []
        for i, point in enumerate(points):
            # Special styling for MPC baseline and best run
            if 'MPC' in point['label']:
                colors.append('red')
                markers.append('s')  # Square
            elif 'best' in point['label'].lower() or i == 0:  # Assume first is best if not marked
                colors.append('green')
                markers.append('*')  # Star
            else:
                colors.append('#4c4c4c')
                markers.append('o')
        
        # Scatter plot
        for i, (point, color, marker) in enumerate(zip(points, colors, markers)):
            ax.scatter(point['MCB'], point['DSC'], 
                      color=color, marker=marker, s=100, 
                      edgecolor='black', linewidth=1, zorder=2)
            
            # Add labels
            ax.annotate(point['label'], 
                       (point['MCB'], point['DSC']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, ha='left')
        
        # Add diagonal reference line (MCB = DSC)
        diag_max = min(mcb_range[1], dsc_range[1])
        ax.plot([0, diag_max], [0, diag_max], 'k--', linewidth=1, alpha=0.5, label='MCB = DSC')
        
        # Labels and title
        ax.set_xlabel('MCB (Miscalibration)', fontsize=12)
        ax.set_ylabel('DSC (Discrimination)', fontsize=12)
        ax.set_title(f'{score_name} CORP Decomposition (Germany; cos(lat) area-weighted)', fontsize=14)
        
        # Set limits
        ax.set_xlim(0, mcb_range[1])
        ax.set_ylim(0, dsc_range[1])
        
        # Grid
        ax.grid(True, linestyle=':', color='lightgray', alpha=0.7)
        
        # Legend for special points
        legend_elements = []
        if any('MPC' in p['label'] for p in points):
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                            markerfacecolor='red', markersize=8, 
                                            label='MPC Baseline'))
        if any('best' in p['label'].lower() for p in points):
            legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', 
                                            markerfacecolor='green', markersize=10, 
                                            label='Best Run'))
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved MCB-DSC scatter plot to: {outfile}")
        
    except Exception as e:
        print(f"Error generating MCB-DSC scatter plot: {e}")
        if 'fig' in locals():
            plt.close(fig)


# Example of how to call it (add this at the end of the file for testing)
if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run evaluation on precipitation forecasting results.')
    parser.add_argument('--output_dir', type=str, required=True, help='Base output directory where run_* folders are located')
    parser.add_argument('--fold', type=int, default=0, help='Fold number to evaluate (default: 0)')
    parser.add_argument('--run_id', type=str, default=None, help='Specific run ID to evaluate (e.g., "run_20240520_123045"). If not provided, uses the most recent run.')
    parser.add_argument('--all_folds', action='store_true', help='Evaluate all folds in the specified directory')
    parser.add_argument('--max_folds', type=int, default=5, help='Maximum number of folds to check when using --all_folds (default: 5)')
    parser.add_argument('--plot_time_index', type=int, default=0, help='Time index to use for plotting P50 maps (default: 0)')
    parser.add_argument('--plot_cdf', action='store_true', help='Plot CDF for a specific grid cell')
    parser.add_argument('--cdf_lat_idx', type=int, default=20, help='Latitude INDEX for CDF plotting (default: 20, ensure it is within grid bounds)') # Updated help
    parser.add_argument('--cdf_lon_idx', type=int, default=30, help='Longitude INDEX for CDF plotting (default: 30, ensure it is within grid bounds)') # Updated help
    parser.add_argument('--batch_size', type=int, default=50, help='Number of grid cells to process in each batch (default: 50)')
    parser.add_argument('--idr_samples', type=int, default=1000, help='Number of recent training samples for IDR (DEPRECATED - uses full aligned, default: 1000)')
    parser.add_argument('--seasonal_plots', action='store_true', help='Generate seasonal plots for validation year')
    parser.add_argument('--skip_seasonal_plots', action='store_true', help='Skip generating seasonal plots')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory {args.output_dir} not found.")
        exit(1)
    
    target_run_dir = None # Renamed
    if args.run_id:
        target_run_dir = os.path.join(args.output_dir, args.run_id)
        if not os.path.exists(target_run_dir):
            print(f"Error: Run directory {target_run_dir} not found.")
            exit(1)
    else:
        # Find the most recent run directory
        all_run_dirs = [d for d in os.listdir(args.output_dir) if d.startswith("run_") and os.path.isdir(os.path.join(args.output_dir, d))] # Renamed, added isdir
        if not all_run_dirs:
            print(f"Error: No run_* directories found in {args.output_dir}")
            exit(1)
        
        all_run_dirs.sort(reverse=True)
        target_run_dir = os.path.join(args.output_dir, all_run_dirs[0])
        print(f"Using most recent run directory: {target_run_dir}")
    
    # Define coordinate arrays (can be overridden if loaded from data)
    # For now, assume DEFAULT_LONGITUDES and DEFAULT_LATITUDES are globally available for plotting
    # In a real scenario, these might be loaded with the data or passed from a config
    
    if args.all_folds:
        fold_numbers = [] # Renamed
        for d_name in os.listdir(target_run_dir): # Renamed
            if d_name.startswith("fold") and os.path.isdir(os.path.join(target_run_dir, d_name)):
                try:
                    fold_numbers.append(int(d_name.replace("fold", "")))
                except ValueError:
                    print(f"Could not parse fold number from directory: {d_name}")
        fold_numbers.sort()
        
        print("\n" + "!"*70)
        print("! WARNING: Per-fold evaluation mode is deprecated!")
        # ... (rest of warning)
        print("!"*70 + "\n")
        
        if args.max_folds > 0 and len(fold_numbers) > args.max_folds:
            print(f"Limiting evaluation to the first {args.max_folds} folds.")
            fold_numbers = fold_numbers[:args.max_folds]
        
        if not fold_numbers:
            print(f"Error: No fold directories found in {target_run_dir}")
            exit(1)
            
        print(f"Evaluating all {len(fold_numbers)} folds: {fold_numbers}")
        
        all_results = [] # Renamed
        for fold_num_main in fold_numbers: # Renamed
            print(f"\n=== Evaluating fold {fold_num_main} ===")
            current_fold_dir = os.path.join(target_run_dir, f"fold{fold_num_main}") # Renamed
            generate_seasonal = args.seasonal_plots and not args.skip_seasonal_plots
            
            # Run evaluation (which now might use current_lats/lons defined inside if data grid differs)
            fold_result = run_evaluation(target_run_dir, fold_num_main, batch_size=args.batch_size, 
                                   generate_seasonal_plots=generate_seasonal) # n_recent_train_samples is effectively ignored
            if fold_result:
                all_results.append(fold_result)
                print(f"Evaluation for fold {fold_num_main} complete.")
                
                print(f"Generating P50 prediction map for fold {fold_num_main}...")
                # Determine lats/lons for plotting (could be default or adjusted if data loading changed them)
                # For simplicity, passing defaults here. run_evaluation internally adjusts if needed for its logic,
                # but plotting functions here will use these. Ideally, load_evaluation_data would return them.
                plot_quantile_map(current_fold_dir, fold_num_main, time_index=args.plot_time_index, 
                                  lats=DEFAULT_LATITUDES, lons=DEFAULT_LONGITUDES) # Pass them explicitly
                
                if args.plot_cdf:
                    mask_path_main = os.path.join(current_fold_dir, "germany_mask.npy") # Renamed
                    try:
                        mask_main = np.load(mask_path_main) # Renamed
                        # Ensure cdf_lat_idx and cdf_lon_idx are within bounds of the loaded mask_main
                        if not (0 <= args.cdf_lat_idx < mask_main.shape[0] and 0 <= args.cdf_lon_idx < mask_main.shape[1]):
                             print(f"Warning: Provided CDF indices ({args.cdf_lat_idx}, {args.cdf_lon_idx}) are out of bounds for mask shape {mask_main.shape}. Attempting mask center.")
                             raise IndexError("CDF indices out of bounds")

                        if mask_main[args.cdf_lat_idx, args.cdf_lon_idx]:
                             print(f"Plotting CDF for specified cell: ({args.cdf_lat_idx}, {args.cdf_lon_idx})")
                             plot_single_cdf(current_fold_dir, fold_num_main, args.cdf_lat_idx, args.cdf_lon_idx, time_index=args.plot_time_index)
                        else: # Fallback to mask center if specified is not in mask
                             print(f"Specified cell ({args.cdf_lat_idx}, {args.cdf_lon_idx}) not in mask. Attempting mask center.")
                             raise ValueError("Specified cell not in mask")

                    except (FileNotFoundError, IndexError, ValueError) as e_cdf_plot: # More specific exceptions
                        print(f"Warning: Could not use specified/default CDF cell ({e_cdf_plot}). Trying to find mask center...")
                        try:
                            mask_fallback = np.load(mask_path_main) # Renamed
                            true_indices = np.where(mask_fallback)
                            if len(true_indices[0]) > 0: 
                                center_lat_idx = int(np.round(np.mean(true_indices[0]))) # Renamed
                                center_lon_idx = int(np.round(np.mean(true_indices[1]))) # Renamed
                                print(f"Plotting CDF for mask center: ({center_lat_idx}, {center_lon_idx})")
                                plot_single_cdf(current_fold_dir, fold_num_main, center_lat_idx, center_lon_idx, time_index=args.plot_time_index)
                            else:
                                print(f"Warning: Mask at {mask_path_main} contains no True values. Skipping CDF plot.")
                        except Exception as e_cdf_fallback: # Renamed
                            print(f"Warning: Error processing mask for CDF plot center: {e_cdf_fallback}. Skipping CDF plot.")
            else:
                print(f"Evaluation for fold {fold_num_main} failed or produced no results.")
        
        print(f"\nCompleted evaluation of {len(all_results)} folds.")
    else:
        # Evaluate just the specified fold
        print("\n" + "!"*70)
        print("! WARNING: Single fold evaluation mode is deprecated!")
        # ...
        print("!"*70 + "\n")
        
        print(f"\n=== Evaluating fold {args.fold} ===")
        current_fold_dir_single = os.path.join(target_run_dir, f"fold{args.fold}") # Renamed
        generate_seasonal_single = args.seasonal_plots and not args.skip_seasonal_plots # Renamed
        
        result_single = run_evaluation(target_run_dir, args.fold, batch_size=args.batch_size, 
                              generate_seasonal_plots=generate_seasonal_single) # n_recent_train_samples ignored
        if result_single:
            print(f"Evaluation complete for fold {args.fold}.")
            
            print(f"Generating P50 prediction map for fold {args.fold}...")
            plot_quantile_map(current_fold_dir_single, args.fold, time_index=args.plot_time_index,
                              lats=DEFAULT_LATITUDES, lons=DEFAULT_LONGITUDES) # Pass them
            
            if args.plot_cdf:
                mask_path_single = os.path.join(current_fold_dir_single, "germany_mask.npy") # Renamed
                try:
                    mask_single = np.load(mask_path_single) # Renamed
                    if not (0 <= args.cdf_lat_idx < mask_single.shape[0] and 0 <= args.cdf_lon_idx < mask_single.shape[1]):
                         print(f"Warning: Provided CDF indices ({args.cdf_lat_idx}, {args.cdf_lon_idx}) are out of bounds for mask shape {mask_single.shape}. Attempting center.")
                         raise IndexError("CDF indices out of bounds for single fold")

                    if mask_single[args.cdf_lat_idx, args.cdf_lon_idx]:
                         print(f"Plotting CDF for specified cell: ({args.cdf_lat_idx}, {args.cdf_lon_idx})")
                         plot_single_cdf(current_fold_dir_single, args.fold, args.cdf_lat_idx, args.cdf_lon_idx, time_index=args.plot_time_index)
                    else:
                         print(f"Specified cell ({args.cdf_lat_idx}, {args.cdf_lon_idx}) not in mask. Attempting mask center.")
                         raise ValueError("Specified cell not in mask for single fold")
                except (FileNotFoundError, IndexError, ValueError) as e_cdf_single: # Renamed
                    print(f"Warning: Could not use specified/default CDF cell ({e_cdf_single}). Trying to find mask center...")
                    try:
                        mask_fallback_single = np.load(mask_path_single) # Renamed
                        true_indices_single = np.where(mask_fallback_single) # Renamed
                        if len(true_indices_single[0]) > 0: 
                            center_lat_idx_s = int(np.round(np.mean(true_indices_single[0]))) # Renamed
                            center_lon_idx_s = int(np.round(np.mean(true_indices_single[1]))) # Renamed
                            print(f"Plotting CDF for mask center: ({center_lat_idx_s}, {center_lon_idx_s})")
                            plot_single_cdf(current_fold_dir_single, args.fold, center_lat_idx_s, center_lon_idx_s, time_index=args.plot_time_index)
                        else:
                            print(f"Warning: Mask at {mask_path_single} contains no True values. Skipping CDF plot.")
                    except Exception as e_cdf_fallback_single: # Renamed
                        print(f"Warning: Error processing mask for CDF plot center: {e_cdf_fallback_single}. Skipping CDF plot.")
        else: # Corresponds to if result_single
            print(f"Evaluation for fold {args.fold} failed or produced no results.")
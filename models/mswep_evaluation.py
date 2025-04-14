import numpy as np
import pandas as pd
import os
import xarray as xr
import json
import matplotlib.pyplot as plt
import matplotlib.colors as colors  # Add colors import for PowerNorm
from isodisreg import idr
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
import argparse
import calendar

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

    Returns
    -------
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

    # --- Strict Checks ---
    if val_preds is None or val_targets is None or train_preds is None or train_targets is None or val_times is None:
         print("Error: Essential data arrays (preds, targets, or times) could not be loaded. Aborting evaluation for this fold.")
         return None, None, None, None, None, None

    # 2. Check if mask was loaded (generate default if missing, but warn)
    if mask is None:
         print("Warning: Mask file not found in fold directory. Creating default mask (all cells valid).")
         mask = np.ones((val_targets.shape[1], val_targets.shape[2]), dtype=bool)
         print(f"Created default mask with shape {mask.shape}")

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
        validation_year = val_times_pd.year[0] # Assume first timestamp represents the year
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
        val_times_pd = pd.to_datetime(val_times) # Still try to convert for return type consistency

    # --- End Strict Checks ---

    print(f"Loaded val_preds shape: {val_preds.shape}")
    print(f"Loaded val_targets shape: {val_targets.shape}")
    print(f"Loaded train_preds shape: {train_preds.shape}")
    print(f"Loaded train_targets shape: {train_targets.shape}")
    print(f"Loaded val_times shape: {val_times.shape} ({len(val_times_pd)} timestamps)")
    print(f"Loaded mask shape: {mask.shape}")

    # Return the loaded (and validated) data
    return val_preds, val_targets, train_preds, train_targets, val_times_pd, mask # Return pandas Timestamps

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
    y = np.asarray(binary_obs).astype(int) # Ensure binary 0/1

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
        uncertainty = np.mean(brier(y, mean_obs)) # Uses scalar mean_obs, broadcasts correctly

        # Calculate Calibrated Forecast using PAVA (Isotonic Regression)
        # Note: CEP_pav handles sorting internally via IsotonicRegression
        pav_x = CEP_pav(x, y) # Gets the calibrated probabilities

        # Calculate Score of Calibrated Forecast (Sc)
        Sc = np.mean(brier(y, pav_x))

        # Calculate Decomposition Components
        # MCB = BS - Sc (Should be >= 0)
        miscalibration = max(0.0, mean_bs - Sc) # Ensure non-negative due to potential float precision
        # DSC = UNC - Sc (Should be >= 0)
        discrimination = max(0.0, uncertainty - Sc) # Ensure non-negative

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

def plot_seasonal_metrics(results_summary: dict, output_dir: str, fold: int):
    """
    Generates and saves bar charts for seasonal CRPS and Brier Score.

    Args:
        results_summary (dict): Dictionary containing evaluation results,
                                including the 'seasonal_metrics' key.
        output_dir (str): Base directory where fold results are saved.
        fold (int): The current fold number (for titles and filenames).
    """
    seasonal_metrics = results_summary.get('seasonal_metrics')
    if not seasonal_metrics:
        print("Seasonal metrics not found in results. Skipping plotting.")
        return

    seasons = list(seasonal_metrics.keys()) # e.g., ['DJF', 'MAM', 'JJA', 'SON']
    # Ensure consistent order if dictionary doesn't preserve it
    season_order = ['DJF', 'MAM', 'JJA', 'SON']
    seasons = [s for s in season_order if s in seasonal_metrics] # Filter based on available keys

    mean_crps_seasonal = [seasonal_metrics[s].get('mean_crps', np.nan) for s in seasons]
    mean_bs_seasonal = [seasonal_metrics[s].get('mean_bs_0.2mm', np.nan) for s in seasons]

    fold_dir = os.path.join(output_dir, f"fold{fold}")
    plot_output_path_crps = os.path.join(fold_dir, f"seasonal_crps_fold{fold}.png")
    plot_output_path_bs = os.path.join(fold_dir, f"seasonal_bs_fold{fold}.png")

    # --- Plot Seasonal CRPS ---
    try:
        plt.figure(figsize=(8, 5))
        plt.bar(seasons, mean_crps_seasonal, color=['blue', 'green', 'red', 'orange'])
        plt.ylabel("Mean CRPS")
        plt.xlabel("Season")
        plt.title(f"Fold {fold}: Spatially Averaged Mean CRPS per Season")
        # Add values on top of bars, handling NaNs
        for i, val in enumerate(mean_crps_seasonal):
             if np.isfinite(val):
                  plt.text(i, val + (max(mean_crps_seasonal, default=0)*0.01), f'{val:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(plot_output_path_crps, dpi=150)
        plt.close()
        print(f"Saved seasonal CRPS plot to: {plot_output_path_crps}")
    except Exception as e:
        print(f"Error generating seasonal CRPS plot: {e}")
        plt.close()

    # --- Plot Seasonal Brier Score ---
    try:
        plt.figure(figsize=(8, 5))
        plt.bar(seasons, mean_bs_seasonal, color=['blue', 'green', 'red', 'orange'])
        plt.ylabel("Mean Brier Score (Thresh=0.2mm)")
        plt.xlabel("Season")
        plt.title(f"Fold {fold}: Spatially Averaged Mean Brier Score per Season")
         # Add values on top of bars, handling NaNs
        for i, val in enumerate(mean_bs_seasonal):
             if np.isfinite(val):
                  plt.text(i, val + (max(mean_bs_seasonal, default=0)*0.01), f'{val:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(plot_output_path_bs, dpi=150)
        plt.close()
        print(f"Saved seasonal Brier Score plot to: {plot_output_path_bs}")
    except Exception as e:
        print(f"Error generating seasonal Brier Score plot: {e}")
        plt.close()

def plot_quantile_map(fold_dir, fold_num, time_index=0):
    """
    Plots a map comparison of the 50th percentile predictions vs. targets for a specific time index.
    Uses PowerNorm for consistent visualization with the training script.
    
    Args:
        fold_dir (str): Directory containing the fold's evaluation results.
        fold_num (int): The fold number (for labeling purposes).
        time_index (int): The time index to plot. Default is 0.
    """
    print(f"Plotting quantile map for fold {fold_num} at time index {time_index}...")
    
    p50_preds_path = os.path.join(fold_dir, "val_preds_p50.npy")
    targets_path = os.path.join(fold_dir, "val_targets.npy")

    try:
        print(f"Loading target and p50 prediction data from {fold_dir}")
        val_targets = np.load(targets_path)
        val_preds_p50 = np.load(p50_preds_path)
    except FileNotFoundError as e:
        print(f"Error: Could not load required files: {e}")
        return
    
    if time_index >= val_targets.shape[0]:
        print(f"Error: Time index {time_index} exceeds available time steps ({val_targets.shape[0]}).")
        return
    
    target_map = val_targets[time_index, :, :]
    p50_map = val_preds_p50[time_index, :, :]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Create a non-linear norm for better visualization of precipitation
    # This will compress high values and expand low values for better visualization
    norm = colors.PowerNorm(gamma=0.5)
    
    im0 = axes[0].imshow(target_map, cmap='Blues', norm=norm)
    axes[0].set_title(f"Target Precipitation (Time Index {time_index}, Power Scale)")
    axes[0].set_xlabel("Longitude Index")
    axes[0].set_ylabel("Latitude Index")
    fig.colorbar(im0, ax=axes[0], label="Precipitation (mm)")
    
    im1 = axes[1].imshow(p50_map, cmap='Blues', norm=norm)
    axes[1].set_title(f"P50 Prediction (Time Index {time_index}, Power Scale)")
    axes[1].set_xlabel("Longitude Index")
    axes[1].set_ylabel("Latitude Index")
    fig.colorbar(im1, ax=axes[1], label="Precipitation (mm)")
    
    fig.suptitle(f"Fold {fold_num}: Target vs P50 Prediction", fontsize=14)
    
    plt.tight_layout()
    
    save_path = os.path.join(fold_dir, f"quantile_p50_map_fold{fold_num}_time{time_index}.png")
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
        cdf_values = single_pred_data.ecdf  # CDF values at those points
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot the CDF as a step function
        plt.plot(np.hstack([points[0], points]), np.hstack([0, cdf_values]), drawstyle='steps-post', 
                 color='blue', linewidth=2, label='Predicted CDF')
        
        # Add vertical line for the actual observation
        plt.axvline(target_obs, color='red', linestyle='--', 
                   label=f'Actual Obs: {target_obs:.2f} mm')
        
        # Add median line for reference
        p50 = predicted_distributions.qpred(quantiles=np.array([0.5]))[time_index]
        plt.axvline(p50, color='green', linestyle='-', 
                   label=f'Median (P50): {p50:.2f} mm')
        
        # Set limits and labels
        plt.xlim(left=0)  # Start from 0 precipitation
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
        
    except (AttributeError, IndexError) as e:
        print(f"Error accessing prediction data structure: {e}")
        print("The isodisreg implementation might have a different internal structure than expected.")
        
        # Alternative approach: Plot CDF by evaluating at multiple threshold points
        try:
            print("Attempting alternative CDF plotting approach...")
            # Generate points to evaluate the CDF
            precipitation_values = np.linspace(0, max(30, target_obs*2), 100)
            
            # Evaluate CDF at these points
            cdf_values = []
            for p in precipitation_values:
                cdf_at_p = predicted_distributions.cdf(thresholds=np.array([p]))[time_index]
                cdf_values.append(cdf_at_p)
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(precipitation_values, cdf_values, color='blue', linewidth=2, label='Predicted CDF')
            
            # Add vertical line for the actual observation
            plt.axvline(target_obs, color='red', linestyle='--', 
                       label=f'Actual Obs: {target_obs:.2f} mm')
            
            # Add median line for reference
            p50 = predicted_distributions.qpred(quantiles=np.array([0.5]))[time_index]
            plt.axvline(p50, color='green', linestyle='-', 
                       label=f'Median (P50): {p50:.2f} mm')
            
            # Set limits and labels
            plt.xlim(left=0)  # Start from 0 precipitation
            plt.ylim(0, 1.05)  # Probability range with slight margin
            plt.xlabel('Precipitation (mm)')
            plt.ylabel('Cumulative Probability')
            plt.title(f'Predicted CDF - Fold {fold_num} - Cell ({lat_idx},{lon_idx}) - Time {time_index}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save the figure
            save_path = os.path.join(fold_dir, f"single_cdf_plot_fold{fold_num}_cell{lat_idx}-{lon_idx}_time{time_index}_alt.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved alternative CDF plot to: {save_path}")
            
        except Exception as e2:
            print(f"Alternative CDF plotting also failed: {e2}")
            return

def run_evaluation(base_output_dir: str, fold: int, batch_size: int = 100, n_recent_train_samples: int = 1000):
    """
    Runs the evaluation pipeline for a specific fold.
    Loads data, applies EasyUQ per cell (using recent train data), and stores results.

    Args:
        base_output_dir (str): Base directory where fold results are saved.
        fold (int): The fold number to evaluate.
        batch_size (int): Number of grid cells to process in each batch to limit memory usage.
        n_recent_train_samples (int): Number of recent training samples for IDR fitting.
    """
    # 1. Load Data
    data = load_evaluation_data(base_output_dir, fold)
    if data[0] is None:
        print(f"Skipping evaluation for fold {fold} due to data loading issues.")
        return None

    val_preds, val_targets, train_preds, train_targets, val_times, mask = data

    # --- OPTIMIZATION: Select recent training data ONCE ---
    if train_preds is not None and train_targets is not None:
        if len(train_preds) >= n_recent_train_samples:
            print(f"Using last {n_recent_train_samples} training samples for IDR fitting.")
            train_preds_recent = train_preds[-n_recent_train_samples:]
            train_targets_recent = train_targets[-n_recent_train_samples:]
        else:
            print(f"Warning: Available training samples ({len(train_preds)}) is less than requested recent samples ({n_recent_train_samples}). Using all available.")
            train_preds_recent = train_preds
            train_targets_recent = train_targets
    else:
        print("Error: Cannot proceed without training data for IDR.")
        return None
    # --- END OPTIMIZATION ---

    if mask is None:
        print("Warning: Mask is None. Assuming all grid cells are valid.")
        # Create a default mask (all True) based on target shape if needed
        if val_targets is not None:
             mask = np.ones(val_targets.shape[1:], dtype=bool)
        else:
             print("Error: Cannot determine grid size without targets or mask.")
             return None

    num_val_samples, grid_lat, grid_lon = val_targets.shape

    if val_times is not None:
         timestamps = val_times # Already a DatetimeIndex from load_evaluation_data
         # Define seasons (adjust months if using a different definition)
         season_map = {
             12: 'DJF', 1: 'DJF', 2: 'DJF', # Winter
             3: 'MAM', 4: 'MAM', 5: 'MAM', # Spring
             6: 'JJA', 7: 'JJA', 8: 'JJA', # Summer
             9: 'SON', 10: 'SON', 11: 'SON' # Autumn
         }
         seasons = ['DJF', 'MAM', 'JJA', 'SON']
         # Get season for each timestamp (used in the loop)
         timestamp_months = timestamps.month
         timestamp_seasons = np.array([season_map[m] for m in timestamp_months])

         # Initialize storage for seasonal aggregates
         seasonal_aggregates = {
             season: {'crps_sum': 0.0, 'bs_sum': 0.0, 'count': 0}
             for season in seasons
         }
         print(f"Initialized seasonal aggregates for: {seasons}")
    else:
         print("Warning: Validation timestamps not loaded, seasonal analysis will not be possible.")
         timestamps = None
         seasonal_aggregates = None
         seasons = []


    # Initialize an array to store the 50th percentile (median) predictions
    val_preds_p50 = np.full((num_val_samples, grid_lat, grid_lon), np.nan)

    # 2. Initialize storage for per-cell results
    # Store predicted distributions, targets, and times for valid cells
    # Using a dictionary where keys are (lat, lon) tuples
    evaluation_results_per_cell = {}
    
    valid_cell_indices = [(lat, lon) for lat in range(grid_lat) for lon in range(grid_lon) if mask[lat, lon]]
    total_valid_cells = len(valid_cell_indices)
    print(f"Found {total_valid_cells} valid grid cells to process")
    
    # Process cells in batches to limit memory usage
    num_batches = (total_valid_cells + batch_size - 1) // batch_size
    print(f"Processing cells in {num_batches} batches of size {batch_size}")
    
    all_cell_mean_crps = []
    all_cell_mean_bs = []
    processed_cells = 0
    
    fold_dir = os.path.join(base_output_dir, f"fold{fold}")
    metrics_path = os.path.join(fold_dir, "cell_metrics.npy")
    cell_metrics = {'crps': {}, 'bs': {}}
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_valid_cells)
        batch_cells = valid_cell_indices[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx+1}/{num_batches} ({len(batch_cells)} cells)")
        
        batch_metrics = {'crps': {}, 'bs': {}}
        
        for lat, lon in tqdm(batch_cells, desc=f"Batch {batch_idx+1}/{num_batches}"):
            train_preds_recent_cell = train_preds_recent[:, lat, lon]
            train_targets_recent_cell = train_targets_recent[:, lat, lon]
            val_preds_cell = val_preds[:, lat, lon]
            val_target_cell = val_targets[:, lat, lon]

            # Apply EasyUQ calibration
            predicted_distributions = apply_easyuq_per_cell(
                train_preds_recent_cell, train_targets_recent_cell, eval_preds_cell=val_preds_cell
            )

            if predicted_distributions is not None:
                # Calculate metrics right away to avoid storing all distributions in memory
                try:
                    # Ensure target is 1D float array for crps calculation
                    y_true_crps = val_target_cell.astype(float).flatten()
                    if y_true_crps.ndim != 1:
                         print(f"Warning: val_target_cell for cell ({lat}, {lon}) is not 1D after flattening. Shape: {y_true_crps.shape}. Skipping CRPS.")
                         crps_values = None
                    else:
                        crps_result = predicted_distributions.crps(y_true_crps)

                        crps_values = None
                        if isinstance(crps_result, list):
                            if len(crps_result) == y_true_crps.shape[0]:
                                try:
                                    crps_values = np.array(crps_result, dtype=float) # Convert list to numpy array
                                except ValueError as e:
                                    print(f"Warning: Could not convert CRPS list to float array for cell ({lat}, {lon}): {e}. Skipping CRPS.")
                            else:
                                print(f"Warning: CRPS list length mismatch for cell ({lat}, {lon}). Got {len(crps_result)}, expected {y_true_crps.shape[0]}. Skipping CRPS.")
                        elif isinstance(crps_result, np.ndarray):
                            crps_values = crps_result
                        else:
                            print(f"Warning: Unexpected type from predicted_distributions.crps() for cell ({lat}, {lon}). Type: {type(crps_result)}. Skipping CRPS.")

                        # Defensive check on crps_values type and shape (now after potential conversion)
                        if isinstance(crps_values, np.ndarray) and crps_values.shape == y_true_crps.shape:
                            valid_crps_mask = np.isfinite(crps_values)
                            if np.any(valid_crps_mask):
                                mean_crps_cell = np.mean(crps_values[valid_crps_mask])
                                if np.isfinite(mean_crps_cell):
                                    all_cell_mean_crps.append(mean_crps_cell)
                                    cell_metrics['crps'][(lat, lon)] = mean_crps_cell

                                if seasonal_aggregates is not None:
                                    valid_crps_values = crps_values[valid_crps_mask]
                                    valid_seasons = timestamp_seasons[valid_crps_mask]
                                    for season in seasons:
                                        season_mask = (valid_seasons == season)
                                        if np.any(season_mask):
                                            seasonal_aggregates[season]['crps_sum'] += np.sum(valid_crps_values[season_mask])
                                            seasonal_aggregates[season]['count'] += np.sum(season_mask)
                        elif crps_values is not None:
                            print(f"Warning: Unexpected output from predicted_distributions.crps() for cell ({lat}, {lon}). Type: {type(crps_values)}, Shape: {getattr(crps_values, 'shape', 'N/A')}. Expected shape: {y_true_crps.shape}. Skipping CRPS calculations for this cell.")
                            # Ensure downstream Brier score calculation can proceed if possible, but skip CRPS accumulators

                        # If CRPS failed, we might need a fallback for BS accumulation
                        # For now, BS relies on valid_crps_mask. If CRPS failed, BS seasonal might not be accumulated.
                        # TODO: Consider decoupling BS seasonal accumulation if CRPS fails often.

                except Exception as e:
                    print(f"Error calculating CRPS or processing its results for cell ({lat}, {lon}): {e}")
                    # Prevent error from stopping the whole batch
                    # Ensure downstream code knows CRPS failed, perhaps by setting a flag or ensuring metrics aren't added

                # Calculate Brier scores for this cell
                try:
                    # Ensure target is 1D float for BS calculation as well
                    y_true_bs = val_target_cell.astype(float).flatten()
                    if y_true_bs.ndim != 1:
                         print(f"Warning: val_target_cell for cell ({lat}, {lon}) is not 1D for BS calc. Shape: {y_true_bs.shape}. Skipping BS.")
                         continue # Skip BS part for this cell

                    prob_le_threshold = predicted_distributions.cdf(thresholds=np.array([0.2]))
                    if isinstance(prob_le_threshold, np.ndarray) and prob_le_threshold.ndim > 1 and prob_le_threshold.shape[1] == 1:
                        prob_le_threshold = prob_le_threshold.flatten()
                    elif not (isinstance(prob_le_threshold, np.ndarray) and prob_le_threshold.ndim == 1 and len(prob_le_threshold) == len(val_target_cell)):
                        raise ValueError(f"Unexpected shape from cdf(): {prob_le_threshold.shape}")

                    predicted_prob_exceed = 1.0 - prob_le_threshold
                    predicted_prob_exceed = np.clip(predicted_prob_exceed, 0.0, 1.0)
                    binary_outcomes = (val_target_cell > 0.2).astype(float)
                    brier_scores = (predicted_prob_exceed - binary_outcomes) ** 2
                    valid_bs_mask = np.isfinite(brier_scores)

                    if np.any(valid_bs_mask):
                        mean_bs_cell = np.mean(brier_scores[valid_bs_mask])
                        if np.isfinite(mean_bs_cell):
                            all_cell_mean_bs.append(mean_bs_cell)
                            cell_metrics['bs'][(lat, lon)] = mean_bs_cell

                        # Accumulate seasonal Brier Score sums
                        # Or should BS have its own count? Let's assume we want BS average over same samples as CRPS.
                        if seasonal_aggregates is not None:
                            valid_bs_values = brier_scores[valid_bs_mask]
                            # We need to align valid_bs_values with the seasons corresponding to the valid_bs_mask
                            valid_seasons_for_bs = timestamp_seasons[valid_bs_mask]
                            for season in seasons:
                                season_mask_bs = (valid_seasons_for_bs == season)
                                if np.any(season_mask_bs):
                                    seasonal_aggregates[season]['bs_sum'] += np.sum(valid_bs_values[season_mask_bs])
                                    # Assuming BS count is tied to valid CRPS count for now. If not, uncomment below
                                    # seasonal_aggregates[season]['bs_count'] = seasonal_aggregates[season].get('bs_count', 0) + np.sum(season_mask_bs)
                    else:
                        print(f"Warning: Non-finite Brier Scores for cell ({lat}, {lon}).")
                except Exception as e:
                    print(f"Error calculating Brier Scores for cell ({lat}, {lon}): {e}")
                
                # Extract the 50th percentile (median) predictions for this cell
                try:
                    median_preds_cell = predicted_distributions.qpred(quantiles=np.array([0.5]))
                    if median_preds_cell is not None and len(median_preds_cell) == num_val_samples:
                        val_preds_p50[:, lat, lon] = median_preds_cell
                except Exception as e:
                    print(f"Error extracting median predictions for cell ({lat}, {lon}): {e}")
                
                # Important: Explicitly delete the distribution object to free memory
                del predicted_distributions
            
            processed_cells += 1
        
        # Update the overall metrics dictionary
        cell_metrics['crps'].update(batch_metrics['crps'])
        cell_metrics['bs'].update(batch_metrics['bs'])
        
        # Periodically save the cell metrics to disk
        try:
            np.save(metrics_path, cell_metrics)
            print(f"Saved intermediate cell metrics after {processed_cells}/{total_valid_cells} cells")
        except Exception as e:
            print(f"Error saving cell metrics: {e}")
        
        # Periodically save the p50 predictions to avoid memory buildup
        if (batch_idx + 1) % 5 == 0 or batch_idx == num_batches - 1:
            print(f"Saving intermediate p50 predictions after {processed_cells}/{total_valid_cells} cells")
            np.save(os.path.join(fold_dir, "val_preds_p50.npy"), val_preds_p50)
        
        batch_metrics = None
        
        # Force garbage collection after each batch
        import gc
        gc.collect()
        
        print(f"Memory freed after batch {batch_idx+1}/{num_batches}")

    print(f"EasyUQ application complete. Processed {processed_cells} valid cells.")

    # Save the 50th percentile predictions (final save)
    np.save(os.path.join(fold_dir, "val_preds_p50.npy"), val_preds_p50)
    print(f"Saved 50th percentile predictions to: {os.path.join(fold_dir, 'val_preds_p50.npy')}")

    print("Calculating overall metrics from accumulated results...")

    overall_mean_crps = np.mean(all_cell_mean_crps) if all_cell_mean_crps else np.nan
    overall_mean_bs = np.mean(all_cell_mean_bs) if all_cell_mean_bs else np.nan

    print(f"\n--- Overall Validation Metrics (Fold {fold}) ---")
    print(f"Spatially Averaged Mean CRPS: {overall_mean_crps:.4f}")
    print(f"Spatially Averaged Mean Brier Score (Thresh=0.2mm): {overall_mean_bs:.4f}")
    print(f"Number of valid cells used for averaging: {len(all_cell_mean_crps)}")

    results_summary = {
        'fold': fold,
        'overall_mean_crps': overall_mean_crps,
        'overall_mean_bs_0.2mm': overall_mean_bs,
        'num_valid_cells': len(all_cell_mean_crps)
    }

    if seasonal_aggregates is None:
        print("Warning: Timestamps not found or seasonal aggregates not initialized. Skipping seasonal analysis.")
        results_summary['seasonal_metrics'] = None
    else:
        seasonal_metrics_results = {}
        print("\nCalculating final seasonal metrics from aggregates...")

        for season in seasons:
            season_data = seasonal_aggregates[season]
            count = season_data['count']
            crps_sum = season_data['crps_sum']
            bs_sum = season_data['bs_sum']
            # bs_count = season_data.get('bs_count', count) # Use separate count if calculated

            if count > 0:
                mean_crps = crps_sum / count
                mean_bs = bs_sum / count # Divide BS sum by CRPS count
                # if bs_count > 0: mean_bs = bs_sum / bs_count # Use separate BS count if available
            else:
                mean_crps = np.nan
                mean_bs = np.nan

            seasonal_metrics_results[season] = {
                'mean_crps': mean_crps,
                'mean_bs_0.2mm': mean_bs,
                'num_samples': count # Total valid samples across all cells for this season
            }
            print(f"  {season}: Mean CRPS = {mean_crps:.4f}, Mean BS = {mean_bs:.4f} (from {count} valid samples)")

        # Add seasonal metrics to the results summary
        results_summary['seasonal_metrics'] = seasonal_metrics_results

    print("Overall and seasonal metrics calculated.")
    
    # Print formatted summary of results
    print(f"\n--- Evaluation Summary for Fold {fold} ---")
    print(f"Overall Mean CRPS: {results_summary.get('overall_mean_crps', 'N/A'):.4f}")
    print(f"Overall Mean BS (Thresh=0.2mm): {results_summary.get('overall_mean_bs_0.2mm', 'N/A'):.4f}")
    print(f"Number of Valid Cells: {results_summary.get('num_valid_cells', 'N/A')}")
    print("\nSeasonal Metrics:")
    print(f"{'Season':<6} {'Mean CRPS':<12} {'Mean BS':<12} {'Samples':<10}")
    print("-" * 45)
    seasonal_metrics = results_summary.get('seasonal_metrics')
    if seasonal_metrics:
        for season, metrics in seasonal_metrics.items():
            crps = metrics.get('mean_crps', np.nan)
            bs = metrics.get('mean_bs_0.2mm', np.nan)
            samples = metrics.get('num_samples', 'N/A')
            # cells = metrics.get('num_cells', 'N/A') # num_cells is less relevant now
            print(f"{season:<6} {crps:<12.4f} {bs:<12.4f} {samples:<10}")
    else:
        print("  (Seasonal metrics not available)")
    print("-" * 45)

    # Note: The seasonal analysis currently uses overall cell metrics rather than season-specific metrics.
    # print("     For more accurate seasonal analysis, implement season-specific metric calculation during batch processing.")
    print("\nNote: Seasonal analysis now calculates metrics based on aggregated scores per season.")

    fold_dir = os.path.join(base_output_dir, f"fold{fold}")
    json_output_path = os.path.join(fold_dir, f"evaluation_summary_fold{fold}.json")
    try:
        with open(json_output_path, 'w') as f:
            # Use the custom encoder to handle numpy types
            json.dump(results_summary, f, indent=4, default=numpy_json_encoder)
        print(f"Evaluation summary saved to: {json_output_path}")
    except TypeError as te:
         print(f"TypeError saving results to JSON: {te}. Check data types in results_summary.")
         # Attempt saving with string conversion as fallback
         try:
             with open(json_output_path.replace('.json', '_fallback.json'), 'w') as f:
                  json.dump(results_summary, f, indent=4, default=str)
             print(f"Fallback summary saved with string conversion.")
         except Exception as e_fallback:
              print(f"Error during fallback JSON saving: {e_fallback}")
    except Exception as e:
        print(f"Error saving evaluation summary to JSON: {e}")
    
    # Generate seasonal performance plots
    plot_seasonal_metrics(results_summary, base_output_dir, fold)
    
    print("Finished evaluation.")
    return results_summary

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
    parser.add_argument('--cdf_lat', type=int, default=20, help='Latitude index for CDF plotting (default: 20)')
    parser.add_argument('--cdf_lon', type=int, default=30, help='Longitude index for CDF plotting (default: 30)')
    parser.add_argument('--batch_size', type=int, default=50, help='Number of grid cells to process in each batch (default: 50)')
    parser.add_argument('--idr_samples', type=int, default=1000, help='Number of recent training samples to use for IDR fitting (default: 1000)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory {args.output_dir} not found.")
        exit(1)
    
    if args.run_id:
        run_dir = os.path.join(args.output_dir, args.run_id)
        if not os.path.exists(run_dir):
            print(f"Error: Run directory {run_dir} not found.")
            exit(1)
    else:
        # Find the most recent run directory
        run_dirs = [d for d in os.listdir(args.output_dir) if d.startswith("run_")]
        if not run_dirs:
            print(f"Error: No run_* directories found in {args.output_dir}")
            exit(1)
        
        # Sort by timestamp (newest first)
        run_dirs.sort(reverse=True)
        run_dir = os.path.join(args.output_dir, run_dirs[0])
        print(f"Using most recent run directory: {run_dir}")
    
    if args.all_folds:
        # Find all fold directories in the run directory
        fold_dirs = [int(d.replace("fold", "")) for d in os.listdir(run_dir) 
                    if d.startswith("fold") and os.path.isdir(os.path.join(run_dir, d))]
        fold_dirs.sort()
        
        # Limit to max_folds if specified
        if args.max_folds > 0 and len(fold_dirs) > args.max_folds:
            print(f"Limiting evaluation to the first {args.max_folds} folds.")
            fold_dirs = fold_dirs[:args.max_folds]
        
        if not fold_dirs:
            print(f"Error: No fold directories found in {run_dir}")
            exit(1)
            
        print(f"Evaluating all {len(fold_dirs)} folds: {fold_dirs}")
        
        results = []
        for fold in fold_dirs:
            print(f"\n=== Evaluating fold {fold} ===")
            fold_dir = os.path.join(run_dir, f"fold{fold}")
            result = run_evaluation(run_dir, fold, batch_size=args.batch_size, n_recent_train_samples=args.idr_samples)
            if result:
                results.append(result)
                print(f"Evaluation for fold {fold} complete.")
                
                print(f"Generating P50 prediction map for fold {fold}...")
                plot_quantile_map(fold_dir, fold, time_index=args.plot_time_index)
                
                if args.plot_cdf:
                    # --- Automatically find center of mask ---
                    mask_path = os.path.join(fold_dir, "germany_mask.npy")
                    try:
                        mask = np.load(mask_path)
                        true_indices = np.where(mask)
                        if len(true_indices[0]) > 0: # Check if mask has any True values
                            center_lat = int(np.round(np.mean(true_indices[0])))
                            center_lon = int(np.round(np.mean(true_indices[1])))
                            print(f"Plotting CDF for mask center: ({center_lat}, {center_lon})")
                            plot_single_cdf(fold_dir, fold, center_lat, center_lon, time_index=args.plot_time_index)
                        else:
                            print(f"Warning: Mask at {mask_path} contains no True values. Skipping CDF plot for center.")
                    except FileNotFoundError:
                         print(f"Warning: Mask file not found at {mask_path}. Cannot determine center for CDF plot. Using defaults.")
                         plot_single_cdf(fold_dir, fold, args.cdf_lat, args.cdf_lon, time_index=args.plot_time_index)
                    except Exception as e:
                        print(f"Warning: Error processing mask for CDF plot center: {e}. Using defaults.")
                        plot_single_cdf(fold_dir, fold, args.cdf_lat, args.cdf_lon, time_index=args.plot_time_index)
                    # --- End automatic center finding ---
            else:
                print(f"Evaluation for fold {fold} failed or produced no results.")
        
        print(f"\nCompleted evaluation of {len(results)} folds.")
    else:
        # Evaluate just the specified fold
        print(f"\n=== Evaluating fold {args.fold} ===")
        fold_dir = os.path.join(run_dir, f"fold{args.fold}")
        result = run_evaluation(run_dir, args.fold, batch_size=args.batch_size, n_recent_train_samples=args.idr_samples)
        if result:
            print(f"Evaluation complete for fold {args.fold}.")
            
            print(f"Generating P50 prediction map for fold {args.fold}...")
            plot_quantile_map(fold_dir, args.fold, time_index=args.plot_time_index)
            
            if args.plot_cdf:
                # --- Automatically find center of mask ---
                mask_path = os.path.join(fold_dir, "germany_mask.npy")
                try:
                    mask = np.load(mask_path)
                    true_indices = np.where(mask)
                    if len(true_indices[0]) > 0: # Check if mask has any True values
                        center_lat = int(np.round(np.mean(true_indices[0])))
                        center_lon = int(np.round(np.mean(true_indices[1])))
                        print(f"Plotting CDF for mask center: ({center_lat}, {center_lon})")
                        plot_single_cdf(fold_dir, args.fold, center_lat, center_lon, time_index=args.plot_time_index)
                    else:
                        print(f"Warning: Mask at {mask_path} contains no True values. Skipping CDF plot for center.")
                except FileNotFoundError:
                    print(f"Warning: Mask file not found at {mask_path}. Cannot determine center for CDF plot. Using defaults.")
                    plot_single_cdf(fold_dir, args.fold, args.cdf_lat, args.cdf_lon, time_index=args.plot_time_index)
                except Exception as e:
                    print(f"Warning: Error processing mask for CDF plot center: {e}. Using defaults.")
                    plot_single_cdf(fold_dir, args.fold, args.cdf_lat, args.cdf_lon, time_index=args.plot_time_index)
                # --- End automatic center finding ---
            else:
                print(f"Evaluation for fold {args.fold} failed or produced no results.")

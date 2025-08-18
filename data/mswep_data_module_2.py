"""
mswep_data_module.py

This module implements a LightningDataModule for MSWEP precipitation forecasting.
It loads newly preprocessed MSWEP daily data from monthly NetCDF files named "mswep_daily_YYYYMM.nc".
Each file contains daily precipitation data on a 41×121 grid with variable "precip".
Samples are created using a 3-lag structure:
    - For each valid day t (from t=3 to T-1), the input is constructed from the previous three days:
          input = stack([data[t-3], data[t-2], data[t-1]])  -> shape (3, 41, 121)
      and the target is:
          target = data[t]  -> shape (41, 121)
This updated approach removes any dropped days and handles a continuous daily time series.

The dataset is then split into training and validation (and optionally test) sets using
a year-based expanding window approach. For each fold, the model is trained on all available
data from the start year (e.g., 2007) up to the validation year (e.g., fold 0 -> 2010), and 
validated on data from the validation year.
A TargetLogScaler is included for applying a log transform to the precipitation targets,
in line with the scientific practices described in the original research paper.

This implementation is fully compatible with EasyUQ (IDR) postprocessing, as it preserves
the grid cell structure needed for cell-wise calibration and ensures proper 
training/validation splits for fitting IDR models.
"""

import os
import re
import numpy as np
import xarray as xr
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
import lightning as L
import pandas as pd  # Add import for pandas to handle timestamps
import math  # Add import for math functions
import calendar # Add import for calendar module

class TargetLogScaler:
    """
    Log transform for precipitation data.

    Applies log(x + offset) to stabilize optimization when training with heavy-tailed
    precipitation distributions. The inverse transform is exp(x) - offset.
    """
    def __init__(self, offset: float = 0.1) -> None:  # Changed from 0.01 to 0.1
        self.offset = offset
        print(f"TargetLogScaler initialized with offset={self.offset:.3f}")

    def fit(self, data):
        # No fitting needed for a simple log transform
        pass

    def transform(self, data):
        if isinstance(data, torch.Tensor):
            return torch.log(data + self.offset)
        else:
            return np.log(data + self.offset)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            # Apply exponential and subtract offset
            result = torch.exp(data) - self.offset
            # Ensure non-negative precipitation (physical constraint)
            result = torch.clamp(result, min=0.0)
            
            # Warn about extremely high values (exceeding 100mm/day)
            with torch.no_grad():
                if result.numel() > 0: # Check if tensor is not empty
                    max_val = result.max().item()
                    if max_val > 100.0:
                        print(f"Warning: Extremely high precipitation values detected: {max_val:.2f} mm")
                        # Optional: Add a reasonable cap for physical consistency
                        # result = torch.clamp(result, max=300.0)
            return result
        else:
            # NumPy version
            result = np.exp(data) - self.offset
            # Ensure non-negative precipitation
            result = np.clip(result, 0.0, None)
            
            # Warn about extremely high values
            if result.size > 0: # Check if array is not empty
                max_val = np.max(result)
                if max_val > 100.0:
                    print(f"Warning: Extremely high precipitation values detected: {max_val:.2f} mm")
                    # Optional: Add a reasonable cap for physical consistency
                    # result = np.clip(result, None, 300.0)
            return result

class MSWEPDataModule(L.LightningDataModule):
    """
    LightningDataModule for MSWEP precipitation data with daily time-based splits.

    This module:
      - Loads newly preprocessed MSWEP daily data from monthly NetCDF files located in a specified directory.
      - Each file is expected to be named in the pattern "mswep_daily_YYYYMM.nc" and to contain a variable "precip"
        with shape (days, 41, 121).
      - Constructs a continuous daily time series by concatenating data from all files.
      - Creates samples using a sliding window with a revised 3-lag structure:
            For each valid day t (from t=3 to T-1):
                input = stack([data[t-3], data[t-2], data[t-1]])  -> shape (3, 41, 121)
                target = data[t]                                   -> shape (41, 121)
      - Applies log transformation to targets in the DataModule for consistency
      - Optionally applies a log-transform to the targets.
      - Splits the combined dataset into training and validation (and optionally test) sets using
        a year-based expanding window. For each fold, the model trains on all available data from 
        the start year (e.g., 2007) up to the validation year and validates on data from the validation year.
      
    EasyUQ Compatibility:
      - Preserves the grid cell structure (41x121) needed for cell-wise IDR calibration
      - Ensures proper temporal separation between training and validation sets
      - Maintains consistent grid cells across time for proper IDR model fitting
      - Enables fold-based training that aligns with the CNN+EasyUQ approach
      
    Attributes:
        data_dir (str): Directory containing monthly NetCDF files for training.
        test_data_dir (str or None): Optional directory for test NetCDF files.
        batch_size (int): Batch size for the data loaders.
        num_workers (int): Number of worker processes for data loading.
        fold (int): Fold index to use when splitting the data (determines the validation year).
        num_total_folds (int): Total number of folds planned for the cross-validation.
        target_source (str): Source for target data ('mswep' or 'hyras').
        apply_log_transform (bool): Whether to apply log transform to targets.
        log_offset (float): Offset for log transformation.
        era5_variables (list): List of ERA5 variable names to load (e.g., ['u', 'v', 'q']).
        era5_pressure_levels (list): List of pressure levels for ERA5 variables (e.g., [300, 500, 700, 850]).
        era5_data_dir (str): Directory containing preprocessed ERA5 predictor files.
    """
    def __init__(self, data_dir, test_data_dir=None, batch_size=32, num_workers=4,
                 fold=0, num_total_folds: int = 5, target_source='mswep',
                 apply_log_transform=True, log_offset=0.1,
                 era5_variables=None, era5_pressure_levels=None, era5_data_dir=None,
                 era5_single_level_variables=None): # Add ERA5 single-level parameter
        super().__init__()
        self.data_dir = data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold = fold
        self.num_total_folds = num_total_folds # Store it, though not directly used by current year-based split logic
        self.target_source = target_source
        self.apply_log_transform = apply_log_transform
        self.log_offset = log_offset
        
        # Store ERA5 configuration
        self.era5_variables = era5_variables if era5_variables is not None else []
        self.era5_pressure_levels = era5_pressure_levels if era5_pressure_levels is not None else []
        self.era5_data_dir = era5_data_dir
        self.era5_single_level_variables = era5_single_level_variables if era5_single_level_variables is not None else []
        
        # Store for ERA5 normalization statistics
        self.era5_stats = {}

        # Initialize target scaler if log transform is enabled
        if self.apply_log_transform:
            self.target_scaler = TargetLogScaler(offset=self.log_offset)
            print(f"MSWEPDataModule: Log transform enabled with offset={self.log_offset}")
        else:
            self.target_scaler = None
            print("MSWEPDataModule: No log transform applied")
            
        # Print ERA5 configuration if provided
        if (self.era5_variables or self.era5_single_level_variables) and self.era5_data_dir:
            print(f"MSWEPDataModule: ERA5 integration enabled")
            if self.era5_variables:
                print(f"  Pressure-level variables: {self.era5_variables}")
                print(f"  Pressure levels: {self.era5_pressure_levels}")
            if self.era5_single_level_variables:
                print(f"  Single-level variables: {self.era5_single_level_variables}")
            print(f"  Data directory: {self.era5_data_dir}")

        # Initialize attributes that will be set in setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.val_times = None
        self._setup_called = False  # Flag to track if setup has been called

    def _calculate_era5_stats(self, era5_data_dict):
        """
        Calculate mean and standard deviation for each ERA5 variable.
        These statistics will be used to standardize the data to zero mean and unit variance.
        
        Args:
            era5_data_dict: Dictionary mapping variable names to numpy arrays
            
        Returns:
            dict: Statistics dictionary with mean and std for each variable
        """
        stats = {}
        
        for var_name, data in era5_data_dict.items():
            # Flatten the data to calculate statistics across all spatial and temporal dimensions
            flat_data = data.flatten()
            
            # Remove NaN values for robust statistics
            valid_data = flat_data[~np.isnan(flat_data)]
            
            if len(valid_data) > 0:
                mean_val = np.mean(valid_data)
                std_val = np.std(valid_data)
                
                # Prevent division by zero in standardization
                if std_val == 0 or np.isnan(std_val):
                    std_val = 1.0
                    
                stats[var_name] = {
                    'mean': mean_val,
                    'std': std_val
                }
                
                print(f"  ERA5 {var_name} statistics: mean={mean_val:.3f}, std={std_val:.3f}")
            else:
                # If all data is NaN, use default values
                stats[var_name] = {
                    'mean': 0.0,
                    'std': 1.0
                }
                print(f"  WARNING: ERA5 {var_name} has no valid data, using default normalization")
                
        return stats

    def _apply_era5_standardization(self, era5_data_dict):
        """
        Apply pre-computed ERA5 standardization statistics to data.
        This is used for test data to ensure it uses the same normalization as training data.
        
        Args:
            era5_data_dict: Dictionary mapping variable names to numpy arrays
            
        Returns:
            dict: Standardized ERA5 data
        """
        if not self.era5_stats:
            print("Warning: No ERA5 statistics available. Returning unstandardized data.")
            return era5_data_dict
        
        standardized_dict = {}
        
        for var_name, data in era5_data_dict.items():
            if var_name in self.era5_stats:
                mean_val = self.era5_stats[var_name]['mean']
                std_val = self.era5_stats[var_name]['std']
                
                # Apply standardization using training statistics
                standardized_dict[var_name] = (data - mean_val) / std_val
                
                # Verify
                standardized_mean = np.nanmean(standardized_dict[var_name])
                standardized_std = np.nanstd(standardized_dict[var_name])
                print(f"  {var_name} test data standardized: mean≈{standardized_mean:.3e}, std≈{standardized_std:.3f}")
            else:
                print(f"  Warning: No standardization stats for {var_name}. Using unstandardized data.")
                standardized_dict[var_name] = data
        
        return standardized_dict

    def prepare_data(self):
        """
        Prepare data if needed.
        Assumes that the monthly NetCDF files already exist in the provided directories.
        """
        pass

    def setup(self, stage=None):
        """
        Set up datasets for training, validation, and testing.

        This method:
          - Loads and concatenates monthly data from the specified directories *only for the required years*.
          - Constructs samples using the updated 3-lag strategy:
                For each valid index i (from 3 to T-1), inputs = [data[i-3], data[i-2], data[i-1]]
                and target = data[i].
          - Splits the complete dataset into training and validation sets using a year-based expanding window.
            For a given fold, trains on all available data from start_year up to validation_year,
            and validates on validation_year.
        """
        # Check if setup has already been called
        if self._setup_called and stage == 'fit':
            print("MSWEPDataModule.setup() already called for stage='fit'. Skipping to prevent duplicate data loading.")
            return
            
        print(f"MSWEPDataModule.setup() called with stage='{stage}'")
        self._setup_called = True if stage == 'fit' else self._setup_called

        # Define constants for the year-based splitting
        start_year = 2007  # First year of available data
        base_val_year = 2010  # First validation year (fold 0)
        current_val_year = base_val_year + self.fold

        # Determine the required end year for loading data (includes validation year)
        required_end_year = current_val_year # We need data up to and including the validation year

        # Load and combine training data *only for the required years*
        train_series, train_times = self._load_data(self.data_dir, start_year, required_end_year) # <-- Pass year range

        # Load target data based on specified source (if different)
        if self.target_source.lower() == 'mswep':
            # Target data source is the same as input source
            target_series = train_series
            print(f"Using MSWEP as target data source.")
        elif self.target_source.lower() == 'hyras':
            print(f"Using HYRAS (regridded) as target data source.")
            # Ensure _load_hyras_data also respects the required_end_year if necessary,
            # or confirm it correctly aligns with the potentially shorter train_times
            hyras_dir = os.path.join(os.path.dirname(self.data_dir), "HYRAS_regridded")
            # Assuming _load_hyras_data aligns correctly to the provided reference_times (train_times)
            target_series, _ = self._load_hyras_data(hyras_dir, train_times)
            
            # Validate the loaded data
            if not isinstance(target_series, np.ndarray):
                raise TypeError("Loaded HYRAS target data must be a numpy array")
            if target_series.shape != train_series.shape:
                raise ValueError(f"Shape mismatch: HYRAS target {target_series.shape} vs MSWEP input {train_series.shape}")
            print(f"Successfully loaded regridded HYRAS data with shape {target_series.shape}")
        else:
            raise ValueError(f"Unknown target_source: '{self.target_source}'. Choose 'mswep' or 'hyras'.")

        # Load ERA5 data if configured
        era5_data_dict = {}
        if (self.era5_variables or self.era5_single_level_variables) and self.era5_data_dir:
            try:
                era5_data_dict, era5_times = self._load_era5_data(start_year, required_end_year)
            except Exception as e:
                print(f"Error loading ERA5 data: {e}")
                raise ValueError(f"Failed to load ERA5 data. Please check your ERA5 configuration and data availability. Error: {e}")
            
            # Align ERA5 data with MSWEP time series if loaded
            if era5_data_dict:
                # Check if ERA5 and MSWEP have compatible time dimensions
                if len(era5_times) != len(train_times):
                    print(f"Warning: ERA5 time dimension ({len(era5_times)}) differs from MSWEP ({len(train_times)})")
                    # Attempt to align by dates
                    train_times_pd = pd.to_datetime(train_times).normalize()
                    era5_times_pd = pd.to_datetime(era5_times).normalize()
                    
                    # Create aligned ERA5 dict
                    aligned_era5_dict = {}
                    for var_name, var_data in era5_data_dict.items():
                        # Create DataFrame for alignment
                        era5_df = pd.DataFrame(
                            data=var_data.reshape(len(era5_times_pd), -1),
                            index=era5_times_pd
                        )
                        # Reindex to match MSWEP times
                        aligned_flat = era5_df.reindex(train_times_pd).values
                        
                        # Check for NaN values after reindexing
                        nan_count = np.isnan(aligned_flat).sum()
                        if nan_count > 0:
                            print(f"  Warning: {nan_count} NaN values found after aligning {var_name}")
                            # Fill NaN values with forward fill, then backward fill for any remaining
                            df_aligned = pd.DataFrame(aligned_flat)
                            df_aligned = df_aligned.ffill().bfill()
                            # If still NaN (e.g., all values are NaN), fill with mean of non-NaN values or 0
                            if df_aligned.isna().any().any():
                                mean_val = np.nanmean(var_data)
                                if np.isnan(mean_val):
                                    mean_val = 0.0
                                df_aligned = df_aligned.fillna(mean_val)
                                print(f"    Filled remaining NaNs with {mean_val:.4f}")
                            aligned_flat = df_aligned.values
                        
                        # Reshape back to spatial dimensions
                        aligned_era5_dict[var_name] = aligned_flat.reshape((len(train_times),) + var_data.shape[1:])
                    
                    era5_data_dict = aligned_era5_dict
                    print(f"Aligned ERA5 data to MSWEP time series")

        # Extract time coordinates for the samples we'll create
        if len(train_times) < 4:
            raise ValueError(f"Loaded data for years {start_year}-{required_end_year} has less than 4 time steps ({len(train_times)}). Cannot create samples.")
        # The times corresponding to targets after _create_samples will be time_coords[3:]
        # because we need 3 previous days for each input
        # With ERA5, we need at least 4 time steps (to access i-4), so samples start from index 4
        if era5_data_dict:
            sample_times = train_times[4:]  # Need extra lag for ERA5 alignment
        else:
            sample_times = train_times[3:]  # Original behavior without ERA5

        # Now call _create_samples with separate input and target sources and time coordinates
        inputs, targets = self._create_samples(input_data=train_series, target_data=target_series, 
                                              time_coords=train_times, era5_data_dict=era5_data_dict)

        # Store original targets 
        targets_original = targets

        # Apply log transform to targets if enabled
        if self.apply_log_transform:
            targets_transformed = self.target_scaler.transform(targets_original)
            print(f"Applied log transform to targets. Original range: [{targets_original.min():.4f}, {targets_original.max():.4f}] -> "
                  f"Log range: [{targets_transformed.min():.4f}, {targets_transformed.max():.4f}]")
        else:
            targets_transformed = targets_original
            print("No log transform applied to targets")

        # Convert all to tensors
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        targets_original_tensor = torch.tensor(targets_original, dtype=torch.float32)
        targets_transformed_tensor = torch.tensor(targets_transformed, dtype=torch.float32)

        # Create dataset with clear structure:
        # - inputs: (N, 5, 41, 121) - precipitation lags + seasonality features
        # - targets_original: (N, 41, 121) - original scale targets for evaluation
        # - targets_transformed: (N, 41, 121) - log-transformed targets for training (if enabled)
        full_data = TensorDataset(inputs_tensor, targets_original_tensor, targets_transformed_tensor)

        # Extract years from sample timestamps
        # Recall that sample_times are the timestamps corresponding to targets
        years = np.array([pd.Timestamp(t).year for t in sample_times])

        # Create masks for training and validation based on years
        train_indices = np.where((years >= start_year) & (years < current_val_year))[0]
        val_indices = np.where(years == current_val_year)[0]

        expected_days = 366 if calendar.isleap(current_val_year) else 365
        expected_samples = expected_days - 3 # Samples generated for a full year
        if len(val_indices) != expected_samples:
            # This warning is informative, not necessarily an error if data isn't perfectly continuous
            # or if start/end of year has partial data after windowing.
             print(f"Warning: Fold {self.fold} validation set generated {len(val_indices)} samples for year {current_val_year}. "
                   f"Expected approximately {expected_samples} for a full year. "
                   f"This might be normal if data for {current_val_year} is incomplete or at edges of the dataset.")


        if len(val_indices) == 0:
            raise ValueError(f"No validation data available for year {current_val_year} (fold {self.fold}) after filtering. Available sample years: {np.unique(years)}")
        if len(train_indices) == 0:
             raise ValueError(f"No training data available for years {start_year}-{current_val_year-1} (fold {self.fold}). Available sample years: {np.unique(years)}")


        self.train_dataset = Subset(full_data, train_indices)
        self.val_dataset = Subset(full_data, val_indices)
        self.val_times = sample_times[val_indices]

        print(f"Fold {self.fold}: Training on years {start_year}-{current_val_year-1} "
              f"({len(train_indices)} samples). Validating on year {current_val_year} "
              f"({len(val_indices)} samples).")
        
        # Clean up memory after creating datasets
        import gc
        del inputs, targets, inputs_tensor  # Delete large temporary arrays
        gc.collect()  # Force garbage collection
        print("Cleaned up temporary arrays to free memory.")

        if self.test_data_dir is not None:
            test_series, test_times = self._load_data(self.test_data_dir, start_year=-1, end_year=-1) # Load all test data

            if self.target_source.lower() == 'mswep':
                test_target_series = test_series
            elif self.target_source.lower() == 'hyras':
                hyras_dir_test = os.path.join(os.path.dirname(self.test_data_dir), "HYRAS_regridded")
                test_target_series, _ = self._load_hyras_data(hyras_dir_test, test_times)
                if test_target_series.shape != test_series.shape:
                    raise ValueError(f"Shape mismatch for test data: HYRAS target {test_target_series.shape} vs MSWEP input {test_series.shape}")
            else: # Should have been caught earlier
                test_target_series = test_series 

            # Load ERA5 test data if configured
            test_era5_data_dict = {}
            if (self.era5_variables or self.era5_single_level_variables) and self.era5_data_dir:
                # Determine test years from the loaded test data
                test_years = np.unique([pd.Timestamp(t).year for t in test_times])
                if len(test_years) > 0:
                    test_start_year = int(test_years.min())
                    test_end_year = int(test_years.max())
                    
                    # Try loading test ERA5 data
                    # First try with 'test' in the path
                    test_era5_dir = self.era5_data_dir.replace('train', 'test') if 'train' in self.era5_data_dir else self.era5_data_dir
                    if os.path.exists(test_era5_dir):
                        test_era5_data_dict, test_era5_times = self._load_era5_data(test_start_year, test_end_year)
                    else:
                        # Fall back to the same directory
                        test_era5_data_dict, test_era5_times = self._load_era5_data(test_start_year, test_end_year)
                    
                    # Align ERA5 data with test MSWEP time series if loaded
                    if test_era5_data_dict:
                        if len(test_era5_times) != len(test_times):
                            print(f"Warning: Test ERA5 time dimension ({len(test_era5_times)}) differs from test MSWEP ({len(test_times)})")
                            # Align by dates
                            test_times_pd = pd.to_datetime(test_times).normalize()
                            test_era5_times_pd = pd.to_datetime(test_era5_times).normalize()
                            
                            # Create aligned ERA5 dict
                            aligned_test_era5_dict = {}
                            for var_name, var_data in test_era5_data_dict.items():
                                # Create DataFrame for alignment
                                era5_df = pd.DataFrame(
                                    data=var_data.reshape(len(test_era5_times_pd), -1),
                                    index=test_era5_times_pd
                                )
                                # Reindex to match test MSWEP times
                                aligned_flat = era5_df.reindex(test_times_pd).values
                                
                                # Check for NaN values after reindexing
                                nan_count = np.isnan(aligned_flat).sum()
                                if nan_count > 0:
                                    print(f"  Warning: {nan_count} NaN values found after aligning {var_name} (test)")
                                    # Fill NaN values with forward fill, then backward fill for any remaining
                                    df_aligned = pd.DataFrame(aligned_flat)
                                    df_aligned = df_aligned.ffill().bfill()
                                    # If still NaN (e.g., all values are NaN), fill with mean of non-NaN values or 0
                                    if df_aligned.isna().any().any():
                                        mean_val = np.nanmean(var_data)
                                        if np.isnan(mean_val):
                                            mean_val = 0.0
                                        df_aligned = df_aligned.fillna(mean_val)
                                        print(f"    Filled remaining NaNs with {mean_val:.4f}")
                                    aligned_flat = df_aligned.values
                                
                                # Reshape back to spatial dimensions
                                aligned_test_era5_dict[var_name] = aligned_flat.reshape((len(test_times),) + var_data.shape[1:])
                            
                            test_era5_data_dict = aligned_test_era5_dict
                            print(f"Aligned test ERA5 data to test MSWEP time series")
                        
                        # Apply standardization to test ERA5 data using training statistics
                        if test_era5_data_dict and self.era5_stats:
                            print("\nApplying standardization to test ERA5 data using training statistics...")
                            test_era5_data_dict = self._apply_era5_standardization(test_era5_data_dict)
                            print("Test ERA5 standardization complete.")

            test_inputs, test_targets_orig = self._create_samples(input_data=test_series, target_data=test_target_series, 
                                                                 time_coords=test_times, era5_data_dict=test_era5_data_dict)
            
            # Apply log transform to test targets if enabled
            if self.apply_log_transform:
                test_targets_transformed = self.target_scaler.transform(test_targets_orig)
                print(f"Applied log transform to test targets. Original range: [{test_targets_orig.min():.4f}, {test_targets_orig.max():.4f}] -> "
                      f"Log range: [{test_targets_transformed.min():.4f}, {test_targets_transformed.max():.4f}]")
            else:
                test_targets_transformed = test_targets_orig
                print("No log transform applied to test targets")
            
            self.test_dataset = TensorDataset(
                torch.tensor(test_inputs, dtype=torch.float32),
                torch.tensor(test_targets_orig, dtype=torch.float32),
                torch.tensor(test_targets_transformed, dtype=torch.float32)
            )
        else:
            self.test_dataset = None

    def _load_hyras_data(self, hyras_dir, reference_times):
        """
        Load regridded HYRAS precipitation data and align it with the reference time series.
        
        Args:
            hyras_dir (str): Directory containing regridded HYRAS NetCDF files.
            reference_times (np.ndarray): Array of datetime64 values to align with.
            
        Returns:
            tuple: (aligned_data, reference_times) where aligned_data is an np.ndarray with the same
                   temporal dimension as reference_times and spatial dimensions matching MSWEP data.
        """
        if not os.path.exists(hyras_dir):
            raise ValueError(f"HYRAS directory {hyras_dir} not found")
        
        min_ref_year = pd.Timestamp(np.min(reference_times)).year
        max_ref_year = pd.Timestamp(np.max(reference_times)).year
        
        print(f"Loading HYRAS data for reference year range: {min_ref_year}-{max_ref_year}")
        
        hyras_datasets = []
        all_hyras_times = []
        
        for year in range(min_ref_year, max_ref_year + 1):
            file_path = os.path.join(hyras_dir, f"hyras_regridded_{year}.nc")
            if not os.path.exists(file_path):
                print(f"Warning: HYRAS file for year {year} not found: {file_path}")
                continue
                
            try:
                ds = xr.open_dataset(file_path)
                precip_var = 'pr'
                if precip_var not in ds:
                    for candidate in ['precip', 'precipitation']:
                        if candidate in ds:
                            precip_var = candidate
                            break
                    else:
                        raise ValueError(f"No precipitation variable found in {file_path}")
                
                data = ds[precip_var].values
                times_current_file = ds[precip_var].coords['time'].values # Ensure this is np.datetime64
                
                hyras_datasets.append(data)
                all_hyras_times.append(times_current_file)
                ds.close()
            except Exception as e:
                print(f"Error loading HYRAS file {file_path}: {e}")
                continue
        
        if not hyras_datasets:
            # Try to find any HYRAS file if specific year range failed, as a fallback for general alignment
            all_hyras_files = sorted([os.path.join(hyras_dir, f) for f in os.listdir(hyras_dir) if f.startswith('hyras_regridded_') and f.endswith('.nc')])
            if not all_hyras_files:
                 raise ValueError(f"No valid HYRAS data found in {hyras_dir} for years {min_ref_year}-{max_ref_year}, and no fallback files found.")
            
            print(f"Warning: No HYRAS data for specific years {min_ref_year}-{max_ref_year}. Attempting to load all available HYRAS files for alignment.")
            hyras_datasets, all_hyras_times = [], []
            for file_path in all_hyras_files:
                try:
                    ds = xr.open_dataset(file_path)
                    precip_var = 'pr' # (logic repeated for brevity, ensure consistent var name)
                    if precip_var not in ds: precip_var = 'precip' # fallback
                    if precip_var not in ds: raise ValueError("No precip var")

                    hyras_datasets.append(ds[precip_var].values)
                    all_hyras_times.append(ds[precip_var].coords['time'].values)
                    ds.close()
                except: continue # Skip files that fail

            if not hyras_datasets:
                 raise ValueError(f"No valid HYRAS data found in {hyras_dir} even after trying all files.")


        combined_hyras_data = np.concatenate(hyras_datasets, axis=0)
        combined_hyras_times = np.concatenate(all_hyras_times, axis=0) # This should be np.datetime64
        
        # Convert reference_times to pandas Timestamps for easier date matching if not already
        # Ensure reference_times are also np.datetime64 for direct comparison if possible
        reference_times_pd = pd.to_datetime(reference_times).normalize()
        combined_hyras_times_pd = pd.to_datetime(combined_hyras_times).normalize()

        # Create a DataFrame for HYRAS data for easier reindexing
        hyras_df = pd.DataFrame(
            data=combined_hyras_data.reshape(len(combined_hyras_times_pd), -1), # Flatten spatial dims
            index=combined_hyras_times_pd
        )
        
        # Reindex HYRAS data to match reference_times, filling missing with NaN
        # This aligns based on date, effectively.
        aligned_hyras_flat = hyras_df.reindex(reference_times_pd).values
        
        # Reshape back to original spatial dimensions
        spatial_dims = combined_hyras_data.shape[1:]
        aligned_data = aligned_hyras_flat.reshape((len(reference_times_pd),) + spatial_dims)
        
        # Count NaNs to see how many were missing
        missing_count = np.sum(np.isnan(aligned_data[:,0,0])) # Check NaNs in one pixel over time
        found_count = len(reference_times_pd) - missing_count
        
        if missing_count > 0:
            print(f"Warning: {missing_count} HYRAS timestamps could not be aligned with reference MSWEP times and were filled with NaNs.")
            # Optional: fill NaNs if appropriate, e.g., with 0 or through interpolation,
            # but for targets, NaNs might be acceptable if handled in loss/metrics.
            # For now, we keep NaNs. If targets must be non-NaN, they need to be filled.
            # Example: np.nan_to_num(aligned_data, nan=0.0, copy=False)

        print(f"Aligned HYRAS data with shape {aligned_data.shape}")
        print(f"Found {found_count} matching times, {missing_count} missing/unaligned times (filled with NaN if any).")
        return aligned_data, reference_times

    def _load_era5_data(self, start_year, end_year):
        """
        Load ERA5 atmospheric variables from preprocessed NetCDF files.
        
        Files are expected to be named: era5_uvq{pressure}_{year}_regrid.nc
        Each file contains all variables (u, v, q) for a specific pressure level and year.
        
        Args:
            start_year (int): The first year of data to load (inclusive).
            end_year (int): The last year of data to load (inclusive).
            
        Returns:
            tuple: (era5_data_dict, time_coords) where:
                - era5_data_dict is a dict mapping variable names to numpy arrays with shape (T, H, W)
                - time_coords is an array of datetime64 values
        """
        if not self.era5_variables and not self.era5_single_level_variables:
            return {}, np.array([])
            
        if not self.era5_data_dir:
            raise ValueError(f"ERA5 data directory not provided")
        
        if not os.path.exists(self.era5_data_dir):
            raise ValueError(f"ERA5 directory {self.era5_data_dir} not found")
        
        print(f"Loading ERA5 data from {self.era5_data_dir} for years {start_year}-{end_year}")
        
        # Dictionary to store loaded data for each variable
        era5_data_dict = {}
        common_times = None
        
        # Check if we should look in regridded subdirectory
        regridded_dir = os.path.join(self.era5_data_dir, "regridded")
        use_regridded = os.path.exists(regridded_dir) and os.listdir(regridded_dir)
        if use_regridded:
            era5_dir = regridded_dir
            print(f"Found regridded subdirectory, using: {era5_dir}")
        else:
            era5_dir = self.era5_data_dir
        
        # Load pressure-level variables
        if self.era5_variables and self.era5_pressure_levels:
            # Since files now contain all variables, we load by pressure level
            # and extract the variables we need
            loaded_pressure_levels = set()
            
            for pressure_level in self.era5_pressure_levels:
                print(f"Loading data for pressure level {pressure_level}hPa...")
                
                yearly_data_by_var = {var: [] for var in self.era5_variables}
                yearly_times_list = []
                
                for year in range(start_year, end_year + 1):
                    # New file naming pattern: era5_uvq{pressure}_{year}_regrid.nc
                    file_path = os.path.join(era5_dir, f"era5_uvq{pressure_level}_{year}_regrid.nc")
                    
                    if os.path.exists(file_path):
                        try:
                            print(f"  Loading year {year} from {os.path.basename(file_path)}")
                            ds = xr.open_dataset(file_path)
                            
                            # Handle different time coordinate names
                            time_coord_name = 'time'
                            if 'valid_time' in ds.dims:
                                time_coord_name = 'valid_time'
                            
                            # Get time coordinates
                            if time_coord_name in ds.coords:
                                times = ds.coords[time_coord_name].values
                            else:
                                raise ValueError(f"No time coordinates found (tried 'time' and 'valid_time')")
                            
                            # Extract each requested variable
                            for var_name in self.era5_variables:
                                if var_name in ds.data_vars:
                                    data = ds[var_name].values
                                    
                                    # Handle extra dimensions
                                    if data.ndim == 4 and data.shape[1] == 1:
                                        data = data.squeeze(axis=1)
                                    
                                    if data.ndim != 3:
                                        raise ValueError(f"Variable {var_name} should be 3D, got shape {data.shape}")
                                    
                                    yearly_data_by_var[var_name].append(data)
                                else:
                                    print(f"  Warning: Variable '{var_name}' not found in {file_path}")
                                    print(f"  Available variables: {list(ds.data_vars)}")
                            
                            yearly_times_list.append(times)
                            ds.close()
                            
                            # Clean up xarray dataset to free memory
                            del ds
                            
                        except Exception as e:
                            print(f"  Error loading {file_path}: {e}")
                            continue
                    else:
                        print(f"  File not found for year {year}")
                
                # Concatenate yearly data for each variable at this pressure level
                if yearly_times_list:
                    concatenated_times = np.concatenate(yearly_times_list, axis=0)
                    
                    for var_name in self.era5_variables:
                        if yearly_data_by_var[var_name]:
                            concatenated_data = np.concatenate(yearly_data_by_var[var_name], axis=0)
                            var_key = f"{var_name}{pressure_level}"
                            era5_data_dict[var_key] = concatenated_data
                            print(f"  Successfully loaded {var_key} with shape {concatenated_data.shape}")
                    
                    # Update common times
                    if common_times is None:
                        common_times = concatenated_times
                    else:
                        if len(concatenated_times) != len(common_times):
                            print(f"  Warning: Time dimension mismatch for pressure level {pressure_level}")
                    
                    loaded_pressure_levels.add(pressure_level)
        
        # Load single-level variables
        if self.era5_single_level_variables:
            print(f"Loading single-level ERA5 variables...")
            
            for var_name in self.era5_single_level_variables:
                # Single-level files are not split by year, just one file per variable
                file_path = os.path.join(era5_dir, f"era5_{var_name}_regrid.nc")
                
                if os.path.exists(file_path):
                    try:
                        print(f"  Loading {var_name} from {os.path.basename(file_path)}")
                        ds = xr.open_dataset(file_path)
                        
                        # Handle different time coordinate names
                        time_coord_name = 'time'
                        if 'valid_time' in ds.dims:
                            time_coord_name = 'valid_time'
                        
                        # Get time coordinates
                        if time_coord_name in ds.coords:
                            times = ds.coords[time_coord_name].values
                        else:
                            raise ValueError(f"No time coordinates found in {file_path}")
                        
                        # Get the data
                        if var_name in ds.data_vars:
                            data = ds[var_name].values
                        else:
                            # Try common alternative names
                            found = False
                            for alt_name in [var_name.upper(), var_name.lower()]:
                                if alt_name in ds.data_vars:
                                    data = ds[alt_name].values
                                    found = True
                                    break
                            if not found:
                                print(f"  Warning: Variable '{var_name}' not found in {file_path}")
                                print(f"  Available variables: {list(ds.data_vars)}")
                                ds.close()
                                continue
                        
                        # Handle extra dimensions
                        if data.ndim == 4 and data.shape[1] == 1:
                            data = data.squeeze(axis=1)
                        
                        if data.ndim != 3:
                            raise ValueError(f"Single-level variable {var_name} should be 3D, got shape {data.shape}")
                        
                        # Filter data by year range if needed
                        if start_year != -1 or end_year != -1:
                            times_pd = pd.to_datetime(times)
                            year_mask = np.ones(len(times), dtype=bool)
                            
                            if start_year != -1:
                                year_mask &= (times_pd.year >= start_year)
                            if end_year != -1:
                                year_mask &= (times_pd.year <= end_year)
                            
                            data = data[year_mask]
                            times = times[year_mask]
                        
                        # Store in dictionary with just the variable name (no pressure level)
                        era5_data_dict[var_name] = data
                        print(f"  Successfully loaded {var_name} with shape {data.shape}")
                        
                        # Update common times
                        if common_times is None:
                            common_times = times
                        else:
                            if len(times) != len(common_times):
                                print(f"  Warning: Time dimension mismatch for single-level variable {var_name}")
                        
                        ds.close()
                        
                        # Clean up to free memory
                        del ds
                        
                    except Exception as e:
                        print(f"  Error loading {file_path}: {e}")
                        continue
                else:
                    print(f"  File not found: {file_path}")
        
        # Check if we got all requested data
        missing_vars = []
        if self.era5_variables:
            for var_name in self.era5_variables:
                for pressure_level in self.era5_pressure_levels:
                    var_key = f"{var_name}{pressure_level}"
                    if var_key not in era5_data_dict:
                        missing_vars.append(var_key)
        
        if self.era5_single_level_variables:
            for var_name in self.era5_single_level_variables:
                if var_name not in era5_data_dict:
                    missing_vars.append(var_name)
        
        if not era5_data_dict:
            error_msg = (f"Failed to load any ERA5 variables. "
                        f"Requested pressure-level: {[f'{v}{p}' for v in self.era5_variables for p in self.era5_pressure_levels]}. "
                        f"Requested single-level: {self.era5_single_level_variables}. "
                        f"Please check that ERA5 files exist in {era5_dir}")
            raise ValueError(error_msg)
        
        if missing_vars:
            print(f"Warning: Some ERA5 variables could not be loaded: {missing_vars}")
            print(f"Successfully loaded: {list(era5_data_dict.keys())}")
        
        print(f"Successfully loaded {len(era5_data_dict)} ERA5 variables: {list(era5_data_dict.keys())}")
        
        # Apply standardization to all ERA5 variables
        if era5_data_dict:
            print("\nStandardizing ERA5 variables to zero mean and unit variance...")
            self.era5_stats = self._calculate_era5_stats(era5_data_dict)
            
            # Apply standardization to each variable
            for var_name, data in era5_data_dict.items():
                if var_name in self.era5_stats:
                    mean_val = self.era5_stats[var_name]['mean']
                    std_val = self.era5_stats[var_name]['std']
                    
                    # Standardize: (x - mean) / std
                    era5_data_dict[var_name] = (data - mean_val) / std_val
                    
                    # Verify standardization
                    standardized_mean = np.nanmean(era5_data_dict[var_name])
                    standardized_std = np.nanstd(era5_data_dict[var_name])
                    print(f"  {var_name} standardized: mean≈{standardized_mean:.3e}, std≈{standardized_std:.3f}")
            
            print("ERA5 standardization complete.")
            
            # Force garbage collection to free up memory from original arrays
            import gc
            gc.collect()
        
        return era5_data_dict, common_times

    def _load_data(self, directory, start_year, end_year):
        """
        Load and combine MSWEP daily precipitation data for a specific year range.

        Each file is assumed to be named in the pattern "mswep_daily_YYYYMM.nc" and to contain
        a variable "precip" with shape (days, 41, 121). Files are filtered based on the year
        range and then concatenated along the time axis.

        Args:
            directory (str): Path to the directory containing monthly NetCDF files.
            start_year (int): The first year of data to load (inclusive). Use -1 to ignore start year.
            end_year (int): The last year of data to load (inclusive). Use -1 to ignore end year.


        Returns:
            tuple: (combined_data, time_coords) where combined_data is an np.ndarray with shape
            (total_days, 41, 121) and time_coords is an array of datetime64 values.
        """
        # Define range_str early for use in all messages and the final print
        range_str = f"{start_year if start_year != -1 else 'any'}-{end_year if end_year != -1 else 'any'}"

        all_files = sorted([f for f in os.listdir(directory) if f.endswith('.nc') and f.startswith('mswep_daily_')])
        if not all_files:
            raise ValueError(f"No NetCDF files matching pattern 'mswep_daily_*.nc' found in {directory}")

        files_to_load = []
        # Filter files based on the year range derived from filename
        for f in all_files:
            try:
                # Extract YYYYMM from filename like "mswep_daily_YYYYMM.nc"
                year_month_str = f.split('_')[-1].split('.')[0]
                year = int(year_month_str[:4])
                # Only include files within the desired year range (ignore if -1)
                if (start_year == -1 or year >= start_year) and \
                   (end_year == -1 or year <= end_year):
                    files_to_load.append(os.path.join(directory, f))
            except (IndexError, ValueError):
                print(f"Warning: Could not parse year from filename {f}. Skipping.")
                continue

        if not files_to_load:
            # Now range_str is already defined
            raise ValueError(f"No NetCDF files found in the year range {range_str} in {directory}")

        # This print statement can also use the pre-defined range_str
        print(f"Found {len(files_to_load)} files to load for years {range_str}.")


        all_data = []
        all_times = []  # Store time coordinates
        # Now loop only through the filtered files
        for file_path in files_to_load: # <-- Use filtered list
            try:
                ds = xr.open_dataset(file_path)
            except Exception as e:
                print(f"Error opening {file_path}: {e}")
                continue
            # Expect the primary variable to be named 'precip'
            precip_var = 'precip'
            if precip_var not in ds:
                for candidate in ['precipitation', 'pr']:
                    if candidate in ds:
                        precip_var = candidate
                        break
                else:
                    print(f"Warning: No precipitation variable found in {file_path}. Available variables: {list(ds.data_vars)}")
                    ds.close()
                    continue
            data = ds[precip_var].values  # Expected shape: (days, 41, 121)

            # Extract time coordinates
            if 'time' in ds[precip_var].coords:
                times = ds[precip_var].coords['time'].values
            else:
                print(f"Warning: No time coordinates found in {file_path}. Creating dummy timestamps.")
                # Create dummy timestamps if not available
                times = np.array([np.datetime64('2000-01-01')] * len(data), dtype='datetime64[D]')
            
            ds.close()
            if data.ndim != 3:
                print(f"Warning: Data in {file_path} does not have 3 dimensions (found shape {data.shape}). Skipping.")
                continue
            if data.shape[1] != 41 or data.shape[2] != 121:
                print(f"Warning: Data in {file_path} has shape {data.shape} but expected spatial dimensions (41, 121). Skipping.")
                continue
            all_data.append(data)
            all_times.append(times)

        if not all_data:
            # range_str is already defined
            raise ValueError(f"No valid data loaded for years {range_str} in {directory}")

        combined_data = np.concatenate(all_data, axis=0)
        combined_times = np.concatenate(all_times, axis=0)
        
        # range_str is already defined, no need to redefine it here.
        print(f"Loaded data from {directory} for {range_str} with shape {combined_data.shape} and {len(combined_times)} time steps")
        return combined_data, combined_times

    def _create_samples(self, input_data, target_data, time_coords, era5_data_dict=None):
        """
        Create samples using a 3-lag structure with seasonality features and optionally ERA5 variables.
        
        For ERA5 temporal alignment:
        - ERA5 data at 18:00 UTC is used to predict precipitation at 00:00 UTC the next day
        - For precipitation lag at day t-1, the corresponding ERA5 data is from day t-2
        - So: i-3 precip → i-4 ERA5, i-2 precip → i-3 ERA5, i-1 precip → i-2 ERA5
        
        ERA5 variables can be either:
        - Pressure-level variables (named as f"{var}{pressure}" in era5_data_dict)
        - Single-level variables (named as just "{var}" in era5_data_dict)
        
        Args:
            input_data: MSWEP precipitation data, shape (T, H, W)
            target_data: Target precipitation data, shape (T, H, W)
            time_coords: Time coordinates array
            era5_data_dict: Optional dict of ERA5 variables, each with shape (T, H, W)
                           Keys can be like "u500", "v850" for pressure levels, or "msl", "tcwv" for single-level
        
        Returns:
            tuple: (inputs, targets) arrays
        """
        T = input_data.shape[0]
        if T < 4:
            raise ValueError(f"Not enough time steps ({T}) to create samples. Need at least 4.")
        if input_data.shape[0] != target_data.shape[0]:
            raise ValueError(f"Input and target data must have the same time dimension. Input: {input_data.shape[0]}, Target: {target_data.shape[0]}")
        if len(time_coords) != T:
            raise ValueError(f"Time coordinates and data must have the same time dimension. Time: {len(time_coords)}, Data: {T}")
        
        # Check if we need ERA5 data from earlier time steps
        if era5_data_dict:
            # We need at least 4 time steps to access i-4 for the first sample at i=3
            if T < 4:
                raise ValueError(f"Not enough time steps ({T}) for ERA5 alignment. Need at least 4.")
            # Verify all ERA5 variables have correct shape
            for var_name, var_data in era5_data_dict.items():
                if var_data.shape != input_data.shape:
                    raise ValueError(f"ERA5 variable {var_name} has shape {var_data.shape}, expected {input_data.shape}")
            
        inputs = []
        targets_for_samples = [] # Renamed to avoid confusion with 'targets' argument name
        
        # Determine starting index based on ERA5 availability
        start_idx = 4 if era5_data_dict else 3  # Need i-4 for ERA5 alignment
        
        for i in range(start_idx, T):
            # Stack precipitation lags
            precip_lags = np.stack([input_data[i-3], input_data[i-2], input_data[i-1]], axis=0)
            
            # Create seasonality features
            try:
                timestamp_i = pd.Timestamp(time_coords[i])
                doy = timestamp_i.dayofyear
            except Exception as e:
                print(f"Warning: Could not process timestamp {time_coords[i]} at index {i}. Using default day 1. Error: {e}")
                doy = 1
            
            sin_doy_val = np.sin(2 * np.pi * doy / 365.25)
            cos_doy_val = np.cos(2 * np.pi * doy / 365.25)
            
            spatial_shape = input_data.shape[1:]
            sin_grid = np.full(spatial_shape, sin_doy_val, dtype=input_data.dtype)
            cos_grid = np.full(spatial_shape, cos_doy_val, dtype=input_data.dtype)
            
            # Start with precipitation lags and seasonality
            channels_to_stack = [
                precip_lags,  # Shape: (3, H, W)
                sin_grid[np.newaxis, :, :],  # Shape: (1, H, W)
                cos_grid[np.newaxis, :, :]   # Shape: (1, H, W)
            ]
            
            # Add ERA5 variables if available
            if era5_data_dict:
                # For each precipitation lag, we need the corresponding ERA5 data offset by one day
                # i-3 precip → i-4 ERA5
                # i-2 precip → i-3 ERA5
                # i-1 precip → i-2 ERA5
                
                # Stack ERA5 variables for each lag
                for lag_offset in [4, 3, 2]:  # Corresponds to i-4, i-3, i-2
                    for var_name in sorted(era5_data_dict.keys()):  # Sort for consistent ordering
                        era5_slice = era5_data_dict[var_name][i-lag_offset]
                        channels_to_stack.append(era5_slice[np.newaxis, :, :])  # Shape: (1, H, W)
            
            # Stack all channels
            sample_with_features = np.concatenate(channels_to_stack, axis=0)
            
            # Log the expected shape
            expected_channels = 5  # 3 precip lags + 2 seasonality
            if era5_data_dict:
                expected_channels += 3 * len(era5_data_dict)  # 3 lags × number of ERA5 variables
            
            if sample_with_features.shape[0] != expected_channels:
                print(f"Warning: Sample at index {i} has {sample_with_features.shape[0]} channels, expected {expected_channels}")
            
            inputs.append(sample_with_features)
            targets_for_samples.append(target_data[i])
        
        inputs_np = np.array(inputs, dtype=np.float32)
        targets_np = np.array(targets_for_samples, dtype=np.float32)
        
        # Update logging to include ERA5 info
        if era5_data_dict:
            print(f"Created {inputs_np.shape[0]} samples with input shape {inputs_np.shape[1:]} "
                  f"(including {len(era5_data_dict)} ERA5 variables) and target shape {targets_np.shape[1:]} "
                  f"from data with {T} time steps.")
        else:
            print(f"Created {inputs_np.shape[0]} samples with input shape {inputs_np.shape[1:]} "
                  f"and target shape {targets_np.shape[1:]} from data with {T} time steps.")
        
        return inputs_np, targets_np


    def train_dataloader(self):
        if self.train_dataset is None: self.setup(stage='fit')
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, # Shuffle training data
                          num_workers=self.num_workers, pin_memory=True, prefetch_factor=2 if self.num_workers > 0 else None)

    def val_dataloader(self):
        if self.val_dataset is None: self.setup(stage='fit')
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, prefetch_factor=2 if self.num_workers > 0 else None)

    def test_dataloader(self):
        if self.test_dataset is None: self.setup(stage='test')
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                              num_workers=self.num_workers, pin_memory=True, prefetch_factor=2 if self.num_workers > 0 else None)
        return None

if __name__ == "__main__":
    # Example usage:
    # Create a dummy data directory and file for testing
    dummy_data_root = "dummy_mswep_data_monthly"
    dummy_mswep_dir = os.path.join(dummy_data_root, "MSWEP_daily")
    dummy_hyras_dir = os.path.join(dummy_data_root, "HYRAS_regridded")
    os.makedirs(dummy_mswep_dir, exist_ok=True)
    os.makedirs(dummy_hyras_dir, exist_ok=True)
    
    # Create dummy NetCDF files for MSWEP (monthly)
    lats = np.arange(30, 30 + 41*1.0, 1.0)
    lons = np.arange(-70, -70 + 121*1.0, 1.0)
    base_start_date = "2007-01-01"
    num_years_data = 8 # 2007 to 2014 to allow for several folds (e.g. 2010, 2011, 2012 as val)

    for year in range(pd.Timestamp(base_start_date).year, pd.Timestamp(base_start_date).year + num_years_data):
        for month in range(1, 13):
            month_start = f"{year}-{month:02d}-01"
            days_in_month = pd.Timestamp(month_start).days_in_month
            times_monthly = pd.date_range(start=month_start, periods=days_in_month, freq='D', name="time")
            
            # MSWEP
            mswep_precip_data = np.random.rand(len(times_monthly), len(lats), len(lons)).astype(np.float32) * 10
            mswep_ds = xr.Dataset(
                {"precip": (("time", "latitude", "longitude"), mswep_precip_data)},
                coords={"time": times_monthly, "latitude": lats, "longitude": lons}
            )
            mswep_file_path = os.path.join(dummy_mswep_dir, f"mswep_daily_{year}{month:02d}.nc")
            mswep_ds.to_netcdf(mswep_file_path)

            # HYRAS (same structure, different values for testing)
            if year <= pd.Timestamp(base_start_date).year + num_years_data -1 : # Create HYRAS for same period
                hyras_precip_data = np.random.rand(len(times_monthly), len(lats), len(lons)).astype(np.float32) * 12 
                hyras_ds = xr.Dataset(
                    {"pr": (("time", "latitude", "longitude"), hyras_precip_data)}, # HYRAS uses 'pr'
                    coords={"time": times_monthly, "latitude": lats, "longitude": lons}
                )
                hyras_file_path = os.path.join(dummy_hyras_dir, f"hyras_regridded_{year}.nc") # Yearly files for HYRAS
                # For simplicity, let's save HYRAS yearly, then the _load_hyras_data will pick it up
                # This means the dummy yearly file would get overwritten each month, so let's accumulate then save.
    
    # Consolidate and save HYRAS yearly after monthly loop
    for year in range(pd.Timestamp(base_start_date).year, pd.Timestamp(base_start_date).year + num_years_data):
        yearly_hyras_data = []
        yearly_hyras_times = []
        for month in range(1,13):
            month_start = f"{year}-{month:02d}-01"
            days_in_month = pd.Timestamp(month_start).days_in_month
            times_monthly = pd.date_range(start=month_start, periods=days_in_month, freq='D', name="time")
            hyras_precip_data_month = np.random.rand(len(times_monthly), len(lats), len(lons)).astype(np.float32) * 12
            yearly_hyras_data.append(hyras_precip_data_month)
            yearly_hyras_times.append(times_monthly)
        
        concat_hyras_data = np.concatenate(yearly_hyras_data, axis=0)
        concat_hyras_times = pd.DatetimeIndex(np.concatenate(yearly_hyras_times))

        hyras_ds_yearly = xr.Dataset(
            {"pr": (("time", "latitude", "longitude"), concat_hyras_data)},
            coords={"time": concat_hyras_times, "latitude": lats, "longitude": lons}
        )
        hyras_file_path = os.path.join(dummy_hyras_dir, f"hyras_regridded_{year}.nc")
        hyras_ds_yearly.to_netcdf(hyras_file_path)
        print(f"Created dummy HYRAS yearly file: {hyras_file_path}")


    print(f"Created dummy data files in {dummy_data_root}")

    num_cv_folds = 3 # e.g., validate on 2010, 2011, 2012
    for current_fold_idx in range(num_cv_folds):
        print(f"\n--- Testing DataModule for Fold {current_fold_idx} ---")
        # Test with MSWEP as target
        dm_mswep_target = MSWEPDataModule(data_dir=dummy_mswep_dir, batch_size=4, num_workers=0,
                                 fold=current_fold_idx, num_total_folds=num_cv_folds, target_source='mswep')
        dm_mswep_target.setup(stage='fit')
        print(f"MSWEP Target - Fold {current_fold_idx}: Train size: {len(dm_mswep_target.train_dataset)}, Val size: {len(dm_mswep_target.val_dataset)}") # type: ignore
        if dm_mswep_target.val_times is not None:
             print(f"  Val times from {pd.to_datetime(dm_mswep_target.val_times[0])} to {pd.to_datetime(dm_mswep_target.val_times[-1])}")

        # Test with HYRAS as target
        dm_hyras_target = MSWEPDataModule(data_dir=dummy_mswep_dir, batch_size=4, num_workers=0,
                                 fold=current_fold_idx, num_total_folds=num_cv_folds, target_source='hyras')
        dm_hyras_target.setup(stage='fit')
        print(f"HYRAS Target - Fold {current_fold_idx}: Train size: {len(dm_hyras_target.train_dataset)}, Val size: {len(dm_hyras_target.val_dataset)}") # type: ignore
        if dm_hyras_target.val_times is not None:
            print(f"  Val times from {pd.to_datetime(dm_hyras_target.val_times[0])} to {pd.to_datetime(dm_hyras_target.val_times[-1])}")


        # Check one batch from train_loader (MSWEP target)
        train_loader = dm_mswep_target.train_dataloader()
        if train_loader and len(train_loader)>0:
            inputs_b, targets_orig_b, targets_transformed_b = next(iter(train_loader))
            print(f"  Train Batch (MSWEP target): Inputs {inputs_b.shape}, Targets Orig {targets_orig_b.shape}, Targets Transformed {targets_transformed_b.shape}")
            # Ensure targets_orig_b and targets_transformed_b are the same since UNetLightningModule handles transform
            assert torch.allclose(targets_orig_b, targets_transformed_b), "Original and transformed targets from DM should be same if UNetLightningModule handles scaling"


    # Clean up dummy data (optional)
    # import shutil
    # shutil.rmtree(dummy_data_root)
    # print(f"Cleaned up dummy data directory: {dummy_data_root}")
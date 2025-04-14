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
    def __init__(self, offset: float = 0.01) -> None:
        self.offset = offset

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
        target_source (str): Source for target data ('mswep' or 'hyras').
    """
    def __init__(self, data_dir, test_data_dir=None, batch_size=32, num_workers=4,
                 fold=0, target_source='mswep'):
        super().__init__()
        self.data_dir = data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold = fold
        self.target_source = target_source

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
        # Define constants for the year-based splitting
        start_year = 2007  # First year of available data
        base_val_year = 2010  # First validation year (fold 0)
        current_val_year = base_val_year + self.fold

        # Determine the required end year for loading data (includes validation year)
        required_end_year = current_val_year # We need data up to and including the validation year

        # Load and combine training data *only for the required years*
        train_series, train_times = self._load_data(self.data_dir, start_year, required_end_year) # <-- Pass year range

        # Load target data based on specified source (if different)
        if self.target_source == 'mswep':
            # Target data source is the same as input source
            target_series = train_series
            print(f"Using MSWEP as target data source.")
        elif self.target_source == 'hyras':
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

        # Extract time coordinates for the samples we'll create
        if len(train_times) < 4:
             raise ValueError(f"Loaded data for years {start_year}-{required_end_year} has less than 4 time steps ({len(train_times)}). Cannot create samples.")
        # The times corresponding to targets after _create_samples will be time_coords[3:]
        # because we need 3 previous days for each input
        sample_times = train_times[3:] # Timestamps corresponding to the targets

        # Now call _create_samples with separate input and target sources and time coordinates
        inputs, targets = self._create_samples(input_data=train_series, target_data=target_series, time_coords=train_times)

        # Store original targets only
        targets_original = targets

        # Convert all to tensors
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        targets_original_tensor = torch.tensor(targets_original, dtype=torch.float32)

        # Create a full dataset with three components:
        # inputs: (N, 5, 41, 121), original targets: (N, 41, 121), original targets again: (N, 41, 121)
        # The second copy of original targets will be transformed by the LightningModule
        full_data = TensorDataset(inputs_tensor, targets_original_tensor, targets_original_tensor)

        # Extract years from sample timestamps
        # Recall that sample_times are the timestamps corresponding to targets
        years = np.array([pd.Timestamp(t).year for t in sample_times])

        # Create masks for training and validation based on years
        train_indices = np.where((years >= start_year) & (years < current_val_year))[0]
        val_indices = np.where(years == current_val_year)[0] # <-- This logic should now work correctly

        # *** Add a check for validation set size ***
        expected_days = 366 if calendar.isleap(current_val_year) else 365
        # We expect N-3 samples for a full year, where N is days in year.
        # Need to adjust expected count based on the 3-day lag.
        # However, a simpler check is just len(val_indices) > 0 and print a warning if not expected_days - 3
        # Let's adjust the warning to be about the number of samples generated for the validation year.
        # The actual number of samples generated will be `days_in_year - 3` if data is continuous.
        expected_samples = expected_days - 3
        if len(val_indices) != expected_samples:
             print(f"Warning: Fold {self.fold} validation set generated {len(val_indices)} samples, "
                   f"expected approximately {expected_samples} samples for year {current_val_year}. "
                   f"Check data continuity/completeness or start/end dates.")

        # Check if we have sufficient data for the requested fold
        if len(val_indices) == 0:
            raise ValueError(f"No validation data available for year {current_val_year} (fold {self.fold}) after filtering loaded data. Available years in samples: {np.unique(years)}")

        # Create the training and validation datasets
        self.train_dataset = Subset(full_data, train_indices)
        self.val_dataset = Subset(full_data, val_indices)

        # --- ADDITION: Store validation timestamps ---
        self.val_times = sample_times[val_indices] # Store the actual timestamps for the validation set

        # Log information about the split
        print(f"Fold {self.fold}: Training on years {start_year}-{current_val_year-1} "
              f"({len(train_indices)} samples). Validating on year {current_val_year} "
              f"({len(val_indices)} samples).")

        # Optionally load test data if provided
        if self.test_data_dir is not None:
            # --- Consider applying year filtering to test data loading if necessary ---
            # For now, assuming test data is loaded entirely, but this might need adjustment
            # depending on how test sets are defined (e.g., specific years).
            # If test data also needs year filtering, calculate test_start_year, test_end_year
            # and pass them to _load_data similarly.
            test_series, test_times = self._load_data(self.test_data_dir, start_year=-1, end_year=-1) # Placeholder: Adjust years if needed

            # Load test target data based on specified source
            if self.target_source == 'mswep':
                test_target_series = test_series
                print(f"Using MSWEP as test target data source.")
            elif self.target_source == 'hyras':
                print(f"Using HYRAS (regridded) as test target data source.")
                hyras_dir = os.path.join(os.path.dirname(self.test_data_dir), "HYRAS_regridded")
                # Assuming _load_hyras_data aligns correctly
                test_target_series, _ = self._load_hyras_data(hyras_dir, test_times)
                if not isinstance(test_target_series, np.ndarray):
                    raise TypeError("Loaded HYRAS test target data must be a numpy array.")
                if test_target_series.shape != test_series.shape:
                     raise ValueError(f"Shape mismatch: HYRAS test target {test_target_series.shape} vs MSWEP test input {test_series.shape}")
                print(f"Successfully loaded regridded HYRAS test data with shape {test_target_series.shape}.")
            else:
                raise ValueError(f"Unknown target_source: '{self.target_source}'. Choose 'mswep' or 'hyras'.")

            # Create samples with separate input and target sources
            test_inputs, test_targets = self._create_samples(input_data=test_series, target_data=test_target_series, time_coords=test_times)

            # Handle test targets - pass original twice
            test_targets_original = test_targets

            # Create test dataset with three components
            self.test_dataset = TensorDataset(
                torch.tensor(test_inputs, dtype=torch.float32),
                torch.tensor(test_targets_original, dtype=torch.float32),
                torch.tensor(test_targets_original, dtype=torch.float32)
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
        
        # Extract years from reference times to know which files to load
        years = np.unique([pd.Timestamp(t).year for t in reference_times])
        print(f"Loading HYRAS data for years: {years}")
        
        # Load each yearly file
        hyras_datasets = []
        all_times = []
        
        for year in years:
            file_path = os.path.join(hyras_dir, f"hyras_regridded_{year}.nc")
            if not os.path.exists(file_path):
                print(f"Warning: HYRAS file for year {year} not found: {file_path}")
                continue
                
            try:
                ds = xr.open_dataset(file_path)
                # Get the precipitation variable (assumed to be 'pr' based on the original file)
                precip_var = 'pr'
                if precip_var not in ds:
                    for candidate in ['precip', 'precipitation']:
                        if candidate in ds:
                            precip_var = candidate
                            break
                    else:
                        raise ValueError(f"No precipitation variable found in {file_path}")
                
                data = ds[precip_var].values
                times = ds[precip_var].coords['time'].values
                
                hyras_datasets.append(data)
                all_times.append(times)
                ds.close()
            except Exception as e:
                print(f"Error loading HYRAS file {file_path}: {e}")
                continue
        
        if not hyras_datasets:
            raise ValueError(f"No valid HYRAS data found in {hyras_dir}")
        
        # Combine all loaded data
        combined_data = np.concatenate(hyras_datasets, axis=0)
        combined_times = np.concatenate(all_times, axis=0)
        
        # Create a time-indexed dictionary for faster lookup
        data_dict = {pd.Timestamp(t).to_datetime64(): d for t, d in zip(combined_times, combined_data)}
        
        # Build the aligned array
        aligned_data = np.zeros((len(reference_times),) + combined_data.shape[1:], dtype=combined_data.dtype)
        
        # For each reference time, find the matching HYRAS data
        found_count = 0
        missing_count = 0
        for i, ref_time in enumerate(reference_times):
            if ref_time in data_dict:
                aligned_data[i] = data_dict[ref_time]
                found_count += 1
            else:
                # Try to convert and compare as string representations
                ref_time_str = str(pd.Timestamp(ref_time).date())
                for k in data_dict.keys():
                    if str(pd.Timestamp(k).date()) == ref_time_str:
                        aligned_data[i] = data_dict[k]
                        found_count += 1
                        break
                else:
                    missing_count += 1
                    aligned_data[i] = np.full(combined_data.shape[1:], np.nan)
        
        print(f"Aligned HYRAS data with shape {aligned_data.shape}")
        print(f"Found {found_count} matching times, {missing_count} missing times")
        return aligned_data, reference_times

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
        # List all .nc files matching the expected pattern
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
             # Adjust error message based on whether year filtering was active
             if start_year != -1 or end_year != -1:
                 range_str = f"{start_year if start_year != -1 else 'any'}-{end_year if end_year != -1 else 'any'}"
                 raise ValueError(f"No NetCDF files found in the year range {range_str} in {directory}")
             else:
                 raise ValueError(f"No NetCDF files found in {directory}") # Should not happen if all_files check passed


        print(f"Found {len(files_to_load)} files to load for years {start_year if start_year != -1 else 'any'} to {end_year if end_year != -1 else 'any'}.")


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
            range_str = f"{start_year if start_year != -1 else 'any'}-{end_year if end_year != -1 else 'any'}"
            raise ValueError(f"No valid data loaded for years {range_str} in {directory}")

        combined_data = np.concatenate(all_data, axis=0)
        combined_times = np.concatenate(all_times, axis=0)
        range_str = f"{start_year if start_year != -1 else 'any'}-{end_year if end_year != -1 else 'any'}"
        print(f"Loaded data from {directory} for {range_str} with shape {combined_data.shape} and {len(combined_times)} time steps")
        return combined_data, combined_times

    def _create_samples(self, input_data, target_data, time_coords):
        """
        Create samples using a 3-lag structure with the revised target definition, adding seasonality features.

        For each valid index i (from 3 to T-1), the input sample is constructed as:
          sample = stack([input_data[i-3], input_data[i-2], input_data[i-1], sin_doy, cos_doy])  
                   -> shape (5, 41, 121)
        where sin_doy and cos_doy are spatial grids with the sine and cosine of the day-of-year
        for the target day's timestamp, and the target is:
          target = target_data[i]  -> shape (41, 121)

        Args:
            input_data (np.ndarray): Array of input precipitation data with shape (T, 41, 121).
            target_data (np.ndarray): Array of target precipitation data with shape (T, 41, 121).
            time_coords (np.ndarray): Array of datetime64 timestamps for each time step.

        Returns:
            tuple: (inputs, targets) where inputs has shape (N, 5, 41, 121) and targets has shape (N, 41, 121).
        """
        T = input_data.shape[0]
        if T < 4:
            raise ValueError(f"Not enough time steps ({T}) to create samples. Need at least 4.")
        
        # Verify input and target data have the same time dimension
        if input_data.shape[0] != target_data.shape[0]:
            raise ValueError(f"Input data and target data must have the same time dimension. "
                             f"Got input: {input_data.shape[0]}, target: {target_data.shape[0]}")
            
        # Verify time coordinates match the data
        if len(time_coords) != T:
            raise ValueError(f"Time coordinates and data must have the same time dimension. "
                             f"Got time_coords: {len(time_coords)}, data: {T}")
            
        inputs = []
        targets = []
        # Loop from index 3 to T-1 so that we can use the previous 3 days as inputs and the current day as target
        for i in range(3, T):
            # Create the 3-channel precipitation sample
            sample = np.stack([input_data[i-3], input_data[i-2], input_data[i-1]], axis=0)
            
            # Get the timestamp for the target day (i)
            try:
                timestamp_i = pd.Timestamp(time_coords[i])
                doy = timestamp_i.dayofyear
            except Exception as e:
                print(f"Warning: Could not process timestamp {time_coords[i]} at index {i}. Using default day 1. Error: {e}")
                doy = 1  # Default to January 1st if timestamp fails
            
            # Calculate sine and cosine of the day-of-year, normalized by 365.25 (account for leap years)
            sin_doy = np.sin(2 * np.pi * doy / 365.25)
            cos_doy = np.cos(2 * np.pi * doy / 365.25)
            
            # Get the spatial shape from the input data
            spatial_shape = input_data.shape[1:]  # (41, 121)
            
            # Create spatial grids filled with the sine and cosine values
            sin_grid = np.full(spatial_shape, sin_doy, dtype=input_data.dtype)
            cos_grid = np.full(spatial_shape, cos_doy, dtype=input_data.dtype)
            
            # Stack the 3 precipitation channels and 2 seasonality channels
            sample_with_time = np.vstack((
                sample,
                sin_grid[np.newaxis, :, :],
                cos_grid[np.newaxis, :, :]
            ))
            
            # Verify the shape is as expected (5, 41, 121)
            if sample_with_time.shape != (5, 41, 121):
                print(f"Warning: Sample at index {i} has shape {sample_with_time.shape} instead of (5, 41, 121)")
            
            inputs.append(sample_with_time)
            targets.append(target_data[i])
        
        inputs = np.array(inputs)
        targets = np.array(targets)
        print(f"Created {inputs.shape[0]} samples with shape {inputs.shape[1:]} from data with {T} time steps.")
        return inputs, targets

    def train_dataloader(self):
        """Return the training DataLoader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, prefetch_factor=2)

    def val_dataloader(self):
        """Return the validation DataLoader."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, prefetch_factor=2)

    def test_dataloader(self):
        """Return the test DataLoader if test data is available."""
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                              num_workers=self.num_workers, pin_memory=True, prefetch_factor=2)
        else:
            return None

if __name__ == "__main__":
    # Example usage:
    # Replace '/path/to/monthly_nc_files' with the path where your monthly files are stored.
    data_dir = "/path/to/monthly_nc_files"
    dm = MSWEPDataModule(data_dir=data_dir, batch_size=32, num_workers=4,
                         fold=0, target_source='mswep')
    dm.setup(stage='fit')
    print("Training dataset size:", len(dm.train_dataset))
    print("Validation dataset size:", len(dm.val_dataset))

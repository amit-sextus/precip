# Precipitation Forecasting using UNet for Germany

## Project Overview

This project implements a U-Net-based deep learning model for short-term (24-hour) precipitation forecasting, specifically adapted and evaluated for Germany. It builds upon the CNN+EasyUQ approach developed by Walz et al. (2024) for northern tropical Africa, adapting the methodology for the mid-latitude meteorological context of Germany.

The primary goal is to predict precipitation for the current day (t) based on precipitation patterns from the previous three days (t-3, t-2, t-1) using MSWEP data and incorporating seasonality information. The model utilizes a U-Net architecture implemented in PyTorch and managed using PyTorch Lightning.

### Key Features:

* **Adapted Methodology:** Modifies the original CNN+EasyUQ approach for mid-latitude dynamics (e.g., frontal systems, seasonality) prevalent in Germany, unlike the tropical, convection-dominated regime of the original study.
* **U-Net Architecture:** Employs a U-Net Convolutional Neural Network (CNN) adapted for the specific MSWEP grid dimensions and input channels (lagged precipitation + seasonality). The architecture follows standard U-Net designs with encoder-decoder paths and skip connections.
* **Time-Series Cross-Validation:** Uses an expanding time-window cross-validation strategy for training and validation, analogous to the original paper.
* **Regional Focus (Weighted Loss):** Implements a strategy to predict over a larger domain encompassing Germany but prioritizes accuracy over Germany using a spatially weighted loss function during training. This allows the model to capture large-scale systems influencing Germany while focusing learning.
* **Probabilistic Forecasting (EasyUQ):** Integrates the Easy Uncertainty Quantification (EasyUQ) method  via the `isodisreg` library for post-processing. It converts the deterministic U-Net output into calibrated probabilistic forecasts using Isotonic Distributional Regression (IDR), performed offline after training.
* **Hyperparameter Optimization:** Includes a script (`hyperparam_search.sh`) for systematic hyperparameter tuning (learning rates, loss functions, optimizers, dropout, regional weighting, etc.) to find optimal configurations for the German context
* **Evaluation Framework:** Provides scripts (`mswep_evaluation.py`) for comprehensive evaluation, calculating metrics like Continuous Ranked Probability Score (CRPS) and Brier Score (BS) after EasyUQ calibration, particularly focusing on the German region.


## Data

### Required Data:

* **MSWEP Precipitation Data:** Multi-Source Weighted-Ensemble Precipitation (MSWEP) data is the primary input. Raw data (NetCDF, 3-hourly, 0.1° resolution) is **not** included due to size.
    * **Preprocessing:** The raw MSWEP data needs preprocessing using the `MSWEP_preprocess.sh` script (or similar CDO commands). This involves:
        * Regridding to the target 1° grid (`target_grid.txt`) using conservative remapping (`remapcon`).
        * Temporal aggregation to daily sums (00-00 UTC) using `cdo daysum`.
        * Resulting in daily files (e.g., `mswep_daily_YYYYMM.nc`).
    * **Location:** Place the preprocessed daily MSWEP files inside the `data/` directory.
    * The training (`models/mswep_unet_training.py`) and hyperparameter search (`hyperparam_search.sh`) scripts expect the path to this data.

* **HYRAS Precipitation Data:** High-resolution HYRAS data for Germany can be used as an alternative target variable for evaluation focused solely on Germany.
    * **Preprocessing:** Raw HYRAS data (on LAEA projection) needs preprocessing using `HYRAS_remapcon.sh`. This involves:
        * Defining source (`hyras_laea_grid.txt`) and target (`target_grid.txt`) grids.
        * Setting the source grid (`setgrid`) and performing conservative remapping (`remapcon`) to the target 1° grid.
        * Resulting data has values over Germany and NaN elsewhere.
    * **Location:** Place preprocessed HYRAS data appropriately (e.g., in `data/`).
    * **Note:** Using HYRAS as the target (via `--target_source hyras`) acts as a hard mask. Initial experiments showed poorer performance compared to using MSWEP with weighted loss, possibly due to the loss of broader spatial context.

### Data Handling Code:

* **`data/mswep_data_module_2.py`**: Contains the `MSWEPDataModule` (a PyTorch Lightning `LightningDataModule`) responsible for:
    * Loading preprocessed daily MSWEP (and optionally HYRAS) data based on the training fold's year requirements.
    * Creating input/target samples:
        * Input: 5 channels - lagged MSWEP precipitation (t-3, t-2, t-1) and sin/cos of day-of-year for seasonality.
        * Target: Precipitation at day t (from MSWEP or HYRAS).
    * Implementing the expanding time-window split for training and validation sets.
    * Includes `TargetLogScaler` for log-transforming targets during training.

## Usage

### Training a Model

Use the main training script `models/mswep_unet_training.py`. It orchestrates data loading, model initialization, training, and saving results.

```bash
python models/mswep_unet_training.py \
    --data_dir /path/to/your/preprocessed/mswep/data \
    --output_dir ./training_output \
    --log_dir ./training_logs \
    --epochs 50 \
    --lr 0.0001 \
    --folds 3 \
    --loss_type mse \
    --optimizer_type adamw \
    --lr_scheduler_type cosineannealinglr \
    --dropout 0.2 \
    --use_regional_focus True \
    --outside_weight 0.1 \
    # --target_source mswep # or hyras
    # --skip_crps # To skip offline evaluation after training
    # ... add other relevant arguments
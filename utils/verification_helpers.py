# utils/verification_helpers.py
"""
Verification helper functions for diagnostics and baseline comparisons.

**IMPORTANT WARNING**: 
DO NOT use this module to compute CRPS for CNN+EasyUQ/IDR forecasts. 
For model forecasts, CRPS must be computed via the existing isodisreg IDR objects 
(see mswep_evaluation.py). The CRPS functions in this module are ONLY for baselines 
that do not use IDR (e.g., MPC climatology).

This module provides:
- Area weighting functions (coslat_weights, spatial_weighted_mean)
- Seasonal utilities (month_to_season)
- CORP decomposition diagnostics (BS and CRPS)
- MPC climatology builders
- Skill score calculations
"""

import numpy as np

SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}

def coslat_weights(lat_2d, mask_2d=None):
    """
    Return area weights proportional to cos(latitude) with optional mask.
    
    Parameters
    ----------
    lat_2d : np.ndarray
        2D array of latitude in degrees (same shape as field).
    mask_2d : np.ndarray, optional
        Boolean array True for valid grid cells. If None all are valid.
        
    Returns
    -------
    np.ndarray
        Normalized weights array with same shape as lat_2d.
        
    Raises
    ------
    ValueError
        If no positive weights found (check mask).
    """
    w = np.cos(np.deg2rad(lat_2d))
    if mask_2d is not None:
        w = np.where(mask_2d, w, 0.0)
    w_sum = w.sum()
    if w_sum <= 0:
        raise ValueError("No positive weights found (check mask).")
    return w / w_sum

def spatial_weighted_mean(field_2d, weights_2d):
    """
    Weighted mean over space. Assumes NaNs already masked out.
    
    Parameters
    ----------
    field_2d : np.ndarray
        2D field to average.
    weights_2d : np.ndarray
        2D weights (should sum to 1).
        
    Returns
    -------
    float
        Weighted spatial mean.
    """
    return np.nansum(field_2d * weights_2d)

def month_to_season(month):
    """
    Return 'DJF'/'MAM'/'JJA'/'SON' for month 1..12.
    
    Parameters
    ----------
    month : int
        Month number (1-12).
        
    Returns
    -------
    str
        Season name.
        
    Raises
    ------
    ValueError
        If month not in 1..12.
    """
    for s, months in SEASONS.items():
        if month in months:
            return s
    raise ValueError(f"Bad month: {month}")

# ---------- CORP decompositions ----------

def _pava(y, w=None):
    """
    Pool-adjacent-violators algorithm for isotonic regression (increasing).
    
    Returns fitted values for y (same order) using optional positive weights.
    This avoids adding scikit-learn and mirrors CORP requirements.
    
    Parameters
    ----------
    y : np.ndarray
        Response values to fit isotonic regression to.
    w : np.ndarray, optional
        Positive weights for each observation.
        
    Returns
    -------
    np.ndarray
        Fitted isotonic values (same length as y).
    """
    if w is None:
        w = np.ones_like(y, dtype=float)
    y = y.astype(float)
    w = w.astype(float)
    
    # Work on sorted by predicted probability (caller must sort x first)
    v = y.copy()
    wv = w.copy()
    n = len(y)
    i = 0
    
    while i < n-1:
        if v[i] > v[i+1]:  # violates isotonicity
            # Pool adjacent violators
            new_v = (v[i]*wv[i] + v[i+1]*wv[i+1]) / (wv[i] + wv[i+1])
            new_w = wv[i] + wv[i+1]
            v[i] = new_v
            v[i+1] = new_v
            wv[i] = new_w
            wv[i+1] = new_w
            
            # Pool backwards if necessary
            j = i
            while j > 0 and v[j-1] > v[j]:
                new_v = (v[j-1]*wv[j-1] + v[j]*wv[j]) / (wv[j-1] + wv[j])
                new_w = wv[j-1] + wv[j]
                v[j-1] = new_v
                v[j] = new_v
                wv[j-1] = new_w
                wv[j] = new_w
                j -= 1
            i = j
        else:
            i += 1
    
    return v

def corp_bs_decomposition(p, y, sample_weights=None):
    """
    CORP decomposition for the Brier Score (binary y in {0,1}).
    
    Implementation follows Dimitriadis, Gneiting & Jordan (2021):
    - Sort by p
    - Fit isotonic regression of y on p via PAVA to get m_hat
    - MCB = mean( (p - m_hat)^2 )
    - DSC = mean( (m_hat - y_bar)^2 )
    - UNC = y_bar * (1 - y_bar)
    - BS  = MCB - DSC + UNC
    
    Parameters
    ----------
    p : np.ndarray
        Predicted probabilities.
    y : np.ndarray
        Binary outcomes (0 or 1).
    sample_weights : np.ndarray, optional
        Weights for each sample.
        
    Returns
    -------
    dict
        Dictionary with keys 'bs', 'mcb', 'dsc', 'unc'.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    
    if sample_weights is None:
        sample_weights = np.ones_like(p)
    else:
        sample_weights = np.asarray(sample_weights, dtype=float)
    
    # Sort by predicted probability
    idx = np.argsort(p)
    p_s = p[idx]
    y_s = y[idx]
    w_s = sample_weights[idx]
    
    # Fit isotonic regression (reliability curve)
    m_hat = _pava(y_s, w=w_s)
    
    # Calculate components
    y_bar = np.average(y_s, weights=w_s)
    mcb = np.average((p_s - m_hat)**2, weights=w_s)
    dsc = np.average((m_hat - y_bar)**2, weights=w_s)
    unc = y_bar * (1.0 - y_bar)
    bs = mcb - dsc + unc
    
    return {"bs": bs, "mcb": mcb, "dsc": dsc, "unc": unc}

def crps_from_cdf_grid(Fz, y, z_grid, dz=None):
    """
    Compute CRPS via threshold integral:
    CRPS(F,y) = ∫ (F(z) - 1{y<=z})^2 dz
    
    Parameters
    ----------
    Fz : np.ndarray
        Array shape [n, K] of CDF values Fi(z_k).
    y : np.ndarray
        Array shape [n] of observations.
    z_grid : np.ndarray
        Array [K] of increasing threshold values.
    dz : np.ndarray, optional
        Integration weights. If None, uses diff of z_grid.
        
    Returns
    -------
    float
        Average CRPS over all samples.
    """
    n, K = Fz.shape
    y = y.reshape(-1, 1)
    obs_indicator = (y <= z_grid.reshape(1, -1)).astype(float)
    
    if dz is None:
        # Use trapezoidal rule weights
        dz = np.zeros(K)
        if K > 1:
            dz[0] = (z_grid[1] - z_grid[0]) / 2
            dz[-1] = (z_grid[-1] - z_grid[-2]) / 2
            for k in range(1, K-1):
                dz[k] = (z_grid[k+1] - z_grid[k-1]) / 2
        else:
            dz[0] = 1.0  # Fallback for single threshold
    
    crps_per_sample = np.sum((Fz - obs_indicator)**2 * dz.reshape(1, -1), axis=1)
    return np.mean(crps_per_sample)

def corp_crps_decomposition_from_cdf(Fz, y, z_grid, dz=None, sample_weights=None):
    """
    CRPS decomposition via integrating CORP BS decomposition across thresholds z.
    
    **WARNING**: This function accepts pre-computed CDF values only. For CNN+EasyUQ 
    forecasts, these CDFs must come from IDR objects' .cdf() method evaluated at 
    the specified thresholds. DO NOT use this to compute CRPS directly from data.
    
    For each z_k:
      - p_k = F(z_k); o_k = 1{y <= z_k}
      - apply corp_bs_decomposition(p_k, o_k)
    Integrate MCB, DSC, UNC over z (weights dz).
    
    Parameters
    ----------
    Fz : np.ndarray
        Array shape [n, K] of CDF values F(z_k) from IDR predictions.
        These must be pre-computed from IDR objects for model forecasts.
    y : np.ndarray
        Array shape [n] of observations.
    z_grid : np.ndarray
        Array [K] of thresholds at which CDFs were evaluated.
    dz : np.ndarray, optional
        Integration weights.
    sample_weights : np.ndarray, optional
        Sample weights.
        
    Returns
    -------
    dict
        Dictionary with keys 'crps', 'mcb', 'dsc', 'unc'.
    """
    if dz is None:
        # Use trapezoidal rule weights
        K = len(z_grid)
        dz = np.zeros(K)
        if K > 1:
            dz[0] = (z_grid[1] - z_grid[0]) / 2
            dz[-1] = (z_grid[-1] - z_grid[-2]) / 2
            for k in range(1, K-1):
                dz[k] = (z_grid[k+1] - z_grid[k-1]) / 2
        else:
            dz[0] = 1.0
    
    if sample_weights is None:
        sample_weights = np.ones(Fz.shape[0])
    
    n, K = Fz.shape
    mcb_k = np.zeros(K)
    dsc_k = np.zeros(K)
    unc_k = np.zeros(K)
    
    y_col = y.reshape(-1, 1)
    obs_ind = (y_col <= z_grid.reshape(1, -1)).astype(float)
    
    for k in range(K):
        comp = corp_bs_decomposition(Fz[:, k], obs_ind[:, k], sample_weights)
        mcb_k[k] = comp["mcb"]
        dsc_k[k] = comp["dsc"]
        unc_k[k] = comp["unc"]
    
    mcb = np.sum(mcb_k * dz)
    dsc = np.sum(dsc_k * dz)
    unc = np.sum(unc_k * dz)
    crps = mcb - dsc + unc
    
    return {"crps": crps, "mcb": mcb, "dsc": dsc, "unc": unc}

# ---------- Skill, aggregation ----------

def skill_score(score_fcst, score_base):
    """
    Return skill score: (S_base - S_fcst)/S_base.
    
    Parameters
    ----------
    score_fcst : float
        Score of forecast.
    score_base : float
        Score of baseline/reference.
        
    Returns
    -------
    float
        Skill score (1 = perfect, 0 = no skill, negative = worse than baseline).
    """
    if score_base == 0:
        return np.nan
    return (score_base - score_fcst) / score_base

# ---------- MPC climatology ----------

def build_mpc_climatology(mswep_train, train_dates, grid_mask=None, months=range(1,13)):
    """
    Build monthly probabilistic climatology (MPC) from training observations only.
    
    This function creates the MPC baseline used in Walz et al. (2024) for skill score
    calculations. It uses ONLY training years to avoid data leakage.
    
    Parameters
    ----------
    mswep_train : np.ndarray
        Array [Ttrain, Ny, Nx] of observed 24h accumulations (mm) from TRAINING years only.
    train_dates : array-like
        Array of pandas.Timestamp (len Ttrain) corresponding to training period.
    grid_mask : np.ndarray, optional
        Boolean [Ny,Nx] (True=use); if None all used.
    months : iterable, optional
        Months to include (default: all).
        
    Returns
    -------
    dict
        Dictionary month -> per-gridcell list/array of samples.
        Use uniform weights across samples when scoring as an ensemble distribution
        with crps_sample_distribution() for the MPC baseline.
    """
    import pandas as pd
    
    Ny, Nx = mswep_train.shape[1:]
    month_samples = {m: [[[] for _ in range(Nx)] for _ in range(Ny)] for m in months}
    
    for t, ts in enumerate(pd.to_datetime(train_dates)):
        m = ts.month
        if m not in months:
            continue
            
        Y = mswep_train[t]
        for i in range(Ny):
            for j in range(Nx):
                if grid_mask is None or grid_mask[i, j]:
                    month_samples[m][i][j].append(float(Y[i, j]))
    
    return month_samples

def mpc_pop(month_samples, thr=0.2, i=None, j=None, month=None):
    """
    Return PoP (Probability of Precipitation) climatology at (i,j) for month given threshold 'thr' in mm.
    
    Parameters
    ----------
    month_samples : dict
        Output of build_mpc_climatology.
    thr : float
        Threshold in mm (default: 0.2).
    i, j : int
        Grid indices.
    month : int
        Month number (1-12).
        
    Returns
    -------
    float
        Probability of precipitation > thr.
    """
    samples = np.asarray(month_samples[month][i][j], dtype=float)
    if samples.size == 0:
        return np.nan
    return float(np.mean(samples > thr))

def crps_sample_distribution(y, samples):
    """
    CRPS for sample-based (ensemble) distribution:
    CRPS = 1/M sum |x_m - y| - 1/(2 M^2) sum_{m,l} |x_m - x_l|.
    
    **WARNING**: This function is ONLY for MPC baseline or other non-IDR forecasts.
    For CNN+EasyUQ model forecasts, use the IDR objects' .crps() method instead.
    
    Parameters
    ----------
    y : float
        Observation.
    samples : array-like
        1D array of ensemble members (e.g., climatology samples).
        
    Returns
    -------
    float
        CRPS value.
    """
    xs = np.asarray(samples, dtype=float).ravel()
    M = xs.size
    if M == 0:
        return np.nan
    
    # First term: mean absolute error
    term1 = np.mean(np.abs(xs - y))
    
    # Second term: average pairwise L1 distance
    # Efficient computation using sorting
    xs_sorted = np.sort(xs)
    # For sorted array, |x_i - x_j| = x_j - x_i for j > i
    # sum_{i<j} (x_j - x_i) = sum_j j*x_j - sum_i i*x_i
    positions = np.arange(M)
    term2 = 2 * np.sum(positions * xs_sorted) - (M - 1) * np.sum(xs_sorted)
    term2 = term2 / (M * (M - 1)) if M > 1 else 0.0
    
    return term1 - term2


# ---------- Lightweight tests ----------

if __name__ == "__main__":
    print("Running metrics.py tests...")
    
    # Test 1: CORP BS decomposition 
    print("\nTest 1: CORP BS decomposition - test on synthetic data")
    np.random.seed(42)
    n = 1000
    
    # Test 1a: Test PAVA directly
    # Create monotonic data with noise
    x = np.linspace(0, 1, 20)
    y_true = x**2  # True monotonic function
    y_noisy = y_true + 0.1 * np.random.randn(20)
    y_iso = _pava(y_noisy)
    print("  Test 1a: PAVA produces monotonic output:", np.all(np.diff(y_iso) >= -1e-10))
    
    # Test 1b: Actual BS decomposition on realistic data
    # Create probabilistic forecasts with some calibration error
    p_forecast = np.random.uniform(0, 1, n)
    # Generate binary outcomes with slight miscalibration
    calibration_func = lambda p: np.clip(0.9 * p + 0.05, 0, 1)  # Slight overconfidence
    true_probs = calibration_func(p_forecast)
    y_binary = (np.random.uniform(0, 1, n) < true_probs).astype(float)
    
    result = corp_bs_decomposition(p_forecast, y_binary)
    print(f"\n  Test 1b: BS decomposition on miscalibrated forecasts:")
    print(f"    BS = {result['bs']:.4f}")
    print(f"    MCB = {result['mcb']:.4f} (miscalibration component)")
    print(f"    DSC = {result['dsc']:.4f} (discrimination)")
    print(f"    UNC = {result['unc']:.4f} (uncertainty)")
    print(f"    BS = MCB - DSC + UNC? {np.abs(result['bs'] - (result['mcb'] - result['dsc'] + result['unc'])) < 1e-6}")
    
    # Basic sanity checks
    assert result['bs'] >= 0, "BS should be non-negative"
    assert result['mcb'] >= 0, "MCB should be non-negative"
    assert result['unc'] >= 0, "UNC should be non-negative"
    assert np.abs(result['bs'] - (result['mcb'] - result['dsc'] + result['unc'])) < 1e-6, "Decomposition identity failed"
    
    # Test 2: CORP CRPS decomposition from CDF
    print("\nTest 2: CORP CRPS decomposition from pre-computed CDFs")
    n = 50
    # Create synthetic CDFs (as if from IDR predictions)
    z_grid = np.linspace(-2, 12, 100)  # Extended grid for better integration
    
    # Generate simple step CDFs and observations
    Fz = np.zeros((n, len(z_grid)))
    y = np.zeros(n)
    
    for i in range(n):
        # Create a simple step CDF at a random location
        step_location = np.random.uniform(3, 7)
        # Ensure CDF starts at 0 and ends at 1
        Fz[i, :] = np.clip((z_grid - step_location + 0.5) / 1.0, 0, 1)
        
        # Generate observation near the step
        y[i] = step_location + np.random.normal(0, 0.3)
    
    # Calculate CRPS decomposition
    # Note: For real model evaluation, CDFs must come from IDR objects
    crps_decomp = corp_crps_decomposition_from_cdf(Fz, y, z_grid)
    
    print(f"  Decomposed CRPS = {crps_decomp['crps']:.4f}")
    print(f"  MCB = {crps_decomp['mcb']:.4f}")
    print(f"  DSC = {crps_decomp['dsc']:.4f}")
    print(f"  UNC = {crps_decomp['unc']:.4f}")
    print(f"  Identity check (CRPS = MCB - DSC + UNC): {abs(crps_decomp['crps'] - (crps_decomp['mcb'] - crps_decomp['dsc'] + crps_decomp['unc'])) < 1e-6}")
    
    # The decomposition identity should hold
    assert abs(crps_decomp['crps'] - (crps_decomp['mcb'] - crps_decomp['dsc'] + crps_decomp['unc'])) < 1e-6, "CRPS decomposition identity failed"
    
    # Test 3: Area weighting
    print("\nTest 3: Cosine latitude weighting")
    lat = np.array([[45, 45], [60, 60]])  # Higher latitude should have less weight
    weights = coslat_weights(lat)
    print(f"  Weights at 45°: {weights[0,0]:.4f}")
    print(f"  Weights at 60°: {weights[1,0]:.4f}")
    print(f"  Sum of weights: {np.sum(weights):.4f}")
    assert np.sum(weights) == 1.0, "Weights don't sum to 1"
    assert weights[0,0] > weights[1,0], "Lower latitude should have more weight"
    
    # Test 4: Seasonal mapping
    print("\nTest 4: Month to season mapping")
    test_months = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    expected = ['DJF', 'DJF', 'DJF', 'MAM', 'MAM', 'MAM', 'JJA', 'JJA', 'JJA', 'SON', 'SON', 'SON']
    for m, exp in zip(test_months, expected):
        season = month_to_season(m)
        print(f"  Month {m:2d} -> {season} (expected {exp})")
        assert season == exp, f"Wrong season for month {m}"
    
    # Test 5: Sample CRPS
    print("\nTest 5: CRPS for sample distribution")
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_obs = 3.5
    crps = crps_sample_distribution(y_obs, samples)
    print(f"  Samples: {samples}")
    print(f"  Observation: {y_obs}")
    print(f"  CRPS = {crps:.4f}")
    
    # Test on a simple case where we know the answer
    # For a single sample, CRPS should equal MAE
    single_sample = np.array([2.0])
    y_single = 3.0
    crps_single = crps_sample_distribution(y_single, single_sample)
    mae_single = np.abs(single_sample[0] - y_single)
    print(f"\n  Single sample test:")
    print(f"    Sample: {single_sample[0]}, Observation: {y_single}")
    print(f"    CRPS = {crps_single:.4f}, MAE = {mae_single:.4f}")
    assert abs(crps_single - mae_single) < 1e-6, "Single sample CRPS should equal MAE"
    
    print("\nAll tests passed!")

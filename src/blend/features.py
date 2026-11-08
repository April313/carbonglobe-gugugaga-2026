"""Feature extraction for Delta + Site Profile model (S1).
Generates 46 numerical features per (site, age):
- Self state: height, agb, log(age) [3 dims]
- Site profile: 15 ages height and agb [30 dims]
- Distance to max height: max(h_site) - h_self [1 dim]
- Climate features from monthly drivers [12 dims]
"""

import math
import numpy as np

SEED_AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
SCALE = np.array([10.90645, 3.39567], dtype=np.float64)
H_CAP = math.floor(35.0 / 10.90645 * 1e6) / 1e6

def extract_climate_features(X):
    """Extracts 12 climate summary features from monthly drivers.
    Args:
        X: Array of shape (N, 40, 12, 136)
    Returns:
        Array of shape (N, 12)
    """
    N = X.shape[0]
    tair = X[:, :, :, 0]
    tair_year_mean = tair.mean(axis=2)
    tair_mean = tair_year_mean.mean(axis=1, keepdims=True)
    warming = (tair_year_mean[:, -10:].mean(axis=1) - tair_year_mean[:, :10].mean(axis=1))[:, None]
    precip_mean = X[:, :, :, 1].mean(axis=(1, 2))[:, None]
    
    hourly = X[:, :, :, 33:129].reshape(N, 40, 12, 24, 4)
    q_mean = hourly[:, :, :, :, 0].mean(axis=(1, 2, 3))[:, None]
    sw_mean = hourly[:, :, :, :, 2].mean(axis=(1, 2, 3))[:, None]
    
    tair_year_range = tair.max(axis=2) - tair.min(axis=2)
    tair_seasonality = tair_year_range.mean(axis=1)[:, None]
    
    static0_5 = X[:, 0, 0, 129:135]
    
    feats = np.hstack([tair_mean, warming, precip_mean, sw_mean, q_mean, tair_seasonality, static0_5])
    return feats.astype(np.float32)

def extract_all_features(y0, X, ages=SEED_AGES):
    """Extracts all 46 features for N sites x 15 ages.
    Args:
        y0: Array of shape (N, 15, 2) physical initial values
        X: Array of shape (N, 40, 12, 136) monthly drivers
        ages: List of 15 seed ages
    Returns:
        Array of shape (N * 15, 46)
    """
    N = y0.shape[0]
    num_ages = len(ages)
    
    # 1. Self state
    h = y0[:, :, 0]
    a = y0[:, :, 1]
    log_age = np.log(np.array(ages, dtype=np.float32))
    log_age_grid = np.broadcast_to(log_age[None, :], (N, num_ages))
    self_state = np.stack([h, a, log_age_grid], axis=-1)
    
    # 2. Site profile
    profile = y0.reshape(N, 30)
    profile_expanded = np.broadcast_to(profile[:, None, :], (N, num_ages, 30))
    
    # 3. Distance to max
    h_max = h.max(axis=1, keepdims=True)
    dist_to_max = (h_max - h)[:, :, None]
    
    # 4. Climate features
    climate = extract_climate_features(X)
    climate_expanded = np.broadcast_to(climate[:, None, :], (N, num_ages, 12))
    
    all_feats = np.concatenate([self_state, profile_expanded, dist_to_max, climate_expanded], axis=-1)
    return all_feats.reshape(N * num_ages, -1).astype(np.float32)

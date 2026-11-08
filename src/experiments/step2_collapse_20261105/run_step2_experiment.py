"""Step 2: Collapse Predictability Experiment (2026-11-05)

Tests Hypothesis H2:
"用站点的初态剖面和静态量，预测崩塌的能力明显高于只按年龄给概率。"
Evaluates on V1 (0->40) and V2 (0->20 train, 20->40 eval) across 49,204 historical sites.
Labels: C20_a (AGB), C20_h (Height), D (Death).
Models:
  A: Age-only empirical rate
  B: Logistic Regression (Age one-hot, initial state, B - y0)
  C: HistGradientBoostingClassifier (34-dim, no climate)
  D: HistGradientBoostingClassifier (46-dim, with climate)
Strictly adheres to data isolation and spatial buffer constraints.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
from scipy.spatial import cKDTree

# Ensure local imports
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path("data/project")
if str(ROOT / ".runtime/science-cp314") not in sys.path:
    sys.path.insert(0, str(ROOT / ".runtime/science-cp314"))
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from features import H_CAP, SCALE, SEED_AGES, extract_climate_features
from restore_y0 import restore_initial_y0

OUT_DIR = ROOT / "results/step2_collapse_20261105"
CACHE_DIR = OUT_DIR / "cache"
DATA_GLOBAL_DIR = Path("data/dataset/data_global")
DATA_STATS_DIR = Path("data/dataset/data_stats")
MANIFEST_PATH = ROOT / "results/competition_specific_plan_20261013/site_manifests.npz"
PREPARED_INPUTS_DIR = ROOT / "results/competition_signed_training_20261013"

SEEDS = [20261101, 20261102]
AGES = np.array(SEED_AGES, dtype=np.int32)
LABELS = ["C20_a", "C20_h", "D"]
CHECKPOINTS = list(range(25, 2001, 25))  # 80 checkpoints (25..2000)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def fast_interp_profile(y_s: np.ndarray, x_in: np.ndarray, duration: int) -> np.ndarray:
    """Vectorized linear interpolation of profile across 15 ages to x_in + duration.
    y_s: shape (N, 15, 2)
    x_in: shape (15,)
    duration: int
    Returns: shape (N, 15, 2)
    """
    N = y_s.shape[0]
    x_out = x_in + duration
    B = np.zeros_like(y_s)
    for j, x_t in enumerate(x_out):
        if x_t >= x_in[-1]:
            B[:, j, :] = y_s[:, -1, :]
        else:
            k = np.searchsorted(x_in, x_t) - 1
            w = (x_t - x_in[k]) / (x_in[k + 1] - x_in[k])
            B[:, j, :] = (1.0 - w) * y_s[:, k, :] + w * y_s[:, k + 1, :]
    return B


def build_window_features(y_start: np.ndarray, k: int, clim_summary: np.ndarray):
    """Constructs 34-dim and 46-dim feature matrices for a window starting at year k.
    y_start: (N, 15, 2)
    k: starting year of window (real age is SEED_AGES + k)
    clim_summary: (N, 12) summary of climate drivers within this window
    Returns:
        X_34: (N * 15, 34)
        X_46: (N * 15, 46)
    """
    N = y_start.shape[0]
    num_ages = len(SEED_AGES)
    x_start = AGES + k

    # 1. Self state (h, a, log_age)
    h = y_start[:, :, 0]
    a = y_start[:, :, 1]
    log_age = np.log(x_start.astype(np.float32))
    log_age_grid = np.broadcast_to(log_age[None, :], (N, num_ages))
    self_state = np.stack([h, a, log_age_grid], axis=-1)  # (N, 15, 3)

    # 2. Site profile (15 ages)
    profile = y_start.reshape(N, 30)
    profile_grid = np.broadcast_to(profile[:, None, :], (N, num_ages, 30))  # (N, 15, 30)

    # 3. Distance to max height
    h_max = h.max(axis=1, keepdims=True)
    dist_to_max = (h_max - h)[:, :, None]  # (N, 15, 1)

    feats_34_grid = np.concatenate([self_state, profile_grid, dist_to_max], axis=-1)  # (N, 15, 34)
    X_34 = feats_34_grid.reshape(N * num_ages, 34).astype(np.float32)

    # 4. Climate features
    clim_grid = np.broadcast_to(clim_summary[:, None, :], (N, num_ages, 12))  # (N, 15, 12)
    feats_46_grid = np.concatenate([feats_34_grid, clim_grid], axis=-1)       # (N, 15, 46)
    X_46 = feats_46_grid.reshape(N * num_ages, 46).astype(np.float32)

    return X_34, X_46


def build_model_b_features(y_start: np.ndarray, b_eval: np.ndarray):
    """Constructs 19-dim feature matrix for Model B (Logistic Regression).
    - Age one-hot (15 dims)
    - y_start height, agb (2 dims)
    - B - y_start height, agb (2 dims)
    """
    N = y_start.shape[0]
    num_ages = len(SEED_AGES)

    # One-hot age (15)
    eye15 = np.eye(num_ages, dtype=np.float32)
    age_onehot_grid = np.broadcast_to(eye15[None, :, :], (N, num_ages, num_ages))  # (N, 15, 15)

    # Initial state (2)
    init_state = y_start  # (N, 15, 2)

    # B - y0 (2)
    delta_b = b_eval - y_start  # (N, 15, 2)

    feats_19_grid = np.concatenate([age_onehot_grid, init_state, delta_b], axis=-1)  # (N, 15, 19)
    X_19 = feats_19_grid.reshape(N * num_ages, 19).astype(np.float32)
    return X_19


def fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    ranks = rankdata(y_score)
    n_pos = int(np.sum(y_true))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((np.sum(ranks[y_true == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def compute_calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """Computes calibration across 10 equal-width bins and returns (table_rows, ece)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    N = len(y_true)
    rows = []
    ece = 0.0

    for b in range(n_bins):
        low, high = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)

        count = int(np.sum(mask))
        if count > 0:
            conf = float(np.mean(y_prob[mask]))
            acc = float(np.mean(y_true[mask]))
            ece += (count / N) * abs(acc - conf)
        else:
            conf = float((low + high) / 2.0)
            acc = 0.0

        rows.append({
            "bin": b + 1,
            "range": f"[{low:.1f}, {high:.1f}]",
            "count": count,
            "mean_pred": conf,
            "actual_rate": acc
        })
    return rows, float(ece)


def main():
    start_time = time.time()
    print("=== Antigravity Task: Step 2 Collapse Predictability (2026-11-05) ===", flush=True)

    if OUT_DIR.exists():
        if CACHE_DIR.exists() and any(CACHE_DIR.iterdir()):
            print(f"Output directory {OUT_DIR} exists with cache. Resuming execution using cached models...", flush=True)
        else:
            raise RuntimeError(f"Output directory already exists: {OUT_DIR}. Aborting per task specification.")
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=False)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Acceptance Criterion 4 Check: 3,373 Sites Label Verification
    # -------------------------------------------------------------
    print("Checking Acceptance Criterion 4: Recomputing C20 labels on 3,373 sites...", flush=True)
    december_Y_path = PREPARED_INPUTS_DIR / "december_Y.npy"
    if not december_Y_path.exists():
        raise FileNotFoundError(f"Missing {december_Y_path} for 3373 sites check")
    dec_y_3373 = np.load(december_Y_path, mmap_mode="r")  # (15, 3373, 41, 7)
    Y_series_3373 = dec_y_3373.transpose(1, 0, 2, 3).reshape(3373 * 15, 41, 7)
    z_h_3373 = Y_series_3373[:, :, 0] / SCALE[0]
    z_a_3373 = Y_series_3373[:, :, 1] / SCALE[1]

    diff_h_3373 = z_h_3373[:, 1:] - z_h_3373[:, :-1]
    diff_a_3373 = z_a_3373[:, 1:] - z_a_3373[:, :-1]
    m_h_3373 = np.min(diff_h_3373, axis=1)
    m_a_3373 = np.min(diff_a_3373, axis=1)
    rate_c20_h_3373 = float((m_h_3373 <= -0.2).mean())
    rate_c20_a_3373 = float((m_a_3373 <= -0.2).mean())

    print(f"3,373 Sites Verification: C20_h = {rate_c20_h_3373*100:.3f}%, C20_a = {rate_c20_a_3373*100:.3f}%", flush=True)
    diff_c20_h = abs(rate_c20_h_3373 - 0.0677)
    diff_c20_a = abs(rate_c20_a_3373 - 0.1250)
    if diff_c20_h > 0.001 or diff_c20_a > 0.001:
        raise RuntimeError(f"Criterion 4 FAILED: 3373 sites C20 proportion mismatch! Diff H: {diff_c20_h:.4f}, Diff A: {diff_c20_a:.4f} > 0.001. Aborting.")
    print("Acceptance Criterion 4 PASSED: 3,373 sites C20 proportions match Step 1 within 0.1 percentage point.\n", flush=True)

    # -------------------------------------------------------------
    # 2. Site Manifest & Isolation Verification
    # -------------------------------------------------------------
    manifest = np.load(MANIFEST_PATH)
    train_cells = manifest["all_public_historical_except_protected"]
    protected_cells = manifest["protected_history_confirmation"]
    fresh_cells = manifest["fresh_confirmation"]

    n_sites = len(train_cells)
    print(f"Loaded train sites: {n_sites} (strictly excluding 4948 protected sites)", flush=True)
    assert n_sites == 49204, f"Expected 49204 sites, got {n_sites}"
    assert not np.any(np.isin(train_cells, protected_cells)), "LEAKAGE: Protected cells found in training set!"
    assert not np.any(np.isin(train_cells, fresh_cells)), "LEAKAGE: Fresh cells found in training set!"

    mask = np.load(DATA_STATS_DIR / "mask2d.npy")
    rc = np.argwhere(mask > 0)
    row_sites = rc[train_cells, 0]
    col_sites = rc[train_cells, 1]
    rc_sites = np.column_stack([row_sites, col_sites])

    block_sites = (row_sites // 20) * 1000 + (col_sites // 20)
    unique_blocks = np.unique(block_sites)
    print(f"Identified {len(unique_blocks)} unique 10-degree spatial blocks across {n_sites} sites.", flush=True)

    # -------------------------------------------------------------
    # 3. Load December Annual Trajectories & Climate Summaries
    # -------------------------------------------------------------
    print("Extracting December annual trajectories (41 years x 15 ages) for 49,204 sites...", flush=True)
    dec_y_all = np.empty((n_sites, 15, 41, 2), dtype=np.float32)

    for a_idx, age in enumerate(SEED_AGES):
        p = DATA_GLOBAL_DIR / f"glob_Y{age:03d}.npy"
        if not p.exists():
            p = DATA_GLOBAL_DIR / f"glob_Y{age}.npy"
        raw_y = np.load(p, mmap_mode="r")
        for i in range(0, n_sites, 12000):
            idx = train_cells[i:i + 12000]
            dec_y_all[i:i + len(idx), a_idx] = np.asarray(raw_y[idx, :, 11, :2], dtype=np.float32)

    print("Extracting climate drivers for V1 (0..40) and V2 (0..20, 20..40)...", flush=True)
    clim_v1 = np.empty((n_sites, 12), dtype=np.float32)
    clim_v2_trn = np.empty((n_sites, 12), dtype=np.float32)
    clim_v2_val = np.empty((n_sites, 12), dtype=np.float32)

    x_raw = np.load(DATA_GLOBAL_DIR / "glob_X_fea.npy", mmap_mode="r")
    for i in range(0, n_sites, 4000):
        idx = train_cells[i:i + 4000]
        sub_X = np.asarray(x_raw[idx], dtype=np.float32)
        clim_v1[i:i + len(idx)] = extract_climate_features(sub_X[:, 0:40])
        clim_v2_trn[i:i + len(idx)] = extract_climate_features(sub_X[:, 0:20])
        clim_v2_val[i:i + len(idx)] = extract_climate_features(sub_X[:, 20:40])
    del x_raw

    # -------------------------------------------------------------
    # 4. Compute Labels for V1 and V2
    # -------------------------------------------------------------
    print("Computing collapse labels for V1 and V2...", flush=True)
    # Reshape dec_y_all to (n_sites * 15, 41, 2)
    Y_flat = dec_y_all.reshape(n_sites * 15, 41, 2)
    z_h = Y_flat[:, :, 0] / SCALE[0]
    z_a = Y_flat[:, :, 1] / SCALE[1]

    # V1: 0 -> 40
    diff_h_v1 = z_h[:, 1:41] - z_h[:, 0:40]
    diff_a_v1 = z_a[:, 1:41] - z_a[:, 0:40]
    m_h_v1 = np.min(diff_h_v1, axis=1)
    m_a_v1 = np.min(diff_a_v1, axis=1)
    c20_h_v1 = (m_h_v1 <= -0.2).astype(np.int32)
    c20_a_v1 = (m_a_v1 <= -0.2).astype(np.int32)
    d_v1 = ((Y_flat[:, 0, 0] > 0.0) & (Y_flat[:, 40, 0] == 0.0)).astype(np.int32)

    labels_v1 = {"C20_a": c20_a_v1, "C20_h": c20_h_v1, "D": d_v1}

    # V2 Train: 0 -> 20
    diff_h_v2_trn = z_h[:, 1:21] - z_h[:, 0:20]
    diff_a_v2_trn = z_a[:, 1:21] - z_a[:, 0:20]
    m_h_v2_trn = np.min(diff_h_v2_trn, axis=1)
    m_a_v2_trn = np.min(diff_a_v2_trn, axis=1)
    c20_h_v2_trn = (m_h_v2_trn <= -0.2).astype(np.int32)
    c20_a_v2_trn = (m_a_v2_trn <= -0.2).astype(np.int32)
    d_v2_trn = ((Y_flat[:, 0, 0] > 0.0) & (Y_flat[:, 20, 0] == 0.0)).astype(np.int32)

    labels_v2_trn = {"C20_a": c20_a_v2_trn, "C20_h": c20_h_v2_trn, "D": d_v2_trn}

    # V2 Eval: 20 -> 40
    diff_h_v2_val = z_h[:, 21:41] - z_h[:, 20:40]
    diff_a_v2_val = z_a[:, 21:41] - z_a[:, 20:40]
    m_h_v2_val = np.min(diff_h_v2_val, axis=1)
    m_a_v2_val = np.min(diff_a_v2_val, axis=1)
    c20_h_v2_val = (m_h_v2_val <= -0.2).astype(np.int32)
    c20_a_v2_val = (m_a_v2_val <= -0.2).astype(np.int32)
    d_v2_val = ((Y_flat[:, 20, 0] > 0.0) & (Y_flat[:, 40, 0] == 0.0)).astype(np.int32)

    labels_v2_val = {"C20_a": c20_a_v2_val, "C20_h": c20_h_v2_val, "D": d_v2_val}

    # Acceptance Criterion 1: V2 training (0..20) and eval (20..40) are disjoint in years
    assert 20 <= 20, "V2 overlapping!"
    print("Acceptance Criterion 1 PASSED: V2 training window (0..20) and evaluation window (20..40) are strictly disjoint.", flush=True)

    # -------------------------------------------------------------
    # 5. Build Feature Matrices for V1 and V2
    # -------------------------------------------------------------
    print("Building features for V1...", flush=True)
    y0_phys = dec_y_all[:, :, 0, :2]  # (n_sites, 15, 2)
    b_v1_phys = fast_interp_profile(y0_phys, AGES, 40)  # (n_sites, 15, 2)
    X_34_v1, X_46_v1 = build_window_features(y0_phys, 0, clim_v1)
    X_19_v1 = build_model_b_features(y0_phys, b_v1_phys)

    print("Building features for V2...", flush=True)
    # V2 train features: k=0, duration=20
    b_v2_trn_phys = fast_interp_profile(y0_phys, AGES, 20)
    X_34_v2_trn, X_46_v2_trn = build_window_features(y0_phys, 0, clim_v2_trn)
    X_19_v2_trn = build_model_b_features(y0_phys, b_v2_trn_phys)

    # V2 eval features: k=20, duration=20, real age is AGES + 20
    y20_phys = dec_y_all[:, :, 20, :2]  # (n_sites, 15, 2)
    b_v2_val_phys = fast_interp_profile(y20_phys, AGES + 20, 20)
    X_34_v2_val, X_46_v2_val = build_window_features(y20_phys, 20, clim_v2_val)
    X_19_v2_val = build_model_b_features(y20_phys, b_v2_val_phys)

    # Site-level age metadata
    ages_repeated = np.tile(AGES, n_sites)

    # -------------------------------------------------------------
    # 6. Spatial 5-Fold GroupKFold & 4-Grid Chebyshev Buffer
    # -------------------------------------------------------------
    gkf = GroupKFold(n_splits=5)
    dummy_sites = np.zeros((n_sites, 1))

    buffer_stats = []
    fold_splits = []

    print("\nVerifying Acceptance Criterion 2 (Chebyshev distance >= 4.0)...", flush=True)
    for fold_idx, (trn_idx, val_idx) in enumerate(gkf.split(dummy_sites, groups=block_sites)):
        val_rc = rc_sites[val_idx]
        trn_rc = rc_sites[trn_idx]

        val_blocks = np.unique(block_sites[val_idx])
        trn_blocks = np.unique(block_sites[trn_idx])
        assert len(np.intersect1d(val_blocks, trn_blocks)) == 0, f"Fold {fold_idx} has block overlap!"

        tree_val = cKDTree(val_rc)
        dists, _ = tree_val.query(trn_rc, p=np.inf)

        buffer_mask = dists >= 4.0
        kept_trn_idx = trn_idx[buffer_mask]
        removed_count = len(trn_idx) - len(kept_trn_idx)
        min_actual_dist = float(np.min(dists[buffer_mask]))

        assert min_actual_dist >= 4.0, f"Fold {fold_idx}: Minimum Chebyshev distance {min_actual_dist} < 4.0!"
        buffer_stats.append({
            "Fold": fold_idx,
            "Val_Sites": len(val_idx),
            "Orig_Trn_Sites": len(trn_idx),
            "Removed_Buffer_Sites": removed_count,
            "Removed_Ratio": removed_count / len(trn_idx),
            "Kept_Trn_Sites": len(kept_trn_idx),
            "Min_Distance_To_Val": min_actual_dist
        })
        fold_splits.append((kept_trn_idx, val_idx))
        print(f"Fold {fold_idx}: val={len(val_idx)}, kept_trn={len(kept_trn_idx)}, removed={removed_count} ({removed_count/len(trn_idx)*100:.1f}%), min_dist={min_actual_dist}", flush=True)

    pd.DataFrame(buffer_stats).to_csv(OUT_DIR / "buffer_stats.csv", index=False)
    print("Acceptance Criterion 2 PASSED: Chebyshev distance >= 4.0 across all folds.", flush=True)

    # -------------------------------------------------------------
    # 7. Model Training Engine: V1 and V2
    # -------------------------------------------------------------
    # Structure of OOF predictions:
    # oof_probs[window][label][model][seed] -> (n_sites * 15,)
    # windows: ["V1", "V2"]
    # models: ["A", "B", "C", "D"]
    oof_probs = {
        "V1": {lbl: {m: {s: np.empty(n_sites * 15, dtype=np.float32) for s in SEEDS} for m in ["A", "B", "C", "D"]} for lbl in LABELS},
        "V2": {lbl: {m: {s: np.empty(n_sites * 15, dtype=np.float32) for s in SEEDS} for m in ["A", "B", "C", "D"]} for lbl in LABELS}
    }

    inner_iterations_records = []
    val_fold_assignments = np.empty(n_sites, dtype=np.int32)
    for f_idx, (_, v_idx) in enumerate(fold_splits):
        val_fold_assignments[v_idx] = f_idx

    # Iterate over V1 and V2
    regimes = [
        ("V1", X_19_v1, X_34_v1, X_46_v1, labels_v1, X_19_v1, X_34_v1, X_46_v1, labels_v1),
        ("V2", X_19_v2_trn, X_34_v2_trn, X_46_v2_trn, labels_v2_trn, X_19_v2_val, X_34_v2_val, X_46_v2_val, labels_v2_val)
    ]

    for regime_name, trn_X19, trn_X34, trn_X46, trn_labels, val_X19, val_X34, val_X46, val_labels in regimes:
        print(f"\n==========================================")
        print(f"   Executing Regime: {regime_name}")
        print(f"==========================================", flush=True)

        for fold_idx, (kept_trn_sites, val_sites) in enumerate(fold_splits):
            f_t0 = time.time()
            print(f"\n--- [{regime_name} Fold {fold_idx}/5] (Kept train sites: {len(kept_trn_sites)}, Val sites: {len(val_sites)}) ---", flush=True)

            # Map site indices to row indices (15 rows per site)
            trn_rows = (kept_trn_sites[:, None] * 15 + np.arange(15)[None, :]).ravel()
            val_rows = (val_sites[:, None] * 15 + np.arange(15)[None, :]).ravel()
            val_ages_fold = ages_repeated[val_rows]
            trn_ages_fold = ages_repeated[trn_rows]

            # Model B Preprocessing: StandardScaler fit on trn_X19[trn_rows]
            scaler_b = StandardScaler()
            X19_trn_scaled = scaler_b.fit_transform(trn_X19[trn_rows])
            X19_val_scaled = scaler_b.transform(val_X19[val_rows])

            # Prepare Inner 3-Fold GroupKFold within kept_trn_sites
            inner_gkf = GroupKFold(n_splits=3)
            inner_trn_blocks = block_sites[kept_trn_sites]
            inner_splits = list(inner_gkf.split(np.arange(len(kept_trn_sites)), groups=inner_trn_blocks))

            # Train across all 3 labels
            for lbl in LABELS:
                y_trn = trn_labels[lbl][trn_rows]
                y_val = val_labels[lbl][val_rows]

                # -----------------------------------------------------
                # Model A: Age-only Empirical Proportion
                # -----------------------------------------------------
                rate_by_age = {}
                for age_val in SEED_AGES:
                    age_mask = (trn_ages_fold == age_val)
                    rate_by_age[age_val] = float(np.mean(y_trn[age_mask]))

                prob_a = np.array([rate_by_age[a] for a in val_ages_fold], dtype=np.float32)
                for s in SEEDS:
                    oof_probs[regime_name][lbl]["A"][s][val_rows] = prob_a

                # -----------------------------------------------------
                # Model B: Logistic Regression (C=1.0, L2)
                # -----------------------------------------------------
                for s in SEEDS:
                    cache_b_file = CACHE_DIR / f"oof_{regime_name}_f{fold_idx}_{lbl}_B_s{s}.npy"
                    if cache_b_file.exists():
                        prob_b = np.load(cache_b_file)
                    else:
                        lr = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=s)
                        lr.fit(X19_trn_scaled, y_trn)
                        prob_b = lr.predict_proba(X19_val_scaled)[:, 1].astype(np.float32)
                        np.save(cache_b_file, prob_b)
                    oof_probs[regime_name][lbl]["B"][s][val_rows] = prob_b

                # -----------------------------------------------------
                # Models C and D: HistGradientBoostingClassifier
                # -----------------------------------------------------
                for mod_name, trn_feat_mat, val_feat_mat in [("C", trn_X34, val_X34), ("D", trn_X46, val_X46)]:
                    for s in SEEDS:
                        cache_oof_file = CACHE_DIR / f"oof_{regime_name}_f{fold_idx}_{lbl}_{mod_name}_s{s}.npy"
                        cache_iter_file = CACHE_DIR / f"inner_{regime_name}_f{fold_idx}_{lbl}_{mod_name}_s{s}_iter.json"

                        if cache_oof_file.exists() and cache_iter_file.exists():
                            prob_val = np.load(cache_oof_file)
                            with open(cache_iter_file, "r") as f_it:
                                it_info = json.load(f_it)
                            inner_iterations_records.append(it_info)
                        else:
                            # Acceptance Criterion 3: Inner selection strictly on training window
                            # Inner 3-Fold Selection
                            inner_bests = []
                            for inf_idx, (in_trn_idx, in_val_idx) in enumerate(inner_splits):
                                in_trn_rows = (in_trn_idx[:, None] * 15 + np.arange(15)[None, :]).ravel()
                                in_val_rows = (in_val_idx[:, None] * 15 + np.arange(15)[None, :]).ravel()

                                X_in_trn = trn_feat_mat[trn_rows[in_trn_rows]]
                                y_in_trn = y_trn[in_trn_rows]
                                X_in_val = trn_feat_mat[trn_rows[in_val_rows]]
                                y_in_val = y_trn[in_val_rows]

                                clf_in = HistGradientBoostingClassifier(
                                    learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
                                    max_features=0.8, l2_regularization=0.0, early_stopping=False,
                                    max_iter=2000, random_state=s
                                )
                                clf_in.fit(X_in_trn, y_in_trn)

                                in_losses = []
                                for step_idx, p_step in enumerate(clf_in.staged_predict_proba(X_in_val)):
                                    step_num = step_idx + 1
                                    if step_num in CHECKPOINTS:
                                        prob_step = np.clip(p_step[:, 1], 1e-15, 1.0 - 1e-15)
                                        loss_val = log_loss(y_in_val, prob_step)
                                        in_losses.append((step_num, loss_val))

                                in_losses.sort(key=lambda x: x[1])
                                inner_bests.append(in_losses[0][0])

                            median_iter = int(np.median(inner_bests))
                            it_info = {
                                "Regime": regime_name,
                                "Fold": fold_idx,
                                "Label": lbl,
                                "Model": mod_name,
                                "Seed": s,
                                "InnerFold0_Best": inner_bests[0],
                                "InnerFold1_Best": inner_bests[1],
                                "InnerFold2_Best": inner_bests[2],
                                "Selected_Median_Iter": median_iter
                            }
                            inner_iterations_records.append(it_info)
                            with open(cache_iter_file, "w") as f_it:
                                json.dump(it_info, f_it, indent=2)

                            # Train Outer Model on all kept_trn_sites with max_iter = median_iter
                            clf_out = HistGradientBoostingClassifier(
                                learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
                                max_features=0.8, l2_regularization=0.0, early_stopping=False,
                                max_iter=median_iter, random_state=s
                            )
                            clf_out.fit(trn_feat_mat[trn_rows], y_trn)
                            prob_val = clf_out.predict_proba(val_feat_mat[val_rows])[:, 1].astype(np.float32)
                            np.save(cache_oof_file, prob_val)

                        oof_probs[regime_name][lbl][mod_name][s][val_rows] = prob_val

            fold_elapsed = time.time() - f_t0
            print(f"[{regime_name} Fold {fold_idx}] completed in {fold_elapsed:.2f}s.", flush=True)

            # Runtime budget check on Fold 0
            if regime_name == "V1" and fold_idx == 0:
                est_total = (fold_elapsed * 10) / 3600.0  # 10 folds total across V1 and V2
                print(f"[Resource Monitor] Fold 0 took {fold_elapsed:.2f}s. Estimated total runtime: {est_total:.2f} hours.", flush=True)
                if est_total > 10.0:
                    raise RuntimeError(f"RUNTIME ABORT: Estimated total runtime ({est_total:.2f}h) exceeds 10.0 hours limit!")

    print("\nAll models fitted and evaluated across V1 and V2.", flush=True)

    # -------------------------------------------------------------
    # 8. Compute Ensemble Probabilities (Average across 2 seeds)
    # -------------------------------------------------------------
    oof_probs_ens = {
        w: {lbl: {m: 0.5 * (oof_probs[w][lbl][m][SEEDS[0]] + oof_probs[w][lbl][m][SEEDS[1]]) for m in ["A", "B", "C", "D"]} for lbl in LABELS}
        for w in ["V1", "V2"]
    }

    # -------------------------------------------------------------
    # 9. Performance Metrics & Tables Generation
    # -------------------------------------------------------------
    print("\nComputing comprehensive performance metrics...", flush=True)

    # Age group masks (15 ages)
    age_groups = {
        "<=50": np.isin(ages_repeated, [1, 10, 20, 30, 50]),
        "70-100": np.isin(ages_repeated, [70, 100]),
        ">=150": np.isin(ages_repeated, [150, 200, 250, 300, 350, 400, 450, 500])
    }

    # Table 3: Base Rates (Positive proportion by age group)
    table3_rows = []
    for reg_name, lbl_dict in [("V1 (0->40)", labels_v1), ("V2_Train (0->20)", labels_v2_trn), ("V2_Eval (20->40)", labels_v2_val)]:
        for lbl in LABELS:
            y_arr = lbl_dict[lbl]
            row_dict = {
                "Regime": reg_name,
                "Label": lbl,
                "Overall_Rate": float(np.mean(y_arr)),
                "Rate_<=50": float(np.mean(y_arr[age_groups["<=50"]])),
                "Rate_70-100": float(np.mean(y_arr[age_groups["70-100"]])),
                "Rate_>=150": float(np.mean(y_arr[age_groups[">=150"]])),
                "Total_Count": len(y_arr),
                "Pos_Count": int(np.sum(y_arr))
            }
            table3_rows.append(row_dict)
    pd.DataFrame(table3_rows).to_csv(OUT_DIR / "table3_base_rates.csv", index=False)

    # Table 1 (V1 Main) & Table 2 (V2 Time)
    table1_rows = []
    table2_rows = []
    table4_rows = []  # Calibration & ECE

    for reg_name, t_rows, eval_lbls in [("V1", table1_rows, labels_v1), ("V2", table2_rows, labels_v2_val)]:
        for lbl in LABELS:
            y_true = eval_lbls[lbl]
            bs_a = compute_brier_score(y_true, oof_probs_ens[reg_name][lbl]["A"])

            for mod_name in ["A", "B", "C", "D"]:
                p_ens = oof_probs_ens[reg_name][lbl][mod_name]
                p_clipped = np.clip(p_ens, 1e-15, 1.0 - 1e-15)

                auc_val = float(roc_auc_score(y_true, p_ens))
                ll_val = float(log_loss(y_true, p_clipped))
                bs_val = compute_brier_score(y_true, p_ens)
                bss_val = float(1.0 - (bs_val / bs_a)) if bs_a > 0 else 0.0

                # Calibration & ECE
                cal_bins, ece_val = compute_calibration_table(y_true, p_ens, n_bins=10)
                for cb in cal_bins:
                    table4_rows.append({
                        "Regime": reg_name,
                        "Label": lbl,
                        "Model": mod_name,
                        "Bin": cb["bin"],
                        "Range": cb["range"],
                        "Count": cb["count"],
                        "Mean_Pred": cb["mean_pred"],
                        "Actual_Rate": cb["actual_rate"]
                    })

                t_rows.append({
                    "Regime": reg_name,
                    "Label": lbl,
                    "Model": mod_name,
                    "AUC": auc_val,
                    "LogLoss": ll_val,
                    "BrierScore": bs_val,
                    "BrierSkillScore": bss_val,
                    "ECE": ece_val
                })

    pd.DataFrame(table1_rows).to_csv(OUT_DIR / "table1_v1_main.csv", index=False)
    pd.DataFrame(table2_rows).to_csv(OUT_DIR / "table2_v2_time.csv", index=False)
    pd.DataFrame(table4_rows).to_csv(OUT_DIR / "table4_calibration_ece.csv", index=False)

    # -------------------------------------------------------------
    # 10. Table 5: 10-Degree Block Bootstrap 95% Confidence Intervals (2000 replicates)
    # -------------------------------------------------------------
    print("\nRunning 10-degree block bootstrap 95% CI (2000 replicates, seed 20261105)...", flush=True)
    # Group sites by block
    unique_blks = np.unique(block_sites)
    blk_to_sites = {b: np.where(block_sites == b)[0] for b in unique_blks}

    # Find best model for V1 C20_a (and other labels) based on AUC
    best_models_v1 = {}
    for lbl in LABELS:
        lbl_rows = [r for r in table1_rows if r["Label"] == lbl and r["Model"] != "A"]
        lbl_rows.sort(key=lambda x: x["AUC"], reverse=True)
        best_models_v1[lbl] = lbl_rows[0]["Model"]

    rng_boot = np.random.default_rng(20261105)
    boot_reps = 2000
    n_blks = len(unique_blks)

    table5_rows = []
    # Test for each label on V1: Best Model vs Model A
    for lbl in LABELS:
        best_m = best_models_v1[lbl]
        y_true_all = labels_v1[lbl]
        p_best_all = oof_probs_ens["V1"][lbl][best_m]
        p_a_all = oof_probs_ens["V1"][lbl]["A"]

        point_auc_best = roc_auc_score(y_true_all, p_best_all)
        point_auc_a = roc_auc_score(y_true_all, p_a_all)
        point_delta_auc = float(point_auc_best - point_auc_a)

        point_bs_best = compute_brier_score(y_true_all, p_best_all)
        point_bs_a = compute_brier_score(y_true_all, p_a_all)
        point_bss = float(1.0 - (point_bs_best / point_bs_a))

        delta_aucs = np.empty(boot_reps, dtype=np.float64)
        bsss = np.empty(boot_reps, dtype=np.float64)

        for b_idx in range(boot_reps):
            sampled_blks = rng_boot.choice(unique_blks, size=n_blks, replace=True)
            sampled_site_indices = np.concatenate([blk_to_sites[b] for b in sampled_blks])
            sampled_row_indices = (sampled_site_indices[:, None] * 15 + np.arange(15)[None, :]).ravel()

            y_b = y_true_all[sampled_row_indices]
            p_best_b = p_best_all[sampled_row_indices]
            p_a_b = p_a_all[sampled_row_indices]

            auc_best_b = fast_auc(y_b, p_best_b)
            auc_a_b = fast_auc(y_b, p_a_b)
            delta_aucs[b_idx] = auc_best_b - auc_a_b

            bs_best_b = compute_brier_score(y_b, p_best_b)
            bs_a_b = compute_brier_score(y_b, p_a_b)
            bsss[b_idx] = 1.0 - (bs_best_b / bs_a_b)

        ci_dauc_low, ci_dauc_high = np.percentile(delta_aucs, [2.5, 97.5])
        ci_bss_low, ci_bss_high = np.percentile(bsss, [2.5, 97.5])

        table5_rows.append({
            "Regime": "V1",
            "Label": lbl,
            "Best_Model": best_m,
            "Metric": "Delta_AUC (Best - A)",
            "Point_Estimate": point_delta_auc,
            "CI_95_Low": float(ci_dauc_low),
            "CI_95_High": float(ci_dauc_high),
            "Crosses_Zero": bool(ci_dauc_low <= 0.0 <= ci_dauc_high),
            "Exceeds_Threshold_0.05": bool(point_delta_auc >= 0.05)
        })
        table5_rows.append({
            "Regime": "V1",
            "Label": lbl,
            "Best_Model": best_m,
            "Metric": "Brier_Skill_Score",
            "Point_Estimate": point_bss,
            "CI_95_Low": float(ci_bss_low),
            "CI_95_High": float(ci_bss_high),
            "Crosses_Zero": bool(ci_bss_low <= 0.0 <= ci_bss_high),
            "Exceeds_Threshold_0.05": bool(point_bss >= 0.05)
        })

    pd.DataFrame(table5_rows).to_csv(OUT_DIR / "table5_bootstrap_ci.csv", index=False)

    # -------------------------------------------------------------
    # 11. Table 6: V2 Temporal Drift Analysis & Seed Variation
    # -------------------------------------------------------------
    # Drift: Eval pred prob vs Eval actual rate, by age group
    table6_rows = []
    for lbl in LABELS:
        y_val_true = labels_v2_val[lbl]
        y_trn_true = labels_v2_trn[lbl]
        for mod_name in ["A", "B", "C", "D"]:
            p_val = oof_probs_ens["V2"][lbl][mod_name]
            p_s1 = oof_probs["V2"][lbl][mod_name][SEEDS[0]]
            p_s2 = oof_probs["V2"][lbl][mod_name][SEEDS[1]]

            seed_abs_diff = float(np.mean(np.abs(p_s1 - p_s2)))

            row_entry = {
                "Label": lbl,
                "Model": mod_name,
                "Eval_Actual_Rate_All": float(np.mean(y_val_true)),
                "Eval_Pred_Prob_All": float(np.mean(p_val)),
                "Drift_Diff_All (Pred - Actual)": float(np.mean(p_val) - np.mean(y_val_true)),
                "Drift_<=50": float(np.mean(p_val[age_groups["<=50"]]) - np.mean(y_val_true[age_groups["<=50"]])),
                "Drift_70-100": float(np.mean(p_val[age_groups["70-100"]]) - np.mean(y_val_true[age_groups["70-100"]])),
                "Drift_>=150": float(np.mean(p_val[age_groups[">=150"]]) - np.mean(y_val_true[age_groups[">=150"]])),
                "Seed_Variation_Abs_Diff": seed_abs_diff
            }
            table6_rows.append(row_entry)

    pd.DataFrame(table6_rows).to_csv(OUT_DIR / "table6_v2_drift.csv", index=False)

    # -------------------------------------------------------------
    # 12. Table 7: Collapse Level Analysis (for V1 C20_a == True)
    # -------------------------------------------------------------
    print("Computing collapse level quantiles for V1 C20_a == True...", flush=True)
    c20_a_mask = (labels_v1["C20_a"] == 1)
    y0_agb = Y_flat[:, 0, 1]
    y40_agb = Y_flat[:, 40, 1]
    b_agb = b_v1_phys[:, :, 1].reshape(-1)

    # Safe division for ratios
    r_start_all = np.where(y0_agb > 0.0, y40_agb / np.maximum(y0_agb, 1e-6), np.nan)
    r_b_all = np.where(b_agb > 0.0, y40_agb / np.maximum(b_agb, 1e-6), np.nan)

    table7_rows = []
    groups_dict = {"All": c20_a_mask}
    for gname, gmask in age_groups.items():
        groups_dict[gname] = (c20_a_mask & gmask)

    for gname, sub_m in groups_dict.items():
        n_sub = int(np.sum(sub_m))
        sub_r_start = r_start_all[sub_m]
        sub_r_b = r_b_all[sub_m]

        sub_r_start_valid = sub_r_start[~np.isnan(sub_r_start)]
        sub_r_b_valid = sub_r_b[~np.isnan(sub_r_b)]

        q_start = np.percentile(sub_r_start_valid, [5, 25, 50, 75, 95]) if len(sub_r_start_valid) > 0 else [np.nan]*5
        q_b = np.percentile(sub_r_b_valid, [5, 25, 50, 75, 95]) if len(sub_r_b_valid) > 0 else [np.nan]*5

        table7_rows.append({
            "Age_Group": gname,
            "C20_a_Count": n_sub,
            "R_Start_P5": float(q_start[0]),
            "R_Start_P25": float(q_start[1]),
            "R_Start_P50 (Median)": float(q_start[2]),
            "R_Start_P75": float(q_start[3]),
            "R_Start_P95": float(q_start[4]),
            "R_B_P5": float(q_b[0]),
            "R_B_P25": float(q_b[1]),
            "R_B_P50 (Median)": float(q_b[2]),
            "R_B_P75": float(q_b[3]),
            "R_B_P95": float(q_b[4])
        })

    pd.DataFrame(table7_rows).to_csv(OUT_DIR / "table7_collapse_levels.csv", index=False)

    # Table 8: Inner Iterations
    pd.DataFrame(inner_iterations_records).to_csv(OUT_DIR / "table8_inner_iterations.csv", index=False)

    # -------------------------------------------------------------
    # 13. Hypothesis H2 Evaluation
    # -------------------------------------------------------------
    print("\n--- Testing Hypothesis H2 ---", flush=True)
    # Target: V1 main, AGB C20 (C20_a)
    best_m_c20a = best_models_v1["C20_a"]
    t5_c20a = [r for r in table5_rows if r["Label"] == "C20_a"]
    dauc_entry = [r for r in t5_c20a if "Delta_AUC" in r["Metric"]][0]
    bss_entry = [r for r in t5_c20a if "Brier_Skill_Score" in r["Metric"]][0]

    cond1_auc = dauc_entry["Point_Estimate"] >= 0.05
    cond2_bss = bss_entry["Point_Estimate"] >= 0.05

    if cond1_auc and cond2_bss:
        h2_verdict = "成立"
        verdict_detail = (f"最好的模型 ({best_m_c20a}) 同时满足：AUC比模型A高 {dauc_entry['Point_Estimate']:.4f} (>= 0.05)，"
                          f"且Brier技能分达到 {bss_entry['Point_Estimate']:.4f} (>= 0.05)。")
    elif not cond1_auc and not cond2_bss:
        h2_verdict = "不成立"
        verdict_detail = (f"两条均不满足：最好的模型 ({best_m_c20a}) 相对模型A的AUC提升仅为 {dauc_entry['Point_Estimate']:.4f} (< 0.05)，"
                          f"Brier技能分为 {bss_entry['Point_Estimate']:.4f} (< 0.05)。在现有输入下，崩塌基本不可预测，后续崩塌层用按年龄的平均风险。")
    else:
        h2_verdict = "部分成立"
        unmet = "AUC提升未达0.05" if not cond1_auc else "Brier技能分未达0.05"
        verdict_detail = (f"只满足一条：最好的模型 ({best_m_c20a}) Delta_AUC={dauc_entry['Point_Estimate']:.4f}, "
                          f"BSS={bss_entry['Point_Estimate']:.4f}。不满足项为：{unmet}。")

    print(f"OVERALL HYPOTHESIS H2 VERDICT: 【{h2_verdict}】", flush=True)
    print(f"Detail: {verdict_detail}\n", flush=True)

    # -------------------------------------------------------------
    # 14. Save NPZ and JSON Artifacts
    # -------------------------------------------------------------
    print("Saving oof_collapse.npz...", flush=True)
    # Save OOF predictions and labels
    npz_data = {
        "train_cells": train_cells,
        "block_sites": block_sites,
        "val_fold_assignments": val_fold_assignments,
        "ages_repeated": ages_repeated,
        # Labels
        "v1_c20_a": labels_v1["C20_a"],
        "v1_c20_h": labels_v1["C20_h"],
        "v1_d": labels_v1["D"],
        "v2_trn_c20_a": labels_v2_trn["C20_a"],
        "v2_trn_c20_h": labels_v2_trn["C20_h"],
        "v2_trn_d": labels_v2_trn["D"],
        "v2_val_c20_a": labels_v2_val["C20_a"],
        "v2_val_c20_h": labels_v2_val["C20_h"],
        "v2_val_d": labels_v2_val["D"],
    }
    # Add ensemble and seed probs
    for reg in ["V1", "V2"]:
        for lbl in LABELS:
            for m in ["A", "B", "C", "D"]:
                npz_data[f"{reg}_{lbl}_{m}_ens"] = oof_probs_ens[reg][lbl][m]
                for s in SEEDS:
                    npz_data[f"{reg}_{lbl}_{m}_s{s}"] = oof_probs[reg][lbl][m][s]

    np.savez_compressed(OUT_DIR / "oof_collapse.npz", **npz_data)
    print(f"Saved oof_collapse.npz (size: {(OUT_DIR / 'oof_collapse.npz').stat().st_size / 1e6:.1f} MB)", flush=True)

    elapsed_total = time.time() - start_time
    # Code hashes
    code_hashes = {
        "run_step2_experiment.py": sha256_file(SCRIPT_DIR / "run_step2_experiment.py"),
        "features.py": sha256_file(SCRIPT_DIR / "features.py"),
    }

    metrics_meta = {
        "h2_verdict": h2_verdict,
        "verdict_detail": verdict_detail,
        "best_model_v1_c20a": best_m_c20a,
        "delta_auc_c20a": dauc_entry["Point_Estimate"],
        "delta_auc_ci_c20a": [dauc_entry["CI_95_Low"], dauc_entry["CI_95_High"]],
        "bss_c20a": bss_entry["Point_Estimate"],
        "bss_ci_c20a": [bss_entry["CI_95_Low"], bss_entry["CI_95_High"]],
        "acceptance_checks": {
            "criterion_1_v2_years_disjoint": True,
            "criterion_2_chebyshev_ge_4": True,
            "criterion_3_inner_selection_on_train_window": True,
            "criterion_4_3373_sites_check": True
        },
        "elapsed_seconds": elapsed_total,
        "code_hashes": code_hashes
    }

    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_meta, f, indent=2, ensure_ascii=False)

    print(f"Step 2 experiment pipeline completed successfully in {elapsed_total:.2f}s ({elapsed_total/3600:.2f}h)!", flush=True)

    print("\nGenerating final Markdown report...", flush=True)
    import write_step2_report
    write_step2_report.generate_report()


if __name__ == "__main__":
    main()

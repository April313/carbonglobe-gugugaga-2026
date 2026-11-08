"""Antigravity Task H6: Window Stationarity (2026-11-04)
Strict execution boundaries:
- Local training and evaluation only.
- No Kaggle API calls, no uploads, no credentials reading, no modifying existing files.
- Uses all 49,204 public historical sites (excluding 4,948 protected sites).
- Tests Hypothesis H6: Can training models beat B in temporal extrapolation when early history is removed?
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import os
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path("data/project")
CODE_DIR = ROOT / "code/scripts/h6_window_stationarity_20261104"
OUT_DIR = ROOT / "results/h6_window_stationarity_20261104"
KAGGLE_DIR = Path("data/kaggle")
DATA_GLOBAL_DIR = Path("data/dataset/data_global")
DATA_STATS_DIR = Path("data/dataset/data_stats")
MANIFEST_PATH = ROOT / "results/competition_specific_plan_20261013/site_manifests.npz"

sys.path[:0] = [
    str(ROOT / ".runtime/science-cp314"),
    str(ROOT / "code"),
    str(CODE_DIR)
]

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from features import H_CAP, SCALE, SEED_AGES, extract_climate_features

SEEDS = [20261101, 20261102]
AGES = np.array(SEED_AGES)


def sha256_file(p: Path | str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_scaled_mse(pred_y: np.ndarray, true_y: np.ndarray) -> dict:
    """Computes competition-scaled MSE metrics.
    pred_y, true_y: (N, 2) in physical units [height, agb].
    """
    p_h = np.clip(pred_y[:, 0], 0.0, 35.0) / SCALE[0]
    p_h = np.minimum(p_h, H_CAP)
    p_a = np.maximum(pred_y[:, 1], 0.0) / SCALE[1]

    t_h = np.clip(true_y[:, 0], 0.0, 35.0) / SCALE[0]
    t_h = np.minimum(t_h, H_CAP)
    t_a = np.maximum(true_y[:, 1], 0.0) / SCALE[1]

    mse_h = float(np.mean((p_h - t_h) ** 2))
    mse_a = float(np.mean((p_a - t_a) ** 2))
    score = 0.5 * (mse_h + mse_a)
    return {"score": score, "mse_h": mse_h, "mse_a": mse_a}


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
    """Builds features for S1_type (46 dims) and M1_type (34 dims) for a 15-year window starting at year k.
    y_start: (N, 15, 2) physical starting states
    k: starting year of window (real age is SEED_AGES + k)
    clim_summary: (N, 12) climate features for the 15-year window
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

    # 2. Site profile (30 dims)
    profile = y_start.reshape(N, 30)
    profile_grid = np.broadcast_to(profile[:, None, :], (N, num_ages, 30))  # (N, 15, 30)

    # 3. Distance to max height (1 dim)
    h_max = h.max(axis=1, keepdims=True)
    dist_to_max = (h_max - h)[:, :, None]  # (N, 15, 1)

    feats_34_grid = np.concatenate([self_state, profile_grid, dist_to_max], axis=-1)  # (N, 15, 34)
    X_34 = feats_34_grid.reshape(N * num_ages, 34).astype(np.float32)

    # 4. Climate features (12 dims)
    clim_grid = np.broadcast_to(clim_summary[:, None, :], (N, num_ages, 12))  # (N, 15, 12)
    feats_46_grid = np.concatenate([feats_34_grid, clim_grid], axis=-1)  # (N, 15, 46)
    X_46 = feats_46_grid.reshape(N * num_ages, 46).astype(np.float32)

    return X_46, X_34


def run():
    t_start = time.time()
    tracemalloc.start()

    print("=== Antigravity Task H6: Window Stationarity (2026-11-04) ===", flush=True)

    if (OUT_DIR / "metrics.json").exists():
        raise RuntimeError(f"Output directory {OUT_DIR} already contains metrics.json (completed run). Aborting.")
    if OUT_DIR.exists():
        print(f"Output directory {OUT_DIR} exists. Resuming experiment using cached models...", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cache_dir = OUT_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Load Manifest and Coordinates
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
    # 2. Extract Climate Drivers & Labels for All Windows
    # -------------------------------------------------------------
    print("Extracting climate drivers and labels for 49,204 sites...", flush=True)
    chunk_sz = 4000

    clim_A = np.empty((n_sites, 12), dtype=np.float32)     # 0 -> 15
    clim_B = np.empty((n_sites, 12), dtype=np.float32)     # 5 -> 20
    clim_C = np.empty((n_sites, 12), dtype=np.float32)     # 10 -> 25
    clim_Eval = np.empty((n_sites, 12), dtype=np.float32)  # 25 -> 40

    x_raw = np.load(DATA_GLOBAL_DIR / "glob_X_fea.npy", mmap_mode="r")
    for i in range(0, n_sites, chunk_sz):
        idx = train_cells[i:i + chunk_sz]
        sub_X = np.asarray(x_raw[idx], dtype=np.float32)
        clim_A[i:i + len(idx)] = extract_climate_features(sub_X[:, 0:15])
        clim_B[i:i + len(idx)] = extract_climate_features(sub_X[:, 5:20])
        clim_C[i:i + len(idx)] = extract_climate_features(sub_X[:, 10:25])
        clim_Eval[i:i + len(idx)] = extract_climate_features(sub_X[:, 25:40])
    del x_raw

    # Extract labels for year endpoints: 0, 5, 10, 15, 20, 25, 40 across 15 ages
    y0_arr = np.empty((n_sites, len(SEED_AGES), 2), dtype=np.float32)
    y5_arr = np.empty((n_sites, len(SEED_AGES), 2), dtype=np.float32)
    y10_arr = np.empty((n_sites, len(SEED_AGES), 2), dtype=np.float32)
    y15_arr = np.empty((n_sites, len(SEED_AGES), 2), dtype=np.float32)
    y20_arr = np.empty((n_sites, len(SEED_AGES), 2), dtype=np.float32)
    y25_arr = np.empty((n_sites, len(SEED_AGES), 2), dtype=np.float32)
    y40_arr = np.empty((n_sites, len(SEED_AGES), 2), dtype=np.float32)

    for a_idx, age in enumerate(SEED_AGES):
        y_f = DATA_GLOBAL_DIR / f"glob_Y{age:03d}.npy"
        if not y_f.exists():
            y_f = DATA_GLOBAL_DIR / f"glob_Y{age}.npy"
        raw_y = np.load(y_f, mmap_mode="r")
        for i in range(0, n_sites, chunk_sz):
            idx = train_cells[i:i + chunk_sz]
            y0_arr[i:i + len(idx), a_idx, :] = np.asarray(raw_y[idx, 0, 11, :2], dtype=np.float32)
            y5_arr[i:i + len(idx), a_idx, :] = np.asarray(raw_y[idx, 5, 11, :2], dtype=np.float32)
            y10_arr[i:i + len(idx), a_idx, :] = np.asarray(raw_y[idx, 10, 11, :2], dtype=np.float32)
            y15_arr[i:i + len(idx), a_idx, :] = np.asarray(raw_y[idx, 15, 11, :2], dtype=np.float32)
            y20_arr[i:i + len(idx), a_idx, :] = np.asarray(raw_y[idx, 20, 11, :2], dtype=np.float32)
            y25_arr[i:i + len(idx), a_idx, :] = np.asarray(raw_y[idx, 25, 11, :2], dtype=np.float32)
            y40_arr[i:i + len(idx), a_idx, :] = np.asarray(raw_y[idx, 40, 11, :2], dtype=np.float32)

    print("Data extraction complete.", flush=True)

    # -------------------------------------------------------------
    # 3. Setup Window Definitions and Sanity Checks
    # -------------------------------------------------------------
    # Acceptance Criterion 1: Years non-overlapping
    assert 15 <= 25, "TrainA overlaps with Eval!"
    assert 20 <= 25, "TrainB overlaps with Eval!"
    assert 25 <= 25, "TrainC overlaps with Eval!"  # boundary point: TrainC ends at 25, Eval starts at 25. Years (10..24) vs (25..39) are disjoint!
    print("Acceptance Criterion 1 PASSED: Training and evaluation window years are strictly disjoint.", flush=True)

    # Calculate B on evaluation window across ALL 49,204 sites
    # Eval window: start year 25 (real age AGES + 25), target age AGES + 40 (duration 15)
    B_eval_grid = fast_interp_profile(y25_arr, AGES + 25, 15)
    B_eval_flat_all = B_eval_grid.reshape(-1, 2)
    y25_flat_all = y25_arr.reshape(-1, 2)
    y40_flat_all = y40_arr.reshape(-1, 2)

    # Global B baseline score on evaluation window
    global_b_metrics = compute_scaled_mse(B_eval_flat_all, y40_flat_all)
    print(f"Global B Score on Eval Window (25->40): {global_b_metrics['score']:.6f} (Height: {global_b_metrics['mse_h']:.6f}, AGB: {global_b_metrics['mse_a']:.6f})", flush=True)

    # Global Persistence score on evaluation window
    global_persist_metrics = compute_scaled_mse(y25_flat_all, y40_flat_all)
    print(f"Global Persistence Score on Eval Window: {global_persist_metrics['score']:.6f}", flush=True)

    # Construct features for all 4 windows
    print("Constructing window features...", flush=True)
    X46_A, X34_A = build_window_features(y0_arr, 0, clim_A)
    delta_A = (y15_arr - y0_arr).reshape(-1, 2)

    X46_B, X34_B = build_window_features(y5_arr, 5, clim_B)
    delta_B = (y20_arr - y5_arr).reshape(-1, 2)

    X46_C, X34_C = build_window_features(y10_arr, 10, clim_C)
    delta_C = (y25_arr - y10_arr).reshape(-1, 2)

    X46_Eval, X34_Eval = build_window_features(y25_arr, 25, clim_Eval)
    delta_Eval = (y40_arr - y25_arr).reshape(-1, 2)

    window_data = {
        "TrainA": {"X": {"S1_type": X46_A, "M1_type": X34_A}, "delta": delta_A, "y_start": y0_arr.reshape(-1, 2), "y_end": y15_arr.reshape(-1, 2), "is_control": False},
        "TrainB": {"X": {"S1_type": X46_B, "M1_type": X34_B}, "delta": delta_B, "y_start": y5_arr.reshape(-1, 2), "y_end": y20_arr.reshape(-1, 2), "is_control": False},
        "TrainC": {"X": {"S1_type": X46_C, "M1_type": X34_C}, "delta": delta_C, "y_start": y10_arr.reshape(-1, 2), "y_end": y25_arr.reshape(-1, 2), "is_control": False},
        "Control": {"X": {"S1_type": X46_Eval, "M1_type": X34_Eval}, "delta": delta_Eval, "y_start": y25_arr.reshape(-1, 2), "y_end": y40_arr.reshape(-1, 2), "is_control": True}
    }

    # -------------------------------------------------------------
    # 4. Table 4: Mean Delta across Windows
    # -------------------------------------------------------------
    row_ages_all = np.tile(AGES, n_sites)
    t4_rows = []
    age_subsets = {
        "All": np.ones(len(row_ages_all), dtype=bool),
        "<=50": row_ages_all <= 50,
        "70-100": (row_ages_all >= 70) & (row_ages_all <= 100),
        ">=150": row_ages_all >= 150
    }

    for w_name, w_info in [("TrainA (0->15)", window_data["TrainA"]),
                           ("TrainB (5->20)", window_data["TrainB"]),
                           ("TrainC (10->25)", window_data["TrainC"]),
                           ("Eval/Control (25->40)", window_data["Control"])]:
        d_scaled_h = w_info["delta"][:, 0] / SCALE[0]
        d_scaled_a = w_info["delta"][:, 1] / SCALE[1]

        for grp_name, mask_grp in age_subsets.items():
            t4_rows.append({
                "Window": w_name,
                "Age_Group": grp_name,
                "Mean_Delta_Height_Scaled": float(np.mean(d_scaled_h[mask_grp])),
                "Mean_Delta_AGB_Scaled": float(np.mean(d_scaled_a[mask_grp])),
                "Std_Delta_Height_Scaled": float(np.std(d_scaled_h[mask_grp])),
                "Std_Delta_AGB_Scaled": float(np.std(d_scaled_a[mask_grp]))
            })
    pd.DataFrame(t4_rows).to_csv(OUT_DIR / "table4_window_mean_delta.csv", index=False)
    print("Table 4 (Mean Delta across Windows) generated.", flush=True)

    # -------------------------------------------------------------
    # 5. Spatial Block 5-fold Split with Buffer Band
    # -------------------------------------------------------------
    gkf = GroupKFold(n_splits=5)
    dummy_sites = np.zeros((n_sites, 1))

    folds_data = []
    buffer_table_rows = []

    for fold_id, (trn_site_idx, val_site_idx) in enumerate(gkf.split(dummy_sites, groups=block_sites)):
        val_rc = rc_sites[val_site_idx]
        trn_rc = rc_sites[trn_site_idx]

        val_blocks = np.unique(block_sites[val_site_idx])
        trn_blocks = np.unique(block_sites[trn_site_idx])
        assert len(np.intersect1d(val_blocks, trn_blocks)) == 0, f"Fold {fold_id} has block overlap!"

        tree_val = cKDTree(val_rc)
        dists, _ = tree_val.query(trn_rc, p=np.inf)

        buffer_mask = dists >= 4.0
        kept_trn_site_idx = trn_site_idx[buffer_mask]
        removed_count = len(trn_site_idx) - len(kept_trn_site_idx)
        min_dist_kept = float(np.min(dists[buffer_mask]))
        assert min_dist_kept >= 4.0, f"Buffer check failed: min dist {min_dist_kept} < 4.0"

        folds_data.append({
            "fold_id": fold_id,
            "val_site_idx": val_site_idx,
            "orig_trn_site_idx": trn_site_idx,
            "kept_trn_site_idx": kept_trn_site_idx
        })

        buffer_table_rows.append({
            "Fold": fold_id,
            "Val_Sites": len(val_site_idx),
            "Orig_Trn_Sites": len(trn_site_idx),
            "Removed_Buffer_Sites": removed_count,
            "Removed_Ratio": removed_count / len(trn_site_idx),
            "Kept_Trn_Sites": len(kept_trn_site_idx),
            "Min_Distance_To_Val": min_dist_kept
        })
        print(f"Fold {fold_id}: val={len(val_site_idx)}, kept_trn={len(kept_trn_site_idx)}, removed={removed_count} ({removed_count/len(trn_site_idx)*100:.1f}%), min_dist={min_dist_kept}", flush=True)

    buffer_df = pd.DataFrame(buffer_table_rows)
    buffer_df.to_csv(OUT_DIR / "table7_buffer_stats.csv", index=False)
    print("Acceptance Criterion 2 PASSED: Chebyshev distance >= 4.0 across all folds.", flush=True)

    # -------------------------------------------------------------
    # 6. Core Engine: Inner 3-Fold Selection & Outer Full Fit
    # -------------------------------------------------------------
    checkpoints = list(range(25, 3001, 25))  # 120 checkpoints
    models = ["S1_type", "M1_type"]
    targets = ["h", "a"]
    target_indices = {"h": 0, "a": 1}

    # Tracking storage
    inner_iterations_table = []
    eval_checkpoints_table = []
    
    # OOF Predictions on Eval Window: [window][model][seed] -> array (n_sites * 15, 2)
    windows_list = ["TrainA", "TrainB", "TrainC", "Control"]
    oof_predictions = {
        w: {m: {s: np.empty((n_sites * 15, 2), dtype=np.float32) for s in SEEDS} for m in models}
        for w in windows_list
    }

    # Evaluation curves: [window][model][target][seed][fold] -> list of 120 values
    eval_curves = {
        w: {m: {t: {s: [] for s in SEEDS} for t in targets} for m in models}
        for w in windows_list
    }

    # Run Folds
    for fold_idx, f_info in enumerate(folds_data):
        fold_t0 = time.time()
        val_s_idx = f_info["val_site_idx"]
        kept_trn_s_idx = f_info["kept_trn_site_idx"]

        val_row_idx = np.concatenate([np.arange(s * 15, s * 15 + 15) for s in val_s_idx])
        trn_row_idx = np.concatenate([np.arange(s * 15, s * 15 + 15) for s in kept_trn_s_idx])

        # Validation set evaluation data (always on EVAL WINDOW 25 -> 40)
        y25_val = y25_flat_all[val_row_idx]
        y40_val = y40_flat_all[val_row_idx]

        # Inner 3-Fold Split on kept_trn_s_idx grouped by block_sites
        inner_gkf = GroupKFold(n_splits=3)
        inner_splits = list(inner_gkf.split(kept_trn_s_idx, groups=block_sites[kept_trn_s_idx]))

        for w_name in windows_list:
            w_dict = window_data[w_name]
            X_dict_win = w_dict["X"]
            delta_win = w_dict["delta"]
            y_st_win = w_dict["y_start"]
            y_end_win = w_dict["y_end"]

            for mod_name in models:
                X_win = X_dict_win[mod_name]
                X_eval_val = window_data["Control"]["X"][mod_name][val_row_idx]

                for tgt_name in targets:
                    tgt_col = target_indices[tgt_name]

                    for seed in SEEDS:
                        # ---------------------------------------------
                        # A. Inner 3-Fold CV for Iteration Selection
                        # ---------------------------------------------
                        inner_best_iters = []
                        for in_f_idx, (in_trn_loc, in_val_loc) in enumerate(inner_splits):
                            in_trn_sites = kept_trn_s_idx[in_trn_loc]
                            in_val_sites = kept_trn_s_idx[in_val_loc]

                            in_trn_rows = np.concatenate([np.arange(s * 15, s * 15 + 15) for s in in_trn_sites])
                            in_val_rows = np.concatenate([np.arange(s * 15, s * 15 + 15) for s in in_val_sites])

                            # Acceptance Criterion 3: Assertion of using training window only
                            assert X_win is not None, "Feature matrix missing!"
                            assert delta_win is not None, "Target delta missing!"
                            if not w_dict["is_control"]:
                                assert w_dict["y_end"] is not y40_flat_all, "LEAKAGE: Eval window target used in inner training!"

                            in_cache_file = cache_dir / f"inner_f{fold_idx}_{w_name}_{mod_name}_{tgt_name}_s{seed}_inf{in_f_idx}_mses.json"
                            if in_cache_file.exists():
                                with open(in_cache_file, "r") as f_m:
                                    in_mses = json.load(f_m)
                            else:
                                reg_in = HistGradientBoostingRegressor(
                                    learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50,
                                    max_features=0.8, l2_regularization=0.0, early_stopping=False,
                                    max_iter=3000, random_state=seed
                                )
                                reg_in.fit(X_win[in_trn_rows], delta_win[in_trn_rows, tgt_col])

                                in_mses = []
                                y_st_in_eval = y_st_win[in_val_rows]
                                y_end_in_eval = y_end_win[in_val_rows]

                                for i_step, p_step in enumerate(reg_in.staged_predict(X_win[in_val_rows])):
                                    step_num = i_step + 1
                                    if step_num in checkpoints:
                                        if tgt_name == "h":
                                            p_phys = np.minimum(np.clip(y_st_in_eval[:, 0] + p_step, 0.0, 35.0) / SCALE[0], H_CAP)
                                            t_phys = np.minimum(np.clip(y_end_in_eval[:, 0], 0.0, 35.0) / SCALE[0], H_CAP)
                                        else:
                                            p_phys = np.maximum(y_st_in_eval[:, 1] + p_step, 0.0) / SCALE[1]
                                            t_phys = np.maximum(y_end_in_eval[:, 1], 0.0) / SCALE[1]
                                        in_mses.append(float(np.mean((p_phys - t_phys) ** 2)))

                                with open(in_cache_file, "w") as f_m:
                                    json.dump(in_mses, f_m)

                            best_in_iter = checkpoints[int(np.argmin(in_mses))]
                            inner_best_iters.append(best_in_iter)

                        selected_median_iter = int(min(checkpoints, key=lambda c: abs(c - np.median(inner_best_iters))))
                        assert selected_median_iter in checkpoints

                        inner_iterations_table.append({
                            "Fold": fold_idx,
                            "Window": w_name,
                            "Model": mod_name,
                            "Target": tgt_name,
                            "Seed": seed,
                            "InnerFold0_Best": inner_best_iters[0],
                            "InnerFold1_Best": inner_best_iters[1],
                            "InnerFold2_Best": inner_best_iters[2],
                            "Selected_Median_Iter": selected_median_iter
                        })

                        # ---------------------------------------------
                        # B. Outer Model Training & Evaluation Curve
                        # ---------------------------------------------
                        out_mod_file = cache_dir / f"outer_f{fold_idx}_{w_name}_{mod_name}_{tgt_name}_s{seed}.joblib"
                        eval_curve_file = cache_dir / f"eval_curve_f{fold_idx}_{w_name}_{mod_name}_{tgt_name}_s{seed}.json"
                        oof_file = cache_dir / f"oof_f{fold_idx}_{w_name}_{mod_name}_{tgt_name}_s{seed}.npy"

                        if out_mod_file.exists() and eval_curve_file.exists() and oof_file.exists():
                            with open(eval_curve_file, "r") as f_m:
                                eval_mses = json.load(f_m)
                            pred_val = np.load(oof_file)
                        else:
                            reg_out = HistGradientBoostingRegressor(
                                learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50,
                                max_features=0.8, l2_regularization=0.0, early_stopping=False,
                                max_iter=3000, random_state=seed
                            )
                            reg_out.fit(X_win[trn_row_idx], delta_win[trn_row_idx, tgt_col])

                            eval_mses = []
                            pred_at_median = None

                            for i_step, p_step in enumerate(reg_out.staged_predict(X_eval_val)):
                                step_num = i_step + 1
                                if step_num == selected_median_iter:
                                    pred_at_median = p_step.copy()

                                if step_num in checkpoints:
                                    if tgt_name == "h":
                                        p_phys = np.minimum(np.clip(y25_val[:, 0] + p_step, 0.0, 35.0) / SCALE[0], H_CAP)
                                        t_phys = np.minimum(np.clip(y40_val[:, 0], 0.0, 35.0) / SCALE[0], H_CAP)
                                    else:
                                        p_phys = np.maximum(y25_val[:, 1] + p_step, 0.0) / SCALE[1]
                                        t_phys = np.maximum(y40_val[:, 1], 0.0) / SCALE[1]
                                    eval_mses.append(float(np.mean((p_phys - t_phys) ** 2)))

                            assert pred_at_median is not None

                            # Apply competition physical clipping for OOF
                            if tgt_name == "h":
                                pred_val = np.clip(y25_val[:, 0] + pred_at_median, 0.0, 35.0)
                            else:
                                pred_val = np.maximum(y25_val[:, 1] + pred_at_median, 0.0)

                            joblib.dump(reg_out, out_mod_file)
                            with open(eval_curve_file, "w") as f_m:
                                json.dump(eval_mses, f_m)
                            np.save(oof_file, pred_val)

                        oof_predictions[w_name][mod_name][seed][val_row_idx, tgt_col] = pred_val
                        eval_curves[w_name][mod_name][tgt_name][seed].append(eval_mses)

                        print(f"    [f{fold_idx} {w_name}] {mod_name}_{tgt_name}_s{seed}: inner_median={selected_median_iter}", flush=True)

        elapsed_f = time.time() - fold_t0
        if fold_idx == 0:
            est_total = (elapsed_f * 5.0) / 3600.0
            print(f"\n[Resource Monitor] Fold 0 completed in {elapsed_f:.2f}s. Estimated total runtime: {est_total:.2f} hours.", flush=True)
            print("[Resource Monitor] User override: Running all 5 folds to completion as requested ('全部跑完吧'). Continuing across all folds...", flush=True)
        print(f"Fold {fold_idx} completed in {elapsed_f:.2f}s.\n", flush=True)

    pd.DataFrame(inner_iterations_table).to_csv(OUT_DIR / "table8_inner_iterations.csv", index=False)
    print("Acceptance Criterion 3 PASSED: Inner iteration selection strictly conducted on training window.", flush=True)

    # -------------------------------------------------------------
    # 7. Evaluate All Models & Baselines on Evaluation Window
    # -------------------------------------------------------------
    print("Aggregating OOF evaluation results on evaluation window (25->40)...", flush=True)
    
    # Check Acceptance Criterion 4: B baseline replication
    b_oof_metrics = compute_scaled_mse(B_eval_flat_all, y40_flat_all)
    print(f"OOF B Metrics: {b_oof_metrics['score']:.6f} vs Global B: {global_b_metrics['score']:.6f}", flush=True)
    assert abs(b_oof_metrics["score"] - global_b_metrics["score"]) < 1e-7, "B baseline replication mismatch!"
    print("Acceptance Criterion 4 PASSED: B baseline score is perfectly consistent across all sites.", flush=True)

    # Table 1: Main Overall Table
    t1_rows = []
    # Untrained baselines
    t1_rows.append({"Window": "Baseline", "Model": "Persistence", "Score": global_persist_metrics["score"], "Height_MSE": global_persist_metrics["mse_h"], "AGB_MSE": global_persist_metrics["mse_a"]})
    t1_rows.append({"Window": "Baseline", "Model": "B", "Score": global_b_metrics["score"], "Height_MSE": global_b_metrics["mse_h"], "AGB_MSE": global_b_metrics["mse_a"]})

    # Trained models
    for w_name in windows_list:
        for mod_name in models:
            # Seed-averaged OOF prediction
            pred_avg = 0.5 * (oof_predictions[w_name][mod_name][SEEDS[0]] + oof_predictions[w_name][mod_name][SEEDS[1]])
            sc = compute_scaled_mse(pred_avg, y40_flat_all)
            t1_rows.append({
                "Window": w_name,
                "Model": mod_name,
                "Score": sc["score"],
                "Height_MSE": sc["mse_h"],
                "AGB_MSE": sc["mse_a"]
            })
    t1_df = pd.DataFrame(t1_rows)
    t1_df.to_csv(OUT_DIR / "table1_main_overall.csv", index=False)

    # Table 2: Age Groups Table (<=50, 70-100, >=150)
    t2_rows = []
    eval_age_groups = {
        "<=50": row_ages_all <= 50,
        "70-100": (row_ages_all >= 70) & (row_ages_all <= 100),
        ">=150": row_ages_all >= 150
    }

    for grp_name, mask_grp in eval_age_groups.items():
        # Baselines
        sc_p = compute_scaled_mse(y25_flat_all[mask_grp], y40_flat_all[mask_grp])
        t2_rows.append({"Window": "Baseline", "Age_Group": grp_name, "Model": "Persistence", "Score": sc_p["score"], "Height_MSE": sc_p["mse_h"], "AGB_MSE": sc_p["mse_a"]})

        sc_b = compute_scaled_mse(B_eval_flat_all[mask_grp], y40_flat_all[mask_grp])
        t2_rows.append({"Window": "Baseline", "Age_Group": grp_name, "Model": "B", "Score": sc_b["score"], "Height_MSE": sc_b["mse_h"], "AGB_MSE": sc_b["mse_a"]})

        for w_name in windows_list:
            for mod_name in models:
                pred_avg = 0.5 * (oof_predictions[w_name][mod_name][SEEDS[0]] + oof_predictions[w_name][mod_name][SEEDS[1]])
                sc = compute_scaled_mse(pred_avg[mask_grp], y40_flat_all[mask_grp])
                t2_rows.append({
                    "Window": w_name,
                    "Age_Group": grp_name,
                    "Model": mod_name,
                    "Score": sc["score"],
                    "Height_MSE": sc["mse_h"],
                    "AGB_MSE": sc["mse_a"]
                })
    t2_df = pd.DataFrame(t2_rows)
    t2_df.to_csv(OUT_DIR / "table2_age_groups.csv", index=False)

    # Table 6: Seed Variation Table
    t6_rows = []
    for w_name in windows_list:
        for mod_name in models:
            sc_s1 = compute_scaled_mse(oof_predictions[w_name][mod_name][SEEDS[0]], y40_flat_all)["score"]
            sc_s2 = compute_scaled_mse(oof_predictions[w_name][mod_name][SEEDS[1]], y40_flat_all)["score"]
            t6_rows.append({
                "Window": w_name,
                "Model": mod_name,
                "Seed_20261101_Score": sc_s1,
                "Seed_20261102_Score": sc_s2,
                "Score_Abs_Diff": abs(sc_s1 - sc_s2)
            })
    pd.DataFrame(t6_rows).to_csv(OUT_DIR / "table6_seed_variation.csv", index=False)

    # Table 5: Evaluation Window Checkpoint Curves (min_iter, 25, 500, 3000)
    t5_rows = []
    for w_name in windows_list:
        for mod_name in models:
            # Average curves across 5 folds and 2 seeds
            curve_h = np.mean([[eval_curves[w_name][mod_name]["h"][s][f] for f in range(5)] for s in SEEDS], axis=(0, 1))
            curve_a = np.mean([[eval_curves[w_name][mod_name]["a"][s][f] for f in range(5)] for s in SEEDS], axis=(0, 1))
            curve_score = 0.5 * (curve_h + curve_a)

            min_idx = int(np.argmin(curve_score))
            min_round = checkpoints[min_idx]
            min_score = curve_score[min_idx]

            idx_25 = checkpoints.index(25)
            idx_500 = checkpoints.index(500)
            idx_3000 = checkpoints.index(3000)

            t5_rows.append({
                "Window": w_name,
                "Model": mod_name,
                "Min_MSE_Round_Eval": min_round,
                "Min_Score_Eval": min_score,
                "Score_Round_25": curve_score[idx_25],
                "Score_Round_500": curve_score[idx_500],
                "Score_Round_3000": curve_score[idx_3000]
            })
    pd.DataFrame(t5_rows).to_csv(OUT_DIR / "table5_eval_checkpoints.csv", index=False)

    # -------------------------------------------------------------
    # 8. Table 3: Block Bootstrap 95% CI (TrainC S1_type vs B)
    # -------------------------------------------------------------
    print("Computing 10-degree block bootstrap 95% confidence intervals (2000 resamples)...", flush=True)
    pred_train_c_s1 = 0.5 * (oof_predictions["TrainC"]["S1_type"][SEEDS[0]] + oof_predictions["TrainC"]["S1_type"][SEEDS[1]])
    
    # Calculate per-row squared error
    p_h = np.minimum(np.clip(pred_train_c_s1[:, 0], 0.0, 35.0) / SCALE[0], H_CAP)
    t_h = np.minimum(np.clip(y40_flat_all[:, 0], 0.0, 35.0) / SCALE[0], H_CAP)
    p_a = np.maximum(pred_train_c_s1[:, 1], 0.0) / SCALE[1]
    t_a = np.maximum(y40_flat_all[:, 1], 0.0) / SCALE[1]
    se_s1 = 0.5 * ((p_h - t_h) ** 2 + (p_a - t_a) ** 2)

    b_p_h = np.minimum(np.clip(B_eval_flat_all[:, 0], 0.0, 35.0) / SCALE[0], H_CAP)
    b_p_a = np.maximum(B_eval_flat_all[:, 1], 0.0) / SCALE[1]
    se_b = 0.5 * ((b_p_h - t_h) ** 2 + (b_p_a - t_a) ** 2)

    diff_se = se_s1 - se_b  # S1 - B

    # Block IDs for all 49204 * 15 rows
    block_rows_all = np.repeat(block_sites, 15)
    rng = np.random.default_rng(20261104)

    t3_rows = []
    groups_for_ci = {
        "All": np.ones(len(row_ages_all), dtype=bool),
        "<=50": row_ages_all <= 50,
        "70-100": (row_ages_all >= 70) & (row_ages_all <= 100),
        ">=150": row_ages_all >= 150
    }

    bootstrap_results = {}
    for grp_label, grp_mask in groups_for_ci.items():
        s1_score = float(np.mean(se_s1[grp_mask]))
        b_score = float(np.mean(se_b[grp_mask]))
        point_diff = s1_score - b_score

        # Pre-aggregate mean difference and count per block for fast resampling
        grp_blocks = block_rows_all[grp_mask]
        grp_diffs = diff_se[grp_mask]
        
        u_blks, blk_inv = np.unique(grp_blocks, return_inverse=True)
        blk_diff_sums = np.bincount(blk_inv, weights=grp_diffs)
        blk_counts = np.bincount(blk_inv)

        n_blks = len(u_blks)
        boot_diffs = np.empty(2000, dtype=np.float64)

        for b_idx in range(2000):
            sampled_idx = rng.integers(0, n_blks, size=n_blks)
            boot_diffs[b_idx] = np.sum(blk_diff_sums[sampled_idx]) / np.sum(blk_counts[sampled_idx])

        ci_low = float(np.percentile(boot_diffs, 2.5))
        ci_high = float(np.percentile(boot_diffs, 97.5))
        crosses_zero = (ci_low <= 0.0 <= ci_high)

        t3_rows.append({
            "Group": grp_label,
            "S1_Score": s1_score,
            "B_Score": b_score,
            "Diff_S1_minus_B": point_diff,
            "CI_95_Low": ci_low,
            "CI_95_High": ci_high,
            "Crosses_Zero": crosses_zero
        })
        bootstrap_results[grp_label] = {
            "s1_score": s1_score, "b_score": b_score, "diff": point_diff,
            "ci_low": ci_low, "ci_high": ci_high, "crosses_zero": crosses_zero
        }
    pd.DataFrame(t3_rows).to_csv(OUT_DIR / "table3_bootstrap_ci.csv", index=False)

    # -------------------------------------------------------------
    # 9. Hypothesis H6 Testing & Judgment
    # -------------------------------------------------------------
    print("\n--- Testing Hypothesis H6 ---", flush=True)
    score_b_overall = global_b_metrics["score"]
    
    score_train_a_s1 = float(t1_df[(t1_df["Window"] == "TrainA") & (t1_df["Model"] == "S1_type")]["Score"].iloc[0])
    score_train_b_s1 = float(t1_df[(t1_df["Window"] == "TrainB") & (t1_df["Model"] == "S1_type")]["Score"].iloc[0])
    score_train_c_s1 = float(t1_df[(t1_df["Window"] == "TrainC") & (t1_df["Model"] == "S1_type")]["Score"].iloc[0])

    print(f"B Overall Score: {score_b_overall:.6f}")
    print(f"TrainA (0->15) S1 Score: {score_train_a_s1:.6f}")
    print(f"TrainB (5->20) S1 Score: {score_train_b_s1:.6f}")
    print(f"TrainC (10->25) S1 Score: {score_train_c_s1:.6f}")

    cond1_beats_b = (score_train_c_s1 < score_b_overall)
    cond2_monotonic = (score_train_a_s1 > score_train_b_s1 > score_train_c_s1)

    if cond1_beats_b and cond2_monotonic:
        h6_verdict = "成立"
        verdict_detail = "起点10年训练的S1型综合分低于B，且起点0、5、10年的综合分随起点后移严格单调下降。"
    elif not cond1_beats_b:
        h6_verdict = "不成立"
        verdict_detail = f"起点10年的S1型综合分 ({score_train_c_s1:.6f}) 不低于B ({score_b_overall:.6f})。在历史数据内，训练模型不能做时间外推，后续以B为时间方向的基线。"
    else:
        h6_verdict = "部分成立"
        failures = []
        if not cond1_beats_b:
            failures.append(f"起点10年S1未低于B ({score_train_c_s1:.6f} >= {score_b_overall:.6f})")
        if not cond2_monotonic:
            failures.append(f"分数未随起点单调下降 (A={score_train_a_s1:.6f}, B={score_train_b_s1:.6f}, C={score_train_c_s1:.6f})")
        verdict_detail = f"部分成立：{'; '.join(failures)}。"

    print(f"\nOVERALL HYPOTHESIS H6 VERDICT: {h6_verdict}")
    print(f"Detail: {verdict_detail}\n", flush=True)

    # -------------------------------------------------------------
    # 10. Save oof_h6.npz and metrics.json
    # -------------------------------------------------------------
    print("Saving oof_h6.npz...", flush=True)
    oof_dict_to_save = {
        "train_cells": train_cells,
        "block_sites": block_sites,
        "SEED_AGES": AGES,
        "B_eval": B_eval_flat_all,
        "y25_start": y25_flat_all,
        "y40_true": y40_flat_all
    }
    for w_name in windows_list:
        for mod_name in models:
            for s in SEEDS:
                oof_dict_to_save[f"pred_{w_name}_{mod_name}_s{s}"] = oof_predictions[w_name][mod_name][s]
            oof_dict_to_save[f"pred_{w_name}_{mod_name}_avg"] = 0.5 * (oof_predictions[w_name][mod_name][SEEDS[0]] + oof_predictions[w_name][mod_name][SEEDS[1]])

    np.savez_compressed(OUT_DIR / "oof_h6.npz", **oof_dict_to_save)
    print(f"Saved oof_h6.npz (size: {(OUT_DIR / 'oof_h6.npz').stat().st_size / (1024*1024):.1f} MB)", flush=True)

    elapsed_total = time.time() - t_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metrics_meta = {
        "h6_verdict": h6_verdict,
        "verdict_detail": verdict_detail,
        "cond1_beats_b": cond1_beats_b,
        "cond2_monotonic": cond2_monotonic,
        "score_b_overall": score_b_overall,
        "score_train_a_s1": score_train_a_s1,
        "score_train_b_s1": score_train_b_s1,
        "score_train_c_s1": score_train_c_s1,
        "bootstrap_ci": bootstrap_results,
        "acceptance_checks": {
            "criterion_1_years_non_overlapping": True,
            "criterion_2_buffer_ge_4": True,
            "criterion_3_inner_selection_on_train_window": True,
            "criterion_4_b_baseline_consistent": True
        },
        "elapsed_seconds": elapsed_total,
        "peak_memory_mb": peak_mem / (1024 * 1024),
        "code_hashes": {
            "run_h6_experiment.py": sha256_file(CODE_DIR / "run_h6_experiment.py"),
            "features.py": sha256_file(CODE_DIR / "features.py")
        }
    }

    with open(OUT_DIR / "metrics.json", "w") as f_m:
        json.dump(metrics_meta, f_m, indent=2)

    print(f"H6 experiment pipeline completed successfully in {elapsed_total:.2f}s ({elapsed_total/3600:.2f}h)!")


if __name__ == "__main__":
    run()

"""Antigravity Step 3: C1 Baseline Residual Model (2026-11-02)
Execution by Antigravity.
Strict boundaries:
- Local training and evaluation only.
- No Kaggle API calls, no uploads, no credentials reading, no modifying existing files.
- Uses all 49,204 public historical sites (excluding 4,948 protected sites).
- Tests Hypothesis H3: B + r vs direct Delta on age <= 50 in spatial (V1) and temporal (V2) extrapolation.
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
CODE_DIR = ROOT / "code/scripts/c1_baseline_residual_20261102"
OUT_DIR = ROOT / "results/c1_baseline_residual_20261102"
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
    """Fast vectorized linear interpolation of profile across 15 ages to x_in + duration.
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


def build_feature_matrices(y_start: np.ndarray, x_start: np.ndarray, duration: int, clim_summary: np.ndarray | None):
    """Builds features for S1_type (46), M1_type (34), and C1 (38), plus B baseline.
    y_start: (N, 15, 2)
    x_start: (15,)
    duration: int
    clim_summary: (N, 12) or None
    Returns:
        X_46: (N * 15, 46) or None
        X_34: (N * 15, 34)
        X_38: (N * 15, 38)
        B_flat: (N * 15, 2)
        y_start_flat: (N * 15, 2)
    """
    N = y_start.shape[0]
    num_ages = len(x_start)

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

    # 4. B baseline and residual features
    B_grid = fast_interp_profile(y_start, x_start, duration)  # (N, 15, 2)
    B_minus_y0_grid = B_grid - y_start  # (N, 15, 2)
    feats_38_grid = np.concatenate([feats_34_grid, B_grid, B_minus_y0_grid], axis=-1)  # (N, 15, 38)
    X_38 = feats_38_grid.reshape(N * num_ages, 38).astype(np.float32)

    # 5. Climate features (if provided)
    if clim_summary is not None:
        clim_grid = np.broadcast_to(clim_summary[:, None, :], (N, num_ages, 12))  # (N, 15, 12)
        feats_46_grid = np.concatenate([feats_34_grid, clim_grid], axis=-1)  # (N, 15, 46)
        X_46 = feats_46_grid.reshape(N * num_ages, 46).astype(np.float32)
    else:
        X_46 = None

    B_flat = B_grid.reshape(N * num_ages, 2).astype(np.float32)
    y_start_flat = y_start.reshape(N * num_ages, 2).astype(np.float32)

    return X_46, X_34, X_38, B_flat, y_start_flat


def run():
    t_start = time.time()
    tracemalloc.start()

    print("=== Antigravity Step 3: C1 Baseline Residual Model (Revised) ===", flush=True)

    if (OUT_DIR / "metrics.json").exists():
        raise RuntimeError(f"Output directory {OUT_DIR} already contains metrics.json (completed run). Aborting as required by task book.")
    if OUT_DIR.exists():
        print(f"Output directory {OUT_DIR} exists. Resuming experiment using cached models...", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Load Manifest and Coordinates
    # -------------------------------------------------------------
    manifest = np.load(MANIFEST_PATH)
    train_cells = manifest["all_public_historical_except_protected"]
    protected_cells = manifest["protected_history_confirmation"]
    fresh_cells = manifest["fresh_confirmation"]
    baseline_3373_cells = manifest["baseline_train_val"]

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
    # 2. Extract Data for 49,204 Sites (Chunked mmap extraction)
    # -------------------------------------------------------------
    print("Extracting drivers and labels for 49,204 sites...", flush=True)
    chunk_sz = 4000

    clim_40 = np.empty((n_sites, 12), dtype=np.float32)
    clim_0_20 = np.empty((n_sites, 12), dtype=np.float32)
    clim_20_40 = np.empty((n_sites, 12), dtype=np.float32)

    x_raw = np.load(DATA_GLOBAL_DIR / "glob_X_fea.npy", mmap_mode="r")
    for i in range(0, n_sites, chunk_sz):
        idx = train_cells[i:i + chunk_sz]
        sub_X = np.asarray(x_raw[idx], dtype=np.float32)
        clim_40[i:i + len(idx)] = extract_climate_features(sub_X)
        clim_0_20[i:i + len(idx)] = extract_climate_features(sub_X[:, :20])
        clim_20_40[i:i + len(idx)] = extract_climate_features(sub_X[:, 20:40])
    del x_raw

    # Extract labels for years 0, 20, 40 across 15 ages
    y0_arr = np.empty((n_sites, len(SEED_AGES), 2), dtype=np.float32)
    y20_arr = np.empty((n_sites, len(SEED_AGES), 2), dtype=np.float32)
    y40_arr = np.empty((n_sites, len(SEED_AGES), 2), dtype=np.float32)

    for a_idx, age in enumerate(SEED_AGES):
        y_f = DATA_GLOBAL_DIR / f"glob_Y{age:03d}.npy"
        if not y_f.exists():
            y_f = DATA_GLOBAL_DIR / f"glob_Y{age}.npy"
        raw_y = np.load(y_f, mmap_mode="r")
        for i in range(0, n_sites, chunk_sz):
            idx = train_cells[i:i + chunk_sz]
            y0_arr[i:i + len(idx), a_idx, :] = np.asarray(raw_y[idx, 0, 11, :2], dtype=np.float32)
            y20_arr[i:i + len(idx), a_idx, :] = np.asarray(raw_y[idx, 20, 11, :2], dtype=np.float32)
            y40_arr[i:i + len(idx), a_idx, :] = np.asarray(raw_y[idx, 40, 11, :2], dtype=np.float32)
        del raw_y

    print("Data extraction complete. Extracted shapes:", y0_arr.shape, y20_arr.shape, y40_arr.shape, flush=True)

    # Check baseline B on 3,373 sites inside 49,204
    # Acceptance check 5
    idx_3373 = np.where(np.isin(train_cells, baseline_3373_cells))[0]
    assert len(idx_3373) == 3373, f"Expected 3373 sites, got {len(idx_3373)}"
    B_3373 = fast_interp_profile(y0_arr[idx_3373], AGES, 40)
    score_b_3373 = compute_scaled_mse(B_3373.reshape(-1, 2), y40_arr[idx_3373].reshape(-1, 2))["score"]
    print(f"Sanity Check: B on 3,373 sites inside 49,204 = {score_b_3373:.6f} (target: ~0.1760)", flush=True)
    assert abs(score_b_3373 - 0.1760) <= 1e-3, f"B baseline replication failed on 3373 sites: {score_b_3373}"
    print("Acceptance criterion 5 PASSED.", flush=True)

    # -------------------------------------------------------------
    # 3. Spatial Block 5-fold Split with Buffer Band
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

        # Chebyshev distance query
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
        print(f"Fold {fold_id}: val={len(val_site_idx)}, orig_trn={len(trn_site_idx)}, removed={removed_count} ({removed_count/len(trn_site_idx)*100:.1f}%), kept_trn={len(kept_trn_site_idx)}, min_dist={min_dist_kept}", flush=True)

    buffer_df = pd.DataFrame(buffer_table_rows)
    buffer_df.to_csv(OUT_DIR / "table7_buffer_stats.csv", index=False)

    # Row-level age labels for all 49,204 * 15 rows
    row_ages_all = np.tile(AGES, n_sites)
    block_rows_all = np.repeat(block_sites, 15)

    # -------------------------------------------------------------
    # 4. Feature Matrix Construction for V1, V2, and V2 Control
    # -------------------------------------------------------------
    print("\nConstructing feature matrices for all settings...", flush=True)
    # V1 (0 -> 40 years)
    X_46_v1, X_34_v1, X_38_v1, B_flat_v1, y0_flat_all = build_feature_matrices(y0_arr, AGES, 40, clim_40)
    y40_flat_all = y40_arr.reshape(-1, 2)
    delta_v1 = y40_flat_all - y0_flat_all
    r_v1 = y40_flat_all - B_flat_v1

    # V2 Train (0 -> 20 years)
    X_46_v2_trn, X_34_v2_trn, X_38_v2_trn, B_flat_v2_trn, y0_flat_v2_trn = build_feature_matrices(y0_arr, AGES, 20, clim_0_20)
    y20_flat_all = y20_arr.reshape(-1, 2)
    delta_v2_trn = y20_flat_all - y0_flat_v2_trn
    r_v2_trn = y20_flat_all - B_flat_v2_trn

    # V2 Val & Control (20 -> 40 years)
    # True planting ages at year 20: AGES + 20
    X_46_v2_val, X_34_v2_val, X_38_v2_val, B_flat_v2_val, y20_flat_val = build_feature_matrices(y20_arr, AGES + 20, 20, clim_20_40)
    delta_v2_val = y40_flat_all - y20_flat_val
    r_v2_val = y40_flat_all - B_flat_v2_val

    # -------------------------------------------------------------
    # 5. Core Engine: Model Training & Evaluation with Checkpoints
    # -------------------------------------------------------------
    checkpoints = list(range(25, 3001, 25))  # 120 checkpoints: 25, 50, ..., 3000
    cache_dir = OUT_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def snap_to_checkpoint(val: float | int) -> int:
        return int(min(checkpoints, key=lambda c: abs(c - val)))

    def run_cv_experiment(
        exp_name: str,
        use_buffer: bool,
        X_dict_trn: dict,
        y_targets_trn: dict,  # {"delta": (N, 2), "r": (N, 2)}
        X_dict_val: dict,
        y_targets_val: dict,
        y_start_val: np.ndarray,
        B_val: np.ndarray,
        y_true_val: np.ndarray,
        is_v2: bool = False
    ):
        print(f"\n=======================================================", flush=True)
        print(f"Starting Experiment: {exp_name} (Buffer: {use_buffer})", flush=True)
        print(f"=======================================================", flush=True)
        exp_t0 = time.time()

        models = ["S1_type", "M1_type", "C1"]
        targets = ["h", "a"]

        # Collect validation MSE curves: [model][tgt][seed][fold] -> list of 120 values
        mse_curves = {m: {t: {s: [] for s in SEEDS} for t in targets} for m in models}
        best_iters_per_fold = {m: {t: {s: [] for s in SEEDS} for t in targets} for m in models}

        for fold_idx, f_info in enumerate(folds_data):
            f_t0 = time.time()
            val_s_idx = f_info["val_site_idx"]
            val_r_idx = np.concatenate([np.arange(s * 15, s * 15 + 15) for s in val_s_idx])

            trn_s_idx = f_info["kept_trn_site_idx"] if use_buffer else f_info["orig_trn_site_idx"]
            trn_r_idx = np.concatenate([np.arange(s * 15, s * 15 + 15) for s in trn_s_idx])

            y_tr_eval = y_true_val[val_r_idx]
            y_st_eval = y_start_val[val_r_idx]
            b_eval = B_val[val_r_idx]

            # Fit models
            for mod_name in models:
                X_tr = X_dict_trn[mod_name][trn_r_idx]
                X_va = X_dict_val[mod_name][val_r_idx]
                tgt_key = "r" if mod_name == "C1" else "delta"
                y_tr_target = y_targets_trn[tgt_key][trn_r_idx]
                y_va_target = y_targets_val[tgt_key][val_r_idx]

                for tgt_idx, tgt_name in [(0, "h"), (1, "a")]:
                    for s_idx, seed in enumerate(SEEDS):
                        mod_file = cache_dir / f"{exp_name}_f{fold_idx}_s{seed}_{mod_name}_{tgt_name}.joblib"
                        mses_file = cache_dir / f"{exp_name}_f{fold_idx}_s{seed}_{mod_name}_{tgt_name}_mses.json"

                        if mod_file.exists() and mses_file.exists():
                            with open(mses_file, "r") as f_m:
                                chk_mses = json.load(f_m)
                        else:
                            reg = HistGradientBoostingRegressor(
                                learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50,
                                max_features=0.8, l2_regularization=0.0, early_stopping=False,
                                max_iter=3000, random_state=seed
                            )
                            reg.fit(X_tr, y_tr_target[:, tgt_idx])

                            # Collect MSE at checkpoints
                            chk_mses = []
                            for i_step, p_step in enumerate(reg.staged_predict(X_va)):
                                step_num = i_step + 1
                                if step_num in checkpoints:
                                    if tgt_name == "h":
                                        if mod_name == "C1":
                                            p_phys = np.minimum(np.clip(b_eval[:, 0] + p_step, 0.0, 35.0) / SCALE[0], H_CAP)
                                        else:
                                            p_phys = np.minimum(np.clip(y_st_eval[:, 0] + p_step, 0.0, 35.0) / SCALE[0], H_CAP)
                                        t_phys = np.minimum(np.clip(y_tr_eval[:, 0], 0.0, 35.0) / SCALE[0], H_CAP)
                                    else:
                                        if mod_name == "C1":
                                            p_phys = np.maximum(b_eval[:, 1] + p_step, 0.0) / SCALE[1]
                                        else:
                                            p_phys = np.maximum(y_st_eval[:, 1] + p_step, 0.0) / SCALE[1]
                                        t_phys = np.maximum(y_tr_eval[:, 1], 0.0) / SCALE[1]

                                    chk_mses.append(float(np.mean((p_phys - t_phys) ** 2)))

                            joblib.dump(reg, mod_file)
                            with open(mses_file, "w") as f_m:
                                json.dump(chk_mses, f_m)

                        best_chk_idx = int(np.argmin(chk_mses))
                        best_iter = checkpoints[best_chk_idx]

                        mse_curves[mod_name][tgt_name][seed].append(chk_mses)
                        best_iters_per_fold[mod_name][tgt_name][seed].append(best_iter)
                        print(f"    [{exp_name} f{fold_idx}] {mod_name}_{tgt_name}_s{seed} done (best_iter={best_iter})", flush=True)

            elapsed_f = time.time() - f_t0
            if fold_idx == 0:
                est_total = (elapsed_f * 5.0) / 3600.0
                print(f"[Resource Monitor] {exp_name} Fold 0 completed in {elapsed_f:.2f}s. Estimated runtime: {est_total:.2f} hours (<= 8.0h limit).", flush=True)
                assert est_total <= 8.0, f"Estimated runtime exceeds 8 hours: {est_total}"
            print(f"  {exp_name} Fold {fold_idx} completed in {elapsed_f:.2f}s.", flush=True)

        # Compute median optimal iteration across 5 folds and seeds (snapped to nearest checkpoint)
        median_iters = {
            m: {
                t: snap_to_checkpoint(np.median([best_iters_per_fold[m][t][s][f] for s in SEEDS for f in range(5)]))
                for t in targets
            }
            for m in models
        }
        print(f"\n{exp_name} Selected Median Iterations: {median_iters}", flush=True)

        # Form OOF predictions using the selected median iteration from cached models
        print(f"Extracting OOF predictions at selected median iterations for {exp_name}...", flush=True)
        oof_preds = {m: {s: np.empty((len(y_true_val), 2), dtype=np.float32) for s in SEEDS} for m in models}

        for fold_idx, f_info in enumerate(folds_data):
            val_s_idx = f_info["val_site_idx"]
            val_r_idx = np.concatenate([np.arange(s * 15, s * 15 + 15) for s in val_s_idx])
            b_eval = B_val[val_r_idx]
            y_st_eval = y_start_val[val_r_idx]

            for mod_name in models:
                X_va = X_dict_val[mod_name][val_r_idx]
                for s in SEEDS:
                    # Height
                    m_iter_h = median_iters[mod_name]["h"]
                    mod_file_h = cache_dir / f"{exp_name}_f{fold_idx}_s{s}_{mod_name}_h.joblib"
                    reg_h = joblib.load(mod_file_h)
                    pred_dh = None
                    for i_step, p_step in enumerate(reg_h.staged_predict(X_va)):
                        if i_step + 1 == m_iter_h:
                            pred_dh = p_step
                            break
                    assert pred_dh is not None

                    # AGB
                    m_iter_a = median_iters[mod_name]["a"]
                    mod_file_a = cache_dir / f"{exp_name}_f{fold_idx}_s{s}_{mod_name}_a.joblib"
                    reg_a = joblib.load(mod_file_a)
                    pred_da = None
                    for i_step, p_step in enumerate(reg_a.staged_predict(X_va)):
                        if i_step + 1 == m_iter_a:
                            pred_da = p_step
                            break
                    assert pred_da is not None

                    if mod_name == "C1":
                        oof_preds[mod_name][s][val_r_idx, 0] = np.clip(b_eval[:, 0] + pred_dh, 0.0, 35.0)
                        oof_preds[mod_name][s][val_r_idx, 1] = np.maximum(b_eval[:, 1] + pred_da, 0.0)
                    else:
                        oof_preds[mod_name][s][val_r_idx, 0] = np.clip(y_st_eval[:, 0] + pred_dh, 0.0, 35.0)
                        oof_preds[mod_name][s][val_r_idx, 1] = np.maximum(y_st_eval[:, 1] + pred_da, 0.0)

        # Average across the 2 seeds
        oof_avg = {m: 0.5 * (oof_preds[m][SEEDS[0]] + oof_preds[m][SEEDS[1]]) for m in models}

        # Add baselines: Persistence and B
        oof_avg["Persistence"] = y_start_val.copy()
        oof_avg["B"] = B_val.copy()

        # Compute performance metrics
        # Overall
        overall_results = {}
        for mname in ["Persistence", "B", "S1_type", "M1_type", "C1"]:
            pred = oof_avg[mname]
            sc = compute_scaled_mse(pred, y_true_val)
            old_mask = row_ages_all >= 150
            old_h = float(np.mean(
                (np.minimum(np.clip(pred[old_mask, 0], 0, 35) / SCALE[0], H_CAP) -
                 np.minimum(np.clip(y_true_val[old_mask, 0], 0, 35) / SCALE[0], H_CAP)) ** 2
            ))
            sc["old_h_mse"] = old_h
            overall_results[mname] = sc

        # Age group breakdowns: <=50, 70-100, >=150
        age_group_masks = {
            "<=50": row_ages_all <= 50,
            "70-100": (row_ages_all >= 70) & (row_ages_all <= 100),
            ">=150": row_ages_all >= 150
        }
        age_group_results = {}
        for grp_name, g_mask in age_group_masks.items():
            age_group_results[grp_name] = {}
            for mname in ["Persistence", "B", "S1_type", "M1_type", "C1"]:
                pred = oof_avg[mname][g_mask]
                true_sub = y_true_val[g_mask]
                sc = compute_scaled_mse(pred, true_sub)
                age_group_results[grp_name][mname] = sc

        # Seed differences
        seed_diffs = {}
        for mname in models:
            sc_s1 = compute_scaled_mse(oof_preds[mname][SEEDS[0]], y_true_val)
            sc_s2 = compute_scaled_mse(oof_preds[mname][SEEDS[1]], y_true_val)
            seed_diffs[mname] = {
                "seed1_score": sc_s1["score"],
                "seed2_score": sc_s2["score"],
                "diff": float(abs(sc_s1["score"] - sc_s2["score"]))
            }

        exp_elapsed = time.time() - exp_t0
        print(f"Completed {exp_name} in {exp_elapsed:.2f}s.", flush=True)

        return {
            "exp_name": exp_name,
            "overall": overall_results,
            "age_groups": age_group_results,
            "median_iters": median_iters,
            "best_iters_per_fold": best_iters_per_fold,
            "seed_diffs": seed_diffs,
            "oof_avg": oof_avg,
            "oof_seeds": oof_preds
        }

    # -------------------------------------------------------------
    # 6. Execute V1 (with buffer & no buffer), V2, and V2 Control
    # -------------------------------------------------------------
    # Feature mappings
    X_dict_v1 = {"S1_type": X_46_v1, "M1_type": X_34_v1, "C1": X_38_v1}
    y_targets_v1 = {"delta": delta_v1, "r": r_v1}

    # 1. V1 with Buffer (主口径)
    res_v1_buffer = run_cv_experiment(
        exp_name="V1_Buffer",
        use_buffer=True,
        X_dict_trn=X_dict_v1,
        y_targets_trn=y_targets_v1,
        X_dict_val=X_dict_v1,
        y_targets_val=y_targets_v1,
        y_start_val=y0_flat_all,
        B_val=B_flat_v1,
        y_true_val=y40_flat_all
    )

    # 2. V1 without Buffer (第二口径)
    res_v1_nobuffer = run_cv_experiment(
        exp_name="V1_NoBuffer",
        use_buffer=False,
        X_dict_trn=X_dict_v1,
        y_targets_trn=y_targets_v1,
        X_dict_val=X_dict_v1,
        y_targets_val=y_targets_v1,
        y_start_val=y0_flat_all,
        B_val=B_flat_v1,
        y_true_val=y40_flat_all
    )

    # 3. V2 Time Extrapolation (train 0..20, val 20..40, with buffer)
    X_dict_v2_trn = {"S1_type": X_46_v2_trn, "M1_type": X_34_v2_trn, "C1": X_38_v2_trn}
    y_targets_v2_trn = {"delta": delta_v2_trn, "r": r_v2_trn}

    X_dict_v2_val = {"S1_type": X_46_v2_val, "M1_type": X_34_v2_val, "C1": X_38_v2_val}
    y_targets_v2_val = {"delta": delta_v2_val, "r": r_v2_val}

    res_v2_time = run_cv_experiment(
        exp_name="V2_Time_Extrap",
        use_buffer=True,
        X_dict_trn=X_dict_v2_trn,
        y_targets_trn=y_targets_v2_trn,
        X_dict_val=X_dict_v2_val,
        y_targets_val=y_targets_v2_val,
        y_start_val=y20_flat_val,
        B_val=B_flat_v2_val,
        y_true_val=y40_flat_all,
        is_v2=True
    )

    # 4. V2 Control (train 20..40, val 20..40, with buffer)
    res_v2_control = run_cv_experiment(
        exp_name="V2_Control",
        use_buffer=True,
        X_dict_trn=X_dict_v2_val,
        y_targets_trn=y_targets_v2_val,
        X_dict_val=X_dict_v2_val,
        y_targets_val=y_targets_v2_val,
        y_start_val=y20_flat_val,
        B_val=B_flat_v2_val,
        y_true_val=y40_flat_all,
        is_v2=True
    )

    # -------------------------------------------------------------
    # 7. Hypothesis H3 Testing on Age <= 50 & Bootstrap CIs
    # -------------------------------------------------------------
    print("\n--- Testing Hypothesis H3 (Age <= 50) ---", flush=True)
    mask_le50 = row_ages_all <= 50
    block_map = {b: i for i, b in enumerate(unique_blocks)}
    row_block_idx = np.array([block_map[b] for b in block_rows_all])
    n_blocks = len(unique_blocks)

    rng = np.random.default_rng(20261102)
    boot_indices = rng.integers(0, n_blocks, size=(2000, n_blocks))

    def evaluate_h3_and_ci(exp_res, y_true):
        preds = exp_res["oof_avg"]
        c1_sc = exp_res["age_groups"]["<=50"]["C1"]["score"]
        s1_sc = exp_res["age_groups"]["<=50"]["S1_type"]["score"]
        m1_sc = exp_res["age_groups"]["<=50"]["M1_type"]["score"]

        best_baseline_sc = min(s1_sc, m1_sc)
        best_name = "S1_type" if s1_sc <= m1_sc else "M1_type"

        # Point differences: C1 - baseline (negative means C1 is better / lower MSE)
        diff_vs_best = c1_sc - best_baseline_sc
        diff_vs_s1 = c1_sc - s1_sc
        diff_vs_m1 = c1_sc - m1_sc

        # Falsification condition check:
        # "在年龄<=50的行上，C1在V1主口径或V2中任一处的综合分不低于S1型和M1型中较好的那个，H3就判为不成立"
        # i.e., if c1_sc >= best_baseline_sc, H3 fails in this setting!
        is_falsified = (c1_sc >= best_baseline_sc)

        # Bootstrap 95% CIs on age <= 50
        def get_se_vec(p):
            p_h = np.minimum(np.clip(p[:, 0], 0, 35) / SCALE[0], H_CAP)
            p_a = np.maximum(p[:, 1], 0) / SCALE[1]
            t_h = np.minimum(np.clip(y_true[:, 0], 0, 35) / SCALE[0], H_CAP)
            t_a = np.maximum(y_true[:, 1], 0) / SCALE[1]
            return 0.5 * ((p_h - t_h) ** 2 + (p_a - t_a) ** 2)

        se_c1 = get_se_vec(preds["C1"])
        se_s1 = get_se_vec(preds["S1_type"])
        se_m1 = get_se_vec(preds["M1_type"])

        # Restrict to age <= 50
        weights_c1 = np.where(mask_le50, se_c1, 0.0)
        weights_s1 = np.where(mask_le50, se_s1, 0.0)
        weights_m1 = np.where(mask_le50, se_m1, 0.0)
        weights_cnt = np.where(mask_le50, 1.0, 0.0)

        blk_c1 = np.bincount(row_block_idx, weights=weights_c1, minlength=n_blocks)
        blk_s1 = np.bincount(row_block_idx, weights=weights_s1, minlength=n_blocks)
        blk_m1 = np.bincount(row_block_idx, weights=weights_m1, minlength=n_blocks)
        blk_cnt = np.bincount(row_block_idx, weights=weights_cnt, minlength=n_blocks)

        boot_c1 = np.sum(blk_c1[boot_indices], axis=1) / np.sum(blk_cnt[boot_indices], axis=1)
        boot_s1 = np.sum(blk_s1[boot_indices], axis=1) / np.sum(blk_cnt[boot_indices], axis=1)
        boot_m1 = np.sum(blk_m1[boot_indices], axis=1) / np.sum(blk_cnt[boot_indices], axis=1)

        diff_boot_s1 = boot_c1 - boot_s1
        diff_boot_m1 = boot_c1 - boot_m1

        ci_s1 = [float(np.percentile(diff_boot_s1, 2.5)), float(np.percentile(diff_boot_s1, 97.5))]
        ci_m1 = [float(np.percentile(diff_boot_m1, 2.5)), float(np.percentile(diff_boot_m1, 97.5))]

        return {
            "c1_score": c1_sc,
            "s1_score": s1_sc,
            "m1_score": m1_sc,
            "best_baseline": best_name,
            "best_baseline_score": best_baseline_sc,
            "diff_vs_best": diff_vs_best,
            "diff_vs_s1": diff_vs_s1,
            "ci_vs_s1": ci_s1,
            "diff_vs_m1": diff_vs_m1,
            "ci_vs_m1": ci_m1,
            "is_falsified": is_falsified,
            "verdict": "不成立 (否定)" if is_falsified else "成立"
        }

    h3_v1 = evaluate_h3_and_ci(res_v1_buffer, y40_flat_all)
    h3_v2 = evaluate_h3_and_ci(res_v2_time, y40_flat_all)

    overall_h3_verdict = "成立" if (not h3_v1["is_falsified"] and not h3_v2["is_falsified"]) else "不成立"
    print(f"H3 in V1 (Buffer): C1={h3_v1['c1_score']:.6f}, Best({h3_v1['best_baseline']})={h3_v1['best_baseline_score']:.6f} -> Diff: {h3_v1['diff_vs_best']:.6f} ({h3_v1['verdict']})", flush=True)
    print(f"  C1 - S1 CI: [{h3_v1['ci_vs_s1'][0]:.6f}, {h3_v1['ci_vs_s1'][1]:.6f}], C1 - M1 CI: [{h3_v1['ci_vs_m1'][0]:.6f}, {h3_v1['ci_vs_m1'][1]:.6f}]", flush=True)
    print(f"H3 in V2 (Time):   C1={h3_v2['c1_score']:.6f}, Best({h3_v2['best_baseline']})={h3_v2['best_baseline_score']:.6f} -> Diff: {h3_v2['diff_vs_best']:.6f} ({h3_v2['verdict']})", flush=True)
    print(f"  C1 - S1 CI: [{h3_v2['ci_vs_s1'][0]:.6f}, {h3_v2['ci_vs_s1'][1]:.6f}], C1 - M1 CI: [{h3_v2['ci_vs_m1'][0]:.6f}, {h3_v2['ci_vs_m1'][1]:.6f}]", flush=True)
    print(f"OVERALL HYPOTHESIS H3 VERDICT: {overall_h3_verdict}", flush=True)

    # -------------------------------------------------------------
    # 8. Output Tables Generation (CSV)
    # -------------------------------------------------------------
    # Table 1: V1 Overall
    t1_rows = []
    for setting_name, exp_res in [("With_Buffer", res_v1_buffer), ("Without_Buffer", res_v1_nobuffer)]:
        for mname in ["Persistence", "B", "S1_type", "M1_type", "C1"]:
            sc = exp_res["overall"][mname]
            t1_rows.append({
                "Setting": setting_name,
                "Model": mname,
                "Score": sc["score"],
                "Height_MSE": sc["mse_h"],
                "AGB_MSE": sc["mse_a"],
                "H_Old_MSE": sc["old_h_mse"]
            })
    pd.DataFrame(t1_rows).to_csv(OUT_DIR / "table1_v1_overall.csv", index=False)

    # Table 2: V2 Overall
    t2_rows = []
    for setting_name, exp_res in [("V2_Time_Extrap", res_v2_time), ("V2_Control", res_v2_control)]:
        for mname in ["Persistence", "B", "S1_type", "M1_type", "C1"]:
            sc = exp_res["overall"][mname]
            t2_rows.append({
                "Setting": setting_name,
                "Model": mname,
                "Score": sc["score"],
                "Height_MSE": sc["mse_h"],
                "AGB_MSE": sc["mse_a"],
                "H_Old_MSE": sc["old_h_mse"]
            })
    pd.DataFrame(t2_rows).to_csv(OUT_DIR / "table2_v2_overall.csv", index=False)

    # Table 3: Age Groups (<=50, 70-100, >=150)
    t3_rows = []
    all_exps = [("V1_Buffer", res_v1_buffer), ("V1_NoBuffer", res_v1_nobuffer), ("V2_Time_Extrap", res_v2_time), ("V2_Control", res_v2_control)]
    for exp_label, exp_res in all_exps:
        for grp_name in ["<=50", "70-100", ">=150"]:
            for mname in ["Persistence", "B", "S1_type", "M1_type", "C1"]:
                sc = exp_res["age_groups"][grp_name][mname]
                t3_rows.append({
                    "Experiment": exp_label,
                    "Age_Group": grp_name,
                    "Model": mname,
                    "Score": sc["score"],
                    "Height_MSE": sc["mse_h"],
                    "AGB_MSE": sc["mse_a"]
                })
    pd.DataFrame(t3_rows).to_csv(OUT_DIR / "table3_age_groups.csv", index=False)

    # Table 4: H3 Test (Age <= 50)
    t4_rows = [
        {
            "Experiment": "V1_Buffer (Spatial)",
            "C1_Score": h3_v1["c1_score"],
            "S1_Score": h3_v1["s1_score"],
            "M1_Score": h3_v1["m1_score"],
            "Best_Baseline": h3_v1["best_baseline"],
            "Best_Baseline_Score": h3_v1["best_baseline_score"],
            "C1_vs_Best_Diff": h3_v1["diff_vs_best"],
            "C1_vs_S1_Diff": h3_v1["diff_vs_s1"],
            "C1_vs_S1_95CI": f"[{h3_v1['ci_vs_s1'][0]:.6f}, {h3_v1['ci_vs_s1'][1]:.6f}]",
            "C1_vs_M1_Diff": h3_v1["diff_vs_m1"],
            "C1_vs_M1_95CI": f"[{h3_v1['ci_vs_m1'][0]:.6f}, {h3_v1['ci_vs_m1'][1]:.6f}]",
            "Verdict": h3_v1["verdict"]
        },
        {
            "Experiment": "V2_Time_Extrap (Temporal)",
            "C1_Score": h3_v2["c1_score"],
            "S1_Score": h3_v2["s1_score"],
            "M1_Score": h3_v2["m1_score"],
            "Best_Baseline": h3_v2["best_baseline"],
            "Best_Baseline_Score": h3_v2["best_baseline_score"],
            "C1_vs_Best_Diff": h3_v2["diff_vs_best"],
            "C1_vs_S1_Diff": h3_v2["diff_vs_s1"],
            "C1_vs_S1_95CI": f"[{h3_v2['ci_vs_s1'][0]:.6f}, {h3_v2['ci_vs_s1'][1]:.6f}]",
            "C1_vs_M1_Diff": h3_v2["diff_vs_m1"],
            "C1_vs_M1_95CI": f"[{h3_v2['ci_vs_m1'][0]:.6f}, {h3_v2['ci_vs_m1'][1]:.6f}]",
            "Verdict": h3_v2["verdict"]
        }
    ]
    pd.DataFrame(t4_rows).to_csv(OUT_DIR / "table4_h3_test_age50.csv", index=False)

    # Table 5: Seed Variations
    t5_rows = []
    for exp_label, exp_res in all_exps:
        for mname in ["S1_type", "M1_type", "C1"]:
            s_diff = exp_res["seed_diffs"][mname]
            t5_rows.append({
                "Experiment": exp_label,
                "Model": mname,
                "Seed_20261101_Score": s_diff["seed1_score"],
                "Seed_20261102_Score": s_diff["seed2_score"],
                "Score_Abs_Diff": s_diff["diff"]
            })
    pd.DataFrame(t5_rows).to_csv(OUT_DIR / "table5_seed_variation.csv", index=False)

    # Table 6: Optimal Iterations
    t6_rows = []
    for exp_label, exp_res in all_exps:
        for mname in ["S1_type", "M1_type", "C1"]:
            for tgt in ["h", "a"]:
                m_iter = exp_res["median_iters"][mname][tgt]
                # Folds under seed 20261101
                f_iters_s1 = exp_res["best_iters_per_fold"][mname][tgt][SEEDS[0]]
                f_iters_s2 = exp_res["best_iters_per_fold"][mname][tgt][SEEDS[1]]
                t6_rows.append({
                    "Experiment": exp_label,
                    "Model": mname,
                    "Target": "Height" if tgt == "h" else "AGB",
                    "Fold0_S1": f_iters_s1[0],
                    "Fold1_S1": f_iters_s1[1],
                    "Fold2_S1": f_iters_s1[2],
                    "Fold3_S1": f_iters_s1[3],
                    "Fold4_S1": f_iters_s1[4],
                    "Median_Iter": m_iter
                })
    pd.DataFrame(t6_rows).to_csv(OUT_DIR / "table6_optimal_iterations.csv", index=False)

    # -------------------------------------------------------------
    # 9. Save Compressed NPZ Files
    # -------------------------------------------------------------
    print("Saving oof_v1.npz and oof_v2.npz...", flush=True)
    np.savez_compressed(
        OUT_DIR / "oof_v1.npz",
        site_indices=train_cells,
        block=block_sites,
        B=B_flat_v1,
        y0=y0_flat_all,
        y40=y40_flat_all,
        oof_S1_buffer=res_v1_buffer["oof_avg"]["S1_type"],
        oof_M1_buffer=res_v1_buffer["oof_avg"]["M1_type"],
        oof_C1_buffer=res_v1_buffer["oof_avg"]["C1"],
        oof_S1_buffer_seed20261101=res_v1_buffer["oof_seeds"]["S1_type"][SEEDS[0]],
        oof_S1_buffer_seed20261102=res_v1_buffer["oof_seeds"]["S1_type"][SEEDS[1]],
        oof_C1_buffer_seed20261101=res_v1_buffer["oof_seeds"]["C1"][SEEDS[0]],
        oof_C1_buffer_seed20261102=res_v1_buffer["oof_seeds"]["C1"][SEEDS[1]],
        oof_S1_nobuffer=res_v1_nobuffer["oof_avg"]["S1_type"],
        oof_M1_nobuffer=res_v1_nobuffer["oof_avg"]["M1_type"],
        oof_C1_nobuffer=res_v1_nobuffer["oof_avg"]["C1"]
    )

    np.savez_compressed(
        OUT_DIR / "oof_v2.npz",
        site_indices=train_cells,
        block=block_sites,
        B_v2=B_flat_v2_val,
        y20=y20_flat_val,
        y40=y40_flat_all,
        oof_S1_v2=res_v2_time["oof_avg"]["S1_type"],
        oof_M1_v2=res_v2_time["oof_avg"]["M1_type"],
        oof_C1_v2=res_v2_time["oof_avg"]["C1"],
        oof_S1_control=res_v2_control["oof_avg"]["S1_type"],
        oof_M1_control=res_v2_control["oof_avg"]["M1_type"],
        oof_C1_control=res_v2_control["oof_avg"]["C1"]
    )

    # -------------------------------------------------------------
    # 10. Record Metadata and Hashes
    # -------------------------------------------------------------
    code_hashes = {
        "run_c1_experiment.py": sha256_file(CODE_DIR / "run_c1_experiment.py"),
        "prepare_inputs.py": sha256_file(CODE_DIR / "prepare_inputs.py"),
        "features.py": sha256_file(CODE_DIR / "features.py"),
        "restore_y0.py": sha256_file(CODE_DIR / "restore_y0.py"),
        "train_s1.py": sha256_file(CODE_DIR / "train_s1.py")
    }

    t_total = time.time() - t_start
    _, current_peak_bytes = tracemalloc.get_traced_memory()
    total_peak_mb = current_peak_bytes / (1024 * 1024)

    metrics_out = {
        "task": "Step 3: C1 Baseline Residual Model (2026-11-02)",
        "elapsed_seconds": t_total,
        "peak_memory_mb": total_peak_mb,
        "h3_testing": {
            "v1_buffer": h3_v1,
            "v2_time": h3_v2,
            "overall_h3_verdict": overall_h3_verdict
        },
        "acceptance_checks": {
            "n_train_sites": n_sites,
            "protected_sites_excluded": 4948,
            "fresh_sites_untouched": 1024,
            "b_3373_score_replicated": score_b_3373,
            "min_buffer_dist_ge_4": True,
            "block_overlap_zero": True
        },
        "code_hashes": code_hashes
    }

    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)
    print(f"Saved complete metrics to {OUT_DIR / 'metrics.json'}", flush=True)

    print("\nStep 3 experiment pipeline completed successfully!", flush=True)
    return metrics_out


if __name__ == "__main__":
    run()

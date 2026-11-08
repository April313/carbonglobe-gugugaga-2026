"""V6 Batch 3: SSP Predictions for S1-type and C3 Models (2026-11-08)

Trains S1-type directly regressing Delta, and C3 collapse decomposition model
(p probability + mu0 uncollapsed regression + mu1 collapsed regression)
on ALL 49,204 historical training sites across 0->40 window.
Generates two official SSP submission CSV files:
- candidate_V6_6_S1type_full.csv
- candidate_V6_7_C3_full.csv

Strict Constraints:
- Pure local offline model fitting and SSP prediction file generation.
- No upload, no Kaggle API calls, no public LB reading, no credentials reading.
- Strictly 49,204 historical training sites. Strictly isolate 4,948 protected sites.
- User accepted original test split label risk on 2026-11-02.
- Mandatory report sentence: "训练使用了原test划分站点的标签，主办方未确认。"
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import tracemalloc

# Setup paths and environment before importing third-party libraries
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path("data/project")
if str(ROOT / ".runtime/science-cp314") not in sys.path:
    sys.path.insert(0, str(ROOT / ".runtime/science-cp314"))
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from src import kaggle_task as kt

from features import H_CAP, SCALE, SEED_AGES, extract_climate_features, extract_all_features
from restore_y0 import restore_initial_y0

OUT_DIR = ROOT / "results/v6_c3_ssp_20261108"
CACHE_DIR = OUT_DIR / "cache"
MODELS_DIR = OUT_DIR / "models"

DATA_GLOBAL_DIR = Path("data/dataset/data_global")
DATA_STATS_DIR = Path("data/dataset/data_stats")
MANIFEST_PATH = ROOT / "results/competition_specific_plan_20261013/site_manifests.npz"

KAGGLE_DIR = Path("data/kaggle")
SAMPLE_CSV_PATH = KAGGLE_DIR / "sample_submission.csv"
G1_CSV_PATH = ROOT / "results/full_sites_20261102/candidate_G1_full_delta.csv"

SEEDS = [20261101, 20261102]
AGES = np.array(SEED_AGES, dtype=np.int32)
REGRESSOR_CHECKPOINTS = list(range(25, 3001, 25))   # 120 checkpoints
CLASSIFIER_CHECKPOINTS = list(range(25, 2001, 25))  # 80 checkpoints


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_window_features(y_start: np.ndarray, k: int, clim_summary: np.ndarray):
    """Constructs 46-dim feature matrix for training window starting at year k.
    y_start: (N, 15, 2)
    k: starting year of window (real age is SEED_AGES + k)
    clim_summary: (N, 12) summary of climate drivers within this window
    Returns:
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

    # 4. Climate features
    clim_grid = np.broadcast_to(clim_summary[:, None, :], (N, num_ages, 12))  # (N, 15, 12)
    feats_46_grid = np.concatenate([feats_34_grid, clim_grid], axis=-1)       # (N, 15, 46)
    X_46 = feats_46_grid.reshape(N * num_ages, 46).astype(np.float32)

    return X_46


def build_submission_csv(
    sample_ids: list[str],
    pred_ssp126: np.ndarray,
    pred_ssp585: np.ndarray,
    out_csv: Path,
):
    """Formats and writes test predictions to Kaggle submission CSV format.
    pred_ssp126, pred_ssp585: (92565, 2) physical clipped predictions [height, agb].
    """
    pred_all = np.vstack([pred_ssp126, pred_ssp585])  # (185130, 2)
    assert len(sample_ids) == len(pred_all), f"Length mismatch: {len(sample_ids)} vs {len(pred_all)}"

    h_phys = np.clip(pred_all[:, 0], 0.0, 35.0)
    a_phys = np.maximum(pred_all[:, 1], 0.0)

    h_scaled = np.minimum(h_phys / SCALE[0], H_CAP)
    a_scaled = a_phys / SCALE[1]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "height", "agb"])
        for row_id, h_val, a_val in zip(sample_ids, h_scaled, a_scaled):
            writer.writerow([row_id, f"{h_val:.6f}", f"{a_val:.6f}"])


def compute_scaled_pred(pred_phys: np.ndarray) -> np.ndarray:
    """Takes physical prediction (N, 2), applies clipping and scaling, returns (N, 2)."""
    h_scaled = np.minimum(np.clip(pred_phys[:, 0], 0.0, 35.0) / SCALE[0], H_CAP)
    a_scaled = np.maximum(pred_phys[:, 1], 0.0) / SCALE[1]
    return np.column_stack([h_scaled, a_scaled])


def main():
    t_start = time.time()
    tracemalloc.start()
    print("=== Antigravity Task: V6 Batch 3 SSP Predictions (2026-11-08) ===", flush=True)

    if (OUT_DIR / "metrics.json").exists():
        raise RuntimeError(f"Output directory {OUT_DIR} already contains metrics.json (completed run). Aborting per task specification.")
    if OUT_DIR.exists():
        print(f"Output directory {OUT_DIR} exists without metrics.json. Resuming execution using cached models...", flush=True)
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=False)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Site Manifest & Data Isolation Verification
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

    block_sites = (row_sites // 20) * 1000 + (col_sites // 20)
    unique_blocks = np.unique(block_sites)
    print(f"Identified {len(unique_blocks)} unique 10-degree spatial blocks across {n_sites} sites.", flush=True)

    # -------------------------------------------------------------
    # 2. Load December Annual Trajectories & Climate Summaries (0->40)
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

    print("Extracting climate drivers for 0..40 window...", flush=True)
    clim_0_40 = np.empty((n_sites, 12), dtype=np.float32)

    x_raw = np.load(DATA_GLOBAL_DIR / "glob_X_fea.npy", mmap_mode="r")
    for i in range(0, n_sites, 4000):
        idx = train_cells[i:i + 4000]
        sub_X = np.asarray(x_raw[idx], dtype=np.float32)
        clim_0_40[i:i + len(idx)] = extract_climate_features(sub_X[:, 0:40])
    del x_raw

    # -------------------------------------------------------------
    # 3. Compute Labels and Physical Targets (0->40)
    # -------------------------------------------------------------
    print("Computing collapse labels and Delta targets...", flush=True)
    Y_flat = dec_y_all.reshape(n_sites * 15, 41, 2)
    z_h = Y_flat[:, :, 0] / SCALE[0]
    z_a = Y_flat[:, :, 1] / SCALE[1]

    diff_h = z_h[:, 1:41] - z_h[:, 0:40]
    diff_a = z_a[:, 1:41] - z_a[:, 0:40]
    m_h = np.min(diff_h, axis=1)
    m_a = np.min(diff_a, axis=1)
    c20_h = (m_h <= -0.2).astype(np.int32)
    c20_a = (m_a <= -0.2).astype(np.int32)

    y0_phys = dec_y_all[:, :, 0, :2].reshape(-1, 2)   # (49204 * 15, 2)
    y40_phys = dec_y_all[:, :, 40, :2].reshape(-1, 2) # (49204 * 15, 2)
    delta_phys = y40_phys - y0_phys

    print(f"Historical collapse rates: C20_h = {np.mean(c20_h)*100:.2f}%, C20_a = {np.mean(c20_a)*100:.2f}%", flush=True)

    # -------------------------------------------------------------
    # 4. Build 46-dim Training Features
    # -------------------------------------------------------------
    print("Constructing 46-dimensional feature matrix on 49,204 historical sites...", flush=True)
    X_train_46 = build_window_features(dec_y_all[:, :, 0, :2], 0, clim_0_40)
    assert X_train_46.shape == (n_sites * 15, 46), f"Shape mismatch: {X_train_46.shape}"

    # -------------------------------------------------------------
    # 5. Inner 3-Fold GroupKFold Selection on ALL 49,204 sites
    # -------------------------------------------------------------
    print("\nPreparing Inner 3-Fold GroupKFold across all 49,204 sites (grouped by 10° blocks)...", flush=True)
    inner_gkf = GroupKFold(n_splits=3)
    inner_splits = list(inner_gkf.split(np.arange(n_sites), groups=block_sites))

    # Assert 0 block overlap between folds
    for inf_idx, (in_trn_sites, in_val_sites) in enumerate(inner_splits):
        trn_b = np.unique(block_sites[in_trn_sites])
        val_b = np.unique(block_sites[in_val_sites])
        assert len(np.intersect1d(trn_b, val_b)) == 0, f"Block overlap in inner fold {inf_idx}!"

    target_defs = {
        "h": {"col": 0, "c20": c20_h, "label_name": "C20_h", "scale": SCALE[0]},
        "a": {"col": 1, "c20": c20_a, "label_name": "C20_a", "scale": SCALE[1]}
    }

    models_to_train = ["s1", "clf", "mu0", "mu1"]
    selected_iterations = {}
    inner_iter_records = []

    for tgt in ["h", "a"]:
        t_col = target_defs[tgt]["col"]
        c20_tgt = target_defs[tgt]["c20"]
        delta_tgt = delta_phys[:, t_col]
        y0_tgt = y0_phys[:, t_col]
        y40_tgt = y40_phys[:, t_col]

        for mod_code in models_to_train:
            for s in SEEDS:
                iter_cache_file = CACHE_DIR / f"selected_iter_{tgt}_{mod_code}_s{s}.json"

                if iter_cache_file.exists():
                    with open(iter_cache_file, "r") as f_it:
                        it_info = json.load(f_it)
                    inner_iter_records.append(it_info)
                    selected_iterations[(tgt, mod_code, s)] = it_info["Selected_Median_Iter"]
                    print(f"[Cache Hit] Iterations for {tgt}_{mod_code}_s{s}: {it_info['Selected_Median_Iter']}", flush=True)
                else:
                    inner_bests = []
                    for in_f_idx, (in_trn_sites, in_val_sites) in enumerate(inner_splits):
                        in_trn_rows = (in_trn_sites[:, None] * 15 + np.arange(15)[None, :]).ravel()
                        in_val_rows = (in_val_sites[:, None] * 15 + np.arange(15)[None, :]).ravel()

                        in_fold_cache = CACHE_DIR / f"inner_mses_inf{in_f_idx}_{tgt}_{mod_code}_s{s}.json"
                        if in_fold_cache.exists():
                            with open(in_fold_cache, "r") as f_inf:
                                in_losses = json.load(f_inf)
                        else:
                            if mod_code == "clf":
                                clf_in = HistGradientBoostingClassifier(
                                    learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
                                    max_features=0.8, l2_regularization=0.0, early_stopping=False,
                                    max_iter=2000, random_state=s
                                )
                                clf_in.fit(X_train_46[in_trn_rows], c20_tgt[in_trn_rows])

                                in_losses = []
                                for step_idx, p_step in enumerate(clf_in.staged_predict_proba(X_train_46[in_val_rows])):
                                    step_num = step_idx + 1
                                    if step_num in CLASSIFIER_CHECKPOINTS:
                                        prob_step = np.clip(p_step[:, 1], 1e-15, 1.0 - 1e-15)
                                        loss_val = log_loss(c20_tgt[in_val_rows], prob_step)
                                        in_losses.append((step_num, loss_val))

                            else:
                                if mod_code == "s1":
                                    in_trn_sub = np.ones(len(in_trn_rows), dtype=bool)
                                    in_val_sub = np.ones(len(in_val_rows), dtype=bool)
                                elif mod_code == "mu0":
                                    in_trn_sub = (c20_tgt[in_trn_rows] == 0)
                                    in_val_sub = (c20_tgt[in_val_rows] == 0)
                                else:  # mu1
                                    in_trn_sub = (c20_tgt[in_trn_rows] == 1)
                                    in_val_sub = (c20_tgt[in_val_rows] == 1)

                                X_in_trn = X_train_46[in_trn_rows[in_trn_sub]]
                                y_in_trn = delta_tgt[in_trn_rows[in_trn_sub]]

                                X_in_val = X_train_46[in_val_rows[in_val_sub]]
                                y0_in_val = y0_tgt[in_val_rows[in_val_sub]]
                                y40_in_val = y40_tgt[in_val_rows[in_val_sub]]

                                assert len(X_in_trn) > 0 and len(X_in_val) > 0, "Empty inner training or validation subset!"

                                reg_in = HistGradientBoostingRegressor(
                                    learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50,
                                    max_features=0.8, l2_regularization=0.0, early_stopping=False,
                                    max_iter=3000, random_state=s
                                )
                                reg_in.fit(X_in_trn, y_in_trn)

                                in_losses = []
                                for step_idx, p_step in enumerate(reg_in.staged_predict(X_in_val)):
                                    step_num = step_idx + 1
                                    if step_num in REGRESSOR_CHECKPOINTS:
                                        if tgt == "h":
                                            p_phys = np.minimum(np.clip(y0_in_val + p_step, 0.0, 35.0) / SCALE[0], H_CAP)
                                            t_phys = np.minimum(np.clip(y40_in_val, 0.0, 35.0) / SCALE[0], H_CAP)
                                        else:
                                            p_phys = np.maximum(y0_in_val + p_step, 0.0) / SCALE[1]
                                            t_phys = np.maximum(y40_in_val, 0.0) / SCALE[1]
                                        in_losses.append((step_num, float(np.mean((p_phys - t_phys) ** 2))))

                            with open(in_fold_cache, "w") as f_inf:
                                json.dump(in_losses, f_inf)

                        in_losses.sort(key=lambda x: x[1])
                        inner_bests.append(in_losses[0][0])

                    chk_list = CLASSIFIER_CHECKPOINTS if mod_code == "clf" else REGRESSOR_CHECKPOINTS
                    med_iter = int(min(chk_list, key=lambda c: abs(c - np.median(inner_bests))))
                    it_info = {
                        "Target": tgt,
                        "Model": mod_code,
                        "Seed": s,
                        "InnerFold0_Best": inner_bests[0],
                        "InnerFold1_Best": inner_bests[1],
                        "InnerFold2_Best": inner_bests[2],
                        "Selected_Median_Iter": med_iter
                    }
                    with open(iter_cache_file, "w") as f_it:
                        json.dump(it_info, f_it, indent=2)
                    inner_iter_records.append(it_info)
                    selected_iterations[(tgt, mod_code, s)] = med_iter
                    print(f"Selected iterations for {tgt}_{mod_code}_s{s}: {inner_bests} -> Median: {med_iter}", flush=True)

    pd.DataFrame(inner_iter_records).to_csv(OUT_DIR / "table_selected_iterations.csv", index=False)

    # -------------------------------------------------------------
    # 6. Full Training on ALL 49,204 sites
    # -------------------------------------------------------------
    print("\n[Step 6] Full Retraining on ALL 49,204 sites...", flush=True)
    trained_models = {}

    for tgt in ["h", "a"]:
        t_col = target_defs[tgt]["col"]
        c20_tgt = target_defs[tgt]["c20"]
        delta_tgt = delta_phys[:, t_col]

        for mod_code in models_to_train:
            for s in SEEDS:
                med_iter = selected_iterations[(tgt, mod_code, s)]
                mod_file = MODELS_DIR / f"model_{tgt}_{mod_code}_s{s}.joblib"

                if mod_file.exists():
                    print(f"[Model Cache Hit] Loading {mod_file.name}...", flush=True)
                    model_obj = joblib.load(mod_file)
                else:
                    t_fit_0 = time.time()
                    if mod_code == "clf":
                        clf_full = HistGradientBoostingClassifier(
                            learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
                            max_features=0.8, l2_regularization=0.0, early_stopping=False,
                            max_iter=med_iter, random_state=s
                        )
                        clf_full.fit(X_train_46, c20_tgt)
                        model_obj = clf_full
                    else:
                        if mod_code == "s1":
                            mask_sub = np.ones(len(c20_tgt), dtype=bool)
                        elif mod_code == "mu0":
                            mask_sub = (c20_tgt == 0)
                        else:  # mu1
                            mask_sub = (c20_tgt == 1)

                        reg_full = HistGradientBoostingRegressor(
                            learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50,
                            max_features=0.8, l2_regularization=0.0, early_stopping=False,
                            max_iter=med_iter, random_state=s
                        )
                        reg_full.fit(X_train_46[mask_sub], delta_tgt[mask_sub])
                        model_obj = reg_full

                    joblib.dump(model_obj, mod_file)
                    print(f"Fitted & saved {mod_file.name} (max_iter={med_iter}) in {time.time() - t_fit_0:.1f}s", flush=True)

                trained_models[(tgt, mod_code, s)] = model_obj

    # -------------------------------------------------------------
    # 7. SSP Predictions on SSP126 & SSP585
    # -------------------------------------------------------------
    print("\n[Step 7] Running Inference on SSP126 and SSP585...", flush=True)
    sample_df = pd.read_csv(SAMPLE_CSV_PATH)
    sample_ids = sample_df["id"].tolist()
    assert len(sample_ids) == 185130, f"Sample submission rows: {len(sample_ids)} != 185130"

    ssp_data = {}
    for scen in ["ssp126", "ssp585"]:
        npz_p = KAGGLE_DIR / f"test_{scen}.npz"
        with np.load(npz_p) as z:
            t_x = z["test_x"]
            t_y0 = z["test_y0"]

        y0_res = restore_initial_y0(t_y0, clip_height=True, clip_agb=True) # (6171, 15, 2)
        X_ssp = extract_all_features(y0_res, t_x).astype(np.float32)       # (92565, 46)
        y0_ssp_flat = y0_res.reshape(-1, 2)                                # (92565, 2)

        ssp_data[scen] = {
            "y0_flat": y0_ssp_flat,
            "X": X_ssp,
            "n_rows": len(y0_ssp_flat)
        }

    # Generate Predictions for S1-type (V6-6) and C3 (V6-7)
    ssp_preds = {
        "V6_6_S1type": {"ssp126": None, "ssp585": None},
        "V6_7_C3": {"ssp126": None, "ssp585": None}
    }
    ssp_p_values = {"ssp126": {"h": None, "a": None}, "ssp585": {"h": None, "a": None}}

    for scen in ["ssp126", "ssp585"]:
        X_scen = ssp_data[scen]["X"]
        y0_scen = ssp_data[scen]["y0_flat"]

        # Seeds predictions
        s1_h_seeds = []
        s1_a_seeds = []
        c3_h_seeds = []
        c3_a_seeds = []
        p_h_seeds = []
        p_a_seeds = []

        for s in SEEDS:
            # S1
            dh_s1 = trained_models[("h", "s1", s)].predict(X_scen)
            da_s1 = trained_models[("a", "s1", s)].predict(X_scen)
            pred_s1_h = np.clip(y0_scen[:, 0] + dh_s1, 0.0, 35.0)
            pred_s1_a = np.maximum(y0_scen[:, 1] + da_s1, 0.0)
            s1_h_seeds.append(pred_s1_h)
            s1_a_seeds.append(pred_s1_a)

            # C3
            p_h = trained_models[("h", "clf", s)].predict_proba(X_scen)[:, 1]
            p_a = trained_models[("a", "clf", s)].predict_proba(X_scen)[:, 1]
            mu0_h = trained_models[("h", "mu0", s)].predict(X_scen)
            mu1_h = trained_models[("h", "mu1", s)].predict(X_scen)
            mu0_a = trained_models[("a", "mu0", s)].predict(X_scen)
            mu1_a = trained_models[("a", "mu1", s)].predict(X_scen)

            # Notice: "不缩放崩塌概率 p" (multiplier = 1.0)
            dh_c3 = (1.0 - p_h) * mu0_h + p_h * mu1_h
            da_c3 = (1.0 - p_a) * mu0_a + p_a * mu1_a

            pred_c3_h = np.clip(y0_scen[:, 0] + dh_c3, 0.0, 35.0)
            pred_c3_a = np.maximum(y0_scen[:, 1] + da_c3, 0.0)
            c3_h_seeds.append(pred_c3_h)
            c3_a_seeds.append(pred_c3_a)

            p_h_seeds.append(p_h)
            p_a_seeds.append(p_a)

        # Ensemble average across the 2 seeds
        ssp_preds["V6_6_S1type"][scen] = np.column_stack([
            0.5 * (s1_h_seeds[0] + s1_h_seeds[1]),
            0.5 * (s1_a_seeds[0] + s1_a_seeds[1])
        ])
        ssp_preds["V6_7_C3"][scen] = np.column_stack([
            0.5 * (c3_h_seeds[0] + c3_h_seeds[1]),
            0.5 * (c3_a_seeds[0] + c3_a_seeds[1])
        ])
        ssp_p_values[scen]["h"] = 0.5 * (p_h_seeds[0] + p_h_seeds[1])
        ssp_p_values[scen]["a"] = 0.5 * (p_a_seeds[0] + p_a_seeds[1])

    # -------------------------------------------------------------
    # 8. Write Submission CSV Files
    # -------------------------------------------------------------
    csv_paths = {
        "V6_6_S1type": OUT_DIR / "candidate_V6_6_S1type_full.csv",
        "V6_7_C3": OUT_DIR / "candidate_V6_7_C3_full.csv"
    }

    for model_key, out_path in csv_paths.items():
        build_submission_csv(
            sample_ids=sample_ids,
            pred_ssp126=ssp_preds[model_key]["ssp126"],
            pred_ssp585=ssp_preds[model_key]["ssp585"],
            out_csv=out_path
        )
        print(f"Generated {out_path.name} (SHA256: {sha256_file(out_path)})", flush=True)

    # -------------------------------------------------------------
    # 9. Acceptance Criterion 1: kt.check_submission_csv
    # -------------------------------------------------------------
    print("\nChecking Acceptance Criterion 1: kt.check_submission_csv...", flush=True)
    check_v6_6 = kt.check_submission_csv(csv_paths["V6_6_S1type"], SAMPLE_CSV_PATH)
    check_v6_7 = kt.check_submission_csv(csv_paths["V6_7_C3"], SAMPLE_CSV_PATH)

    assert check_v6_6["rows_match_185130"], "V6_6 row count mismatch!"
    assert check_v6_6["ids_match_sample_order"], "V6_6 ID order mismatch!"
    assert check_v6_6["no_nan_inf"], "V6_6 contains NaN/Inf!"
    assert check_v6_6["scale_sanity"], "V6_6 scale sanity check failed!"

    assert check_v6_7["rows_match_185130"], "V6_7 row count mismatch!"
    assert check_v6_7["ids_match_sample_order"], "V6_7 ID order mismatch!"
    assert check_v6_7["no_nan_inf"], "V6_7 contains NaN/Inf!"
    assert check_v6_7["scale_sanity"], "V6_7 scale sanity check failed!"
    print("Acceptance Criterion 1 PASSED: Both submission files validated cleanly against sample submission.\n", flush=True)

    # -------------------------------------------------------------
    # 10. Acceptance Criterion 2: Reload Models Determinism Verification
    # -------------------------------------------------------------
    print("Checking Acceptance Criterion 2: Reloading models from disk to verify deterministic SSP reproduction...", flush=True)
    # Clear in-memory models
    del trained_models
    gc.collect()

    reloaded_models = {}
    for tgt in ["h", "a"]:
        for mod_code in models_to_train:
            for s in SEEDS:
                mod_file = MODELS_DIR / f"model_{tgt}_{mod_code}_s{s}.joblib"
                reloaded_models[(tgt, mod_code, s)] = joblib.load(mod_file)

    max_diff_s1 = 0.0
    max_diff_c3 = 0.0

    for scen in ["ssp126", "ssp585"]:
        X_scen = ssp_data[scen]["X"]
        y0_scen = ssp_data[scen]["y0_flat"]

        # Recompute S1
        s1_h_re = 0.5 * (
            np.clip(y0_scen[:, 0] + reloaded_models[("h", "s1", SEEDS[0])].predict(X_scen), 0.0, 35.0) +
            np.clip(y0_scen[:, 0] + reloaded_models[("h", "s1", SEEDS[1])].predict(X_scen), 0.0, 35.0)
        )
        s1_a_re = 0.5 * (
            np.maximum(y0_scen[:, 1] + reloaded_models[("a", "s1", SEEDS[0])].predict(X_scen), 0.0) +
            np.maximum(y0_scen[:, 1] + reloaded_models[("a", "s1", SEEDS[1])].predict(X_scen), 0.0)
        )
        re_s1_sc = compute_scaled_pred(np.column_stack([s1_h_re, s1_a_re]))
        orig_s1_sc = compute_scaled_pred(ssp_preds["V6_6_S1type"][scen])
        max_diff_s1 = max(max_diff_s1, float(np.max(np.abs(re_s1_sc - orig_s1_sc))))

        # Recompute C3
        p_h_0 = reloaded_models[("h", "clf", SEEDS[0])].predict_proba(X_scen)[:, 1]
        p_h_1 = reloaded_models[("h", "clf", SEEDS[1])].predict_proba(X_scen)[:, 1]
        mu0_h_0 = reloaded_models[("h", "mu0", SEEDS[0])].predict(X_scen)
        mu1_h_0 = reloaded_models[("h", "mu1", SEEDS[0])].predict(X_scen)
        mu0_h_1 = reloaded_models[("h", "mu0", SEEDS[1])].predict(X_scen)
        mu1_h_1 = reloaded_models[("h", "mu1", SEEDS[1])].predict(X_scen)

        dh_0 = (1.0 - p_h_0) * mu0_h_0 + p_h_0 * mu1_h_0
        dh_1 = (1.0 - p_h_1) * mu0_h_1 + p_h_1 * mu1_h_1
        c3_h_re = 0.5 * (np.clip(y0_scen[:, 0] + dh_0, 0.0, 35.0) + np.clip(y0_scen[:, 0] + dh_1, 0.0, 35.0))

        p_a_0 = reloaded_models[("a", "clf", SEEDS[0])].predict_proba(X_scen)[:, 1]
        p_a_1 = reloaded_models[("a", "clf", SEEDS[1])].predict_proba(X_scen)[:, 1]
        mu0_a_0 = reloaded_models[("a", "mu0", SEEDS[0])].predict(X_scen)
        mu1_a_0 = reloaded_models[("a", "mu1", SEEDS[0])].predict(X_scen)
        mu0_a_1 = reloaded_models[("a", "mu0", SEEDS[1])].predict(X_scen)
        mu1_a_1 = reloaded_models[("a", "mu1", SEEDS[1])].predict(X_scen)

        da_0 = (1.0 - p_a_0) * mu0_a_0 + p_a_0 * mu1_a_0
        da_1 = (1.0 - p_a_1) * mu0_a_1 + p_a_1 * mu1_a_1
        c3_a_re = 0.5 * (np.maximum(y0_scen[:, 1] + da_0, 0.0) + np.maximum(y0_scen[:, 1] + da_1, 0.0))

        re_c3_sc = compute_scaled_pred(np.column_stack([c3_h_re, c3_a_re]))
        orig_c3_sc = compute_scaled_pred(ssp_preds["V6_7_C3"][scen])
        max_diff_c3 = max(max_diff_c3, float(np.max(np.abs(re_c3_sc - orig_c3_sc))))

    print(f"Reload verification maximum scaled difference: S1 = {max_diff_s1:.2e}, C3 = {max_diff_c3:.2e}", flush=True)
    assert max_diff_s1 <= 1e-5, f"Criterion 2 FAILED: S1 max diff {max_diff_s1} > 1e-5"
    assert max_diff_c3 <= 1e-5, f"Criterion 2 FAILED: C3 max diff {max_diff_c3} > 1e-5"
    print("Acceptance Criterion 2 PASSED: Reloaded model predictions match within 1e-5.\n", flush=True)

    # -------------------------------------------------------------
    # 11. Acceptance Criterion 4: Comparison with G1 & Average p on SSP
    # -------------------------------------------------------------
    print("Checking Acceptance Criterion 4: Computing differences against G1 and average p on SSP...", flush=True)
    assert G1_CSV_PATH.exists(), f"Missing G1 reference file: {G1_CSV_PATH}"
    g1_df = pd.read_csv(G1_CSV_PATH)
    v6_6_df = pd.read_csv(csv_paths["V6_6_S1type"])
    v6_7_df = pd.read_csv(csv_paths["V6_7_C3"])

    g1_h = g1_df["height"].values
    g1_a = g1_df["agb"].values

    v6_6_h = v6_6_df["height"].values
    v6_6_a = v6_6_df["agb"].values

    v6_7_h = v6_7_df["height"].values
    v6_7_a = v6_7_df["agb"].values

    def get_rms_and_corr(a_vals, b_vals):
        rms = float(np.sqrt(np.mean((a_vals - b_vals) ** 2)))
        corr = float(np.corrcoef(a_vals, b_vals)[0, 1])
        return rms, corr

    rms_h_s1_g1, corr_h_s1_g1 = get_rms_and_corr(v6_6_h, g1_h)
    rms_a_s1_g1, corr_a_s1_g1 = get_rms_and_corr(v6_6_a, g1_a)
    rms_comp_s1_g1 = 0.5 * (rms_h_s1_g1 + rms_a_s1_g1)

    rms_h_c3_g1, corr_h_c3_g1 = get_rms_and_corr(v6_7_h, g1_h)
    rms_a_c3_g1, corr_a_c3_g1 = get_rms_and_corr(v6_7_a, g1_a)
    rms_comp_c3_g1 = 0.5 * (rms_h_c3_g1 + rms_a_c3_g1)

    rms_h_c3_s1, corr_h_c3_s1 = get_rms_and_corr(v6_7_h, v6_6_h)
    rms_a_c3_s1, corr_a_c3_s1 = get_rms_and_corr(v6_7_a, v6_6_a)
    rms_comp_c3_s1 = 0.5 * (rms_h_c3_s1 + rms_a_c3_s1)

    comparison_records = [
        {
            "Comparison": "V6_6_S1type vs G1",
            "Height_RMS": rms_h_s1_g1, "Height_Corr": corr_h_s1_g1,
            "AGB_RMS": rms_a_s1_g1, "AGB_Corr": corr_a_s1_g1,
            "Composite_RMS": rms_comp_s1_g1
        },
        {
            "Comparison": "V6_7_C3 vs G1",
            "Height_RMS": rms_h_c3_g1, "Height_Corr": corr_h_c3_g1,
            "AGB_RMS": rms_a_c3_g1, "AGB_Corr": corr_a_c3_g1,
            "Composite_RMS": rms_comp_c3_g1
        },
        {
            "Comparison": "V6_7_C3 vs V6_6_S1type",
            "Height_RMS": rms_h_c3_s1, "Height_Corr": corr_h_c3_s1,
            "AGB_RMS": rms_a_c3_s1, "AGB_Corr": corr_a_c3_s1,
            "Composite_RMS": rms_comp_c3_s1
        }
    ]
    pd.DataFrame(comparison_records).to_csv(OUT_DIR / "table_g1_comparisons.csv", index=False)

    # Average p on SSP across age groups
    ssp_ages_repeated = np.tile(AGES, 6171) # (92565,) per scenario
    age_masks = {
        "All": np.ones(len(ssp_ages_repeated), dtype=bool),
        "<=50": ssp_ages_repeated <= 50,
        "70-100": (ssp_ages_repeated >= 70) & (ssp_ages_repeated <= 100),
        ">=100": ssp_ages_repeated >= 100,
        ">=150": ssp_ages_repeated >= 150
    }

    p_records = []
    for scen in ["ssp126", "ssp585"]:
        p_h_scen = ssp_p_values[scen]["h"]
        p_a_scen = ssp_p_values[scen]["a"]

        for grp_name, g_mask in age_masks.items():
            p_records.append({
                "Scenario": scen,
                "Age_Group": grp_name,
                "Mean_p_Height": float(np.mean(p_h_scen[g_mask])),
                "Mean_p_AGB": float(np.mean(p_a_scen[g_mask]))
            })

    # Combined across both scenarios
    p_h_both = np.concatenate([ssp_p_values["ssp126"]["h"], ssp_p_values["ssp585"]["h"]])
    p_a_both = np.concatenate([ssp_p_values["ssp126"]["a"], ssp_p_values["ssp585"]["a"]])
    ages_both = np.concatenate([ssp_ages_repeated, ssp_ages_repeated])
    both_masks = {
        "All": np.ones(len(ages_both), dtype=bool),
        "<=50": ages_both <= 50,
        "70-100": (ages_both >= 70) & (ages_both <= 100),
        ">=100": ages_both >= 100,
        ">=150": ages_both >= 150
    }
    for grp_name, g_mask in both_masks.items():
        p_records.append({
            "Scenario": "Combined_SSP",
            "Age_Group": grp_name,
            "Mean_p_Height": float(np.mean(p_h_both[g_mask])),
            "Mean_p_AGB": float(np.mean(p_a_both[g_mask]))
        })

    pd.DataFrame(p_records).to_csv(OUT_DIR / "table_ssp_average_p.csv", index=False)
    print("Acceptance Criterion 4 PASSED: Comparisons and SSP average p tables generated.\n", flush=True)

    # -------------------------------------------------------------
    # 12. Save Metrics & Checksums
    # -------------------------------------------------------------
    total_elapsed = time.time() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    code_hashes = {
        "run_v6_c3_ssp.py": sha256_file(SCRIPT_DIR / "run_v6_c3_ssp.py"),
        "features.py": sha256_file(SCRIPT_DIR / "features.py"),
        "restore_y0.py": sha256_file(SCRIPT_DIR / "restore_y0.py"),
        "prepare_inputs.py": sha256_file(SCRIPT_DIR / "prepare_inputs.py")
    }

    results_hashes = {
        "candidate_V6_6_S1type_full.csv": sha256_file(csv_paths["V6_6_S1type"]),
        "candidate_V6_7_C3_full.csv": sha256_file(csv_paths["V6_7_C3"]),
        "table_selected_iterations.csv": sha256_file(OUT_DIR / "table_selected_iterations.csv"),
        "table_g1_comparisons.csv": sha256_file(OUT_DIR / "table_g1_comparisons.csv"),
        "table_ssp_average_p.csv": sha256_file(OUT_DIR / "table_ssp_average_p.csv")
    }

    metrics = {
        "task": "V6 Batch 3 SSP Predictions: S1-type and C3",
        "date": "2026-11-08",
        "elapsed_seconds": total_elapsed,
        "elapsed_hours": total_elapsed / 3600.0,
        "peak_memory_mb": peak_mem / (1024 * 1024),
        "acceptance_checks": {
            "criterion_1_kt_check_passed": True,
            "criterion_2_reload_diff_le_1e5": bool(max(max_diff_s1, max_diff_c3) <= 1e-5),
            "max_reload_diff_s1": max_diff_s1,
            "max_reload_diff_c3": max_diff_c3,
            "criterion_3_train_sites_49204": int(n_sites),
            "criterion_4_comparisons_and_p_reported": True,
            "criterion_5_sha256_recorded": True
        },
        "selected_iterations": {
            f"{tgt}_{mod}_{s}": selected_iterations[(tgt, mod, s)]
            for tgt, mod, s in selected_iterations
        },
        "g1_comparisons": comparison_records,
        "code_hashes": code_hashes,
        "results_hashes": results_hashes
    }

    with open(OUT_DIR / "metrics.json", "w") as f_met:
        json.dump(metrics, f_met, indent=2)

    print(f"\nPipeline completed in {total_elapsed:.2f}s ({total_elapsed/3600.0:.2f}h). Peak RAM: {peak_mem/(1024*1024):.1f} MB.", flush=True)


if __name__ == "__main__":
    main()

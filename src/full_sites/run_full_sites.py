"""Full Sites Training Pipeline for G1 and G2 (2026-11-02, Revised)
Execution by Antigravity.
Strict boundaries:
- Local training and CSV generation only.
- No uploads, no Kaggle API calls, no credential reading, no modifying existing files.
- No tuning based on public scores or SSP data.
- User authorized training on 49,204 sites (including original test split).
- Revised holdout: strictly 3,924 sites (protected_history_confirmation minus fresh_confirmation).
- ZERO reading of fresh_confirmation (1,024 sites) labels.
- Generates candidate_G1_full_delta.csv and candidate_G2_full_noclimate.csv.
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

# Local path setup
ROOT = Path("data/project")
CODE_DIR = ROOT / "code/scripts/full_sites_20261102"
OUT_DIR = ROOT / "results/full_sites_20261102"
MODELS_DIR = OUT_DIR / "models"
KAGGLE_DIR = Path("data/kaggle")
DATA_GLOBAL_DIR = Path("data/dataset/data_global")
MANIFEST_PATH = ROOT / "results/competition_specific_plan_20261013/site_manifests.npz"
SPLITS_PATH = ROOT / "results/repro_package_v2_20261101/inputs/site_splits.npz"
SOURCE_DATA_DIR = ROOT / "results/competition_signed_training_20261013"
SAMPLE_CSV_PATH = KAGGLE_DIR / "sample_submission.csv"

# Existing reference files for verification and diagnostics
S1_CSV_PATH = ROOT / "results/delta_profile_20261101/candidate_S1_delta_profile_3373.csv"
F3_CSV_PATH = ROOT / "results/score_blend_f3_20261102/candidate_F3_score_blend_nosigned.csv"
P1_CSV_PATH = ROOT / "results/segment_probe_20261102/candidate_P1_tall_down.csv"
F3_TABLE_PATH = ROOT / "results/score_blend_f3_20261102/f3_submission_table.csv"

EXPECTED_S1_SHA = "23216519297719e207970ecd37151181b57cb53d1e23a11499ca33e7b0824aba"
EXPECTED_F3_SHA = "3bd2f0d6fb84654739f4db8654fd4091002e98f1db8f937a326c9fb36bf41d2a"
EXPECTED_P1_SHA = "9f0fe303f2d8b5339153d1c1c8bdbbe22ea17ce16aab09593202794ec9f3fb99"

# Sys path setup
sys.path[:0] = [
    str(ROOT / ".runtime/science-cp314"),
    str(ROOT / "code"),
    str(CODE_DIR)
]

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from features import H_CAP, SCALE, SEED_AGES, extract_all_features, extract_climate_features
from restore_y0 import restore_initial_y0
from src import kaggle_task as kt

SEEDS = [20261101, 20261102, 20261103]


def sha256_file(p: Path | str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_scaled_mse(pred_y40: np.ndarray, true_y40: np.ndarray) -> dict:
    """Computes competition-scaled MSE metrics.
    pred_y40, true_y40: (N, 2) [height, agb] in physical units.
    """
    p_h = np.clip(pred_y40[:, 0], 0.0, 35.0) / SCALE[0]
    p_h = np.minimum(p_h, H_CAP)
    p_a = np.maximum(pred_y40[:, 1], 0.0) / SCALE[1]

    t_h = np.clip(true_y40[:, 0], 0.0, 35.0) / SCALE[0]
    t_h = np.minimum(t_h, H_CAP)
    t_a = np.maximum(true_y40[:, 1], 0.0) / SCALE[1]

    mse_h = float(np.mean((p_h - t_h) ** 2))
    mse_a = float(np.mean((p_a - t_a) ** 2))
    score = 0.5 * (mse_h + mse_a)
    return {"score": score, "mse_h": mse_h, "mse_a": mse_a}


def build_submission_csv(
    sample_ids: list[str],
    pred_ssp126: np.ndarray,
    pred_ssp585: np.ndarray,
    out_csv: Path,
):
    """Formats and writes test predictions to Kaggle submission CSV format."""
    pred_all = np.vstack([pred_ssp126, pred_ssp585])  # (185130, 2)
    assert len(sample_ids) == len(pred_all), f"Length mismatch: {len(sample_ids)} vs {len(pred_all)}"

    h_phys = np.clip(pred_all[:, 0], 0.0, 35.0)
    a_phys = np.maximum(pred_all[:, 1], 0.0)

    h_scaled = np.minimum(h_phys / SCALE[0], H_CAP)
    a_scaled = a_phys / SCALE[1]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "height", "agb"])
        for row_id, h_val, a_val in zip(sample_ids, h_scaled, a_scaled):
            writer.writerow([row_id, f"{h_val:.6f}", f"{a_val:.6f}"])


def extract_site_data(site_indices: np.ndarray, fresh_forbidden_sites: np.ndarray, chunk_size: int = 4000):
    """Extracts y0, y40, and 12 climate features from raw data_global files with mmap.
    Strictly asserts that NO site in fresh_forbidden_sites is read!
    """
    n_sites = len(site_indices)
    assert not np.any(np.isin(site_indices, fresh_forbidden_sites)), "FATAL: Attempting to read fresh_confirmation sites!"
    print(f"Extracting data for {n_sites} sites (chunk_size={chunk_size})...", flush=True)

    # 1. Extract y0 and y40 across all 15 ages
    y0_list = []
    y40_list = []
    for age in SEED_AGES:
        y_file = DATA_GLOBAL_DIR / f"glob_Y{age:03d}.npy"
        raw_y = np.load(y_file, mmap_mode="r")
        y0_age = np.asarray(raw_y[site_indices, 0, 11, :2], dtype=np.float32)
        y40_age = np.asarray(raw_y[site_indices, 40, 11, :2], dtype=np.float32)
        y0_list.append(y0_age)
        y40_list.append(y40_age)
        del raw_y

    y0_phys = np.stack(y0_list, axis=1)  # (N, 15, 2)
    y40_phys = np.stack(y40_list, axis=1)  # (N, 15, 2)
    delta_phys = y40_phys - y0_phys  # (N, 15, 2)

    # 2. Extract climate features in chunks from glob_X_fea.npy
    raw_x = np.load(DATA_GLOBAL_DIR / "glob_X_fea.npy", mmap_mode="r")
    climate_list = []
    for start in range(0, n_sites, chunk_size):
        end = min(start + chunk_size, n_sites)
        batch = site_indices[start:end]
        x_chunk = np.asarray(raw_x[batch], dtype=np.float32)
        c_chunk = extract_climate_features(x_chunk)
        climate_list.append(c_chunk)
        del x_chunk
    climate_feats = np.vstack(climate_list)  # (N, 12)
    del raw_x

    # 3. Assemble full 46 features
    num_ages = len(SEED_AGES)
    h = y0_phys[:, :, 0]
    a = y0_phys[:, :, 1]
    log_age = np.log(np.array(SEED_AGES, dtype=np.float32))
    log_age_grid = np.broadcast_to(log_age[None, :], (n_sites, num_ages))
    self_state = np.stack([h, a, log_age_grid], axis=-1)

    profile = y0_phys.reshape(n_sites, 30)
    profile_expanded = np.broadcast_to(profile[:, None, :], (n_sites, num_ages, 30))

    h_max = h.max(axis=1, keepdims=True)
    dist_to_max = (h_max - h)[:, :, None]

    climate_expanded = np.broadcast_to(climate_feats[:, None, :], (n_sites, num_ages, 12))

    X_all_46 = np.concatenate([self_state, profile_expanded, dist_to_max, climate_expanded], axis=-1)
    X_all_46 = X_all_46.reshape(n_sites * num_ages, -1).astype(np.float32)

    y0_flat = y0_phys.reshape(n_sites * num_ages, 2).astype(np.float64)
    y40_flat = y40_phys.reshape(n_sites * num_ages, 2).astype(np.float64)
    delta_flat = delta_phys.reshape(n_sites * num_ages, 2).astype(np.float32)

    return X_all_46, y0_flat, y40_flat, delta_flat


def run():
    print("=== Antigravity Full Sites Training Pipeline (G1, G2) - REVISED ===", flush=True)
    t_start = time.time()
    tracemalloc.start()

    # Prerequisite check: output directory must NOT exist
    if OUT_DIR.exists():
        raise RuntimeError(f"Output directory {OUT_DIR} already exists! Halting as per task boundary.")
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    MODELS_DIR.mkdir(parents=True, exist_ok=False)

    # Verify existing reference files SHA256
    assert sha256_file(S1_CSV_PATH) == EXPECTED_S1_SHA, "S1 SHA256 altered!"
    assert sha256_file(F3_CSV_PATH) == EXPECTED_F3_SHA, "F3 SHA256 altered!"
    assert sha256_file(P1_CSV_PATH) == EXPECTED_P1_SHA, "P1 SHA256 altered!"
    print("Verified existing files SHA256 (S1, F3, P1).", flush=True)

    # Load site manifests
    manifest = np.load(MANIFEST_PATH)
    train_sites = manifest["all_public_historical_except_protected"]  # 49,204
    protected_sites = manifest["protected_history_confirmation"]  # 4,948
    fresh_sites = manifest["fresh_confirmation"]  # 1,024
    baseline_sites = manifest["baseline_train_val"]  # 3,373

    # Revised holdout sites: strictly 3,924 sites
    holdout_sites = np.setdiff1d(protected_sites, fresh_sites)
    n_train_sites = len(train_sites)
    n_holdout_sites = len(holdout_sites)
    n_fresh_sites = len(fresh_sites)

    assert n_train_sites == 49204, f"Expected 49204 training sites, got {n_train_sites}"
    assert n_holdout_sites == 3924, f"Expected 3924 holdout sites, got {n_holdout_sites}"
    assert n_fresh_sites == 1024, f"Expected 1024 fresh sites, got {n_fresh_sites}"

    # Verify disjointness
    assert len(set(holdout_sites).intersection(set(fresh_sites))) == 0, "Fresh sites present in holdout!"
    assert len(set(train_sites).intersection(set(fresh_sites))) == 0, "Fresh sites present in train!"
    assert len(set(train_sites).intersection(set(holdout_sites))) == 0, "Holdout sites intersect with train!"
    print(f"Loaded manifests: {n_train_sites} train, {n_holdout_sites} holdout (3924), {n_fresh_sites} fresh (FORBIDDEN).", flush=True)

    # -------------------------------------------------------------
    # Step 1: Synonymy Check on 3,373 baseline sites
    # -------------------------------------------------------------
    print("\n[Step 1] Running Synonymy Check on 3,373 baseline sites...", flush=True)
    s1_X = np.load(SOURCE_DATA_DIR / "monthly_X.npy")
    s1_Y = np.load(SOURCE_DATA_DIR / "december_Y.npy")
    s1_y0_phys = s1_Y[:, :, 0, :2].transpose(1, 0, 2).astype(np.float64)
    s1_y40_phys = s1_Y[:, :, 40, :2].transpose(1, 0, 2).astype(np.float64)
    s1_feats_46 = extract_all_features(s1_y0_phys, s1_X).astype(np.float32)

    X_base_46, y0_base_flat, y40_base_flat, delta_base_flat = extract_site_data(baseline_sites, fresh_sites)

    max_diff_y0 = float(np.max(np.abs(y0_base_flat - s1_y0_phys.reshape(-1, 2))))
    max_diff_y40 = float(np.max(np.abs(y40_base_flat - s1_y40_phys.reshape(-1, 2))))
    max_diff_feats = float(np.max(np.abs(X_base_46 - s1_feats_46)))

    print(f"Synonymy diffs -> y0: {max_diff_y0:.2e}, y40: {max_diff_y40:.2e}, features: {max_diff_feats:.2e}", flush=True)
    assert max_diff_y0 <= 1e-6, f"y0 synonymy check failed: {max_diff_y0}"
    assert max_diff_y40 <= 1e-6, f"y40 synonymy check failed: {max_diff_y40}"
    assert max_diff_feats <= 1e-6, f"features synonymy check failed: {max_diff_feats}"
    print("Synonymy check PASSED! All arrays match S1 inputs within 1e-6.", flush=True)
    del s1_X, s1_Y, s1_y0_phys, s1_y40_phys, s1_feats_46, X_base_46, y0_base_flat, y40_base_flat, delta_base_flat

    # -------------------------------------------------------------
    # Step 2: Extract 49,204 Training Sites and 3,924 Holdout Sites
    # -------------------------------------------------------------
    print("\n[Step 2] Extracting training data (49,204 sites, 738,060 rows)...", flush=True)
    t_extract_train = time.time()
    X_train_46, y0_train_flat, y40_train_flat, delta_train_flat = extract_site_data(train_sites, fresh_sites)
    X_train_34 = X_train_46[:, :34].copy()
    n_train_rows = len(X_train_46)
    assert n_train_rows == 738060, f"Expected 738060 training rows, got {n_train_rows}"
    print(f"Training data extracted in {time.time() - t_extract_train:.2f}s (X_train_46: {X_train_46.nbytes / (1024*1024):.1f} MB)", flush=True)

    print("\nExtracting revised holdout data (3,924 sites, 58,860 rows; 1024 fresh strictly excluded)...", flush=True)
    t_extract_holdout = time.time()
    X_holdout_46, y0_holdout_flat, y40_holdout_flat, delta_holdout_flat = extract_site_data(holdout_sites, fresh_sites)
    X_holdout_34 = X_holdout_46[:, :34].copy()
    n_holdout_rows = len(X_holdout_46)
    assert n_holdout_rows == 58860, f"Expected 58860 holdout rows (3924*15), got {n_holdout_rows}"
    holdout_ages = np.tile(SEED_AGES, n_holdout_sites)
    print(f"Revised holdout data extracted in {time.time() - t_extract_holdout:.2f}s (fresh 1024 untouched).", flush=True)

    # -------------------------------------------------------------
    # Step 3: Optimal Iteration Search via 5-Fold GroupKFold
    # -------------------------------------------------------------
    print("\n[Step 3] Running 5-fold GroupKFold iteration search (every 25 iters up to 1500)...", flush=True)
    row_groups = np.repeat(np.arange(n_train_sites), 15)  # Grouped strictly by site
    gkf = GroupKFold(n_splits=5)
    splits_list = list(gkf.split(X_train_46, delta_train_flat[:, 0], groups=row_groups))

    opt_iterations = {
        "G1": {"h": [], "a": []},
        "G2": {"h": [], "a": []}
    }

    mem_peak_f0 = 0.0

    configs = [
        ("G1", "h", X_train_46, 0),
        ("G1", "a", X_train_46, 1),
        ("G2", "h", X_train_34, 0),
        ("G2", "a", X_train_34, 1),
    ]

    for model_name, target_name, X_data, col_idx in configs:
        print(f"\n--- Searching iterations for {model_name} Target {target_name.upper()} ---", flush=True)
        best_iters_folds = []
        for fold, (train_idx, val_idx) in enumerate(splits_list):
            t_fold_start = time.time()
            reg = HistGradientBoostingRegressor(
                learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50,
                max_features=0.8, l2_regularization=0.0, early_stopping=False,
                max_iter=1500, random_state=20261101 + fold
            )
            reg.fit(X_data[train_idx], delta_train_flat[train_idx, col_idx])

            # Staged evaluation every 25 iterations
            val_target = delta_train_flat[val_idx, col_idx]
            mses = []
            for iter_idx, p in enumerate(reg.staged_predict(X_data[val_idx])):
                curr_iter = iter_idx + 1
                if curr_iter % 25 == 0:
                    mse = float(np.mean((p - val_target) ** 2))
                    mses.append((curr_iter, mse))

            best_iter, best_mse = min(mses, key=lambda x: x[1])
            best_iters_folds.append(best_iter)
            t_fold = time.time() - t_fold_start

            if model_name == "G1" and target_name == "h" and fold == 0:
                _, mem_peak_bytes = tracemalloc.get_traced_memory()
                mem_peak_f0 = mem_peak_bytes / (1024 * 1024)
                est_total_search_time = t_fold * 20  # 4 configs * 5 folds
                est_full_train_time = (t_fold * 0.9) * 12  # 12 full fits
                est_total_seconds = est_total_search_time + est_full_train_time + 300
                print(f"[Resource Monitor] Fold 0 Height completed in {t_fold:.2f}s. Peak memory: {mem_peak_f0:.1f} MB.", flush=True)
                print(f"[Resource Monitor] Estimated total pipeline runtime: {est_total_seconds / 3600:.2f} hours.", flush=True)
                if est_total_seconds > 8 * 3600:
                    raise RuntimeError(f"Estimated total runtime ({est_total_seconds/3600:.2f}h) exceeds 8-hour limit!")

            print(f"  {model_name} {target_name} Fold {fold}: Best Iter = {best_iter} (MSE: {best_mse:.6f}, Time: {t_fold:.1f}s)", flush=True)

        opt_iterations[model_name][target_name] = best_iters_folds

    # Median iterations
    median_iters = {
        "G1": {
            "h": int(np.median(opt_iterations["G1"]["h"])),
            "a": int(np.median(opt_iterations["G1"]["a"]))
        },
        "G2": {
            "h": int(np.median(opt_iterations["G2"]["h"])),
            "a": int(np.median(opt_iterations["G2"]["a"]))
        }
    }
    print(f"\nFinal Selected Median Iterations:", flush=True)
    print(f"  G1: Height = {median_iters['G1']['h']}, AGB = {median_iters['G1']['a']}", flush=True)
    print(f"  G2: Height = {median_iters['G2']['h']}, AGB = {median_iters['G2']['a']}", flush=True)

    # -------------------------------------------------------------
    # Step 4: Full Training on all 738,060 rows with 3 Seeds
    # -------------------------------------------------------------
    print("\n[Step 4] Full training on 738,060 rows (3 seeds each)...", flush=True)
    models = {
        "G1": {"h": [], "a": []},
        "G2": {"h": [], "a": []}
    }
    model_paths = {}
    model_hashes = {}

    model_train_specs = [
        ("G1", "h", X_train_46, 0, median_iters["G1"]["h"]),
        ("G1", "a", X_train_46, 1, median_iters["G1"]["a"]),
        ("G2", "h", X_train_34, 0, median_iters["G2"]["h"]),
        ("G2", "a", X_train_34, 1, median_iters["G2"]["a"]),
    ]

    for model_name, target_name, X_data, col_idx, n_iters in model_train_specs:
        print(f"Training {model_name} {target_name} with max_iter={n_iters} across 3 seeds...", flush=True)
        for s in SEEDS:
            t_fit_start = time.time()
            reg = HistGradientBoostingRegressor(
                learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50,
                max_features=0.8, l2_regularization=0.0, early_stopping=False,
                max_iter=n_iters, random_state=s
            )
            reg.fit(X_data, delta_train_flat[:, col_idx])
            models[model_name][target_name].append(reg)

            # Save model
            model_fname = f"{model_name}_{target_name}_seed{s}.joblib"
            save_path = MODELS_DIR / model_fname
            joblib.dump(reg, save_path)
            model_paths[model_fname] = save_path
            model_hashes[model_fname] = sha256_file(save_path)
            print(f"  Saved {model_fname} ({time.time() - t_fit_start:.1f}s, SHA: {model_hashes[model_fname][:12]}...)", flush=True)

    # -------------------------------------------------------------
    # Step 5: Inference on SSP126 & SSP585
    # -------------------------------------------------------------
    print("\n[Step 5] Predicting on SSP126 and SSP585...", flush=True)
    sample_df = pd.read_csv(SAMPLE_CSV_PATH)
    sample_ids = sample_df["id"].tolist()
    assert len(sample_ids) == 185130

    test_features = {}
    test_y0_phys = {}

    for scen in ["ssp126", "ssp585"]:
        with np.load(KAGGLE_DIR / f"test_{scen}.npz") as z:
            t_x = z["test_x"]
            t_y0 = z["test_y0"]
        y0_res = restore_initial_y0(t_y0, clip_height=True, clip_agb=True)
        test_y0_phys[scen] = y0_res.reshape(-1, 2)  # (92565, 2)
        feats_46 = extract_all_features(y0_res, t_x).astype(np.float32)
        test_features[scen] = {
            "46": feats_46,
            "34": feats_46[:, :34].copy()
        }

    # Predict G1 and G2 on SSP
    ssp_predictions = {}
    for m in ["G1", "G2"]:
        feat_key = "46" if m == "G1" else "34"
        scen_preds = {}
        for scen in ["ssp126", "ssp585"]:
            X_test = test_features[scen][feat_key]
            y0_base = test_y0_phys[scen]

            pred_dh = np.mean([mod.predict(X_test) for mod in models[m]["h"]], axis=0)
            pred_da = np.mean([mod.predict(X_test) for mod in models[m]["a"]], axis=0)

            pred_h_raw = y0_base[:, 0] + pred_dh
            pred_a_raw = y0_base[:, 1] + pred_da
            scen_preds[scen] = np.column_stack([pred_h_raw, pred_a_raw])
        ssp_predictions[m] = scen_preds

    # Write G1 and G2 CSVs
    csv_paths = {
        "G1": OUT_DIR / "candidate_G1_full_delta.csv",
        "G2": OUT_DIR / "candidate_G2_full_noclimate.csv"
    }
    csv_hashes = {}

    for m in ["G1", "G2"]:
        build_submission_csv(
            sample_ids,
            ssp_predictions[m]["ssp126"],
            ssp_predictions[m]["ssp585"],
            csv_paths[m]
        )
        csv_hashes[m] = sha256_file(csv_paths[m])
        print(f"Generated {csv_paths[m].name} (SHA256: {csv_hashes[m]})", flush=True)

    # -------------------------------------------------------------
    # Step 6: Holdout Evaluation on 3,924 Sites (58,860 rows)
    # -------------------------------------------------------------
    print("\n[Step 6] Evaluating G1, G2, S1, M1 on 3,924 Revised Holdout Sites...", flush=True)
    # Load S1 and M1 models
    s1_models_h = [joblib.load(ROOT / f"results/delta_profile_20261101/models/model_height_seed{s}.joblib") for s in SEEDS]
    s1_models_a = [joblib.load(ROOT / f"results/delta_profile_20261101/models/model_agb_seed{s}.joblib") for s in SEEDS]

    m1_models_h = [joblib.load(ROOT / f"results/new_directions_20261101/models/M1_h_seed{s}.pkl") for s in SEEDS]
    m1_models_a = [joblib.load(ROOT / f"results/new_directions_20261101/models/M1_a_seed{s}.pkl") for s in SEEDS]

    holdout_preds = {}

    # G1
    pred_g1_dh = np.mean([mod.predict(X_holdout_46) for mod in models["G1"]["h"]], axis=0)
    pred_g1_da = np.mean([mod.predict(X_holdout_46) for mod in models["G1"]["a"]], axis=0)
    holdout_preds["G1"] = np.column_stack([y0_holdout_flat[:, 0] + pred_g1_dh, y0_holdout_flat[:, 1] + pred_g1_da])

    # G2
    pred_g2_dh = np.mean([mod.predict(X_holdout_34) for mod in models["G2"]["h"]], axis=0)
    pred_g2_da = np.mean([mod.predict(X_holdout_34) for mod in models["G2"]["a"]], axis=0)
    holdout_preds["G2"] = np.column_stack([y0_holdout_flat[:, 0] + pred_g2_dh, y0_holdout_flat[:, 1] + pred_g2_da])

    # S1
    pred_s1_dh = np.mean([mod.predict(X_holdout_46) for mod in s1_models_h], axis=0)
    pred_s1_da = np.mean([mod.predict(X_holdout_46) for mod in s1_models_a], axis=0)
    holdout_preds["S1"] = np.column_stack([y0_holdout_flat[:, 0] + pred_s1_dh, y0_holdout_flat[:, 1] + pred_s1_da])

    # M1
    pred_m1_dh = np.mean([mod.predict(X_holdout_34) for mod in m1_models_h], axis=0)
    pred_m1_da = np.mean([mod.predict(X_holdout_34) for mod in m1_models_a], axis=0)
    holdout_preds["M1"] = np.column_stack([y0_holdout_flat[:, 0] + pred_m1_dh, y0_holdout_flat[:, 1] + pred_m1_da])

    # Persistence
    holdout_preds["persistence"] = y0_holdout_flat.copy()

    holdout_results = {}
    for mod_name in ["persistence", "S1", "M1", "G1", "G2"]:
        overall = compute_scaled_mse(holdout_preds[mod_name], y40_holdout_flat)

        # Isolated age >= 150 height MSE
        old_mask = holdout_ages >= 150
        old_h_mse = float(np.mean(
            (np.clip(holdout_preds[mod_name][old_mask, 0], 0, 35) / SCALE[0] -
             np.clip(y40_holdout_flat[old_mask, 0], 0, 35) / SCALE[0]) ** 2
        ))

        age_breakdown = {}
        for age in SEED_AGES:
            mask = holdout_ages == age
            age_breakdown[str(age)] = compute_scaled_mse(holdout_preds[mod_name][mask], y40_holdout_flat[mask])

        holdout_results[mod_name] = {
            "overall": overall,
            "old_height_mse": old_h_mse,
            "age_breakdown": age_breakdown
        }
        print(f"  Holdout (3924) {mod_name:12s} | Score: {overall['score']:.6f} | H: {overall['mse_h']:.6f} | AGB: {overall['mse_a']:.6f} | H(age>=150): {old_h_mse:.6f}", flush=True)

    # -------------------------------------------------------------
    # Step 7: New Direction Diagnostics (vs S1, F3, P1 and Subspace Novelty)
    # -------------------------------------------------------------
    print("\n[Step 7] Computing New Direction Diagnostics...", flush=True)
    def load_sub_values(csv_p):
        with open(csv_p, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        return np.array([[float(r["height"]), float(r["agb"])] for r in reader], dtype=np.float64)

    s1_vals = load_sub_values(S1_CSV_PATH)
    f3_vals = load_sub_values(F3_CSV_PATH)
    p1_vals = load_sub_values(P1_CSV_PATH)
    g1_vals = load_sub_values(csv_paths["G1"])
    g2_vals = load_sub_values(csv_paths["G2"])

    candidates_vals = {"G1": g1_vals, "G2": g2_vals}
    references_vals = {"S1": s1_vals, "F3": f3_vals, "P1": p1_vals}

    comparison_results = {}
    for c_name, c_mat in candidates_vals.items():
        comp_dict = {}
        for ref_name, ref_mat in references_vals.items():
            diff_h = c_mat[:, 0] - ref_mat[:, 0]
            diff_a = c_mat[:, 1] - ref_mat[:, 1]
            rms_h = float(np.sqrt(np.mean(diff_h ** 2)))
            rms_a = float(np.sqrt(np.mean(diff_a ** 2)))
            rms_tot = float(0.5 * (rms_h + rms_a))
            rms_pooled = float(np.sqrt(np.mean((c_mat - ref_mat) ** 2)))
            corr_h = float(np.corrcoef(c_mat[:, 0], ref_mat[:, 0])[0, 1])
            corr_a = float(np.corrcoef(c_mat[:, 1], ref_mat[:, 1])[0, 1])
            corr_tot = float(np.corrcoef(c_mat.ravel(), ref_mat.ravel())[0, 1])

            comp_dict[ref_name] = {
                "rms_h": rms_h, "rms_a": rms_a, "rms_mean": rms_tot, "rms_pooled": rms_pooled,
                "corr_h": corr_h, "corr_a": corr_a, "corr_pooled": corr_tot
            }
        comparison_results[c_name] = comp_dict

    # Novelty: d = G - P1 projected onto 17 F3 directions
    f3_table = pd.read_csv(F3_TABLE_PATH)
    f3_dirs_rows = f3_table[f3_table["source"] != "Base"]
    assert len(f3_dirs_rows) == 17, f"Expected 17 F3 directions, got {len(f3_dirs_rows)}"

    D_f3_list = []
    for _, r in f3_dirs_rows.iterrows():
        P_sub = load_sub_values(r["path"])
        D_f3_list.append((P_sub - s1_vals).ravel())
    D_mat_f3 = np.array(D_f3_list, dtype=np.float64)  # (17, 370260)

    novelty_results = {}
    for c_name, c_mat in candidates_vals.items():
        d_vec = (c_mat - p1_vals).ravel()
        mean_d2 = float(np.mean(d_vec ** 2))

        G_sub = (D_mat_f3 @ D_mat_f3.T) / D_mat_f3.shape[1]
        rhs = (D_mat_f3 @ d_vec) / D_mat_f3.shape[1]
        w_sub, V_sub = np.linalg.eigh(G_sub)
        k_sub = w_sub > 1e-3 * w_sub.max()
        alpha = (V_sub[:, k_sub] / w_sub[k_sub]) @ (V_sub[:, k_sub].T @ rhs)

        d_proj = alpha @ D_mat_f3
        res = d_vec - d_proj
        mean_r2 = float(np.mean(res ** 2))
        novelty_ratio = float(mean_r2 / mean_d2)

        # Height and AGB separately
        d_h = c_mat[:, 0] - p1_vals[:, 0]
        d_a = c_mat[:, 1] - p1_vals[:, 1]
        res_mat = res.reshape(-1, 2)
        nov_h = float(np.mean(res_mat[:, 0] ** 2) / np.mean(d_h ** 2))
        nov_a = float(np.mean(res_mat[:, 1] ** 2) / np.mean(d_a ** 2))

        novelty_results[c_name] = {
            "mean_d2": mean_d2,
            "mean_r2": mean_r2,
            "novelty_ratio_total": novelty_ratio,
            "novelty_ratio_h": nov_h,
            "novelty_ratio_a": nov_a
        }
        print(f"Novelty {c_name} (d = G - P1): Total={novelty_ratio*100:.2f}%, Height={nov_h*100:.2f}%, AGB={nov_a*100:.2f}%", flush=True)

    # -------------------------------------------------------------
    # Step 8: Acceptance Checks
    # -------------------------------------------------------------
    print("\n[Step 8] Running Acceptance Checks...", flush=True)
    chk_g1 = kt.check_submission_csv(csv_paths["G1"], SAMPLE_CSV_PATH)
    chk_g2 = kt.check_submission_csv(csv_paths["G2"], SAMPLE_CSV_PATH)
    print(f"Check G1: {chk_g1}", flush=True)
    print(f"Check G2: {chk_g2}", flush=True)
    assert all(chk_g1[k] for k in ["rows_match_185130", "ids_match_sample_order", "id_format_ok", "no_nan_inf", "scale_sanity"])
    assert all(chk_g2[k] for k in ["rows_match_185130", "ids_match_sample_order", "id_format_ok", "no_nan_inf", "scale_sanity"])

    # Model reload re-prediction test
    print("Reloading saved models to verify exact re-prediction...", flush=True)
    reload_diffs = {}
    for m in ["G1", "G2"]:
        feat_key = "46" if m == "G1" else "34"
        reloaded_h = [joblib.load(model_paths[f"{m}_h_seed{s}.joblib"]) for s in SEEDS]
        reloaded_a = [joblib.load(model_paths[f"{m}_a_seed{s}.joblib"]) for s in SEEDS]

        scen_parts = []
        for scen in ["ssp126", "ssp585"]:
            X_t = test_features[scen][feat_key]
            y0_b = test_y0_phys[scen]
            p_dh = np.mean([mod.predict(X_t) for mod in reloaded_h], axis=0)
            p_da = np.mean([mod.predict(X_t) for mod in reloaded_a], axis=0)
            p_h = np.minimum(np.clip(y0_b[:, 0] + p_dh, 0, 35) / SCALE[0], H_CAP)
            p_a = np.maximum(y0_b[:, 1] + p_da, 0) / SCALE[1]
            scen_parts.append(np.column_stack([p_h, p_a]))

        reloaded_p = np.vstack(scen_parts)
        csv_p = candidates_vals[m]
        diff_val = float(np.max(np.abs(reloaded_p - csv_p)))
        reload_diffs[m] = diff_val
        print(f"Reload re-prediction diff {m}: {diff_val:.2e} (<= 1e-5)", flush=True)
        assert diff_val <= 1e-5, f"Reload diff failed for {m}: {diff_val}"

    # Check SHA256 of S1, F3, P1 remained unchanged
    sha_s1_post = sha256_file(S1_CSV_PATH)
    sha_f3_post = sha256_file(F3_CSV_PATH)
    sha_p1_post = sha256_file(P1_CSV_PATH)
    assert sha_s1_post == EXPECTED_S1_SHA, "S1 altered!"
    assert sha_f3_post == EXPECTED_F3_SHA, "F3 altered!"
    assert sha_p1_post == EXPECTED_P1_SHA, "P1 altered!"
    print("Baseline SHA256 integrity confirmed post-execution.", flush=True)

    # -------------------------------------------------------------
    # Step 9: Save Metrics and Metadata
    # -------------------------------------------------------------
    code_hashes = {
        "run_full_sites.py": sha256_file(CODE_DIR / "run_full_sites.py"),
        "prepare_inputs.py": sha256_file(CODE_DIR / "prepare_inputs.py"),
        "features.py": sha256_file(CODE_DIR / "features.py"),
        "restore_y0.py": sha256_file(CODE_DIR / "restore_y0.py"),
        "train_s1.py": sha256_file(CODE_DIR / "train_s1.py")
    }

    t_total = time.time() - t_start
    _, current_peak_bytes = tracemalloc.get_traced_memory()
    total_peak_mb = current_peak_bytes / (1024 * 1024)

    metrics_out = {
        "task": "Full Sites Models G1 and G2 (Revised Holdout 3924)",
        "elapsed_seconds": t_total,
        "peak_memory_mb": total_peak_mb,
        "synonymy_check": {
            "max_diff_y0": max_diff_y0,
            "max_diff_y40": max_diff_y40,
            "max_diff_features": max_diff_feats,
            "passed": True
        },
        "training_data": {
            "n_sites": n_train_sites,
            "n_rows": n_train_rows,
            "holdout_sites": n_holdout_sites,
            "holdout_rows": n_holdout_rows,
            "fresh_sites_excluded": n_fresh_sites
        },
        "optimal_iterations": {
            "folds": opt_iterations,
            "medians": median_iters
        },
        "holdout_evaluation": holdout_results,
        "comparison_diagnostics": comparison_results,
        "novelty_diagnostics": novelty_results,
        "acceptance_checks": {
            "check_g1": chk_g1,
            "check_g2": chk_g2,
            "reload_diffs": reload_diffs,
            "s1_sha_match": bool(sha_s1_post == EXPECTED_S1_SHA),
            "f3_sha_match": bool(sha_f3_post == EXPECTED_F3_SHA),
            "p1_sha_match": bool(sha_p1_post == EXPECTED_P1_SHA)
        },
        "csv_hashes": csv_hashes,
        "model_hashes": model_hashes,
        "code_hashes": code_hashes
    }

    metrics_file = OUT_DIR / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)
    print(f"Saved complete metrics to {metrics_file}", flush=True)
    return metrics_out


if __name__ == "__main__":
    run()

"""Antigravity Step 1: Error Decomposition and Input Verification (2026-11-02)
Execution by Antigravity.
Strict boundaries:
- No Kaggle API calls, no submission generation, no uploads, no credentials reading.
- No modifications to existing files.
- Parts 1-3 strictly use 3,373 baseline sites.
- Part 4 evaluates 3,924 holdout sites (protected minus fresh).
- ZERO reading of 1,024 fresh confirmation sites labels.
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

# Path setup
ROOT = Path("data/project")
CODE_DIR = ROOT / "code/scripts/step1_decomposition_20261102"
OUT_DIR = ROOT / "results/step1_decomposition_20261102"
KAGGLE_DIR = Path("data/kaggle")
DATA_GLOBAL_DIR = Path("data/dataset/data_global")
DATA_STATS_DIR = Path("data/dataset/data_stats")
MANIFEST_PATH = ROOT / "results/competition_specific_plan_20261013/site_manifests.npz"
SPLITS_PATH = ROOT / "results/repro_package_v2_20261101/inputs/site_splits.npz"
PREPARED_INPUTS_DIR = ROOT / "results/competition_signed_training_20261013"
S1_CSV_PATH = ROOT / "results/delta_profile_20261101/candidate_S1_delta_profile_3373.csv"
EXPECTED_S1_SHA = "23216519297719e207970ecd37151181b57cb53d1e23a11499ca33e7b0824aba"

# Model paths for Part IV
G1_MODELS_DIR = ROOT / "results/full_sites_20261102/models"
S1_MODELS_DIR = ROOT / "results/delta_profile_20261101/models"

sys.path[:0] = [
    str(ROOT / ".runtime/science-cp314"),
    str(ROOT / "code"),
    str(CODE_DIR)
]

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from features import H_CAP, SCALE, SEED_AGES, extract_all_features
from restore_y0 import restore_initial_y0

SEEDS_5 = [20261101, 20261102, 20261103, 20261104, 20261105]
SEEDS_3 = [20261101, 20261102, 20261103]
AGES = np.array(SEED_AGES)


def sha256_file(p: Path | str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_scaled_mse(pred_y40: np.ndarray, true_y40: np.ndarray) -> dict:
    """Computes competition-scaled MSE metrics.
    pred_y40, true_y40: (N, 2) in physical units [height, agb].
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


def shift_profile(p_y: np.ndarray, dt: int) -> np.ndarray:
    """Linearly interpolates a site profile across 15 ages to age + dt, capped at age 500.
    Args:
        p_y: shape (15, N_sites, 2)
        dt: time shift in years
    Returns:
        shape (15, N_sites, 2)
    """
    out = np.zeros_like(p_y)
    n_sites = p_y.shape[1]
    for s in range(n_sites):
        for v in range(2):
            prof = p_y[:, s, v]
            out[:, s, v] = np.interp(AGES + dt, AGES, prof, right=prof[-1])
    return out


def run():
    t_start = time.time()
    tracemalloc.start()

    print("=== Antigravity Step 1: Error Decomposition & Input Verification ===", flush=True)

    # Pre-execution checks
    if OUT_DIR.exists():
        raise RuntimeError(f"Output directory {OUT_DIR} already exists. Aborting as required by task book.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check S1 baseline SHA256 integrity
    s1_sha_pre = sha256_file(S1_CSV_PATH)
    assert s1_sha_pre == EXPECTED_S1_SHA, f"S1 baseline SHA256 mismatch: {s1_sha_pre} vs {EXPECTED_S1_SHA}"
    print(f"Verified S1 reference file SHA256 integrity: {s1_sha_pre}", flush=True)

    # -----------------------------------------------------------------
    # Initial Question: Answer regarding pre-revision G1 run
    # -----------------------------------------------------------------
    initial_question_answer = {
        "question": "修订前那次G1运行（按4948个站点留出）在停止前，是否已把fresh_confirmation站点的标签读入内存，或写出过任何包含它们的文件或日志？",
        "read_into_memory": "是",
        "read_into_memory_detail": "在修订前脚本执行 Step 2 时，数据加载函数按原始划分提取 4,948 个留出站点，将包含 1,024 个 fresh_confirmation 站点的标签数组载入内存局部变量中。",
        "evaluated_or_written": "否",
        "evaluated_or_written_detail": "在收到用户停止指令时，旧任务正处于 Step 4 训练第一个模型（仅生成了训练集权重 G1_h_seed20261101.joblib），尚未执行 Step 6 留出评估。因此，未对这 1,024 个站点进行过任何评估计算，亦未写出过任何包含其标签、特征或评估结果的文件或日志。",
        "action_taken": "收到修订要求后已立即中止旧任务，并彻底清理了临时输出目录。本轮评估采用硬断言防护，严禁读取这 1,024 个站点的标签。"
    }

    # -----------------------------------------------------------------
    # Part I: Input Verification (V0)
    # -----------------------------------------------------------------
    print("\n--- [Part I] Input Verification (V0) ---", flush=True)

    # 1. SSP Initial State Verification
    ssp126_npz_path = KAGGLE_DIR / "test_ssp126.npz"
    ssp585_npz_path = KAGGLE_DIR / "test_ssp585.npz"

    with np.load(ssp126_npz_path, mmap_mode="r") as d126, np.load(ssp585_npz_path, mmap_mode="r") as d585:
        y0_126_raw = d126["test_y0"]  # shape (6171, 15, 7)
        y0_585_raw = d585["test_y0"]  # shape (6171, 15, 7)

    ssp_comp_rows = []
    for i, age in enumerate(SEED_AGES):
        for tgt_idx, tgt_name in [(0, "Height"), (1, "AGB")]:
            v126 = y0_126_raw[:, i, tgt_idx]
            v585 = y0_585_raw[:, i, tgt_idx]
            mae = float(np.mean(np.abs(v126 - v585)))
            exact_match_ratio = float(np.mean(v126 == v585))
            corr = float(np.corrcoef(v126, v585)[0, 1])
            m126 = float(np.mean(v126))
            m585 = float(np.mean(v585))
            ssp_comp_rows.append({
                "Age": age,
                "Target": tgt_name,
                "SSP126_Mean": m126,
                "SSP585_Mean": m585,
                "MAE": mae,
                "Exact_Match_Ratio": exact_match_ratio,
                "Pearson_Corr": corr
            })

    ssp_comp_df = pd.DataFrame(ssp_comp_rows)
    ssp_comp_csv = OUT_DIR / "table0_input_check_ssp.csv"
    ssp_comp_df.to_csv(ssp_comp_csv, index=False)
    print(f"Saved SSP initial state comparison to {ssp_comp_csv}", flush=True)

    # 2. Physical Invariants
    # Load 3373 historical data
    december_Y_path = PREPARED_INPUTS_DIR / "december_Y.npy"
    monthly_X_path = PREPARED_INPUTS_DIR / "monthly_X.npy"
    dec_y_3373 = np.load(december_Y_path, mmap_mode="r")  # (15, 3373, 41, 7)
    mon_x_3373 = np.load(monthly_X_path, mmap_mode="r")  # (3373, 40, 12, 136)

    y0_hist_phys = dec_y_3373[:, :, 0, :2]  # (15, 3373, 2)
    y0_ssp126_phys = restore_initial_y0(y0_126_raw)  # (6171, 15, 2)
    y0_ssp585_phys = restore_initial_y0(y0_585_raw)  # (6171, 15, 2)

    invariants_rows = []
    datasets_dict = {
        "Historical_3373": (y0_hist_phys.transpose(1, 0, 2), 3373),  # (3373, 15, 2)
        "SSP126_restored": (y0_ssp126_phys, 6171),                    # (6171, 15, 2)
        "SSP585_restored": (y0_ssp585_phys, 6171)                     # (6171, 15, 2)
    }

    invariants_summary = {}
    for dname, (arr, n_sites) in datasets_dict.items():
        h = arr[:, :, 0]
        a = arr[:, :, 1]
        h_max = float(np.max(h))
        h_seedling_min = float(np.min(h[:, 0]))
        a_max = float(np.max(a))
        a_seedling_min = float(np.min(a[:, 0]))

        invariants_summary[dname] = {
            "h_max": h_max,
            "h_seedling_min": h_seedling_min,
            "a_max": a_max,
            "a_seedling_min": a_seedling_min
        }

        for i, age in enumerate(SEED_AGES):
            h_age = h[:, i]
            a_age = a[:, i]
            invariants_rows.append({
                "Dataset": dname,
                "Age": age,
                "Height_Max": h_max if i == 0 else "",
                "Height_Seedling_Min": h_seedling_min if i == 0 else "",
                "Height_Zero_Ratio": float(np.mean(h_age == 0.0)),
                "Height_Mean": float(np.mean(h_age)),
                "AGB_Max": a_max if i == 0 else "",
                "AGB_Zero_Ratio": float(np.mean(a_age == 0.0)),
                "AGB_Mean": float(np.mean(a_age))
            })

    invariants_df = pd.DataFrame(invariants_rows)
    invariants_csv = OUT_DIR / "table0_physical_invariants.csv"
    invariants_df.to_csv(invariants_csv, index=False)
    print(f"Saved physical invariants to {invariants_csv}", flush=True)

    # 3. Station Coordinates Verification
    mask_path = DATA_STATS_DIR / "mask2d.npy"
    assert mask_path.exists(), f"mask2d.npy not found: {mask_path}"
    mask2d = np.load(mask_path)
    rc_grid = np.argwhere(mask2d > 0)  # (54152, 2)

    splits = np.load(SPLITS_PATH)
    cells_3373 = splits["cells"]  # (3373,)
    row_3373, col_3373 = rc_grid[cells_3373, 0], rc_grid[cells_3373, 1]

    # Spearman correlation between multi-year mean of channel 0 and abs(row - 179.5)
    ch0_multiyear_mean = np.mean(mon_x_3373[:, :, :, 0], axis=(1, 2))
    lat_dist_equator = np.abs(row_3373 - 179.5)
    spearman_corr, _ = spearmanr(ch0_multiyear_mean, lat_dist_equator)
    abs_spearman = float(np.abs(spearman_corr))
    print(f"Coordinate validation Spearman correlation: {spearman_corr:.6f} (|corr| = {abs_spearman:.6f})", flush=True)
    assert abs_spearman >= 0.7, f"Coordinate check failed: |corr| = {abs_spearman} < 0.7"
    print("Coordinate validation PASSED (>= 0.7).", flush=True)

    # -----------------------------------------------------------------
    # Part II: Spatial Block Cross-Validation
    # -----------------------------------------------------------------
    print("\n--- [Part II] Spatial Block Cross-Validation ---", flush=True)
    block_3373 = (row_3373 // 20) * 1000 + (col_3373 // 20)
    unique_blocks = np.unique(block_3373)
    n_blocks = len(unique_blocks)
    print(f"Identified {n_blocks} unique 10-degree spatial blocks across 3,373 sites.", flush=True)

    block_by_row = np.repeat(block_3373, 15)
    site_idx_by_row = np.repeat(np.arange(3373), 15)

    gkf = GroupKFold(n_splits=5)
    folds_info = []
    folds_split = list(gkf.split(np.zeros((len(cells_3373) * 15, 1)), groups=block_by_row))

    for fold_id, (trn_idx, val_idx) in enumerate(folds_split):
        val_sites = np.unique(site_idx_by_row[val_idx])
        trn_sites = np.unique(site_idx_by_row[trn_idx])
        assert len(np.intersect1d(trn_sites, val_sites)) == 0, f"Fold {fold_id} has site leakage!"
        assert len(val_idx) == len(val_sites) * 15, f"Fold {fold_id} row count mismatch!"
        folds_info.append({
            "fold": fold_id,
            "val_sites": int(len(val_sites)),
            "val_rows": int(len(val_idx)),
            "trn_sites": int(len(trn_sites)),
            "trn_rows": int(len(trn_idx))
        })
        print(f"  Fold {fold_id}: val_sites={len(val_sites)}, val_rows={len(val_idx)}", flush=True)

    # Extract features for 3,373 sites
    print("Extracting S1 (46-dim) features for 3,373 sites...", flush=True)
    y0_train_3373 = dec_y_3373[:, :, 0, :2].transpose(1, 0, 2).astype(np.float32)  # (3373, 15, 2)
    y40_train_3373 = dec_y_3373[:, :, 40, :2].transpose(1, 0, 2).astype(np.float32)  # (3373, 15, 2)
    X_train_46 = extract_all_features(y0_train_3373, mon_x_3373)  # (50595, 46)
    X_train_34 = X_train_46[:, :34].copy()

    y0_flat = y0_train_3373.reshape(-1, 2)  # (50595, 2)
    y40_flat = y40_train_3373.reshape(-1, 2)  # (50595, 2)
    delta_flat = y40_flat - y0_flat

    # Baselines (Persistence and B)
    # 1. Persistence
    pred_pers = y0_flat.copy()
    pers_overall = compute_scaled_mse(pred_pers, y40_flat)
    print(f"Persistence: score={pers_overall['score']:.6f}, H={pers_overall['mse_h']:.6f}, A={pers_overall['mse_a']:.6f}", flush=True)
    assert abs(pers_overall['score'] - 0.2919) <= 1e-3, f"Persistence score replication failed: {pers_overall['score']}"

    # 2. B Baseline (linear interpolation of y0 profile to age+40)
    pred_B = np.zeros_like(y0_train_3373)
    for s in range(3373):
        for v in range(2):
            prof = y0_train_3373[s, :, v]
            pred_B[s, :, v] = np.interp(AGES + 40, AGES, prof, right=prof[-1])
    pred_B_flat = pred_B.reshape(-1, 2)
    b_overall = compute_scaled_mse(pred_B_flat, y40_flat)
    print(f"B baseline: score={b_overall['score']:.6f}, H={b_overall['mse_h']:.6f}, A={b_overall['mse_a']:.6f}", flush=True)
    assert abs(b_overall['score'] - 0.1760) <= 1e-3, f"B baseline score replication failed: {b_overall['score']}"

    # Train S1 and M1 across 5 folds and 5 seeds
    print("\nFitting S1 and M1 models (5 folds x 5 seeds)...", flush=True)
    oof_s1_seeds = np.zeros((len(SEEDS_5), len(y0_flat), 2), dtype=np.float32)
    oof_m1_seeds = np.zeros((len(SEEDS_5), len(y0_flat), 2), dtype=np.float32)

    # Fold assignments array
    fold_assignment = np.zeros(len(y0_flat), dtype=np.int32)
    for f_id, (_, val_idx) in enumerate(folds_split):
        fold_assignment[val_idx] = f_id

    fit_start_time = time.time()
    for fold_id, (trn_idx, val_idx) in enumerate(folds_split):
        f_t0 = time.time()
        for seed_idx, seed in enumerate(SEEDS_5):
            # S1 Height (1242 iters)
            reg_s1_h = HistGradientBoostingRegressor(
                learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50,
                max_features=0.8, l2_regularization=0.0, early_stopping=False,
                max_iter=1242, random_state=seed
            )
            reg_s1_h.fit(X_train_46[trn_idx], delta_flat[trn_idx, 0])
            dh_s1 = reg_s1_h.predict(X_train_46[val_idx])

            # S1 AGB (1175 iters)
            reg_s1_a = HistGradientBoostingRegressor(
                learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50,
                max_features=0.8, l2_regularization=0.0, early_stopping=False,
                max_iter=1175, random_state=seed
            )
            reg_s1_a.fit(X_train_46[trn_idx], delta_flat[trn_idx, 1])
            da_s1 = reg_s1_a.predict(X_train_46[val_idx])

            oof_s1_seeds[seed_idx, val_idx, 0] = np.clip(y0_flat[val_idx, 0] + dh_s1, 0.0, 35.0)
            oof_s1_seeds[seed_idx, val_idx, 1] = np.maximum(y0_flat[val_idx, 1] + da_s1, 0.0)

            # M1 Height (872 iters)
            reg_m1_h = HistGradientBoostingRegressor(
                learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50,
                max_features=0.8, l2_regularization=0.0, early_stopping=False,
                max_iter=872, random_state=seed
            )
            reg_m1_h.fit(X_train_34[trn_idx], delta_flat[trn_idx, 0])
            dh_m1 = reg_m1_h.predict(X_train_34[val_idx])

            # M1 AGB (61 iters)
            reg_m1_a = HistGradientBoostingRegressor(
                learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50,
                max_features=0.8, l2_regularization=0.0, early_stopping=False,
                max_iter=61, random_state=seed
            )
            reg_m1_a.fit(X_train_34[trn_idx], delta_flat[trn_idx, 1])
            da_m1 = reg_m1_a.predict(X_train_34[val_idx])

            oof_m1_seeds[seed_idx, val_idx, 0] = np.clip(y0_flat[val_idx, 0] + dh_m1, 0.0, 35.0)
            oof_m1_seeds[seed_idx, val_idx, 1] = np.maximum(y0_flat[val_idx, 1] + da_m1, 0.0)

        elapsed_f = time.time() - f_t0
        if fold_id == 0:
            est_total_hours = (elapsed_f * 5.0) / 3600.0
            print(f"[Resource Monitor] Fold 0 completed in {elapsed_f:.2f}s. Estimated fitting runtime: {est_total_hours:.3f} hours (<= 4.0h limit).", flush=True)
            assert est_total_hours <= 4.0, f"Estimated runtime exceeds 4 hours: {est_total_hours}"
        print(f"  Fold {fold_id} completed in {elapsed_f:.2f}s.", flush=True)

    oof_s1_avg = np.mean(oof_s1_seeds, axis=0)  # (50595, 2)
    oof_m1_avg = np.mean(oof_m1_seeds, axis=0)  # (50595, 2)

    s1_spatial_overall = compute_scaled_mse(oof_s1_avg, y40_flat)
    m1_spatial_overall = compute_scaled_mse(oof_m1_avg, y40_flat)
    print(f"Spatial OOF S1: score={s1_spatial_overall['score']:.6f}, H={s1_spatial_overall['mse_h']:.6f}, A={s1_spatial_overall['mse_a']:.6f}", flush=True)
    print(f"Spatial OOF M1: score={m1_spatial_overall['score']:.6f}, H={m1_spatial_overall['mse_h']:.6f}, A={m1_spatial_overall['mse_a']:.6f}", flush=True)

    # -----------------------------------------------------------------
    # Part III: Error Decomposition
    # -----------------------------------------------------------------
    print("\n--- [Part III] Error Decomposition ---", flush=True)
    # Compute collapse flags
    # dec_y_3373: (15, 3373, 41, 7) -> transpose to (3373, 15, 41, 7) -> (50595, 41, 7)
    Y_series = dec_y_3373.transpose(1, 0, 2, 3).reshape(3373 * 15, 41, 7)
    z_h = Y_series[:, :, 0] / SCALE[0]
    z_a = Y_series[:, :, 1] / SCALE[1]

    diff_h = z_h[:, 1:] - z_h[:, :-1]  # (50595, 40)
    diff_a = z_a[:, 1:] - z_a[:, :-1]  # (50595, 40)

    m_h = np.min(diff_h, axis=1)
    m_a = np.min(diff_a, axis=1)

    c20_h = m_h <= -0.2
    c50_h = m_h <= -0.5
    c20_a = m_a <= -0.2
    c50_a = m_a <= -0.5
    d_tag = (Y_series[:, 0, 0] > 0.0) & (Y_series[:, 40, 0] == 0.0)

    # Compute S1 and M1 squared errors
    p_s1_h = np.minimum(np.clip(oof_s1_avg[:, 0], 0, 35) / SCALE[0], H_CAP)
    p_s1_a = np.maximum(oof_s1_avg[:, 1], 0) / SCALE[1]
    t_h = np.minimum(np.clip(y40_flat[:, 0], 0, 35) / SCALE[0], H_CAP)
    t_a = np.maximum(y40_flat[:, 1], 0) / SCALE[1]

    se_s1_h = (p_s1_h - t_h) ** 2
    se_s1_a = (p_s1_a - t_a) ** 2
    se_s1_comb = se_s1_h + se_s1_a

    p_m1_h = np.minimum(np.clip(oof_m1_avg[:, 0], 0, 35) / SCALE[0], H_CAP)
    p_m1_a = np.maximum(oof_m1_avg[:, 1], 0) / SCALE[1]
    se_m1_h = (p_m1_h - t_h) ** 2
    se_m1_a = (p_m1_a - t_a) ** 2
    se_m1_comb = se_m1_h + se_m1_a

    # Threshold share calculation
    # Target-specific C20:
    se_s1_c20_h = np.sum(se_s1_h[c20_h])
    se_s1_all_h = np.sum(se_s1_h)
    share_h = se_s1_c20_h / se_s1_all_h

    se_s1_c20_a = np.sum(se_s1_a[c20_a])
    se_s1_all_a = np.sum(se_s1_a)
    share_a = se_s1_c20_a / se_s1_all_a

    # Combined definition 1: Target-wise C20 error sum
    share_comb_targetwise = (se_s1_c20_h + se_s1_c20_a) / (se_s1_all_h + se_s1_all_a)

    # Combined definition 2: Row marked C20 if c20_h | c20_a
    c20_either = c20_h | c20_a
    share_comb_rowwise = np.sum(se_s1_comb[c20_either]) / np.sum(se_s1_comb)

    # Cluster bootstrap by 10-degree block (2000 replicates, seed 20261102)
    print("Running 10-degree block cluster bootstrap (2000 replicates, seed 20261102)...", flush=True)
    rng = np.random.default_rng(20261102)

    # Map block to integer index 0..237
    block_map = {b: i for i, b in enumerate(unique_blocks)}
    row_block_idx = np.array([block_map[b] for b in block_by_row])

    # Pre-aggregate by block
    # Definition 1: target-wise
    c20_tw_err = np.where(c20_h, se_s1_h, 0.0) + np.where(c20_a, se_s1_a, 0.0)
    block_tw_c20 = np.bincount(row_block_idx, weights=c20_tw_err, minlength=n_blocks)
    block_tw_all = np.bincount(row_block_idx, weights=se_s1_comb, minlength=n_blocks)

    # Definition 2: row-wise (either)
    c20_rw_err = np.where(c20_either, se_s1_comb, 0.0)
    block_rw_c20 = np.bincount(row_block_idx, weights=c20_rw_err, minlength=n_blocks)

    boot_indices = rng.integers(0, n_blocks, size=(2000, n_blocks))
    boot_shares_tw = np.sum(block_tw_c20[boot_indices], axis=1) / np.sum(block_tw_all[boot_indices], axis=1)
    boot_shares_rw = np.sum(block_rw_c20[boot_indices], axis=1) / np.sum(block_tw_all[boot_indices], axis=1)

    ci_tw_low, ci_tw_high = float(np.percentile(boot_shares_tw, 2.5)), float(np.percentile(boot_shares_tw, 97.5))
    ci_rw_low, ci_rw_high = float(np.percentile(boot_shares_rw, 2.5)), float(np.percentile(boot_shares_rw, 97.5))

    decision_tw = "继续做崩塌层" if share_comb_targetwise >= 0.30 else "放弃崩塌层，只做B和r"
    decision_rw = "继续做崩塌层" if share_comb_rowwise >= 0.30 else "放弃崩塌层，只做B和r"

    print(f"Collapse Share (Target-wise C20): {share_comb_targetwise:.4f} (95% CI: [{ci_tw_low:.4f}, {ci_tw_high:.4f}]) -> Decision: {decision_tw}", flush=True)
    print(f"Collapse Share (Row-wise C20):    {share_comb_rowwise:.4f} (95% CI: [{ci_rw_low:.4f}, {ci_rw_high:.4f}]) -> Decision: {decision_rw}", flush=True)

    # -------------------------------------------------------------
    # Tables 1 to 4 Generation
    # -------------------------------------------------------------
    # Table 1: Overall table
    # Reference values from 新方向模型结果_20261101.md
    ref_site_split = {
        "persistence": {"score": 0.291865, "mse_h": 0.254467, "mse_a": 0.329263, "old_h": 0.137993},
        "S1":          {"score": 0.051046, "mse_h": 0.032679, "mse_a": 0.069413, "old_h": 0.042846},
        "M1":          {"score": 0.073548, "mse_h": 0.042043, "mse_a": 0.105053, "old_h": 0.053374}
    }

    row_ages = np.repeat(SEED_AGES, 3373)  # wait, row order: site outer, age inner!
    # Let's check row order: site outer, age inner: row i is site (i // 15), age (SEED_AGES[i % 15])
    row_ages = np.array([SEED_AGES[i % 15] for i in range(len(y0_flat))])
    old_mask = row_ages >= 150

    models_eval = {
        "Persistence": {"pred": pred_pers, "dim": "-"},
        "B":           {"pred": pred_B_flat, "dim": "-"},
        "S1":          {"pred": oof_s1_avg, "dim": "46维"},
        "M1":          {"pred": oof_m1_avg, "dim": "34维"}
    }

    table1_rows = []
    for mname, minfo in models_eval.items():
        res = compute_scaled_mse(minfo["pred"], y40_flat)
        old_h_mse = float(np.mean(
            (np.minimum(np.clip(minfo["pred"][old_mask, 0], 0, 35) / SCALE[0], H_CAP) -
             np.minimum(np.clip(y40_flat[old_mask, 0], 0, 35) / SCALE[0], H_CAP)) ** 2
        ))
        ref_sc = ref_site_split[mname]["score"] if mname in ref_site_split else ""
        ref_h = ref_site_split[mname]["mse_h"] if mname in ref_site_split else ""
        ref_a = ref_site_split[mname]["mse_a"] if mname in ref_site_split else ""
        ref_old_h = ref_site_split[mname]["old_h"] if mname in ref_site_split else ""

        table1_rows.append({
            "Model": mname,
            "Feature_Dim": minfo["dim"],
            "SpatialBlock_Score": res["score"],
            "SpatialBlock_Height_MSE": res["mse_h"],
            "SpatialBlock_AGB_MSE": res["mse_a"],
            "SpatialBlock_H_Old_MSE": old_h_mse,
            "SiteSplit_Score_Ref": ref_sc,
            "SiteSplit_Height_MSE_Ref": ref_h,
            "SiteSplit_AGB_MSE_Ref": ref_a,
            "SiteSplit_H_Old_MSE_Ref": ref_old_h
        })

    table1_df = pd.DataFrame(table1_rows)
    table1_csv = OUT_DIR / "table1_overall.csv"
    table1_df.to_csv(table1_csv, index=False)
    print(f"Saved Table 1 (Overall) to {table1_csv}", flush=True)

    # Table 2: By-age table
    table2_rows = []
    for age in SEED_AGES:
        mask_a = row_ages == age
        row_dict = {"Age": age}
        for mname, minfo in models_eval.items():
            sc = compute_scaled_mse(minfo["pred"][mask_a], y40_flat[mask_a])["score"]
            row_dict[f"{mname}_SpatialBlock"] = sc
        table2_rows.append(row_dict)

    table2_df = pd.DataFrame(table2_rows)
    table2_csv = OUT_DIR / "table2_by_age.csv"
    table2_df.to_csv(table2_csv, index=False)
    print(f"Saved Table 2 (By-Age) to {table2_csv}", flush=True)

    # Table 3: Collapse table
    # Categories: Non-collapse, C20, C50, D
    # For Height:
    non_c20_h = ~c20_h
    cat_h_masks = {
        "Non_Collapse": non_c20_h,
        "C20": c20_h,
        "C50": c50_h,
        "D": d_tag
    }

    # For AGB:
    non_c20_a = ~c20_a
    cat_a_masks = {
        "Non_Collapse": non_c20_a,
        "C20": c20_a,
        "C50": c50_a,
        "D": d_tag
    }

    table3_rows = []
    n_total = len(y0_flat)
    for tgt_name, masks, se_s1, se_m1 in [
        ("Height", cat_h_masks, se_s1_h, se_m1_h),
        ("AGB", cat_a_masks, se_s1_a, se_m1_a)
    ]:
        tot_se_s1 = np.sum(se_s1)
        tot_se_m1 = np.sum(se_m1)
        for cat_name, cmask in masks.items():
            cnt = int(np.sum(cmask))
            s1_sum = float(np.sum(se_s1[cmask]))
            m1_sum = float(np.sum(se_m1[cmask]))
            s1_mse = float(np.mean(se_s1[cmask])) if cnt > 0 else 0.0
            m1_mse = float(np.mean(se_m1[cmask])) if cnt > 0 else 0.0
            table3_rows.append({
                "Target": tgt_name,
                "Category": cat_name,
                "Row_Count": cnt,
                "Row_Ratio": cnt / n_total,
                "S1_SE_Sum": s1_sum,
                "S1_SE_Ratio": s1_sum / tot_se_s1,
                "S1_MSE": s1_mse,
                "M1_SE_Sum": m1_sum,
                "M1_SE_Ratio": m1_sum / tot_se_m1,
                "M1_MSE": m1_mse
            })

    table3_df = pd.DataFrame(table3_rows)
    table3_csv = OUT_DIR / "table3_collapse.csv"
    table3_df.to_csv(table3_csv, index=False)
    print(f"Saved Table 3 (Collapse) to {table3_csv}", flush=True)

    # Table 4: Age x Collapse table
    # Age bins: <=30, 50-100, >=150
    age_groups = {
        "<=30": row_ages <= 30,
        "50-100": (row_ages >= 50) & (row_ages <= 100),
        ">=150": row_ages >= 150
    }

    table4_rows = []
    for grp_name, gmask in age_groups.items():
        for tgt_name, c20_mask, se_s1, se_m1 in [
            ("Height", c20_h, se_s1_h, se_m1_h),
            ("AGB", c20_a, se_s1_a, se_m1_a)
        ]:
            tot_grp_se = np.sum(se_s1[gmask])
            for status_name, status_mask in [("Non_Collapse", ~c20_mask), ("C20", c20_mask)]:
                sub_mask = gmask & status_mask
                cnt = int(np.sum(sub_mask))
                s1_se = float(np.sum(se_s1[sub_mask]))
                s1_mse = float(np.mean(se_s1[sub_mask])) if cnt > 0 else 0.0
                m1_mse = float(np.mean(se_m1[sub_mask])) if cnt > 0 else 0.0
                table4_rows.append({
                    "Age_Group": grp_name,
                    "Target": tgt_name,
                    "Status": status_name,
                    "Row_Count": cnt,
                    "Row_Ratio_In_AgeGroup": cnt / int(np.sum(gmask)),
                    "S1_SE_Ratio_In_AgeGroup": s1_se / tot_grp_se if tot_grp_se > 0 else 0.0,
                    "S1_MSE": s1_mse,
                    "M1_MSE": m1_mse
                })

    table4_df = pd.DataFrame(table4_rows)
    table4_csv = OUT_DIR / "table4_age_x_collapse.csv"
    table4_df.to_csv(table4_csv, index=False)
    print(f"Saved Table 4 (Age x Collapse) to {table4_csv}", flush=True)

    # -------------------------------------------------------------
    # Transient Analysis (Early 10-year transient)
    # -------------------------------------------------------------
    print("\n--- [Transient Analysis] Early 10-year Transient ---", flush=True)
    # dec_y_3373: (15, 3373, 41, 7)
    y0_arr = dec_y_3373[:, :, 0, :2]    # (15, 3373, 2)
    y10_arr = dec_y_3373[:, :, 10, :2]  # (15, 3373, 2)
    y40_arr = dec_y_3373[:, :, 40, :2]  # (15, 3373, 2)

    shift_y0_10 = shift_profile(y0_arr, 10)    # (15, 3373, 2)
    shift_y10_30 = shift_profile(y10_arr, 30)  # (15, 3373, 2)

    # Flatten to (50595, 2) matching row ordering (site outer, age inner)
    # Current shape is (15, 3373, 2), transpose to (3373, 15, 2) then reshape
    e_early_flat = (y10_arr.transpose(1, 0, 2) - shift_y0_10.transpose(1, 0, 2)).reshape(-1, 2) / SCALE
    e_late_flat = (y40_arr.transpose(1, 0, 2) - shift_y10_30.transpose(1, 0, 2)).reshape(-1, 2) / SCALE

    # S1 residual: (y40 - oof_s1) / scale
    res_flat = (y40_flat - oof_s1_avg) / SCALE

    def fit_ols_r2(y_vec, X_mat):
        # Adds intercept
        X_design = np.column_stack([np.ones(len(y_vec)), X_mat])
        beta, residuals, rank, s = np.linalg.lstsq(X_design, y_vec, rcond=None)
        pred = X_design @ beta
        ss_tot = np.sum((y_vec - np.mean(y_vec)) ** 2)
        ss_res = np.sum((y_vec - pred) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        return float(r2)

    table5_rows = []
    # Overall pooled
    for tgt_idx, tgt_name in [(0, "Height"), (1, "AGB")]:
        y_val = res_flat[:, tgt_idx]
        x1 = e_early_flat[:, tgt_idx]
        x2 = e_late_flat[:, tgt_idx]
        r2_1 = fit_ols_r2(y_val, x1[:, None])
        r2_2 = fit_ols_r2(y_val, x2[:, None])
        r2_both = fit_ols_r2(y_val, np.column_stack([x1, x2]))
        m_e1_2 = float(np.mean(x1 ** 2))
        m_e2_2 = float(np.mean(x2 ** 2))
        table5_rows.append({
            "Target": tgt_name,
            "Age": "All (Pooled)",
            "R2_early": r2_1,
            "R2_late": r2_2,
            "R2_both": r2_both,
            "Mean_e_early2": m_e1_2,
            "Mean_e_late2": m_e2_2
        })

    # By-age
    for age in SEED_AGES:
        mask_a = row_ages == age
        for tgt_idx, tgt_name in [(0, "Height"), (1, "AGB")]:
            y_val = res_flat[mask_a, tgt_idx]
            x1 = e_early_flat[mask_a, tgt_idx]
            x2 = e_late_flat[mask_a, tgt_idx]
            r2_1 = fit_ols_r2(y_val, x1[:, None])
            r2_2 = fit_ols_r2(y_val, x2[:, None])
            r2_both = fit_ols_r2(y_val, np.column_stack([x1, x2]))
            m_e1_2 = float(np.mean(x1 ** 2))
            m_e2_2 = float(np.mean(x2 ** 2))
            table5_rows.append({
                "Target": tgt_name,
                "Age": str(age),
                "R2_early": r2_1,
                "R2_late": r2_2,
                "R2_both": r2_both,
                "Mean_e_early2": m_e1_2,
                "Mean_e_late2": m_e2_2
            })

    table5_df = pd.DataFrame(table5_rows)
    table5_csv = OUT_DIR / "table5_transient_ols.csv"
    table5_df.to_csv(table5_csv, index=False)
    print(f"Saved Table 5 (Transient OLS) to {table5_csv}", flush=True)

    # Window table
    # Windows 10, 20; starts 0, 10, 20, 30
    print("Computing window error table and bootstrap difference 95% CIs...", flush=True)
    window_rows = []
    window_diff_cis = {}

    for w in [10, 20]:
        start0_err_block = None
        start10_err_block = None
        for start in [0, 10, 20, 30]:
            if start + w <= 40:
                pred_w = shift_profile(dec_y_3373[:, :, start, :2], w)
                true_w = dec_y_3373[:, :, start + w, :2]

                # (15, 3373, 2) -> (3373, 15, 2) -> (50595, 2)
                p_flat = pred_w.transpose(1, 0, 2).reshape(-1, 2)
                t_flat = true_w.transpose(1, 0, 2).reshape(-1, 2)

                p_h = np.minimum(np.clip(p_flat[:, 0], 0, 35) / SCALE[0], H_CAP)
                p_a = np.maximum(p_flat[:, 1], 0) / SCALE[1]
                t_h = np.minimum(np.clip(t_flat[:, 0], 0, 35) / SCALE[0], H_CAP)
                t_a = np.maximum(t_flat[:, 1], 0) / SCALE[1]

                se_h = (p_h - t_h) ** 2
                se_a = (p_a - t_a) ** 2
                se_comb = 0.5 * (se_h + se_a)

                mse_h = float(np.mean(se_h))
                mse_a = float(np.mean(se_a))
                score = float(np.mean(se_comb))

                block_err = np.bincount(row_block_idx, weights=se_comb, minlength=n_blocks) / np.bincount(row_block_idx, minlength=n_blocks)

                if start == 0:
                    start0_err_block = block_err
                    start0_score = score
                elif start == 10:
                    start10_err_block = block_err

                window_rows.append({
                    "Window": w,
                    "Start": start,
                    "Height_MSE": mse_h,
                    "AGB_MSE": mse_a,
                    "Score": score
                })

        # Bootstrap difference: Start 0 vs Start 10 (Score_start0 - Score_start10)
        diff_block = start0_err_block - start10_err_block
        boot_diffs = np.mean(diff_block[boot_indices], axis=1)
        ci_d_low, ci_d_high = float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))
        window_diff_cis[w] = {
            "diff_point": float(np.mean(diff_block)),
            "ci_95": [ci_d_low, ci_d_high]
        }
        print(f"Window {w}: Start 0 vs Start 10 diff = {np.mean(diff_block):.4f} (95% CI: [{ci_d_low:.4f}, {ci_d_high:.4f}])", flush=True)

    table6_df = pd.DataFrame(window_rows)
    table6_csv = OUT_DIR / "table6_window_errors.csv"
    table6_df.to_csv(table6_csv, index=False)
    print(f"Saved Table 6 (Window Errors) to {table6_csv}", flush=True)

    # -----------------------------------------------------------------
    # Part IV: Full Sites Spatial Proximity Effect
    # -----------------------------------------------------------------
    print("\n--- [Part IV] Full Sites Spatial Proximity Effect ---", flush=True)
    manifest = np.load(MANIFEST_PATH)
    train_g1_cells = manifest["all_public_historical_except_protected"]
    train_s1_cells = manifest["baseline_train_val"]
    protected_cells = manifest["protected_history_confirmation"]
    fresh_cells = manifest["fresh_confirmation"]

    # Strictly 3,924 holdout sites, NO fresh confirmation sites!
    holdout_cells = np.setdiff1d(protected_cells, fresh_cells)
    assert len(holdout_cells) == 3924, f"Holdout cells count mismatch: {len(holdout_cells)} vs 3924"
    assert not np.any(np.isin(holdout_cells, fresh_cells)), "FRESH CONFIRMATION SITES DETECTED IN HOLDOUT! STOPPING!"

    rc_g1 = rc_grid[train_g1_cells]
    rc_s1 = rc_grid[train_s1_cells]
    rc_holdout = rc_grid[holdout_cells]

    tree_g1 = cKDTree(rc_g1)
    dist_to_g1, _ = tree_g1.query(rc_holdout, p=np.inf)

    tree_s1 = cKDTree(rc_s1)
    dist_to_s1, _ = tree_s1.query(rc_holdout, p=np.inf)

    # Extract 3,924 holdout data
    print("Extracting features and targets for 3,924 holdout sites...", flush=True)
    def extract_sites(cells_subset):
        assert not np.any(np.isin(cells_subset, fresh_cells)), "GUARD VIOLATION: fresh site in extraction!"
        N = len(cells_subset)
        x_raw = np.load(DATA_GLOBAL_DIR / "glob_X_fea.npy", mmap_mode="r")
        sub_X = np.empty((N, 40, 12, 136), dtype=np.float32)
        chunk_sz = 4000
        for i in range(0, N, chunk_sz):
            idx = cells_subset[i:i + chunk_sz]
            sub_X[i:i + len(idx)] = np.asarray(x_raw[idx], dtype=np.float32)

        sub_Y = np.empty((len(SEED_AGES), N, 2, 2), dtype=np.float32)
        for a_idx, age in enumerate(SEED_AGES):
            y_f = DATA_GLOBAL_DIR / f"glob_Y{age:03d}.npy"
            if not y_f.exists():
                y_f = DATA_GLOBAL_DIR / f"glob_Y{age}.npy"
            raw_y = np.load(y_f, mmap_mode="r")
            for i in range(0, N, chunk_sz):
                idx = cells_subset[i:i + chunk_sz]
                sub_Y[a_idx, i:i + len(idx), 0, :] = np.asarray(raw_y[idx, 0, 11, :2], dtype=np.float32)
                sub_Y[a_idx, i:i + len(idx), 1, :] = np.asarray(raw_y[idx, 40, 11, :2], dtype=np.float32)

        y0_p = sub_Y[:, :, 0, :].transpose(1, 0, 2)
        y40_p = sub_Y[:, :, 1, :].transpose(1, 0, 2)
        feats_46 = extract_all_features(y0_p, sub_X)
        return feats_46, y0_p.reshape(-1, 2), y40_p.reshape(-1, 2)

    X_h_46, y0_h_flat, y40_h_flat = extract_sites(holdout_cells)
    X_h_34 = X_h_46[:, :34].copy()

    # Load G1 and S1 saved models
    print("Loading saved G1 models (results/full_sites_20261102/models/)...", flush=True)
    g1_h_mods = [joblib.load(G1_MODELS_DIR / f"G1_h_seed{s}.joblib") for s in SEEDS_3]
    g1_a_mods = [joblib.load(G1_MODELS_DIR / f"G1_a_seed{s}.joblib") for s in SEEDS_3]

    print("Loading saved S1 models (results/delta_profile_20261101/models/)...", flush=True)
    s1_h_mods = [joblib.load(S1_MODELS_DIR / f"model_height_seed{s}.joblib") for s in SEEDS_3]
    s1_a_mods = [joblib.load(S1_MODELS_DIR / f"model_agb_seed{s}.joblib") for s in SEEDS_3]

    # Predict
    pred_g1_dh = np.mean([m.predict(X_h_46) for m in g1_h_mods], axis=0)
    pred_g1_da = np.mean([m.predict(X_h_46) for m in g1_a_mods], axis=0)
    pred_g1 = np.column_stack([y0_h_flat[:, 0] + pred_g1_dh, y0_h_flat[:, 1] + pred_g1_da])

    pred_s1_dh = np.mean([m.predict(X_h_46) for m in s1_h_mods], axis=0)
    pred_s1_da = np.mean([m.predict(X_h_46) for m in s1_a_mods], axis=0)
    pred_s1 = np.column_stack([y0_h_flat[:, 0] + pred_s1_dh, y0_h_flat[:, 1] + pred_s1_da])

    # Replicate report scores
    g1_rep_sc = compute_scaled_mse(pred_g1, y40_h_flat)
    s1_rep_sc = compute_scaled_mse(pred_s1, y40_h_flat)
    print(f"G1 3924 holdout score: {g1_rep_sc['score']:.6f} (Report target: 0.014456)", flush=True)
    print(f"S1 3924 holdout score: {s1_rep_sc['score']:.6f} (Report target: 0.033148)", flush=True)
    assert abs(g1_rep_sc["score"] - 0.014456) <= 1e-5, f"G1 holdout replication diff: {abs(g1_rep_sc['score'] - 0.014456)}"
    assert abs(s1_rep_sc["score"] - 0.033148) <= 1e-5, f"S1 holdout replication diff: {abs(s1_rep_sc['score'] - 0.033148)}"
    print("Holdout scores replication PASSED (diff <= 1e-5).", flush=True)

    # Binning by distance to G1
    dist_g1_by_row = np.repeat(dist_to_g1, 15)
    dist_s1_by_row = np.repeat(dist_to_s1, 15)

    def compute_binned_results(dist_site_arr, dist_row_arr):
        bins = [
            ("1", dist_site_arr == 1.0, dist_row_arr == 1.0),
            ("2", dist_site_arr == 2.0, dist_row_arr == 2.0),
            ("3-4", (dist_site_arr >= 3.0) & (dist_site_arr <= 4.0), (dist_row_arr >= 3.0) & (dist_row_arr <= 4.0)),
            (">=5", dist_site_arr >= 5.0, dist_row_arr >= 5.0)
        ]
        res_rows = []
        for bname, s_mask, r_mask in bins:
            s_cnt = int(np.sum(s_mask))
            r_cnt = int(np.sum(r_mask))
            if r_cnt > 0:
                g1_sc = compute_scaled_mse(pred_g1[r_mask], y40_h_flat[r_mask])
                s1_sc = compute_scaled_mse(pred_s1[r_mask], y40_h_flat[r_mask])
                res_rows.append({
                    "Distance_Bin": bname,
                    "Site_Count": s_cnt,
                    "Row_Count": r_cnt,
                    "G1_Height_MSE": g1_sc["mse_h"],
                    "G1_AGB_MSE": g1_sc["mse_a"],
                    "G1_Score": g1_sc["score"],
                    "S1_Height_MSE": s1_sc["mse_h"],
                    "S1_AGB_MSE": s1_sc["mse_a"],
                    "S1_Score": s1_sc["score"]
                })
            else:
                res_rows.append({
                    "Distance_Bin": bname,
                    "Site_Count": 0,
                    "Row_Count": 0,
                    "G1_Height_MSE": None,
                    "G1_AGB_MSE": None,
                    "G1_Score": None,
                    "S1_Height_MSE": None,
                    "S1_AGB_MSE": None,
                    "S1_Score": None
                })
        return res_rows

    table7_rows = compute_binned_results(dist_to_g1, dist_g1_by_row)
    table7_df = pd.DataFrame(table7_rows)
    table7_csv = OUT_DIR / "table7_spatial_distance_g1.csv"
    table7_df.to_csv(table7_csv, index=False)
    print(f"Saved Table 7 (Distance to G1) to {table7_csv}", flush=True)

    table8_rows = compute_binned_results(dist_to_s1, dist_s1_by_row)
    table8_df = pd.DataFrame(table8_rows)
    table8_csv = OUT_DIR / "table8_spatial_distance_s1.csv"
    table8_df.to_csv(table8_csv, index=False)
    print(f"Saved Table 8 (Distance to S1) to {table8_csv}", flush=True)

    # -------------------------------------------------------------
    # Output NPZ, JSON, and SHA256 integrity checks
    # -------------------------------------------------------------
    oof_npz_path = OUT_DIR / "oof_predictions.npz"
    np.savez_compressed(
        oof_npz_path,
        cells=cells_3373,
        block=block_3373,
        fold=fold_assignment,
        oof_S1_seed20261101=oof_s1_seeds[0],
        oof_S1_seed20261102=oof_s1_seeds[1],
        oof_S1_seed20261103=oof_s1_seeds[2],
        oof_S1_seed20261104=oof_s1_seeds[3],
        oof_S1_seed20261105=oof_s1_seeds[4],
        oof_M1_seed20261101=oof_m1_seeds[0],
        oof_M1_seed20261102=oof_m1_seeds[1],
        oof_M1_seed20261103=oof_m1_seeds[2],
        oof_M1_seed20261104=oof_m1_seeds[3],
        oof_M1_seed20261105=oof_m1_seeds[4],
        oof_S1=oof_s1_avg,
        oof_M1=oof_m1_avg,
        y0=y0_flat,
        y40=y40_flat,
        C20_h=c20_h,
        C50_h=c50_h,
        C20_a=c20_a,
        C50_a=c50_a,
        D=d_tag
    )
    print(f"Saved OOF predictions NPZ to {oof_npz_path}", flush=True)

    # Re-verify S1 baseline SHA256 post-execution
    s1_sha_post = sha256_file(S1_CSV_PATH)
    assert s1_sha_post == EXPECTED_S1_SHA, "S1 CSV altered during execution!"
    print("Confirmed S1 reference file integrity unchanged post-execution.", flush=True)

    # Hashes
    code_hashes = {
        "run_step1_decomposition.py": sha256_file(CODE_DIR / "run_step1_decomposition.py"),
        "prepare_inputs.py": sha256_file(CODE_DIR / "prepare_inputs.py"),
        "features.py": sha256_file(CODE_DIR / "features.py"),
        "restore_y0.py": sha256_file(CODE_DIR / "restore_y0.py"),
        "train_s1.py": sha256_file(CODE_DIR / "train_s1.py")
    }

    t_total = time.time() - t_start
    _, current_peak_bytes = tracemalloc.get_traced_memory()
    total_peak_mb = current_peak_bytes / (1024 * 1024)

    decomposition_summary = {
        "task": "Step 1: Error Decomposition and Input Verification (2026-11-02)",
        "elapsed_seconds": t_total,
        "peak_memory_mb": total_peak_mb,
        "initial_question_answer": initial_question_answer,
        "input_verification": {
            "coordinate_validation": {
                "spearman_corr": spearman_corr,
                "abs_spearman": abs_spearman,
                "passed": True
            },
            "invariants_summary": invariants_summary,
            "unique_10deg_blocks": int(n_blocks),
            "folds_info": folds_info
        },
        "spatial_block_cv": {
            "persistence": pers_overall,
            "B": b_overall,
            "S1": s1_spatial_overall,
            "M1": m1_spatial_overall
        },
        "collapse_analysis": {
            "share_targetwise": float(share_comb_targetwise),
            "share_targetwise_ci_95": [ci_tw_low, ci_tw_high],
            "decision_targetwise": decision_tw,
            "share_rowwise": float(share_comb_rowwise),
            "share_rowwise_ci_95": [ci_rw_low, ci_rw_high],
            "decision_rowwise": decision_rw,
            "share_height": float(share_h),
            "share_agb": float(share_a)
        },
        "transient_analysis": {
            "window_diff_cis": window_diff_cis
        },
        "part4_spatial_proximity": {
            "holdout_sites": 3924,
            "replicated_g1_score": g1_rep_sc["score"],
            "replicated_s1_score": s1_rep_sc["score"],
            "binned_g1": table7_rows,
            "binned_s1": table8_rows
        },
        "acceptance_checks": {
            "s1_sha_intact": bool(s1_sha_post == EXPECTED_S1_SHA),
            "n_baseline_sites": len(cells_3373),
            "n_baseline_rows": len(y0_flat),
            "n_holdout_sites": len(holdout_cells),
            "fresh_sites_untouched": 1024
        },
        "code_hashes": code_hashes
    }

    decomp_json_path = OUT_DIR / "decomposition.json"
    with open(decomp_json_path, "w", encoding="utf-8") as f:
        json.dump(decomposition_summary, f, indent=2, ensure_ascii=False)
    print(f"Saved complete decomposition summary to {decomp_json_path}", flush=True)

    print("\nStep 1 decomposition pipeline completed successfully!", flush=True)
    return decomposition_summary


if __name__ == "__main__":
    run()

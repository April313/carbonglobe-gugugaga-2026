"""Training and inference pipeline for S1 (Delta + Site Profile Model).
Trains 3-seed ensemble of HistGradientBoostingRegressor on 3373 sites.
Outputs candidate_S1_delta_profile_3373.csv.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

# Local module imports
from features import SEED_AGES, SCALE, H_CAP, extract_all_features
from restore_y0 import restore_initial_y0

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_pipeline(config_path):
    cfg = load_config(config_path)
    base_dir = Path(config_path).resolve().parent
    
    # Resolve paths relative to config file if not absolute
    raw_dir = Path(cfg['raw_data_dir'])
    if not raw_dir.is_absolute():
        raw_dir = (base_dir / raw_dir).resolve()
        
    hist_dir_key = cfg.get('prepared_inputs_dir') or cfg.get('historical_data_dir') or cfg.get('inputs_dir', 'inputs')
    hist_dir = Path(hist_dir_key)
    if not hist_dir.is_absolute():
        hist_dir = (base_dir / hist_dir).resolve()
        
    sample_sub_path = Path(cfg['sample_submission_path'])
    if not sample_sub_path.is_absolute():
        sample_sub_path = (base_dir / sample_sub_path).resolve()
        
    inputs_dir = Path(cfg.get('inputs_dir', 'inputs'))
    if not inputs_dir.is_absolute():
        inputs_dir = (base_dir / inputs_dir).resolve()
        
    output_dir = Path(cfg.get('output_dir', 'outputs'))
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Training S1: Delta + Site Profile Model ===")
    t0 = time.time()
    
    # 1. Load splits and training data
    splits_path = inputs_dir / 'site_splits.npz'
    splits = np.load(splits_path)
    groups = splits['groups']
    cells = splits['cells']
    N_sites = len(cells)
    assert N_sites == 3373, f"Expected 3373 sites, got {N_sites}"
    
    december_Y = np.load(hist_dir / 'december_Y.npy') # (15, 3373, 41, 7)
    monthly_X = np.load(hist_dir / 'monthly_X.npy')   # (3373, 40, 12, 136)
    
    y0_train = december_Y[:, :, 0, :2].transpose(1, 0, 2).astype(np.float32)
    y40_train = december_Y[:, :, 40, :2].transpose(1, 0, 2).astype(np.float32)
    delta_train = y40_train - y0_train
    
    print("Extracting features on training data...")
    X_train_all = extract_all_features(y0_train, monthly_X)
    delta_flat = delta_train.reshape(N_sites * 15, 2)
    row_groups = np.repeat(groups, 15)
    
    # 2. 5-fold GroupKFold search for optimal iterations
    print("Running 5-fold GroupKFold iteration search (up to 1500 iters)...")
    gkf = GroupKFold(n_splits=5)
    best_iters_h = []
    best_iters_a = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_all, delta_flat[:, 0], groups=row_groups)):
        # Height
        reg_h = HistGradientBoostingRegressor(
            learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50,
            max_features=0.8, l2_regularization=0.0, early_stopping=False,
            max_iter=1500, random_state=20261101 + fold
        )
        reg_h.fit(X_train_all[train_idx], delta_flat[train_idx, 0])
        mses_h = [np.mean((p - delta_flat[val_idx, 0]) ** 2) for p in reg_h.staged_predict(X_train_all[val_idx])]
        best_iters_h.append(int(np.argmin(mses_h)) + 1)
        
        # AGB
        reg_a = HistGradientBoostingRegressor(
            learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50,
            max_features=0.8, l2_regularization=0.0, early_stopping=False,
            max_iter=1500, random_state=20261101 + fold
        )
        reg_a.fit(X_train_all[train_idx], delta_flat[train_idx, 1])
        mses_a = [np.mean((p - delta_flat[val_idx, 1]) ** 2) for p in reg_a.staged_predict(X_train_all[val_idx])]
        best_iters_a.append(int(np.argmin(mses_a)) + 1)
        
    median_iter_h = int(np.median(best_iters_h))
    median_iter_a = int(np.median(best_iters_a))
    print(f"Optimal iterations: Height={best_iters_h} -> Median={median_iter_h}, AGB={best_iters_a} -> Median={median_iter_a}")
    
    # 3. Full training with 3 seeds
    print("Full training on 3373 sites with 3 seeds...")
    seeds = [20261101, 20261102, 20261103]
    models_h = []
    models_a = []
    
    for seed in seeds:
        reg_h = HistGradientBoostingRegressor(
            learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50,
            max_features=0.8, l2_regularization=0.0, early_stopping=False,
            max_iter=median_iter_h, random_state=seed
        )
        reg_h.fit(X_train_all, delta_flat[:, 0])
        models_h.append(reg_h)
        
        reg_a = HistGradientBoostingRegressor(
            learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50,
            max_features=0.8, l2_regularization=0.0, early_stopping=False,
            max_iter=median_iter_a, random_state=seed
        )
        reg_a.fit(X_train_all, delta_flat[:, 1])
        models_a.append(reg_a)
        
    # 4. Predict on SSP126 & SSP585
    print("Predicting on SSP126 and SSP585...")
    with open(sample_sub_path, 'r', encoding='utf-8') as f:
        reader = list(csv.reader(f))
    sample_ids = [r[0] for r in reader[1:]]
    assert len(sample_ids) == 185130
    
    s1_parts = []
    for sc in ['ssp126', 'ssp585']:
        ssp_data = np.load(raw_dir / f'test_{sc}.npz')
        test_x = np.asarray(ssp_data['test_x'], dtype=np.float32)
        test_y0 = np.asarray(ssp_data['test_y0'], dtype=np.float64)
        
        y0_restored = restore_initial_y0(test_y0)
        X_ssp = extract_all_features(y0_restored, test_x)
        
        preds_dh = np.mean([m.predict(X_ssp) for m in models_h], axis=0)
        preds_da = np.mean([m.predict(X_ssp) for m in models_a], axis=0)
        
        y0_flat_ssp = y0_restored.reshape(-1, 2)
        raw_pred_h = y0_flat_ssp[:, 0] + preds_dh
        raw_pred_a = y0_flat_ssp[:, 1] + preds_da
        
        h_clipped = np.clip(raw_pred_h, 0.0, 35.0)
        a_clipped = np.maximum(raw_pred_a, 0.0)
        
        h_scaled = np.minimum(h_clipped / SCALE[0], H_CAP)
        a_scaled = a_clipped / SCALE[1]
        s1_parts.append(np.column_stack([h_scaled, a_scaled]))
        
    s1_scaled = np.concatenate(s1_parts, axis=0)
    assert len(s1_scaled) == 185130
    
    # 5. Write submission CSV
    out_csv_path = output_dir / 'candidate_S1_delta_profile_3373.csv'
    with open(out_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'height', 'agb'])
        for row_id, (h_val, a_val) in zip(sample_ids, s1_scaled):
            writer.writerow([row_id, f"{h_val:.6f}", f"{a_val:.6f}"])
            
    print(f"S1 finished in {time.time()-t0:.1f}s. Saved to: {out_csv_path}")
    return out_csv_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train S1 Delta + Site Profile model.")
    parser.add_argument('--config', default='config.json', help="Path to config.json")
    args = parser.parse_args()
    run_pipeline(args.config)

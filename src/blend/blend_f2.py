"""Closed-form public score blending for F2 (non-signed 13 submissions + S1 base).
Solves G c = -g using eigendecomposition with eigenvalue cutoff tol = 1e-3.
Outputs candidate_F2_score_blend_nosigned.csv.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from features import H_CAP

def load_csv_values(csv_path):
    with open(csv_path, encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
    return np.array([[float(r['height']), float(r['agb'])] for r in reader], dtype=np.float64)

def run_blending(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    base_dir = Path(config_path).resolve().parent
    
    inputs_dir = Path(cfg.get('inputs_dir', 'inputs'))
    if not inputs_dir.is_absolute():
        inputs_dir = (base_dir / inputs_dir).resolve()
        
    output_dir = Path(cfg.get('output_dir', 'outputs'))
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sample_sub_path = Path(cfg['sample_submission_path'])
    if not sample_sub_path.is_absolute():
        sample_sub_path = (base_dir / sample_sub_path).resolve()
        
    print("=== Solving F2 Closed-Form Public Score Blend ===")
    t0 = time.time()
    
    table_path = inputs_dir / 'f2_submission_table.csv'
    with open(table_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        
    # S1 is the base (ref 57860559)
    s1_row = [r for r in rows if r['ref'] == '57860559'][0]
    other_rows = [r for r in rows if r['ref'] != '57860559']
    
    s1_csv_path = inputs_dir / 'f2_csvs' / s1_row['filename']
    B = load_csv_values(s1_csv_path) # (185130, 2)
    S_B = float(s1_row['score'])
    
    M = len(other_rows)
    assert M == 13, f"Expected 13 non-signed submissions, got {M}"
    
    D_list = []
    scores = []
    for r in other_rows:
        csv_p = inputs_dir / 'f2_csvs' / r['filename']
        val = load_csv_values(csv_p)
        D_list.append((val - B).ravel())
        scores.append(float(r['score']))
        
    D_mat = np.array(D_list, dtype=np.float64) # (13, 370260)
    scores = np.array(scores, dtype=np.float64)
    
    # Gram matrix: mean over all 370260 entries
    G = (D_mat @ D_mat.T) / D_mat.shape[1]
    G_diag = np.diag(G)
    g = (scores - S_B - G_diag) / 2.0
    
    # Eigendecomposition solver
    w, V = np.linalg.eigh(G)
    tol = 1e-3
    keep = w > tol * w.max()
    c = -(V[:, keep] / w[keep]) @ (V[:, keep].T @ g)
    
    pred_score = float(S_B + 2.0 * g @ c + c @ G @ c)
    sum_abs_c = float(np.sum(np.abs(c)))
    kept_count = int(keep.sum())
    
    print(f"F2 solved: Kept components = {kept_count}/{M}, sum|c_i| = {sum_abs_c:.4f}, predicted score = {pred_score:.6f}")
    
    # Blend and clip
    p = B + (c @ D_mat).reshape(-1, 2)
    p[:, 0] = np.clip(p[:, 0], 0.0, H_CAP)
    p[:, 1] = np.maximum(p[:, 1], 0.0)
    
    # Load sample IDs
    with open(sample_sub_path, encoding='utf-8') as f:
        sample_reader = list(csv.reader(f))
    sample_ids = [r[0] for r in sample_reader[1:]]
    assert len(sample_ids) == len(p)
    
    out_csv_path = output_dir / 'candidate_F2_score_blend_nosigned.csv'
    with open(out_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'height', 'agb'])
        for row_id, (h_val, a_val) in zip(sample_ids, p):
            writer.writerow([row_id, f"{h_val:.6f}", f"{a_val:.6f}"])
            
    print(f"F2 finished in {time.time()-t0:.1f}s. Saved to: {out_csv_path}")
    return out_csv_path, pred_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Blend F2 submissions using public scores.")
    parser.add_argument('--config', default='config.json', help="Path to config.json")
    args = parser.parse_args()
    run_blending(args.config)

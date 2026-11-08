"""Full Execution Pipeline for F3 Closed-Form Public Score Blending (2026-11-02)
Execution by Antigravity.
Strict boundaries:
- No training, no uploading, no Kaggle API calls.
- No reading credentials, no reading hidden labels.
- No modifying existing files.
- S1 retained as base B.
- Solves G c = -g with tol = 1e-3, height clipping [0, 3.209109], AGB >= 0.
"""

import sys
import os
import json
import math
import time
import hashlib
import csv
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path('data/project')
sys.path[:0] = [
    str(ROOT / '.runtime/science-cp314'),
    str(ROOT / 'code')
]

import numpy as np
import pandas as pd
from src import kaggle_task as kt

SEED_AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
SCALE = np.array([10.90645, 3.39567], dtype=np.float64)
H_CAP = math.floor(35 / 10.90645 * 1e6) / 1e6 # 3.209109

OUT_DIR = ROOT / 'results/score_blend_f3_20261102'
RAW = Path('data/kaggle')
REPORT_PATH = Path('data/reports/F3融合结果_20261102.md')
F2_CSV_PATH = ROOT / 'results/score_blend_20261101/candidate_F2_score_blend_nosigned.csv'

EXPECTED_S1_SHA = '23216519297719e207970ecd37151181b57cb53d1e23a11499ca33e7b0824aba'
EXPECTED_M1_SHA = '5e1d9806f69dec75ec1302f5f7e0aabf50736c82efa4de019f45baf208a8e999'
EXPECTED_M2_SHA = 'f5512033e2106f1c113eb54260038f1b4b368d6d6745fd604073f4847af123ff'
EXPECTED_M3_SHA = 'cb81a4fef0da2e40a156eab07a7754eb66c88f1a18629944d9615856ace9a782'
EXPECTED_B2_SHA = 'd23e0df2915e816d29355600bf49c1bdeb130ea14de7721dc8db4bbbd54c1e9a'

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def load_csv_values(csv_path):
    with open(csv_path, encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
    return np.array([[float(r['height']), float(r['agb'])] for r in reader], dtype=np.float64)

def run():
    print("=== Antigravity F3 Closed-Form Public Score Blending Task ===")
    t_start = time.time()
    
    # Prerequisite: output directory must not exist
    if OUT_DIR.exists():
        raise RuntimeError(f"Output directory {OUT_DIR} already exists! Halting as per task boundary.")
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    
    sample_csv = RAW / 'sample_submission.csv'
    sample_ids = kt.submission_ids(sample_csv)
    assert len(sample_ids) == 185130, f"Expected 185,130 sample IDs, got {len(sample_ids)}"
    
    # 1. Gather all inputs
    # F2 submission table
    f2_table_path = ROOT / 'results/score_blend_20261101/submission_table.csv'
    f2_table = pd.read_csv(f2_table_path)
    
    # S1 base
    s1_row = f2_table[f2_table['ref'] == 57860559].iloc[0]
    s1_path = Path(s1_row['path'])
    s1_sha = sha256_file(s1_path)
    assert s1_sha == EXPECTED_S1_SHA, f"S1 SHA256 mismatch: {s1_sha} != {EXPECTED_S1_SHA}"
    print(f"Verified S1 base file: ref 57860559, score 0.091, sha={s1_sha}")
    
    # 13 non-signed F2 directions
    f2_13_df = f2_table[(f2_table['signed'] == False) & (f2_table['ref'] != 57860559) & (f2_table['reason'] == '纳入')]
    assert len(f2_13_df) == 13, f"Expected 13 non-signed F2 directions, got {len(f2_13_df)}"
    
    all_directions = []
    for _, r in f2_13_df.iterrows():
        p = Path(r['path'])
        sha = sha256_file(p)
        assert sha == r['sha256'], f"SHA mismatch for F2 direction {r['ref']}"
        all_directions.append({
            'source': 'F2原有',
            'name': f"ref_{r['ref']}",
            'ref': str(r['ref']),
            'score': float(r['score']),
            'path': str(p),
            'sha256': sha
        })
        
    # M1, M2, M3, B2
    new_specs = [
        ('M1', '57890637', 0.103, ROOT / 'results/new_directions_20261101/candidate_M1_noclimate.csv', EXPECTED_M1_SHA),
        ('M2', '57890610', 0.101, ROOT / 'results/new_directions_20261101/candidate_M2_logratio.csv', EXPECTED_M2_SHA),
        ('M3', '57874589', 0.125, ROOT / 'results/new_directions_20261101/candidate_M3_mlp.csv', EXPECTED_M3_SHA),
        ('B2', '57863058', 0.136, ROOT / 'results/y0_restore_v2_20261101/candidate_B2_y0restore_nosigned_cap35.csv', EXPECTED_B2_SHA)
    ]
    for src_name, ref, sc, p, expected_sha in new_specs:
        sha = sha256_file(p)
        assert sha == expected_sha, f"SHA mismatch for {src_name} ({ref}): {sha} != {expected_sha}"
        all_directions.append({
            'source': src_name,
            'name': src_name,
            'ref': ref,
            'score': sc,
            'path': str(p),
            'sha256': sha
        })
        
    assert len(all_directions) == 17, f"Expected exactly 17 directions, got {len(all_directions)}"
    print(f"Loaded and verified all 17 directions + S1 base.")
    
    # Save f3_submission_table.csv
    f3_table_rows = [{
        'source': 'Base',
        'ref': '57860559',
        'score': 0.091,
        'path': str(s1_path),
        'sha256': s1_sha,
        'reason': '基准B (S1)'
    }]
    for d in all_directions:
        f3_table_rows.append({
            'source': d['source'],
            'ref': d['ref'],
            'score': d['score'],
            'path': d['path'],
            'sha256': d['sha256'],
            'reason': '纳入F3求解方向'
        })
    f3_table_path = OUT_DIR / 'f3_submission_table.csv'
    with open(f3_table_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'ref', 'score', 'path', 'sha256', 'reason'])
        writer.writeheader()
        writer.writerows(f3_table_rows)
    print(f"Saved submission table: {f3_table_path}")
    
    # 2. Preload data and construct Gram matrix
    B = load_csv_values(s1_path) # (185130, 2)
    S_B = 0.091
    
    D_list = []
    scores = []
    for d in all_directions:
        P_i = load_csv_values(d['path'])
        D_list.append((P_i - B).ravel())
        scores.append(d['score'])
        
    D_mat = np.array(D_list, dtype=np.float64) # (17, 370260)
    scores = np.array(scores, dtype=np.float64)
    M = len(all_directions)
    
    # Gram matrix: mean over all 370260 elements
    G = (D_mat @ D_mat.T) / D_mat.shape[1]
    G_diag = np.diag(G)
    g = (scores - S_B - G_diag) / 2.0
    
    # 3. Eigendecomposition solver
    w, V = np.linalg.eigh(G)
    tol_base = 1e-3
    w_max = float(w.max())
    w_min = float(w.min())
    keep = w > tol_base * w_max
    c = -(V[:, keep] / w[keep]) @ (V[:, keep].T @ g)
    
    pred_score = float(S_B + 2.0 * g @ c + c @ G @ c)
    sum_abs_c = float(np.sum(np.abs(c)))
    kept_count = int(keep.sum())
    
    print(f"\n--- F3 Solve Complete ---")
    print(f"Kept components: {kept_count}/{M}")
    print(f"sum|c_i|: {sum_abs_c:.6f}")
    print(f"Predicted score: {pred_score:.6f}")
    
    # 4. Prediction and clipping
    p = B + (c @ D_mat).reshape(-1, 2)
    p_final = p.copy()
    p_final[:, 0] = np.clip(p_final[:, 0], 0.0, H_CAP)
    p_final[:, 1] = np.maximum(p_final[:, 1], 0.0)
    
    clip_diff = p_final - p
    clip_energy = float(np.mean(clip_diff ** 2))
    h_clipped_cnt = int(((p[:, 0] < 0) | (p[:, 0] > H_CAP)).sum())
    a_clipped_cnt = int((p[:, 1] < 0).sum())
    total_clipped_rows = int(((p[:, 0] < 0) | (p[:, 0] > H_CAP) | (p[:, 1] < 0)).sum())
    print(f"Clipping energy mean((p_final - p)^2): {clip_energy:.6e}")
    print(f"Clipped rows: {total_clipped_rows}/{len(p)} ({total_clipped_rows/len(p)*100:.3f}%) [H: {h_clipped_cnt}, A: {a_clipped_cnt}]")
    
    # 5. Stability check 1: Leave-One-Out (LOO) Prediction
    print("\n--- Stability Check 1: LOO Prediction ---")
    loo_errors = []
    loo_predictions = []
    for i in range(M):
        idx_sub = [j for j in range(M) if j != i]
        G_sub = G[np.ix_(idx_sub, idx_sub)]
        G_col = G[idx_sub, i]
        g_sub = g[idx_sub]
        
        w_s, V_s = np.linalg.eigh(G_sub)
        k_s = w_s > tol_base * w_s.max()
        beta = (V_s[:, k_s] / w_s[k_s]) @ (V_s[:, k_s].T @ G_col)
        g_i_pred = float(beta @ g_sub)
        S_i_pred = float(S_B + G_diag[i] + 2.0 * g_i_pred)
        loo_predictions.append(S_i_pred)
        loo_errors.append(S_i_pred - scores[i])
    loo_rmse = float(np.sqrt(np.mean(np.array(loo_errors) ** 2)))
    print(f"LOO RMSE: {loo_rmse:.6f} across {M} directions")
    
    # 6. Stability check 2: Rounding Sensitivity (200 trials, seed 20261102, +/- 0.0005)
    print("\n--- Stability Check 2: Rounding Sensitivity ---")
    rng = np.random.default_rng(20261102)
    noisy_scores = []
    max_p_diffs = []
    for _ in range(200):
        noise_scores = rng.uniform(-0.0005, 0.0005, M)
        noise_B = rng.uniform(-0.0005, 0.0005)
        s_noisy = scores + noise_scores
        S_B_noisy = S_B + noise_B
        g_noisy = (s_noisy - S_B_noisy - G_diag) / 2.0
        c_noisy = -(V[:, keep] / w[keep]) @ (V[:, keep].T @ g_noisy)
        ps_noisy = float(S_B_noisy + 2.0 * g_noisy @ c_noisy + c_noisy @ G @ c_noisy)
        noisy_scores.append(ps_noisy)
        p_noisy = B + (c_noisy @ D_mat).reshape(-1, 2)
        max_p_diffs.append(float(np.max(np.abs(p_noisy - p))))
        
    p95_noisy_score = float(np.percentile(noisy_scores, 95))
    p05_noisy_score = float(np.percentile(noisy_scores, 5))
    noisy_summary = {
        'trials': 200,
        'seed': 20261102,
        'noise_range': [-0.0005, 0.0005],
        'min': float(np.min(noisy_scores)),
        'p05': p05_noisy_score,
        'median': float(np.median(noisy_scores)),
        'p95': p95_noisy_score,
        'max': float(np.max(noisy_scores)),
        'max_pred_score_diff': float(np.max(np.abs(np.array(noisy_scores) - pred_score))),
        'max_p_diff': float(np.max(max_p_diffs))
    }
    print(f"Rounding sensitivity: p05={p05_noisy_score:.6f}, median={noisy_summary['median']:.6f}, p95={p95_noisy_score:.6f}, max_p_diff={noisy_summary['max_p_diff']:.6f}")
    
    # 7. Stability check 3: tol sensitivity (1e-2 and 1e-4)
    print("\n--- Stability Check 3: tol Sensitivity ---")
    tol_results = {}
    for tol_val in [1e-2, 1e-4]:
        k_tol = w > tol_val * w_max
        c_tol = -(V[:, k_tol] / w[k_tol]) @ (V[:, k_tol].T @ g)
        ps_tol = float(S_B + 2.0 * g @ c_tol + c_tol @ G @ c_tol)
        tol_results[str(tol_val)] = {
            'kept': int(k_tol.sum()),
            'sum_abs_c': float(np.sum(np.abs(c_tol))),
            'pred_score': ps_tol
        }
        print(f"  tol={tol_val}: kept={k_tol.sum()}/{M}, sum|c|={np.sum(np.abs(c_tol)):.4f}, pred_score={ps_tol:.6f}")
        
    # 8. Check Stop Conditions
    print("\n--- Stop Condition Evaluation ---")
    stop_conditions = {
        'pred_score_ge_0082': bool(pred_score >= 0.082),
        'loo_rmse_gt_001': bool(loo_rmse > 0.01),
        'sum_abs_c_gt_3': bool(sum_abs_c > 3.0),
        'clip_energy_gt_1e4': bool(clip_energy > 1e-4),
        'noisy_p95_ge_0082': bool(p95_noisy_score >= 0.082)
    }
    any_triggered = any(stop_conditions.values())
    for cond_name, triggered in stop_conditions.items():
        print(f"  {cond_name}: {triggered}")
    print(f"Any stop condition triggered: {any_triggered}")
    
    # 9. Output CSV
    cand_csv_name = 'candidate_F3_score_blend_nosigned.csv'
    cand_csv_path = OUT_DIR / cand_csv_name
    with open(cand_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'height', 'agb'])
        for row_id, (h_val, a_val) in zip(sample_ids, p_final):
            writer.writerow([row_id, f"{h_val:.6f}", f"{a_val:.6f}"])
    f3_sha256 = sha256_file(cand_csv_path)
    print(f"Saved candidate CSV: {cand_csv_path} (SHA256: {f3_sha256})")
    
    # 10. Acceptance Checks
    print("\n--- Acceptance Check 1: kt.check_submission_csv ---")
    chk = kt.check_submission_csv(cand_csv_path, sample_csv)
    print("kt.check_submission_csv result:", chk)
    assert all(chk[k] for k in ['rows_match_185130', 'ids_match_sample_order', 'id_format_ok', 'no_nan_inf', 'scale_sanity']), f"Check failed for {cand_csv_name}"
    
    print("\n--- Acceptance Check 2: Reconstruct candidate with saved c (diff <= 1e-6) ---")
    reconstructed_p = B.copy()
    for idx, d in enumerate(all_directions):
        p_val = load_csv_values(d['path'])
        reconstructed_p += c[idx] * (p_val - B)
    reconstructed_p[:, 0] = np.clip(reconstructed_p[:, 0], 0.0, H_CAP)
    reconstructed_p[:, 1] = np.maximum(reconstructed_p[:, 1], 0.0)
    
    csv_loaded_p = load_csv_values(cand_csv_path)
    max_recon_diff = float(np.max(np.abs(reconstructed_p - csv_loaded_p)))
    print(f"Max reconstruction difference: {max_recon_diff:.2e}")
    assert max_recon_diff <= 1e-6, f"Reconstruction difference exceeds 1e-6: {max_recon_diff}"
    
    print("\n--- Acceptance Check 3: S1 file SHA256 integrity ---")
    s1_check_sha = sha256_file(s1_path)
    assert s1_check_sha == EXPECTED_S1_SHA, f"S1 SHA256 altered: {s1_check_sha} != {EXPECTED_S1_SHA}"
    print(f"S1 SHA256 matches: {EXPECTED_S1_SHA}")
    
    # 11. Prediction Changes Relative to S1 and F2
    print("\n--- Computing Diagnostics vs S1 and F2 ---")
    assert F2_CSV_PATH.exists(), f"F2 CSV path {F2_CSV_PATH} not found!"
    F2_vals = load_csv_values(F2_CSV_PATH)
    
    sc_len = 92565
    scenarios = ['ssp126', 'ssp585']
    
    # Scenario x age x target breakdown vs S1 and vs F2
    sc_age_summary = []
    for sc_idx, sc in enumerate(scenarios):
        sc_slice = slice(sc_idx * sc_len, (sc_idx + 1) * sc_len)
        p_sc = p_final[sc_slice]
        b_sc = B[sc_slice]
        f2_sc = F2_vals[sc_slice]
        
        for age_idx, age in enumerate(SEED_AGES):
            row_indices = np.arange(age_idx, sc_len, 15)
            p_age = p_sc[row_indices]
            b_age = b_sc[row_indices]
            f2_age = f2_sc[row_indices]
            
            diff_vs_s1 = (p_age - b_age).mean(axis=0)
            diff_vs_f2 = (p_age - f2_age).mean(axis=0)
            
            sc_age_summary.append({
                'scenario': sc,
                'age': int(age),
                'scaled_f3_mean_h': float(p_age[:, 0].mean()),
                'scaled_f3_mean_a': float(p_age[:, 1].mean()),
                'scaled_diff_vs_s1_h': float(diff_vs_s1[0]),
                'scaled_diff_vs_s1_a': float(diff_vs_s1[1]),
                'phys_diff_vs_s1_h': float(diff_vs_s1[0] * SCALE[0]),
                'phys_diff_vs_s1_a': float(diff_vs_s1[1] * SCALE[1]),
                'scaled_diff_vs_f2_h': float(diff_vs_f2[0]),
                'scaled_diff_vs_f2_a': float(diff_vs_f2[1]),
                'phys_diff_vs_f2_h': float(diff_vs_f2[0] * SCALE[0]),
                'phys_diff_vs_f2_a': float(diff_vs_f2[1] * SCALE[1])
            })
            
    # RMS differences vs S1 and vs F2
    rms_vs_s1_h = float(np.sqrt(np.mean((p_final[:, 0] - B[:, 0]) ** 2)))
    rms_vs_s1_a = float(np.sqrt(np.mean((p_final[:, 1] - B[:, 1]) ** 2)))
    rms_vs_s1_tot = float(0.5 * (rms_vs_s1_h + rms_vs_s1_a))
    
    rms_vs_f2_h = float(np.sqrt(np.mean((p_final[:, 0] - F2_vals[:, 0]) ** 2)))
    rms_vs_f2_a = float(np.sqrt(np.mean((p_final[:, 1] - F2_vals[:, 1]) ** 2)))
    rms_vs_f2_tot = float(0.5 * (rms_vs_f2_h + rms_vs_f2_a))
    
    # Mean of squares
    sq_h = float(np.mean(p_final[:, 0] ** 2))
    sq_a = float(np.mean(p_final[:, 1] ** 2))
    sq_mean = float(0.5 * (sq_h + sq_a))
    
    # Mean values
    mean_h = float(p_final[:, 0].mean())
    mean_a = float(p_final[:, 1].mean())
    
    # 12. Save full results metadata
    coefficients_list = []
    for idx, d in enumerate(all_directions):
        coefficients_list.append({
            'source': d['source'],
            'name': d['name'],
            'ref': d['ref'],
            'score': d['score'],
            'c_i': float(c[idx]),
            'abs_c_i': float(np.abs(c[idx])),
            'loo_pred_score': float(loo_predictions[idx]),
            'loo_error': float(loo_errors[idx]),
            'path': d['path'],
            'sha256': d['sha256']
        })
    with open(OUT_DIR / 'f3_coefficients.json', 'w', encoding='utf-8') as f:
        json.dump(coefficients_list, f, indent=2, ensure_ascii=False)
        
    metrics = {
        'task': 'F3 Closed-Form Public Score Blend',
        'elapsed_seconds': float(time.time() - t_start),
        'base_s1': {
            'ref': '57860559',
            'score': 0.091,
            'sha256': EXPECTED_S1_SHA,
            'path': str(s1_path)
        },
        'f3_solution': {
            'pred_score': pred_score,
            'sum_abs_c': sum_abs_c,
            'kept_components': kept_count,
            'total_components': M,
            'w_max': w_max,
            'w_min': w_min,
            'tol': tol_base,
            'clip_energy': clip_energy,
            'clipped_rows_total': total_clipped_rows,
            'clipped_rows_fraction': float(total_clipped_rows / len(p_final)),
            'clipped_h_rows': h_clipped_cnt,
            'clipped_a_rows': a_clipped_cnt
        },
        'stability_checks': {
            'loo_rmse': loo_rmse,
            'rounding_sensitivity': noisy_summary,
            'tol_sensitivity': tol_results,
            'stop_conditions': stop_conditions,
            'any_stop_triggered': any_triggered
        },
        'acceptance_checks': {
            'check_submission_csv': chk,
            'max_reconstruction_diff': max_recon_diff,
            's1_sha256_match': bool(s1_check_sha == EXPECTED_S1_SHA)
        },
        'candidate_f3': {
            'filename': cand_csv_name,
            'path': str(cand_csv_path),
            'sha256': f3_sha256,
            'means': {'height': mean_h, 'agb': mean_a},
            'mean_squares': {'height': sq_h, 'agb': sq_a, 'total': sq_mean},
            'rms_vs_s1': {'height': rms_vs_s1_h, 'agb': rms_vs_s1_a, 'total': rms_vs_s1_tot},
            'rms_vs_f2': {'height': rms_vs_f2_h, 'agb': rms_vs_f2_a, 'total': rms_vs_f2_tot}
        },
        'scenario_age_breakdown': sc_age_summary
    }
    with open(OUT_DIR / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        
    print(f"Metrics saved to {OUT_DIR / 'metrics.json'}")
    return metrics, coefficients_list

if __name__ == '__main__':
    run()

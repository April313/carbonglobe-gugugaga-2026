"""Stage 1: Generate Probe Files P1 and P2 for Segment Correction (2026-11-02)
Execution by Antigravity.
Strict boundaries:
- No training, no uploading, no Kaggle API calls.
- No reading credentials, no reading hidden labels.
- No modifying existing files.
- Generates candidate_P1_tall_down.csv and candidate_P2_young_up.csv.
- Verifies relative difference vs Claude reference <= 1%.
- Halts after Stage 1 to wait for public scores.
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
    str(ROOT / 'code'),
    str(Path(__file__).resolve().parent)
]

import numpy as np
import pandas as pd
from src import kaggle_task as kt
from restore_y0 import restore_initial_y0

SCALE = np.array([10.90645, 3.39567], dtype=np.float64)
H_CAP = math.floor(35 / 10.90645 * 1e6) / 1e6 # 3.209109
SEED_AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]

OUT_DIR = ROOT / 'results/segment_probe_20261102'
REPORT_PATH = Path('data/reports/分段修正探测结果_20261102.md')

F3_CSV_PATH = ROOT / 'results/score_blend_f3_20261102/candidate_F3_score_blend_nosigned.csv'
EXPECTED_F3_SHA = '3bd2f0d6fb84654739f4db8654fd4091002e98f1db8f937a326c9fb36bf41d2a'

SSP126_NPZ_PATH = Path('data/kaggle/test_ssp126.npz')
SSP585_NPZ_PATH = Path('data/kaggle/test_ssp585.npz')
SAMPLE_CSV_PATH = Path('data/kaggle/sample_submission.csv')

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def run():
    print("=== Antigravity Segment Probe Generation (Stage 1) ===")
    t0 = time.time()
    
    # 1. Output directory prerequisite check
    if OUT_DIR.exists():
        raise RuntimeError(f"Output directory {OUT_DIR} already exists! Halting as per task boundary.")
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    
    # 2. Check F3 input integrity
    assert F3_CSV_PATH.exists(), f"F3 file not found: {F3_CSV_PATH}"
    f3_sha_init = sha256_file(F3_CSV_PATH)
    assert f3_sha_init == EXPECTED_F3_SHA, f"F3 SHA256 mismatch: {f3_sha_init} != {EXPECTED_F3_SHA}"
    print(f"Verified F3 base SHA256: {f3_sha_init}")
    
    # 3. Load F3 predictions
    f3_df = pd.read_csv(F3_CSV_PATH)
    sample_df = pd.read_csv(SAMPLE_CSV_PATH)
    assert len(f3_df) == 185130, f"Expected 185130 rows in F3, got {len(f3_df)}"
    assert (f3_df['id'] == sample_df['id']).all(), "F3 IDs do not match sample_submission IDs!"
    sample_ids = f3_df['id'].tolist()
    F = f3_df[['height', 'agb']].values.astype(np.float64) # (185130, 2)
    
    # 4. Load initial state test_y0 and restore physical Y
    assert SSP126_NPZ_PATH.exists(), f"SSP126 npz not found: {SSP126_NPZ_PATH}"
    assert SSP585_NPZ_PATH.exists(), f"SSP585 npz not found: {SSP585_NPZ_PATH}"
    npz126 = np.load(SSP126_NPZ_PATH)
    npz585 = np.load(SSP585_NPZ_PATH)
    
    raw_y0_126 = npz126['test_y0'] # (6171, 15, 7)
    raw_y0_585 = npz585['test_y0'] # (6171, 15, 7)
    assert raw_y0_126.shape == (6171, 15, 7), f"Unexpected shape for raw_y0_126: {raw_y0_126.shape}"
    assert raw_y0_585.shape == (6171, 15, 7), f"Unexpected shape for raw_y0_585: {raw_y0_585.shape}"
    
    Y_126 = restore_initial_y0(raw_y0_126, clip_height=True, clip_agb=True).reshape(-1, 2) # (92565, 2)
    Y_585 = restore_initial_y0(raw_y0_585, clip_height=True, clip_agb=True).reshape(-1, 2) # (92565, 2)
    Y = np.vstack([Y_126, Y_585]) # (185130, 2) physical units: height [0, 35], agb >= 0
    Y_s = Y / SCALE # (185130, 2) scaled units
    
    # Verify line-by-line scenario, site, age alignment with F3 IDs
    print("Verifying line-by-line alignment of scenario, site, age...")
    row_ages = []
    for idx, row_id in enumerate(sample_ids):
        parts = row_id.split('_')
        sc = parts[0]
        site = int(parts[1])
        age = int(parts[2])
        
        expected_sc = 'ssp126' if idx < 92565 else 'ssp585'
        local_idx = idx % 92565
        expected_site = local_idx // 15
        expected_age = SEED_AGES[local_idx % 15]
        
        assert sc == expected_sc and site == expected_site and age == expected_age, \
            f"ID mismatch at row {idx}: {row_id} vs {expected_sc}_{expected_site}_{expected_age}"
        row_ages.append(age)
    row_ages = np.array(row_ages, dtype=np.int64)
    print("Alignment verified: all 185,130 rows match scenario, site, age exactly.")
    
    # 5. Direction P1: tall forest downscaled, moves height only
    # tall = 1 / (1 + exp(-(Y_h - 20) / 2)), Y_h in meters
    Y_h = Y[:, 0]
    tall = 1.0 / (1.0 + np.exp(-(Y_h - 20.0) / 2.0))
    d1_h = -tall * F[:, 0]
    d1_a = np.zeros_like(d1_h)
    d1 = np.column_stack([d1_h, d1_a])
    
    P1_unclipped = F + 0.06 * d1
    P1 = P1_unclipped.copy()
    P1[:, 0] = np.clip(P1[:, 0], 0.0, H_CAP)
    P1[:, 1] = np.maximum(P1[:, 1], 0.0)
    
    # 6. Direction P2: young forest (age <= 30) growth amplified, moves both columns
    # young = 1 if age <= 30 else 0
    young = (row_ages <= 30).astype(np.float64)
    d2 = young[:, None] * (F - Y_s)
    
    P2_unclipped = F + 0.14 * d2
    P2 = P2_unclipped.copy()
    P2[:, 0] = np.clip(P2[:, 0], 0.0, H_CAP)
    P2[:, 1] = np.maximum(P2[:, 1], 0.0)
    
    # 7. Metrics calculation
    # P1 metrics
    p1_diff = P1 - F
    p1_mean_sq_diff = float(np.mean(p1_diff ** 2))
    p1_clip_diff = P1 - P1_unclipped
    p1_clip_energy = float(np.mean(p1_clip_diff ** 2))
    p1_h_mean_change = float(np.mean(p1_diff[:, 0]) * SCALE[0])
    p1_a_mean_change = float(np.mean(p1_diff[:, 1]) * SCALE[1])
    
    # P2 metrics
    p2_diff = P2 - F
    p2_mean_sq_diff = float(np.mean(p2_diff ** 2))
    p2_clip_diff = P2 - P2_unclipped
    p2_clip_energy = float(np.mean(p2_clip_diff ** 2))
    p2_h_mean_change = float(np.mean(p2_diff[:, 0]) * SCALE[0])
    p2_a_mean_change = float(np.mean(p2_diff[:, 1]) * SCALE[1])
    
    # Reference values from task
    ref_vals = {
        'P1': {'mean_sq': 0.00196, 'clip_energy': 0.0, 'h_change': -0.347, 'a_change': 0.0},
        'P2': {'mean_sq': 0.00204, 'clip_energy': 1.46e-6, 'h_change': 0.175, 'a_change': 0.0362}
    }
    
    diff_p1_sq = abs(p1_mean_sq_diff - ref_vals['P1']['mean_sq']) / ref_vals['P1']['mean_sq']
    diff_p1_h = abs(p1_h_mean_change - ref_vals['P1']['h_change']) / abs(ref_vals['P1']['h_change'])
    diff_p2_sq = abs(p2_mean_sq_diff - ref_vals['P2']['mean_sq']) / ref_vals['P2']['mean_sq']
    diff_p2_clip = abs(p2_clip_energy - ref_vals['P2']['clip_energy']) / ref_vals['P2']['clip_energy']
    diff_p2_h = abs(p2_h_mean_change - ref_vals['P2']['h_change']) / abs(ref_vals['P2']['h_change'])
    diff_p2_a = abs(p2_a_mean_change - ref_vals['P2']['a_change']) / abs(ref_vals['P2']['a_change'])
    
    print("\n--- P1 Calculated Metrics vs Reference ---")
    print(f"mean((P1-F)^2):   {p1_mean_sq_diff:.6f} (Ref: ~0.00196, rel diff: {diff_p1_sq*100:.3f}%)")
    print(f"Clipping energy:  {p1_clip_energy:.6e} (Ref: 0)")
    print(f"Height mean diff: {p1_h_mean_change:+.6f} m (Ref: ~ -0.347 m, rel diff: {diff_p1_h*100:.3f}%)")
    print(f"AGB mean diff:    {p1_a_mean_change:+.6f} Mg/ha (Ref: 0)")
    
    print("\n--- P2 Calculated Metrics vs Reference ---")
    print(f"mean((P2-F)^2):   {p2_mean_sq_diff:.6f} (Ref: ~0.00204, rel diff: {diff_p2_sq*100:.3f}%)")
    print(f"Clipping energy:  {p2_clip_energy:.6e} (Ref: ~1.46e-6, rel diff: {diff_p2_clip*100:.3f}%)")
    print(f"Height mean diff: {p2_h_mean_change:+.6f} m (Ref: ~ +0.175 m, rel diff: {diff_p2_h*100:.3f}%)")
    print(f"AGB mean diff:    {p2_a_mean_change:+.6f} Mg/ha (Ref: ~ +0.0362 Mg/ha, rel diff: {diff_p2_a*100:.3f}%)")
    
    # Check 1% relative threshold
    assert diff_p1_sq <= 0.01, f"P1 mean_sq relative diff exceeds 1%: {diff_p1_sq}"
    assert diff_p1_h <= 0.01, f"P1 height change relative diff exceeds 1%: {diff_p1_h}"
    assert p1_clip_energy == 0.0, f"P1 clipping energy non-zero: {p1_clip_energy}"
    assert p1_a_mean_change == 0.0, f"P1 AGB change non-zero: {p1_a_mean_change}"
    
    assert diff_p2_sq <= 0.01, f"P2 mean_sq relative diff exceeds 1%: {diff_p2_sq}"
    assert diff_p2_clip <= 0.01, f"P2 clipping energy relative diff exceeds 1%: {diff_p2_clip}"
    assert diff_p2_h <= 0.01, f"P2 height change relative diff exceeds 1%: {diff_p2_h}"
    assert diff_p2_a <= 0.01, f"P2 AGB change relative diff exceeds 1%: {diff_p2_a}"
    print("\nAll relative differences vs Claude reference are well within 1% threshold.")
    
    # 8. Save CSV files
    p1_csv_name = 'candidate_P1_tall_down.csv'
    p2_csv_name = 'candidate_P2_young_up.csv'
    p1_csv_path = OUT_DIR / p1_csv_name
    p2_csv_path = OUT_DIR / p2_csv_name
    
    def write_csv(path, vals):
        with open(path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'height', 'agb'])
            for row_id, (h_val, a_val) in zip(sample_ids, vals):
                writer.writerow([row_id, f"{h_val:.6f}", f"{a_val:.6f}"])
                
    write_csv(p1_csv_path, P1)
    write_csv(p2_csv_path, P2)
    
    p1_sha = sha256_file(p1_csv_path)
    p2_sha = sha256_file(p2_csv_path)
    print(f"Saved {p1_csv_name}: SHA256={p1_sha}")
    print(f"Saved {p2_csv_name}: SHA256={p2_sha}")
    
    # 9. Acceptance checks
    print("\n--- Acceptance Check 1: kt.check_submission_csv ---")
    chk_p1 = kt.check_submission_csv(p1_csv_path, SAMPLE_CSV_PATH)
    chk_p2 = kt.check_submission_csv(p2_csv_path, SAMPLE_CSV_PATH)
    print(f"P1 check: {chk_p1}")
    print(f"P2 check: {chk_p2}")
    assert all(chk_p1[k] for k in ['rows_match_185130', 'ids_match_sample_order', 'id_format_ok', 'no_nan_inf', 'scale_sanity']), "P1 check failed"
    assert all(chk_p2[k] for k in ['rows_match_185130', 'ids_match_sample_order', 'id_format_ok', 'no_nan_inf', 'scale_sanity']), "P2 check failed"
    
    print("\n--- Acceptance Check 2: ID and sequence alignment with F3 ---")
    p1_df = pd.read_csv(p1_csv_path)
    p2_df = pd.read_csv(p2_csv_path)
    assert (p1_df['id'] == f3_df['id']).all(), "P1 IDs do not match F3 exactly!"
    assert (p2_df['id'] == f3_df['id']).all(), "P2 IDs do not match F3 exactly!"
    print("P1 and P2 IDs and sequence strictly match F3.")
    
    print("\n--- Acceptance Check 3: F3 file integrity ---")
    f3_sha_post = sha256_file(F3_CSV_PATH)
    assert f3_sha_post == EXPECTED_F3_SHA, f"F3 file altered! {f3_sha_post} != {EXPECTED_F3_SHA}"
    print(f"F3 file SHA256 unchanged: {EXPECTED_F3_SHA}")
    
    # 10. Record code hashes and metrics
    code_dir = Path(__file__).resolve().parent
    code_hashes = {
        'generate_probes.py': sha256_file(code_dir / 'generate_probes.py'),
        'restore_y0.py': sha256_file(code_dir / 'restore_y0.py'),
        'features.py': sha256_file(code_dir / 'features.py')
    }
    
    metrics = {
        'task': 'Segment Probe Generation (Stage 1)',
        'elapsed_seconds': float(time.time() - t0),
        'base_f3': {
            'path': str(F3_CSV_PATH),
            'sha256': EXPECTED_F3_SHA
        },
        'P1': {
            'filename': p1_csv_name,
            'path': str(p1_csv_path),
            'sha256': p1_sha,
            'mean_sq_diff_vs_f3': p1_mean_sq_diff,
            'clipping_energy': p1_clip_energy,
            'height_mean_change_m': p1_h_mean_change,
            'agb_mean_change_mgha': p1_a_mean_change,
            'relative_diffs_vs_ref': {
                'mean_sq_diff': diff_p1_sq,
                'height_change': diff_p1_h
            },
            'acceptance': chk_p1
        },
        'P2': {
            'filename': p2_csv_name,
            'path': str(p2_csv_path),
            'sha256': p2_sha,
            'mean_sq_diff_vs_f3': p2_mean_sq_diff,
            'clipping_energy': p2_clip_energy,
            'height_mean_change_m': p2_h_mean_change,
            'agb_mean_change_mgha': p2_a_mean_change,
            'relative_diffs_vs_ref': {
                'mean_sq_diff': diff_p2_sq,
                'clipping_energy': diff_p2_clip,
                'height_change': diff_p2_h,
                'agb_change': diff_p2_a
            },
            'acceptance': chk_p2
        },
        'code_hashes': code_hashes
    }
    
    with open(OUT_DIR / 'probe_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        
    print(f"Metrics saved to {OUT_DIR / 'probe_metrics.json'}")
    return metrics

if __name__ == '__main__':
    run()

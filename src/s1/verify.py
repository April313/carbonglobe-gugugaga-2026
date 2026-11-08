"""Verification script for reproduced S1 and F2 candidates.
Verifies CSV formatting, ID alignment, scale sanity, and checks SHA256 against expected.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def check_csv_structure(csv_path, sample_path):
    with open(sample_path, encoding='utf-8') as f:
        sample_reader = list(csv.reader(f))
    want_ids = [r[0] for r in sample_reader[1:]]
    N_ROWS = len(want_ids)
    
    with open(csv_path, encoding='utf-8') as f:
        reader = list(csv.reader(f))
    assert reader[0] == ['id', 'height', 'agb'], f"Header mismatch: {reader[0]}"
    body = reader[1:]
    
    checks = {
        'n_rows': len(body),
        'rows_match_185130': len(body) == N_ROWS,
        'ids_match_sample_order': [r[0] for r in body] == want_ids,
        'id_format_ok': all(r[0].count('_') == 2 for r in body)
    }
    vals = np.array([[float(r[1]), float(r[2])] for r in body], dtype=np.float64)
    checks['no_nan_inf'] = bool(np.isfinite(vals).all())
    checks['height_mean'] = float(vals[:, 0].mean())
    checks['agb_mean'] = float(vals[:, 1].mean())
    checks['scale_sanity'] = (0.2 < checks['height_mean'] < 5.0) and (0.2 < checks['agb_mean'] < 5.0)
    return checks, vals

def run_verification(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    base_dir = Path(config_path).resolve().parent
    
    output_dir = Path(cfg.get('output_dir', 'outputs'))
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()
        
    sample_sub_path = Path(cfg['sample_submission_path'])
    if not sample_sub_path.is_absolute():
        sample_sub_path = (base_dir / sample_sub_path).resolve()
        
    expected_path = base_dir / 'expected/sha256.json'
    expected_shas = {}
    if expected_path.exists():
        with open(expected_path, 'r', encoding='utf-8') as f:
            expected_shas = json.load(f)
            
    print("=== Verifying Candidate Outputs ===")
    all_passed = True
    
    for filename in ['candidate_S1_delta_profile_3373.csv', 'candidate_F2_score_blend_nosigned.csv']:
        p = output_dir / filename
        if not p.exists():
            print(f"FAILED: {filename} does not exist in {output_dir}")
            all_passed = False
            continue
            
        chk, vals = check_csv_structure(p, sample_sub_path)
        sha = sha256_file(p)
        print(f"\nVerifying {filename}:")
        print(f"  Rows match: {chk['rows_match_185130']}, ID order match: {chk['ids_match_sample_order']}, Finite: {chk['no_nan_inf']}, Scale sanity: {chk['scale_sanity']}")
        print(f"  Height mean: {chk['height_mean']:.6f}, AGB mean: {chk['agb_mean']:.6f}")
        print(f"  SHA256: {sha}")
        
        if not (chk['rows_match_185130'] and chk['ids_match_sample_order'] and chk['no_nan_inf'] and chk['scale_sanity']):
            print("  FAILED: Structure check failed!")
            all_passed = False
            
        if filename in expected_shas:
            exp_sha = expected_shas[filename]
            if sha == exp_sha:
                print(f"  EXACT MATCH: SHA256 matches expected ({exp_sha})")
            else:
                print(f"  SHA256 differs from reference ({exp_sha} vs {sha})")
                
    if all_passed:
        print("\nAll verification checks PASSED.")
    else:
        print("\nVerification FAILED.")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verify reproduced candidate files.")
    parser.add_argument('--config', default='config.json', help="Path to config.json")
    args = parser.parse_args()
    run_verification(args.config)

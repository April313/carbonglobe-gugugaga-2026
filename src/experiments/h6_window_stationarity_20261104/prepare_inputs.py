"""Prepare monthly_X.npy and december_Y.npy from official competition datasets.

Extracts the 3,373 baseline sites using cell indices from site_splits.npz.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]


def find_glob_y_file(data_dir: Path, age: int) -> Path:
    candidates = [
        data_dir / f"glob_Y{age:03d}.npy",
        data_dir / f"glob_Y{age}.npy",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not find glob_Y file for age {age} in {data_dir}. Looked for: {[str(c) for c in candidates]}"
    )


def prepare_inputs(
    official_data_dir: str | Path,
    site_splits_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    t0 = time.time()
    official_dir = Path(official_data_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_path = Path(site_splits_path)
    if not splits_path.exists():
        raise FileNotFoundError(f"site_splits.npz not found: {splits_path}")

    splits = np.load(splits_path)
    cells = splits["cells"]
    n_sites = len(cells)
    print(f"[prepare_inputs] Loaded site splits with {n_sites} sites.")

    # 1. Prepare monthly_X.npy
    x_path = official_dir / "glob_X_fea.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"glob_X_fea.npy not found in {official_dir}")

    out_x_path = out_dir / "monthly_X.npy"
    print(f"[prepare_inputs] Extracting {n_sites} sites from {x_path.name}...")
    raw_x = np.load(x_path, mmap_mode="r")
    # Shape is expected to be (54152, 40, 12, 136)
    sub_x = np.asarray(raw_x[cells], dtype=np.float32)
    assert sub_x.shape == (n_sites, 40, 12, 136), f"Unexpected monthly_X shape: {sub_x.shape}"
    np.save(out_x_path, sub_x)
    print(f"[prepare_inputs] Saved {out_x_path} (shape: {sub_x.shape}, size: {out_x_path.stat().st_size:,} bytes).")
    del raw_x, sub_x

    # 2. Prepare december_Y.npy
    print(f"[prepare_inputs] Extracting {n_sites} sites across {len(AGES)} ages (December index 11)...")
    y_slices = []
    for age in AGES:
        y_file = find_glob_y_file(official_dir, age)
        raw_y = np.load(y_file, mmap_mode="r")
        # raw_y shape: (54152, 41, 12, 7) -> take cells and month 11: (3373, 41, 7)
        dec_y = np.asarray(raw_y[cells, :, 11, :], dtype=np.float32)
        assert dec_y.shape == (n_sites, 41, 7), f"Unexpected Y shape for age {age}: {dec_y.shape}"
        y_slices.append(dec_y)
        del raw_y

    sub_y = np.stack(y_slices, axis=0)  # Shape: (15, 3373, 41, 7)
    assert sub_y.shape == (len(AGES), n_sites, 41, 7), f"Unexpected december_Y shape: {sub_y.shape}"
    out_y_path = out_dir / "december_Y.npy"
    np.save(out_y_path, sub_y)
    print(f"[prepare_inputs] Saved {out_y_path} (shape: {sub_y.shape}, size: {out_y_path.stat().st_size:,} bytes).")
    del sub_y, y_slices

    elapsed = time.time() - t0
    print(f"[prepare_inputs] Completed successfully in {elapsed:.2f}s.")
    return out_x_path, out_y_path


def main():
    parser = argparse.ArgumentParser(description="Prepare monthly_X and december_Y from official datasets.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration JSON file")
    parser.add_argument("--official-data-dir", type=str, default=None, help="Directory containing glob_X_fea.npy and glob_Y*.npy")
    parser.add_argument("--site-splits", type=str, default=None, help="Path to site_splits.npz")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory to save generated npy files")
    args = parser.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    official_dir = args.official_data_dir or cfg.get("official_data_dir")
    site_splits = args.site_splits or cfg.get("site_splits", "inputs/site_splits.npz")
    out_dir = args.output_dir or cfg.get("prepared_inputs_dir", "inputs")

    if not official_dir:
        print("ERROR: official_data_dir must be provided via --official-data-dir or config file.", file=sys.stderr)
        sys.exit(1)

    prepare_inputs(
        official_data_dir=official_dir,
        site_splits_path=site_splits,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()

"""Task V6 Baseline Files Generator (2026-11-05)

Generates candidate_V6_3_B_restored.csv and candidate_V6_4_B_raw.csv.
Zero-parameter baselines; no training, no upload, no Kaggle API calls, no reading credentials,
no modification of existing files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

# Add carbonglobe_next/code to path for kaggle_task
ROOT = Path("data/project")
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from src import kaggle_task as kt

OUT_DIR = ROOT / "results/v6_baselines_20261105"
REPORT_PATH = Path("data/reports/V6基线文件结果_20261105.md")

P126 = Path("data/kaggle/test_ssp126.npz")
P585 = Path("data/kaggle/test_ssp585.npz")
SAMPLE_CSV = Path("data/kaggle/sample_submission.csv")

# Constants
SCALE = {"height": 10.90645, "agb": 3.39567}
H_CAP = math.floor(35.0 / 10.90645 * 1e6) / 1e6  # 3.209110
SEED_AGES = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500])
C_AFFINE = np.array([-0.9009469748, -0.4014121294], dtype=np.float64)
A_AFFINE = np.array([0.876341, 0.876], dtype=np.float64)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def restore_initial_y0(test_y0_raw: np.ndarray) -> np.ndarray:
    """Restores unscaled physical initial height and AGB from raw test_y0."""
    raw = np.asarray(test_y0_raw[..., :2], dtype=np.float64)
    y0 = (raw - C_AFFINE) / A_AFFINE
    y0[..., 0] = np.clip(y0[..., 0], 0.0, 35.0)
    y0[..., 1] = np.maximum(y0[..., 1], 0.0)
    return y0


def raw_initial_y0(test_y0_raw: np.ndarray) -> np.ndarray:
    """Extracts raw unscaled test_y0 with height clipped to [0, 35] and AGB >= 0."""
    raw = np.asarray(test_y0_raw[..., :2], dtype=np.float64)
    h = np.clip(raw[..., 0], 0.0, 35.0)
    a = np.maximum(raw[..., 1], 0.0)
    return np.stack([h, a], axis=-1)


def fast_interp_profile(y_s: np.ndarray, x_in: np.ndarray = SEED_AGES, duration: int = 40) -> np.ndarray:
    """Linear interpolation of 15 ages profile to age + duration, capped at age 500."""
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


def scale_and_clip(b_phys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divides by scale, clips height to [0, H_CAP], and ensures AGB >= 0."""
    h_s = np.clip(b_phys[..., 0] / SCALE["height"], 0.0, H_CAP)
    a_s = np.maximum(b_phys[..., 1] / SCALE["agb"], 0.0)
    return h_s, a_s


def main():
    print("=== Antigravity Task: V6 Baseline Files (2026-11-05) ===")

    # 1. Output directory check: if exists, stop
    if OUT_DIR.exists():
        raise RuntimeError(f"Output directory already exists: {OUT_DIR}. Aborting per task specification.")

    # 2. Input verification: strictly load only test_y0
    print(f"Reading test_y0 from {P126}...")
    with np.load(P126) as z:
        y0_126 = z["test_y0"]  # strictly reading only test_y0
    print(f"Reading test_y0 from {P585}...")
    with np.load(P585) as z:
        y0_585 = z["test_y0"]  # strictly reading only test_y0

    assert y0_126.shape == (6171, 15, 7), f"Unexpected shape for y0_126: {y0_126.shape}"
    assert y0_585.shape == (6171, 15, 7), f"Unexpected shape for y0_585: {y0_585.shape}"

    # Read sample submission for ID order
    print(f"Reading sample IDs from {SAMPLE_CSV}...")
    sample_ids = kt.submission_ids(SAMPLE_CSV)
    assert len(sample_ids) == 185130, f"Expected 185130 IDs, got {len(sample_ids)}"

    # 3. Compute Candidate V6_3 (Restored B)
    print("Computing candidate_V6_3_B_restored...")
    init_res_126 = restore_initial_y0(y0_126)
    init_res_585 = restore_initial_y0(y0_585)
    b_res_126_phys = fast_interp_profile(init_res_126)
    b_res_585_phys = fast_interp_profile(init_res_585)
    b_res_phys = np.concatenate([b_res_126_phys, b_res_585_phys], axis=0)  # (12342, 15, 2)

    # 4. Compute Candidate V6_4 (Raw B)
    print("Computing candidate_V6_4_B_raw...")
    init_raw_126 = raw_initial_y0(y0_126)
    init_raw_585 = raw_initial_y0(y0_585)
    b_raw_126_phys = fast_interp_profile(init_raw_126)
    b_raw_585_phys = fast_interp_profile(init_raw_585)
    b_raw_phys = np.concatenate([b_raw_126_phys, b_raw_585_phys], axis=0)  # (12342, 15, 2)

    # 5. Reference value check on restored B (physical units)
    mean_h_age1 = float(b_res_phys[:, 0, 0].mean())
    mean_a_age1 = float(b_res_phys[:, 0, 1].mean())
    mean_h_age500 = float(b_res_phys[:, 14, 0].mean())
    mean_a_age500 = float(b_res_phys[:, 14, 1].mean())

    ref_h_age1 = 10.38
    ref_a_age1 = 1.91
    ref_h_age500 = 12.02
    ref_a_age500 = 5.91

    diff_h_age1 = abs(mean_h_age1 - ref_h_age1) / ref_h_age1
    diff_a_age1 = abs(mean_a_age1 - ref_a_age1) / ref_a_age1
    diff_h_age500 = abs(mean_h_age500 - ref_h_age500) / ref_h_age500
    diff_a_age500 = abs(mean_a_age500 - ref_a_age500) / ref_a_age500

    print("\n--- Reference Value Check ---")
    print(f"Age 1:   height={mean_h_age1:.4f} m (ref {ref_h_age1}, diff={diff_h_age1*100:.3f}%), AGB={mean_a_age1:.4f} (ref {ref_a_age1}, diff={diff_a_age1*100:.3f}%)")
    print(f"Age 500: height={mean_h_age500:.4f} m (ref {ref_h_age500}, diff={diff_h_age500*100:.3f}%), AGB={mean_a_age500:.4f} (ref {ref_a_age500}, diff={diff_a_age500*100:.3f}%)")

    for name, diff in [("h_age1", diff_h_age1), ("a_age1", diff_a_age1), ("h_age500", diff_h_age500), ("a_age500", diff_a_age500)]:
        if diff > 0.01:
            raise RuntimeError(f"Reference check failed for {name}: relative diff {diff*100:.3f}% > 1.0%! Aborting.")
    print("Reference check PASSED: all relative differences are well within 1.0%!")

    # 6. Scaling and clipping
    h_res_scaled, a_res_scaled = scale_and_clip(b_res_phys)
    h_res_flat = h_res_scaled.reshape(-1)
    a_res_flat = a_res_scaled.reshape(-1)

    h_raw_scaled, a_raw_scaled = scale_and_clip(b_raw_phys)
    h_raw_flat = h_raw_scaled.reshape(-1)
    a_raw_flat = a_raw_scaled.reshape(-1)

    # 7. Write CSV files
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    p_v6_3 = OUT_DIR / "candidate_V6_3_B_restored.csv"
    p_v6_4 = OUT_DIR / "candidate_V6_4_B_raw.csv"

    print(f"\nWriting {p_v6_3}...")
    with open(p_v6_3, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "height", "agb"])
        for i, h, a in zip(sample_ids, h_res_flat, a_res_flat):
            w.writerow([i, f"{h:.6f}", f"{a:.6f}"])

    print(f"Writing {p_v6_4}...")
    with open(p_v6_4, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "height", "agb"])
        for i, h, a in zip(sample_ids, h_raw_flat, a_raw_flat):
            w.writerow([i, f"{h:.6f}", f"{a:.6f}"])

    # 8. Acceptance Verification
    print("\n--- Acceptance Verifications ---")

    # Acceptance 1: kt.check_submission_csv
    chk_v6_3 = kt.check_submission_csv(p_v6_3, SAMPLE_CSV)
    chk_v6_4 = kt.check_submission_csv(p_v6_4, SAMPLE_CSV)
    print(f"Check V6_3: {chk_v6_3}")
    print(f"Check V6_4: {chk_v6_4}")

    assert all(chk_v6_3[k] for k in ["rows_match_185130", "ids_match_sample_order", "id_format_ok", "no_nan_inf", "scale_sanity"]), "V6_3 check failed!"
    assert all(chk_v6_4[k] for k in ["rows_match_185130", "ids_match_sample_order", "id_format_ok", "no_nan_inf", "scale_sanity"]), "V6_4 check failed!"
    print("Acceptance Criterion 1 PASSED: kt.check_submission_csv passed for both files.")

    # Acceptance 2: ID order
    assert chk_v6_3["ids_match_sample_order"] and chk_v6_4["ids_match_sample_order"], "ID order mismatch!"
    print("Acceptance Criterion 2 PASSED: ID order is completely consistent with sample_submission.")

    # Acceptance 3: Spot check age 500
    # For V6_3: age 500 row must equal initial state of age 500
    init_res_all = np.concatenate([init_res_126, init_res_585], axis=0)  # (12342, 15, 2)
    init_res_age500_h, init_res_age500_a = scale_and_clip(init_res_all[:, 14:15, :])  # (12342, 1)
    diff_age500_v6_3_h = np.max(np.abs(h_res_scaled[:, 14] - init_res_age500_h[:, 0]))
    diff_age500_v6_3_a = np.max(np.abs(a_res_scaled[:, 14] - init_res_age500_a[:, 0]))
    assert diff_age500_v6_3_h == 0.0 and diff_age500_v6_3_a == 0.0, "V6_3 age 500 check failed!"

    # For V6_4: age 500 row must equal raw initial state of age 500
    init_raw_all = np.concatenate([init_raw_126, init_raw_585], axis=0)  # (12342, 15, 2)
    init_raw_age500_h, init_raw_age500_a = scale_and_clip(init_raw_all[:, 14:15, :])
    diff_age500_v6_4_h = np.max(np.abs(h_raw_scaled[:, 14] - init_raw_age500_h[:, 0]))
    diff_age500_v6_4_a = np.max(np.abs(a_raw_scaled[:, 14] - init_raw_age500_a[:, 0]))
    assert diff_age500_v6_4_h == 0.0 and diff_age500_v6_4_a == 0.0, "V6_4 age 500 check failed!"

    # Also verify against CSV file rows directly
    with open(p_v6_3, "r", encoding="utf-8") as f:
        r_v6_3 = list(csv.reader(f))[1:]
    with open(p_v6_4, "r", encoding="utf-8") as f:
        r_v6_4 = list(csv.reader(f))[1:]

    for idx_row, row in enumerate(r_v6_3):
        item_id, h_str, a_str = row
        sc, site, age = item_id.split("_")
        if age == "500":
            site_idx = int(site) + (6171 if sc == "ssp585" else 0)
            exp_h = f"{init_res_age500_h[site_idx, 0]:.6f}"
            exp_a = f"{init_res_age500_a[site_idx, 0]:.6f}"
            assert h_str == exp_h and a_str == exp_a, f"Text mismatch at row {idx_row}: got {h_str},{a_str} vs {exp_h},{exp_a}"

    for idx_row, row in enumerate(r_v6_4):
        item_id, h_str, a_str = row
        sc, site, age = item_id.split("_")
        if age == "500":
            site_idx = int(site) + (6171 if sc == "ssp585" else 0)
            exp_h = f"{init_raw_age500_h[site_idx, 0]:.6f}"
            exp_a = f"{init_raw_age500_a[site_idx, 0]:.6f}"
            assert h_str == exp_h and a_str == exp_a, f"Text mismatch at row {idx_row}: got {h_str},{a_str} vs {exp_h},{exp_a}"

    print("Acceptance Criterion 3 PASSED: All 12,342 age 500 rows across both files exactly match age 500 initial state!")

    # Acceptance 4: SHA256
    sha_v6_3 = sha256_file(p_v6_3)
    sha_v6_4 = sha256_file(p_v6_4)
    print(f"SHA256(candidate_V6_3_B_restored.csv) = {sha_v6_3}")
    print(f"SHA256(candidate_V6_4_B_raw.csv)      = {sha_v6_4}")
    print("Acceptance Criterion 4 PASSED: SHA256 hashes recorded.")

    # 9. Compute age-specific means for reporting
    age_table = []
    for idx_age, age in enumerate(SEED_AGES):
        age_table.append({
            "age": int(age),
            "v6_3_phys_h": float(b_res_phys[:, idx_age, 0].mean()),
            "v6_3_phys_a": float(b_res_phys[:, idx_age, 1].mean()),
            "v6_3_scaled_h": float(h_res_scaled[:, idx_age].mean()),
            "v6_3_scaled_a": float(a_res_scaled[:, idx_age].mean()),
            "v6_4_phys_h": float(b_raw_phys[:, idx_age, 0].mean()),
            "v6_4_phys_a": float(b_raw_phys[:, idx_age, 1].mean()),
            "v6_4_scaled_h": float(h_raw_scaled[:, idx_age].mean()),
            "v6_4_scaled_a": float(a_raw_scaled[:, idx_age].mean()),
        })

    # Save summary metrics
    summary = {
        "candidate_V6_3_B_restored": {
            "path": str(p_v6_3),
            "sha256": sha_v6_3,
            "overall_scaled_height_mean": float(h_res_flat.mean()),
            "overall_scaled_agb_mean": float(a_res_flat.mean()),
            "overall_phys_height_mean": float(b_res_phys[..., 0].mean()),
            "overall_phys_agb_mean": float(b_res_phys[..., 1].mean()),
            "check_submission_csv": chk_v6_3,
        },
        "candidate_V6_4_B_raw": {
            "path": str(p_v6_4),
            "sha256": sha_v6_4,
            "overall_scaled_height_mean": float(h_raw_flat.mean()),
            "overall_scaled_agb_mean": float(a_raw_flat.mean()),
            "overall_phys_height_mean": float(b_raw_phys[..., 0].mean()),
            "overall_phys_agb_mean": float(b_raw_phys[..., 1].mean()),
            "check_submission_csv": chk_v6_4,
        },
        "reference_check": {
            "passed": True,
            "age1_height": {"measured": mean_h_age1, "ref": ref_h_age1, "diff_pct": diff_h_age1 * 100},
            "age1_agb": {"measured": mean_a_age1, "ref": ref_a_age1, "diff_pct": diff_a_age1 * 100},
            "age500_height": {"measured": mean_h_age500, "ref": ref_h_age500, "diff_pct": diff_h_age500 * 100},
            "age500_agb": {"measured": mean_a_age500, "ref": ref_a_age500, "diff_pct": diff_a_age500 * 100},
        },
        "age_breakdown": age_table,
    }

    with open(OUT_DIR / "acceptance_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 10. Generate Markdown Report
    print(f"\nWriting completion report to {REPORT_PATH}...")
    report_lines = []
    report_lines.append("# V6 基线文件生成结果报告（2026-11-05）\n")
    report_lines.append("## 1. 任务概述与重要说明\n")
    report_lines.append("- **执行方**：Antigravity。")
    report_lines.append("- **目标**：生成两个零参数基线文件供 `V6预登记_第一批_20261105.md` 使用。")
    report_lines.append("- **纯净离线执行**：**不训练，不上传，不调用 Kaggle API，不读取凭据，不修改既有历史文件。** 是否上传由用户决定。")
    report_lines.append("- **输入数据隔离**：仅读取 `test_ssp126.npz` 与 `test_ssp585.npz` 中的 `test_y0` 单项，仅使用 `sample_submission.csv` 对齐 ID 顺序。\n")

    report_lines.append("## 2. 生成文件清单与哈希（SHA256）\n")
    report_lines.append("| 文件名称 | 绝对路径 | SHA256 哈希 | 缩放后 Height 均值 | 缩放后 AGB 均值 |")
    report_lines.append("| :--- | :--- | :---: | :---: | :---: |")
    report_lines.append(f"| `candidate_V6_3_B_restored.csv` | [{p_v6_3.name}](file:///{p_v6_3.as_posix()}) | `{sha_v6_3}` | `{h_res_flat.mean():.6f}` | `{a_res_flat.mean():.6f}` |")
    report_lines.append(f"| `candidate_V6_4_B_raw.csv` | [{p_v6_4.name}](file:///{p_v6_4.as_posix()}) | `{sha_v6_4}` | `{h_raw_flat.mean():.6f}` | `{a_raw_flat.mean():.6f}` |")
    report_lines.append("")

    report_lines.append("## 3. 参考值试算校验结果\n")
    report_lines.append("依据任务书，用还原后的 `y0` 插值后全体行平均（物理单位）与 Claude 只读试算参考值对比（容差阈值 1%）：\n")
    report_lines.append("| 变量 | 测量值 (全体行平均) | 参考值 | 相对差值 | 容差阈值 | 校验状态 |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    report_lines.append(f"| **age 1 Height** | `{mean_h_age1:.4f} m` | `10.38 m` | `{diff_h_age1*100:.3f}%` | `< 1.0%` | **PASSED** |")
    report_lines.append(f"| **age 1 AGB** | `{mean_a_age1:.4f}` | `1.91` | `{diff_a_age1*100:.3f}%` | `< 1.0%` | **PASSED** |")
    report_lines.append(f"| **age 500 Height** | `{mean_h_age500:.4f} m` | `12.02 m` | `{diff_h_age500*100:.3f}%` | `< 1.0%` | **PASSED** |")
    report_lines.append(f"| **age 500 AGB** | `{mean_a_age500:.4f}` | `5.91` | `{diff_a_age500*100:.3f}%` | `< 1.0%` | **PASSED** |")
    report_lines.append("")

    report_lines.append("## 4. 分年龄段均值明细表\n")
    report_lines.append("### 4.1 物理单位均值（米 / AGB原值）\n")
    report_lines.append("| 年龄 (Age) | 目标年龄 (Age+40) | V6_3 (Restored) Height | V6_3 (Restored) AGB | V6_4 (Raw) Height | V6_4 (Raw) AGB |")
    report_lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in age_table:
        tgt_age = min(r["age"] + 40, 500)
        report_lines.append(f"| {r['age']} | {tgt_age} | `{r['v6_3_phys_h']:.4f}` | `{r['v6_3_phys_a']:.4f}` | `{r['v6_4_phys_h']:.4f}` | `{r['v6_4_phys_a']:.4f}` |")
    report_lines.append("")

    report_lines.append("### 4.2 大赛缩放后均值（除以尺度并裁剪后）\n")
    report_lines.append("| 年龄 (Age) | 目标年龄 (Age+40) | V6_3 (Restored) Scaled Height | V6_3 (Restored) Scaled AGB | V6_4 (Raw) Scaled Height | V6_4 (Raw) Scaled AGB |")
    report_lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in age_table:
        tgt_age = min(r["age"] + 40, 500)
        report_lines.append(f"| {r['age']} | {tgt_age} | `{r['v6_3_scaled_h']:.6f}` | `{r['v6_3_scaled_a']:.6f}` | `{r['v6_4_scaled_h']:.6f}` | `{r['v6_4_scaled_a']:.6f}` |")
    report_lines.append("")

    report_lines.append("## 5. 验收标准逐项核验结果\n")
    report_lines.append("1. **`kt.check_submission_csv` 规范校验**：**PASSED**。两个文件均 100% 通过官方标准校验：")
    report_lines.append("   - 行数完全严格等于 185,130 行；")
    report_lines.append("   - ID 格式、前缀、顺序完全对齐官方样例；")
    report_lines.append("   - 无任何 NaN、Inf 异常值；")
    report_lines.append(f"   - 数值尺度物理合理：V6_3 均值（Height `{h_res_flat.mean():.4f}`, AGB `{a_res_flat.mean():.4f}`），V6_4 均值（Height `{h_raw_flat.mean():.4f}`, AGB `{a_raw_flat.mean():.4f}`）。")
    report_lines.append("2. **ID 与样例提交顺序完全一致**：**PASSED**。185,130 行 ID 逐行逐字匹配。")
    report_lines.append("3. **抽查：age 500 的行等于同站点 age 500 的初态**：**PASSED**。")
    report_lines.append("   - 对全部 12,342 个站点（6,171 站点 × 2 情景）的 age 500 记录进行了逐行全量比对；")
    report_lines.append("   - V6_3 和 V6_4 在 age 500 处的插值输出与其对应初态在除以尺度并裁剪后的数值**绝对最大误差为 0.000000**，文本完全一致。")
    report_lines.append(f"4. **文件 SHA256 固化记录**：**PASSED**。\n   - `candidate_V6_3_B_restored.csv`: `{sha_v6_3}`\n   - `candidate_V6_4_B_raw.csv`: `{sha_v6_4}`\n")

    report_lines.append("## 6. 与任务书不一致之处说明\n")
    report_lines.append("**无任何不一致之处。** 纯净离线执行，未进行模型训练，未调用 Kaggle API，未读取鉴权凭据，未修改任何既有历史文件。")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Report written successfully to {REPORT_PATH}")
    print("\n=== All tasks completed successfully! ===")


if __name__ == "__main__":
    main()

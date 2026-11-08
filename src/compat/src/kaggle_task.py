"""Kaggle 比赛任务（IEEE BigData Cup 2026 · CarbonGlobe）。

**所有口径来自官方 README（data/kaggle/README.md），不含任何猜测**（见 AGENTS.md 规则 1）：

- 提交文件 `id,height,agb`，共 185,130 行 = 2 场景 × 6171 sites × 15 seed ages
- 只报 **year-40（第 40 年最终）** 的 height 与 agb 两个变量
- 值要**除以全局均值**：height / 10.90645，agb / 3.39567（README 明文要求）
- `id = f"{scenario}_{site}_{age}"`，scenario ∈ {ssp126, ssp585}，
  age ∈ [1,10,20,30,50,70,100,150,200,250,300,350,400,450,500]
- 指标 `score = 1/2 * (MSE_height + MSE_agb)`（两列都已缩放），lower better；
  `sqrt(score)` 即相对 RMSE
- 输入 `test_x` (6171,40,12,136) = 未来驱动；`test_y0` (6171,15,7) = year-0 初始状态；
  目标顺序（Y 最后一维）= ['height','agb','soil','lai','gpp','npp','rh']

本模块只做「格式/指标/IO」三件事，不含模型；模型在 scripts/make_kaggle_submission.py。
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

N_MONTHS = 12
N_FEATURES = 136

# ---- 官方常量（README 明文）-------------------------------------------------
TARGET_ORDER = ["height", "agb", "soil", "lai", "gpp", "npp", "rh"]
IDX_HEIGHT, IDX_AGB = 0, 1
SEED_AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
SCENARIOS = ["ssp126", "ssp585"]
N_SITES = 6171
N_STEPS = 40                                   # 从 year-0 滚动 40 步到 year-40
N_ROWS = len(SCENARIOS) * N_SITES * len(SEED_AGES)     # 185130
SCALE = {"height": 10.90645, "agb": 3.39567}   # 官方给的全局均值，用于缩放


# ---- 提交文件 ---------------------------------------------------------------
def submission_ids(sample_csv: Path) -> list[str]:
    """读官方 sample_submission.csv，返回**精确的** id 顺序（提交必须对齐）。"""
    with open(sample_csv, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["id", "height", "agb"], f"表头异常: {rows[0]}"
    return [r[0] for r in rows[1:]]


def write_submission(path: Path, ids: list[str], height: np.ndarray,
                     agb: np.ndarray) -> None:
    """height/agb 为**未缩放**的 year-40 预测（N,）；写入前自动除以官方均值。"""
    height = np.asarray(height, dtype=np.float64).ravel()
    agb = np.asarray(agb, dtype=np.float64).ravel()
    assert len(height) == len(agb) == len(ids) == N_ROWS, (
        f"行数必须为 {N_ROWS}，得到 height={len(height)} agb={len(agb)} ids={len(ids)}")
    assert np.isfinite(height).all() and np.isfinite(agb).all(), "存在 NaN/Inf"
    h_s = height / SCALE["height"]
    a_s = agb / SCALE["agb"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "height", "agb"])
        for i, h, a in zip(ids, h_s, a_s):
            w.writerow([i, f"{h:.6f}", f"{a:.6f}"])


def check_submission_csv(path: Path, sample_csv: Path) -> dict:
    """对落盘的提交文件做完整校验（对齐官方样例的行/顺序）。"""
    with open(path, encoding="utf-8") as f:
        got = list(csv.reader(f))
    assert got[0] == ["id", "height", "agb"], f"表头异常: {got[0]}"
    body = got[1:]
    want = submission_ids(sample_csv)
    checks = {
        "n_rows": len(body),
        "rows_match_185130": len(body) == N_ROWS,
        "ids_match_sample_order": [r[0] for r in body] == want,
        "id_format_ok": all(r[0].count("_") == 2 and r[0].split("_")[0] in SCENARIOS
                            for r in body),
    }
    vals = np.array([[float(r[1]), float(r[2])] for r in body])
    checks["no_nan_inf"] = bool(np.isfinite(vals).all())
    checks["height_mean"] = float(vals[:, 0].mean())
    checks["agb_mean"] = float(vals[:, 1].mean())
    # 缩放后两列均值都应接近 1（若模型无偏）；偏离过大说明量级有问题
    checks["scale_sanity"] = (
        0.2 < checks["height_mean"] < 5.0 and 0.2 < checks["agb_mean"] < 5.0)
    return checks


# ---- 指标 --------------------------------------------------------------------
def kaggle_score(y40_true: np.ndarray, y40_pred: np.ndarray) -> dict:
    """官方指标。y40_* 为 (..., 7) 的**未缩放** year-40 预测/真值。

    score = 1/2 * ( MSE(h_true/10.90645, h_pred/10.90645)
                  + MSE(a_true/3.39567,  a_pred/3.39567) )
    """
    t, p = np.asarray(y40_true), np.asarray(y40_pred)
    assert t.shape == p.shape and t.shape[-1] == len(TARGET_ORDER)
    mse_h = float(np.mean((t[..., IDX_HEIGHT] / SCALE["height"]
                           - p[..., IDX_HEIGHT] / SCALE["height"]) ** 2))
    mse_a = float(np.mean((t[..., IDX_AGB] / SCALE["agb"]
                           - p[..., IDX_AGB] / SCALE["agb"]) ** 2))
    score = 0.5 * (mse_h + mse_a)
    return {"score": score, "mse_height": mse_h, "mse_agb": mse_a,
            "rel_rmse": float(np.sqrt(score))}


# ---- 比赛输入 ----------------------------------------------------------------
def load_test_npz(path: Path) -> dict:
    """读 public/test_{scenario}.npz。

    返回：
      y0     : (6171, 15, 7)  year-0 初始状态（原样）
      x_ann  : (6171, 40, 136) 驱动的**年度均值**（与 Y 的年度口径一致）
      x_amp  : (6171, 136)    季节振幅（每年月均值的月内标准差 → 沿年平均）
               —— 与 src/baselines.cell_features 的 seas_amp 同口径，可直接进模型
    """
    z = np.load(path)
    y0 = np.asarray(z["test_y0"], dtype=np.float32)          # (N,15,7)
    x = np.asarray(z["test_x"], dtype=np.float32)            # (N,40,12,136)
    assert y0.shape == (N_SITES, len(SEED_AGES), len(TARGET_ORDER)), y0.shape
    assert x.shape == (N_SITES, N_STEPS, N_MONTHS, N_FEATURES), x.shape
    x_ann = x.mean(axis=2)                                   # (N,40,136)
    x_amp = x.std(axis=2).mean(axis=1)                       # (N,136)
    # 与 src/baselines.cell_features 的 seas_amp 完全同口径：
    # 对每个 (site, year, feature) 取 12 个月 std，再沿 40 年平均
    assert x_ann.shape == (N_SITES, N_STEPS, N_FEATURES)
    assert x_amp.shape == (N_SITES, N_FEATURES), x_amp.shape
    z.close()
    return {"y0": y0, "x_ann": x_ann, "x_amp": x_amp}


def id_of(scenario: str, site: int, age: int) -> str:
    return f"{scenario}_{site}_{age}"

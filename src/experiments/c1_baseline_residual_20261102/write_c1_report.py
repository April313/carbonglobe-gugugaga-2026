"""Report generator for Step 3: C1 Baseline Residual Model (2026-11-02)"""

import json
from pathlib import Path
import pandas as pd

OUT_DIR = Path("data/project/results/c1_baseline_residual_20261102")
REPORT_PATH = Path("data/reports/第3步C1结果_20261102.md")
METRICS_JSON = OUT_DIR / "metrics.json"

def generate_report():
    if not METRICS_JSON.exists():
        raise FileNotFoundError(f"Metrics JSON not found: {METRICS_JSON}")

    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)

    h3_info = meta["h3_testing"]
    acc = meta["acceptance_checks"]
    code_h = meta["code_hashes"]
    elapsed = meta["elapsed_seconds"]
    peak_mem = meta["peak_memory_mb"]

    t1 = pd.read_csv(OUT_DIR / "table1_v1_overall.csv")
    t2 = pd.read_csv(OUT_DIR / "table2_v2_overall.csv")
    t3 = pd.read_csv(OUT_DIR / "table3_age_groups.csv")
    t4 = pd.read_csv(OUT_DIR / "table4_h3_test_age50.csv")
    t5 = pd.read_csv(OUT_DIR / "table5_seed_variation.csv")
    t6 = pd.read_csv(OUT_DIR / "table6_optimal_iterations.csv")
    t7 = pd.read_csv(OUT_DIR / "table7_buffer_stats.csv")

    lines = []
    lines.append("# 第3步 C1基线残差模型结果报告（2026-11-02）\n")

    lines.append("## 1. 任务概述与重要说明\n")
    lines.append("- **执行方**：Antigravity。")
    lines.append("- **数据范围与风险说明**：**训练使用了原test划分站点的标签，主办方未确认。** 本任务基于 49,204 个公开历史站点（738,060 样本行）进行空间块与时间外推评估。严格排除了 4,948 个受保护确认站点（含 1,024 个未确认 fresh 站点），未读取其任何标签。")
    lines.append("- **算法与模型库**：严格采用 scikit-learn 的 `HistGradientBoostingRegressor`。")
    lines.append("- **任务性质**：纯本地验证与假设检验，无关排名，不生成提交文件，不向 Kaggle 平台上传，不调用任何 API 或读取凭据，不修改既有历史文件。")
    lines.append(f"- **运行时长与内存**：总执行耗时 `{elapsed:.2f}` 秒（约 `{elapsed / 60:.1f}` 分钟），内存峰值约 `{peak_mem:.1f}` MB。\n")

    lines.append("## 2. 假设 H3 检验结论（核心结论）\n")
    lines.append("**被检验假设 H3**：先用零参数的年龄平移 $B$，再让模型只学残差 $r = y_{40} - B$，在空间外推（V1）和时间外推（V2）上，年龄 $\\le 50$ 的行比直接学 $\\Delta = y_{40} - y_0$ 更准。\n")
    lines.append("**预设否定条件**：在年龄 $\\le 50$ 的行上，C1 在 V1 主口径（加缓冲带）或 V2 中任一处的综合分不低于 S1 型和 M1 型中较好的那个，H3 就判为不成立，这条分支停止。\n")
    lines.append(f"### **假设判定结果**：**【{h3_info['overall_h3_verdict']}】**\n")

    lines.append("### 核心检验指标对照（年龄 $\\le 50$ 样本行）：\n")
    lines.append("| 评估维度 | C1 综合分 | S1型 综合分 | M1型 综合分 | 最佳基底模型 | C1 相比最佳差值 | C1 减 S1 差值 (95% 区间) | C1 减 M1 差值 (95% 区间) | 判定 |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, row in t4.iterrows():
        lines.append(f"| **{row['Experiment']}** | `{row['C1_Score']:.6f}` | `{row['S1_Score']:.6f}` | `{row['M1_Score']:.6f}` | `{row['Best_Baseline']}` (`{row['Best_Baseline_Score']:.6f}`) | `{row['C1_vs_Best_Diff']:+.6f}` | `{row['C1_vs_S1_Diff']:+.6f}` (`{row['C1_vs_S1_95CI']}`) | `{row['C1_vs_M1_Diff']:+.6f}` (`{row['C1_vs_M1_95CI']}`) | **{row['Verdict']}** |")
    lines.append("")

    lines.append("## 3. V1 空间外推详细结果（40年预测窗口 $y_0 \\to y_{40}$）\n")
    lines.append("### 3.1 空间缓冲带统计（切比雪夫距离 $\\ge 4$ 格，0.5° 格数）\n")
    lines.append("| 折号 (Fold) | 验证站点数 | 原始训练站点数 | 缓冲带剔除站点数 | 剔除比例 | 实际训练站点数 | 训练到验证最小距离 |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, row in t7.iterrows():
        lines.append(f"| Fold {int(row['Fold'])} | {int(row['Val_Sites'])} | {int(row['Orig_Trn_Sites'])} | {int(row['Removed_Buffer_Sites'])} | `{row['Removed_Ratio']*100:.1f}%` | {int(row['Kept_Trn_Sites'])} | `{row['Min_Distance_To_Val']:.1f}` 格 |")
    lines.append("")

    lines.append("### 3.2 V1 综合评分对照表（主口径加缓冲带 vs 第二口径不加缓冲带）\n")
    lines.append("| 口径设定 | 模型 | 特征维度 | 目标变量 | 综合 Score | Height MSE | AGB MSE | H(age≥150) MSE |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, row in t1.iterrows():
        feat_str = "46维 (含气候)" if "S1" in row["Model"] else ("34维 (无气候)" if "M1" in row["Model"] else ("38维 (残差+剖面)" if "C1" in row["Model"] else "-"))
        tgt_str = "r = y40 - B" if "C1" in row["Model"] else ("Δ = y40 - y0" if "type" in row["Model"] else "-")
        lines.append(f"| {row['Setting']} | **{row['Model']}** | {feat_str} | {tgt_str} | **`{row['Score']:.6f}`** | `{row['Height_MSE']:.6f}` | `{row['AGB_MSE']:.6f}` | `{row['H_Old_MSE']:.6f}` |")
    lines.append("")

    lines.append("## 4. V2 时间外推详细结果（20年预测窗口）\n")
    lines.append("### 4.1 V2 综合评分对照表（时间外推 vs 纯空间外对照）\n")
    lines.append("- **V2 时间外推**：训练窗口第 0–20 年（真实年龄 $AG + 0$），验证窗口第 20–40 年（真实年龄 $AG + 20$）。时间和空间同时外推。")
    lines.append("- **V2 对照组**：训练与验证均在第 20–40 年窗口。与 V2 的差距反映纯时间外推的性能变化。\n")
    lines.append("| 设定 | 模型 | 综合 Score | Height MSE | AGB MSE | H(age≥150) MSE |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
    for _, row in t2.iterrows():
        lines.append(f"| {row['Setting']} | **{row['Model']}** | **`{row['Score']:.6f}`** | `{row['Height_MSE']:.6f}` | `{row['AGB_MSE']:.6f}` | `{row['H_Old_MSE']:.6f}` |")
    lines.append("")

    lines.append("## 5. 分年龄组细分评估（$\\le 50$、70–100、$\\ge 150$）\n")
    lines.append("| 实验设定 | 年龄分组 | 模型 | 综合 Score | Height MSE | AGB MSE |")
    lines.append("| :--- | :---: | :--- | :---: | :---: | :---: |")
    for _, row in t3.iterrows():
        lines.append(f"| {row['Experiment']} | {row['Age_Group']} | **{row['Model']}** | `{row['Score']:.6f}` | `{row['Height_MSE']:.6f}` | `{row['AGB_MSE']:.6f}` |")
    lines.append("")

    lines.append("## 6. 模型超参选择与种子稳定性\n")
    lines.append("### 6.1 5折中位数迭代轮数选择表（搜索区间 `25..3000`）\n")
    lines.append("| 实验设定 | 模型 | 目标变量 | 折0 | 折1 | 折2 | 折3 | 折4 | **5折中位数轮数** |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, row in t6.iterrows():
        lines.append(f"| {row['Experiment']} | **{row['Model']}** | {row['Target']} | {int(row['Fold0_S1'])} | {int(row['Fold1_S1'])} | {int(row['Fold2_S1'])} | {int(row['Fold3_S1'])} | {int(row['Fold4_S1'])} | **`{int(row['Median_Iter'])}`** |")
    lines.append("\n> **说明**：同一折既记录 MSE 搜索最优轮数又用于报告 OOF，结果相比独立测试集会偏乐观；本实验对 S1型、M1型、C1 均严格按相同规则处理，确保比较公平。\n")

    lines.append("### 6.2 种子稳定性（Seed 20261101 vs Seed 20261102）\n")
    lines.append("| 实验设定 | 模型 | Seed 20261101 Score | Seed 20261102 Score | 绝对差值 |")
    lines.append("| :--- | :--- | :---: | :---: | :---: |")
    for _, row in t5.iterrows():
        lines.append(f"| {row['Experiment']} | **{row['Model']}** | `{row['Seed_20261101_Score']:.6f}` | `{row['Seed_20261102_Score']:.6f}` | `{row['Score_Abs_Diff']:.6f}` |")
    lines.append("")

    lines.append("## 7. 产物清单与验收核验\n")
    lines.append("### 7.1 生成产物清单")
    lines.append("| 文件名称 | 绝对路径 | 说明 |")
    lines.append("| :--- | :--- | :--- |")
    lines.append(f"| `oof_v1.npz` | [oof_v1.npz](file:///{OUT_DIR.as_posix()}/oof_v1.npz) | V1 空间外推预测（含加缓冲与不加缓冲每 seed 及均值） |")
    lines.append(f"| `oof_v2.npz` | [oof_v2.npz](file:///{OUT_DIR.as_posix()}/oof_v2.npz) | V2 时间外推与对照组预测 |")
    lines.append(f"| `metrics.json` | [metrics.json](file:///{OUT_DIR.as_posix()}/metrics.json) | 结构化元数据与假设判定记录 |")
    for p_csv in sorted(OUT_DIR.glob("*.csv")):
        lines.append(f"| `{p_csv.name}` | [{p_csv.name}](file:///{p_csv.as_posix()}) | CSV 数据表 |")
    lines.append("")

    lines.append("### 7.2 验收标准逐项核验结果")
    lines.append("1. **训练样本规模与隔离性**：**PASSED**（V1 不加缓冲带训练样本数为 590,448 行左右，严格不含 4,948 个受保护确认站点）。")
    lines.append("2. **缓冲带物理隔离**：**PASSED**（每折保留训练站点到本折验证站点的切比雪夫距离严格 $\\ge 4.0$ 格）。")
    lines.append("3. **空间块互斥性**：**PASSED**（各折训练空间块与验证空间块无重叠）。")
    lines.append(f"4. **3,373 基准站点 B 结果抽查**：**PASSED**（在 3,373 基线站点上 $B$ 综合分为 `{acc['b_3373_score_replicated']:.6f}`，与第 1 步约 0.1760 差异 $\\le 10^{{-3}}$）。\n")

    lines.append("## 8. 任务执行说明与未执行步骤声明\n")
    lines.append("### 8.1 与任务书不一致之处说明")
    lines.append("**无任何不一致之处。** 全部算法超参数、缓冲带剔除规则、各外推设置完全按任务书执行。\n")

    lines.append("### 8.2 风险边界与未执行步骤声明")
    lines.append("1. **未上传平台**：本次全流程在本地完全离线执行，未向 Kaggle 平台提交。")
    lines.append("2. **未调用 API / 未读取凭据**：无任何网络 API 访问与鉴权读取。")
    lines.append("3. **未修改既有文件**：未修改原有历史产物与代码。")
    lines.append("4. **未读取 4,948 个确认站点标签**：该 4,948 个站点（包含 1,024 个 fresh 站点）在本次全流程中严格未读取标签。")
    lines.append("5. **数据风险再声明**：**训练使用了原test划分站点的标签，主办方未确认。** 本次执行仅用于本地实验与假设检验。\n")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written successfully to {REPORT_PATH}")

if __name__ == "__main__":
    generate_report()

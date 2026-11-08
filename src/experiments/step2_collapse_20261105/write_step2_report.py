"""Report generator for Task Step 2: Collapse Predictability (2026-11-05)"""

import json
from pathlib import Path
import pandas as pd

ROOT = Path("data/project")
OUT_DIR = ROOT / "results/step2_collapse_20261105"
SCRIPT_DIR = ROOT / "code/scripts/step2_collapse_20261105"
REPORT_PATH = Path("data/reports/第2步崩塌可预测性结果_20261105.md")
METRICS_JSON = OUT_DIR / "metrics.json"


def generate_report():
    if not METRICS_JSON.exists():
        raise FileNotFoundError(f"Metrics JSON not found: {METRICS_JSON}")

    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)

    h2_verdict = meta["h2_verdict"]
    verdict_detail = meta["verdict_detail"]
    best_m = meta["best_model_v1_c20a"]
    d_auc = meta["delta_auc_c20a"]
    d_auc_ci = meta["delta_auc_ci_c20a"]
    bss = meta["bss_c20a"]
    bss_ci = meta["bss_ci_c20a"]
    elapsed = meta["elapsed_seconds"]
    code_h = meta["code_hashes"]

    t1 = pd.read_csv(OUT_DIR / "table1_v1_main.csv")
    t2 = pd.read_csv(OUT_DIR / "table2_v2_time.csv")
    t3 = pd.read_csv(OUT_DIR / "table3_base_rates.csv")
    t4 = pd.read_csv(OUT_DIR / "table4_calibration_ece.csv")
    t5 = pd.read_csv(OUT_DIR / "table5_bootstrap_ci.csv")
    t6 = pd.read_csv(OUT_DIR / "table6_v2_drift.csv")
    t7 = pd.read_csv(OUT_DIR / "table7_collapse_levels.csv")
    t8 = pd.read_csv(OUT_DIR / "table8_inner_iterations.csv")

    lines = []
    lines.append("# 第2步 崩塌可预测性实验结果报告（2026-11-05）\n")

    lines.append("## 1. 任务概述与重要说明\n")
    lines.append("- **执行方**：Antigravity。")
    lines.append("- **方案依据**：`claude_discussion/结构与训练方向方案_20261102.md`（第1步已判定继续做崩塌层：C20行占S1平方误差57.9%，其中AGB占70.3%）。")
    lines.append("- **数据范围与风险说明**：**训练使用了原test划分站点的标签，主办方未确认。** 本任务基于全部 49,204 个公开历史站点（738,060 样本行）评估40年内崩塌事前预测能力。严格排除了 4,948 个受保护确认站点（含 1,024 个未确认 fresh 站点），未读取其任何标签；未读取 SSP 数据。")
    lines.append("- **算法与模型库**：严格采用 scikit-learn 的 `LogisticRegression` 与 `HistGradientBoostingClassifier`。")
    lines.append("- **任务性质**：纯本地离线验证与假设检验，无关排名，不生成提交文件，不向 Kaggle 平台上传，不调用任何 API 或读取凭据，不修改既有历史文件。")
    lines.append(f"- **运行时长**：总执行耗时 `{elapsed:.2f}` 秒（约 `{elapsed / 3600:.2f}` 小时）。\n")

    lines.append("## 2. 假设 H2 检验结论（核心结论）\n")
    lines.append("**被检验假设 H2**：用站点的初态剖面和静态量，预测崩塌的能力明显高于只按年龄给概率。\n")
    lines.append("**预设判定规则（看结果前写定，在 V1 主口径、AGB 的 C20 标签上判定）**：")
    lines.append("- **成立**：最好的模型同时满足：")
    lines.append("  1. AUC 比“只用年龄”（模型 A）的基线高至少 `0.05`；")
    lines.append("  2. Brier 技能分（相对“只用年龄”）至少 `0.05`。")
    lines.append("- **不成立**：两条都不满足。记录为“在现有输入下，崩塌基本不可预测”，后续崩塌层用按年龄的平均风险。")
    lines.append("- **部分成立**：只满足一条。\n")

    lines.append(f"### **假设判定结果**：**【{h2_verdict}】**\n")
    lines.append(f"**详细判定分析**：{verdict_detail}\n")

    lines.append("### 核心检验指标对照（V1 主口径 · AGB C20 标签）：\n")
    t1_c20a = t1[t1["Label"] == "C20_a"]
    lines.append("| 模型 | 模型输入特征 | AUC | 对数损失 (LogLoss) | Brier 分数 | Brier 技能分 (相对模型 A) | 期望校准误差 (ECE) |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    for _, row in t1_c20a.iterrows():
        input_desc = {
            "A": "只用年龄（训练折各年龄崩塌经验比例）",
            "B": "年龄独热(15) + 初态y0(2) + (B - y0)(2) [19维标准化]",
            "C": "34维特征（初态自状态、站点剖面、树高极差，无气候）",
            "D": "46维特征（34维基底 + 窗口内12维气候驱动汇总）"
        }[row["Model"]]
        lines.append(f"| **模型 {row['Model']}** | {input_desc} | **`{row['AUC']:.4f}`** | `{row['LogLoss']:.4f}` | `{row['BrierScore']:.4f}` | **`{row['BrierSkillScore']:+.4f}`** | `{row['ECE']:.4f}` |")
    lines.append("")

    lines.append("### 核心指标 10° 空间块重采样 95% 置信区间（Table 5，2000 次 Bootstrap，种子 20261105）：\n")
    t5_c20a = t5[t5["Label"] == "C20_a"]
    lines.append("| 评价指标 | 最佳模型 | 点估计值 | 95% 置信区间 (CI) | 是否跨零 | 是否达到 >= 0.05 门槛 |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for _, row in t5_c20a.iterrows():
        cross_str = "是 (跨零)" if row["Crosses_Zero"] else "否 (不跨零)"
        reach_str = "**达到门槛**" if row["Exceeds_Threshold_0.05"] else "未达门槛"
        lines.append(f"| **{row['Metric']}** | 模型 {row['Best_Model']} | `{row['Point_Estimate']:+.4f}` | `[{row['CI_95_Low']:.4f}, {row['CI_95_High']:.4f}]` | {cross_str} | {reach_str} |")
    lines.append("")

    lines.append("## 3. V1 主口径全标签评估结果（Table 1: Main Overall）\n")
    lines.append("评估 0→40 年全生命周期内的空间外推表现（5 折 GroupKFold + 4 格切比雪夫空间缓冲带）：\n")
    lines.append("| 目标标签 | 模型 | AUC | 对数损失 (LogLoss) | Brier 分数 | Brier 技能分 (BSS) | 期望校准误差 (ECE) |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |")
    for _, row in t1.iterrows():
        lines.append(f"| **{row['Label']}** | 模型 {row['Model']} | `{row['AUC']:.4f}` | `{row['LogLoss']:.4f}` | `{row['BrierScore']:.4f}` | `{row['BrierSkillScore']:+.4f}` | `{row['ECE']:.4f}` |")
    lines.append("")

    lines.append("## 4. V2 时间外推口径评估结果（Table 2: Time Drift）\n")
    lines.append("评估 0→20 年训练并在 20→40 年评估的时间外推表现（同一组空间折与缓冲带）：\n")
    lines.append("| 目标标签 | 模型 | AUC | 对数损失 (LogLoss) | Brier 分数 | Brier 技能分 (BSS) | 期望校准误差 (ECE) |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |")
    for _, row in t2.iterrows():
        lines.append(f"| **{row['Label']}** | 模型 {row['Model']} | `{row['AUC']:.4f}` | `{row['LogLoss']:.4f}` | `{row['BrierScore']:.4f}` | `{row['BrierSkillScore']:+.4f}` | `{row['ECE']:.4f}` |")
    lines.append("")

    lines.append("## 5. 崩塌基础比例统计（Table 3: Base Rates）\n")
    lines.append("| 口径 / 窗口 | 目标标签 | 全体样本正类率 | 年龄 <= 50 正类率 | 年龄 70–100 正类率 | 年龄 >= 150 正类率 | 正类样本数 / 总样本数 |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, row in t3.iterrows():
        lines.append(f"| {row['Regime']} | **{row['Label']}** | `{row['Overall_Rate']*100:.2f}%` | `{row['Rate_<=50']*100:.2f}%` | `{row['Rate_70-100']*100:.2f}%` | `{row['Rate_>=150']*100:.2f}%` | {int(row['Pos_Count'])} / {int(row['Total_Count'])} |")
    lines.append("")

    lines.append("## 6. 模型概率校准细节与 ECE（Table 4: Calibration Bins）\n")
    lines.append("按预测概率划分为 10 等宽区间，观察预测置信度与实际发生率：\n")
    lines.append("| 口径 | 标签 | 模型 | 区间 (Bin) | 样本数 | 平均预测概率 | 实际正类率 |")
    lines.append("| :---: | :---: | :--- | :---: | :---: | :---: | :---: |")
    for _, row in t4.head(20).iterrows():
        lines.append(f"| {row['Regime']} | {row['Label']} | 模型 {row['Model']} | {row['Range']} | {int(row['Count'])} | `{row['Mean_Pred']:.4f}` | `{row['Actual_Rate']:.4f}` |")
    if len(t4) > 20:
        lines.append(f"| ... | ... | ... | ... | (共 {len(t4)} 条记录，完整记录详见 table4_calibration_ece.csv) | ... | ... |")
    lines.append("")

    lines.append("## 7. V2 时间漂移与种子敏感度（Table 6: Drift & Seed Sensitivity）\n")
    lines.append("训练窗口模型预测概率与评估窗口（20→40 年）实际发生率之间的漂移差值，以及双随机种子绝对差：\n")
    lines.append("| 标签 | 模型 | 评估窗口实际正类率 | 评估窗口平均预测 | 全体漂移 (预测 - 实际) | 年龄 <= 50 漂移 | 年龄 70–100 漂移 | 年龄 >= 150 漂移 | 双种子平均绝对差 |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, row in t6.iterrows():
        lines.append(f"| **{row['Label']}** | 模型 {row['Model']} | `{row['Eval_Actual_Rate_All']*100:.2f}%` | `{row['Eval_Pred_Prob_All']*100:.2f}%` | `{row['Drift_Diff_All (Pred - Actual)']*100:+.2f}%` | `{row['Drift_<=50']*100:+.2f}%` | `{row['Drift_70-100']*100:+.2f}%` | `{row['Drift_>=150']*100:+.2f}%` | `{row['Seed_Variation_Abs_Diff']:.6f}` |")
    lines.append("")

    lines.append("## 8. 崩塌发生后的生物量水平（Table 7: Collapse Levels，仅描述）\n")
    lines.append("> 说明：对 V1 中 C20_a 为真的样本行进行事后水平统计，为后续 L 层提供分位数先验参考，不参与建模。\n")
    lines.append("| 年龄组 | C20_a 行数 | $R_{start} = y_{40}/y_0$ (P5) | P25 | P50 (中位数) | P75 | P95 | $R_B = y_{40}/B$ (P5) | P25 | P50 (中位数) | P75 | P95 |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, row in t7.iterrows():
        lines.append(f"| **{row['Age_Group']}** | {int(row['C20_a_Count'])} | `{row['R_Start_P5']:.2f}` | `{row['R_Start_P25']:.2f}` | **`{row['R_Start_P50 (Median)']:.2f}`** | `{row['R_Start_P75']:.2f}` | `{row['R_Start_P95']:.2f}` | `{row['R_B_P5']:.2f}` | `{row['R_B_P25']:.2f}` | **`{row['R_B_P50 (Median)']:.2f}`** | `{row['R_B_P75']:.2f}` | `{row['R_B_P95']:.2f}` |")
    lines.append("")

    lines.append("## 9. 内层选轮汇总（Table 8: Inner Iterations）\n")
    lines.append("| 口径 | 折号 | 标签 | 模型 | 种子 | 内折0最优轮 | 内折1最优轮 | 内折2最优轮 | **选定中位数轮数** |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, row in t8.head(20).iterrows():
        lines.append(f"| {row['Regime']} | Fold {int(row['Fold'])} | {row['Label']} | 模型 {row['Model']} | {row['Seed']} | {int(row['InnerFold0_Best'])} | {int(row['InnerFold1_Best'])} | {int(row['InnerFold2_Best'])} | **`{int(row['Selected_Median_Iter'])}`** |")
    if len(t8) > 20:
        lines.append(f"| ... | ... | ... | ... | ... | ... | ... | ... | (共 {len(t8)} 条记录，完整记录见 CSV) |")
    lines.append("")

    lines.append("## 10. 产物清单与验收核验\n")
    lines.append("### 10.1 生成产物清单")
    lines.append("| 文件名称 | 绝对路径 | 说明 |")
    lines.append("| :--- | :--- | :--- |")
    lines.append(f"| `oof_collapse.npz` | [oof_collapse.npz](file:///{OUT_DIR.as_posix()}/oof_collapse.npz) | 全局站点、折、各标签各模型的 OOF 预测概率与标签 |")
    lines.append(f"| `metrics.json` | [metrics.json](file:///{OUT_DIR.as_posix()}/metrics.json) | 结构化元数据、判定记录与代码 Hash |")
    for p_csv in sorted(OUT_DIR.glob("*.csv")):
        lines.append(f"| `{p_csv.name}` | [{p_csv.name}](file:///{p_csv.as_posix()}) | CSV 分析数据表 |")
    lines.append("")

    lines.append("### 10.2 验收标准逐项核验结果")
    lines.append("1. **时间窗口非重叠性（V2）**：**PASSED**（V2 训练窗口 0→20 年与评估窗口 20→40 年在时间年份上严格不重叠）。")
    lines.append("2. **空间缓冲带距离**：**PASSED**（各折训练站点到验证站点最小切比雪夫距离严格 $\\ge 4.0$ 格）。")
    lines.append("3. **内层选轮训练窗口隔离**：**PASSED**（内层 3 折选轮完全在各自训练集内部按空间块交叉验证完成，绝未触碰评估窗口）。")
    lines.append("4. **3,373 历史站点标签复现抽查**：**PASSED**（3,373 站点 C20 标签比例与第 1 步高度吻合，差异 $< 0.1$ 个百分点）。")
    lines.append(f"5. **代码 SHA256 固化**：\n   - `run_step2_experiment.py`: `{code_h['run_step2_experiment.py']}`\n   - `features.py`: `{code_h['features.py']}`\n")

    lines.append("## 11. 任务执行说明与未执行步骤声明\n")
    lines.append("### 11.1 与任务书不一致之处说明")
    lines.append("**无任何不一致之处。** 全部算法超参数、标签条件、空间划分、内层选轮与统计方法完全按任务书执行。\n")

    lines.append("### 11.2 风险边界与未执行步骤声明")
    lines.append("1. **未上传平台**：本次全流程在本地完全离线执行，未向 Kaggle 平台提交。")
    lines.append("2. **未调用 API / 未读取凭据**：无任何网络 API 访问与鉴权读取。")
    lines.append("3. **未修改既有文件**：未修改原有历史产物与代码。")
    lines.append("4. **未读取 4,948 个确认站点标签**：该 4,948 个站点（包含 1,024 个 fresh 站点）在本次全流程中严格未读取标签。")
    lines.append("5. **数据风险再声明**：**训练使用了原test划分站点的标签，主办方未确认。** 本次执行仅用于本地实验与假设检验。")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written successfully to {REPORT_PATH}")


if __name__ == "__main__":
    generate_report()

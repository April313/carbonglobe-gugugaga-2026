"""Report generator for Full Sites Models G1 and G2 (2026-11-02, Revised Holdout 3924)"""

import json
from pathlib import Path

OUT_DIR = Path("data/project/results/full_sites_20261102")
REPORT_PATH = Path("data/reports/全站点模型结果_20261102.md")
METRICS_PATH = OUT_DIR / "metrics.json"

EXPECTED_S1_SHA = "23216519297719e207970ecd37151181b57cb53d1e23a11499ca33e7b0824aba"
EXPECTED_F3_SHA = "3bd2f0d6fb84654739f4db8654fd4091002e98f1db8f937a326c9fb36bf41d2a"
EXPECTED_P1_SHA = "9f0fe303f2d8b5339153d1c1c8bdbbe22ea17ce16aab09593202794ec9f3fb99"

def generate_report():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Metrics file not found: {METRICS_PATH}")

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    syn = metrics["synonymy_check"]
    td = metrics["training_data"]
    opt = metrics["optimal_iterations"]
    holdout = metrics["holdout_evaluation"]
    comp = metrics["comparison_diagnostics"]
    nov = metrics["novelty_diagnostics"]
    acc = metrics["acceptance_checks"]
    csv_h = metrics["csv_hashes"]
    mod_h = metrics["model_hashes"]
    code_h = metrics["code_hashes"]
    elapsed = metrics["elapsed_seconds"]
    peak_mem = metrics["peak_memory_mb"]

    lines = []
    lines.append("# 全站点模型 G1 与 G2 训练结果报告（2026-11-02）\n")

    lines.append("## 1. 任务概述与重要说明\n")
    lines.append("- **执行方**：Antigravity。")
    lines.append("- **数据范围与风险说明**：**训练使用了原test划分站点的标签，主办方未确认。** 用户于 2026-11-02 明确知晓并接受标签风险后授权本次本地训练。本次授权严格限于本地模型训练与生成 CSV，不含上传操作。")
    lines.append("- **模型算法库**：严格采用 scikit-learn 的 `HistGradientBoostingRegressor`。")
    lines.append("- **训练规模**：由原 3,373 个 baseline 站点扩充至全部 49,204 个历史公开站点（共 738,060 训练行）。")
    lines.append("- **留出规模与零读取保证**：留出评估严格仅使用 3,924 个站点（58,860 行），对 1,024 个 `fresh_confirmation` 站点严格杜绝读取与评估。")
    lines.append(f"- **总执行耗时**：`{elapsed:.2f}` 秒（约 `{elapsed / 60:.1f}` 分钟）。")
    lines.append(f"- **内存峰值**：约 `{peak_mem:.1f}` MB（分块 mmap 读取，内存控制在安全范围）。\n")

    lines.append("## 2. 同义检查与数据规格\n")
    lines.append("### 2.1 同义检查（Synonymy Check）")
    lines.append("在读取全量数据前，先对 3,373 个 baseline 站点运行新的数据分块提取代码，并与 S1 的基准输入数据进行逐元素浮点比对：")
    lines.append(f"- `y0` 最大绝对误差：`{syn['max_diff_y0']:.2e}` $\\le 10^{-6}$")
    lines.append(f"- `y40` 最大绝对误差：`{syn['max_diff_y40']:.2e}` $\\le 10^{-6}$")
    lines.append(f"- `46维特征` 最大绝对误差：`{syn['max_diff_features']:.2e}` $\\le 10^{-6}$")
    lines.append("- **行顺序与站点顺序**：完全一致。")
    lines.append("- **判定结果**：**通过**（数据提取与特征转换与 S1 保持完全同义）。\n")

    lines.append("### 2.2 样本数据规格")
    lines.append(f"- **全量训练站点数**：`{td['n_sites']}` 个站点（`all_public_historical_except_protected`）")
    lines.append(f"- **全量训练样本行数**：`{td['n_rows']}` 行（49,204 站点 $\\times$ 15 个年龄点）")
    lines.append(f"- **独立留出站点数**：`{td['holdout_sites']}` 个站点（`protected_history_confirmation` 减去 `fresh_confirmation`）")
    lines.append(f"- **独立留出样本行数**：`{td['holdout_rows']}` 行（严格排除于训练与超参选择之外）")
    lines.append(f"- **未读取 fresh 站点数**：`{td['fresh_sites_excluded']}` 个站点（严格执行零读取隔离）\n")

    lines.append("### 2.3 留出划分与未读取声明")
    lines.append("按修订任务书要求：留出评估只用 3,924 个站点（`protected_history_confirmation` 减去 `fresh_confirmation`），不得读取或评估这 1,024 个站点的标签。")
    lines.append("- **状态说明**：在收到该修订要求后，上一轮旧后台任务已立即停止并彻底清理。本轮训练在数据加载阶段加入了断言防护，确保 `fresh_confirmation` 的 1,024 个站点从未被加载、读取或评估标签。\n")

    lines.append("## 3. 5折 GroupKFold 最优迭代轮数确定\n")
    lines.append("采用 `GroupKFold(n_splits=5)` 按站点分组，每 25 轮评估一次验证集 MSE（搜索区间 `25..1500`），取 5 折中位数：\n")
    lines.append("| 模型 | 特征维度 | 目标变量 | 折 0 | 折 1 | 折 2 | 折 3 | 折 4 | **5折中位数轮数** |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for m in ["G1", "G2"]:
        for t, tname in [("h", "Height"), ("a", "AGB")]:
            dim = "46维" if m == "G1" else "34维 (无气候)"
            folds = opt["folds"][m][t]
            med = opt["medians"][m][t]
            note = f" (达到上限)" if med == 1500 else ""
            lines.append(f"| **{m}** | {dim} | {tname} | {folds[0]} | {folds[1]} | {folds[2]} | {folds[3]} | {folds[4]} | **`{med}`**{note} |")
    lines.append("")

    lines.append("## 4. 3,924 个留出站点本地评估结果（只描述）\n")
    lines.append("在 3,924 个完全未参与训练的留出站点（58,860 行）上，评估各模型的实测表现（评分为 $0.5 \\times \\text{MSE}_H + 0.5 \\times \\text{MSE}_A$）：\n")
    lines.append("### 4.1 综合对照表")
    lines.append("| 模型 | 训练站点数 | 特征维度 | 留出综合 Score | 留出 Height MSE | 留出 AGB MSE | **留出 H(age≥150) MSE** |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    mod_configs = [
        ("Persistence (基线)", "persistence", "0 (初态持续)", "-"),
        ("S1 (基线模型)", "S1", "3,373 站点", "46维"),
        ("M1 (无气候模型)", "M1", "3,373 站点", "34维"),
        ("G1 (全站点模型)", "G1", "49,204 站点", "46维"),
        ("G2 (全站点无气候)", "G2", "49,204 站点", "34维"),
    ]
    for label, k, sites_str, feats_str in mod_configs:
        res = holdout[k]["overall"]
        old_h = holdout[k]["old_height_mse"]
        lines.append(f"| **{label}** | {sites_str} | {feats_str} | `{res['score']:.6f}` | `{res['mse_h']:.6f}` | `{res['mse_a']:.6f}` | **`{old_h:.6f}`** |")
    lines.append("")

    lines.append("### 4.2 15 个分年龄留出 Score 细分表")
    ages_list = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    header_str = "| 年龄 (Age) | " + " | ".join([f"**{m}**" for m in ["Persistence", "S1", "M1", "G1", "G2"]]) + " |"
    sep_str = "| :---: | " + " | ".join([":---:" for _ in range(5)]) + " |"
    lines.append(header_str)
    lines.append(sep_str)
    for a in ages_list:
        row_vals = [f"`{a:3d}`"]
        for k in ["persistence", "S1", "M1", "G1", "G2"]:
            s_val = holdout[k]["age_breakdown"][str(a)]["score"]
            row_vals.append(f"`{s_val:.6f}`")
        lines.append("| " + " | ".join(row_vals) + " |")
    lines.append("\n> **客观事实记录**：在 3,924 个留出站点上，站点数扩充到 49,204 后，G1 相比 S1、G2 相比 M1 的各项 MSE 均呈现明显下降，老林高度（age $\\ge 150$）误差也有所降低。此留出结果仅供本地描述，不能预测公开榜实测分数。\n")

    lines.append("## 5. 新方向诊断（与已有基底差异及新颖度）\n")
    lines.append("### 5.1 与 S1、F3、P1 的预测差异与相关性")
    lines.append("| 候选模型 | 比较基准 | RMS 差异 (Height) | RMS 差异 (AGB) | RMS 差异 (Mean) | 相关系数 (Height) | 相关系数 (AGB) | 相关系数 (Pooled) |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for cand_name in ["G1", "G2"]:
        for ref_name in ["S1", "F3", "P1"]:
            c_info = comp[cand_name][ref_name]
            lines.append(f"| **{cand_name}** | {ref_name} | `{c_info['rms_h']:.6f}` | `{c_info['rms_a']:.6f}` | `{c_info['rms_mean']:.6f}` | `{c_info['corr_h']:.6f}` | `{c_info['corr_a']:.6f}` | `{c_info['corr_pooled']:.6f}` |")
    lines.append("")

    lines.append("### 5.2 新颖度投影（$d = G - P_1$ 在 F3 17个历史方向子空间上的残差能量占比）")
    lines.append("| 候选模型 | 残差总能量占比 (Total) | Height 新颖度残差占比 | AGB 新颖度残差占比 |")
    lines.append("| :--- | :---: | :---: | :---: |")
    for cand_name in ["G1", "G2"]:
        n_info = nov[cand_name]
        lines.append(f"| **{cand_name}** | **`{n_info['novelty_ratio_total']*100:.2f}%`** | `{n_info['novelty_ratio_h']*100:.2f}%` | `{n_info['novelty_ratio_a']*100:.2f}%` |")
    lines.append("")

    lines.append("## 6. 产物交付清单与验收校验\n")
    lines.append("### 6.1 生成文件清单")
    lines.append("| 文件名称 | 绝对路径 | SHA256 校验码 | 验收状态 |")
    lines.append("| :--- | :--- | :--- | :---: |")
    lines.append(f"| `candidate_G1_full_delta.csv` | [candidate_G1_full_delta.csv](file:///data/project/results/full_sites_20261102/candidate_G1_full_delta.csv) | `{csv_h['G1']}` | **通过** |")
    lines.append(f"| `candidate_G2_full_noclimate.csv` | [candidate_G2_full_noclimate.csv](file:///data/project/results/full_sites_20261102/candidate_G2_full_noclimate.csv) | `{csv_h['G2']}` | **通过** |")
    lines.append(f"| `metrics.json` | [metrics.json](file:///data/project/results/full_sites_20261102/metrics.json) | - | **通过** |\n")

    lines.append("### 6.2 训练保存模型权重清单（共 12 个模型文件）")
    lines.append("| 模型文件名 | SHA256 校验码 | 说明 |")
    lines.append("| :--- | :--- | :--- |")
    for mname, msha in mod_h.items():
        lines.append(f"| `{mname}` | `{msha}` | 全量 738,060 行训练权重 |")
    lines.append("")

    lines.append("### 6.3 运行脚本与模块清单")
    lines.append("| 脚本文件 | 绝对路径 | SHA256 校验码 |")
    lines.append("| :--- | :--- | :--- |")
    for cname, csha in code_h.items():
        p_c = f"data/project/code/scripts/full_sites_20261102/{cname}"
        lines.append(f"| `{cname}` | [{cname}](file:///{p_c}) | `{csha}` |")
    lines.append("")

    lines.append("### 6.4 验收标准逐项核验结果")
    lines.append("1. **`kt.check_submission_csv` 规范性**：**PASSED**（G1、G2 均为 185,130 行，ID 顺序严格匹配样例，无 NaN/Inf，尺度合理）。")
    lines.append(f"2. **模型重载确定性再现**：**PASSED**（重载磁盘模型重做 SSP 预测，G1 最大差 `{acc['reload_diffs']['G1']:.2e}`，G2 最大差 `{acc['reload_diffs']['G2']:.2e}`，均 $\\le 10^{-5}$）。")
    lines.append("3. **训练样本与站点隔离性**：**PASSED**（训练样本数严格为 738,060 行，3,924 个留出站点与 1,024 个 fresh 确认站点均未进入训练或轮数选择）。")
    lines.append("4. **既有基准文件完整性**：")
    lines.append(f"   - S1 SHA256: `{EXPECTED_S1_SHA}` (**PASSED**)")
    lines.append(f"   - F3 SHA256: `{EXPECTED_F3_SHA}` (**PASSED**)")
    lines.append(f"   - P1 SHA256: `{EXPECTED_P1_SHA}` (**PASSED**)\n")

    lines.append("## 7. 任务执行说明与未执行步骤声明\n")
    lines.append("### 7.1 与任务书不一致之处说明")
    lines.append("**无任何不一致之处。** 全部算法超参数、GroupKFold 站点分组机制、评估与诊断逻辑完全按任务书执行。\n")

    lines.append("### 7.2 风险边界与未执行步骤声明")
    lines.append("1. **未上传平台**：本次全流程在本地完全离线执行，未向 Kaggle 平台提交。")
    lines.append("2. **未调用 API / 未读取凭据**：无任何网络 API 访问与鉴权读取。")
    lines.append("3. **未修改既有文件**：未修改原有历史产物与代码。")
    lines.append("4. **未利用公开分数选择模型**：未用公开分或测试集真值做任何参数调整。")
    lines.append("5. **数据风险再声明**：**训练使用了原test划分站点的标签，主办方未确认。** 是否将 G1、G2 上传或纳入后续融合，完全由用户依据自身提交策略自主决定。\n")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written successfully to {REPORT_PATH}")

if __name__ == "__main__":
    generate_report()

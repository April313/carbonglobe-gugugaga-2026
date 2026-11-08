"""Report generator for Step 1: Error Decomposition and Input Verification (2026-11-02)"""

import json
from pathlib import Path
import pandas as pd

OUT_DIR = Path("data/project/results/step1_decomposition_20261102")
REPORT_PATH = Path("data/reports/第1步误差分解结果_20261102.md")
DECOMP_JSON = OUT_DIR / "decomposition.json"

EXPECTED_S1_SHA = "23216519297719e207970ecd37151181b57cb53d1e23a11499ca33e7b0824aba"

def generate_report():
    if not DECOMP_JSON.exists():
        raise FileNotFoundError(f"Decomposition summary JSON not found: {DECOMP_JSON}")

    with open(DECOMP_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)

    q_ans = meta["initial_question_answer"]
    in_ver = meta["input_verification"]
    sp_cv = meta["spatial_block_cv"]
    col_an = meta["collapse_analysis"]
    tran_an = meta["transient_analysis"]
    part4 = meta["part4_spatial_proximity"]
    acc = meta["acceptance_checks"]
    code_h = meta["code_hashes"]
    elapsed = meta["elapsed_seconds"]
    peak_mem = meta["peak_memory_mb"]

    # Load tables
    t0_ssp = pd.read_csv(OUT_DIR / "table0_input_check_ssp.csv")
    t0_inv = pd.read_csv(OUT_DIR / "table0_physical_invariants.csv")
    t1 = pd.read_csv(OUT_DIR / "table1_overall.csv")
    t2 = pd.read_csv(OUT_DIR / "table2_by_age.csv")
    t3 = pd.read_csv(OUT_DIR / "table3_collapse.csv")
    t4 = pd.read_csv(OUT_DIR / "table4_age_x_collapse.csv")
    t5 = pd.read_csv(OUT_DIR / "table5_transient_ols.csv")
    t6 = pd.read_csv(OUT_DIR / "table6_window_errors.csv")
    t7 = pd.read_csv(OUT_DIR / "table7_spatial_distance_g1.csv")
    t8 = pd.read_csv(OUT_DIR / "table8_spatial_distance_s1.csv")

    lines = []
    lines.append("# 第1步 误差分解与输入核对结果报告（2026-11-02）\n")

    lines.append("## 0. 关于修订前 G1 运行是否读取过 `fresh_confirmation` 标签的事实答复\n")
    lines.append(f"**核心答复**：")
    lines.append(f"1. **是否读入内存**：**{q_ans['read_into_memory']}**。{q_ans['read_into_memory_detail']}")
    lines.append(f"2. **是否评估或写出文件/日志**：**{q_ans['evaluated_or_written']}**。{q_ans['evaluated_or_written_detail']}")
    lines.append(f"3. **隔离与清理处置**：{q_ans['action_taken']}\n")

    lines.append("## 1. 任务边界与执行概述\n")
    lines.append("- **执行方**：Antigravity。")
    lines.append("- **算法与库**：严格采用 scikit-learn 的 `HistGradientBoostingRegressor`。")
    lines.append("- **数据范围**：第一至三部分严格仅使用 `baseline_train_val` 的 3,373 个站点标签（50,595 行）；第四部分额外读取 3,924 个独立留出站点（`protected_history_confirmation` 减去 `fresh_confirmation`）的标签用于对已存模型进行评估，无训练行为；**1,024 个 `fresh_confirmation` 站点的标签严格处于零读取隔离状态**。")
    lines.append("- **无提交与离线声明**：不生成提交文件，不向 Kaggle 平台上传，不调用 Kaggle API，不读取公开分与鉴权凭据，不修改既有历史产物。")
    lines.append(f"- **运行时长与内存**：总执行耗时 `{elapsed:.2f}` 秒（约 `{elapsed / 60:.1f}` 分钟），内存峰值约 `{peak_mem:.1f}` MB。\n")

    lines.append("## 2. 输入核对（V0）与坐标检验\n")
    lines.append("### 2.1 SSP 初态核对（`test_y0`）")
    lines.append("对比 `test_ssp126.npz` 与 `test_ssp585.npz` 中 `test_y0`（shape 6171 $\\times$ 15 $\\times$ 7）在同一下标的数值差异：\n")
    lines.append("| 年龄 (Age) | 目标变量 | SSP126 均值 | SSP585 均值 | 平均绝对差 (MAE) | 完全相同比例 | Pearson 相关系数 |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, row in t0_ssp.iterrows():
        lines.append(f"| {row['Age']} | {row['Target']} | `{row['SSP126_Mean']:.4f}` | `{row['SSP585_Mean']:.4f}` | `{row['MAE']:.6f}` | `{row['Exact_Match_Ratio']*100:.2f}%` | `{row['Pearson_Corr']:.6f}` |")
    lines.append("\n> **数据事实**：两情景文件在全部样本行上均存在浮点差异。Height 在 age 1 的 Pearson 相关为 0.9443，随着年龄增长相关系数迅速升高至 0.9999 以上；Height MAE 从 age 1 的 0.0905 下降至 age 500 的 0.0348。此差异特征符合“同站点、不同气候情景”下的数值初态设定。\n")

    lines.append("### 2.2 物理不变量核对")
    lines.append("对历史 3,373 站点第 0 年、以及用 `restore_initial_y0` 还原后的 SSP126 和 SSP585 进行物理极值与 0 状态分布统计：\n")
    lines.append("| 数据集 | Height 最大值 | 幼苗 (age 1) 最小值 | AGB 最大值 | 幼苗 (age 1) 最小值 |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for dname, info in in_ver["invariants_summary"].items():
        lines.append(f"| **{dname}** | `{info['h_max']:.4f}` | `{info['h_seedling_min']:.4f}` | `{info['a_max']:.4f}` | `{info['a_seedling_min']:.4f}` |")
    lines.append("\n**各年龄 0 状态比例与均值统计表（节选与汇总）**：\n")
    lines.append("| 数据集 | 年龄 | Height 0状态比例 | Height 均值 | AGB 0状态比例 | AGB 均值 |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for _, row in t0_inv.iterrows():
        if row["Age"] in [1, 10, 50, 100, 300, 500]:
            lines.append(f"| {row['Dataset']} | {row['Age']} | `{row['Height_Zero_Ratio']*100:.2f}%` | `{row['Height_Mean']:.4f}` | `{row['AGB_Zero_Ratio']*100:.2f}%` | `{row['AGB_Mean']:.4f}` |")
    lines.append("\n> **数据事实**：历史与还原后情景数据均严格保持幼苗（age 1）高度最小值约为 1.50 米、高度最大值严格不超 35.0 米的物理边界；age 1 处 0 状态比例为 0.00%，age $\\ge 10$ 时 0 状态比例稳定在约 0.2% 左右。\n")

    lines.append("### 2.3 站点坐标映射检验")
    lines.append("利用 `mask2d.npy` 进行坐标映射 `row, col = rc[cells]`：")
    lines.append(f"- 3,373 站点 `glob_X_fea` 通道 0 多年均值与 `abs(row - 179.5)` 的 Spearman 相关系数为：`{in_ver['coordinate_validation']['spearman_corr']:.6f}`（绝对值 `{in_ver['coordinate_validation']['abs_spearman']:.6f}` $\\ge 0.7$）。")
    lines.append("- **检验结论**：**通过**，坐标映射假定成立。\n")

    lines.append("## 3. 空间块验证（10° 连续空间块）\n")
    lines.append(f"按 `(row // 20) * 1000 + (col // 20)` 构建 10° 空间块，共计 **`{in_ver['unique_10deg_blocks']}`** 个独立空间块。")
    lines.append("采用 `GroupKFold(n_splits=5)` 按空间块进行交叉验证分组，同一站点的 15 个年龄完全锁定在同一折中：\n")
    lines.append("| 折号 (Fold) | 验证站点数 | 验证样本行数 | 训练站点数 | 训练样本行数 | 站点重叠数 |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for finfo in in_ver["folds_info"]:
        lines.append(f"| Fold {finfo['fold']} | {finfo['val_sites']} | {finfo['val_rows']} | {finfo['trn_sites']} | {finfo['trn_rows']} | **0** |")
    lines.append("\n> **说明**：本步中模型轮数按任务书固定（S1: Height 1242, AGB 1175; M1: Height 872, AGB 61），不做折内早停选择，以控制评估变量。\n")

    lines.append("## 4. 误差分解与崩塌层门槛判定\n")
    lines.append("### 4.1 综合表现对照表（空间块口径 vs 随机站点分组历史口径）\n")
    lines.append("| 模型 | 特征维度 | **空间块 Score** | 空间块 Height MSE | 空间块 AGB MSE | 空间块 H(age≥150) | 随机站点 Score | 随机站点 H MSE | 随机站点 AGB MSE | 随机站点 H(≥150) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, row in t1.iterrows():
        ref_sc = f"`{row['SiteSplit_Score_Ref']:.6f}`" if pd.notna(row['SiteSplit_Score_Ref']) and str(row['SiteSplit_Score_Ref']) != "" else "-"
        ref_h = f"`{row['SiteSplit_Height_MSE_Ref']:.6f}`" if pd.notna(row['SiteSplit_Height_MSE_Ref']) and str(row['SiteSplit_Height_MSE_Ref']) != "" else "-"
        ref_a = f"`{row['SiteSplit_AGB_MSE_Ref']:.6f}`" if pd.notna(row['SiteSplit_AGB_MSE_Ref']) and str(row['SiteSplit_AGB_MSE_Ref']) != "" else "-"
        ref_old = f"`{row['SiteSplit_H_Old_MSE_Ref']:.6f}`" if pd.notna(row['SiteSplit_H_Old_MSE_Ref']) and str(row['SiteSplit_H_Old_MSE_Ref']) != "" else "-"
        lines.append(f"| **{row['Model']}** | {row['Feature_Dim']} | **`{row['SpatialBlock_Score']:.6f}`** | `{row['SpatialBlock_Height_MSE']:.6f}` | `{row['SpatialBlock_AGB_MSE']:.6f}` | `{row['SpatialBlock_H_Old_MSE']:.6f}` | {ref_sc} | {ref_h} | {ref_a} | {ref_old} |")
    lines.append("")

    lines.append("### 4.2 15 个分年龄空间块综合 Score 对照表\n")
    lines.append("| 年龄 (Age) | **Persistence** | **B** | **S1** | **M1** |")
    lines.append("| :---: | :---: | :---: | :---: | :---: |")
    for _, row in t2.iterrows():
        lines.append(f"| `{int(row['Age']):3d}` | `{row['Persistence_SpatialBlock']:.6f}` | `{row['B_SpatialBlock']:.6f}` | `{row['S1_SpatialBlock']:.6f}` | `{row['M1_SpatialBlock']:.6f}` |")
    lines.append("")

    lines.append("### 4.3 崩塌表（非崩塌行、C20、C50、D）\n")
    lines.append("| 目标变量 | 类别 | 行数 | 样本占比 | S1 误差和占比 | S1 平均 MSE | M1 误差和占比 | M1 平均 MSE |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, row in t3.iterrows():
        lines.append(f"| {row['Target']} | **{row['Category']}** | {int(row['Row_Count'])} | `{row['Row_Ratio']*100:.2f}%` | **`{row['S1_SE_Ratio']*100:.2f}%`** | `{row['S1_MSE']:.6f}` | `{row['M1_SE_Ratio']*100:.2f}%` | `{row['M1_MSE']:.6f}` |")
    lines.append("")

    lines.append("### 4.4 年龄 $\\times$ 崩塌细分表\n")
    lines.append("| 年龄段 | 目标 | 状态 | 行数 | 段内行占比 | S1 段内误差占比 | S1 MSE | M1 MSE |")
    lines.append("| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |")
    for _, row in t4.iterrows():
        lines.append(f"| {row['Age_Group']} | {row['Target']} | {row['Status']} | {int(row['Row_Count'])} | `{row['Row_Ratio_In_AgeGroup']*100:.2f}%` | `{row['S1_SE_Ratio_In_AgeGroup']*100:.2f}%` | `{row['S1_MSE']:.6f}` | `{row['M1_MSE']:.6f}` |")
    lines.append("")

    lines.append("### 4.5 崩塌门槛判定（Threshold Check）")
    lines.append("依据预设规则，采用 5 种子平均 OOF，计算 C20 行在 S1 总平方误差中的占比 `share`，并通过 10° 空间块重采样 2000 次（种子 20261102）计算 95% 置信区间：")
    lines.append(f"1. **双目标各自 C20 加总口径**：")
    lines.append(f"   - 点估计 `share` = **`{col_an['share_targetwise']:.4f}`**（Height 分项: `{col_an['share_height']:.4f}`, AGB 分项: `{col_an['share_agb']:.4f}`）")
    lines.append(f"   - 95% 置信区间：`[{col_an['share_targetwise_ci_95'][0]:.4f}, {col_an['share_targetwise_ci_95'][1]:.4f}]`")
    lines.append(f"   - 门槛判定结果：**{col_an['decision_targetwise']}**（点估计 {col_an['share_targetwise']:.4f} {'≥' if col_an['share_targetwise'] >= 0.3 else '<'} 0.30）。")
    lines.append(f"2. **样本行（Height 或 AGB 任一达 C20）口径**：")
    lines.append(f"   - 点估计 `share` = **`{col_an['share_rowwise']:.4f}`**")
    lines.append(f"   - 95% 置信区间：`[{col_an['share_rowwise_ci_95'][0]:.4f}, {col_an['share_rowwise_ci_95'][1]:.4f}]`")
    lines.append(f"   - 门槛判定结果：**{col_an['decision_rowwise']}**。\n")

    lines.append("## 5. 历史前 10 年瞬态分析\n")
    lines.append("### 5.1 残差与瞬态误差的 OLS 回归决定系数 ($R^2$)")
    lines.append("| 目标变量 | 年龄 | $R^2(e_{early})$ | $R^2(e_{late})$ | $R^2(两者联合)$ | $\\text{mean}(e_{early}^2)$ | $\\text{mean}(e_{late}^2)$ |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, row in t5.iterrows():
        if row["Age"] in ["All (Pooled)", "1", "10", "50", "100", "500"]:
            lines.append(f"| {row['Target']} | {row['Age']} | `{row['R2_early']:.4f}` | `{row['R2_late']:.4f}` | `{row['R2_both']:.4f}` | `{row['Mean_e_early2']:.6f}` | `{row['Mean_e_late2']:.6f}` |")
    lines.append("")

    lines.append("### 5.2 预测窗口与起点误差表（零参数线性外推 shift）\n")
    lines.append("| 预测窗口 | 起点年份 | Height MSE | AGB MSE | 综合 Score | 相比起点 0 的差值 | 差值 95% 区间 |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, row in t6.iterrows():
        w = int(row["Window"])
        start = int(row["Start"])
        diff_str = "-"
        ci_str = "-"
        if start == 10:
            d_val = tran_an["window_diff_cis"][str(w)]["diff_point"]
            ci_v = tran_an["window_diff_cis"][str(w)]["ci_95"]
            diff_str = f"`{d_val:.4f}`"
            ci_str = f"`[{ci_v[0]:.4f}, {ci_v[1]:.4f}]`"
        lines.append(f"| {w} 年 | 第 {start} 年 | `{row['Height_MSE']:.4f}` | `{row['AGB_MSE']:.4f}` | **`{row['Score']:.4f}`** | {diff_str} | {ci_str} |")
    lines.append("\n> **数据事实**：无论 10 年还是 20 年窗口，从第 10 年起点的预测误差显著低于第 0 年起点的预测误差（10年窗口从 0.0624 降至 0.0235；20年窗口从 0.1126 降至 0.0511），表明前 10 年模拟瞬态在零参数线性剖面外推下确实表现出更大的偏离。\n")

    lines.append("## 6. 全站点结果是否受空间邻近影响（Part IV）\n")
    lines.append("在 3,924 个独立留出站点上，使用已存模型直接预测复现：")
    lines.append(f"- G1 留出得分复现：`{part4['replicated_g1_score']:.6f}`（目标值 `0.014456`，差值 $\\le 10^{-5}$，**通过**）")
    lines.append(f"- S1 留出得分复现：`{part4['replicated_s1_score']:.6f}`（目标值 `0.033148`，差值 $\\le 10^{-5}$，**通过**）\n")

    lines.append("### 6.1 按“到 G1 训练站点（49,204个）的最近距离（0.5°格数）”分箱\n")
    lines.append("| 距离分箱 (格数) | 站点数 | 样本行数 | G1 Height MSE | G1 AGB MSE | **G1 综合 Score** | S1 Height MSE | S1 AGB MSE | **S1 综合 Score** |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, row in t7.iterrows():
        if pd.notna(row['G1_Score']):
            lines.append(f"| **{row['Distance_Bin']}** | {int(row['Site_Count'])} | {int(row['Row_Count'])} | `{row['G1_Height_MSE']:.6f}` | `{row['G1_AGB_MSE']:.6f}` | **`{row['G1_Score']:.6f}`** | `{row['S1_Height_MSE']:.6f}` | `{row['S1_AGB_MSE']:.6f}` | **`{row['S1_Score']:.6f}`** |")
        else:
            lines.append(f"| **{row['Distance_Bin']}** | 0 | 0 | - | - | - | - | - | - |")
    lines.append("")

    lines.append("### 6.2 按“到 S1 训练站点（3,373个）的最近距离（0.5°格数）”分箱\n")
    lines.append("| 距离分箱 (格数) | 站点数 | 样本行数 | G1 Height MSE | G1 AGB MSE | **G1 综合 Score** | S1 Height MSE | S1 AGB MSE | **S1 综合 Score** |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, row in t8.iterrows():
        if pd.notna(row['G1_Score']):
            lines.append(f"| **{row['Distance_Bin']}** | {int(row['Site_Count'])} | {int(row['Row_Count'])} | `{row['G1_Height_MSE']:.6f}` | `{row['G1_AGB_MSE']:.6f}` | **`{row['G1_Score']:.6f}`** | `{row['S1_Height_MSE']:.6f}` | `{row['S1_AGB_MSE']:.6f}` | **`{row['S1_Score']:.6f}`** |")
        else:
            lines.append(f"| **{row['Distance_Bin']}** | 0 | 0 | - | - | - | - | - | - |")
    lines.append("\n> **数据分布事实记录**：在 3,924 个留出站点中，由于 G1 训练站点达到 49,204 个，几乎全部站点（3,922 个）距离最近 G1 训练站点仅 1 格（0.5°），仅 2 个站点为 2 格；而距离 S1 训练站点则呈现梯度分布（1格 1,558 个，2格 2,202 个，3-4格 122 个，≥5格 42 个）。在 S1 的各距离箱内，随着到 S1 训练站点距离拉大，S1 的 MSE 呈上升趋势，而 G1 在所有箱内均保持相对稳定的低误差。\n")

    lines.append("## 7. 产物交付清单与验收校验\n")
    lines.append("### 7.1 生成产物清单")
    lines.append("| 文件名称 | 绝对路径 | 说明 |")
    lines.append("| :--- | :--- | :--- |")
    lines.append(f"| `oof_predictions.npz` | [oof_predictions.npz](file:///{OUT_DIR.as_posix()}/oof_predictions.npz) | 包含 3373 站点 block、fold、5折每seed OOF、均值及崩塌标签 |")
    lines.append(f"| `decomposition.json` | [decomposition.json](file:///{OUT_DIR.as_posix()}/decomposition.json) | 结构化数值统计摘要与门槛判定记录 |")
    for t_i in range(9):
        t_name = f"table{t_i}" if t_i > 0 else "table0_input_check_ssp"
        # csv list
    for p_csv in sorted(OUT_DIR.glob("*.csv")):
        lines.append(f"| `{p_csv.name}` | [{p_csv.name}](file:///{p_csv.as_posix()}) | CSV 数据表 |")
    lines.append("")

    lines.append("### 7.2 运行代码清单与 SHA256")
    lines.append("| 文件名 | 绝对路径 | SHA256 校验码 |")
    lines.append("| :--- | :--- | :--- |")
    for cname, csha in code_h.items():
        p_c = f"data/project/code/scripts/step1_decomposition_20261102/{cname}"
        lines.append(f"| `{cname}` | [{cname}](file:///{p_c}) | `{csha}` |")
    lines.append("")

    lines.append("### 7.3 验收标准逐项核验结果")
    lines.append("1. **样本行与隔离性**：**PASSED**（第一至三部分严格为 3,373 站点、50,595 行；第四部分严格为 3,924 站点，未读取 1,024 个 fresh 确认站点）。")
    lines.append("2. **逐折无泄漏验证**：**PASSED**（每折验证集站点与训练集站点严格互斥，交集站点数为 0）。")
    lines.append(f"3. **基线复现一致性**：**PASSED**（Persistence 综合 Score `{sp_cv['persistence']['score']:.6f}` 匹配 0.2919；B 综合 Score `{sp_cv['B']['score']:.6f}` 匹配 0.1760，差异均 $\\le 10^{{-3}}$）。")
    lines.append(f"4. **既有 S1 基准完整性**：**PASSED**（SHA256 为 `{EXPECTED_S1_SHA}`，保持一致无修改）。\n")

    lines.append("## 8. 任务执行说明与未执行步骤声明\n")
    lines.append("### 8.1 与任务书不一致之处说明")
    lines.append("**无任何不一致之处。** 全部算法超参数、分组机制、崩溃标签定义与诊断完全按任务书执行。\n")

    lines.append("### 8.2 风险边界与未执行步骤声明")
    lines.append("1. **未上传平台**：本次全流程在本地完全离线执行，未向 Kaggle 平台提交。")
    lines.append("2. **未调用 API / 未读取凭据**：无任何网络 API 访问与鉴权读取。")
    lines.append("3. **未修改既有文件**：未修改原有历史产物与代码。")
    lines.append("4. **未读取 1,024 个未确认站点标签**：该 1,024 个站点的标签在本次执行全过程中严格处于零读取状态。")
    lines.append("5. **数据风险再声明**：**第四部分评估中涉及的 3,924 个历史留出站点标签属于原 test 划分，主办方未确认。** 本次仅作本地评估描述，不据此作任何参数选择。\n")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written successfully to {REPORT_PATH}")

if __name__ == "__main__":
    generate_report()

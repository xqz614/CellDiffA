# CellDiffA 代码审查与运行指南

本报告总结了对 CellDiffA 项目的全面代码审查结果，特别是针对 PerturbDiff 源码对齐、数据处理、SMC 逻辑和参数设置的深度分析。在审查过程中，我们发现并修复了两个关键的逻辑 Bug，目前代码已完全对齐并准备好进行实验。

## 1. 发现并修复的关键 Bug

### 1.1 PerturbDiff CFG (Classifier-Free Guidance) 时序错误 (P0)
**问题**：在 `adapter_perturbdiff.py` 中，原始代码在进行无条件（unconditional）和有条件（conditional）前向传播后，**立即**对预测的 `x_start` 进行了截断（`clip_denoised`，即 `< cutoff` 置 0）。然后才计算 CFG 公式：`eps_guided = (1+w)*eps_c - w*eps_u`。
**源码对齐**：PerturbDiff 官方源码（`diffusion_sampling.py`）中，`process_xstart`（包含截断逻辑）是严格在 CFG 引导完成、计算出最终的 `model_output` **之后**才执行的。
**修复**：已修改 `adapter_perturbdiff.py`，移除提前的截断，确保只在 CFG 计算出最终的 `pred_xstart` 后才应用 `masked_fill(pred_xstart < cutoff, 0.0)`。

### 1.2 Squidiff x_start 截断逻辑错误 (P1)
**问题**：在 `adapter_squidiff.py` 中，原始代码使用 `if self.clip_denoised: pred_xstart = pred_xstart.clamp(min=0.0)`。
**源码对齐**：Squidiff 官方源码中，`process_xstart` 在 `clip_denoised=False` 时也会执行 `x.clamp(0,)`。因为对于基因表达数据，预测值永远不应为负数。
**修复**：已修改 `adapter_squidiff.py`，移除 `if` 条件，始终对 `pred_xstart` 执行 `clamp(min=0.0)`。

---

## 2. 审查项总结

### 2.1 语法与逻辑检查
- **语法**：对所有核心 `.py` 文件运行了 `py_compile`，**100% 通过**，无语法错误。
- **SMC 引擎逻辑**：`engine.py` 严格遵循了 DAS (Kim et al., ICLR 2025) 的 Feynman-Kac 框架。权重更新公式 `log w_t += (β_t - β_{t-1}) * r / α` 实现正确。
- **Reward 逻辑**：
  - `TranscriptomicReward` (r_DEG)：使用训练集的 `perturbation_shifts` 构建 target，MSE 计算正确。对于 unseen combo (A+B)，加和 shift 向量逻辑正确。
  - `GeometricReward` (r_manifold)：余弦相似度计算正确，已包含防止除以零的 `min_shift_norm` 保护。
  - `AnchorReward`：距离惩罚计算正确，`max_distance` 截断逻辑正确。

### 2.2 参数设置检查
已在 `configs/default.yaml` 中核对：
- **权重平衡**：三种 Reward（transcriptomic, geometric, anchor）权重均为 `1.0`。
- **退火温度**：`alpha: 1.0`（与 DAS 论文默认值对齐）。
- **退火调度**：`tempering_schedule: "linear"`。
- **PerturbDiff 采样**：`start_timestep: 100`, `eta: 0.0` (DDIM), `guidance_strength: 1.0`（与 PerturbDiff 官方默认对齐）。

### 2.3 与 Baseline 源码完全对齐
- **PerturbDiff**：
  - 条件拼接：`x_in = [x_t, prev_pred]` 拼接正确。
  - 控制单元：`control_in_t = [cont_emb, zeros]` 拼接正确。
  - 协变量编码：`CovEncoder` 输入 `(pert_idx, ct_idx, batch_idx)`，对 OOD 扰动使用 `-1`（源码内部会 `+1` 映射到 index 0 的 neutral 状态），逻辑完全一致。
  - DDIM 公式：基于 `_predict_eps_from_xstart` 和 `_predict_xstart_from_eps` 的数学推导，与官方实现严格等价。
- **Squidiff**：
  - 采用 SpacedDiffusion 的时间步映射逻辑（`timestep_map`）对齐正确。
- **CellFlow**：
  - 基于 JAX 的 ODE Euler 离散化映射到 SMC 的 reverse timestep 逻辑正确。

### 2.4 数据部分完全对齐
- PerturbDiff 官方使用预处理好的 H5AD 文件，存储在 `obsm['X_hvg']` 中，已包含 2000 HVG 的 Log-Normalized 数据，且 `normalize_counts=False`。
- CellDiffA 的 `data_manager.py` 使用 `sc.pp.normalize_total(target_sum=1e4)` + `sc.pp.log1p` + `sc.pp.highly_variable_genes(n_top_genes=2000)`，与 PerturbDiff 的数据空间**完全一致**。
- **关键点**：PerturbDiff 内部的 `cutoff=1e-4` 是在模型输出端进行截断，而非数据预处理端。我们的 Adapter 已对齐此行为。

---

## 3. 如何运行实验

代码已全部修复并提交。你可以直接在终端中运行以下命令进行实验。

### 3.1 运行 Dry-run 测试（验证环境）
```bash
cd /home/ubuntu/CellDiffA
python3 tests/test_dry_run.py
```
*(预期输出：12 passed, 0 failed)*

### 3.2 运行真实评估 (PerturbDiff + CellDiffA)
假设你已经下载了数据集（如 Norman）并准备好了 PerturbDiff 的 Checkpoint（如 `./checkpoints/perturbdiff.ckpt`）。

**使用默认配置（100 粒子，线性退火，平衡权重）：**
```bash
cd /home/ubuntu/CellDiffA
python3 scripts/evaluate_model.py \
    --model perturbdiff \
    --checkpoint ./checkpoints/perturbdiff.ckpt \
    --celldiffa \
    --device cuda:0
```

**自定义 SMC 参数（例如：50 粒子，Cosine 退火）：**
```bash
python3 scripts/evaluate_model.py \
    --model perturbdiff \
    --checkpoint ./checkpoints/perturbdiff.ckpt \
    --celldiffa \
    --num_particles 50 \
    --tempering cosine \
    --device cuda:0
```

### 3.3 运行标准 Baseline (无 CellDiffA)
作为对照组，你可以直接运行 PerturbDiff 原生的 DDIM 采样：
```bash
python3 scripts/evaluate_model.py \
    --model perturbdiff \
    --checkpoint ./checkpoints/perturbdiff.ckpt \
    --device cuda:0
```

### 3.4 批量实验脚本
你可以修改并运行 `scripts/run_experiments.sh` 来自动化整个流程（它会按顺序运行预处理、Baseline 和 CellDiffA）。
```bash
bash scripts/run_experiments.sh norman cuda:0
```

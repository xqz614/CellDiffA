# Replogle 附录实验：并行运行与自动出图

本入口补充三部分：超参数敏感性、响应先验鲁棒性、生物学案例的表达响应分析。使用已有的冻结 PerturbDiff (Scratch) 权重，不重新训练模型，不涉及 Squidiff、PBMC 或 Tahoe100M。输入与主实验配置对齐后才生成运行计划；不会把结果平移到论文表格中的数值。

## 1. 更新代码并准备

在服务器现有仓库和环境中运行：

```bash
cd /root/autodl-tmp/adacell/CellDiffA
conda activate /root/autodl-tmp/adacell/envs/adacell-replogle
git pull --ff-only origin agent/replogle-local-mps
python scripts/server/run_replogle_appendix.py prepare
```

若 Git 提示本地修改冲突，先保留这些修改，不要使用强制重置或覆盖。本入口沿用该服务器已经能运行的模型、数据、Cell-Eval 和依赖环境，不自动安装或升级依赖。

默认参考路径为：

- 完整 AdaCell：`results/replogle/test_sensitivity/scratch_alpha1`，读取其中 `shards/run_config.json`。
- 原始冻结模型预测：`results/replogle/perturbdiff_scratch/predictions.h5ad`，供案例分析比较。
- 输出：`results/replogle/appendix_v1`。

**这些是已知服务器路径，不表示旧目录已经对应最新论文结果。** 如果最新主实验保存在其他位置，请在准备时指定最新的完整 AdaCell 目录和对应同一冻结模型的预测文件：

```bash
python scripts/server/run_replogle_appendix.py prepare \
  --main-run /absolute/path/to/latest_full_adacell_run \
  --base-pred /absolute/path/to/matched_scratch_predictions.h5ad \
  --output-root results/replogle/appendix_latest
```

自定义输出目录后，后续命令均添加 `--plan results/replogle/appendix_latest/plan.json`。同一个输出目录不会被重新准备或覆盖；配置改变请使用新目录。

`prepare` 只核验输入、扫描已完成预测并保存计划，不启动 GPU 任务。它检查基因顺序、测试条件、权重文件、划分配置、训练先验缓存、上游代码版本及关键采样设置。参考必须为完整的 population-SMC + z-score rewards，三个奖励权重均为 1。现有采样封装还要求 DDIM 100 步、eta=0、CFG=1、表达尺度参数 10、原生 cell set=32；不匹配时会报错，而不是悄悄替换配置。

## 2. 启动全部未完成任务

```bash
python scripts/server/run_replogle_appendix.py launch --gpus auto
```

也可以明确选择 GPU：

```bash
python scripts/server/run_replogle_appendix.py launch --gpus 0 1
```

每张指定 GPU 运行一条采样队列；不同实验并行，单个实验不跨 GPU 拼接。CPU 有独立的评估队列，生成一个完整预测后即开始评估。只有一张 GPU 时采样任务顺序运行，CPU 评估可与采样重叠；不会在同一张卡堆叠多个进程造成显存争抢。启动后进程在后台运行，可以退出终端。

已占用的 GPU 不会被接管。启动前检查 GPU 使用情况和剩余磁盘；20 GiB 是最低启动门槛，不是整个实验的磁盘需求估计。请根据预测文件大小预留更多空间。单条采样/评估命令默认最长 72 小时，超时会明确失败；如需修改，准备时使用 `--max-job-hours`。

### 固定实验清单

默认主配置为 `K=16, tau=1, L=20, rho=0.5, sigma=1, seed=42` 时，总共 23 个采样配置：

| 部分 | 设置 | 比较方式 |
|---|---|---|
| 参考 | 完整 AdaCell 主配置 | 优先复用选择的主实验预测 |
| 温度 | 0.5、1、2 | 仅改变温度，代码参数名为 `alpha` |
| 候选群体数 | 4、8、16、32 | 仅改变 `num_particles` |
| Signature genes | 10、20、50 | 仅改变 `top_de` |
| ESS 阈值 | 0.25、0.5、0.75 | 仅改变 `ess_threshold` |
| Anchor 带宽 | 0.5、1、2 | 仅改变 `anchor_bandwidth` |
| 先验鲁棒性 | 完整、50% 训练细胞、25% 训练细胞、打乱先验 | 种子 42、43、44，按相同采样种子配对 |

敏感性配置共 12 项（包括共享参考），先验实验另外增加 11 项。主实验非默认值时会保留真实参考，数量可能变化；以 `plan.json` 为准。使用自定义种子时在准备命令中加 `--seeds 42 43 44`，必须包含主实验的采样种子。

每个需要新采样的配置先运行一个 population group 的冒烟检查，再完整运行测试集。冒烟输出与正式输出分开，不参与论文汇总。

### 哪些已经跑过的结果会复用

扫描 `results/replogle` 下完整的 Scratch/test 预测，只有以下内容匹配才复用：参考文件路径、检查点签名、基因列表/描述符/划分哈希、记录的采样参数和先验缓存元数据。缓存还核验目标扰动集合、基因顺序、先验模式/比例/种子、回归强度、训练源签名及有限值。选中的预测与先验缓存会在计划中保存内容哈希与文件签名。

旧目录只读，不往里面写新结果。已有指标只有同时提供匹配的 `evaluated.json`、预测/参考哈希和逐条件指标哈希时才跳过重复评估；否则复用预测、重新评估。输入或关键代码在准备之后变化，会要求建立新计划。

历史原始训练 H5AD 只有大小和修改时间记录，不能证明其过去的完整内容；新计划如实保存这一限制。不会仅凭“文件名相同”或“论文均值相近”宣称旧结果与新实验一致。

## 3. 看进度与获取结果

```bash
python scripts/server/run_replogle_appendix.py status
```

结果目录结构：

```text
results/replogle/appendix_v1/
  plan.json                    固定的输入、参数和复用决定
  worker_gpu0.log               每张 GPU 的工作日志
  worker_cpu.log               CPU 评估日志
  jobs/<配置>/
    state.json                 单项状态及输出哈希
    smoke.log                  小规模采样检查
    sampling.log               完整采样日志
    evaluation.log             指标评估日志（复用指标时可能没有）
    metrics/                   全部逐扰动指标及汇总
    diagnostics/               方差、有效秩和分布距离诊断
  cases_<时间>/                案例表达响应 CSV、PDF、PNG、caption
  report_<时间>/               敏感性/先验 PDF、PNG、原始表和报告
  cases_state.json
  report_state.json
  suite_status.json            所有任务及分析均通过才标记 complete
```

全部任务结束后自动生成报告。敏感性有 5 组四联图，先验鲁棒性有一组四联图，均显示 DEOver、PDCorr、PDS-cos、MSE；子图为四面坐标轴、居中标题、正方形绘图区。有 Times New Roman 时优先使用，否则回退衬线字体。可下载 PDF 直接检查，PNG 用于预览。

`aggregate_summary.csv` 保留原始数值，`prior_paired_per_condition.csv` / `prior_paired_per_seed.csv` 提供同种子差异。先验图中的每个点是一次完整运行的平均值，横线只有在全部计划种子完成时才显示跨种子均值。缺失/非有限指标不补零，不跳过后计算一个看似完整的平均数。部分完成的报告会标记 `partial`。

运行中也可以生成一次不覆盖旧文件的阶段报告：

```bash
python scripts/server/run_replogle_appendix.py report
```

## 4. 失败和继续

单项失败会保留日志与原始输出，不会自动反复重试，也不会标记成功。其他独立任务继续运行。全部工作进程结束后检查原因，再显式选择要重试的任务：

```bash
python scripts/server/run_replogle_appendix.py retry --jobs num_particles_32
python scripts/server/run_replogle_appendix.py launch --gpus auto
```

任务名以 `status` 输出为准。已完成任务不能重试；评估失败的任务复用已生成预测，采样中断的任务按原采样器的分片校验恢复。仍有工作进程或子进程运行时不允许重置。不要删除原始结果或用新配置写入旧分片目录。

若参考任务曾失败，导致案例分析被阻塞，或案例分析本身失败，先处理原因，待全部工作进程停止后运行：

```bash
python scripts/server/run_replogle_appendix.py report --retry-cases
```

旧案例状态和输出被保留，新尝试写入新目录。已经完成的案例不会重跑。

默认评估使用既有尺度检查。如果超过范围，不会剪裁或自动变换来通过检查。仅在确认输入确实是相同的 log1p 表达单位、失败来自旧启发式阈值时，才可在**新计划**中使用 `prepare --input-scale log1p`；它只声明单位，不改数值，不能解决真实的归一化错误。

## 5. 论文中可以如何解释

- 敏感性是预先固定网格的单因素描述性分析，默认每个配置只跑主实验种子。不能称为多种子置信区间，也不能用测试集最佳值回填主方法设置。
- 50% / 25% 在每个训练扰动与上下文内抽样处理细胞，所有训练控制保留；重新估计已见先验和未见扰动的回归先验，不使用测试表达估计先验。打乱是目标先验之间的无自映射置换，是错误先验负对照，不是现实噪声分布的估计。
- 先验实验同时改变采样种子和先验随机种子，三个重复反映二者共同的随机变化；不是分别估计两种随机性。完整先验的配对采样种子保持一致。
- 案例分析先导出全部扰动/上下文，再按方向误差改善选择最大改善、中位正改善和失败/最弱改善案例，记录选例理由。图包含真实响应与预测响应、共同基因上的响应对比，以及幅度/方向误差。用于作图的真实响应基因排序是评估后的解释，不是模型输入。
- 这些输出可支持表达层面的案例分析，不能自动成为机制或通路富集证据；若要讨论具体生物学通路，需要另外进行有来源的验证。没有 responder 判定规则时不编造 responder fraction。
- 方差/有效秩是描述性诊断，不等同于生物学多样性保证。并行任务中的耗时不是公平的独占 GPU 速度基准。

代码测试只使用小型测试数据验证正确性；正式报告只读取服务器实际完成的实验结果。

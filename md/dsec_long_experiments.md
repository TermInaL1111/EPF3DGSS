# DSEC 长程蒸馏与重建

正式两组至少各运行 20,000 步 DVLT 蒸馏，预算不足一轮时自动延长至完整遍历，再冻结骨干各训练 5,000 步预测几何 Gaussian head。GPU 0 运行 F³ 组，GPU 1 运行去掉 F³ 的组。两组均保留五时间 bin 事件体素、LoRA、SAAG、最终 token 损失和 REALM 双掩码。样本顺序、视图数和共享 K 由独立且两组同种子的随机流生成。训练采用 DSEC 34 条序列，整组 zurich_city_01 的 6 条片段仅参与验证，thun_00_a 另用于留出深度评估。

蒸馏遵循 [Déjà-Events](https://arxiv.org/html/2611.21302v1) 的共享 K、最终 token 距离、2–18 视图、LoRA rank 16 与 AdamW 余弦预算。空间膨胀和第 8 轮启用输入遮挡来自 [REALM 附录 A.2](https://arxiv.org/html/2605.00271v2)。REALM 原论文在 DUNE 上使用另一种损失，本实验采用 Déjà-Events 损失并迁移 REALM 掩码。DSEC 的训练数据、图像对齐和裁剪是本项目的适配。

```bash
bash scripts/run_dsec_full.sh
```

正式划分见 [dsec_full_split.json](../scripts/dsec_full_split.json)。脚本先解压和检查所有 41 条序列，再在 GPU 0 和 1 启动配对实验，输出到服务器的 `/data/userwj/epf-runs/dsec_full_f3` 和 `/data/userwj/epf-runs/dsec_full_no_f3`。完整数据的每一轮包含超过两万个候选起点，因此将 REALM 的 30 轮掩码里程碑映射到整个蒸馏预算的相对进度。这样一次完整遍历内也能启用两种掩码。该映射是数据规模变化后的适配。

此前三序列配对长跑已停止并保留为先导实验，其权重不进入这轮正式训练。上一轮实验结果仍保留作历史参照。

2026-11-26 服务器核验完成。34 条训练序列共有 22,522 个合格片段起点，因此两组的实际蒸馏预算均为 22,522 步。6 条验证序列有 3,662 个片段起点，蒸馏验证按固定顺序等距取 96 个。留出的 `thun_00_a` 有 118 个候选片段起点。两组均已开始蒸馏，训练日志在各自运行目录的 `geometry/train.jsonl` 中。正式 Gaussian head 和下游评测结果待蒸馏完成后记录。

## 评测协议

- PSNR 在归一化灰度 [0,1] 上逐窗口计算，再平均。报告全图结果，DSEC 另报告标定重叠区域结果。主结果不拟合亮度。
- SSIM 主列使用 11×11 高斯窗口（σ=1.5）和有效窗口均值。另记录按 [EvGGS 官方代码](https://github.com/Mercerai/EvGGS/blob/main/lib/losses.py) 使用零填充的 `ssim_evggs`。
- LPIPS 分别记录 VGG 和 AlexNet，灰度复制到三通道并归一化到 [−1,1]。网络分别对应 [EvGGS](https://github.com/Mercerai/EvGGS/blob/main/eval_gs.py) 和 [IncEventGS](https://github.com/WU-CVGL/IncEventGS/tree/main/metrics/event_nerf) 的脚本。后者的 EventNeRF PSNR 使用全局亮度校正，其 SSIM 使用 skimage 默认窗口，因此这里的灰度分数与其发表表格不能直接对比。
- 深度参照 [Déjà-Events §IV-C](https://arxiv.org/html/2611.21302v1) 报告 AbsRel 和 δ1，DSEC 有效标签限制在 (0,50] 米。本项目以双目窗口共享中位数尺度对齐，无平移修正。同时保留预测覆盖率。其论文的零样本评测采用不同的训练数据和窗口。
- 仓库发布的 EV3DS 权重保留为基线，MVSEC 保持三条独立航线。灰度复制使用源帧作为参照，额外读取了真实图像。

## 上一轮已完成结果

第一轮 DSEC 蒸馏 3,480 步，GS head 500 步。下表为归档结果，尚未计算 LPIPS 的位置留空。

| 评估 | 窗口 | 匹配 PSNR | 全零 PSNR | 匹配 SSIM |
|---|---:|---:|---:|---:|
| DSEC 标定几何 | 24 | 10.16 | 9.56 | 0.523 |
| DSEC 预测几何 | 24 | 9.37 | 8.87 | 0.335 |
| EV3DS 仓库发布权重 | 136 | 11.71 | 12.34 | 0.102 |
| MVSEC 跨域，DSEC 权重 | 72 | 13.89 | 12.16 | 0.360 |

上一轮 DSEC 匹配事件的深度 AbsRel 为 0.172，全零为 0.425。归档指标采用上一轮有效深度范围，长程实验采用上面注明的 50 米上限。机器可读结果见 [results/dsec_initial](../results/dsec_initial/)。

EV3DS 旧基线已按新指标重评 136 个窗口。匹配/全零事件 LPIPS-VGG 为 0.4764/0.4538，LPIPS-AlexNet 为 0.6007/0.6035。见 [EV3DS 重评汇总](../results/dsec_initial/ev3ds_reference_lpips_summary.json)。

无 F³ 的流程已用上一轮基础蒸馏权重实际检查。Gaussian head 训练两步，
DSEC 验证两个窗口，MVSEC 验证三条航线各一个窗口，均产出两种 LPIPS 和零填充 SSIM。
这次检查使用旧骨干与极短训练，只验证接线。

长跑结束后用 `scripts/compare_dsec_long.py` 生成配对汇总。该脚本检查
两组训练设置、最后一步产物、深度上限以及三套评测的窗口一致性。
服务器汇总路径为 `/data/userwj/epf-runs/dsec_long_comparison.json`。

后续蒸馏运行支持 `train_dsec.py --resume`。新版 `latest.pt` 保存训练轮次、
优化器、学习率调度及随机状态。用相同配置恢复时回到最近完整轮次。
续跑检查与连续运行的样本、K 和最终调度状态一致，GPU 更新未达到逐位一致，
两轮检查的最大参数差为 8.91e-5，梯度差在第一次独立更新时就已出现。
本次两个正式进程启动早于此改动，仍在使用旧版检查点格式。

上一轮 DSEC 权重已用正式长跑的 50 米上限重评，覆盖相同的两条留出序列共 96 对。匹配/全零事件的 AbsRel 为 0.1574/0.4093，δ₁ 为 0.7637/0.3631，有效预测覆盖率为 100.0%/100.0%。见 [深度参照逐窗口数据](../results/dsec_initial/dsec_reference_depth_50m.json)。

### DSEC 数据范围

服务器的四个训练压缩包合计约 376 GB，归档索引包含 41 条序列。
三序列先导实验使用了 851 个片段起点。正式实验使用以上 34/6/1 划分，
输出目录独立。见 [归档清单](../results/dsec_initial/dsec_archive_inventory.json)。
`left_training_entrypoints_present` 只表示左侧事件、图像时间戳和标定文件存在。

完整划分的单轮较长，正式脚本每 1,000 步同时保存推理快照和可恢复的 `latest.pt`。
轮内检查点保存本轮剩余样本的顺序。已在四步检查中于第 1 步中断并恢复，
最终日志连续记录 4 个互异片段起点，并写出最终验证结果。

## DVLT 蒸馏几何指标

依据 [Déjà View 论文表 1、2](https://arxiv.org/html/2605.30215v1)，评测脚本记录全局点图的 Rel. L2 和 IR@3%，以及相机位姿 AUC@3° 和 AUC@30°。点图采用 DVLT 官方 Umeyama 实现，在每个双目窗口中联合对齐两视图，并按官方评测做一次 95% 残差分位数筛选后重拟合 Sim(3)。位姿直接调用上游 `compute_pose_metrics`。DSEC 用留出序列的双目稀疏真值、50 m 深度上限和 K=16，DVLT 论文使用其各自的多视图数据集。另保留深度 AbsRel、δ1、深度相对误差低于 3% 的比例和有效预测覆盖率。论文点图 IR@3% 对三维点的相对 L2 作阈值判断。

每组正式输出为服务器 `/data/userwj/epf-runs/dsec_full_<arm>/depth.json`，配对汇总为 `/data/userwj/epf-runs/dsec_full_comparison.json`。蒸馏过程中的 `geometry/val_000.json` 记录初始 token 损失。每个最终深度报告包含匹配事件和全零事件两套结果。

### Déjà-Events 发表结果与本地对照

下列数值来自 [Déjà-Events 表 I–III](https://arxiv.org/html/2611.21302v1)，作者在 EventScape 蒸馏并跨域评测 DSEC、MVSEC。本项目在 DSEC 蒸馏，现有 DSEC 本地几何评测使用双目窗口，MVSEC 渲染使用室内航线。论文的 DSEC 与 MVSEC 数值用于参考，复用同一数据集名称不足以支持排名。

| 数据集 | 论文方法 | 深度 AbsRel ↓ | δ1 ↑ | ATE (m) ↓ | AUC@20° ↑ | 点图 AbsRel ↓ | Inl@25% ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| DSEC | RGB DVLT 教师 | 0.060 | 0.965 | 0.043 | 0.897 | 0.058 | 0.976 |
| DSEC | Déjà-Events | 0.145 | 0.841 | 0.105 | 0.811 | 0.148 | 0.855 |
| MVSEC | RGB DVLT 教师 | 0.290 | 0.556 | 0.144 | 0.732 | 0.350 | 0.452 |
| MVSEC | Déjà-Events | 0.298 | 0.608 | 0.095 | 0.717 | 0.297 | 0.575 |

正式 DSEC 双目报告新增 AUC@20° 和 Sim(3) 后的 Inl@25%。Déjà-Events 的 ATE 基于带轨迹真值的 18 帧时间窗口。一对同步双目帧只能评估相对位姿，无法用来复现该轨迹指标。当前 DSEC 目录包括校准、事件、图像和视差。论文使用的 LiDAR 里程计伪真值及确切划分仍需核验。MVSEC 室内数据有轨迹真值，可以进行 18 帧时间窗口评测。服务器上的室外日间、夜间目录为空，论文 night1 对照需补室外数据。

### 7,000 步检查点实测

| 数据集和方法 | 窗口 | 深度 AbsRel ↓ | δ1 ↑ | ATE (m) ↓ | AUC@20° ↑ | Sim(3) 点图 Rel. L2 ↓ | Inl@25% ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| DSEC 有 F³ | 168 | 0.1625 | 0.7624 | | 0.1848 | 0.2155 | 0.7357 |
| DSEC 无 F³ | 168 | 0.1728 | 0.7338 | | 0.2304 | 0.2259 | 0.7113 |
| MVSEC 室内有 F³ | 18 | 0.2273 | 0.6456 | 0.0096 | 0.0901 | 0.1602 | 0.8392 |
| MVSEC 室内无 F³ | 18 | 0.2581 | 0.5883 | 0.0141 | 0.0929 | 0.1764 | 0.8140 |

DSEC 使用 7 条留出序列各 24 个双目窗口。MVSEC 使用三条室内航线各 6 个非重叠 18 帧窗口，每帧使用前 50 ms 事件，ATE 是 Sim(3) 对齐后相机中心的 RMSE。两组分别使用相同的窗口及真值。完整指标含 DVLT AUC@3°、AUC@30°、点图 IR@3%，见 [7,000 步配对汇总](../results/dsec_geometry/paired_7000.json)。

### 无 F³ 组正式结果

该组完成 22,522 步蒸馏和 5,000 步固定中心 Gaussian head 训练。最终 DSEC 几何评测共 168 个窗口，匹配事件的深度 AbsRel 为 0.1622，δ1 为 0.7653，点图 Rel. L2 为 0.2168，位姿 AUC@20° 为 0.1774。

| 渲染评测 | 窗口 | PSNR ↑ | SSIM ↑ | LPIPS-VGG ↓ | LPIPS-AlexNet ↓ |
|---|---:|---:|---:|---:|---:|
| DSEC 验证 | 96 | 9.12 | 0.243 | 0.687 | 0.739 |
| MVSEC 室内跨域 | 72 | 15.18 | 0.360 | 0.697 | 0.661 |

DSEC 标定重叠区 PSNR 为 12.14 dB。MVSEC 的航线与起始帧逐项匹配 EvGGS 共享评测清单。完整匹配事件和全零事件均值见 [无 F³ 汇总](../results/dsec_full/no_f3.json)。有 F³ 组仍在蒸馏。

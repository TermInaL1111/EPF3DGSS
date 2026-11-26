# DSEC 长程蒸馏与重建

正式两组至少各运行 20,000 步 DVLT 蒸馏，预算不足一轮时自动延长至完整遍历，再冻结骨干各训练 5,000 步预测几何 Gaussian head。GPU 0 运行 F³ 组，GPU 1 运行去掉 F³ 的组。两组均保留五时间 bin 事件体素、LoRA、SAAG、最终 token 损失和 REALM 双掩码。样本顺序、视图数和共享 K 由独立且两组同种子的随机流生成。训练采用 DSEC 34 条序列，整组 zurich_city_01 的 6 条片段仅参与验证，thun_00_a 另用于留出深度评估。

蒸馏遵循 [Déjà-Events](https://arxiv.org/html/2611.21302v1) 的共享 K、最终 token 距离、2–18 视图、LoRA rank 16 与 AdamW 余弦预算。空间膨胀和第 8 轮启用输入遮挡来自 [REALM 附录 A.2](https://arxiv.org/html/2605.00271v2)。REALM 原论文在 DUNE 上使用另一种损失，本实验采用 Déjà-Events 损失并迁移 REALM 掩码。DSEC 的训练数据、图像对齐和裁剪是本项目的适配。

```bash
bash scripts/run_dsec_full.sh
```

正式划分见 [dsec_full_split.json](../scripts/dsec_full_split.json)。脚本先解压和检查所有 41 条序列，再在 GPU 0 和 1 启动配对实验，输出到服务器的 `/data/userwj/epf-runs/dsec_full_f3` 和 `/data/userwj/epf-runs/dsec_full_no_f3`。完整数据的每一轮包含超过两万个候选起点，因此将 REALM 的 30 轮掩码里程碑映射到整个蒸馏预算的相对进度。这样一次完整遍历内也能启用两种掩码。该映射是数据规模变化后的适配。

此前三序列配对长跑已停止并保留为先导实验，其权重不进入这轮正式训练。上一轮实验结果仍保留作历史参照。

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

# DSEC 长程蒸馏与重建

两组各运行 20,000 步 DVLT 蒸馏，再冻结骨干各训练 5,000 步预测几何 Gaussian head。GPU 0 运行 F³ 组，GPU 1 运行去掉 F³ 的组。两组均保留五时间 bin 事件体素、LoRA、SAAG、最终 token 损失和 REALM 双掩码。样本顺序、视图数和共享 K 由独立且两组同种子的随机流生成。训练采用 DSEC 三条序列，验证使用 zurich_city_01_a，thun_00_a 另用于留出深度评估。

蒸馏遵循 [Déjà-Events](https://arxiv.org/html/2611.21302v1) 的共享 K、最终 token 距离、2–18 视图、LoRA rank 16 与 AdamW 余弦预算。空间膨胀和第 8 轮启用输入遮挡来自 [REALM 附录 A.2](https://arxiv.org/html/2605.00271v2)。REALM 原论文在 DUNE 上使用另一种损失，本实验采用 Déjà-Events 损失并迁移 REALM 掩码。DSEC 的训练数据、图像对齐和裁剪是本项目的适配。

```bash
bash scripts/run_dsec_long.sh f3
# 在另一个终端运行
bash scripts/run_dsec_long.sh no_f3
```

两组首个短跑验证了相同的样本顺序、视图数量及 K。长跑输出位于服务器的 `/data/userwj/epf-runs/dsec_long_f3` 和 `/data/userwj/epf-runs/dsec_long_no_f3`。正式分数随结果产出更新。

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

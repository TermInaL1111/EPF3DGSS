# 外观头与评测协议决定（2026-11-28）

## 1. 相机协议

主表固定为：

> **已知传感器内参 + 事件预测的相对源位姿**。

- 不再使用“训练参考相机、测试射线拟合相机”。
- 射线拟合内参不可用：fx 偏 1.9×（DSEC）/3.4×（EventScape），主点偏 10–17 px；相对位姿仍准（0.57–0.76°，0.20–0.23 m）。
- 这条假设比 EvGGS 的 GT source pose 更弱；和 IncEventGS 的“内参已知、位姿估计”同一层级。
- 若要 intrinsic-free：在蒸馏阶段给 ray head 加教师 rays 或 K 监督；主表先不做。

## 2. 为什么渲染糊

**不是全局动态范围窄。** 实测分位数：

| 数据集 | 目标 p1 / p99 | 预测 p1 / p99 |
|---|---|---|
| DSEC | 0.024 / 0.843 | 0.031 / 0.871 |
| EventScape | 0.000 / 0.686 | 0.095 / 0.626 |

真正的问题是**空间低频**：预测灰度在几像素内几乎不变。来源：

- 体素内先平均特征，再回归一个灰度；
- F³ 特征 96×128，上采样到 480×640；
- DSEC 高斯投影半径中位数 4.4 px，大而半透明；
- 训练损失 L1+0.2 SSIM，偏向平滑均值；
- 12k 步、4,034 窗，约 3 epoch，明显欠拟合。

target-colour oracle（保留几何、只换目标帧颜色）在 DSEC 21.68 dB、EventScape 抽样 26.86 dB，说明颜色头是主缺口；oracle 21.7 也是当前几何的上界。

## 3. 解决顺序

| 优先级 | 改动 | 作用 |
|---|---|---|
| **A** | **octree + per-pixel colour**：480×640 预测灰度图，按叶内点平均成高斯颜色，加 per-leaf 残差 | 直接修空间低频；目标 DSEC 19–20 |
| B | 提高外观分辨率：F³ cell 5→2/1，或 event-voxel CNN | 纹理/边缘 |
| C | 缩小高斯/更多 primitive + 覆盖率正则 | 突破 oracle 21.7，往 22–24 |
| D | LPIPS/DISTS + 前景加权 + 源视图渲染损失 | 感知质量 |
| E | 12k→40k 步 | 欠拟合 |

**先做 A**；A 不动几何（octree 选叶、尺度、不透明度保持），只换颜色，便于归因。

## 4. 数据集协议

- **EventScape 主训练/主表**：合成、深度稠密、512×256。
- **DSEC held-out 测试序列**：DSEC 训练序列参与混合训练，`thun_00_a` 只用于测试；不再宣称 zero-shot。DSEC 深度稀疏（约 20% 有效），用来暴露几何差异。
- 协议变更原因：EventScape-only 的颜色头在 DSEC 上 7.44 dB，低于全零事件 9.20；混合训练 4k 已恢复到 17.43。
- EventScape 需要同时报 **full-gray** 和 **background-black** 两种目标口径，后者用于和 EvGGS 官方协议对齐。

## 5. 对比协议（关键）

PSNR 不能跨口径比较：

- EvGGS 官方 29.3 dB 是 **EV3DS + 背景置黑**；
- 同一批 EV3DS 窗口、**保留背景**时，EvGGS 只有 **11.55 dB**；
- EPF 在同一共享协议下 **11.84 dB**，PSNR 不输，但 LPIPS 更差（VGG 0.485 vs 0.452，Alex 0.623 vs 0.497）。
- MVSEC 72 窗：EvGGS 12.47，EPF 12.60–13.60，PSNR 不输，LPIPS 更差。

因此：

1. 主表必须固定：目标背景、分辨率/裁剪、target frame、源位姿来源、是否 per-scene。
2. IncEventGS 是 per-scene optimization，不能和 two-frame feed-forward 放同一张表；单独比 quality/runtime/是否需要 per-scene 训练。
3. Deja-Events/Event3R 只出几何，放 geometry 表，不放 PSNR 表。
4. 若固定协议后所有方法都低：报告相对排序 + source-copy/oracle 上界；若所有方法都低于 source-copy（DSEC 22.7 / ES 25.5），说明协议或指标需要调整。
5. target-colour oracle 只放 supplementary / 分析，不进主表。

建议表结构：

- 表 A（NVS，共享协议）：EPF / EvGGS / IncEventGS；PSNR、SSIM、LPIPS、coverage、runtime。
- 表 B（几何）：Deja-Events / Event3R / EPF；AbsRel、δ1、ATE、AUC。
- 表 C（分析）：camera（predicted/known-K/GT）、depth（pred/ref）、appearance（oracle）、target offset（2/4/8）。

## 6. 下一实验

`octree_pixel`：

- 保留 octree 的选叶、尺度、旋转、不透明度；
- 新增 480×640 per-pixel 灰度分支（24→24→1 conv），按叶内点平均，颜色 = 叶平均 + 0.25·tanh(残差)；
- EventScape 训练 12k，`target_offset=4`，已知内参 + 预测位姿；
- EventScape 主表 + DSEC zero-shot；
- 指标：PSNR/SSIM/LPIPS-VGG/Alex、前景 PSNR、coverage；定性图。
- 成功标准：DSEC ≥ 19.5，EventScape ≥ 19.5，LPIPS-VGG 明显低于当前 0.634/0.641。
- 若 A 后仍糊：B+C+D+E。

## 7. 当前基线（同协议）

| 数据集 | 方法 | PSNR | SSIM | LPIPS-VGG |
|---|---|---:|---:|---:|
| DSEC | octree target2 | 17.56 | 0.708 | 0.617 |
| DSEC | octree target4 | 17.79 | 0.722 | 0.634 |
| DSEC | VG²GT octree | 15.44 | 0.702 | 0.640 |
| EventScape | octree target2 | 17.88 | 0.592 | 0.619 |
| EventScape | octree target4 | 17.78 | 0.589 | 0.641 |
| EventScape | VG²GT octree | 17.66 | 0.587 | 0.655 |

VG²GT-style 结论：fine-voxel 在 DSEC 崩到 11.86；octree hybrid 15.44；self-splitting 在没有 volume rendering/多样性约束时拖坏几何。详细结果见 `md/vg2gt_head_results.md`。

## 8. 第一轮结果（octree + per-pixel colour，EventScape-only 12k）

| 数据集 | 方法 | PSNR | SSIM | LPIPS-VGG | LPIPS-Alex | 全零 PSNR | 高斯数 |
|---|---|---:|---:|---:|---:|---:|---:|
| EventScape | octree target4 | 17.78 | 0.589 | 0.641 | 0.667 | 13.40 | 328k |
| EventScape | **octree_pixel 12k** | **18.20** | **0.607** | **0.563** | **0.515** | 14.60 | 327k |
| DSEC | octree target4 | 17.79 | 0.722 | 0.634 | 0.606 | 9.47 | 219k |
| DSEC | octree_pixel zero-shot | **7.44** | 0.367 | 0.685 | 0.716 | 9.20 | 218k |

结论：

- **EventScape 上 per-pixel colour 有效**：PSNR +0.42，SSIM +0.019，LPIPS-VGG −0.078，LPIPS-Alex −0.152；视觉上颜色/边缘明显改善。
- **DSEC zero-shot 失败**：7.44 dB，比全零事件 9.20 还低；几何和相机没变，是 EventScape-only 训练的颜色头在真实事件上严重域外。原 octree（DSEC+ES 混合训练）在同一 DSEC test 是 17.79。
- 因此协议改为 **DSEC+EventScape 混合训练 + DSEC held-out 测试序列**（`thun_00_a` 不参与训练）；不再写 zero-shot。
- 混合训练 4k 的 test：DSEC 17.43 / 0.694 / LPIPS 0.650，EventScape 17.76 / 0.598 / LPIPS 0.588；DSEC 已从 7.44 恢复。
- 新增 **center-color** 头：在 Gaussian 中心投影处采样 per-pixel 颜色，而不是叶内平均。7.5k EventScape test：18.10 / 0.603 / LPIPS 0.578（leaf-average 12k 是 18.20 / 0.607 / 0.563）；还需训到 12k，并做 mixed 版本。

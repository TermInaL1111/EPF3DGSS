# 训练与检查点记录

本文件记录 2026-11-28 这轮外观头/高斯头改版的全部训练运行。机器可读清单在
[results/run_manifest.json](../results/run_manifest.json) 和
[results/run_manifest.csv](../results/run_manifest.csv)，其中包含每个
`latest.pt` 的绝对路径与 SHA-256。

## 固定条件

- 几何：冻结 `eventscape_realm_20k/geometry/latest.pt` 事件 DVLT；backbone、depth/ray decoder、event adapter 不训练。
- 相机：已知传感器内参 + 事件预测的相对源位姿。
- 目标：`target_offset=4`（源帧 i、i+2，目标 i+4）。
- 训练只更新 `gs_head.*`、`field_features.*`、`geometry_features.*`。
- EventScape-only 运行不包含 DSEC；混合运行包含 DSEC 训练序列 + EventScape，DSEC 测试序列 `thun_00_a` 留出。

## 运行清单

| run | head | 训练域 | steps | 状态 | DSEC PSNR / LPIPS | EventScape PSNR / LPIPS |
|---|---|---|---:|---|---:|---:|
| `octree_target4_12k` | octree | mixed | 12000 | finished | 17.79 / 0.634 | 17.78 / 0.641 |
| `vg2gt_target4_12k` | VG²GT fine voxel | mixed | 4600 | stopped（DSEC 几何崩） | 11.86 / 0.657 | 17.72 / 0.657 |
| `vg2gt_octree_target4_12k` | VG²GT octree hybrid | mixed | 12000 | finished | 15.44 / 0.640 | 17.66 / 0.655 |
| `octree_pixel_es_12k` | octree_pixel | EventScape | 12000 | finished | —（zero-shot 失败） | **18.20 / 0.563** |
| `octree_pixel_dsec_zeroshot` | octree_pixel | EventScape | 12000 | zero-shot | 7.44 / 0.685 | — |
| `octree_pixel_mixed_12k` | octree_pixel | mixed | 12000 | finished | **17.39 / 0.648** | 17.93 / 0.578 |
| `octree_pixel_center_es_12k` | octree_pixel_center | EventScape | 12000 | finished | — | 18.13 / 0.566 |
| `octree_pixel_center_mixed_12k` | octree_pixel_center | mixed | 12000 | finished | 17.77 / 0.635 | 17.73 / 0.639 |
| `octree_pixel_fine_es_12k` | octree_pixel_fine（2.5 cm） | EventScape | 4900 | stopped | — | — |
| `octree_pixel_es_40k` | octree_pixel | EventScape | 3900 | stopped（增益很小） | — | — |
| `octree_target8_eval` | octree | mixed | 12000 | target-8 评测 | 18.06（9 窗子集） | 17.68（336 窗子集） |

## 关键结论

- 相机协议修复：预测相机 12.04/9.35 → 已知内参 + 预测位姿 17.56/17.88。
- per-pixel colour 在 EventScape 上把 LPIPS-VGG 从 0.641 降到 0.563，PSNR 17.78→18.20；但 DSEC zero-shot 只有 7.44，颜色头域外失败。
- 混合训练把 DSEC 拉回 17.39，EventScape 17.93；DSEC 比 octree 低 0.40，EventScape LPIPS 更好。
- center-color（中心投影采样）12k 为 18.13，不如 leaf-average 18.20；不保留。
- fine 2.5 cm 叶子（442k 高斯）正在训练，是冲 19+ 的下一刀。
- VG²GT-style fine-voxel 在 DSEC 崩到 11.86；octree hybrid 15.44；self-splitting 在没有 volume rendering/多样性约束时拖坏几何。

## 检查点说明

- 每个训练目录下都有 `latest.pt`（约 4.3 MB）、`config.json`、`train.jsonl`、`val_*.json`；
  完成评测的运行还有 `val_test*.json` 和 `preview_test*.png`。
- `.pt` 被仓库 `.gitignore` 排除，不随 git 提交；它们保存在服务器的
  `/data/userwj/epf-runs/<run>/latest.pt`，SHA-256 见 `results/run_manifest.json`。
- 复现某个头需要同时有：本仓库对应 commit 的代码 + `eventscape_realm_20k/geometry/latest.pt` + 该 run 的 `latest.pt`。
- 评测命令统一为：`--head <head> --steps 0 --init-head <run>/latest.pt --known-source-intrinsics --target-offset 4`。

## 待补

- 两组各续训 12,000 步。GPU 0 仅监督目标视图，GPU 1 额外监督两个源视图。4,000 步权重已保存，4,034 步完成 144 个配对验证窗。正式测试按每个数据集的目标位移分三组。 [验证结果](../results/source_view_supervision/validation_004034.json) · [产物路径与哈希](../results/source_view_supervision/validation_004034_artifacts.json)
- 已统一 MVSEC 的两个渲染器的投影位置。EvGGS 官方权重的 72 窗复评排在 GPU 0 完成续训之后。IncEventGS 保留文献与指标对照。 [评测协议](evggs_baseline.md)

## 代码与几何版本

- 代码 commit：`9d0e321`（per-pixel colour head、VG²GT-style 变体、run manifest）。
- 冻结几何：`/data/userwj/epf-runs/eventscape_realm_20k/geometry/latest.pt`
  - SHA-256：`8e866eecf4466c15eead4eb1a4b751270cf5c79dc826842bac34b61947e55de2`
  - 大小：`12196026` bytes
- 头部 `latest.pt` 的 SHA-256 见 `results/run_manifest.json`。

## weights/ 归档

已把 4 个关键 run 的最终头复制到 `weights/`，文件名可读，源文件与副本 SHA-256 一致：

| run | 归档文件 | 字节 |
|---|---|---:|
| `octree_target4_12k` | `weights/octree_target4_12k.pt` | 4,331,981 |
| `octree_pixel_es_12k` | `weights/octree_pixel_es_12k.pt` | 4,354,057 |
| `octree_pixel_mixed_12k` | `weights/octree_pixel_mixed_12k.pt` | 4,354,057 |
| `octree_pixel_center_es_12k` | `weights/octree_pixel_center_es_12k.pt` | 4,354,057 |

`results/run_manifest.json` 现在同时记录 `checkpoint`（服务器原路径）、
`checkpoint_sha256`、`archived_weights` 和 `archived_weights_sha256`。

注意：`.gitignore` 仍排除 `*.pt`，所以 `weights/` 是本地归档，不随 git 提交。
若需要版本化权重，需单独用 Git LFS / 对象存储 / release asset。

## 中间 checkpoint 留档

`train_dsec_gaussians.py` 新增：

- `--save-every N`：每 N 步额外保存 `step_XXXXXX.pt`；
- 训练结束时总会保存 `step_<final>.pt`。

旧运行没有这些中间权重；只有 `latest.pt`（最终）和 `val_*.json` 指标。

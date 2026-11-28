# 11.28 DeepSeek 交接文档

**交接时间**：2026-11-28 06:55（America/New_York）  
**工作分支**：`codex/dsec-long-training`  
**交接范围**：事件 DVLT 几何 → Gaussian Head 的相机协议、外观头、VG²GT-style 消融、训练/评测记录，以及 Draftpaper 项目初始化。  
**当前结论**：相机协议已修复，深度不是主瓶颈；per-pixel colour 在 EventScape 到 **18.20 dB / LPIPS 0.563**，但 DSEC 必须混合训练（held-out 17.39 dB）；**19–21 dB 尚未达到**，fine 2.5 cm、deep decoder、high-res F³ 三个方向都停在约 16.13 dB（4k 验证点），没有继续加容量的收益。

---

## 1. 交接后我做了什么

按时间顺序：

1. **定位相机协议错配**：训练用参考源相机、测试用 DVLT 射线拟合相机；量化 fitted K/主点/射线角误差，确认位姿是准的。
2. **修复协议**：主表改为“已知传感器内参 + 事件预测的相对源位姿”；DSEC 12.04→17.56 dB，EventScape 9.35→17.88 dB。
3. **深度置换对照**：固定相机，替换参考深度，PSNR 只变 −0.14/−0.15 dB；排除“深度放错”作为主因。
4. **外观诊断**：target-colour oracle 显示几何可支撑 21.68/26.86 dB；剩余差距在颜色头与大高斯的空间低频，不是全局动态范围。
5. **实现 VG²GT-style 头**：fine-voxel、octree hybrid、per-pixel colour；fine/hybrid 在 DSEC 崩，形成负结果。
6. **实现 octree + per-pixel colour**：EventScape 18.20 / LPIPS 0.563；DSEC zero-shot 7.44（域外失败）；混合训练 held-out 17.39 / 17.93。
7. **协议改为混合训练 + DSEC held-out**：更新 goal 和文档，不再宣称 zero-shot。
8. **继续尝试 center-color、2.5 cm 叶子、deep decoder、high-res F³ detail field**：全部在 2k/4k 验证点与 5 cm 基线持平或略差；确认 18.2 是当前架构/信息量下的平台。
9. **建立记录**：`results/run_manifest.json`、`weights/` 4 个关键头、`md/run_log.md`、`md/appearance_head_plan.md`、`md/vg2gt_head_results.md`；更新 `md/1128.md`；把误加到 `md/11.18.md` 的内容撤回。
10. **Draftpaper 初始化**：安装项目级 skill、克隆/安装 CLI、建项目、导入 5 图 5 表和 10 篇 curated 文献；CVPR 模板仍是 fallback。

---

## 2. 关键结论（可直接写论文/汇报）

### 2.1 相机

| 源相机 | DSEC PSNR | EventScape PSNR |
|---|---:|---:|
| DVLT 射线拟合（旧） | 12.04 | 9.35 |
| 只换内参 | **17.56** | **17.88** |
| 只换位姿 | 12.03 | 9.36 |
| 内参 + 位姿 | 17.67 | 17.91 |
| 已知内参 + 预测位姿（主协议） | **17.56** | **17.88** |

射线拟合误差：

| 数据集 | fx 比 | fy 比 | 主点误差 | 射线角 | 相对旋转 | 相对平移 |
|---|---:|---:|---:|---:|---:|---:|
| DSEC | 1.924 | 1.289 | 16.78 px | 7.84° | 0.765° | 0.230 m |
| EventScape | 3.426 | 1.535 | 10.02 px | 17.40° | 0.566° | 0.202 m |

**结论**：射线头在事件域未标定，内参不可用；相对位姿准。主表写 “intrinsics calibrated, source poses predicted”，比 EvGGS 的 GT source pose 更弱。

### 2.2 深度

- 参考深度置换：DSEC −0.144 dB，EventScape −0.149 dB。
- 参考深度有效率：DSEC 20.1%，EventScape 78.6%。
- 重投影误差约 1.4–1.6 px。

**结论**：深度不是当前 PSNR 主因；DSEC 参考深度稀疏，置换对照偏弱。

### 2.3 外观

- target-colour oracle（保留几何、只换目标帧投影颜色）：DSEC **21.68 dB**，EventScape 抽样 **26.86 dB**；当前 17.67/17.56。
- 预测灰度的全局动态范围并不窄（DSEC p1/p99 0.031/0.871 vs 目标 0.024/0.843）；问题是**空间低频**。
- 主因：体素内平均特征、F³ 96×128 上采样 5×、DSEC 高斯投影半径中位数 4.4 px、L1+SSIM 偏向平滑。

### 2.4 VG²GT-style 负结果

| 数据集 | 头 | PSNR | SSIM | LPIPS-VGG |
|---|---|---:|---:|---:|
| DSEC | octree | 17.79 | 0.722 | 0.634 |
| DSEC | VG²GT fine-voxel | 11.86 | 0.625 | 0.657 |
| DSEC | VG²GT octree hybrid | 15.44 | 0.702 | 0.640 |
| EventScape | octree | 17.78 | 0.589 | 0.641 |
| EventScape | VG²GT fine-voxel | 17.72 | 0.588 | 0.657 |
| EventScape | VG²GT octree hybrid | 17.66 | 0.587 | 0.655 |

原因：固定 5 cm 网格在 DSEC 稀疏深度上保留大量离群点；self-splitting 共享叶特征、中心/尺度接近，重叠后反而糊。target-colour oracle：fine 12.62、hybrid 16.02，远低于 octree 21.68。

### 2.5 当前最好的外观头

| 设置 | DSEC PSNR / LPIPS | EventScape PSNR / LPIPS |
|---|---:|---:|
| octree target 4 | 17.79 / 0.634 | 17.78 / 0.641 |
| octree_pixel，EventScape-only | 7.44（zero-shot 失败） | **18.20 / 0.563** |
| octree_pixel，mixed | **17.39 / 0.648** | 17.93 / 0.578 |
| octree_pixel_center，EventScape | — | 18.13 / 0.566 |
| octree_pixel_center，mixed | 17.77 / 0.635 | 17.73 / 0.639 |

- EventScape per-pixel colour：+0.42 dB，LPIPS 0.641→0.563；渲染图能看出 STOP、护栏、树和路面标线。
- DSEC zero-shot 失败（7.44 < 全零 9.20）；混合训练恢复 17.39，但仍比 octree 低 0.40 dB。
- center-color 不占优（18.13 < 18.20）。

### 2.6 平台期证据

| 变体 | 验证点 | PSNR | LPIPS | 与 5 cm 基线 |
|---|---:|---:|---:|---:|
| fine 2.5 cm | 4k | 16.112 | 0.5537 | 16.132 / 0.5536（持平） |
| deep decoder | 2k | 16.859 | 0.5536 | 16.845 / 0.5568（+0.014） |
| high-res F³ detail | 4k | 16.134 | 0.5535 | 16.132 / 0.5536（持平） |
| center-color | 12k | 18.13（ES test） | 0.566 | 18.20 / 0.563（略差） |

**结论**：加容量/加分辨率/加叶子密度都不改变 4k 验证点；19–21 不是“再调一调”能到的，需要新的信息源或监督。

---

## 3. 代码改动

| 文件 | 改动 |
|---|---|
| `event_dvlt.py` | 注册 `octree_pixel` / `octree_pixel_center` / `octree_pixel_fine` / `octree_pixel_deep` / `vg2gt*`；confidence 分支；`detail_field` 高分辨率 F³；`source_intrinsics` 传给 center 头 |
| `voxel_gaussians.py` | `per_pixel_color`、`color_head`、`center_color`、`voxel_size`、`color_width/color_layers`、中心投影采样、颜色残差 |
| `vg2gt_gaussians.py` | fine-voxel self-splitting（bounded scale、center residual、多视图支持过滤） |
| `vg2gt_pixel_gaussians.py` | fine-voxel + per-pixel colour |
| `vg2gt_octree_gaussians.py` | octree 多尺度选叶 + self-splitting |
| `train_dsec_gaussians.py` | head choices；`--target-offset`；`--known-source-intrinsics`；EventScape-only；eval-only；`--save-every`；final `step_XXXXXX.pt`；detail field 数据流 |
| `dsec_temporal_gaussian_data.py` | `target_offset`、dict 相机、缺失目标窗过滤 |
| `eventscape_gaussian_data.py` | `target_offset`、dict 相机、EventScape-only |
| `scripts/diagnose_gaussian_blur.py` | 单窗几何/尺度/颜色/oracle 诊断，scale bound/min points/min views 干预 |
| `scripts/extend_temporal_cameras.py` | test/val 相机扩展到 offset 8 |
| `scripts/summarize_nvs_runs.py` | 多 run 同窗汇总 |
| `scripts/summarize_vg2gt_results.py` | VG²GT-style 对比表 |
| `scripts/build_run_manifest.py` | run/checkpoint/SHA/archived weights 清单 |

当前 **uncommitted**（交接时）：`event_dvlt.py`、`train_dsec_gaussians.py`、`voxel_gaussians.py` 的 detail-field/deep/fine 改动；交接 commit 会包含这些文件。

---

## 4. 记录与产物

| 内容 | 路径/说明 |
|---|---|
| 机器可读 run 清单 | `results/run_manifest.json` / `.csv`（path、SHA-256、指标、archived weights） |
| 训练/检查点记录 | `md/run_log.md` |
| 外观头设计与结果 | `md/appearance_head_plan.md` |
| VG²GT-style 负结果 | `md/vg2gt_head_results.md` |
| 11.28 汇报 | `md/1128.md` 第四节的表 9–11 |
| 关键头归档 | `weights/octree_target4_12k.pt`、`octree_pixel_es_12k.pt`、`octree_pixel_mixed_12k.pt`、`octree_pixel_center_es_12k.pt` |
| 定性图 | `paper/octree_pixel_es_comparison.png`、`paper/vg2gt_head_comparison.png`、`paper/eventscape_camera_control.png` |
| 几何 checkpoint | `/data/userwj/epf-runs/eventscape_realm_20k/geometry/latest.pt`，SHA `8e866eec...e55de2` |

`.pt` 被 `.gitignore` 排除，`weights/` 是本地归档，不随 git 提交；manifest 记录 SHA。

### Git

```text
9d0e321 Add per-pixel colour event Gaussian head, VG2GT ablations, and run manifest
c5f4e7b Record code and frozen geometry versions in run log
daa165c Archive key head checkpoints and record archived-weight hashes
973d7af Document 11.28 ...（误加到 11.18，已被 61e6659 撤回）
61e6659 Move 11.28 camera and appearance results into the 1128 report
```

交接 commit 见 `git log` 最新一条（包含 detail-field 代码与本文件）。

---

## 5. 当前运行/未完成

| 项 | 状态 |
|---|---|
| `octree_pixel_center_mixed_12k` | 完成：DSEC 17.77 / 0.718 / LPIPS 0.635；EventScape 17.73 / 0.588 / 0.639 |
| `octree_pixel_detail_es_12k` | 已停在 4.5k；4k val 与 5 cm 持平（16.134 vs 16.132） |
| `octree_pixel_fine_es_12k` | 已停在 4.9k；4k val 16.112 vs 16.132 |
| `octree_pixel_deep_es_12k` | 已停在 3.1k；2k val +0.014 dB |
| 19–21 dB | **未达到** |
| 同协议 EvGGS/IncEventGS 表 | **未完成**（需固定背景/裁剪/target frame/pose） |
| Draftpaper 文献 live search | 返回 0 最终项；已用 `--from-json` 导入 10 篇 curated |
| CVPR 模板 | fallback 到通用 Overleaf article，不是官方 `cvpr.sty` |

---

## 6. 风险与坑

1. **checkpoint 命名**：每个 run 目录一个 `latest.pt`，训练中每 100 步覆盖；旧 run 的中间 4k/8k 权重已被覆盖。新 run 用 `--save-every` 留 `step_XXXXXX.pt`。
2. **zero-shot 不成立**：EventScape-only 颜色头在 DSEC 7.44 dB；必须混合训练或 few-shot，论文不要再写 zero-shot。
3. **PSNR 不能跨口径比**：EvGGS 官方 29.3 dB 是 EV3DS 置黑背景；同批保留背景只有 11.55 dB。主表必须固定背景/裁剪/target frame/pose。
4. **target-colour oracle 不是 baseline**：它用目标帧颜色，只作 supplementary 上界。
5. **4k/2k 验证是单窗**，噪声大；最终以 test 全窗均值为准。
6. **EventScape 上四种头都 17.7–18.2**，区分度低；DSEC 才暴露几何差异。
7. **Draftpaper 项目在 `.gitignore`**（`draftpaper/`、`refs/`、`.dsh/`），不随主仓库提交；需要单独备份。

---

## 7. 下一步建议（按优先级）

1. **换信息源，而不是继续加容量**：当前 fine/deep/detail 都平台在 16.13（4k）。建议做 **源视图渲染损失**（源帧 i/i+2 的灰度也监督）或 **RGB 教师外观特征蒸馏**；否则 19–21 没有信息基础。
2. **固定同协议对比**：先定目标背景（全灰/置黑）、裁剪、target frame、源位姿来源，再跑 EvGGS/IncEventGS；不要用 29.3 vs 17.8 直接比。
3. **DSEC held-out**：用混合训练 + `thun_00_a` 只测试，目标 17.5+；若做 few-shot，明确标注。
4. **论文定位**：如果 19+ 短期达不到，建议把贡献放在“相机/深度/外观误差分解 + 可部署的 known-K/pose-free 协议 + 大规模同协议评测”，而不是 SOTA PSNR；per-pixel colour 的 LPIPS 0.641→0.563 是真实收益。
5. **记录**：新 run 加 `--save-every 4000`；完成后更新 `run_manifest.json`、`weights/`、`run_log.md`。
6. **Draftpaper**：下一步 `preflight-research-feasibility`；补官方 CVPR 模板；文献用 curated JSON 继续。

---

## 8. 复现命令

### 训练 EventScape per-pixel colour

```bash
source scripts/server_env.sh
export CUDA_VISIBLE_DEVICES=0
deps=/data/userwj/projects/epf-deps
export PYTHONPATH="$PWD:$PYTHONPATH:$deps/OctreeSplat:$deps/octree-cpu-deps"
run=/data/userwj/epf-runs
python -u train_dsec_gaussians.py "$EPF_DSEC" "$EPF_ASSETS" "$deps/fast-feature-fields" \
  "$run/eventscape_realm_20k/geometry/latest.pt" "$run/octree_pixel_es_12k" \
  --eventscape-root /data/userwj/datasets/EventScape \
  --eventscape-cameras "$run/eventscape_rgbd_cameras" \
  --geometry predicted --head octree_pixel --steps 12000 --val-limit 1 --lpips \
  --known-source-intrinsics --target-offset 4 --save-every 4000
```

### 评测

```bash
python -u train_dsec_gaussians.py "$EPF_DSEC" "$EPF_ASSETS" "$deps/fast-feature-fields" \
  "$run/eventscape_realm_20k/geometry/latest.pt" <eval_out> \
  --eventscape-root /data/userwj/datasets/EventScape \
  --eventscape-cameras "$run/eventscape_rgbd_cameras" \
  --geometry predicted --head octree_pixel --steps 0 --val-limit 1 --lpips \
  --known-source-intrinsics --target-offset 4 \
  --init-head "$run/octree_pixel_es_12k/latest.pt"
```

混合训练把 `--eventscape-root/--eventscape-cameras` 与 `--dsec-temporal-cameras` 同时传入；DSEC zero-shot 只传 `--dsec-temporal-cameras`。

---

## 9. 交接检查清单

- [ ] `git log` 最新 commit 包含 detail-field 代码与本文件。
- [ ] `results/run_manifest.json` 中 4 个 archived weights 的 SHA 与 `weights/` 一致。
- [ ] `octree_pixel_center_mixed_12k` 的 test 已完成并补进 manifest（若未完成，标 pending）。
- [ ] 确认论文主表使用 known-K + predicted pose，而不是旧 12.04/9.35。
- [ ] 确认不再写 DSEC zero-shot。
- [ ] 同协议 EvGGS/IncEventGS 表尚未完成，交接时不要声称已比较。
- [ ] 19–21 dB 未达到；不要把 18.20 当最终 SOTA。

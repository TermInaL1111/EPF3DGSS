# EPF3DGSS 接续目标

以 `8ed1172` 的交接为起点，面向 CCF A/B 投稿完成可复现的单目事件前馈 3DGS 研究与论文。当前优先改善画质，并验证收益能否跨场景重现。DSEC 和 EventScape 的混合 center-color 已完成，21 窗和 463 窗的 PSNR 分别为 17.7679 和 17.7246 dB。19–21 dB 是待验证目标。 [交接文档](1128deepseek.md) · [运行清单](../results/run_manifest.json)

## 下一项配对实验

1. 以 `octree_pixel_mixed_12k` 为共同起点，用同一采样顺序和训练步数，对照仅目标灰度监督与额外的两个源视图灰度渲染监督。主推理协议为已知传感器内参、事件预测源位姿。源灰度只在训练期使用。对 DSEC 使用事件相机与灰度相机的已标定变换，分别以两个源时刻的灰度相机渲染监督。
2. 按 DSEC 留出序列、EventScape Town05、MVSEC 跨域分别报告 PSNR、SSIM、LPIPS-VGG、LPIPS-Alex、真实/全零差和定性图。报告 dB 差和方向一致的百分比。对长目标位移单独汇总。若源视图监督收益不足，再做训练期 RGB 教师外观蒸馏的配对实验。
3. 保留 Déjà-Events 与 REALM 的 DVLT 长程蒸馏和深度、点图、位姿评测。新视角比较使用 EvGGS 的共同目标、背景、窗口和相机条件，单列 EV3DS 原协议。IncEventGS 保留指标与文献对照。
4. 只用 GPU 0、1。记录训练时间与 4k、8k、12k 检查点，按 ETA 等待。核对产物后更新运行清单、日志、1128 汇报及 GitHub。

## Draftpaper_loop

沿用已初始化的本地项目和 0.43.2 CLI（源码 `4cf6b89`）。Draftpaper 消费实验仓库的逐窗结果、清单、图表和文献。主仓库继续维护可复现的训练与评测。

从项目根目录执行：

```bash
project="$PWD/draftpaper/event-only-pose-free-feed-forward-3d-ga_c40f620d"
cd refs/Draftpaper_loop_repo
.venv/bin/python -m draftpaper_cli.cli session-preflight --project "$project"
.venv/bin/python -m draftpaper_cli.cli verify-next-action --project "$project"
```

运行时身份检查和 `preflight-research-feasibility` 已通过。随后 `generate-plan` 因将英文项目描述带进中文版研究计划而被语言校验拒绝。现有英文项目描述也仍把 self-splitting 写成主头，需要按当前证据更新。项目已有本地图表和文献导入。稿件编排前还需核验文献、统一实验口径并替换当前通用 LaTeX 模板。 [Draftpaper_loop 上游](https://github.com/xiejhhhhhh/Draftpaper_loop)

## 源视图监督运行

GPU 0 运行仅目标帧组，GPU 1 运行源视图监督组。两组从 `octree_pixel_mixed_12k` 续训 12,000 步，AdamW 学习率为 0.0001，随机种子为 11。源事件对应第 `i` 和 `i+2` 帧，目标为第 `i+4` 帧。源视图组将两个源灰度重建的平均 L1 + SSIM 损失按 0.5 加到目标损失上。沿用 4,034 个 DSEC 和 EventScape 训练窗。各轮随机打乱后不放回采样，两个组使用相同顺序。

DSEC 与 EventScape 留出评测沿用 target-offset 4。MVSEC 沿用 EvGGS 的 72 个固定窗口，源为 `i,i+4`，目标为 `i+2`。它是不同目标间隔的跨域测试。主结果使用已知内参和预测源位姿，另留相同标定源相机的对照。两个组完成后，汇总脚本核验配对窗口、逐步样本顺序和源损失记录，并计算相对变化。 [训练和评测入口](../scripts/run_source_view_supervision.sh) · [配对汇总](../scripts/summarize_source_view_supervision.py)

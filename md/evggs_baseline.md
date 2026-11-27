# EvGGS 基线

使用官方 [EvGGS](https://github.com/Mercerai/EvGGS) 提供的 `pretrain_evggs.pth`（340,000 步）。从 README 的 OneDrive 链接下载 `CKPT.zip`，验证 ZIP 后严格加载 `network`。权重 SHA-256 为 `1ba5a707a16611f01f4921a33f3834c6b0f492c6d14776308eaebd92488e5fcf`。压缩包同时含 `ft_evggs.pth`，此前卷积先导实验的权重摘要对应这个微调文件。本次主表采用官方合成预训练配置指定的权重。EvGGS 代码为 `4c97c29da989a8555e04e2a0020b46ada8861665`，Graphdeco rasterizer 为 `59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d`。

- EV3DS 共享协议沿用 EPF 的 `i, i+4 → i+2`，每个源用同一个事件包，评测两组同时使用 512×480 中央裁剪和未遮黑灰度目标。EvGGS 使用标定源位姿，EPF 自己预测源位姿。两组都使用相同的目标帧索引和已知目标相机。基础指标为 PSNR、SSIM、LPIPS-VGG、LPIPS-AlexNet，额外记录前景 PSNR 和零填充 SSIM。
- MVSEC 使用同一组三条室内航线，每条 24 个固定窗口。两组沿用同一个 640×480 虚拟相机、事件包、目标相机和 128×96 灰度目标。EvGGS 使用标定源位姿、官方预测前景和深度范围 0.8 m，以米为单位对真实数据作零样本推理。
- EV3DS 官方输入协议使用两个连续事件包组成一侧输入、`i, i+2 → i+1` 和遮黑目标，同时记录原图指标。全划分评估排除了会跨首尾拼接事件的窗口。此结果单独报告。
- EvGGS 源视图深度按有效真值像素计算原尺度 AbsRel、δ1，保留预测覆盖率。MVSEC 通过标定将深度映射到虚拟相机，按 25 ms 时间差和 0.1–50 m 真值范围筛选。深度汇总对有有效真值的源视图取平均。

## 复跑

在项目环境中安装 `yacs`、`natsort` 和上述版本的 Graphdeco rasterizer。设置 `CUDA_VISIBLE_DEVICES=1` 后执行以下命令。

```bash
source scripts/server_env.sh
export CUDA_VISIBLE_DEVICES=1
repo=/data/userwj/projects/epf-deps/EvGGS
weights="$EPF_ASSETS/evggs/CKPT/pretrain_evggs.pth"
flight="$EPF_MVSEC/indoor_flying"
python evaluate_evggs.py ev3ds "$repo" "$weights" "$EPF_EV3DS" ev3ds_val.json
python evaluate_evggs.py ev3ds "$repo" "$weights" "$EPF_EV3DS" ev3ds_test.json --split test
python evaluate_evggs.py ev3ds "$repo" "$weights" "$EPF_EV3DS" ev3ds_native_val.json --native
python evaluate_evggs.py ev3ds "$repo" "$weights" "$EPF_EV3DS" ev3ds_native_test.json --native --split test
python evaluate_evggs.py mvsec "$repo" "$weights" "$flight" mvsec.json --calibration "$flight/calibration/camchain-imucam-indoor_flying.yaml"
```

用 `scripts/summarize_evggs.py` 计算汇总。使用 `--epf` 时，脚本逐项检查场景或航线和帧索引的顺序。

## 共享窗口结果

| 数据集 | 方法 | 窗口 | PSNR ↑ | SSIM ↑ | LPIPS-VGG ↓ | LPIPS-AlexNet ↓ | 前景 PSNR ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| EV3DS 验证 | EvGGS | 136 | 11.55 | 0.051 | 0.452 | 0.497 | 16.66 |
| EV3DS 验证 | EPF 发布权重 | 136 | 11.84 | 0.127 | 0.485 | 0.623 | 12.42 |
| EV3DS 测试 | EvGGS | 128 | 11.56 | 0.054 | 0.455 | 0.498 | 15.98 |
| MVSEC | EvGGS | 72 | 12.47 | 0.053 | 0.656 | 0.543 | |
| MVSEC | EPF 发布权重 | 72 | 13.50 | 0.213 | 0.710 | 0.684 | |

EPF 两行均使用仓库的 EV3DS 发布权重。正式 DSEC 两个训练组还在蒸馏，完成重建后将用相同 MVSEC 窗口加入对照。EvGGS 的原尺度深度在 EV3DS 验证集为 AbsRel 0.1037、δ1 0.9255，在测试集为 0.1148、0.9099，在 MVSEC 为 0.9294、0.0036（135 个有效源视图）。当前表格的深度结果来自 EvGGS。

可机读汇总在 [results/evggs](../results/evggs/)，逐窗口产物保存在服务器的 `/data/userwj/epf-runs/evggs`。

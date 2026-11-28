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

## EV3DS 原生协议

| 划分 | 场景 | 窗口 | 遮黑 PSNR ↑ | 遮黑 SSIM ↑ | LPIPS-VGG ↓ | LPIPS-AlexNet ↓ | 原图 PSNR ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 验证 | 17 | 3,366 | 29.31 | 0.956 | 0.057 | 0.056 | 11.62 |
| 测试 | 16 | 3,168 | 27.74 | 0.943 | 0.067 | 0.066 | 11.67 |

初次原生测试调用遗漏了测试划分参数。已重跑并核对官方测试场景清单、产物的划分字段和每场景 198 个窗口。

## 官方验证脚本复跑

直接运行 EvGGS 的 `eval_gs.py`，使用官方 `val` 划分和 `pretrain_evggs.pth`，保留全部 3,417 个窗口，得到 **PSNR 29.29363078 dB、SSIM 0.95764233、LPIPS-VGG 0.05682888**。评分使用加载器的 `cim`，即背景置黑的灰度目标。官方 PSNR 对每张图用 `20 log10(1 / sqrt(MSE))`，最后对所有窗口取平均。运行时仅为当前 NumPy 补上游使用的 `np.int` 别名。

共享验证表的 **11.55 dB** 来自 136 个配对窗口和保留背景的灰度目标。上表原生验证的 **29.31 dB** 使用背景置黑的目标，并排除每场景最后三个回绕窗口。引用官方验证复跑时使用 **29.2936 dB**。官方入口固定使用验证划分，原生测试行来自本项目评测入口。

在同一组 3,366 个 EV3DS 窗口上，EvGGS 对置黑目标是 29.31 dB，对原灰度目标是 11.62 dB。这个差值来自目标图像口径，比较方法时需要固定目标。若复现 EV3DS 原生口径，两种方法须同时使用官方窗口、相机和同一张背景置黑的真值，再用官方全图公式计分。前景 PSNR 另外只在掩码内计误差，两种分数分别报告。

MVSEC 对照使用 72 个共享窗口和同一 128×96 灰度目标，双方提供标定源位姿并以黑色作为未覆盖区域的渲染背景。现有 [11.28 汇报的 MVSEC 表](1128.md#2-单目时序新视角合成) 记录早期评测，其中 EvGGS 将 OpenCV 整数像素中心的主点直接传给光栅化器，使投影相对目标偏移半个原始像素。修订版将双方的渲染主点统一为 (320, 240)，并在结果中记录内参。EvGGS 官方权重在 EV3DS 上训练，EPF 高斯头在 DSEC 和 EventScape 上训练，该表比较的是各自权重的跨域表现。

按首帧到目标帧的位移等量分成三组，每组 24 窗。EPF 体素头相对 EvGGS 的 PSNR 差在短、中、长位移组分别为 +1.255、−0.651、−0.220 dB。三个分组中，EPF 体素头的两种 LPIPS 都更高。共享窗口总均值的 PSNR 优势主要来自短位移组。 [9 窗同帧渲染](../paper/mvsec_evggs_nvs.png) · [导出命令](../scripts/run_mvsec_preview.sh) · [制图脚本](../scripts/plot_mvsec_preview.py)

## DSEC 训练权重在 MVSEC 的正式对照

无 F³ 组完成 22,522 步 DSEC 蒸馏和 5,000 步 Gaussian head 训练后，在同一批 72 个 MVSEC 室内窗口上达到 15.18 dB PSNR、0.360 SSIM、0.697 LPIPS-VGG、0.661 LPIPS-AlexNet。相对 EvGGS，PSNR 高 2.71 dB，SSIM 高 0.307，两种 LPIPS 分别高 0.041 和 0.118。EvGGS 使用标定源位姿，EPF 预测源位姿，两者的权重训练域也不同。有 F³ 组正在蒸馏。 [可机读配对结果](../results/dsec_full/no_f3_vs_evggs_mvsec.json)

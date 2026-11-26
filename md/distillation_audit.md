# 蒸馏实现核对（2026-11-26）

依据用户要求：先核对 `11.18.md` 引用的论文与官方代码，再推进训练。
DSEC 为主训练、EV3DS 为基线、MVSEC 为跨域评估；最多使用 GPU 0、1。

## 已获取的官方来源

源码路径与完整提交号见 `scripts/upstream_sources.json`。

- REALM：`/data/userwj/projects/epf-deps/REALM`，包含递归子模块。
- F³：`/data/userwj/projects/epf-deps/fast-feature-fields`。
- DVLT：`/data/userwj/projects/epf-deps/dvlt`。
- REALM 原文：https://arxiv.org/html/2605.00271v2 （§3.3、附录 A.2/B）。
- Déjà-Events 原文：https://arxiv.org/html/2611.21302v1 （§III、IV-A）。

已检查的 REALM 提交提供模型加载、事件表示、embedding、评估代码。
其中 `realm/realm/dune/losses.py` 是 DUNE 的损失，不能冒充完整 REALM 蒸馏训练入口。
`model_factory.py` 从发布 checkpoint 的 config 读取 LoRA 参数；代码里的 fallback 不是论文实验配置。
截至本次核查，尚未定位 Déjà-Events 官方训练仓库；不能声称已克隆其训练实现。

## 必须修正的差异

| 项目 | 依据 | 当前草稿问题 / 后续处理 |
|---|---|---|
| 教师目标 | Déjà-Events §III-B，匹配最终 tokens，包含 camera/register | 草稿只映射 patch tokens；需重新设计同视场配对后保留完整目标 |
| 循环次数 | Déjà-Events Eq.6，两模型共享每次采样的 K | 固定 K 的缓存不满足要求；更换为在线教师或按 K 区分的缓存 |
| REALM 空间课程 | REALM §3.3、附录 A.2，活动区域逐渐膨胀 | 当前 coverage 是标定有效性，不是活动掩码；随机遮挡增加不等同空间膨胀 |
| 输入遮挡 | REALM 附录 A.2，warmup 后启用 token dropping | 当前从首步开始递增；需对齐原文并明确在 DVLT 上的迁移方式 |
| 损失与优化器 | 两篇论文各自有目标和配置 | 要分别提供可追溯配置；组合方案不得称为任一原论文复现 |
| DSEC 对齐 | REALM 附录 B，先对齐 RGB 与事件 | 草稿在教师 token 层做稀疏重投影，是额外设计，不能当作官方做法 |
| 数据窗口 | REALM 附录 B 有 DSEC 的取样规则 | 草稿固定 20ms 只是工程假设，必须改为有依据的配置 |
| F³ 接口 | 官方 hubconf.py 与发布模型 | 已严格加载官方权重；接入 DVLT 的 adapter 属本项目扩展，需独立标记/消融 |

## 运行状态

- DSEC 4 步 smoke 已结束，只证明工程链路能够反向传播；不作为论文复现或方法收益证据。
- 已停止后续 DSEC 缓存→2000步训练的进程链；旧训练入口已由下文的时间配对训练替换。
- EV3DS F³ 缓存已完成（`/data/userwj/checkpoints/epf/f3_dense_full.pt`）。
- 保留数据、预训练权重及环境，后续实现核对通过后复用。

## 下一步验收

1. 按公式独立实现并验证 Déjà-Events 蒸馏、共享 K 与 SAAG。
2. 按 REALM 论文/公开代码明确活动掩码、空间扩张与输入遮挡，标注缺少官方训练源码的部分。
3. DSEC 建立同视场 RGB/Event 配对审计，区分有效性掩码与活动掩码；不使用未经验证的 token 重投影目标启动正式训练。
4. 在同数据划分下区分基础蒸馏、加入 REALM、加入 F³ 的配置，先验证短程、再训练。
5. 接续 Gaussian head、EV3DS 基线和 MVSEC 跨域评估，所有命令限制 GPU 0、1。

## 已落实的公式模块

`distillation.py` 分开实现 Déjà-Events Eq.7 和 REALM Eq.1 的空间项，
以及 REALM 附录 A.2 的活动掩码膨胀和 token-drop 概率课程。
明确标注为本地按论文转写，未冒充官方训练代码；未将两个损失随意相加。

CPU 检查通过：相同特征距离为零；单活动 patch 在 epoch 1/10/15/20
的覆盖数为 1/25/81/169；epoch 8 启用 30% dropout；已知输入损失
为 0.7；无效 patch 梯度为零；教师不接收梯度。
尚未接入正式训练，仍需完整验证共享 K、CLS/特殊 tokens、DSEC 配对及模型接线。

### 官方循环接口验证

`sample_shared_steps` 直接调用已锁定 DVLT 的 `_sample_K`（Beta 分布、round-to-bin），
不另行猜测整数取整方法。`teacher_final_tokens` 使用官方 inference solver，
显式指定 K，并保留 camera/register tokens；返回后恢复模型原配置。

GPU 0 实测：K=12 时该接口的全部 tokens 与官方 `forward_inference` 的
深度 decoder 输入逐元素一致（rtol=atol=0）；SAAG correction 为零时，
项目 `gated_update` 与官方 frame attention block 输出逐元素一致。
这证明接口和门控初始等价性，不代表非零 SAAG、完整学生或数据配对已验证。

### DSEC 原始事件坐标的 RGB 对齐审计

检查 REALM 官方 `utils/transforms.py::Warper._build_maps`：使用旋转映射，
未使用平移或逐像素深度。`audit_dsec_alignment.py` 将该几何方式适配到
DSEC 的 raw-event / rectified-RGB 标定格式，不再先重映射教师 tokens。

两条初始序列各抽四帧，用官方稀疏视差投影作为几何参照：忽略平移导致的
RGB 原图坐标误差中位数为 1.95–3.88 px，95 分位为 5.16–8.32 px。
这不是完整对齐精度：只覆盖有视差点，未衡量时间窗运动、遮挡和动态物体。
14px 统计阈值只是原图像素阈值，不能当作模型 patch 验收阈值。

报告：`/data/userwj/epf-runs/dsec_alignment/alignment.json`。
同目录包含八张 warped-RGB / 最近150000原始事件叠加图。
人工检查第78帧可见主要建筑轮廓匹配，但近处道路和运动边缘仍有时间累积痕迹。
正式训练需要保留这种近似及有效视场的记录，扩展更多序列并核对完整学生流程。

### DSEC 时间配对训练已接通

`train_dsec.py` 使用同一左事件相机连续时间视图，帧步长为 2，
每帧从图像时间戳向前取 150000 个原始事件，再裁到双方共同视场。
RGB 先用 REALM 的旋转映射方法对齐，再送进在线 DVLT 教师。
共同裁剪区为 560×420，视图缩放到 DVLT 的 224×168 网格。
公开论文的 REALM 训练使用 DUNE，本项目只将其空间膨胀和输入遮挡课程迁移至
DVLT。最终损失用 Déjà-Events Eq.7，启用课程时始终保留 camera/register tokens。
F³ 是可选适配器，使用官方权重。两视图、裁剪和分辨率为本项目配置。

GPU 0 的八步检查覆盖第八轮输入遮挡和独立序列验证。两帧对验证损失从
0.28542 到 0.28033。首轮 30 轮任务的配置与结果存于
`/data/userwj/epf-runs/dsec_temporal_realm_f3`。该任务使用
zurich_city_02_a 的 116 对以及 zurich_city_01_a 的 48 个抽样验证对。

MVSEC 的 `dvlt_real_transfer.py --geometry-only` 用 DSEC checkpoint 和冻结
DVLT 几何头测相对旋转。使用完整 EV3DS Gaussian checkpoint 时，保留原先的
灰度渲染和旋转指标。两类输出衡量的模型阶段不同。

MVSEC 几何短程评估已成功输出三条航线共 6 个窗口的匹配/全零事件旋转误差。
完整 EV3DS 验证和 MVSEC 发布权重参照使用 GPU 1。

### 灰度 Gaussian head 与 EV3DS 基线

`train_dsec_gaussians.py` 使用同步 DSEC 左右事件相机及 20 ms 事件窗。
双目视差提供稀疏几何，经过标定投影和 z-buffer 后进入参考组。
预测组由冻结的 DVLT 学生预测两路深度、相机和内参，并学习全局尺度。
两组在同一划分上使用右侧 RGB 灰度图渲染监督，报告全图 PSNR 和全零事件对照。
稀疏参考组的未覆盖像素会降低全图 PSNR，结果只能代表该协议下的标定参考。
该阶段从单相机时间配对扩展到同步双目，两阶段的视图和坐标分布存在差异。

EV3DS 发布权重已完成独立验证集全部 136 个窗口。匹配事件的平均 PSNR 为
11.7104 dB，零事件为 12.3365 dB。对应前景 PSNR 为 12.4162 和 13.0725 dB。
本轮结果没有观察到匹配事件提高 EV3DS 渲染 PSNR。
逐窗口数据见 `/data/userwj/epf-runs/ev3ds_reference.json`。

DSEC 两组先执行各两步集成检查。正式首轮各 500 步，在 30 轮几何蒸馏及检查完成后
串行运行。预测组完成后，使用其 checkpoint 进行 MVSEC 跨域渲染评估。

### 阶段 A 几何检查与 MVSEC 参照

`evaluate_dsec_geometry.py` 对独立序列的同步双目事件运行冻结模型。
使用每个双目对共享的中位数深度尺度，计算稀疏深度和点图误差。
相对旋转和翻译方向直接与事件相机的标定比较。

第 7 轮快照在 zurich_city_01_a 的 24 个窗口上完成检查。
匹配事件和全零事件的深度 AbsRel 分别为 0.1670、0.4816，
旋转误差为 0.9020°、0.8228°，平移方向误差为 17.6748°、89.3707°。
这组数据用于同一时刻的双目验证，源输入和训练时的单相机时间配对有差异。
逐窗口数据见 `/data/userwj/epf-runs/dsec_geometry_interim.json`。

额外三条序列 interlaken_00_c、zurich_city_00_a、thun_00_a 的 RGB、事件和视差
已读通，分别有 267、468、118 个时间帧对。
EV3DS 发布权重在 MVSEC 三条航线共 72 个窗口的匹配/全零 PSNR 为
13.5011/13.1703 dB，旋转误差为 1.8448°/1.2748°。
逐窗口数据见 `/data/userwj/epf-runs/mvsec_ev3ds_reference.json`。

### 蒸馏对照与米制尺度初始化

新增 F³（无 REALM 掩码）和基础 Déjà-Events 式蒸馏两组对照。
固定初始数据划分、种子、48 个验证窗口、30 轮余弦学习率计划，
首批各执行 10 轮并与主实验第 10 轮比较。不同模型的随机数消耗可能不同，
逐轮记录 K 和样本编号。结果目录为 `dsec_ablation_f3` 与 `dsec_ablation_base`。

第 7 轮的稀疏几何检查显示，预测深度与米制深度的匹配尺度约为 6–7。
预测几何 Gaussian head 在训练集均匀抽 12 帧，先逐帧取深度比值中位数，
再用跨帧中位数初始化一个可学习的全局尺度。原始 EV3DS 项目的尺度初值
约为 0.5。新初值仅使用 DSEC 训练标签，记录到 `scale_initialization.json`。

### 渲染指标与视场检查

三套渲染入口现在统一计算灰度 PSNR 和 SSIM。SSIM 用 11×11 高斯核
（σ=1.5）在完整落入图像的窗口上求均值。`scripts/summarize_render.py`
输出匹配和全零事件的均值、同窗差值，以及序列、场景或航线汇总。
EV3DS 136 窗重评的匹配/全零 SSIM 为 0.10152/0.21675，
原 PSNR 结果得到复核。

DSEC 验证保存首个窗口的真值、匹配、全零三联图。
早期预览显示，预测几何的渲染只占目标图像中央的一部分。
因此同时报告全图 PSNR 与标定确定的名义视场重叠区域 PSNR，
以便查看视场覆盖对指标的影响。该区域来自旋转投影的无穷远近似，
不能用来断定实际可见性或近距离遮挡。训练仍按全图灰度误差执行。

灰度 Gaussian head 的正式训练使用 0.8 L1 + 0.2 (1-SSIM)。
权重是本项目固定的初始实验设置，两组几何使用相同的系数。
验证同时保留全图和名义重叠区域 PSNR，优化使用全图目标。

MVSEC 新评估记录两源帧和目标帧的位移，外加首帧灰度复制和恒等旋转参照。
复制首帧使用了额外的灰度输入，只衡量短基线下不重建场景的参照水平。
已完成的 72 窗参照 SSIM 为匹配 0.21289、全零 0.11892。

### 从事件包导出可渲染场景

`reconstruct.py` 支持预测几何 DSEC Gaussian checkpoint 和 EV3DS 发布权重。
DSEC 路径使用 16 个循环，EV3DS 发布权重沿用 12 个循环。
两路输入均为 `[4,N]` 的 NPY（x、y、时间戳、极性）。
DSEC 双目输入需与 Gaussian head 训练协议一致，使用标准 DSEC 校正映射、
640×480 坐标和同一时间戳前的 20 ms 事件窗。

```bash
cd /home/userwj/EPF3DGSS
source scripts/server_env.sh
CUDA_VISIBLE_DEVICES=1 python reconstruct.py "$EPF_ASSETS" \
  /data/userwj/epf-runs/dsec_gs_predicted/latest.pt \
  /data/userwj/projects/epf-deps/fast-feature-fields "$EPF_ASSETS/f3.pth" \
  left.npy right.npy scene.npz
```

导出中心、尺度、四元数、灰度、不透明度、相对位姿和源深度。
两个事件文件是推理时唯一的样本输入。标定几何对照的 checkpoint 需要额外的
源几何标签，场景导出入口会拒绝该权重。

### 正式实验入口

从仓库根目录加载 `scripts/server_env.sh`。训练与评估分别设置
`CUDA_VISIBLE_DEVICES=0` 或 `CUDA_VISIBLE_DEVICES=1`。
当前各单卡任务的输入、输出和验证划分如下。

| 任务 | 入口 | 产物目录或报告 |
|---|---|---|
| DSEC 30 轮蒸馏 | `train_dsec.py --realm --f3` | `dsec_temporal_realm_f3/` |
| DSEC 10 轮对照 | `train_dsec.py --stop-after 10`，分别开启和关闭 F³ | `dsec_ablation_f3/`、`dsec_ablation_base/` |
| 相同验证窗口对照 | `scripts/compare_distillation.py --epoch 10` | `dsec_distillation_epoch10.json` |
| 最终 DSEC 几何评估 | `evaluate_dsec_geometry.py`，原验证序列与三条额外序列各 24 帧 | `dsec_geometry_final.json` |
| 灰度重建 | `train_dsec_gaussians.py --geometry calibrated/predicted`，各 500 步 | `dsec_gs_calibrated/`、`dsec_gs_predicted/` |
| EV3DS 灰度参照 | `train_event_dvlt.py::evaluate`，发布权重与 136 个窗口 | `ev3ds_reference_ssim.json` |
| MVSEC 跨域几何及渲染 | `dvlt_real_transfer.py`，三条航线各 24 个窗口 | `mvsec_dsec_temporal_realm_f3.json`、`mvsec_dsec_gaussians.json` |

表中产物路径相对于 `/data/userwj/epf-runs`，训练完成后才会写出最终结果。

首轮 DSEC 组合蒸馏已完成 30 轮、3,480 步。第 30 轮的 48 个验证窗口
中匹配事件的平均 token 损失为 0.18880，全零事件为 0.38764。
最终 checkpoint 的 48 个适配参数张量均为有限数值。
后续标定与预测几何训练使用同一份第 30 轮 checkpoint。

第 10 轮的三组蒸馏对照已用完全相同的 48 个验证窗口汇总。
基础组、F³ 组、F³ 加 REALM 掩码组的匹配 token 损失分别为
0.17616、0.17395、0.20493。见 `dsec_distillation_epoch10.json`。
结论限定为该划分、单随机种子与固定第 10 轮，完整组合组仍按原计划跑到第 30 轮。

第 30 轮组合组在 MVSEC 三条独立航线的 72 个窗口上完成几何评估。
匹配事件、零事件、恒等旋转的平均旋转误差分别为 1.0985°、0.7345°、0.6664°。
源帧平均平移 0.0545 m。该短基线协议的旋转结果较弱。
分航线结果见 `mvsec_dsec_temporal_realm_f3_summary.json`。

第 30 轮组合组的 DSEC 几何评估覆盖四条留出序列各 24 对。
匹配/全零事件的深度 AbsRel 为 0.1718/0.4254，点图相对误差为
0.2309/0.4482，旋转误差为 0.8759°/0.2770°。
统一尺度的确定方式与第 7 轮几何检查相同。
基础与 F³ 单独组在第 10 轮的同一批 36 对上，匹配深度 AbsRel 为
0.1737/0.1748，点图相对误差为 0.2342/0.2347，
相对平移方向误差为 17.94°/16.50°。当前对照没有显示 F³ 提升深度精度。

完成几何评估后，预测 Gaussian 组移至 GPU 0，与 GPU 1 的标定组并行。
预测组使用训练标签估计的全局初始尺度为 8.7053。
同输出路径的进程由文件锁串行，同参数下可复用已通过最终验证的运行结果。

两组 Gaussian head 都完成了 500 步，使用相同的 24 个留出窗口。

| 几何来源 | 匹配 PSNR dB | 全零 PSNR dB | 匹配 SSIM | 全零 SSIM | 匹配重叠 PSNR dB |
|---|---:|---:|---:|---:|---:|
| 标定与稀疏视差 | 10.160 | 9.564 | 0.523 | 0.478 | 12.969 |
| DVLT 学生预测 | 9.367 | 8.869 | 0.335 | 0.260 | 12.968 |

在标定的名义重叠区域内，两组 PSNR 接近。全图与 SSIM 仍有明显差异，
预测组的边缘覆盖和纹理细节较弱。定量结果和并排预览可在
`dsec_gs_final_comparison.json` 及各组的 `preview_000500.png` 查看。

预测几何组完成了 MVSEC 三条航线的 72 窗口跨域渲染。
匹配/全零事件的 PSNR 为 13.890/12.162 dB，SSIM 为 0.360/0.311。
首帧灰度复制的 PSNR 为 24.399 dB，目标帧平均平移 0.0274 m。
完整结果见 `mvsec_dsec_gaussians_summary.json`。

两组最终 checkpoint 均已核对为第 500 步，参数有限。
预测组正式权重从 DSEC 留出序列的两个事件包导出了
`dsec_gs_predicted/heldout_frame42_scene.npz`，包含 614,400 个高斯。
渲染对照位于同目录的 `heldout_frame42_render.png`。

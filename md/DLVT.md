# **DVLT：Déjà View（Looping Transformer）**

**DVLT（Déjà View Looping Transformer）是 NVIDIA 于 2026 年提出的轻量级、多视图、Pose-free 前馈三维重建模型。**

它可以理解为 VGGT / DA3 的一种架构替代方案：**不再堆叠大量不同参数的 Transformer Blocks，而是重复调用同一个共享 Block，逐步完善多视图几何特征。**

论文：[Déjà View: Looping Transformers for Multi-View 3D Reconstruction](https://arxiv.org/abs/2605.30215?utm_source=chatgpt.com)

代码：[NVIDIA 官方 GitHub](https://github.com/nv-tlabs/dvlt?utm_source=chatgpt.com)

预训练权重：[nvidia/dvlt](https://huggingface.co/nvidia/dvlt?utm_source=chatgpt.com)

## **1. DVLT 的输入与输出**

**Multi-view RGB Images**

[B, N, 3, H, W]，无需输入相机位姿

**DINOv2 Image Encoder**

**Shared Multi-view Transformer**

循环 K 次

逐步更新所有视图的 Tokens

**Depth / Ray / Camera Heads**

**Depth + Camera Pose + 3D Points**

输入多个没有已知位姿的 RGB 视图，DVLT 一次模型调用即可预测相机和稠密三维几何，不需要像 DUSt3R 那样先逐对预测再执行全局对齐。

arXiv+1

## **2. DVLT 的核心：循环 Transformer**

传统 VGGT 使用不同的网络层：

\[ z_{k+1}=F_{\theta_k}(z_k) \]

DVLT 使用共享参数：

\[ z_{k+1}=z_k+g_kF_\theta(z_k) \]

其中 \(g_k\) 表示控制每次更新的门控；这是便于理解的简化形式。

即使循环 16 次，它仍然只存储一份共享 Transformer 参数。

这并不是简单减少层数，而是通过反复处理同一组多视图特征，逐步完善几何关系。论文发现，在相同训练数据与计算预算下，共享参数的循环版本也优于对应的不共享参数版本。

arXiv

## **3. DVLT 和 VGGT / DA3 比较**

|特性|VGGT|DA3|DVLT|
|---|---|---|---|
|输入|多视图 RGB|多视图 RGB|多视图 RGB|
|相机位姿输入|不需要|视具体模式|不需要|
|原生多视图|是|是|是|
|几何输出|Depth、Point、Camera|Depth、Ray、Camera 等|Depth、Ray、Camera|
|Transformer|多层不同参数|多层不同参数|**共享 Block 循环**|
|参数规模|约 1B|随模型版本变化|**约 117M**|
|推理循环次数|固定架构|固定架构|**可调 K**|

DVLT 官方报告，它在五个涵盖室内、室外、物体和驾驶场景的测试基准上，以大幅减少的参数量取得了与更大模型竞争的结果。

NVIDIA

## **4. DVLT 与 Déjà-Events 是什么关系？**

这两者不能混为一谈。

- **DVLT：** 原始 RGB 多视图几何模型。
    
- **Déjà-Events：** 在 DVLT 基础上适配 Event 的模型。
    

Déjà-Events 的核心做法是冻结 RGB DVLT 教师，让 Event Student 的最终 Tokens 对齐 RGB 教师，并加入根据事件稀疏程度调整特征更新的 SAAG 门控。

因此，Déjà-Events 并不是重新从头训练一套三维几何模型。

arXiv

## **5. 对你的 Event-only 前馈 3DGS，有什么价值？**

我认为 DVLT 有两个值得关注的地方。

**第一，作为 Event 几何 Backbone。**

\[ \text{Multi-view Events} \rightarrow \text{Event Encoder} \rightarrow \text{DVLT} \rightarrow \text{Depth + Camera + Points} \]

然后连接 Gaussian Head：

\[ \text{Points + Event Features} \rightarrow \text{Gray Gaussians} \]

**第二，循环更新机制可以扩展到 3DGS。**

例如第 \(k\) 次循环：

\[ z^{k+1}=F_\theta(z^k,E_{1:N}) \]

根据更新后的特征预测更准确的 Point Maps，最终解码 Gaussian 参数。

但原版 DVLT 不是 3DGS 模型，也没有证明循环更新会自动提高 Gaussian 重建质量；这需要额外设计和实验。

### **我的建议**

既然 Déjà-Events 已经提供了 Event → DVLT 几何适配的研究实例，**现在不一定需要再训练一个 F³ + VGGT 来完成完全相同的几何任务**。

更有价值的是验证：

\[ \boxed{ \text{Déjà-Events Geometry} + \text{Event Features} \rightarrow \text{Feed-forward Gray 3DGS} } \]

在你的 4090 上，可以先用公开的 RGB DVLT 权重测试多视图推理、Depth/Camera/Point 的质量和显存，再决定是否需要适配 Event。

**一个需要特别区分的点：DVLT 的循环次数可调，但它仍然是前馈推理，不是逐场景 3DGS 优化。** 它迭代的是网络内部的特征状态，而不是针对每个新场景单独训练参数。
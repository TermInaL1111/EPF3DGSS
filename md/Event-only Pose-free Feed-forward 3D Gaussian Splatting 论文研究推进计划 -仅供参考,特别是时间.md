# Event-only Pose-free Feed-forward 3D Gaussian Splatting 论文研究推进计划

**研究周期：** 16 周（建议）

**目标投稿：** CCF A/B 类计算机视觉、人工智能相关会议

**计算资源：** 本轮在 A40 服务器上仅使用 GPU 0、1，单卡显存为 48GB。

**研究目标：** 利用多视图异步事件流，在不依赖推理期相机位姿、不执行逐场景优化的条件下，通过一次前馈网络预测场景几何及灰度 3D Gaussians，实现高质量的新视角合成。

---

## 一、研究背景与问题定义

### 1.1 研究背景

目前，VGGT、Depth Anything 3、DVLT 等 RGB 多视图几何基础模型，可以直接从多张图像预测相机参数、深度和三维点云，为前馈 3D Gaussian Splatting 提供可靠的三维几何初始化。

但事件相机具有异步触发、稀疏采样、时间分辨率高和运动依赖性强等特点，其数据形式与 RGB 图像存在明显差异，难以直接复用 RGB 基础模型。

现有研究包括：

- **F³：** 通过事件预测任务学习事件时空表示。
    
- **REALM：** 使用跨模态特征蒸馏，将事件特征迁移至预训练 RGB 视觉模型的表征空间。
    
- **Déjà-Events：** 通过 RGB DVLT 教师蒸馏与稀疏感知门控，实现原生多视图事件几何预测。
    
- **EvGGS：** 利用事件观测实现前馈 Gaussian 重建。
    
- **GradSplat：** 利用网络内部的渲染梯度反馈改善前馈 Gaussian 的参数预测。
    

这些工作分别解决了事件编码、跨模态迁移、多视图几何和 Gaussian 参数估计等问题，但尚不能直接代替本课题希望构建的完整 Event-only、Pose-free、灰度前馈 3DGS 系统。

### 1.2 核心研究问题

本研究重点解决两个问题：

**问题一：** 如何结合 F³ 的事件时间特征与 RGB DVLT 的多视图几何知识，使事件几何模型既具备跨视图几何预测能力，又保留对后续 Gaussian 重建有用的原生事件信息？

**问题二：** 如何融合多个事件视图的几何、运动与稀疏观测信息，使预测的三维 Gaussians 具有更好的几何一致性与灰度新视角合成质量？

### 1.3 最终任务定义

输入：

[  
E_{1:N}={E_1,E_2,\ldots,E_N}  
]

其中，每个 (E_i) 为围绕参考时间戳构建的事件观测窗口。

输出：

[  
\hat{\mathcal G}={\mu_k,\Sigma_k,\alpha_k,g_k}_{k=1}^{M}  
]

其中：

- (\mu_k)：Gaussian 三维中心；
    
- (\Sigma_k)：三维协方差；
    
- (\alpha_k)：不透明度；
    
- (g_k)：灰度外观参数。
    

网络同时预测相机参数与场景几何，最终使用可微 Gaussian Rasterization 合成指定新视角的灰度图像。

推理时不输入真实 RGB、Depth GT 或 Camera Pose GT。

需要说明的是，新视角渲染仍需要指定目标相机视角；Pose-free 指的是无需提供**输入观测的真实相机外参**，不等于不需要定义目标视角。

---

## 二、总体技术路线

### 2.1 设计框架

**阶段 A：事件多视图几何学习**

Multi-view Events  
↓  
F³ Event Encoder  
↓  
Event-to-DVLT Adapter  
↓  
DVLT Student + REALM-style Masked Distillation  
↓  
Depth / Camera / Point Maps

**阶段 B：事件感知 Gaussian 生成**

DVLT Geometry Features + F³ Temporal Features  
↓  
3D Geometry Lifting  
↓  
Event-aware Multi-view Feature Fusion  
↓  
Gaussian Parameter Head  
↓  
Gray 3DGS

**阶段 C：灰度重建和联合训练**

Predicted Gray Gaussians  
↓  
Differentiable Rendering  
↓  
Grayscale Reconstruction Loss  
↓  
可选 Event Temporal Consistency Loss

### 2.2 计划中的论文贡献

**贡献一：面向前馈三维重建的事件几何–时间双分支表示。**

以 F³ 和 DVLT 为基础，利用跨模态知识蒸馏获得多视图几何特征，同时保留未被 RGB Teacher 强制对齐的原生事件时间特征，为 Gaussian 外观和几何预测提供联合信息。

这属于待实验验证的方法设计；F³、LoRA 和 Masked Distillation 本身不作为新的算法贡献。

**贡献二：事件感知的多视图三维 Gaussian 融合机制。**

根据多视图几何一致性、可见性以及事件时空统计信息，对来自不同视图的三维特征进行自适应加权，预测统一坐标系下的灰度 Gaussians。

该模块为重点研究内容。

**可选贡献三：事件时空一致性约束。**

利用多时间戳灰度渲染与原始事件的亮度变化关系，辅助训练 Gaussian 几何及外观属性。

只有在独立实验中证明显著收益后，才将其写入论文主要贡献。

---

## 三、阶段一：数据准备和基础环境（第 1–2 周）

### 3.1 数据集选择

**主训练数据：DSEC；辅助验证与基线：EV3DS、MVSEC**

不再使用 TartanAir2。先复现 EV3DS 基线，再接入 DSEC 真实事件与图像进行适配；MVSEC 用于独立跨域评测。DSEC 的事件相机与图像相机需按标定对齐，不可直接按像素配对。深度、位姿或标定缺失时禁用相应监督，不把教师预测当成真值。按序列划分训练与验证，避免相邻帧泄漏。

本服务器最多使用两张 A40（GPU 0、1），所有 GPU 任务必须显式设置 CUDA_VISIBLE_DEVICES。

训练所需的数据包括：

|数据|用途|
|---|---|
|Event Stream|网络输入|
|同步 RGB|DVLT RGB Teacher 蒸馏|
|Depth GT|几何评测及可选监督|
|Camera Pose GT|训练期评测、坐标对齐和消融|
|Camera Intrinsics|三维反投影与渲染|
|RGB 转换的 Gray GT|灰度新视角重建监督|

后期选择具备可靠标定与必要评测真值的真实事件数据集，用于跨域验证。

### 3.2 数据处理

1. 完成事件、RGB、Depth、Pose 的时间同步。
    
2. 对连续事件流进行时间窗口划分。
    
3. 每个样本构造 2–4 个事件视图。
    
4. 将 RGB、Depth 与事件坐标系统一。
    
5. 生成训练与验证所需的相机内外参。
    
6. 生成 Gray GT。
    
7. 按场景和轨迹划分训练、验证及测试集。
    

初始实验采用短时间窗口，长度以 F³ 预训练模型配置为依据。

同时记录每个窗口内的相机位移和旋转量，防止窗口运动过大引入几何模糊。

### 3.3 环境搭建

完成以下模型的安装和推理验证：

- F³ Event Encoder；
    
- 原始 RGB DVLT；
    
- DVLT Camera / Depth / Point Map 输出；
    
- 可微 Gaussian Rasterizer；
    
- 灰度 PSNR、SSIM、LPIPS 评测代码。
    

### 本阶段交付成果

- 标准化 Event Multi-view Dataset。
    
- 可重复的数据加载脚本。
    
- RGB DVLT 几何结果。
    
- F³ Event Feature 输出。
    
- 数据同步与坐标系检查报告。
    

**阶段验收标准：** 能从同一轨迹正确读取两个或多个 Event 视图及其对应真值，并验证 RGB DVLT 的几何输出在该数据上合理。

---

## 四、阶段二：F³ 与 DVLT 跨模态适配（第 3–5 周）

### 4.1 模型实现

首先冻结 F³，提取事件特征：

[  
F_E=\operatorname{F^3}(E)  
]

通过 Event Adapter 调整特征维度和空间尺寸：

[  
Z_E=A_\theta(F_E)  
]

将其输入 DVLT Student：

[  
H_E=\operatorname{DVLT}_{Event}(Z_E)  
]

RGB Teacher：

[  
H_R=\operatorname{DVLT}_{RGB}(I)  
]

训练时主要更新：

- Event Adapter；
    
- DVLT 中的 LoRA；
    
- 必要的输入归一化或事件门控参数。
    

其他预训练模块优先冻结。

### 4.2 训练策略

**实验 1：基础蒸馏**

采用 Déjà-Events 式的教师最终 Token 对齐：

[  
\mathcal L_{\mathrm{distill}}  
=d(H_E,H_R)  
]

**实验 2：REALM 式掩码蒸馏**

在基础蒸馏中加入空间特征掩码或事件 Token Masking，比较模型在事件稀疏区域的几何预测表现。

**实验 3：多视图中间特征蒸馏**

除最终 Tokens 外，选择少量 DVLT 循环阶段进行特征监督，验证是否改善多视图几何一致性。

这里应注意，DVLT 共享循环 Block，不应直接套用普通 Transformer 的逐层监督规则，需要按循环阶段对齐。

### 4.3 几何评测

重点比较：

- Relative Pose Rotation Error；
    
- Translation Direction Error；
    
- Depth AbsRel / RMSE；
    
- Point Map Error；
    
- Cross-view Reprojection Error。
    

对照模型至少包含：

1. 原始 EventVGGT（权重可用时）。
    
2. Déjà-Events 原方法或尽可能接近的复现。
    
3. F³ + DVLT 基础蒸馏。
    
4. F³ + DVLT + REALM 式蒸馏。
    

如果 Déjà-Events 的完整代码或权重未发布，需清晰区分官方结果与自行复现结果。

### 本阶段交付成果

- Event-to-DVLT Adapter。
    
- 蒸馏训练代码。
    
- Event 多视图 Depth、Camera、Point Map 预测模型。
    
- 不同蒸馏策略的对照结果。
    

**阶段验收标准：** 在未见轨迹上，Event Student 能产生稳定的相机和点云预测，并且能与公开事件几何基线进行定量比较。

---

## 五、阶段三：构建基础 Gray Gaussian Head（第 6–7 周）

此阶段的重点不是设计新模块，而是先验证完整的 Event → 3DGS 流程。

### 5.1 Gaussian 几何初始化

使用 DVLT 输出的 Depth / Point Maps 构建 Gaussian 中心：

[  
\mu_i(u)=P_i(u)  
]

Gaussian 其他参数由轻量 MLP 预测：

[  
(s_i,q_i,\alpha_i,g_i)=H_\theta(F_i,P_i)  
]

初始实现采用单通道灰度 Gaussian，不使用 VAE/RAE 或大规模扩散模型。

### 5.2 训练策略

冻结 F³ 与 DVLT，首先仅训练 Gaussian Head。

使用 Gray GT 渲染监督：

\lambda_1\mathcal L_{L1}  
+\lambda_s\mathcal L_{\mathrm{SSIM}}  
]

建议采用两组实验：

**实验 A：GT Geometry + Gaussian Head**

使用训练期真实 Depth/Pose 构建几何，确定 Gaussian Head 和 Rasterizer 的性能上限。

**实验 B：Predicted Geometry + Gaussian Head**

使用 Event DVLT 的预测 Depth/Pose/Point Maps，评估实际 Event-only 推理效果。

两组都应在未参与训练的新视角上评价。

### 本阶段交付成果

- 可运行的 Gray Gaussian Head。
    
- Gray 3DGS Rasterizer。
    
- Event-only、Pose-free 前馈 3DGS 基线。
    
- GT Geometry 与 Predicted Geometry 对照结果。
    

**阶段验收标准：** 能输入多视图 Event，直接预测统一三维场景并渲染新视角灰度图；无需针对测试场景训练参数或进行 Bundle Adjustment。

---

## 六、阶段四：核心创新——Event-aware Gaussian Fusion（第 8–10 周）

此阶段为论文最主要的方法研究。

### 6.1 问题

不同事件视图对同一三维表面的观察质量存在差异。

例如：

- 部分视图产生丰富事件；
    
- 部分视图事件稀疏；
    
- 部分视图的几何预测不稳定；
    
- 部分视图存在遮挡；
    
- 不同时间窗口内的事件响应受相机运动影响。
    

普通的平均融合不能区分这些情况。

### 6.2 研究方法

首先使用 DVLT 预测的三维点，将多个视图的 F³ 特征映射到同一三维空间。

对于三维位置 (x)，收集：

[  
{f_i(x)}_{i\in\mathcal V(x)}  
]

构建融合权重：

[  
a_i(x)=h_\theta\left(  
f_i(x),c_i(x),\rho_i(x),v_i(x)  
\right)  
]

其中：

- (f_i)：事件特征；
    
- (c_i)：几何一致性或预测置信度；
    
- (\rho_i)：事件活动和时间统计；
    
- (v_i)：可见性信息。
    

然后：

[  
w_i(x)=\operatorname{Softmax}_i(a_i(x))  
]

[  
F_{3D}(x)=\sum_iw_i(x)f_i(x)  
]

最后：

[  
G(x)=H_{\mathrm{GS}}(x,F_{3D}(x))  
]

### 6.3 必须完成的消融

|实验|Fusion 方法|
|---|---|
|F0|无跨视图融合，直接预测并合并 Gaussians|
|F1|Mean Pooling|
|F2|普通 Attention Fusion|
|F3|Geometry Confidence Fusion|
|F4|Event Statistics Fusion|
|F5|**完整 Event-aware Geometry Fusion**|

特别注意：事件稀疏不一定代表错误观测，事件密度应与几何可靠性联合建模，而不是直接作为置信度。

### 本阶段交付成果

- Event-aware Gaussian Fusion 模块。
    
- 完整消融结果。
    
- 不同事件密度和运动条件下的可视化。
    
- 与普通 Gaussian Head 的定量对比。
    

**阶段验收标准：** 在相同几何主干和训练条件下，新模块对新视角灰度重建具有稳定收益，并且不是简单依靠增加模型参数或计算量取得的改善。

---

## 七、阶段五：事件时空监督与联合优化（第 11–12 周）

这一阶段属于可选增强。若阶段四已有足够明确的论文贡献，应优先完善实验，而不是继续增加复杂模块。

### 7.1 事件一致性监督

同一组 Gaussians 在多个时间戳下渲染：

[  
\hat I(t_k)=\mathcal R(G,T(t_k))  
]

计算相邻时刻对数亮度变化：

\log\hat I(t_{k+1})  
-\log\hat I(t_k)  
]

与对应区间的事件累计量或可微模拟事件比较：

\sum_k\rho(  
\Delta\hat L_k-CS_k  
)  
]

该公式作为初步近似，后续需要评估事件阈值、极性、时间积分和传感器噪声的影响。

### 7.2 训练策略

首先冻结 DVLT：

[  
\mathcal L=  
\mathcal L_{\mathrm{gray}}  
+  
\lambda_e\mathcal L_{\mathrm{event}}  
]

验证事件监督是否改善灰度 NVS。

如果有效，再尝试解冻部分 DVLT LoRA，进行小学习率联合训练。

### 7.3 研究边界

不强制复现 GradSplat 的完整梯度更新网络。

优先验证简单事件时空约束；只有在有明确收益和足够训练资源时，再研究学习式 Gaussian Refinement。

### 本阶段交付成果

- Event Temporal Loss 训练代码。
    
- 有无 Event Loss 对照结果。
    
- 不同运动速度条件下的误差分析。
    

**阶段验收标准：** 验证事件时空监督是否产生独立收益。如果效果不稳定，将其保留为补充实验，而不强行作为主要贡献。

---

## 八、阶段六：完整实验与论文撰写（第 13–16 周）

### 8.1 主实验

重点评价：

**新视角合成质量**

- 灰度 PSNR；
    
- SSIM；
    
- LPIPS；
    
- 不同视角间隔的 NVS 表现。
    

**几何质量**

- Depth AbsRel / RMSE；
    
- Relative Pose Error；
    
- Point Map Error；
    
- 跨视图重投影误差。
    

**计算效率**

- 总参数量；
    
- 可训练参数量；
    
- 峰值显存；
    
- 推理时间；
    
- 不同输入视图数量下的性能。
    

### 8.2 对比方法

根据代码与数据协议可用性，选择：

- EvGGS；
    
- EventVGGT + 相同 Gaussian Head；
    
- Déjà-Events + 相同 Gaussian Head；
    
- Event3R + 几何/GS 转换基线；
    
- Event → RGB 重建 → RGB 几何模型 → 3DGS；
    
- 本文提出的完整模型。
    

所有基线必须明确说明是否使用真实输入 Pose、Depth GT、训练期 RGB、测试时优化等额外信息。

对于无法在相同协议下运行的方法，不直接比较 PSNR 排名，而是单独报告原论文结果及协议差异。

### 8.3 论文结构

**1. Introduction**

事件相机的优势、传统事件重建限制、现有多视图几何方法与前馈 3DGS 之间的空缺。

**2. Related Work**

Event Representation、Event–RGB Distillation、Event Multi-view Geometry、Feed-forward 3DGS。

**3. Method**

3.1 Overview  
3.2 Event Geometry Representation  
3.3 Cross-modal Geometry Distillation  
3.4 Event-aware Gaussian Fusion  
3.5 Rendering and Training Objectives

**4. Experiments**

4.1 Datasets and Implementation Details  
4.2 Comparison with Existing Methods  
4.3 Ablation Studies  
4.4 Efficiency and Generalization  
4.5 Limitations

**5. Conclusion**

总结事件感知多视图几何融合在 Pose-free、前馈灰度 3DGS 中的作用。

---

## 九、16 周时间安排

|周次|工作内容|关键交付|
|---|---|---|
|W1|环境搭建、数据集同步|可运行数据管线|
|W2|F³、RGB DVLT 推理|特征与几何基线|
|W3|Event Adapter + 基础蒸馏|Event-DVLT 初始模型|
|W4|REALM 式掩码蒸馏|蒸馏策略对照|
|W5|多视图几何评测|Depth/Pose/Point 实验|
|W6|Gray GS Head、GT 几何上限|灰度 Gaussian 基线|
|W7|预测几何接入 3DGS|完整前馈系统|
|W8|三维特征融合|多视图融合模型|
|W9|Event-aware 权重设计|核心模块初版|
|W10|Fusion 消融与调参|主要创新实验|
|W11|Event Temporal Loss|时空监督实验|
|W12|小规模联合微调|完整模型|
|W13|多场景基准与强基线|主对比表|
|W14|泛化、效率与可视化|完整实验结果|
|W15|论文初稿与图表|完整论文|
|W16|补充实验、修改、投稿准备|投稿版本|

---

## 十、算力和风险控制

### 10.1 计算资源

**1×4090：**

主要完成数据准备、2–3 视图蒸馏、小型 Gray GS Head、初步融合实验。

采用 BF16、冻结特征缓存、梯度累积等方式控制显存。

**本轮 A40 服务器（最多两张卡）：**

当前以单卡运行单个实验，第二张卡并行运行对照或独立评估。更多视图和联合训练按两卡上限规划。

具体训练耗时需要通过 Step/s 实测确定，不能直接用论文的少量可训练参数推断 GPU 时间。

### 10.2 主要技术风险

|风险|影响|应对策略|
|---|---|---|
|F³ 与 DVLT 特征接口不匹配|蒸馏难以收敛|先测试简单 Adapter 与基础教师蒸馏|
|Event Student 几何精度不足|Gaussian 初始化错误|使用 GT Geometry 建立上限，分析误差来源|
|事件缺乏绝对亮度|灰度恢复存在歧义|训练期加入 Gray GT，统一亮度评测协议|
|多视图 Gaussian 重复或错位|浮点噪声与重影|引入几何一致性、可见性与空间融合|
|Event Fusion 收益不明显|方法贡献不足|优先分析视图置信度和几何误差|
|Event Temporal Loss 不稳定|联合训练退化|保持为可选模块，先冻结几何|
|训练成本过高|无法完成全部实验|控制视图数量、特征尺寸和模块规模|

---

## 十一、阶段性决策条件

**第 5 周：几何是否可用？**

如果 F³ + DVLT 无法稳定预测多视图几何，不应继续投入大量蒸馏成本。可切换到已有 EventVGGT 等可用几何模型，先完成 Gaussian 研究。

**第 7 周：完整 3DGS 是否可用？**

如果 GT Geometry 下能够成功重建，而 Predicted Geometry 下效果很差，优先修复相机坐标和点云一致性问题。

如果 GT Geometry 下也无法完成灰度重建，则优先检查 Gaussian Head、属性预测和渲染损失。

**第 10 周：核心创新是否有效？**

如果 Event-aware Fusion 相对普通融合没有稳定收益，应缩小研究范围，重新检查事件时空特征是否提供了与几何特征不同的信息。

**第 12 周：是否增加 Refinement？**

只有在完整模型已收敛、主要消融充分的情况下，才加入复杂的 Event-guided Refinement。

---

## 十二、最终预期成果

### 基础成果

实现一个完整的 Event-only、Pose-free、Feed-forward 灰度 3DGS 重建系统。

### 方法成果

提出面向前馈 Gaussian 重建的事件几何–时间信息融合方法，并通过消融实验验证其贡献。

### 实验成果

形成跨场景灰度 NVS、几何精度、效率、输入视图数量及事件稀疏性分析等完整实验。

### 论文成果

完成一篇以 **Event-aware Multi-view Gaussian Reconstruction** 为核心的研究论文，根据实验质量与会议征稿范围选择合适的 CCF A/B 投稿目标。

论文暂定题目：

**EventSplat: Event-aware Multi-view Geometric Fusion for Pose-free Feed-forward 3D Gaussian Splatting**

该名称仅为项目暂定名称，正式投稿前需要检查是否与已有论文重名。

---

## 十三、工作优先级

**P0：必须完成**

- 可靠的多视图 Event 几何预测；
    
- 简单 Gray Gaussian Head；
    
- Pose-free 前馈重建；
    
- GT 与预测几何对照；
    
- 基础 NVS 和几何评测。
    

**P1：论文重点**

- F³ 与 DVLT 的有效适配；
    
- REALM 式蒸馏消融；
    
- Event-aware Gaussian Fusion；
    
- 跨场景主实验。
    

**P2：有余力再做**

- 连续时间 Event Loss；
    
- Gaussian Gradient Refinement；
    
- 复杂 Latent 3DGS；
    
- 大规模联合训练。
    

**总体原则：先形成可投稿的完整基础系统，再以可测量的方式增加方法创新，避免多个模块同时开发导致无法判断性能收益来源。**
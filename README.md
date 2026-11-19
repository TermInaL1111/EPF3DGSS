# EPF3DGS

研究目标是用训练期 RGB 视觉基础模型先验，从多视图事件包一次预测场景高斯与相对位姿，支持灰度新视角渲染。原始设计见 md/方案.md。

## 当前主线：Déjà-Events 风格的事件 DVLT 学生接 Gaussian head

训练时，同场景、同源帧编号的 RGB 对送入冻结 DVLT 教师。事件体素与 F³ 特征进入事件适配的 DVLT 学生。学生保留 DINOv2、帧内及跨帧循环 Transformer、深度与射线解码器；用 LoRA、依赖事件密度的门控与 REALM 思路的渐进掩码进行 token 蒸馏。新的 GS head 读取**学生自己的几何 tokens、深度、相机预测**和事件细节。推理入口为 reconstruct.py。EvGGS 仅用于外部对照。

| 环节 | 当前实验设置 |
|---|---|
| 输入 | 两包原始事件，原传感器 640×480 |
| 学生 DVLT | 224×168，14 像素 patch，每视图 16×12 个空间 token，12 次共享循环 |
| Gaussian head | 每源视图 640×480 个候选；双视图合计 614,400 个，预测深度残差、三轴尺度、旋转、灰度和不透明度 |
| 训练 | 冻结的 RGB DVLT 教师监督最终 tokens，先蒸馏后加入渲染损失 |
| 位姿 | 学生利用自身射线预测恢复源相机；目标相机仅用于指定渲染视角 |

event_dvlt.py 实现推理骨干与读出头；train_event_dvlt.py 实现训练与空事件对照。以下旧版实验使用小卷积学生，用于方法对照；旧权重 epf3dgs_ev3ds_pilot.pt 对应旧学生。原方案见 md/方案.md。

主线学生的增量权重见 [event_dvlt_gs.pt](weights/event_dvlt_gs.pt)，搭配原始 DVLT 与 F³ 权重使用。两包输入均为形状 (4,N) 的 x、y、t、p 数组。当前配置的复现命令如下。

```bash
assets=/cloud/cloud-ssd4/epf-assets
data=/cloud/cloud-ssd1/datasets/EV3DS
export PYTHONPATH=/cloud/cloud-ssd4/dvlt/src
python train_event_dvlt.py "$data" "$assets/f3_dense.pt" "$assets" "$assets/event_dvlt_v1"
python reconstruct.py "$assets" weights/event_dvlt_gs.pt /cloud/cloud-ssd4/fast-feature-fields "$assets/f3.pth" "$data/Event/Arms/0004.npy" "$data/Event/Arms/0008.npy" scene.npz
```

每包输入是同一运动相机、一个源时刻附近的 (x,y,t,p) 事件列表。每个包同时包含两种极性。464 个源帧的内容配对核查中，463 个 NPY 包与同编号 RGB 变化的匹配度最高，Towel/0043 更接近下一帧。NPY 与 H5 使用不同时间数值。逐帧核查见 [sync.json](results/event_dvlt/sync.json)。

逐窗口结果见 [EV3DS 和 MVSEC 实验](results/event_dvlt/)，结果解读见[论文工作稿](paper/manuscript.md)。

384 步对照共享初始化、事件掩码和训练窗口。两组从第一步开始使用渲染损失，分别设置 token 蒸馏权重为 1 和 0。下面的评估覆盖全部 17 个验证物体，输入包括匹配事件、同一帧号的下一物体事件和全零输入。

```bash
for weight in 1 0; do
  python train_event_dvlt.py "$data" "$assets/f3_dense.pt" "$assets" "$assets/dvlt_joint384_w$weight" --warmup 0 --steps 384 --token-weight "$weight"
done
python evaluate_dvlt.py "$data" "$assets" /cloud/cloud-ssd4/fast-feature-fields "$assets/dvlt_joint384_w1/event_dvlt_gs.pt" "$assets/dvlt_joint384_w0/event_dvlt_gs.pt" full_validation.json event_conditioning.png
python sync_audit.py "$data" sync.json
```

<details>
<summary>早期卷积学生实验归档</summary>

## EV3DS 配对先导实验

学生的 forward 接口仅接收两个事件包。当前原型固定使用 EV3DS 标定，每包按时间二分并分正负极性。共享卷积编码器和双视图融合头预测相对旋转、平移，以及每个源视图的 48 个各向同性高斯。渲染器通过透视投影雅可比计算屏幕协方差，按深度进行透明度合成。训练有相对位姿监督。目标相机只传给渲染器。当前实现将首源视图定义为场景坐标系。

实验固定数据、初始化、批次次序和训练步数，仅改变特征蒸馏损失权重。RGB 教师来自 DVLT 发布权重中的 DINOv2 图像编码器，用 timm 加载并在低分辨率上抽取最终 patch 特征。多视图循环几何教师、F³ 和事件感知融合属于后续研究模块。完整模型与权重说明见 [DVLT 官方仓库](https://github.com/nv-tlabs/dvlt)。

运行需要 CUDA PyTorch、timm 1.0.31、Pillow、numpy 和 safetensors。EV3DS 根目录应含 Scenes、Event、Poses 和场景划分文件。下列命令下载固定版本的教师并运行训练、验证和输入置换诊断。

```bash
assets=/cloud/cloud-ssd4/epf-assets
data=/cloud/cloud-ssd1/datasets/EV3DS
mkdir -p "$assets"
curl -fL https://huggingface.co/nvidia/dvlt/resolve/c4e33d824c3f997d39bb9cbbde4e2cc4f1dd8846/model.safetensors -o "$assets/dvlt.safetensors"
OMP_NUM_THREADS=2 python pilot.py "$data" "$assets/dvlt.safetensors" "$assets/run10"
OMP_NUM_THREADS=2 python audit.py "$data" "$assets/run10"
```

同一场景的源帧为 i 和 i+4，目标帧为 i+2，事件取对应源帧之前的时间窗。每个场景取八个三元组。验证指标为逐帧灰度 PSNR，先在场景内平均，再对场景等权平均。此协议衡量合成事件、共享相机轨迹下的插值。外推、新轨迹和真实传感器需要独立评估。

前景加权实验在 pilot.py 后加上 `--foreground` 并使用独立输出目录。前景由目标帧深度大于零的像素定义，训练期和评估期按下采样像素覆盖率加权。audit.py 会同时测量匹配、跨场景置换、全零事件的全图和前景指标。论文工作稿见 [paper/manuscript.md](paper/manuscript.md)，逐次实验记录见 [results](results/)。

## 多视图教师和事件场验证

教师实现固定为 nv-tlabs/dvlt 的 d9f1c0c 和 grasp-lyrl/fast-feature-fields 的 833e2c6。geometry_teacher.py 使用发布配置严格加载 RGB 教师，保留十二次循环，在同一次前向传播中提取循环前后的源视图特征。输入为 448×336。特征按相同方式池化到 8×6。另行缓存两源视图的深度。教师位姿通过官方的射线拟合得到。验证场景参与教师质量审计，学生训练缓存仅含训练场景。

F³ 前端使用官方小模型，原始坐标在 1280×720 训练画布居中填充。每包时间映射至二十个相对年龄桶。该权重使用无极性的哈希编码；保留的四个计数通道为学生提供有符号事件信息。对中间 32 通道特征裁剪、面积池化并减去固定参考响应，然后与四通道基线拼接。空事件输入按零表示。源码和权重依照各自上游许可使用。

```bash
a=/cloud/cloud-ssd4/epf-assets
d=/cloud/cloud-ssd1/datasets/EV3DS
curl -fL https://huggingface.co/nvidia/dvlt/raw/c4e33d824c3f997d39bb9cbbde4e2cc4f1dd8846/config.json -o "$a/config.json"
curl -fL https://richeek-penn.s3.amazonaws.com/prolev/ff/patchff_fullcardaym3ed_small_20ms.pth -o "$a/f3.pth"
python event_field.py "$d" /cloud/cloud-ssd4/fast-feature-fields "$a/f3.pth" "$a/f3_features.pt"
PYTHONPATH=/cloud/cloud-ssd4/dvlt/src python geometry_teacher.py "$d" "$a" "$a/multiview"
python pilot.py "$d" "$a/dvlt.safetensors" "$a/loop10" --foreground --cached-features "$a/multiview/loop_features.pt" --event-fields "$a/f3_features.pt"
python audit.py "$d" "$a/loop10" --event-fields "$a/f3_features.pt"
python student_depth.py "$d" "$a/loop10" --event-fields "$a/f3_features.pt"
python infer.py "$a/loop10/student_0.1.pt" "$d/Event/Arms/0004.npy" "$d/Event/Arms/0008.npy" "$a/scene.npz" --f3-repository /cloud/cloud-ssd4/fast-feature-fields --f3-checkpoint "$a/f3.pth"
```

本机 DVLT 依赖装在 /cloud/cloud-ssd4/epf-env。运行 geometry_teacher.py 时使用此环境的 Python 和 `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`。训练和教师质量审计均在 RTX 4090 上执行。用 `encoder_features.pt` 替换循环特征可复现同分辨率单图对照。`--permute-teacher` 在相同时间索引间置换训练场景，用来检验教师和事件配对的作用。`--depth-targets "$a/multiview/depth_targets.pt" --depth-weighting event` 进一步对齐两源视图的相对对数深度，按局部事件计数加权；uniform 为对照。

baseline_shift.py 在 EV3DS 上改变源间距和顺序。mvsec_transfer.py 使用官方室内标定将真实 DAVIS 图像及事件投影到学生的固定视场，对三个室内飞行序列各取 24 个窗口。学生不使用 RGB 或外参；渲染和旋转误差评估使用插值后的参考位姿。source-copy 使用源灰度图，作为额外信息参照。

## 高密度 3DGS 与位姿校验

128×96 版本的 32×24 网格为每个源视图生成 768 个各向异性高斯。每个高斯预测旋转、三轴尺度、中心、不透明度与灰度。渲染使用 gsplat 1.5.3。训练蒸馏、事件加权深度损失和跨视图灰度融合可以独立消融。论文中的高密度融合对照只使用循环层特征监督。几何融合按预测深度一致性对重投影灰度加权，事件融合再乘上两个视图的局部事件支持度比值。

运行 event_field.py 加 `--scale 4` 写入单独的高密度缓存。运行 pilot.py 加 `--scale 4 --fusion event` 并传入该缓存。geometry 和 none 分别是只使用深度一致性和不融合的对照。在本机先启用 /cloud/cloud-ssd4/epf-env，并设置 `TORCH_EXTENSIONS_DIR=/cloud/cloud-ssd4/epf-assets/extensions`。infer.py 可从训练权重恢复输入密度和融合方式。

当前代码从每帧 NPZ 内的 Camera 对象读取位姿。results 下 scene_dense 开头的实验使用此协议。此前已推送的低分辨率实验使用公共 Poses 文本；复现这些记录请使用提交 1eada6c。pose_labels.py 对比了两种标注。验证集 Arms 第 36 帧的 NPZ 将相机变换记为单位矩阵，相邻帧均处于轨道上，需在位姿指标中披露该异常。全局文本和逐帧 NPZ 在绝对坐标上也可能不同。

原始分辨率 EvGGS 对照使用本地锁定的上游权重评估存档。evggs_comparison.py 根据场景、源帧、目标帧匹配，检查前景比例，并在 640×480 渲染后裁出与 EvGGS 一致的 512×480 区域。存档的上游版本、权重和逐样本文件摘要已写入 results/evggs_native_comparison.json。两者使用的事件包数、源位姿和背景监督不同。

论文工作稿可用安装了 pypandoc_binary 和 typst 的 Python 运行 `python paper/build.py` 生成。当前稿件记录负结果以及相机标注敏感性。[PDF](paper/manuscript.pdf)

仓库随附 weights/epf3dgs_ev3ds_pilot.pt，来自 scene_dense_event_10，含学生参数，不含 F³ 或 RGB 教师权重。使用 infer.py 的两个源事件文件即可生成 1,536 个高斯；提供官方 F³ 权重及源码路径。三种稠密融合方案的定量结果见 results/dense_summary.py。

</details>

# EPF3DGS

研究目标是用训练期 RGB 视觉基础模型先验，从多视图事件包一次预测场景高斯与相对位姿，支持灰度新视角渲染。原始设计见 md/方案.md。

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

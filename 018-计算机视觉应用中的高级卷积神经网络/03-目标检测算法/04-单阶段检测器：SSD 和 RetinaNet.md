# 单阶段检测器：SSD 和 RetinaNet

来源：[原文](https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-3-object-detection-algorithms/single-stage-detectors-ssd-retinanet)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管 Faster R-CNN 等两阶段检测器通过首先提出区域然后进行分类来实现高精度，但这种顺序过程可能会带来计算开销。单阶段检测器通过同时执行定位和分类来简化此过程，在网络的一次前向传播中直接从特征图中预测边界框和类别概率。这种架构选择通常会带来更快的推理 (inference)速度，使其适用于实时应用。两种有影响力的单阶段检测器是：单次多框检测器 (SSD) 和 RetinaNet。

### 单次多框检测器 (SSD)

SSD 通过在基础网络（如 VGG 或 ResNet）的一次前向传播中，从具有不同分辨率的多个特征图中进行预测，来应对检测不同尺度物体的挑战。较早的特征图（更接近输入端）具有更高的空间分辨率并捕获更精细的细节，使其适用于检测较小的物体。较晚的特征图分辨率较低但感受野较大，使其能够检测更大的物体。

**架构和多尺度特征图：**
SSD 从一个标准分类网络（主干网络）开始，该网络在最终分类层之前被截断。然后附加了几个辅助卷积层，这些层空间分辨率逐渐降低，同时通道深度增加。与仅从最终特征层预测的模型不同，SSD 从主干网络和辅助结构的各个阶段的选定特征图中生成预测。

对于每个选定的特征图，一组卷积滤波器会预测：

1. **边界框偏移量：** 相对于预定义默认框（类似于锚框）的调整。
2. **类别置信度分数：** 每个物体类别（包括背景类别）的概率。

**默认框（锚点）：**
在选定特征图上的每个位置，SSD 会关联一组具有不同宽高比和尺度的默认框。这些默认框平铺特征图，作为初始提议。网络会预测偏移量 ($\Delta cx, \Delta cy, \Delta w, \Delta h$) 来调整这些默认框的位置和大小，使其更好地与真实物体匹配，同时预测每个类别的置信度分数。默认框的尺度对于高分辨率特征图通常较小，对于低分辨率特征图则较大，从而使框的大小与该特征层预期的物体大小对齐 (alignment)。

**训练：**
训练期间，每个真实边界框都会与具有最高 Jaccard 交并比（IoU）的默认框匹配。与任何真实框的 IoU 大于某个阈值（例如 0.5）的默认框也被视为正匹配。所有其他默认框都被标记 (token)为负（背景）。损失函数 (loss function)是以下各项的加权和：

- **定位损失（例如，Smooth L1）：** 仅针对正匹配计算，惩罚预测框偏移中的误差。
- **置信度损失（例如，交叉熵）：** 对正负匹配都计算，惩罚分类错误。通常使用难负例挖掘来平衡大量负框。

SSD 在速度和精度之间取得了良好的平衡，通常比两阶段检测器明显更快。然而，由于它使用相对较低分辨率的特征图来检测较大物体，并严重依赖这些跨多尺度的预定义默认框，因此与那些更细致分析细节或拥有专门提议阶段的方法相比，它有时难以准确检测非常小的物体。

### RetinaNet：使用焦点损失解决类别不平衡问题

对于像 SSD 这样的密集型单阶段检测器，一个主要挑战是训练期间极端的类别不平衡。绝大多数默认框或锚点位置对应背景类别，而只有一小部分代表实际物体。应用于所有位置的标准交叉熵损失意味着那些容易分类的背景样本会共同主导损失值和梯度更新，阻碍网络为较稀有的前景物体类别学习有效的表示。

RetinaNet 引入了**焦点损失**，专门用于解决这种不平衡。它是一种动态加权的交叉熵损失，其加权因子随着对正确类别置信度的增加而衰减至零。直观来说，焦点损失在训练期间会自动降低易分类样本（通常是大量的背景负例）的贡献，并将模型的注意力集中在难以分类的样本（通常是前景物体或模糊的背景区域）上。

二分类的标准交叉熵（CE）损失可以写为：
$CE(p, y) = \begin{cases} -\log(p) & \text{如果 } y=1 \\ -\log(1-p) & \text{如果 } y=0 \end{cases}$
其中 $y \in \{0, 1\}$ 是真实类别，$p \in [0, 1]$ 是模型对类别 $y=1$ 的估计概率。我们可以更紧凑地将其重写为 $CE(p_t) = -\log(p_t)$，其中 $p_t$ 定义为：
$p_t = \begin{cases} p & \text{如果 } y=1 \\ 1-p & \text{如果 } y=0 \end{cases}$
$p_t$ 表示真实类别的概率。

焦点损失在标准交叉熵损失中增加了一个调制因子 $(1 - p_t)^\gamma$，带有一个可调的聚焦参数 (parameter) $\gamma \ge 0$：
$FL(p_t) = -(1 - p_t)^\gamma \log(p_t)$
可选地，也可以添加一个 $\alpha$ 平衡参数：
$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$
其中 $\alpha_t$ 对于类别 1 是 $\alpha$，对于类别 0 是 $1-\alpha$。

**焦点损失的特点：**

1. **降低易分类样本的权重 (weight)：** 当一个样本容易分类时（ $p_t \rightarrow 1$ ），调制因子 $(1 - p_t)^\gamma$ 趋近于 0，从而降低该样本的损失贡献。具有高置信度的易分类背景样本（ $p \rightarrow 0$，因此 $p_t \rightarrow 1$ ）的损失会显著降低。
2. **关注难分类样本：** 当一个样本被错误分类时（ $p_t$ 较小），调制因子趋近于 1，损失类似于标准交叉熵。这些难分类样本的相对贡献会增加。
3. **聚焦参数 $\gamma$：** 增加 $\gamma$ 会增强调制因子的效果。当 $\gamma = 0$ 时，焦点损失等同于标准交叉熵。经验上，在原始 RetinaNet 论文中，$\gamma=2$ 被发现效果良好。



[交互图表：焦点损失与交叉熵的对比](https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-3-object-detection-algorithms/single-stage-detectors-ssd-retinanet#plot-2xupq8)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        0.01,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.99
      ],
      "y": [
        4.605,
        2.303,
        1.609,
        1.204,
        0.916,
        0.693,
        0.511,
        0.357,
        0.223,
        0.105,
        0.01
      ],
      "mode": "lines",
      "name": "CE (γ=0)",
      "line": {
        "color": "#fa5252"
      }
    },
    {
      "x": [
        0.01,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.99
      ],
      "y": [
        2.258,
        0.933,
        0.515,
        0.295,
        0.167,
        0.087,
        0.041,
        0.016,
        0.005,
        0.001,
        0.0
      ],
      "mode": "lines",
      "name": "FL (γ=2)",
      "line": {
        "color": "#4263eb"
      }
    },
    {
      "x": [
        0.01,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.99
      ],
      "y": [
        1.082,
        0.219,
        0.066,
        0.021,
        0.007,
        0.002,
        0.0004,
        7e-05,
        5e-06,
        0.0,
        0.0
      ],
      "mode": "lines",
      "name": "FL (γ=5)",
      "line": {
        "color": "#12b886"
      }
    }
  ],
  "layout": {
    "title": {
      "text": "焦点损失与交叉熵的对比"
    },
    "xaxis": {
      "title": {
        "text": "正确类别的概率 (p_t)"
      }
    },
    "yaxis": {
      "title": {
        "text": "损失值"
      }
    },
    "legend": {
      "orientation": "h",
      "yanchor": "bottom",
      "y": 1.02,
      "xanchor": "right",
      "x": 1
    }
  }
}
```

</details>



> 该图表说明了焦点损失（FL）在 $\gamma > 0$ 时，如何与标准交叉熵（CE，相当于 $\gamma = 0$）相比，减少了已正确分类样本（高 $p_t$）的损失。更高的 $\gamma$ 值会增强这种效果，将训练集中在 $p_t$ 较低的难分类样本上。

**RetinaNet 架构：**
虽然焦点损失是主要贡献，但 RetinaNet 检测器本身通常采用基于 ResNet 等主干网络构建的特征金字塔网络（FPN）。FPN 生成一个在所有级别上都具有丰富语义的多尺度特征金字塔，提高了对各种尺度物体的检测能力，补充了焦点损失的效果。锚框应用于特征金字塔的每个级别进行预测。

通过使用焦点损失有效管理类别不平衡，RetinaNet 表现出单阶段检测器可以达到甚至超越 Faster R-CNN 等流行两阶段检测器的精度，同时保持更高的速度。

### SSD 和 RetinaNet 总结

SSD 和 RetinaNet 都体现了单阶段方法，直接预测框和类别，无需专门的区域提议步骤。

- **SSD** 引入了在单次网络传播中使用多个特征图进行多尺度检测的理念，提供了显著的速度优势。其主要局限性可能是精度，特别是对于小物体。
- **RetinaNet** 识别并通过新颖的**焦点损失**解决了密集单阶段检测器中固有的重要类别不平衡问题。结合 FPN 主干网络，它弥补了与两阶段方法的精度差距，同时保持了单阶段设计的效率优势。

在这些（以及 YOLO 等其他检测器）之间进行选择时，通常需要考虑特定应用在速度、精度、物体尺寸分布和场景复杂性方面的要求。RetinaNet 通常比 SSD 提供更高的精度，特别是在有挑战性的场景中，这得益于焦点损失以及通常使用的 FPN，而如果最高速度是绝对优先事项且其精度权衡可以接受，则 SSD 可能更受青睐。

## 参考资料

- [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) — Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg (2016)
  Journal: European Conference on Computer Vision (ECCV); Pages: 21-37; DOI: [10.1007/978-3-319-46448-0_2](https://doi.org/10.1007/978-3-319-46448-0_2)
  介绍单次多盒检测器 (SSD) 架构的原始论文，详细说明了其多尺度特征图、默认框和用于快速目标检测的训练策略。
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) — Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár (2017)
  Journal: IEEE International Conference on Computer Vision (ICCV); Pages: 2979-2987; DOI: [10.48550/arXiv.1708.02002](https://doi.org/10.48550/arXiv.1708.02002)
  本文介绍了 RetinaNet 这一单阶段检测器，并提出了 Focal Loss 函数，旨在解决密集目标检测中固有的严重类别不平衡问题。
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144) — Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie (2017)
  Journal: IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 936-944; DOI: [10.48550/arXiv.1612.03144](https://doi.org/10.48550/arXiv.1612.03144)
  描述了特征金字塔网络 (FPN) 架构，该架构从单分辨率输入构建多尺度特征金字塔，常与 RetinaNet 结合使用，以改善跨尺度检测。

---

[上一节](03-%E5%8D%95%E9%98%B6%E6%AE%B5%E6%A3%80%E6%B5%8B%E5%99%A8%EF%BC%9AYOLO%E7%B3%BB%E5%88%97.md) · [下一节](05-%E9%94%9A%E6%A1%86%EF%BC%9A%E8%AE%BE%E8%AE%A1%E4%B8%8E%E4%BC%98%E5%8C%96.md)

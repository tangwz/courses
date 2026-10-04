# CNN核心操作：卷积

来源：[原文](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-7-intro-specialized-architectures/cnn-convolution-operation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

如前所述，标准前馈网络独立处理输入特征，因此会丢失图像等数据中内含的空间关系。卷积神经网络 (neural network) (CNN)通过使用一种称为**卷积**的特殊操作作为其主要构件来解决此问题。此操作使网络能够识别和检测输入中的局部模式，从而维持空间层级。

### 滤波器（核）：一种特征检测器

卷积操作的中心是**滤波器**，也称为**核**。可以将滤波器视为一个小的、可学习的权重 (weight)矩阵。它的作用是扫描输入数据并识别特定的特征或模式。例如，在图像中，一个滤波器可以学会检测垂直边缘，另一个可以检测水平边缘，还有一个可能对特定的纹理或颜色梯度有强烈反应。

滤波器在空间上通常较小（例如 3x3、5x5 像素），但会延伸到输入体量的完整深度。如果输入是彩色图像（具有红、绿、蓝通道），则滤波器的深度也将为 3。

### 卷积过程：滑动与计算

卷积操作包含系统地将滤波器在输入数据（例如图像或来自前一层的特征图）上滑动。在每个位置，滤波器会覆盖输入的一小块区域。主要计算包含三个步骤：

1. **逐元素相乘：** 将滤波器中的权重 (weight)乘以其当前覆盖的输入区域中的对应值。
2. **求和：** 将所有逐元素相乘的结果相加。
3. **添加偏置 (bias)（可选）：** 通常，一个可学习的偏置项会加到总和中。

这个单独计算出的值（总和 + 偏置）表示滤波器在输入中该特定位置的响应。一个强烈的正值表明滤波器旨在识别的特征在该位置强烈存在。

然后，滤波器滑动到下一个位置，该过程重复进行，直到覆盖整个输入。

> 一个 3x3 的滤波器覆盖输入的一个区域。对应元素相乘，结果求和（加偏置）以在输出特征图中生成一个值。然后滤波器移动到下一个位置。

### 特征图（激活图）

通过将单个滤波器应用于整个输入所得到的输出是一个 2D 矩阵，称为**特征图**或**激活图**。该图中的每个元素代表着滤波器在输入中特定空间位置的响应。高激活值表示滤波器识别出的特征在该位置强烈存在。

通常，一个卷积层会使用多个滤波器（例如 32、64 或更多），每个滤波器初始化不同，从而识别不同的特征。将所有这些滤波器应用于相同的输入体量，会形成一系列 2D 特征图，构成卷积层的输出体量。此输出体量的深度等于所用滤波器的数量。

### 控制滑动：步幅

**步幅**规定了滤波器在输入上移动时的步长。

- 步幅为 1 ($S=1$) 表示滤波器每次移动一个像素（水平和垂直方向）。这会导致感受野重叠，并通常生成更大的输出特征图。
- 步幅为 2 ($S=2$) 表示滤波器每次移动时，其起始位置会平移两个像素的距离。这会导致更少的重叠，更快地处理输入，并生成更小的输出特征图（下采样）。

选择步幅是一个设计决策，会影响输出的空间维度和计算成本。

### 维持维度：填充

当应用滤波器时，尤其是在输入边界附近，输出特征图的尺寸自然会比输入小。此外，输入最边缘的像素被滤波器覆盖的次数少于中心像素，这可能会引起信息丢失。

**填充**通过在应用卷积之前在输入体量的边界周围添加额外像素（通常值为零）来处理这些问题。

有两种常见的填充策略：

1. **有效填充（Valid Padding）：** 这表示不添加任何填充。输出尺寸将小于输入尺寸。输出维度的公式（假设输入为 $W \times W$ 的正方形，滤波器为 $F \times F$，步幅为 $S$）是：
   
   $$
   W_{out} = \lfloor \frac{W - F}{S} \rfloor + 1
   $$
   
2. **相同填充（Same Padding）：** 这里的目的是添加足够的零填充，使得输出特征图的高度和宽度与输入特征图相同（当使用步幅 $S=1$ 时）。这非常常见，因为它有助于构建更深的网络，而无需顾虑每一层的空间维度急剧缩小。对于“相同填充”（步幅为 $S=1$），所需的填充量 $P$ 通常为 $P = (F - 1) / 2$（适用于奇数滤波器尺寸）。

### 卷积总结

简而言之，卷积操作使用可学习的滤波器扫描输入数据，进行逐元素乘法和求和以生成特征图。这些图显示了特定模式（由滤波器学到）在不同空间位置的存在。滤波器数量、滤波器尺寸、步幅和填充等参数 (parameter)可以控制特征的提取方式以及输出的空间维度。此过程是 CNN 有效处理图像等网格状数据的基础。

这是一个使用 PyTorch 定义 2D 卷积层的简单示例：

```python
import torch
import torch.nn as nn

# 示例：3 通道输入（例如 RGB 图像），批次大小 1
# 输入维度：(批次大小, 通道数, 高度, 宽度)
input_tensor = torch.randn(1, 3, 64, 64)

# 定义一个卷积层
# in_channels=3: 与输入深度（RGB）匹配
# out_channels=16: 我们希望使用 16 个不同的滤波器
# kernel_size=3: 每个滤波器将是 3x3
# stride=1: 滤波器每次移动一个像素
# padding=1: 使用“相同”填充（对于 3x3 核，步幅 1）
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

# 应用卷积
output_feature_map = conv_layer(input_tensor)

# 打印输出形状
# 输出：torch.Size([1, 16, 64, 64])
# 批次大小 1，16 个特征图（每个滤波器一个），高度 64，宽度 64
# 注意：由于 padding='same' (padding=1) 和 stride=1，高度和宽度保持为 64。
print(output_feature_map.shape)
```

这种基本操作在多个层中重复进行，让 CNN 能够学到分层特征，从早期层中的简单边缘和纹理发展到后续层中更复杂的物体部分。

## 参考资料

- [Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/document/726791) — Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner (1998)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 86; Pages: 2278-2324; DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791)
  这篇开创性论文介绍了LeNet-5，一个开创性的卷积神经网络，详细阐述了卷积操作及其应用。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  第9章“卷积网络”对深度学习中的卷积及相关概念进行了全面的学术论述。
- [Convolutional Neural Networks (CNNs)](http://cs231n.github.io/convolutional-networks/) — Fei-Fei Li, Justin Johnson, Serena Yeung, et al. (Stanford University CS231n Course Notes) (2023)
  这些广为使用的课程笔记对CNNs进行了直观的解释，包括卷积操作、滤波器、步幅和填充。
- [Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) — PyTorch Developers (2024)
  Publisher: PyTorch Foundation
  PyTorch 2D卷积层的官方文档，提供了关于其参数和行为的准确细节，补充了代码示例。

---

[上一节](02-%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%20%28CNNs%29-%20%E5%8A%A8%E5%9B%A0.md) · [下一节](04-CNN%E6%A0%B8%E5%BF%83%E6%93%8D%E4%BD%9C%EF%BC%9A%E6%B1%A0%E5%8C%96.md)

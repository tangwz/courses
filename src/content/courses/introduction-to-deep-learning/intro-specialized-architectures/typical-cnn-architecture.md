---
course: "introduction-to-deep-learning"
chapter: "intro-specialized-architectures"
lesson: "typical-cnn-architecture"
sourceId: 5075
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-7-intro-specialized-architectures/typical-cnn-architecture"
title: "典型CNN架构"
description: "概述CNN的常见结构：卷积层、池化层和全连接层。"
order: 5
plots: []
sourceHash: "c40836e4ee894b770087e1c31c4c8692d791b38d1492a1ca34f21039dd3047b4"
sourceCorrections: []
---

卷积神经网络 (neural network)（CNN）将卷积和池化等基本操作组合成典型架构。尽管这些架构可能因具体任务和数据集而有很大差异，但一种常见模式已经形成，特别是对于图像分类问题。

一个标准CNN通常包含两个主要部分：

1. **特征提取层：** 此部分由交替的卷积层和池化层组成。其主要作用是自动从输入数据中学习分层特征（例如，图像中的边缘、角点、纹理，以及最终更复杂的模式）。
2. **分类/回归层：** 此部分通常包含一个或多个全连接层（也称为密集层），类似于多层感知机（MLP）中的层。它接收卷积层提取的高级特征，并使用它们来完成最终任务，例如对图像进行分类或预测一个值。

让我们分解一下流程和组成部分：

### 特征提取堆栈

输入图像（或其他网格状数据）首先通过一系列卷积层和池化层。

- **卷积层：** 对输入应用一组可学习的滤波器（核）。每个滤波器检测特定的模式或特征。此层的输出是一组特征图，其中每个图都凸显了检测到特定特征的位置。从数学角度看，这包含之前讨论的卷积操作。
- **激活函数 (activation function)：** 紧随卷积之后，对特征图逐元素应用非线性激活函数。修正线性单元（ReLU）激活函数 ($f(x) = \max(0, x)$) 在CNN中非常常用，因为它简单且能有效缓解梯度消失问题。其作用是引入非线性，使网络能够学习更复杂的关系。
- **池化层：** 激活后，通常会插入一个池化层（最常见的是最大池化）。池化减小特征图的空间维度（宽度和高度），使表示更紧凑且计算效率更高。它还有助于对输入图像中的小幅平移或失真产生一定程度的平移不变性。最大池化是获取在特征图上滑动的局部小窗口中的最大值。

`卷积 -> 激活 (ReLU) -> 池化` 的这一序列通常形成一个“块”，并且可以堆叠多个这样的块。随着我们网络越靠后（堆叠更多块），卷积层倾向于在早期层检测到的特征基础上，学习日益复杂和抽象的特征。滤波器大小可能保持较小（例如，3x3），但后面层中的滤波器数量通常会增加，使网络能够捕获更广泛的特征种类。

### 展平步骤

在特征提取堆栈的最后一个池化层之后，得到的特征图通常是3D张量（高 x 宽 x 通道/滤波器数量）。然而，标准全连接层需要1D向量 (vector)作为输入。因此，会执行一个“展平”操作。这只是简单地将3D特征图重塑为一个单一的、长的1D向量，有效地将所有学习到的特征激活排列起来。

### 分类/回归堆栈

展平后的向量 (vector)随后被送入一个或多个全连接层。

- **全连接（密集）层：** 全连接层中的每个神经元都接收来自前一层中*每个*神经元。这些层根据展平向量中包含的高级特征执行分类或回归。它们学习这些特征的非线性组合。ReLU激活也常用于隐藏的密集层之间。
- **输出层：** 最后一层是全连接层，其神经元数量与所需的输出格式相对应。
  - 对于**多类别分类**，它通常有 $N$ 个神经元，其中 $N$ 是类别的数量，并使用Softmax激活函数 (activation function)来产生每个类别的概率。
  - 对于**二元分类**，它可能有一个单独的神经元，使用Sigmoid激活函数产生一个介于0和1之间的概率。
  - 对于**回归**，它会有一个神经元（或更多，取决于要预测的值的数量），使用线性激活函数（或无激活）。

### 架构可视化

下图描绘了图像分类的常见CNN结构：

> CNN的典型流程：输入图像通过卷积层和池化层进行特征提取。得到的特征图被展平为一个向量 (vector)，然后由全连接层处理以进行最终分类或回归。

### PyTorch示例

下面展示了如何使用PyTorch的`nn.Sequential`定义一个类似于上面图示的简单CNN架构：

```python
import torch
import torch.nn as nn

# 假设输入图像是32x32像素，具有3个颜色通道（RGB）
# 并且我们想要分类为10个类别

model = nn.Sequential(
    # 特征提取块 1
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), # 输出: 32x32x32
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2), # 输出: 16x16x32

    # 特征提取块 2
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # 输出: 16x16x64
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2), # 输出: 8x8x64

    # 展平
    nn.Flatten(), # 输出: 8 * 8 * 64 = 4096 个特征

    # 分类层
    nn.Linear(in_features=8*8*64, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=10) # 10个类别的输出层
    # 注意: Softmax通常由损失函数（例如，CrossEntropyLoss）隐式应用
)

# 示例用法: 创建一个虚拟输入张量
dummy_input = torch.randn(1, 3, 32, 32) # (批量大小, 通道数, 高度, 宽度)
output = model(dummy_input)
print(output.shape) # 预期输出: torch.Size([1, 10])
```

这种结构使网络能够学习特征的空间层次结构。早期层检测简单的模式，如边缘和角点，而后面层则将这些模式组合起来，识别与最终任务更复杂的结构。池化层具有稳定性并减少计算负担，而全连接层则整合学习到的特征进行预测。

## 参考资料

- [Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/document/726791) — Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner (1998)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 86; Pages: 2278-2324; DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791)
  介绍了最早成功的CNN架构之一LeNet-5，展示了用于光学字符识别的卷积层和池化层。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本标准教科书，对卷积网络提供了广泛的介绍，包括其理论基础和架构设计。
- [Convolutional Neural Networks for Visual Recognition (CS231n) Lecture Notes](https://cs231n.github.io/slides/2017/cs231n_2017_lecture5.pdf) — Fei-Fei Li, Justin Johnson, and Serena Yeung (2017)
  Publisher: Stanford University
  提供关于卷积神经网络的详细讲义和解释，涵盖其架构、常见层和实际考虑因素。
- [Deep Sparse Rectifier Neural Networks](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf) — Xavier Glorot, Antoine Bordes, Yoshua Bengio (2011)
  Journal: Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics (AISTATS); Publisher: PMLR; Volume: 15; Pages: 315-323
  介绍并分析了修正线性单元（ReLU）作为激活函数的使用，强调了它们在训练深度网络中的优势。
- [torch.nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) — PyTorch Developers (2024)
  Publisher: PyTorch Foundation
  PyTorch `nn.Sequential` 模块及相关层（如 `Conv2d`、`MaxPool2d`、`Flatten` 和 `Linear`）的官方文档，用于构建CNN。

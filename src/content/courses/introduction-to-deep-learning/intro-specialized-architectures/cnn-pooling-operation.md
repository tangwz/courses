---
course: "introduction-to-deep-learning"
chapter: "intro-specialized-architectures"
lesson: "cnn-pooling-operation"
sourceId: 5074
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-7-intro-specialized-architectures/cnn-pooling-operation"
title: "CNN核心操作：池化"
description: "了解池化层（最大池化、平均池化）的用途和类型。"
order: 4
plots: []
sourceHash: "16c0d81fe29454569546c7ef5f65f4ad48a238e8679fc2234de4307ee4c2dc83"
sourceCorrections: []
---

卷积层从输入中提取特征后，会生成特征图。这些特征图虽然包含信息，但通常保持较高的空间分辨率（宽度和高度）。在后续层中处理这些大型特征图可能计算成本高昂，并可能使网络对输入中特征的精确位置过于敏感。池化层提供了一种系统地减小特征图空间维度的方法来解决此问题。

池化，也称为子采样或降采样，它对特征图局部区域的信息进行概括。它有助于创建更紧凑、对输入图像中的小偏移或失真具有稍强鲁棒性的表示。可以将其视为创建了一个较低分辨率的概要，它保留了前一个卷积层检测到的必要特征。

### 池化工作原理

池化操作独立地在每个输入特征图上滑动一个窗口（通常称为池化滤波器或核）。对于窗口的每个位置，它将该窗口内的值聚合成一个单一的输出值。窗口随后根据指定的步幅滑动，类似于卷积。

与卷积层不同，池化层通常没有可学习的参数 (parameter)（权重 (weight)或偏差）。聚合函数（如最大值或平均值）是固定的。这使得它们更简单且计算成本更低。

定义池化操作的主要参数有：

- **池化窗口大小：** 要聚合区域的维度（高 x 宽）（例如，2x2、3x3）。
- **步幅：** 窗口在每一步水平和垂直方向上移动的像素数量。步幅等于窗口大小会导致非重叠的池化区域。
- **填充：** 通常，池化层不使用填充（“有效”填充），因为目标是降维。但是，如果需要保持特定的输出维度，有时可以使用“相同”填充，尽管这不太常见。

### 常见的池化类型

池化操作是卷积神经网络 (neural network) (CNN) 中常用的技术，主要有两种类型：

#### 最大池化

最大池化从池化窗口覆盖的特征图区域中选择最大值。

- **操作：** 对于每个窗口位置，输出 = $max(区域值)$。
- **理念：** 它捕获局部区域内最显著的特征（激活值最高的那个）。这种方法倾向于保留强信号，并在许多分类任务中有效。如果网络在某个区域内检测到强边缘或角点，最大池化会确保该信息被传递。

考虑一个4x4输入特征图和步幅为2的2x2最大池化操作：

```
输入特征图：
[[ 1,  3,  2,  4],
 [ 5,  6,  7,  8],
 [ 9,  0,  1,  2],
 [ 3,  4,  5,  6]]

池化窗口 (2x2), 步幅 (2)

左上窗口：max(1, 3, 5, 6) = 6
右上窗口：max(2, 4, 7, 8) = 8
左下窗口：max(9, 0, 3, 4) = 9
右下窗口：max(1, 2, 5, 6) = 6

输出特征图 (2x2)：
[[ 6,  8 ],
 [ 9,  6 ]]
```

#### 平均池化

平均池化计算池化窗口内所有元素的平均值。

- **操作：** 对于每个窗口位置，输出 = $average(区域值)$。
- **理念：** 它提供了区域中特征的平滑、概括性概要。虽然在分类CNN的中间层中不如最大池化常见，但它在某些架构中或作为全连接层之前减小特征图的最后一步（全局平均池化）时可能有用。

使用相同的4x4输入图，步幅为2的2x2平均池化操作将得到：

```
输入特征图：
[[ 1,  3,  2,  4],
 [ 5,  6,  7,  8],
 [ 9,  0,  1,  2],
 [ 3,  4,  5,  6]]

池化窗口 (2x2), 步幅 (2)

左上窗口：avg(1, 3, 5, 6) = 15/4 = 3.75
右上窗口：avg(2, 4, 7, 8) = 21/4 = 5.25
左下窗口：avg(9, 0, 3, 4) = 16/4 = 4.00
右下窗口：avg(1, 2, 5, 6) = 14/4 = 3.50

输出特征图 (2x2)：
[[ 3.75, 5.25 ],
 [ 4.00, 3.50 ]]
```

#### 全局池化

一种变体是全局池化（包括全局最大池化或全局平均池化）。它不使用小窗口，而是将池化操作应用于每个特征图的*整个*空间维度（宽度和高度）。如果输入特征图的维度为 $H \times W \times C$，全局池化会生成大小为 $1 \times 1 \times C$ 的输出，有效地将每个通道的特征图缩减为一个单一数值。这通常在网络的末端使用，以在送入最终分类层之前大幅减小维度。

### 最大池化可视化

这是一个小例子，说明了步幅为2的2x2最大池化操作。

> 示例：将步幅为2的2x2最大池化操作应用于4x4输入特征图。输出图中的每个值对应于输入中2x2非重叠区域的最大值。

### 在CNN架构中的作用

池化层通常插入在连续的卷积层之间（通常在对卷积输出应用激活函数 (activation function)之后）。CNN中的常见模式是：

CONV -> ReLU -> POOL -> CONV -> ReLU -> POOL -> ... -> Flatten -> Fully Connected -> Output

这种结构允许网络首先学习分层特征（CONV + ReLU），然后周期性地降低空间分辨率，同时保留重要信息（POOL）。这种缩减有助于管理计算负载，并对特征定位的变化建立一定的容忍度。

总之，池化是CNN中的一项基本操作，它减小特征图的空间维度，降低计算量，并引入一定程度的平移不变性，与卷积层的特征提取作用相辅相成。最大池化是使用最频繁的变体，因为它在保留强特征激活方面有效。

## 参考资料

- [Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/document/726791) — Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner (1998)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 86; Pages: 2278-2324; DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791)
  介绍了卷积神经网络的基础概念，包括子采样（池化），为现代CNN架构奠定了基础。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: 326-330
  一本全面的教科书，详细解释了卷积网络中池化层的理论和实践。
- [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b4b574a76f146438993c126-Paper.pdf) — Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton (2012)
  Journal: Advances in Neural Information Processing Systems 25; Publisher: Curran Associates, Inc.; Volume: 25; Pages: 1097-1105; DOI: [10.5591/978-1-57766-314-2](https://doi.org/10.5591/978-1-57766-314-2)
  这篇里程碑式的论文展示了深度CNN的强大能力，主要采用了最大池化进行下采样，为其在ImageNet上的突破性表现做出了贡献。
- [Convolutional Neural Networks for Visual Recognition - Stanford CS231n](http://cs231n.github.io/convolutional-networks/#pool) — Andrej Karpathy, Justin Johnson, Fei-Fei Li, et al. (2024)
  为学生提供了关于卷积神经网络的清晰易懂的解释，包括池化层的作用和机制。

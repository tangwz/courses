---
course: "cnns-for-computer-vision"
chapter: "cnn-foundations-modern-architectures"
lesson: "cnn-building-blocks-review"
sourceId: 2511
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-1-cnn-foundations-modern-architectures/cnn-building-blocks-review"
title: "卷积神经网络构建模块简要回顾"
description: "为高级学习者简明回顾卷积神经网络核心组成部分：卷积、池化和激活函数。"
order: 1
plots: []
sourceHash: "d8a20c03c6301c58312ddcb32da0d9e13ce5317fea5c9dba250f76241fa56521"
sourceCorrections: []
---

为审视塑造现代计算机视觉的精巧架构做准备，将概述构成几乎所有卷积神经网络 (neural network) (CNN) 基础的基本组成部分。这将确保对术语和机制有共同的理解。

### 卷积层：学习空间层次特征

任何卷积神经网络 (neural network) (CNN)的核心都是卷积层。它的主要作用是在输入图像（或前一层的特征图）中识别局部模式，例如边缘、拐角和纹理。这是通过将小型滤波器（也称为卷积核）滑动通过输入体来实现的。

卷积层的重要方面包括：

- **滤波器（卷积核）：** 这些是可学习参数 (parameter)（权重 (weight)）的小矩阵。每个滤波器专门用于识别特定类型的特征。例如，一个滤波器可能学习识别水平边缘，而另一个则识别特定的颜色模式。滤波器的深度与输入体的深度（通道数）相匹配。
- **特征图：** 将滤波器应用于输入后的输出是一个二维激活图，或称特征图。此图突出显示了输入中检测到滤波器特定模式的区域。一个卷积层通常会应用多个滤波器，生成一组特征图（一个体）作为其输出。
- **步长：** 此参数定义了滤波器在输入上移动的步长。步长为 1 意味着滤波器每次移动一个像素，而步长为 2 意味着它每隔一个像素跳过，这通常会导致输出特征图尺寸更小。
- **填充：** 通常，会在输入体的边界周围添加零填充。这使得滤波器能更有效地处理输入的边缘，并能控制输出特征图的空间尺寸。'Same' 填充旨在保持输出空间尺寸与输入相同（在步长为 1 的情况下），而 'valid' 填充不使用填充，可能减小输出尺寸。
- **参数共享：** 卷积层的一个重要优势是参数共享。单个滤波器内的权重在整个输入空间维度上被重复使用。与全连接层相比，这大大减少了参数的数量，并使网络对特征的位置具有一定的平移不变性。

### 激活函数 (activation function)：引入非线性特性

在卷积操作之后，通常会将激活函数逐元素应用于结果特征图。其作用是为网络引入非线性。如果没有非线性，堆叠多个卷积层将等同于单个更大的卷积层，从而限制了网络对数据中复杂关系的建模能力。

现代卷积神经网络 (neural network) (CNN)中最常用的激活函数是整流线性单元 (ReLU)：


$$
ReLU(x) = max(0, x)
$$


ReLU 计算效率高，并有助于缓解在非常深的网络中早期激活函数（如 Sigmoid 或 Tanh）遇到的梯度消失问题。虽然 Leaky ReLU、参数 (parameter)化 ReLU (PReLU) 或 GELU 等变体存在并用于更精巧的架构中，但标准 ReLU 仍然是一个可靠的基准。

### 池化层：下采样与不变性

池化层通常插入在连续的卷积层之间。它们的主要目标是：

1. **维度降低：** 它们逐渐减小特征图的空间尺寸（宽度和高度）。这减少了后续层的参数 (parameter)数量和计算负担。
2. **平移不变性：** 池化使得表示对输入图像中的微小平移或形变更为稳定。通过汇总局部区域的特征，特征的确切位置变得不那么重要。

常见的池化策略包括：

- **最大池化：** 从池化窗口覆盖的特征图区域中选择最大值。它倾向于保留最强的激活。
- **平均池化：** 计算池化窗口内的平均值。它提供了更平滑的下采样。

与卷积层类似，池化层具有窗口大小和步长。例如，常用的 $2 \times 2$ 的窗口和 2 的步长，有效地将特征图的宽度和高度减半，同时保持深度不变。

> 卷积神经网络 (neural network) (CNN)块内典型的操作序列，将输入体转换为一组抽象特征图。

这三个组件以各种配置堆叠，构成了大多数卷积神经网络的基本处理单元。理解它们各自的作用以及如何相互配合是考察旨在构建更深、能力更强的架构创新所必需的，我们将在后面介绍这些创新。

## 参考资料

- [Gradient-based learning applied to document recognition](https://doi.org/10.1109/5.726791) — Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner (1998)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 86; Pages: 2278-2324; DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791)
  这篇论文介绍了LeNet-5架构，将卷积层、池化和参数共享确立为CNNs的核心组件。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本内容全面的教科书，详细解释了卷积网络、其基本层和激活函数。
- [CS231n: Convolutional Neural Networks for Visual Recognition - Lecture Notes on Architectures of Convolutional Networks](http://cs231n.github.io/convolutional-networks/) — Andrej Karpathy, Justin Johnson, and Fei-Fei Li (2023)
  这份权威课程笔记提供了卷积层、池化和激活函数的清晰解释及可视化内容。
- [Rectified Linear Units for Restricted Boltzmann Machines](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf) — Vinod Nair and Geoffrey Hinton (2010)
  Journal: Proceedings of the 27th International Conference on Machine Learning (ICML); Pages: 807-814
  这篇论文介绍了修正线性单元（ReLU）激活函数，该函数后来成为深度神经网络的标准组件。

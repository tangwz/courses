# 卷积神经网络 (CNN) 概述

来源：[原文](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-7-introduction-common-architectures/cnn-overview)

[返回章节目录](README.md) · [返回课程目录](../README.md)

标准神经网络 (neural network)层，如 `nn.Linear`，将输入数据视为一个扁平向量 (vector)。尽管功能强大，但这种方法未能内在理解图像等数据中存在的空间结构。对于图像来说，相互靠近的像素通常是关联的，它们构成边缘、纹理或物体的一部分。当直接应用于图像时，全连接层面临两个主要问题：

1. **参数 (parameter)效率低下：** 将一张中等大小的图像（例如，224x224 像素，3 个颜色通道）展平为向量会导致输入维度非常大。即使将其连接到一个中等大小的隐藏层，也需要大量权重 (weight)，使得模型容易过拟合 (overfitting)，并且计算成本高昂。
2. **空间信息丢失：** 展平图像会丢弃像素的 2D（或包含通道的 3D）排列。网络会丢失关于哪些像素最初是相邻的信息。

卷积神经网络 (CNN) 是一种专门设计用于处理具有网格状拓扑数据（如图像（2D 网格）或时间序列数据（1D 网格））的神经网络。它们通过结合两个主要思想来解决标准网络的局限：局部感受野（通过卷积）和空间下采样（通过池化）。

### 卷积操作：识别局部模式

CNN 的核心组成部分是**卷积层**。卷积层不将每个输入单元连接到每个输出单元，而是使用小的过滤器（也称为核），它们在输入数据上滑动。每个过滤器都是一个小的权重 (weight)矩阵。

想象一个微小的放大镜（即过滤器）在输入图像上滑动。在每个位置，过滤器会与其当前覆盖的图像区域执行元素级乘法，并将结果求和以在输出中生成一个单一值。这个过程在整个输入图像上重复进行，生成一个输出**特征图**。

> 过滤器对输入的局部区域施加权重，以计算输出特征图中的一个值。

这种滑动过滤器方法具有两个显著优点：

1. **局部连接：** 特征图中的每个单元仅连接到输入的一个小区域（过滤器大小）。这使得网络能够在早期层中学习到局部模式，如边缘或角落。
2. **参数 (parameter)共享：** 相同的过滤器（具有相同的权重集合）在输入图像的不同位置重复使用。这与全连接层相比**大幅减少了**参数数量，并使网络对特征的平移具有等变性。如果一个模式（如垂直边缘）被过滤器学习，它可以在图像中任何位置检测到该模式。

通常，一个卷积层会使用多个过滤器，每个过滤器学习识别不同类型的特征（例如，一个过滤器识别水平边缘，另一个识别垂直边缘，还有一个识别特定纹理）。这些过滤器的输出堆叠在一起，形成该层的最终输出体。PyTorch 主要通过 `nn.Conv2d` 层来实现图像数据的这一操作。

### 激活函数 (activation function)

就像在标准网络中一样，非线性激活函数（例如 ReLU，在 PyTorch 中实现为 `nn.ReLU`）通常在卷积操作之后进行元素级应用。这使得网络能够学习特征之间复杂的非线性关系。

### 池化操作：下采样与不变性

在通过卷积层检测到特征后，通常有益于使表示更紧凑并对小的空间变异具有抵抗力。这通过使用**池化层**来实现。

最常见的类型是**最大池化**。它也涉及在特征图上滑动一个窗口（通常小于卷积过滤器且不重叠或带步幅）。但是，它不应用学习到的权重 (weight)，而只是简单地取出该窗口内的*最大*值。

> 最大池化选择特征图局部窗口内的最大值。

池化提供多项益处：

1. **维度降低：** 它减少了特征图的空间维度（高度和宽度），降低了后续层的计算负担。
2. **平移不变性（局部）：** 通过用其最大激活来概括局部区域，池化使表示对特征在该区域内的确切位置更具稳定性。

PyTorch 提供了 `nn.MaxPool2d` 等池化层。

### 典型 CNN 架构

一个典型的 CNN 架构通常会堆叠这些组件：

1. 一个或多个**卷积 -> 激活 -> 池化**层块。早期层倾向于使用较小的过滤器来捕捉精细细节，而后期层可能使用较大的过滤器，或依赖于早期层的池化特征来捕捉更大空间区域上的更复杂模式。
2. 在经过多个卷积和池化层之后，得到的特征图通常会**展平**为一个向量 (vector)。
3. 然后，这个向量被馈入一个或多个**全连接 (`nn.Linear`) 层**，类似于标准前馈网络，用于最终的分类或回归。

> 一个典型的 CNN 架构流程。

CNN **运用**卷积和池化，直接从网格状数据中自动学习特征的分层表示，这使得它们在图像识别、物体检测等任务中表现非常出色，甚至在文本得到适当表示时，也能用于自然语言处理。在下一节中，你将看到如何在 PyTorch 中实现像 `nn.Conv2d` 和 `nn.MaxPool2d` 这样的构建模块，以构建你的第一个 CNN。

## 参考资料

- [Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/document/726791) — Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner (1998)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 86; Pages: 2278-2324; DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791)
  这篇论文介绍了LeNet-5，这是一种卷积网络架构，确立了CNN的许多基础概念，包括卷积、池化和图像识别中的参数共享。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  第9章提供了卷积网络的详细理论背景，涵盖了卷积、池化和各种架构考虑因素的原理。
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks/) — Fei-Fei Li, Justin Johnson, Serena Yeung, and Richard Zhang (2023)
  Publisher: Stanford University
  全面的在线课程笔记，提供了CNN架构、卷积、池化以及图像任务实践考虑因素的易于理解的解释。
- [Build the Neural Network](https://pytorch.org/tutorials/beginner/basics/build_model_tutorial.html) — PyTorch Documentation Team (2021)
  Publisher: PyTorch
  官方PyTorch教程，演示如何使用nn.Conv2d、nn.MaxPool2d和其他核心PyTorch模块构建卷积神经网络。

---

[上一节](../06-%E5%AE%9E%E7%8E%B0%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF/11-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%8C%E6%95%B4%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.md) · [下一节](02-%E5%9C%A8PyTorch%E4%B8%AD%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84CNN.md)

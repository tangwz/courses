---
course: "introduction-to-neural-networks"
chapter: "building-first-neural-network"
lesson: "setup-network-architecture"
sourceId: 1982
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-5-building-first-neural-network/setup-network-architecture"
title: "搭建网络结构"
description: "为一个特定任务定义层、神经元和激活函数。"
order: 1
plots: []
sourceHash: "d0e2072412cc00edef623bb0d4beb3ce062f271306ec80fbe77a05a5666df34f"
sourceCorrections: []
---

你已掌握了主要运作方式：单个神经元如何计算，数据如何前向流动以进行预测，以及反向传播 (backpropagation)与梯度下降 (gradient descent)相结合如何使网络从错误中学习。现在，是时候扮演架构师的角色，在搭建（训练）开始之前，为你的神经网络 (neural network)设计蓝图了。设置网络结构包括对其结构做出具体选择，这从根本上决定了它将如何处理信息和进行学习。

可以将这一步想象成决定一栋建筑应有多少层和多少房间。这些决定取决于建筑的用途。同样，你的网络结构也很大程度上取决于你试图解决的具体问题以及数据的性质。

你需要定义的主要组成部分是：

1. **层数：** 你的网络将有多少层？至少，你需要一个输入层和一个输出层。通常，一个或多个*隐藏层*位于它们之间。
2. **每层的神经元（或单元）数量：** 每层在计算上应有多“宽”？
3. **每层的激活函数 (activation function)：** 哪个函数将在每层的线性计算之后引入非线性？

我们来详细分析每个部分的考量因素。

### 输入层

输入层是网络的入口点。它不像其他层那样执行计算（传统意义上不应用加权和或激活函数 (activation function)）。它的主要作用是接收输入数据。

- **神经元数量：** 输入层中的神经元数量直接由数据集中特征的数量决定。如果你处理的是每个样本有15个特征的表格数据，你的输入层将有15个神经元。如果你处理的是28x28像素的灰度图像，通常会将图像展平为一个$28 \times 28 = 784$像素的向量 (vector)，这意味着你的输入层需要784个神经元。你已在第2章中了解了如何准备数据（缩放、编码）；预处理的结果决定了输入层的形状。

### 隐藏层

隐藏层是输入层和输出层之间的中间层。它们是大多数复杂特征学习发生的地方。

- **层数（深度）：** 增加更多隐藏层（使网络变得“更深”）允许网络有可能学习更复杂、分层的特征。早期层可能学习简单的模式（如图像中的边缘或纹理），而更深的层将这些结合起来以识别更抽象的事物（如物体或人脸）。然而，更深的网络训练起来更具挑战性（例如，梯度消失问题，稍后讨论），并且计算成本更高。对于许多标准问题，从一到两个隐藏层开始是一个合理的起点。
- **每层神经元数量（宽度）：** 增加隐藏层中的神经元数量（使其“更宽”）赋予网络学习该层内部模式的更大能力。神经元过少可能导致*欠拟合 (underfitting)*（网络无法捕捉数据的复杂性），而过多则可能增加计算成本以及*过拟合 (overfitting)*的风险（网络对训练数据学习得太好，包括噪声，而在新数据上表现不佳）。没有单一的万能公式；选择宽度通常需要一些尝试，但常见做法是隐藏层的神经元数量与输入/输出大小相关，有时会随着接近输出层而减小。
- **激活函数 (activation function)：** 隐藏层几乎总是使用非线性激活函数。正如第1章所讨论的，这种非线性对于网络学习超越简单线性关系的复杂映射是必不可少的。ReLU（修正线性单元）因其计算效率和在缓解梯度消失问题方面的有效性，目前是隐藏层最受欢迎的选择。Tanh或Sigmoid也是可能的选择，但在现代深度隐藏层中较少见。通常，单个隐藏层内的所有神经元都使用相同的激活函数。

### 输出层

输出层是最后一层，产生网络的预测。它的配置很大程度上取决于任务的类型：

- **神经元数量：**

  - **回归：** 对于预测连续值（例如，预测房价）的任务，输出层通常只有一个神经元。
  - **二分类：** 对于只有两个可能结果类别（例如，垃圾邮件 vs. 非垃圾邮件，猫 vs. 狗）的任务，输出层通常有一个神经元。
  - **多分类：** 对于有两个以上互斥类别（例如，分类手写数字0-9）的任务，输出层每个类别有一个神经元。因此，对于数字分类（10个类别），输出层将有10个神经元。
- **激活函数 (activation function)：** 此处的选择对于正确格式化输出非常重要：

  - **回归：** 通常不使用激活函数（或使用*线性*激活，$f(x)=x$），允许输出神经元产生任何实数值。
  - **二分类：** Sigmoid激活函数是常用的。它将输出压缩到0到1之间，这可以解释为属于正类别的概率。 $\sigma(z) = \frac{1}{1 + e^{-z}}$
  - **多分类：** Softmax激活函数是标准选择。它接受最后一层的原始输出（logits），并将它们转换为所有类别的概率分布，确保输出总和为1。对于$K$个类别和原始输出$z_1, ..., z_K$： $\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$ 每个输出神经元随后代表其对应类别的预测概率。

### 示例：设计一个简单分类器结构

假设我们想构建一个网络来分类像MNIST（手写数字0-9）数据集中的图像。这些图像是28x28像素的灰度图。

1. **输入层：** 图像是$28 \times 28 = 784$像素。展平后，我们的输入层需要784个神经元。
2. **隐藏层：** 我们可以从一个简单的结构开始，或许是两个隐藏层。第一个隐藏层选择128个神经元，第二个选择64个。这是一个常见的经验法则，随着层数加深而减小宽度。对于激活函数 (activation function)，两个隐藏层都将使用ReLU。
3. **输出层：** 我们有10个可能的类别（数字0到9）。因此，输出层需要10个神经元。由于这是一个多分类问题，我们将使用Softmax激活函数。

这是该结构的可视化：

> 用于MNIST分类的简单前馈神经网络 (neural network)结构。输入层接收展平的像素数据，经过两个宽度递减的ReLU隐藏层，最后到达一个用于输出类别概率的10神经元Softmax输出层。虚线表示相邻层之间的完全连接。

这定义了我们网络的结构。下一步，我们稍后将介绍，是初始化这些神经元之间所有连接的权重 (weight)和偏置 (bias)。请记住，结构设计通常是一个迭代过程。你可能会从一个像这样的简单结构开始，训练它，评估其性能，然后重新审视结构（例如，改变层的大小，添加/移除层，尝试不同的激活函数）以提升结果。像TensorFlow和PyTorch这样的深度学习 (deep learning)框架提供了便捷的方式来定义这些层并尝试不同的结构，抽象化了大部分手动设置。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本权威的深度学习教材，涵盖了深度学习的数学和概念基础，包括关于前馈神经网络、层和激活函数的详细章节。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  提供使用 Keras 和 TensorFlow 实现神经网络的实用、代码驱动指南，其中有专门章节介绍如何为各种机器学习任务设计和配置网络架构。
- [CS231n: Convolutional Neural Networks for Visual Recognition - Neural Networks (notes)](https://cs231n.github.io/neural-networks-1/) — Stanford University, Fei-Fei Li, Justin Johnson, and Serena Yeung (2017)
  提供清晰简洁的讲义，解释神经网络的基础知识，包括层、激活函数和基本的架构设计等概念。
- [Keras layers API: tf.keras.layers.Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) — TensorFlow Developers (2024)
  Publisher: TensorFlow
  Keras 中 `Dense`（全连接）层的官方文档，展示了其参数和在构建神经网络架构中的用法。

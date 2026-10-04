---
course: "introduction-to-deep-learning"
chapter: "neural-network-foundations"
lesson: "artificial-neuron-model"
sourceId: 5020
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-1-neural-network-foundations/artificial-neuron-model"
title: "人工神经元：数学模型"
description: "界定人工神经元的组成部分：输入、权重、偏置、求和与激活。"
order: 3
plots: []
sourceHash: "b313eaae0890ee0ba232c33a5c5ca878947d283d4fd924750494a7c2ef745dcd"
sourceCorrections: []
---

尽管生物神经元提供了一个有趣的起点，但构建计算模型需要更精确的数学抽象。让我们剖析一下人工神经元的标准模型，它在网络中常被称为单元或节点。这个模型以简化且易于计算的形式，捕捉了生物神经元信号处理的本质。

可以将人工神经元看作是一个处理单元，它接收多个输入，执行计算，并产生一个输出。以下是它的基本组成部分：

### 输入与权重 (weight)

人工神经元接收一个或多个输入信号。这些输入，表示为$x_1, x_2, ..., x_n$，代表输入神经元的特征或信息。例如，在图像分类任务中，这些输入可以是图像一小块区域的像素值。

每个输入连接都有一个关联权重，表示为$w_1, w_2, ..., w_n$。这些权重是网络在训练过程中学习到的重要参数 (parameter)。权重表示其相应输入信号的重要程度或强度。一个大的正权重意味着该输入强烈地激发神经元，而一个大的负权重意味着该输入强烈地抑制神经元。一个接近零的权重表示该输入对神经元的输出影响很小。

### 求和函数

神经元内部的第一步是计算所有输入的加权和。这汇集了所有输入信号的影响，并由各自的权重 (weight)调节。在数学上，这通常表示为：


$$
z = (w_1 x_1 + w_2 x_2 + \dots + w_n x_n) + b
$$


这本质上是输入的线性组合。

### 偏置 (bias)

请注意求和函数中的附加项$b$。这是偏置项。你可以将偏置视为一种移动激活函数 (activation function)触发点的方式，使神经元更容易或更难被激活（产生非零输出）。或者，它可以被视为与常数输入1（$x_0 = 1$, $w_0 = b$）关联的权重 (weight)。如果没有偏置，神经元的加权和$w_1 x_1 + \dots + w_n x_n$将始终通过原点，从而限制了其灵活性。偏置允许神经元模拟不一定通过原点的关系。与权重一样，偏置是训练期间调整的可学习参数 (parameter)。

### 激活函数 (activation function)

求和结果$z$随后通过一个激活函数，通常表示为$g(\cdot)$。这个函数为神经元的输出引入非线性。


$$
\text{输出} = a = g(z) = g(\sum_{i=1}^{n} w_i x_i + b)
$$


为什么非线性很重要？如果神经元只执行线性变换（如加权和），那么堆叠多层神经元整体上仍只会产生线性变换。这将严重限制网络可以学习的函数的复杂性。生物神经元表现出非线性发放行为，激活函数模仿了这一点。常见示例包括 Sigmoid、Tanh 和 ReLU 函数，我们将在下一章中详细介绍它们。现在，请理解激活函数决定了传递给其他神经元或用作最终网络输出的最终信号。

### 模型可视化

我们可以将这个数学模型可视化如下：

> 人工神经元模型的图示。输入（$x_i$）乘以权重 (weight)（$w_i$），与偏置 (bias)（$b$）一起求和得到$z$，然后通过激活函数 (activation function)$g(\cdot)$产生最终输出$a$。

这个数学模型，简单而强大，构成了几乎所有神经网络 (neural network)的基本模型。我们接下来将讨论的感知器，本质上就是这个模型，只是使用了特定类型的激活函数（一个阶跃函数）。通过将许多这样的神经元分层连接起来，我们可以构建能够学习数据中错综复杂模式的网络。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的深度学习教科书，涵盖了人工神经元及其组件的数学模型等基本概念。
- [Principles of Neurodynamics: Perceptrons and the Theory of Brain Mechanisms](https://archive.org/details/principlesofneur00rose) — Frank Rosenblatt (1962)
  Publisher: Spartan Books
  介绍感知器（现代人工神经元模型的直接前身）的基础性书籍，详细阐述了其数学基础。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — Michael A. Nielsen (2015)
  Publisher: Determination Press
  一本易于理解的在线书籍，清晰地解释了神经网络的基础知识，包括感知器和人工神经元的架构。

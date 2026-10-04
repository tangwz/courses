---
course: "introduction-to-neural-networks"
chapter: "neural-network-foundations"
lesson: "network-layers"
sourceId: 1919
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-1-neural-network-foundations/network-layers"
title: "网络结构：层与连接"
description: "了解输入层、隐藏层和输出层及其如何构成网络。"
order: 4
plots: []
sourceHash: "03d22c370c34ca56ca0a5dd63cd5df3dc475c1ac7397fe17e87338e2675bc5c6"
sourceCorrections: []
---

既然我们了解了人工神经元的组成部分，包括权重 (weight)、偏置 (bias)和激活函数 (activation function)，那么现在来看看这些单元是如何组织起来以构建一个功能性神经网络 (neural network)的。单个神经元是强大的计算单元，但正是它们集体排列成层以及层之间的连接，才使网络能够从数据中学习复杂的模式。

### 分层架构

神经网络 (neural network)通常按层组织。每层包含一个或多个神经元。我们通常区分三种类型的层：

1. **输入层：** 这是网络的入口点。它不执行应用权重 (weight)和激活函数 (activation function)意义上的任何计算（或者可以认为它具有恒等激活函数）。相反，它只是保存输入到网络的初始数据。输入层中的神经元数量直接对应于输入数据中的特征数量。例如，如果您根据面积（平方英尺）和卧室数量预测房价，您的输入层将有两个神经元。
2. **隐藏层：** 隐藏层位于输入层和输出层之间，是大部分计算发生的地方。隐藏层中的神经元从前一层（可以是输入层或另一个隐藏层）的所有神经元接收输入，计算它们的加权和加上偏置 (bias)（$z$），并应用激活函数（$a = f(z)$）。一个隐藏层的输出（$a$）随后作为下一层的输入。一个网络可以有零个、一个或多个隐藏层。具有多个隐藏层的网络通常被称为“深度”神经网络。这些中间层使网络能够逐步学习更复杂的表示以及输入特征的组合。选择隐藏层的数量以及每层中神经元的数量是网络设计的一个重要部分。
3. **输出层：** 这是网络的最后一层，产生最终结果。输出层的结构在很大程度上取决于网络设计的具体任务：

   - **回归：** 用于预测连续值（如房价），输出层通常只有一个神经元，通常带有线性激活函数（或不带激活函数）。
   - **二分类：** 用于将输入分类到两个类别之一（例如，垃圾邮件或非垃圾邮件），输出层通常有一个带有 Sigmoid 激活函数的神经元，输出介于 0 和 1 之间的概率。
   - **多分类：** 用于将输入分类到多个类别之一（例如，数字识别 0-9），输出层通常每个类别有一个神经元，通常使用 Softmax 激活函数。Softmax 确保输出表示的概率在所有类别中总和为 1。

### 层间连接

在我们最初讨论的标准前馈网络中，层通常是**全连接的**（或**密集连接的**）。这意味着一个层中的每个神经元将其输出信号发送到后续层中的*每个*神经元。这些连接中的每一个都具有一个相关的权重 (weight)，表示连接的强度。

一个简单的网络通常包含以下层：

- 输入层（$L_0$），包含 $n_0$ 个神经元（特征）。
- 隐藏层（$L_1$），包含 $n_1$ 个神经元。
- 输出层（$L_2$），包含 $n_2$ 个神经元。

在一个全连接设置中：

- $L_1$ 中的每个 $n_1$ 神经元从 $L_0$ 中的所有 $n_0$ 神经元接收输入。这需要 $n_0 \times n_1$ 个连接 $L_0$ 到 $L_1$ 的权重。$L_1$ 中的每个神经元也有其自身的偏置 (bias)项。
- $L_2$ 中的每个 $n_2$ 神经元从 $L_1$ 中的所有 $n_1$ 神经元接收输入。这需要 $n_1 \times n_2$ 个连接 $L_1$ 到 $L_2$ 的权重。$L_2$ 中的每个神经元也有其自身的偏置项。

下图展示了一个简单的全连接前馈网络，包含一个输入层（3个神经元）、一个隐藏层（4个神经元）和一个输出层（2个神经元）。

> 一个简单的前馈神经网络 (neural network)架构。输入节点（蓝色）将数据传递给隐藏层节点（绿色），隐藏层节点处理数据并将其结果传递给输出节点（红色）。一个层中的每个节点都连接到下一层中的每个节点。

这种分层和连接的结构确定了数据在网络中流动的路径。当我们将数据输入到输入层时，它向前传播通过隐藏层，在每个步骤中经历转换（通过权重和偏置进行线性组合，然后是非线性激活），直到到达输出层，输出层产生最终预测。这个过程被称为**前向传播**，是第 3 章的主题。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教材，涵盖神经网络架构的基本概念，包括层和连接。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — Michael A. Nielsen (2015)
  Publisher: Determination Press
  一本易于理解的在线书籍，清晰地阐述了神经网络结构、层及其工作原理。
- [Neural Networks and Deep Learning (Course 1 of the Deep Learning Specialization)](https://www.coursera.org/learn/neural-networks-deep-learning) — Andrew Ng, Kian Katanforoosh, Younes Bensouda Mourri (n.d.)
  Publisher: DeepLearning.AI
  一个广受认可的在线课程，介绍神经网络的架构，包括输入层、隐藏层和输出层。
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/10.1007/978-0-387-45528-0) — Christopher M. Bishop (2006)
  Publisher: Springer; DOI: [10.1007/978-0-387-45528-0](https://doi.org/10.1007/978-0-387-45528-0)
  对神经网络基础提供了严谨的数学处理，有助于理解网络结构的基本原理。

---
course: "introduction-to-deep-learning"
chapter: "neural-network-foundations"
lesson: "biological-inspiration-neuron"
sourceId: 5019
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-1-neural-network-foundations/biological-inspiration-neuron"
title: "生物学上的启发：神经元"
description: "了解生物神经元作为神经网络中人工神经元的构想模型。"
order: 2
plots: []
sourceHash: "afeda1dfbd97f5652be74b2a293628350a05efadccdf4fcf8cc0f9764757924a"
sourceCorrections: []
---

虽然人工神经网络 (neural network)（ANNs）本质上是软件中实现的数学模型，但其最初的构想在很大程度上受到大脑主要处理单元——生物神经元的结构和功能的启发。了解这种生物对应物有助于理解早期人工神经网络的设计选择，并提供了一个有用的思维模型，即使现代深度学习 (deep learning)已发生很大变化。

### 大脑的处理单元

人脑包含数十亿神经元，形成一个庞大且相互连接的网络。每个神经元都充当一个微小的信息处理器。尽管生化上很复杂，但我们可以为我们的目的简化其功能。一个典型的神经元由三个主要部分参与信号传输：

1. **树突：** 这些树状分支充当输入通道。它们通过称为突触的连接从其他神经元接收电化学信号。
2. **胞体（细胞体）：** 神经元的中心部分。它收集并汇总树突接收到的输入信号。如果组合信号强度在短时间内超过某个激活阈值，胞体就会产生一个电脉冲。
3. **轴突：** 一个长的、电缆状的突起，将电脉冲（“输出信号”）从胞体传输出去。轴突终止于分支，这些分支与其他神经元的树突形成突触，从而将信号传递下去。

**突触**是神经元之间的连接点。它们不只是被动的连接；突触的强度或“权重 (weight)”决定了突触前（发送）神经元发出的信号对突触后（接收）神经元有多大影响。这种突触强度会随时间改变，这被认为是学习和记忆在大脑中的基本机制。

本质上，一个生物神经元接收多个加权输入（由突触强度调节的信号），在胞体中整合它们，如果总输入超过阈值，它会“发放”（放电），沿着轴突发送信号，从而可能激活其他神经元。

> 生物神经元的组成部分与人工神经元要素的简化比较。

### 从生物学到计算

人工神经元（我们将在下一节中对其进行数学定义）是一种受到这一生物过程启发的高度简化的抽象：

- **输入：** 对应于树突接收的信号。
- **权重 (weight)：** 类似于突触强度，决定了每个输入的影响。
- **求和：** 模拟胞体中信号的整合。
- **激活函数 (activation function)：** 代表胞体的阈值发放机制。如果加权输入总和超过某个值，人工神经元就会产生一个输出信号。
- **输出：** 对应于沿轴突传递的信号。

请记住，这是一种*启发*，而非直接复制。人工神经网络 (neural network)不模拟复杂的电化学动态、神经递质、精确的脉冲时间或大脑复杂的3D结构。它们捕捉了一个核心计算原理：组合加权输入并应用非线性激活。这种简化使得人工神经网络在计算上可行，并且能有效解决复杂问题，即使其底层机制与生物实际情况有很大不同。理解这一来源提供了背景，有助于我们接下来对人工神经元的数学形式进行阐述。

## 参考资料

- [A Logical Calculus of the Ideas Immanent in Nervous Activity](https://link.springer.com/article/10.1007/BF02478259) — Warren S. McCulloch and Walter Pitts (1943)
  Journal: Bulletin of Mathematical Biophysics; Publisher: Springer; Volume: 5; Pages: 115-133; DOI: [10.1007/BF02478259](https://doi.org/10.1007/BF02478259)
  这篇开创性论文提出了第一个神经元的数学模型，奠定了人工神经网络的基础。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本有影响力的教材全面介绍了深度学习，包括人工神经网络背后的历史背景和生物学启发。

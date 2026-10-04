---
course: "introduction-to-neural-networks"
chapter: "neural-network-foundations"
lesson: "feedforward-example"
sourceId: 1921
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-1-neural-network-foundations/feedforward-example"
title: "一个简单的前馈网络示例"
description: "呈现基本前馈神经网络的结构和信息流。"
order: 5
plots: []
sourceHash: "dcf86dfa11050171547a5849f2b99742f8cb5fec2dca2d475ed0539410300928"
sourceCorrections: []
---

让我们将这些部分组合起来，展示一个基本的前馈神经网络 (neural network)。我们已经讨论了单个神经元、它们的参数 (parameter)（权重 (weight)和偏置 (bias)）、激活函数 (activation function)，以及它们如何组织成层。现在，设想我们想构建一个简单的网络，例如，基于两个输入特征来预测一个单一的输出值。

考虑一个具有以下结构的网络：

- 一个**输入层**，包含2个神经元，对应我们的两个输入特征（$x_1, x_2$）。这一层不执行计算；它只是将输入值向前传递。
- 一个**隐藏层**，包含3个神经元（$h_1, h_2, h_3$）。每个隐藏神经元都接收来自输入层*所有*神经元的输入。
- 一个**输出层**，包含1个神经元（$o_1$）。这个神经元接收来自隐藏层*所有*神经元的输入，并产生最终预测值（$\hat{y}$）。

这被称为“前馈”网络，因为信息严格地沿一个方向流动：从输入层到隐藏层再到输出层，没有回环。

信息传播方式如下：

1. **输入到隐藏层：**

   - 每个输入特征（$x_1, x_2$）都连接到每个隐藏神经元（$h_1, h_2, h_3$）。
   - 对于第一个隐藏神经元（$h_1$），输入信号与相应的权重相乘（例如，从 $x_1 \to h_1$ 的 $w_{11}$，从 $x_2 \to h_1$ 的 $w_{21}$）。
   - 计算加权和（$z_{h1}$）：$z_{h1} = (x_1 \times w_{11}) + (x_2 \times w_{21}) + b_{h1}$，其中 $b_{h1}$ 是神经元 $h_1$ 的偏置。
   - 将激活函数 $f$（如ReLU或Sigmoid）应用于此和，得到神经元的输出：$a_{h1} = f(z_{h1})$。
   - 类似的计算（$z = \sum(weight \times input) + bias$，随后是 $a = f(z)$）独立地对其他隐藏神经元（$h_2, h_3$）进行，每个都使用自己的一组权重和偏置。
2. **隐藏到输出层：**

   - 隐藏层神经元的输出（$a_{h1}, a_{h2}, a_{h3}$）成为输出层的输入。
   - 每个隐藏神经元输出都连接到输出神经元（$o_1$）。
   - 对于输出神经元（$o_1$），使用隐藏层激活值和一组新权重（例如，从 $a_{h1} \to o_1$ 的 $w_{h1, o1}$ 等）以及它自己的偏置（$b_{o1}$）来计算加权和（$z_{o1}$）：$z_{o1} = (a_{h1} \times w_{h1, o1}) + (a_{h2} \times w_{h2, o1}) + (a_{h3} \times w_{h3, o1}) + b_{o1}$。
   - 将最终激活函数 $g$（可能与隐藏层的激活函数不同，取决于任务）应用于此和，以获得网络的预测值：$\hat{y} = a_{o1} = g(z_{o1})$。

整个过程，从输入初始值 $x_1, x_2$ 到获得最终输出 $\hat{y}$，称为**前向传播**，我们将在后续内容中进行详细讲解。

下面的图表呈现了这种简单的网络结构。

> 一个简单的前馈神经网络，包含2个输入神经元、1个由3个神经元组成的隐藏层和1个输出神经元。箭头表示前向传播过程中信息流的方向。每个连接代表一个权重，隐藏层和输出层中的每个神经元都有一个相关的偏置（未明确画出）。

"本示例体现了基本架构。网络可以有更多的层（使其“更深”）和每层更多的神经元，但加权和、激活函数以及分层连接的基本原理保持不变。层的具体数量、神经元的数量以及激活函数的选择，都是解决问题时需要作出的设计考量。在后续章节中，我们将学习如何高效地执行这些计算，以及网络如何找到其权重和偏置的合适值。"

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  权威教材，涵盖前馈神经网络的数学和概念基础，包括其架构和前向传播。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — Michael A. Nielsen (2015)
  Publisher: Determination Press
  备受赞誉的在线书籍，为神经网络提供了易于理解的介绍，包括前馈网络和前向传播机制的详细解释。
- [CS230: Deep Learning - Lecture Notes](http://cs230.stanford.edu/) — Andrew Ng and the Stanford CS230 Staff (2018-2019)
  Publisher: Stanford University
  顶尖大学课程的官方材料，为神经网络架构（包括前馈网络及其计算流程）提供基础解释。

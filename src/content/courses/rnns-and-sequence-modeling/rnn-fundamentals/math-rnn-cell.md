---
course: "rnns-and-sequence-modeling"
chapter: "rnn-fundamentals"
lesson: "math-rnn-cell"
sourceId: 2532
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-2-rnn-fundamentals/math-rnn-cell"
title: "RNN 单元的数学表述"
description: "展示简单 RNN 中隐藏状态和输出计算的方程。"
order: 4
plots: []
sourceHash: "17fc9f19d9d39ebe0892522f9eee88c38694353306d08bc599b91cc4d9378479"
sourceCorrections: []
---

RNN 逐个处理序列元素并更新隐藏状态。我们将形式化每个时间步 $t$ 在简单 RNN 单元内部进行的计算。

回想一下，该单元接收两个输入：序列的当前输入特征向量 (vector) $x_t$ 和在上一时间步计算的隐藏状态 $h_{t-1}$。其目标是计算新的隐藏状态 $h_t$，此状态包含从过去到当前步的信息，并且可能包含与此特定时间步相关的输出 $y_t$。

核心计算通常由以下两个方程定义：

1. **计算新的隐藏状态 ($h_t$):**

   
   $$
   h_t = f(W_{hh}h_{t-1} + W_{xh}x_t + b_h)
   $$
   
2. **计算输出 ($y_t$):**

   
   $$
   y_t = g(W_{hy}h_t + b_y)
   $$
   

我们来详细说明这些方程的每个组成部分：

- $x_t$: 这是当前时间步 $t$ 的输入向量。如果您在处理文本，这可以是位置 $t$ 处词语的嵌入 (embedding)向量。如果是时间序列数据，它可能是时间 $t$ 的传感器读数。假设 $x_t$ 的维度是 $D$（即 $x_t \in \mathbb{R}^D$）。
- $h_{t-1}$: 这是来自*上一*时间步 $t-1$ 的隐藏状态向量。它用作网络的历史记忆。我们假设隐藏状态的大小是 $N$（即 $h_{t-1} \in \mathbb{R}^N$）。对于第一个时间步 ($t=0$)， $h_{-1}$ 通常被初始化，常常是一个零向量。
- $h_t$: 这是为当前时间步 $t$ 新计算的隐藏状态向量。它的大小也是 $N$（即 $h_t \in \mathbb{R}^N$）。
- $W_{xh}$: 这是转换当前输入 $x_t$ 的权重 (weight)矩阵。它连接输入层到隐藏层。其维度必须与矩阵乘法兼容：如果 $x_t$ 是 $D$ 维的，且 $h_t$ 是 $N$ 维的，那么 $W_{xh}$ 必须具有 $N \times D$ 的形状。
- $W_{hh}$: 这是转换上一隐藏状态 $h_{t-1}$ 的权重矩阵。它表示从上一时间步的隐藏层到当前时间步的隐藏层的循环连接。其维度必须将一个 $N$ 维向量 ($h_{t-1}$) 连接到另一个 $N$ 维向量 ($h_t$)，因此 $W_{hh}$ 具有 $N \times N$ 的形状。
- $b_h$: 这是添加到隐藏状态计算中的偏置 (bias)向量。它与隐藏状态具有相同的维度 $N$（即 $b_h \in \mathbb{R}^N$）。
- $f$: 这是按元素应用以计算隐藏状态的激活函数 (activation function)。在简单 RNN 中，这通常是双曲正切函数（`tanh`）。`tanh` 函数将值压缩到 [-1, 1] 的范围，这有助于调整流经网络的激活值。其他函数如 ReLU 也可能被使用，但 `tanh` 是基本 RNN 隐藏状态的传统选择。

  
  $$
  \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
  $$
  
- $y_t$: 这是在时间步 $t$ 计算的输出向量。其维度，我们称之为 $K$（即 $y_t \in \mathbb{R}^K$），取决于具体任务。例如，在语言建模中，您可能预测下一个词，因此 $K$ 将是词汇量大小。在时间序列预测中，如果您预测单个值， $K$ 可能为 1。请注意，根据应用情况，不一定在每个时间步都需要输出。
- $W_{hy}$: 这是转换当前隐藏状态 $h_t$ 以生成输出 $y_t$ 的权重矩阵。它连接隐藏层到输出层。要将 $N$ 维隐藏状态转换为 $K$ 维输出， $W_{hy}$ 必须具有 $K \times N$ 的形状。
- $b_y$: 这是添加到输出计算中的偏置向量。它与输出具有相同的维度 $K$（即 $b_y \in \mathbb{R}^K$）。
- $g$: 这是按元素应用以计算最终输出 $y_t$ 的激活函数。 $g$ 的选择很大程度上取决于任务的性质。

  - 对于回归任务（如预测连续值）， $g$ 可能只是恒等函数（不应用激活）。
  - 对于每一步的二元分类， $g$ 通常是 sigmoid 函数。
  - 对于多类别分类（如从词汇表 (vocabulary)中预测下一个词）， $g$ 通常是 `softmax` 函数。

一个重要的方面是 **参数 (parameter)共享**。*相同*的一组权重矩阵 ($W_{xh}$，$W_{hh}$，$W_{hy}$) 和偏置向量 ($b_h$，$b_y$) 用于在*每个时间步*执行计算。这使得模型计算效率高，并使其能够将序列中某一点学到的模式推广到其他点，而与序列长度无关。网络学习一组单一的规则，用于如何根据当前输入和其记忆来更新状态并生成输出。

这是一个单个 RNN 单元内计算过程的可视化：

> 一个简单 RNN 单元内的计算流程。输入 $x_t$ 和 $h_{t-1}$ 分别通过权重矩阵 $W_{xh}$ 和 $W_{hh}$ 转换，与偏置 $b_h$ 相加，经过激活函数 $f$ 产生 $h_t$。然后 $h_t$ 通过 $W_{hy}$ 转换，与偏置 $b_y$ 相加，并经过激活函数 $g$ 产生 $y_t$。

这些方程定义了信息通过 RNN 单元的单个时间步前向传播。在训练期间，网络的参数 ($W_{xh}$，$W_{hh}$，$W_{hy}$，$b_h$，$b_y$) 会进行调整，以最小化基于预测输出 $y_t$ 与真实目标值对比的损失函数 (loss function)。此学习过程涉及计算梯度，在 RNN 的背景下，这需要一种称为随时间反向传播 (backpropagation)（BPTT）的技术，这是我们接下来将讨论的话题。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 10
  这本权威教科书为循环神经网络提供了全面的数学和理论基础，涵盖了基本单元公式及其组成部分。
- [Finding structure in time](https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1) — Jeffrey L. Elman (1990)
  Journal: Cognitive Science; Publisher: John Wiley & Sons; Volume: 14; Pages: 179-211; DOI: [10.1207/s15516709cog1402_1](https://doi.org/10.1207/s15516709cog1402_1)
  这篇基础论文介绍了简单循环网络（Elman网络），明确描述了构成所讨论的RNN单元基础的架构和循环连接。

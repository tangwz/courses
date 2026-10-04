---
course: "rnns-and-sequence-modeling"
chapter: "long-short-term-memory-lstm"
lesson: "lstm-information-flow"
sourceId: 2575
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-5-long-short-term-memory-lstm/lstm-information-flow"
title: "LSTM单元中的信息流动"
description: "展示LSTM单元中信息和梯度流经门与状态的路径。"
order: 7
plots: []
sourceHash: "c16b999e28c45ae6f5728a62ac89c2561b2210c90edf5faefb7f3cd431f6a0f7"
sourceCorrections: []
---

信息在长短期记忆 (LSTM) 单元中于单个时间步内以特定的方式流动。这种流动对于理解LSTM如何有效地处理长序列上下文 (context)，以及如何弥补简单循环神经网络 (neural network) (RNN) 的不足之处非常重要。

在每个时间步 $t$，LSTM单元接收以下三个输入：

1. 当前输入向量 (vector) $x_t$。
2. 前一个隐藏状态 $h_{t-1}$。
3. 前一个细胞状态 $c_{t-1}$。

这些输入与门和细胞状态交互，产生两个输出：

1. 新的隐藏状态 $h_t$。
2. 新的细胞状态 $c_t$。

让我们按照数据路径进行说明：

### 1. 遗忘门：决定舍弃什么

第一步涉及遗忘门（$f_t$）。它的作用是决定旧细胞状态 $c_{t-1}$ 的哪些部分不再相关并应被丢弃。它会查看前一个隐藏状态 $h_{t-1}$ 和当前输入 $x_t$。一个Sigmoid激活函数 (activation function)（$\sigma$）会将细胞状态向量 (vector)中每个数值的输出压缩到0和1之间。


$$
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
$$


这里，$[h_{t-1}, x_t]$ 表示这两个向量的拼接。$W_f$ 和 $b_f$ 分别是遗忘门的权重 (weight)矩阵和偏置 (bias)向量，它们在训练期间学习得到。

输出为1表示“完全保留此信息”，而输出为0则表示“完全丢弃此信息”。这个门的输出 $f_t$ 随后与前一个细胞状态 $c_{t-1}$ 进行按元素相乘（$\odot$）。

### 2. 输入门：决定存储什么新信息

接下来，单元需要确定当前输入 $x_t$ 和前一个隐藏状态 $h_{t-1}$ 的哪些新信息应该添加到细胞状态中。这涉及两部分：

- **输入门层（$i_t$）：** 另一个Sigmoid层决定我们将更新哪些值。
  
  $$
  i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
  $$
  
- **候选值（$\tilde{c}_t$）：** 一个$tanh$层创建一个新的候选值向量 (vector)，这些值*可能*被添加到状态中。
  
  $$
  \tilde{c}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)
  $$
  

$W_i, b_i$ 和 $W_C, b_C$ 分别是这些层的权重 (weight)和偏置 (bias)。$tanh$ 函数的输出值在-1和1之间。

### 3. 更新细胞状态：结合旧与新

现在我们将旧的细胞状态 $c_{t-1}$ 更新为新的细胞状态 $c_t$。我们结合遗忘门和输入门的结果：

- 首先，我们应用遗忘门的决定：$f_t \odot c_{t-1}$。这会丢弃标记 (token)为遗忘的信息。
- 然后，我们确定要添加的新信息：$i_t \odot \tilde{c}_t$。这根据我们决定更新每个状态值的程度来缩放候选值。
- 最后，我们将这两部分加在一起：
  
  $$
  c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
  $$
  

这种加性交互与简单RNN中重复的矩阵乘法有显著的不同。它使得梯度在反向传播 (backpropagation)过程中能够更轻松地通过时间流动，从而减轻梯度消失问题。细胞状态就像一条传送带，传输信息，只伴随着微小的线性交互（与 $f_t$ 相乘和加上 $i_t \odot \tilde{c}_t$），使得在许多步骤中保持上下文 (context)变得更容易。

### 4. 输出门：决定输出什么

最后，我们需要决定隐藏状态 $h_t$（以及可能作为此时间步的输出）应该是什么。这个输出将是细胞状态 $c_t$ 的一个过滤版本。

- **输出门层（$o_t$）：** 一个Sigmoid层确定细胞状态的哪些部分将作为输出。
  
  $$
  o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
  $$
  
- **过滤细胞状态：** 我们将更新后的细胞状态 $c_t$ 通过$tanh$函数（将值压缩到-1和1之间），然后将其按元素与输出门 $o_t$ 的输出相乘：
  
  $$
  h_t = o_t \odot \tanh(c_t)
  $$
  

产生的 $h_t$ 是传递到下一个时间步的隐藏状态。如果需要用于预测，它也可以作为单元在时间步 $t$ 的输出。$W_o$ 和 $b_o$ 是输出门的权重 (weight)和偏置 (bias)。

### 流程可视化

以下图表说明了这些组件如何连接以及数据在单个时间步内如何通过LSTM单元流动：

> 该图说明了在单个时间步 $t$ 内，信息和计算在LSTM单元中的流动方式。输入 $x_t$、$h_{t-1}$、$c_{t-1}$ 经过遗忘门（$f_t$）、输入门（$i_t$）和输出门（$o_t$）以及候选状态（$\tilde{c}_t$）的处理，以计算新的细胞状态 $c_t$ 和隐藏状态 $h_t$。Sigmoid（$\sigma$）和$tanh$激活函数 (activation function)控制着门控和状态更新。按元素相乘（$\odot$）和加法（+）组合中间结果。红色虚线表示 $c_t$ 和 $h_t$ 传递到下一个时间步。

通过精心调节在每个步骤中哪些信息被保留、丢弃、添加和输出，LSTM单元为梯度在训练期间更有效地流动创建了通道。细胞状态作为显式记忆通道，受门保护，使得网络能够在长时间内学习和记忆信息，这对处理复杂的序列建模任务非常重要。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  这篇开创性论文介绍了长短期记忆（LSTM）架构，详细阐述了其基本设计，旨在解决循环神经网络中的梯度消失和梯度爆炸问题。
- [Deep Learning](http://www.deeplearningbook.org) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础性教材，其中包含详细的循环神经网络章节，提供了对LSTM架构及其运行流程的全面解释。
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — Christopher Olah (2015)
  一篇广受好评、高度可视化的博客文章，通过直观的分步解释展示了LSTM单元的运作方式，使信息流易于理解。

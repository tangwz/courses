# GRU 单元结构

来源：[原文](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-6-gated-recurrent-units-gru/gru-cell-architecture)

[返回章节目录](README.md) · [返回课程目录](../README.md)

门控循环单元（GRU）是一种循环神经网络 (neural network) (RNN)架构，旨在有效捕捉序列数据中的依赖关系。它们通过使用门控机制来实现这一点，这些机制能够调节信息流。一个 GRU 单元包含两个主要门：**重置门**和**更新门**。这些门控制着新输入数据和先前的隐藏状态信息如何被用于更新当前的隐藏状态。GRU 通过将单元状态和隐藏状态的功能直接合并为一个单一的隐藏状态向量 (vector) $h_t$ 来简化其内部结构。这种设计通常比其他循环架构（例如 LSTM，其通常使用三个门以及单独的单元和隐藏状态）更为简单。

现在，我们来了解在时间步 $t$ 的单个 GRU 单元内的组成部分和信息流。该单元接收当前输入 $x_t$ 和来自上一时间步的隐藏状态 $h_{t-1}$。然后，它计算新的隐藏状态 $h_t$，该状态也作为此时间步的输出。

组成部分包括：

1. **重置门 ($r_t$)**：决定如何将新输入 $x_t$ 与之前的隐藏状态 $h_{t-1}$ 组合以计算候选隐藏状态。具体来说，它控制在提出新状态时，应“遗忘”或忽略多少之前的状态信息。
2. **更新门 ($z_t$)**：决定应保留多少之前的隐藏状态 $h_{t-1}$ 以及应将多少新计算的候选隐藏状态 $\tilde{h}_t$ 包含在最终隐藏状态 $h_t$ 中。
3. **候选隐藏状态 ($\tilde{h}_t$)**：下一个隐藏状态的“建议”值，根据当前输入和之前隐藏状态的潜在重置版本计算得出。
4. **最终隐藏状态 ($h_t$)**：当前时间步 GRU 单元的输出状态，通过在之前状态 $h_{t-1}$ 和候选状态 $\tilde{h}_t$ 之间进行插值计算得出，由更新门引导。

以下是 GRU 单元结构的视图：

> GRU 单元结构的一个简化视图。输入 $x_t$ 和 $h_{t-1}$ 送入重置门 ($r_t$) 和更新门 ($z_t$)。重置门调节 $h_{t-1}$ 对候选状态 $h̃_t$ 的影响。更新门控制 $h_{t-1}$ 和 $h̃_t$ 之间的混合，以生成最终输出 $h_t$。

现在，我们来看看单元内执行的计算。

### 重置门 ($r_t$)

重置门决定在计算候选隐藏状态 $\tilde{h}_t$ 时，应忽略多少来自之前隐藏状态 $h_{t-1}$ 的信息。它以当前输入 $x_t$ 和之前隐藏状态 $h_{t-1}$ 作为输入。

计算方式如下：


$$
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
$$


这里，$W_r$ 和 $U_r$ 是权重 (weight)矩阵，$b_r$ 是偏置 (bias)向量 (vector)，$\sigma$ 是 sigmoid 激活函数 (activation function)。sigmoid 函数输出 0 到 1 之间的值。一个接近 0 的值表示“重置”或忽略之前状态中对应的元素，而一个接近 1 的值表示在计算候选状态时“保留”它。

### 更新门 ($z_t$)

更新门控制隐藏状态更新新信息的程度以及保留旧信息的程度。它决定了之前隐藏状态 $h_{t-1}$ 有多少会传递到最终隐藏状态 $h_t$。与重置门类似，它使用当前输入 $x_t$ 和之前隐藏状态 $h_{t-1}$。

计算方式如下：


$$
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
$$


同样，$W_z$、$U_z$ 和 $b_z$ 是学习参数 (parameter)（权重 (weight)和偏置 (bias)），$\sigma$ 是 sigmoid 函数。$z_t$ 的值接近 1 表示新的隐藏状态 $h_t$ 主要应基于候选状态 $\tilde{h}_t$，而接近 0 的值则表示保留大部分之前的状态 $h_{t-1}$。

### 候选隐藏状态 ($\tilde{h}_t$)

候选隐藏状态的计算方式与简单 RNN 中的隐藏状态类似，但涉及重置门的修改。它的目标是捕获来自当前输入 $x_t$ 的新信息，并可能通过之前状态 $h_{t-1}$ 的相关部分进行调节。

计算涉及当前输入 $x_t$ 和之前隐藏状态 $h_{t-1}$，并与重置门的输出 $r_t$ 进行元素级乘法 ($\odot$)：


$$
\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)
$$


这里，$W_h$、$U_h$ 和 $b_h$ 是学习参数 (parameter)。`tanh` 激活函数 (activation function)将输出压缩到 -1 和 1 之间。元素级乘法 $r_t \odot h_{t-1}$ 很重要：如果 $r_t$ 中的一个元素接近 0，则来自 $h_{t-1}$ 的相应元素对 $\tilde{h}_t$ 的计算贡献很小，从而允许单元在生成候选状态时有效地“遗忘”不相关的过去信息。

### 最终隐藏状态 ($h_t$)

当前时间步的最终隐藏状态 $h_t$ 是通过在之前隐藏状态 $h_{t-1}$ 和候选隐藏状态 $\tilde{h}_t$ 之间进行线性插值计算的。更新门 $z_t$ 控制此插值。

计算方式如下：


$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$


此方程展示了 GRU 如何更新其状态。向量 (vector) $z_t$ 进行元素级操作：

- 如果 $z_t$ 的一个元素接近 1，$h_t$ 中的相应元素将主要由候选状态 $\tilde{h}_t$ 决定，从而加入新信息。
- 如果 $z_t$ 的一个元素接近 0，$h_t$ 中的相应元素将主要由之前状态 $h_{t-1}$ 决定，从而保留过去信息。

这种机制使 GRU 能够在长序列中保持信息（当 $z_t$ 在许多时间步接近 0 时），或者根据新输入快速更新（当 $z_t$ 接近 1 时）。值得注意的是，GRU 没有像 LSTM 那样独立的单元状态；隐藏状态 $h_t$ 携带所有必要信息前进。

这种结构，凭借其两个门和组合状态表示，提供了一种有效但可能计算效率更高的方式来处理序列数据，与 LSTM 相比，我们将在本章后面进行更直接的比较。

## 参考资料

- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio (2014)
  Journal: EMNLP 2014; Pages: 1724-1734; DOI: [10.48550/arXiv.1406.1078](https://doi.org/10.48550/arXiv.1406.1078)
  介绍门控循环单元（GRU）作为循环神经网络新架构的奠基性论文。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 10 (Sequence Modeling: Recurrent and Recursive Nets)
  提供循环神经网络的全面理论概述，包括 GRU 架构及其机制的详细解释。
- [torch.nn.GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html) — PyTorch Core Team (2024)
  Journal: PyTorch Documentation; Publisher: PyTorch Foundation
  PyTorch 中 GRU 模块的官方文档，详细说明其参数、数学公式和在实际深度学习应用中的用法。

---

[上一节](01-%E4%BB%8B%E7%BB%8DGRU%EF%BC%9A%E4%B8%80%E7%A7%8D%E6%9B%B4%E7%AE%80%E6%B4%81%E7%9A%84%E9%97%A8%E6%8E%A7%E6%9E%B6%E6%9E%84.md) · [下一节](03-%E6%9B%B4%E6%96%B0%E9%97%A8.md)

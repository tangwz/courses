---
course: "nlp-fundamentals"
chapter: "nlp-sequence-models-intro"
lesson: "lstm-networks"
sourceId: 2506
sourceUrl: "https://apxml.com/zh/courses/nlp-fundamentals/chapter-5-nlp-sequence-models-intro/lstm-networks"
title: "长短期记忆（LSTM）网络"
description: "理解LSTM的架构以及它们如何解决梯度消失问题。"
order: 4
plots: []
sourceHash: "9c3fa79721e35ea1e338b3e4df7034432805157b9604e6efc417e1bcc4931c65"
sourceCorrections: []
---

循环神经网络 (neural network)（RNN）在学习序列中的长距离依赖关系时面临挑战。主要的困难在于梯度消失问题，这导致在反向传播 (backpropagation)过程中，梯度变得过小，无法有效更新网络在较早时间步的权重 (weight)。这种局限性阻碍了它们捕捉远距离元素间关系的能力。为了解决这一问题，一种更先进的循环架构被提出：长短期记忆（LSTM）网络，由Hochreiter和Schmidhuber于1997年首次引入。

LSTM 被明确设计用于长期记忆信息。其核心改进是*细胞状态*（$C_t$），常被视为网络的“记忆”，它能够在许多时间步中相对不变地传递信息。与简单RNN中单一的隐藏状态变换不同，LSTM使用一套*门*系统来精细调节信息进出此细胞状态。

### LSTM细胞结构

一个LSTM单元顺序处理数据，接收当前时间步的输入（$x_t$）和来自前一时间步的隐藏状态（$h_{t-1}$），以计算新的隐藏状态（$h_t$）并更新其内部细胞状态（$C_t$）。这个过程由三个主要门控制：

1. **遗忘门（$f_t$）：** 这个门决定从细胞状态中丢弃哪些信息。它会查看$h_{t-1}$和$x_t$，并为前一细胞状态$C_{t-1}$中的每个数值输出一个0到1之间的数。1表示“完全保留”，而0表示“完全丢弃”。它使用sigmoid激活函数 (activation function)（$\sigma$）。
   $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$
   其中，$[h_{t-1}, x_t]$表示前一隐藏状态和当前输入的拼接，$W_f$是权重 (weight)，$b_f$是遗忘门的偏置 (bias)。
2. **输入门（$i_t$）：** 这个门决定哪些新信息将被存储到细胞状态中。它由两部分组成：

   - 首先，一个sigmoid层决定哪些值需要更新（$i_t$）。
     $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$
   - 其次，一个`tanh`层创建新的候选值向量 (vector)$\tilde{C}_t$，这些值可以添加到状态中。
     $\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)$
     这两部分结合起来更新细胞状态。
3. **更新细胞状态：** 旧细胞状态$C_{t-1}$更新为新细胞状态$C_t$。我们将旧状态乘以$f_t$，从而遗忘我们之前决定遗忘的信息。然后我们添加$i_t * \tilde{C}_t$。这是新的候选信息，根据我们决定更新每个状态值的程度进行缩放。
   $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$
   注意这里的加法运算。这种加法交互对于梯度更好地流动很重要，相比之下，简单RNN中是重复的乘法运算。
4. **输出门（$o_t$）：** 这个门决定下一个隐藏状态$h_t$（此时间步的输出）应该是什么。输出将基于过滤后的细胞状态。

   - 首先，一个sigmoid层决定细胞状态的哪些部分需要输出。
     $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$
   - 然后，将（已更新的）细胞状态$C_t$通过`tanh`函数（将值推到-1和1之间），并乘以sigmoid门$o_t$的输出。这确保我们只输出我们决定要输出的部分。
     $h_t = o_t * \tanh(C_t)$

隐藏状态$h_t$随后被传递到下一个时间步（作为$h_{t-1}$），也可以作为模型在当前时间步的输出。

### LSTM细胞的可视化

一个LSTM细胞的结构，其门控着信息流经细胞状态，可以如下所示：

> 单个LSTM细胞内的数据流。箭头表示数据依赖关系。标有'X'的圆表示按元素乘法，'+'表示按元素加法，'\u03c3'表示sigmoid函数，'tanh'表示双曲正切函数。细胞状态就像一条传送带，门控着信息的移除和添加。

### LSTM如何缓解梯度消失

LSTM能够优于简单RNN处理长距离依赖关系的机制在于细胞状态和门控机制。

- **加法更新：** 细胞状态$C_t$主要通过加法更新（$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$）。这种加法交互使得梯度更容易随时间反向流动，而不会像简单RNN中重复的矩阵乘法那样快速消失。
- **门控：** 门充当调节器。遗忘门可以学习通过保持其激活值接近1来保留细胞状态中某些来自遥远过去的重要信息。同样，输入门可以学习阻止不相关的新信息破坏细胞状态。这种选择性记忆使网络能够在长序列中保持上下文 (context)。

可以将细胞状态视为一条随时间传递信息的传送带。遗忘门从传送带上移除项目，输入门添加新项目，输出门从传送带上读取项目以决定即时输出（隐藏状态）。这种结构在长时间内保留信号方面更有效。

LSTM在各种NLP任务中取得了巨大成功，包括机器翻译、情感分析和序列标注，这正是因为它们能够捕获长距离上下文。虽然它们比简单RNN更复杂，但其有效性通常足以弥补额外的计算开销。

在下一节中，我们将介绍门控循环单元（GRU），它是LSTM的一个略微简化的变体，通常能达到相当的性能。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍长短期记忆（LSTM）网络的原始学术论文。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 10
  一本综合性教材，在循环神经网络的背景下，详细解释了LSTM的理论。

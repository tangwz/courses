# 长短期记忆（LSTM）门控机制

来源：[原文](https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-1-revisiting-sequence-modeling-limitations/lstm-gating-mechanisms)

[返回章节目录](README.md) · [返回课程目录](../README.md)

简单循环神经网络 (neural network)（RNN）内部的核心数学运算，即跨时间步重复进行的矩阵乘法，直接导致了梯度消失和梯度爆炸问题。训练深度循环网络变得不稳定，使得捕获序列中相距较远元素之间的依赖关系变得困难。长短期记忆（LSTM）网络由Hochreiter和Schmidhuber于1997年提出，专门设计用于通过一个更精巧的内部结构来解决这些问题，该结构内含*门控机制*。

LSTM的核心创新是除了隐藏状态（$h_t$）之外，还引入了**单元状态**（$C_t$）。可以将单元状态想象成一条信息高速公路或记忆传送带。它贯穿整个序列，只有轻微的线性交互。信息可以被添加到单元状态中或从其中移除，这些操作由称为**门**的结构精确地调控。

这些门由一个sigmoid神经网络层和一个逐点乘法操作组成。sigmoid层输出0到1之间的数字，用于描述每个分量应该通过多少。值为1表示“让所有信息通过”，而值为0表示“不让任何信息通过”。一个LSTM单元通常包含三个这样的门，以保护和控制单元状态。

让我们在特定时间步 $t$ 查看每个门，考虑当前输入 $x_t$、前一个隐藏状态 $h_{t-1}$ 和前一个单元状态 $C_{t-1}$。

### 遗忘门 ($f_t$)

第一步是决定我们要从单元状态中丢弃哪些信息。这个决定由遗忘门做出。它查看 $h_{t-1}$ 和 $x_t$，并为单元状态 $C_{t-1}$ 中的每个数字输出一个介于0和1之间的值。


$$
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
$$


这里，$[h_{t-1}, x_t]$ 表示前一个隐藏状态和当前输入向量 (vector)的拼接。$W_f$ 代表遗忘门的权重 (weight)矩阵，$b_f$ 代表其偏置 (bias)向量。sigmoid函数 $\sigma$ 将输出压缩到 [0, 1] 的范围。接近0的值表示忘记 $C_{t-1}$ 中相应的信息，而接近1的值表示保留它。

### 输入门 ($i_t$) 和候选值 ($\tilde{C}_t$)

接下来，我们需要决定将哪些新信息存储到单元状态中。这包括两个部分：

1. **输入门** ($i_t$) 是另一个sigmoid层，它决定我们将更新哪些值。
2. 一个 `tanh` 层创建新的候选值向量 (vector) $\tilde{C}_t$，这些值可以被添加到状态中。


$$
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
$$


$$
\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)
$$


类似于遗忘门，$W_i, b_i, W_C, b_C$ 是在训练过程中学到的权重 (weight)矩阵和偏置 (bias)向量。`tanh` 函数输出介于-1和1之间的值，代表对单元状态的潜在更新（正向或负向）。

现在，我们将旧的单元状态 $C_{t-1}$ 更新为新的单元状态 $C_t$。我们将旧状态乘以 $f_t$，忘记了我们之前决定忘记的内容。然后我们加上 $i_t * \tilde{C}_t$。这是新的候选值，根据我们决定更新每个状态值的程度进行缩放。


$$
C_t = f_t * C_{t-1} + i_t * \tilde{C}_t
$$


符号 $*$ 表示逐元素乘法。

### 输出门 ($o_t$)

最后，我们需要决定输出什么。这个输出将基于我们的单元状态，但会是一个经过筛选的版本。首先，我们运行一个sigmoid层，它决定我们将输出单元状态的哪些部分。


$$
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
$$


然后，我们将单元状态 $C_t$ 通过 `tanh` （将值推至-1和1之间）并将其乘以sigmoid门 $o_t$ 的输出，这样我们只输出了我们决定要输出的部分。这个结果就是新的隐藏状态 $h_t$。


$$
h_t = o_t * \tanh(C_t)
$$


这个 $h_t$ 被传递到下一个时间步，也可以作为当前时间步LSTM单元的输出进行预测。

> LSTM单元的内部结构。门（sigmoid $\sigma$）控制信息进出单元状态 ($C_t$) 的流动，由绿色路径表示。隐藏状态 ($h_t$) 是单元状态的筛选版本。

### 门控如何缓解梯度消失问题

主要的发现是单元状态的加性交互。新信息被*添加*到单元状态中（通过 $i_t * \tilde{C}_t$），旧信息被*移除*（通过乘以 $f_t$），而不像简单RNN那样通过矩阵乘法和非线性操作重复转换。遗忘门允许单元状态在需要时长时间保留信息（通过将 $f_t$ 设置接近1）。

这种结构创建了梯度可以反向传播 (backpropagation)而不迅速消失的路径。门学习控制这种流动，根据上下文 (context)打开或关闭对单元状态的访问。如果遗忘门大部分是打开的 ($f_t \approx 1$) 并且输入门大部分是关闭的 ($i_t \approx 0$)，单元状态可以将其信息在许多时间步内大致不变地传递，从而保留梯度。

尽管LSTM代表了显著的进展，并使得许多以前对简单RNN来说难以处理的序列建模任务取得了进展，但它们并不是一个完美的解决方案。它们仍然按顺序处理信息，限制了训练和推理 (inference)过程中的并行化。此外，尽管它们在捕获更长依赖方面远优于简单RNN，但它们在处理数千时间步中存在细微依赖的极长序列时仍然可能遇到困难。门控机制的复杂性也增加了相比简单模型的计算开销。这些未解决的问题为Transformer等架构的出现创造了条件，该架构完全放弃了循环。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍长短期记忆（LSTM）架构的原始论文，详细阐述了其旨在克服循环神经网络中梯度消失问题而进行的设计。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  第10章“序列建模：循环和递归网络”对LSTM进行了严谨的数学处理，介绍了其历史及其在深度学习中的作用。
- [Lecture Notes on Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM)](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGlz73W3xiTPb7RhQtX4h3T70hc7EtqNekpZn4buhPSgJMgTYRVBxAgYI1szC3SVpe6Rlgu5eJxJ7j2M3pKvfVVUeCwMQ48OhrMvnmYhVE3tkrW_c9FuLFiscYDmNG9HkDhx-1wYMFL-lNuk3_o9oTSMIvbdekh_3wu3nhFZNKIdyu70dldP6QlQUvmKm051T1HiDViLL-NihXvuDs=) — Abigail See, Chris Potts (2019)
  Publisher: Stanford University
  来自一所顶尖大学自然语言处理课程的讲义，提供了对RNN、梯度消失问题和LSTM工作机制的结构化和深入解释。

---

[上一节](02-%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E4%B8%8E%E6%A2%AF%E5%BA%A6%E7%88%86%E7%82%B8%E9%97%AE%E9%A2%98.md) · [下一节](04-%E9%97%A8%E6%8E%A7%E5%BE%AA%E7%8E%AF%E5%8D%95%E5%85%83%20%28GRU%29%20%E6%9E%B6%E6%9E%84.md)

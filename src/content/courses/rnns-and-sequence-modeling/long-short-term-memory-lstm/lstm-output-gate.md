---
course: "rnns-and-sequence-modeling"
chapter: "long-short-term-memory-lstm"
lesson: "lstm-output-gate"
sourceId: 2573
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-5-long-short-term-memory-lstm/lstm-output-gate"
title: "输出门"
description: "详细说明输出门在生成隐藏状态中的功能和计算。"
order: 6
plots: []
sourceHash: "97ac98e5b97e6a4a4de18481592285bde712a42a728a9e760e19aee17e13b583"
sourceCorrections: []
---

长短期记忆（LSTM）单元通过多个门控机制管理其内部状态（$C_t$）。遗忘门负责舍弃不相关信息，输入门则整合新信息。**输出门**是LSTM单元的最后一个组成部分。这个门决定更新后的细胞状态哪些部分应作为隐藏状态（$h_t$）传递到下一个时间步，并可能作为当前时间步的网络输出。

将细胞状态（$C_t$）视为LSTM的内部长期记忆。输出门起到过滤作用，决定该记忆的哪些方面与紧接的下一步和当前输出相关。并非细胞状态中存储的所有信息都立即需要，输出门允许LSTM选择性地显示仅相关联的部分。

### 过滤细胞状态

输出门的操作包含两个主要步骤：

1. **决定细胞状态的哪些部分输出：** 与其他门类似，这涉及一个sigmoid层。它接收前一个隐藏状态（$h_{t-1}$）和当前输入（$x_t$），并为细胞状态中的每个数值输出0到1之间的值。接近1的值表示“让这部分通过”，而接近0的值表示“阻止这部分”。这个决定向量 (vector)通常表示为 $o_t$。

   计算公式为：

   
   $$
   o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
   $$
   

   这里，$\sigma$ 是sigmoid激活函数 (activation function)，$W_o$ 代表权重 (weight)矩阵，$b_o$ 是输出门的偏置 (bias)向量。项 $[h_{t-1}, x_t]$ 表示前一个隐藏状态和当前输入向量在与权重相乘前进行了拼接。
2. **生成隐藏状态：** 更新后的细胞状态（$C_t$，由遗忘门和输入门操作产生）首先通过双曲正切（$\tanh$）激活函数。这会将值压缩到-1和1之间，有助于调整网络信号的数值范围。然后，这个转换后的细胞状态与输出门的激活向量（$o_t$）进行元素级乘法（$\odot$）。这种乘法起到过滤作用：$\tanh(C_t)$ 中对应 $o_t$ 中接近零值的部分会减弱，而对应 $o_t$ 中接近一值的部分则基本保持不变地通过。这个过滤的结果就是新的隐藏状态，$h_t$。

   计算公式为：

   
   $$
   h_t = o_t \odot \tanh(C_t)
   $$
   

这个结果 $h_t$ 有两个用途：它是当前时间步的输出（如果层配置为返回序列或它是最终输出层），并且它成为序列中下一个时间步的 $h_{t-1}$。

### 输出门数据流示意

以下图表示意了输出门机制内的数据流，以及它如何产生隐藏状态 $h_t$：

> 使用当前输入（$x_t$）、前一个隐藏状态（$h_{t-1}$）和更新后的细胞状态（$C_t$）计算输出门激活（$o_t$）和最终隐藏状态（$h_t$）的数据流。

### 重要性

输出门是LSTM发挥作用的一个重要环节。通过在生成隐藏状态前过滤细胞状态，LSTM能够维持一个丰富的内部表示（$C_t$），其中包含可能在多个时间步内都相关的信息，同时只显示当前上下文 (context)或任务所需的必要信息（$h_t$）。这种受控显示有助于防止内部状态在某些部分不立即相关时直接在后续计算中引起问题，从而与简单RNN相比，有助于获得更稳定的梯度和更好的长距离依赖学习。

总之，输出门通过以下方式完成LSTM循环：

1. 使用sigmoid函数来确定细胞状态的哪些部分与输出相关（$o_t$）。
2. 对细胞状态应用 $\tanh$ 函数以进行缩放。
3. 将缩放后的细胞状态与输出门的过滤器（$o_t$）相乘，以生成隐藏状态（$h_t$）。

这个隐藏状态 $h_t$ 随后将过滤后的相关信息传递到下一个时间步，影响未来的计算，并可能作为网络在当前步骤的输出。理解这三个门（遗忘、输入、输出）和细胞状态是理解LSTM网络在序列建模中作用的基础。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: The MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍长短期记忆（LSTM）网络架构的原始研究论文，提供了包括输出门在内的其组件的基础描述。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本权威教材，对循环神经网络提供了全面的解释，包括LSTM单元及其门控机制的详细结构和功能。
- [CS224n: Natural Language Processing with Deep Learning - Lecture Notes](http://web.stanford.edu/class/cs224n/) — Christopher Manning (2023)
  Publisher: Stanford University
  来自顶尖大学课程的教育资料，对LSTM网络提供了清晰易懂的解释，侧重于各种门（如输出门）的作用和操作。请参阅关于循环神经网络和LSTM的讲义。

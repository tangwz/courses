---
course: "rnns-and-sequence-modeling"
chapter: "long-short-term-memory-lstm"
lesson: "lstm-input-gate"
sourceId: 2570
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-5-long-short-term-memory-lstm/lstm-input-gate"
title: "输入门"
description: "详细说明输入门在更新细胞状态时的功能和计算方法。"
order: 4
plots: []
sourceHash: "49ad6c4fa08d17a5e2c3d46cdc03e41e55f3b7c0aa26d24d76a22eb492d209ad"
sourceCorrections: []
---

简单RNN在处理长序列时会遇到困难，因为在随时间反向传播 (backpropagation)过程中梯度可能消失或爆炸。LSTM引入了*门控机制*，正是为解决此问题而设。这些门像是调节器，仔细控制信息如何流入、流经和流出LSTM的核心记忆部件——细胞状态。

这些重要的调节器之一就是**输入门**。它的作用是决定从当前输入($x_t$)和前一个隐藏状态($h_{t-1}$)中，*哪些新信息*应该被存入细胞状态($C_t$)。它并非独立运作；它与遗忘门（遗忘门决定从*旧*细胞状态中舍弃什么）一同工作，以有效管理细胞的记忆。

输入门的决策过程包含两个主要部分：

1. **决定更新哪些值：** 首先，一个sigmoid层决定细胞状态的哪些部分应该被更新。sigmoid函数，常用$\sigma$表示，将其输入压缩到0到1的范围。一个接近1的值表示“允许此信息通过”，而一个接近0的值表示“阻止此信息”。该层接收前一个隐藏状态($h_{t-1}$)和当前输入($x_t$)，并生成一个输出向量 (vector)，我们称之为$i_t$。

   计算公式如下：

   
   $$
   i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
   $$
   

   这里，$[h_{t-1}, x_t]$ 表示前一个隐藏状态与当前输入向量的拼接。$W_i$是权重 (weight)矩阵，$b_i$是专门用于输入门这一部分的偏置 (bias)向量。sigmoid函数($\sigma$)是逐元素应用的。$i_t$中的每个元素都对应于细胞状态中的一个元素，作为该特定元素的过滤器或门值。
2. **生成候选值：** 同时，一个`tanh`层生成一个*新候选值*向量，记作$\tilde{C}_t$（读作“C-tilde sub t”）。这些是*可能*被添加到细胞状态的潜在值。与sigmoid层类似，该层也使用前一个隐藏状态($h_{t-1}$)和当前输入($x_t$)。`tanh`激活函数 (activation function)将其输入压缩到-1到1的范围。

   计算公式如下：

   
   $$
   \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
   $$
   

   同样，$W_C$和$b_C$是针对这个特定层的权重矩阵和偏置向量。输出$\tilde{C}_t$表示从当前输入和前一个语境中提取的*新信息*，其值缩放到-1到1之间。

可以将$i_t$看作看门人，决定每个潜在的新信息片段（$\tilde{C}_t$）*有多少*应该被实际考虑加入记忆。$\tilde{C}_t$包含潜在的更新，而$i_t$包含用于缩放这些候选值的过滤值（介于0和1之间）。

> 该图显示了输入门的两个组成部分。它接收当前输入($x_t$)和前一个隐藏状态($h_{t-1}$)，通过并行的sigmoid和tanh层进行处理，并生成门激活($i_t$)和候选值($\tilde{C}_t$)。然后将它们逐元素结合（$i_t * \tilde{C}_t$），形成用于细胞状态的更新信息。

连接输入门与细胞状态的重要一步，涉及将这两个层的输出进行组合。这通常通过逐元素乘法完成：$i_t * \tilde{C}_t$。这个乘积表示*经过过滤的候选值*，即经过输入门选择和缩放的新信息。

这个结果向量（$i_t * \tilde{C}_t$）被添加到（适当遗忘的）前一个细胞状态$C_{t-1}$，以形成新的细胞状态$C_t$。我们将在下一节讨论更新细胞状态时，详细考察这个加法过程。目前，重要的是输入门提供了选择性地将新信息整合到LSTM记忆中的机制。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter and Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍了长短期记忆（LSTM）架构，详细阐述了记忆单元及其门控机制的最初设计。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供深度学习的全面论述，涵盖循环神经网络和LSTM架构及其门控系统。
- [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) — Christopher Olah (2015)
  提供对LSTM网络的清晰、直观的解释，使包括输入门在内的每个门的功能都易于理解。
- [CS224n: Natural Language Processing with Deep Learning, Lecture Notes 6: Recurrent Neural Networks and LSTMs](https://nlp.stanford.edu/courses/cs224n/2019/lectures/lecture6_rnn.pdf) — Abigail See (2019)
  Publisher: Stanford University; Pages: 6-10
  大学课程讲义，涵盖循环神经网络和LSTM，解释其组成部分和运作方式。

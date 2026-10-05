# 介绍GRU：一种更简洁的门控架构

来源：[原文](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-6-gated-recurrent-units-gru/introducing-grus)

[返回章节目录](README.md) · [返回课程目录](../README.md)

简单的循环神经网络 (neural network)（RNN）在理念上简洁，但难以捕捉长序列中的依赖关系。梯度消失问题常会阻碍梯度在许多时间步中有效传播，使得网络难以学习远距离元素间的关系。

长短期记忆（LSTM）网络在第五章中进行了详细描述，通过引入专用记忆单元和多个门控机制（遗忘门、输入门和输出门）来精细控制信息流，从而提供了一个有效的解决方案。LSTM已被证实非常有效，但其内部结构相对复杂且参数 (parameter)数量较多。

在LSTM开发的同时，出现了另一种门控循环单元 (GRU)：门控循环单元，简称GRU。GRU由Cho等人于2014年提出，旨在处理长距离依赖关系方面达到类似的能力，但采用更精简的架构。GRU背后的主要思想是简化门控机制，同时保持其缓解梯度问题的有效性。

### 简化门控结构

与LSTM相比，GRU引入了两项主要简化：

1. **单元状态与隐藏状态合并：** GRU不像LSTM那样维护独立的单元状态（$c_t$）。它们只有一个隐藏状态（$h_t$）。这个隐藏状态兼具双重作用：携带过往信息，并作为当前时间步的输出。
2. **更少的门：** GRU只使用两个门，而不是LSTM中使用的三个（或四个，取决于看待方式）。它们是**重置门**和**更新门**。

我们简单看一下这两个门的作用：

- **重置门（$r_t$）：** 这个门决定了在计算新的候选隐藏状态时，来自*上一个*隐藏状态（$h_{t-1}$）的多少信息应该被忽略或“重置”。如果重置门在某些维度上的值接近0，它有效地使单元在这些维度上表现得像是在处理一个新序列的开始，从而使其能够遗忘不相关的过往信息。
- **更新门（$z_t$）：** 这个门的作用类似于LSTM中的遗忘门和输入门，但以组合方式。它决定了来自上一个隐藏状态（$h_{t-1}$）的多少信息应该保留并直接传递到当前隐藏状态（$h_t$）。反之，它也决定了应该整合多少*新计算出的候选*隐藏状态。

下图对LSTM和GRU单元的内部结构进行了高层次的比较。

> 高层次地比较了LSTM和GRU单元的内部组成部分和信息流。注意GRU中没有独立的单元状态，且门更少。

### 简化的潜在优势

这种降低的复杂度提供了几个潜在优势：

- **更少的参数 (parameter)：** 只有两个门且没有独立的单元状态，GRU通常比相同数量隐藏单元的LSTM具有更少的训练参数。这可以使GRU的训练计算成本更低，并潜在地更不容易过拟合 (overfitting)，尤其是在数据量不是非常大的数据集上。
- **更快的训练：** 更少的参数通常意味着每个训练步骤的计算速度更快。

然而，LSTM和GRU之间的性能差异通常取决于任务。两种架构在所有序列建模问题中都不是普遍优越的。尽管GRU提供了简洁性，但LSTM凭借其独特的单元状态和独立的门，可能对信息流提供更精细的控制，这对某些复杂任务可能有利。

在接下来的部分，我们将详细检查GRU架构，包括控制其门和状态更新的具体公式。然后，我们将直接比较其机制和性能特点与LSTM的异同，以帮助您决定何时选择哪种更适合您的应用。

## 参考资料

- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio (2014)
  Journal: EMNLP 2014; Pages: 1724-1734; DOI: [10.48550/arXiv.1406.1078](https://doi.org/10.48550/arXiv.1406.1078)
  提出门控循环单元（GRU）架构的原始研究论文。
- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  提出长短期记忆（LSTM）网络的开创性论文，GRU是对其进行简化的架构。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 10: Sequence Modeling: Recurrent and Recursive Networks
  一本内容详尽的教科书，其中包含对循环神经网络、LSTM和GRU的细致说明。

---

[上一节](../05-%E9%95%BF%E7%9F%AD%E6%9C%9F%E8%AE%B0%E5%BF%86%20%28LSTM%29%20%E7%BD%91%E7%BB%9C/08-LSTM%20%E7%9A%84%E4%BC%98%E5%8A%BF.md) · [下一节](02-GRU%20%E5%8D%95%E5%85%83%E7%BB%93%E6%9E%84.md)

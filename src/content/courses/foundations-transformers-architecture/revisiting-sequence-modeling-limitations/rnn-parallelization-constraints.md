---
course: "foundations-transformers-architecture"
chapter: "revisiting-sequence-modeling-limitations"
lesson: "rnn-parallelization-constraints"
sourceId: 1928
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-1-revisiting-sequence-modeling-limitations/rnn-parallelization-constraints"
title: "循环模型中的并行化限制"
description: "了解为何RNN的按序特性阻碍了训练并行化。"
order: 6
plots: []
sourceHash: "848ffd34609a853bb77681198037cd15ab647ac68bcb5afb0b0a4c8f0c5cc084"
sourceCorrections: []
---

循环模型（如RNN、LSTM和GRU）逐步处理序列。这种按序处理的特性，尽管对时间序列或语言建模来说很直观，但却对计算并行化带来了基本限制，尤其是在训练阶段。

### 序列依赖性瓶颈

任何RNN中的主要操作是计算时间步 $t$ 的隐藏状态 $h_t$，其依据是该时间步的输入 $x_t$ 和*前一个*时间步的隐藏状态 $h_{t-1}$。这种关系通常表示为：


$$
h_t = f(h_{t-1}, x_t)
$$


其中 $f$ 代表RNN单元执行的变换（可以是简单的RNN更新、LSTM单元或GRU单元）。这个公式清楚地显示了固有的序列依赖性：计算 $h_t$ 需要 $h_{t-1}$ 的结果，而 $h_{t-1}$ 又需要 $h_{t-2}$，依此类推，直到最初的状态 $h_0$。

> 隐藏状态 $h_t$ 的计算按序依赖于前一个状态 $h_{t-1}$。这种依赖关系阻止了在单个序列中跨时间步的并行计算。

这种时间上的依赖性造成了瓶颈。现代硬件加速器，例如GPU和TPU，擅长并行执行大规模矩阵运算。然而，在RNN处理的单个序列中，时间步 $t$ 的计算必须等到时间步 $t-1$ 的计算完成后才能开始。单个时间步*内部*的操作（例如LSTM门内的矩阵乘法）可以并行化，但*跨*时间步的计算仍然是按序进行的。

### 对训练效率的影响

这种限制明显制约了训练期间可获得的加速效果。尽管可以在小批量数据（数据并行）中对*不同序列*进行RNN训练的并行化，但每个*单独序列*的处理仍受其长度限制，这是因为前向传播是按序进行的。

此外，网络中的反向传播 (backpropagation)（称为时间反向传播，BPTT）也受到同样的按序处理限制。为了计算损失函数 (loss function)对时间步 $t$ 所用参数 (parameter)的梯度，BPTT需要逐步地将梯度反向传播通过序列。时间步 $t$ 的梯度计算依赖于时间步 $t+1$ 的梯度计算。这反映了前向传播的依赖关系，阻止了梯度计算在时间上的并行化。

实际结果是，在非常长的序列上训练RNN变得计算成本高且缓慢。增加序列长度会导致该序列的计算时间大致呈线性增长，并且无法充分发挥并行硬件的性能意味着GPU在部分计算过程中可能未被充分利用。

这种限制是促使Transformer等架构出现的原因之一，其目的是捕获序列依赖性而不依赖于按序的循环，从而在训练期间实现更大的并行化。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 10: Sequence Modeling: Recurrent and Recursive Nets
  全面介绍了循环神经网络、其计算图和通过时间反向传播算法，详细阐述了限制并行化的固有序列依赖性。
- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter and Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: The MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍了长短期记忆（LSTM）网络的基础架构，展示了广泛使用的循环单元设计，该设计具有本文讨论的序列依赖性。
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://aclanthology.org/D14-1179/) — Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 1724-1734; DOI: [10.3115/v1/D14-1179](https://doi.org/10.3115/v1/D14-1179)
  介绍了门控循环单元（GRU），这是另一个重要的循环模型，进一步展示了受序列计算限制的架构。

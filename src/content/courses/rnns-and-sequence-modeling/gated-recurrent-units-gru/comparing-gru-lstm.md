---
course: "rnns-and-sequence-modeling"
chapter: "gated-recurrent-units-gru"
lesson: "comparing-gru-lstm"
sourceId: 2592
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-6-gated-recurrent-units-gru/comparing-gru-lstm"
title: "GRU与LSTM的比较"
description: "讨论GRU与LSTM架构之间的异同及权衡。"
order: 7
plots: []
sourceHash: "b1d6a801328fce36c2d1fbe6983a53f5635c247247745a985f5befc79b0e90e5"
sourceCorrections: []
---

长短期记忆网络（LSTM）和门控循环单元（GRU）是循环神经网络 (neural network) (RNN)中两种主要的架构。一个主要的问题是，这两种强大的架构如何相互比较，以及何时应该选择其中一个。两者都是为了解决简单RNN的不足而开发的，特别是通过引入门控机制来应对梯度消失问题。然而，它们通过不同的设计达成此目标，这导致了结构、计算成本以及有时性能上的差异。

### 架构差异：门和状态管理

最明显的区别在于它们的内部结构，具体体现在门的数量以及它们管理内存的方式。

- **LSTM：** 使用三个不同的门：

  1. **遗忘门 ($f_t$)：** 决定从单元状态中舍弃哪些信息。
  2. **输入门 ($i_t$)：** 确定将哪些新信息存入单元状态。
  3. **输出门 ($o_t$)：** 控制单元状态的哪一部分作为隐藏状态输出。
     重要的是，LSTM在**隐藏状态 ($h_t$)** 旁边维护一个独立的**单元状态 ($c_t$)**。单元状态就像一个内部记忆传送带，通过遗忘门和输入门相对温和地修改。隐藏状态是单元状态的过滤版本，由输出门决定。这种分离使得对记忆的持久性和暴露拥有更专门的控制。
- **GRU：** 仅使用两个门：

  1. **重置门 ($r_t$)：** 决定在提出新的候选状态时，遗忘多少先前的隐藏状态。
  2. **更新门 ($z_t$)：** 决定保留多少先前的隐藏状态 ($h_{t-1}$) 以及将多少新的候选隐藏状态 ($\tilde{h}_t$) 合并到最终隐藏状态 ($h_t$) 中。
     GRU *没有* 独立的单元状态。它将LSTM的单元状态和隐藏状态的功能合并为一个单一的隐藏状态 ($h_t$)。更新门直接平衡了对过去信息 ($h_{t-1}$) 的保留和对新信息 ($\tilde{h}_t$) 的整合。

以下是突出结构差异的简化视图：

> LSTM和GRU单元中信息流与组成部分的宏观比较。LSTM利用由三个门管理的独立单元和隐藏状态，而GRU则使用由两个门管理的单一隐藏状态。

### 计算效率与参数 (parameter)数量

因为GRU的门较少且没有独立的单元状态，所以它们通常比LSTM的计算效率更高。

- **更少的参数：** 对于相同数量的隐藏单元，GRU层的可训练权重 (weight)和偏置 (bias)将少于LSTM层。这种差异源于输出门的缺失以及更新门的合并作用。
- **更快的计算：** 门控计算更少意味着GRU单元中的每个时间步计算通常比LSTM单元更快。
- **更低的内存需求：** 更少的参数也意味着在训练和推理 (inference)过程中对内存的需求会略低。

这种效率提升会很明显，尤其是在构建深层网络（堆叠RNN）或处理非常大的数据集或长序列，并且训练时间是重要因素时。

### 性能考量

GRU更简单的架构是否会导致更差的性能？不一定。比较LSTM和GRU的经验结果通常是混合的，并且高度依赖于具体的任务和数据集。

- **普遍的等效性：** 在许多序列建模任务上，GRU的表现已显示与LSTM相当。其简化的门控可能足以捕捉所需的时间依赖性。
- **潜在的LSTM优势场景：** 一些研究表明，在需要建模极长依赖或对记忆内容进行非常精确控制的任务上，LSTM*可能*具有轻微优势，这可能是由于其专用的单元状态充当了更受保护的记忆通道。
- **数据量：** 对于较小的数据集，参数 (parameter)更少的GRU可能比LSTM具有略好的泛化能力，或者更不容易过拟合 (overfitting)。
- **收敛性：** GRU有时在训练期间收敛更快，这归因于其更简单的结构以及与更新门在 $h_{t-1}$ 和 $h_t$ 之间直接连接相关的可能更平滑的梯度流。

没有明确的规则说明哪一个普遍优越。选择通常归结为针对您的具体问题进行经验评估。

### 何时选择？实用建议

考虑到它们的异同，以下是选择GRU和LSTM的实用方法：

1. **从GRU开始：** 如果计算资源有限，训练时间是一个考量因素，或者您正在处理相对较小的数据集，GRU通常是一个不错的默认选择，因为它在许多任务上具有效率和可比的性能。
2. **如果需要，尝试LSTM：** 如果您拥有充足的计算资源，并希望尽可能地提升性能，尤其是在可能涉及非常复杂或长距离依赖的任务上（如机器翻译或长文档摘要），尝试LSTM是值得的。增加的复杂度*可能*会带来性能优势。
3. **经验性测试：** 最可靠的方法通常是尝试两种架构（最初可以在较小规模上进行），并在您的特定验证集上评估它们的性能。对两种架构进行超参数 (parameter) (hyperparameter)调整对于公平比较很重要。

总而言之，LSTM和GRU都代表了对简单RNN的重大改进。GRU提供了一种精简的设计，参数更少，计算可能更快，性能通常与LSTM相当。LSTM则提供了更复杂的门控机制和独立的单元状态，这在需要精细记忆控制的特定情况下可能带来优势。最佳选择通常取决于您项目的具体约束和要求。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍长短期记忆 (LSTM) 架构的原始论文，是理解其门控机制的基础。
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Pages: 1724-1734; DOI: [10.48550/arXiv.1406.1078](https://doi.org/10.48550/arXiv.1406.1078)
  在机器翻译的编码器-解码器框架中引入门控循环单元 (GRU) 的论文。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 10
  提供循环神经网络、LSTM 和 GRU 全面解释的标准教科书章节。

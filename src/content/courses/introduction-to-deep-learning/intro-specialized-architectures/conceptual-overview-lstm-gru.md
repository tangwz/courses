---
course: "introduction-to-deep-learning"
chapter: "intro-specialized-architectures"
lesson: "conceptual-overview-lstm-gru"
sourceId: 5080
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-7-intro-specialized-architectures/conceptual-overview-lstm-gru"
title: "概述：LSTM与GRU"
description: "介绍长短期记忆（LSTM）和门控循环单元（GRU）作为解决RNN挑战的方法。"
order: 10
plots: []
sourceHash: "a6a3cf208eedd6fcbd9b7aa9678596b40326cd6dbea33d0a05a730f3656765ec"
sourceCorrections: []
---

简单的循环神经网络 (neural network)（RNN）虽然其设计思路简洁，但在学习长序列模式时面临困难。其主要问题常在于时间上的反向传播 (backpropagation)过程。梯度可能呈指数级缩小（梯度消失），这使得网络难以学习远距离元素间的关联；或呈指数级增大（梯度爆炸），导致训练不稳定。这一局限性显著影响了它们在需要长时记忆任务上的表现。

为应对这些难题，更复杂的循环架构被发展出来，其中最著名的是长短期记忆（LSTM）网络和门控循环单元（GRU）。这些架构引入了被称为“门”的机制，用于调节循环单元内信息的流通，使它们能够在长时间内选择性地记忆或遗忘信息。

### 长短期记忆 (LSTM)

LSTM通过引入一个专门的*细胞状态*与隐藏状态并行，来处理梯度问题。可将细胞状态（$c_t$）想象成一条信息高速公路，它允许信息在序列中相对不变地流动，除非被明确修改。细胞状态的修改由三个主要的门来控制：

1. **遗忘门：** 它决定从细胞状态中丢弃哪些信息。它查看前一隐藏状态（$h_{t-1}$）和当前输入（$x_t$），并对前一细胞状态（$c_{t-1}$）中的每条信息输出一个0到1之间的数值。1表示“完全保留”，而0表示“完全丢弃”。
2. **输入门：** 它决定将哪些新信息存储到细胞状态中。此门有两个部分：首先，一个 sigmoid 层（$\sigma$）决定更新哪些值（即“输入门”本身）；其次，一个 tanh 层创建一个新的候选值向量 (vector)（$\tilde{c}_t$），这些值可以被添加到状态中。
3. **输出门：** 它决定将什么作为隐藏状态（$h_t$）输出。它首先运行一个 sigmoid 层来决定细胞状态的哪些部分应该被输出。然后，它将（已更新的）细胞状态通过 `tanh` 函数（将值推到-1和1之间）并将其乘以 sigmoid 门的输出，以便只输出选定的部分。

这些门使用如 sigmoid（$\sigma$）之类的激活函数 (activation function)（将值压缩到0和1之间）来控制信息流通。通过学习这些门的参数 (parameter)，LSTM能够学习复杂的依赖关系并在多个时间步中保留重要信息，从而缓解了梯度消失问题。

> LSTM单元内的信息流，突显遗忘门、输入门和输出门在管理细胞状态和隐藏状态方面的作用。

### 门控循环单元 (GRU)

门控循环单元（GRU）是一种较新的循环架构，其引入是为了简化LSTM。它将遗忘门和输入门合并为一个*更新门*，并将细胞状态和隐藏状态合并。它还引入了一个*重置门*。

1. **重置门：** 它决定在提出新的候选隐藏状态时，前一隐藏状态（$h_{t-1}$）应被遗忘多少。如果重置门的输出接近0，则前一隐藏状态将被很大程度上忽略。
2. **更新门：** 类似于LSTM的遗忘门和输入门的组合。它决定保留前一隐藏状态（$h_{t-1}$）的多少，以及将（使用重置门的影响计算出的）*新候选*隐藏状态的多少纳入最终隐藏状态（$h_t$）。

GRU比LSTM有更少的参数 (parameter)（因为它们缺少单独的输出门和细胞状态），有时在计算上更高效。从实践来看，它们在许多任务上的表现常与LSTM相近，虽然没有绝对的优胜者；最佳选择通常取决于具体的数据集和问题。

> GRU单元内的信息流，显示重置门和更新门如何控制组合到新隐藏状态中的信息。

LSTM和GRU都通过引入门控机制，相对于简单的RNN取得了显著进步。这些门使网络能够学习在长序列中哪些信息应保留或丢弃，使它们成为自然语言处理、时间序列分析等领域中建模序列数据的有力工具。虽然我们不会在本入门课程中完整实现它们，但理解它们的作用对于判断何时标准前馈网络或简单循环网络可能不足是重要的。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: The MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍长短期记忆（LSTM）架构的原始学术论文。
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://doi.org/10.48550/arXiv.1406.1078) — Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio (2014)
  Journal: EMNLP 2014; Pages: 1724-1734; DOI: [10.48550/arXiv.1406.1078](https://doi.org/10.48550/arXiv.1406.1078)
  本文介绍了门控循环单元（GRU）作为一种新型循环神经网络架构。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本内容全面的教科书，涵盖了循环神经网络，包括对LSTM和GRU的详细解释。

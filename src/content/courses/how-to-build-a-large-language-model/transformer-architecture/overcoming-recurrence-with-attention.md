---
course: "how-to-build-a-large-language-model"
chapter: "transformer-architecture"
lesson: "overcoming-recurrence-with-attention"
sourceId: 5962
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-4-transformer-architecture/overcoming-recurrence-with-attention"
title: "通过注意力机制解决循环问题"
description: "说明注意力机制作为顺序处理的替代方案的缘由。"
order: 1
plots: []
sourceHash: "6a447e588bb43bc61c33bbd0b2451d0272b84f087e79ccff5facfa856f1c9c62"
sourceCorrections: []
---

循环神经网络 (neural network)（RNN），包括长短期记忆网络（LSTM）和门控循环单元（GRU），都逐步处理序列。每个隐藏状态 $h_t$ 都是根据输入 $x_t$ 和前一个隐藏状态 $h_{t-1}$ 计算得出。这种顺序处理方式，虽然在建模序列时很直观，但在处理大型语言模型所需的规模时，会带来显著的难题。

### 循环的瓶颈

核心局限直接源于这种顺序处理：

1. **并行化受限：** $h_t$ 的计算*必须*等待 $h_{t-1}$ 计算完成后才能进行。这种依赖性阻止了在单个训练示例中沿着时间维度（序列长度）进行计算的并行化。尽管可以在批次中的不同序列之间进行并行化，但每个序列*内部*的处理本质上仍是顺序的。对于很长的序列，这会成为主要的计算瓶颈，从而限制训练速度。想象一下处理一个有数千个词的文档；计算必须逐词进行。
2. **难以处理长距离依赖：** 序列早期部分的信息必须通过整个循环连接链才能影响后续部分的计算。尽管LSTM和GRU设计了门控机制以缓解梯度消失问题，但在很长的距离上保持准确的信息仍然很困难。两个相距较远的词元 (token) $x_i$ 和 $x_j$ 之间的信息流动路径长度与 $|i - j|$ 成正比。这意味着梯度在长路径上仍然可能消失或爆炸，使得模型难以学习序列中相距较远的词之间的关系。

考虑一个简化的RNN更新过程：

```python
import torch

# 占位输入序列（batch_size=1, seq_len=5, features=10）
input_seq = torch.randn(1, 5, 10)
# 初始隐藏状态（batch_size=1, hidden_size=20）
h_prev = torch.zeros(1, 20)
# 简单的RNN单元（非实际实现）
rnn_cell = lambda input_t, h_prev: torch.tanh(
    input_t @ torch.randn(10, 20) + h_prev @ torch.randn(20, 20)
) # 简化版

hidden_states = []
# 顺序处理循环
for t in range(input_seq.shape[1]): # 遍历序列长度
    input_t = input_seq[:, t, :]
    h_t = rnn_cell(input_t, h_prev)
    hidden_states.append(h_t)
    h_prev = h_t # 更新隐藏状态以用于下一步

# hidden_states 现在包含每个时间步的状态
# 注意：时间步't'的计算明确依赖于
# 't-1'的结果
```

这个循环突显了顺序依赖性。我们无法在计算 `t=2` 对应的 `h_t` 之前计算 `t=3` 对应的 `h_t`。

### 引入注意力机制 (attention mechanism)：直接连接

Transformer架构通过完全去除循环，从根本上打破了这种顺序链条。它不再逐步传递信息，而是使用**注意力机制**。注意力机制的核心思想是让模型在处理序列中的一个元素（例如一个词）时，能够直接查看并从序列中的*所有*其他元素中获取信息。

想象您正在翻译句子“The cat sat on the mat”。当处理词语“sat”时，注意力机制使模型能够直接评估“The”、“cat”、“on”、“the”和“mat”的相关性，以理解“sat”的语境。它计算一组注意力分数，表示序列中其他每个词对当前词的重要性。然后，这些分数用于创建其他词表示的加权和，为“sat”提供一个语境感知的表示。

重要的是，这种“查看”过程不依赖于序列中词之间的距离。模型可以在“The”和“mat”之间建立直接连接，就像在“on”和“the”之间一样容易。序列中任意两个词元 (token)之间的路径长度变为常数，实际上是 $O(1)$，因为注意力机制直接计算它们之间的两两交互。这极大地简化了长距离依赖的学习，与RNN中序列长度为 $n$ 时 $O(n)$ 的路径长度相比。

### 实现并行化

通过消除顺序依赖性（$h_t$ 依赖于 $h_{t-1}$），Transformer实现了跨序列长度的大规模并行化。理论上，生成每个词元 (token)表示所需的计算可以同时进行。尽管单个词元的计算*内部*存在依赖性（例如，在应用注意力分数之前计算它们），但层内整个序列表示的总体计算可以比在RNN中更有效地并行化。这一特性对训练当前流行的大型模型非常重要。

从循环到注意力机制 (attention mechanism)的转变可以被形象化为从链式结构转向全连接图（在同一层内），其中每个词元都可以直接与其他所有词元交互。

> 信息流比较。在RNN中（左图），信息顺序流动。在像Transformer这样的基于注意力机制的模型中（右图），每个输出可以同时直接关注所有输入。

这种从顺序处理到可并行注意力机制的根本性转变是Transformer架构在大型序列建模中如此成功的主要原因。以下章节将查看实现这一原理的特定机制，例如缩放点积注意力（Scaled Dot-Product Attention）和多头注意力（Multi-Head Attention）。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  本文介绍了Transformer架构，该架构完全依赖于注意力机制并移除了循环结构，从而克服了RNN在序列建模方面的局限性。
- [Long Short-Term Memory](https://www.mitpressjournals.org/doi/pdf/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter and Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: The MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  这篇基础论文介绍了长短期记忆（LSTM）网络，这是一种循环神经网络，旨在解决梯度消失问题并改进对长序列依赖关系的处理。
- [Stanford CS224N: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) — Diyi Yang, Tatsunori Hashimoto (2025)
  Publisher: Stanford University
  这门大学课程提供了自然语言处理深度学习的全面材料，涵盖了RNN、LSTM、GRU、注意力机制和Transformer架构，解释了它们的架构演变和应用。
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Pages: 1724-1734; DOI: [10.48550/arXiv.1406.1078](https://doi.org/10.48550/arXiv.1406.1078)
  本文介绍了门控循环单元（GRU），它是LSTM的一个简化变体，能有效建模序列，并有助于解决循环网络中的长距离依赖问题。

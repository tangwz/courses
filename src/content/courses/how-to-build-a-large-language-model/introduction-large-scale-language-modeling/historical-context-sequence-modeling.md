---
course: "how-to-build-a-large-language-model"
chapter: "introduction-large-scale-language-modeling"
lesson: "historical-context-sequence-modeling"
sourceId: 5948
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-1-introduction-large-scale-language-modeling/historical-context-sequence-modeling"
title: "序列建模的历史背景"
description: "回顾从N-gram和RNN到大型Transformer模型兴起的历程。"
order: 2
plots: []
sourceHash: "79bfe23ff7452682c9d2296df0693ad47540e23c32ee924772544e5b4b6ca999"
sourceCorrections: []
---

要理解当今大型语言模型的结构和功能，需要了解为处理序列数据（尤其是文本）而发展起来的技术。从简单的统计方法到复杂的神经网络 (neural network)的演进，表明了捕获语言中语境和依赖关系的持续难题。

### 早期统计模型：N-gram

在深度学习 (deep learning)普及之前，统计方法构成了语言建模的起点。其中最基本的是**N-gram模型**。这些模型基于马尔可夫假设运行：下一个词的概率仅依赖于前 $n-1$ 个词。例如，一个三元模型（$n=3$）会估算给定前两个词 $w_{i-1}$ 和 $w_{i-2}$ 时词 $w_i$ 的概率：

$P(w_i | w_1, ..., w_{i-1}) \approx P(w_i | w_{i-2}, w_{i-1})$

这些概率通常通过计算大型文本语料库中的出现次数来估算：

$P(w_i | w_{i-2}, w_{i-1}) = \frac{\text{计数}(w_{i-2}, w_{i-1}, w_i)}{\text{计数}(w_{i-2}, w_{i-1})}$

尽管简单且易于理解，N-gram模型面临明显不足：

1. **稀疏性：** 许多有效的词序列可能从未在训练语料库中出现，导致概率为零。人们开发了平滑技术（例如拉普拉斯平滑、Kneser-Ney平滑）来缓解此问题，但它们不能完全解决问题，特别是当 $n$ 较大时。
2. **语境窗口：** N-gram只能考虑一个固定的短语境（$n-1$个词）。它们从根本上无法捕获长距离依赖关系，这在人类语言中大量存在（例如，跨子句的主谓一致）。增加 $n$ 会指数级地加剧稀疏性问题。
3. **缺乏泛化能力：** N-gram将词视为离散单元。它们不能轻易地基于语义相似性进行泛化（例如，知道“cat”与“dog”相似）。

> N-gram模型根据固定窗口内的前置词来预测下一个词。

### 循环神经网络 (neural network)（RNN）的出现

神经网络提供了一种方法来处理N-gram的不足之处。\*\*循环神经网络（RNN）\*\*是专门为序列数据设计的。与前馈网络不同，RNN具有循环连接，使其能够维持一个内部*隐藏状态*（$h_t$），该状态理论上可以包含序列中所有先前时间步的信息。

在每个时间步 $t$，RNN接收当前输入 $x_t$ 和前一个隐藏状态 $h_{t-1}$，以计算新的隐藏状态 $h_t$ 和可能的输出 $y_t$：

$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$
$y_t = W_{hy}h_t + b_y$

这里，$W_{hh}$、$W_{xh}$、$W_{hy}$、$b_h$ 和 $b_y$ 是学习得到的参数 (parameter)（权重 (weight)矩阵和偏置 (bias)），它们在所有时间步共享。$\tanh$ 函数是一种常用的激活函数 (activation function)。

> RNN单元处理输入 $x_t$ 和前一状态 $h_{t-1}$，以生成下一个状态 $h_t$ 和输出 $y_t$。状态随着时间传递。

尽管RNN有望建模任意长距离依赖关系，但由于**梯度消失问题**，训练它们变得困难。在时间反向传播 (backpropagation)过程中，梯度在多个时间步相乘可能指数级缩小，阻止与早期时间步相关的权重得到充分更新。这意味着，在实践中，简单RNN难以学习相对短窗口内的依赖关系，效果与N-gram类似。相关的**梯度爆炸问题**（即梯度指数级增长）也可能发生，尽管通常通过梯度裁剪等方式更容易控制。

### 门控RNN：LSTM和GRU

为了解决梯度消失问题，人们设计了更复杂的循环单元。

\*\*长短期记忆（LSTM）\*\*网络引入了门控机制：

1. **遗忘门：** 决定从细胞状态中丢弃哪些信息。
2. **输入门：** 决定哪些新信息要存储到细胞状态中。
3. **输出门：** 决定根据过滤后的细胞状态输出什么。

这些门通过一个独立的*细胞状态*（$C_t$）控制信息随时间流动，使网络能够选择性地长时间记住或遗忘信息。

```python
import torch
import torch.nn as nn

# 示例：在PyTorch中实例化一个LSTM层
# 每个时间步的输入特征数
input_size = 10
# 隐藏状态中的特征数
hidden_size = 20
# 循环层数
num_layers = 2

# 创建一个LSTM层
lstm_layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

# 示例输入 (批量大小=3, 序列长度=5, 输入特征数=10)
batch_size = 3
seq_len = 5
example_input = torch.randn(batch_size, seq_len, input_size)

# 初始隐藏状态和细胞状态（可选，默认为零）
h0 = torch.randn(num_layers, batch_size, hidden_size)
c0 = torch.randn(num_layers, batch_size, hidden_size)

# 前向传播
output, (hn, cn) = lstm_layer(example_input, (h0, c0))

print("Output shape:", output.shape) # (批量大小, 序列长度, 隐藏状态大小)
print("Final hidden state shape:", hn.shape)
# (层数, 批量大小, 隐藏状态大小)
print("Final cell state shape:", cn.shape)
# (层数, 批量大小, 隐藏状态大小)
```

\*\*门控循环单元（GRU）\*\*提供了一种稍微简化的替代方案，它有两个门（更新门和重置门），并且没有独立的细胞状态。GRU在许多任务上通常能达到与LSTM相近的性能，但参数 (parameter)更少。

与简单RNN相比，LSTM和GRU都明显提升了捕获长距离依赖关系的能力。它们成为许多自然语言处理任务的标准，常用于\*\*序列到序列（Seq2Seq）\*\*架构中。Seq2Seq模型由一个*编码器*RNN和一个*解码器*RNN组成，编码器RNN将输入序列处理成一个上下文 (context)向量 (vector)（通常是最终隐藏状态），解码器RNN则根据该上下文向量生成输出序列。尽管在机器翻译、摘要等任务中取得了成功，Seq2Seq模型仍然面临瓶颈：无论输入序列的长度如何，其全部含义都必须被压缩到一个固定大小的上下文向量中。

### 注意力机制 (attention mechanism)

**注意力机制**被引入以缓解Seq2Seq的瓶颈问题。解码器不再仅仅依赖最终的编码器隐藏状态，而是在生成输出的每一步中被允许“关注”*整个*输入序列的不同部分。它计算当前解码器状态与所有编码器隐藏状态之间的注意力得分，生成一个针对该解码步骤的加权上下文 (context)向量 (vector)。这使得模型在生成相应输出词时，能够专注于相关的输入词，明显提升了性能，特别是对于更长的序列。

> 注意力机制允许解码器状态 $s_t$ 选择性地对编码器隐藏状态（$h_1, h_2, h_3$）进行加权，以形成上下文向量 $c_t$。

### Transformer：摆脱循环

尽管注意力机制 (attention mechanism)提升了基于RNN的模型，但循环的序列性仍然是训练效率的瓶颈。处理一个序列需要 $O(\text{序列长度})$ 的顺序操作，阻碍了并行化。

在论文“Attention Is All You Need”（Vaswani 等，2017）中提出的**Transformer**架构，通过完全放弃循环机制，彻底改变了序列建模方式。它仅依赖于注意力机制，特别是**自注意力 (self-attention)**。自注意力允许模型在为特定词编码表示时，衡量输入序列中所有其他词的重要性。

Transformer层内的计算（自注意力和前馈网络）可以在序列位置上大部分并行执行。这种并行能力是一项重大进展，使得在比以往使用RNN更大的数据集上训练更大规模的模型成为可能。通过并行化获得的计算效率，加上自注意力在处理复杂依赖关系（包括短距离和长距离）方面的有效性，为构建本课程主要关注的超大型语言模型创造了条件。Transformer架构所带来的扩展特性和计算需求是我们将在后续讨论中多次提及的核心内容。

## 参考资料

- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Pearson Education
  一本全面的教科书，涵盖自然语言处理的基础概念，包括N-gram模型及其统计估计。
- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍长短期记忆（LSTM）网络的原始论文，解决了RNN中的梯度消失问题。
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio (2014)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1409.0473](https://doi.org/10.48550/arXiv.1409.0473)
  这篇论文介绍了注意力机制，使模型能够在生成输出序列时关注相关的输入部分。
- [Attention Is All You Need](https://doi.org/10.48550/arXiv.1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  这篇开创性论文介绍了Transformer架构，该架构完全依赖于注意力机制，并显著提高了序列建模的并行化能力。

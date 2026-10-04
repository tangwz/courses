---
course: "introduction-to-transformer-models"
chapter: "transformer-encoder-decoder-architecture"
lesson: "final-linear-softmax"
sourceId: 3015
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-3-transformer-encoder-decoder-architecture/final-linear-softmax"
title: "最终线性层和Softmax"
description: "解释解码器输出如何转换为词汇表上的最终概率分布。"
order: 11
plots: []
sourceHash: "ed94fbbcd1855d2a9d8773137ff508c14990d776f3524cc2dfeea911c7e40937"
sourceCorrections: []
---

Transformer 解码器生成一系列输出向量 (vector)。这一生成过程涉及多个层，这些层包含掩码自注意力 (self-attention)机制 (attention mechanism)和编码器-解码器注意力机制，用于提炼序列信息。输出序列中位置 $t$ 处的每个向量，代表了解码器对于在该输入序列和生成到位置 $t-1$ 的词元 (token)的条件下，该位置应是哪个词元的判断。

然而，这些输出向量通常是高维的，且包含连续值。它们不能直接告诉我们词汇表 (vocabulary)中哪个具体的词最有可能出现在该位置。为了将这些内部表示转换为实际预测的词，我们需要两个最终步骤：一个线性变换和一个Softmax激活函数 (activation function)。

### 最终线性层

第一步是将解码器堆栈产生的高维输出向量 (vector)投影到一个大小与我们目标词汇表 (vocabulary)匹配的向量。例如，如果我们的模型需要从一个包含50,000个独立词元 (token)的词汇表中预测词，这个线性层会将解码器的输出向量（例如，维度为 $d_{model} = 512$）转换为一个维度为50,000的向量。

这通过一个标准的全连接线性层实现，该层有时也称为投影层。在数学上，如果 $x_{out} \in \mathbb{R}^{d_{model}}$ 是顶层解码器层针对特定位置的输出向量，且词汇表大小为 $V$，线性层执行以下操作：


$$
\text{对数几率} = W_{proj} x_{out} + b_{proj}
$$


在这里，$W_{proj}$ 是一个形状为 $(V, d_{model})$ 的权重 (weight)矩阵，$b_{proj}$ 是一个形状为 $(V, 1)$ 的偏置 (bias)向量。$W_{proj}$ 和 $b_{proj}$ 都是可学习的参数 (parameter)，在训练过程中进行调整。结果向量，通常称为`logits`，维度为 $V$。`logits`向量中的每个元素对应词汇表中一个特定词元的分数。分数越高表示对应的词元可能性越大，根据模型在此阶段的判断。

### Softmax函数

`logits`向量 (vector)提供了分数，但它们不是概率。这些分数可以是正的或负的，并且它们的总和不为1。为了将这些分数转换为词汇表 (vocabulary)上正确的概率分布，我们应用Softmax函数。

Softmax函数将`logits`向量作为输入，并输出一个相同维度 ($V$) 的概率向量。对于一个`logits`向量 $z = [z_1, z_2, ..., z_V]$，Softmax函数计算词汇表中第 $i$ 个词元 (token)的概率 $P_i$ 如下：


$$
P_i = \text{Softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
$$


这个函数有两个理想的属性：

1. **归一化 (normalization)：** 所有输出概率的总和为1 ($\sum_{i=1}^{V} P_i = 1$)。
2. **单调性：** 它保持了输入分数的顺序。具有最高logit分数的词元在应用Softmax后也将具有最高的概率。

结果向量 $P = [P_1, P_2, ..., P_V]$ 表示模型对整个词汇表的预测概率分布，针对输出序列中的当前位置。

> 从最终解码器输出向量经过线性层和Softmax函数，以生成目标词汇表的概率分布。

在推断（生成新文本）期间，模型通常从该分布中选择概率最高的词元作为当前位置的预测词元。这个预测的词元随后通常被送回解码器作为输入，用于生成序列中的下一个词元。在训练期间，这个概率分布与实际目标词元进行比较，使用损失函数 (loss function)（例如交叉熵）来计算误差并更新模型的参数 (parameter)。

线性层与Softmax函数的这种组合，是将Transformer解码器学习到的内部表示映射回可观测词元空间的标准方式，使模型能够做出具体的预测。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍Transformer模型的原始论文，详细阐述其架构，包括用于输出生成的最终线性投影和softmax。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教科书，涵盖了深度学习的基本组成部分，例如用于分类的线性变换和softmax激活函数。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2025)
  Publisher: Pearson
  一本标准的自然语言处理教科书，解释了神经网络模型，特别是序列模型，如何利用线性层和softmax从词汇表中预测标记。
- [Stanford CS224N: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) — Diyi Yang, Tatsunori Hashimoto (2023)
  Publisher: Stanford University
  提供深度学习在自然语言处理中的教育材料，通常涵盖线性层和softmax在序列模型输出层中的应用。

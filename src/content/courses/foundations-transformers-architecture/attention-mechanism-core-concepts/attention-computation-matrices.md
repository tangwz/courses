---
course: "foundations-transformers-architecture"
chapter: "attention-mechanism-core-concepts"
lesson: "attention-computation-matrices"
sourceId: 1945
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-2-attention-mechanism-core-concepts/attention-computation-matrices"
title: "计算考量与矩阵运算"
description: "使用矩阵乘法并行化实现高效注意力计算。"
order: 6
plots: []
sourceHash: "fa35f4b9b4ea941b93eccccc71408fa74f5a83b3f240a9d09472690e545cc27a"
sourceCorrections: []
---

通过矩阵运算实现的高效性赋予了注意力机制 (attention mechanism)实际效果，尽管其定义利用查询、键和值来进行概念理解。这种方法使模型能够同时计算序列中所有位置的注意力分数和上下文 (context)向量 (vector)，使其非常契合现代并行硬件如GPU和TPU。

我们来回顾一下缩放点积注意力公式：
$\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

与一次只计算一个查询的注意力不同，我们同时处理整个序列。假设我们有一个长度为 $n$ 的输入序列。每个位置的查询、键和值向量堆叠起来形成矩阵：

- $Q$: 一个维度为 $(n \times d_k)$ 的矩阵，其中每行 $i$ 是位置 $i$ 的查询向量 $q_i$。
- $K$: 一个维度为 $(n \times d_k)$ 的矩阵，其中每行 $j$ 是位置 $j$ 的键向量 $k_j$。
- $V$: 一个维度为 $(n \times d_v)$ 的矩阵，其中每行 $j$ 是位置 $j$ 的值向量 $v_j$。值得一提的是，值的维度 $d_v$ 有时可能与 $d_k$ 不同。

现在，我们使用这些矩阵逐步分析计算过程：

### 1. 计算点积分数：$QK^T$

注意力计算的主要部分是确定每个查询应关注每个元素的程度。这通过计算每个查询 $q_i$ 与每个元素 $k_j$ 之间的点积来完成。矩阵乘法 $QK^T$ 并行执行所有这些点积：

$\text{分数} = Q K^T$

结果得到的 `分数` 矩阵的维度为 $(n \times d_k) \times (d_k \times n) = (n \times n)$。此矩阵中的每个元素 $(i, j)$ 代表位置 $i$ 的查询与位置 $j$ 的键之间的原始对齐 (alignment)分数。值越高意味着相关性越强。

### 2. 应用缩放：$/\sqrt{d_k}$

如前所述，缩放可以避免点积变得过大，从而可能将softmax函数推向梯度很小的区域。这种缩放按元素应用于 `分数` 矩阵：

$\text{缩放分数} = \frac{\text{分数}}{\sqrt{d_k}}$

维度仍为 $(n \times n)$。

### 3. 使用Softmax函数归一化 (normalization)分数：$\text{softmax}(\cdot)$

为了将缩放分数转换为概率（注意力权重 (weight)），softmax函数独立地应用于 `缩放分数` 矩阵的每一行。对于给定行 $i$，softmax确保查询 $i$ 对所有键 $j=1...n$ 分配的注意力权重总和为1。

$W = \text{softmax}(\text{缩放分数})_{\text{按行}}$

结果得到的注意力权重矩阵 $W$ 的维度也为 $(n \times n)$。$W_{ij}$ 代表位置 $i$ 的查询对位置 $j$ 的键（及关联值）关注的比例。

### 4. 计算值的加权和：$W V$

最后，注意力权重 (weight) $W$ 用于计算值向量 (vector) $V$ 的加权和。这通过矩阵乘法完成：

$\text{输出} = W V$

输出矩阵的维度为 $(n \times n) \times (n \times d_v) = (n \times d_v)$。`输出` 矩阵的每行 $i$ 是位置 $i$ 对应的上下文 (context)向量。它是序列中所有值向量的组合，根据为查询 $i$ 计算的注意力分布进行加权。

### 并行化带来的效率提升

这种基于矩阵的表述的精妙之处在于其可并行性。每个步骤，主要涉及矩阵乘法，都可以在为这类操作设计的硬件上高效执行。与按顺序处理标记 (token)（$t=1, 2, ..., n$）的循环模型不同，注意力机制 (attention mechanism)能够基本并行地计算所有位置对 $(i, j)$ 之间的交互。这解决了限制RNN和LSTM的顺序瓶颈，从而实现更快的训练和对更长序列的处理（在适用情况下）。

> 流程图描绘了使用矩阵运算计算缩放点积注意力的过程。维度表示为（行 x 列），其中 n 是序列长度，dk 是键的维度，dv 是值的维度。

了解这种矩阵表述是基础。它不仅说明了注意力如何高效计算，而且作为在深度学习 (deep learning)框架中实现注意力层的基础，这些框架高度依赖优化的矩阵运算。下一节将通过动手实践来实现缩放点积注意力机制，从而加深您的理解。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30; Publisher: Curran Associates, Inc.; Pages: 5998-6008; DOI: [10.55988/neurips-2017-1011](https://doi.org/10.55988/neurips-2017-1011)
  这篇论文介绍了Transformer架构和缩放点积注意力机制，提供了其原始公式并讨论了并行化的好处。
- [Dive into Deep Learning](https://d2l.ai/chapter_attention-mechanisms/attention-scoring-functions.html) — Aston Zhang, Zack C. Lipton, Mu Li, Alex J. Smola, et al. (2023)
  Publisher: Cambridge University Press
  一本在线教材，提供了深度学习模型的解释和实际实现，其中包含关于注意力机制及其矩阵运算的章节。
- [Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/) — Lewis Tunstall, Leandro von Werra, Thomas Wolf (2022)
  Publisher: O'Reilly Media
  一本实用指南，扩展了Transformer的理论方面，展示了矩阵运算如何在现代自然语言处理应用中实现和利用。

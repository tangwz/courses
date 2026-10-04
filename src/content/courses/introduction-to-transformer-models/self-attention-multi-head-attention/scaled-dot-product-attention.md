---
course: "introduction-to-transformer-models"
chapter: "self-attention-multi-head-attention"
lesson: "scaled-dot-product-attention"
sourceId: 2975
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-2-self-attention-multi-head-attention/scaled-dot-product-attention"
title: "缩放点积注意力机制"
description: "解释缩放点积注意力的公式和运作方式，包括缩放因子。"
order: 3
plots: []
sourceHash: "6dbdbc6a4c3e03cdd6a4cbca3d2df2d832e9540035c3051afea09c4f8b1eb15e"
sourceCorrections: []
---

缩放点积注意力是自注意力 (self-attention)机制 (attention mechanism)中主要的计算方法。它计算出序列中每个元素应如何关注（包括其自身在内的）所有其他元素。输入嵌入 (embedding)被转换为三种不同的向量 (vector)：查询 ($Q$)、键 ($K$) 和值 ($V$)。

- **查询 ($Q$):** 代表当前词语或位置，它在寻求信息。“我应该关注什么？”
- **键 ($K$):** 代表所有提供信息的词语或位置，如同标签或标识符。“我代表着什么。”
- **值 ($V$):** 代表与每个键相关的实际内容或含义。“我持有这些信息。”

目的是使用特定位置的查询来衡量其与所有位置的键的匹配度，然后使用这些匹配度分数对值进行加权求和。这个加权和将成为查询位置的输出表示，并富含来自整个序列的上下文 (context)信息。

### 缩放点积注意力公式

其计算以简洁的公式表示，该公式最初在“Attention Is All You Need”论文中提出：


$$
\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


我们来逐步分解：

1. **计算相似度分数 ($QK^T$)**
   第一步是计算我们关注位置的查询向量 (vector)与序列中所有位置的键向量的点积。如果我们把所有查询表示为一个矩阵 $Q$，所有键表示为一个矩阵 $K$，我们可以通过矩阵乘法 $QK^T$ 同时计算所有这些分数。

   - 为何使用点积？点积衡量两个向量的相似度或对齐 (alignment)度。查询 $q_i$ 与键 $k_j$ 之间点积越大，表明位置 $j$ 对位置 $i$ 的相关性越高。
   - 维度：如果 $Q$ 的维度是 ($seq\_len, d_k$)，而 $K$ 的维度是 ($seq\_len, d_k$)，那么它们的转置 $K^T$ 的维度是 ($d_k, seq\_len$)。结果矩阵 $QK^T$ 的维度将是 ($seq\_len, seq\_len$)，该矩阵的元素 $(i, j)$ 代表从位置 $i$ 到位置 $j$ 的原始注意力分数。
2. **缩放分数 ($\frac{...}{\sqrt{d_k}}$)**
   下一步是通过除以向量维度的平方根 $\sqrt{d_k}$ 来缩小这些分数。

   - 为何缩放？点积可能变得非常大，特别是当维度 $d_k$ 很大时。softmax 函数的输入过大会导致梯度变得非常小，这会使训练不稳定或缓慢。除以 $\sqrt{d_k}$ 可以抵消这种影响，更好地控制 softmax 输入的方差。这有助于稳定训练过程。
   - $d_k$: 这是键（和查询）向量的维度。在许多 Transformer 的实现中，查询、键和值向量的维度可能相同，但缩放因子专门与点积计算中使用的维度 ($d_k$) 相关。
3. **应用 Softmax ($\text{softmax}(...)$)**
   缩放后的分数随后通过 softmax 函数处理。softmax 会按行应用于缩放分数的矩阵。

   - 目的：Softmax 将分数转换为和为 1 的正值。结果矩阵中的每个值都可以解释为注意力权重 (weight)。位置 $(i, j)$ 的权重表示位置 $i$ 的查询应关注位置 $j$ 的值的注意力比例。权重越高，重要性越高。
   > 流程图展示了缩放点积注意力所涉及的步骤，从输入查询、键和值矩阵到最终的输出上下文 (context)向量。
4. **乘以值矩阵 ($...V$)**
   最后，注意力权重矩阵（softmax 的输出）与值矩阵 $V$ 相乘。

   - 结果：这一步计算值向量的加权和。对于每个位置 $i$，输出向量 $z_i$ 被计算为 $\sum_{j} \text{注意力\_权重}_{ij} v_j$。简单来说，值向量 $v_j$ 对输出 $z_i$ 的贡献与其所关注的位置 $j$ 的程度成比例。
   - 最终的输出矩阵维度为 ($seq\_len, d_v$)，其中 $d_v$ 是值向量的维度。该矩阵的每一行都是对应输入元素的一个包含了上下文信息的表示，它基于计算出的注意力权重整合了整个序列的信息。

### 原理与总结

缩放点积注意力为模型提供了一种高效的方式，来衡量序列中不同元素相互之间的重要性。通过比较每个元素的查询与其所有其他元素的键，它会生成注意力分数。这些分数经过缩放并通过 softmax 归一化 (normalization)后，决定了如何将值向量 (vector)组合成每个元素的精炼表示。这种机制使模型能够在处理任何给定部分时动态地集中于输入序列的相关部分，从而为 Transformer 模型如何在文本中理解上下文 (context)奠定了基础。

在下一节中，我们将了解这种机制如何通过使用多个“头”进行扩展，以同时捕捉不同类型的关系。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; DOI: [10.5591/978-0-9883408-0-3_2](https://doi.org/10.5591/978-0-9883408-0-3_2)
  介绍了Transformer架构和Scaled Dot-Product Attention机制。
- [Natural Language Processing with Transformers: Building Innovative Applications with 🤗 Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098103981/) — Lewis Tunstall, Leandro von Werra, and Thomas Wolf (2022)
  Publisher: O'Reilly Media
  一份全面的Transformer模型指南，包含注意力机制的详细说明和实践应用。
- [CS224N: Natural Language Processing with Deep Learning - Lecture 8: Transformers and Attention](http://web.stanford.edu/class/cs224n/slides/cs224n-2023-lecture08-transformers.pdf) — Christopher Manning (2023)
  一份来自知名大学课程的讲义，提供了关于Transformer和注意力机制的学术概述。
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) — Alexander Rush (2018)
  对‘Attention Is All You Need’论文的深度代码实现与说明，为注意力机制提供了实践见解。

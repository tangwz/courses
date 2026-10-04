---
course: "introduction-to-transformer-models"
chapter: "transformer-encoder-decoder-architecture"
lesson: "positional-information-need"
sourceId: 2995
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-3-transformer-encoder-decoder-architecture/positional-information-need"
title: "位置信息的必要性"
description: "讨论Transformer模型为何需要显式位置信息，因为它缺乏循环性。"
order: 3
plots: []
sourceHash: "cdbff5b7de6507d3117ff0adf54753370cd4319ab9e31f9e632d96ab50794b32"
sourceCorrections: []
---

比较循环神经网络 (neural network)（RNN）等序列模型与Transformer架构，会发现它们在处理序列数据的方式上存在一个基本区别。RNN模型就其本质而言，一个接一个地处理输入标记 (token)。这种序列处理自然地包含了标记的顺序。步骤$t$的隐藏状态是步骤$t$的输入和步骤$t-1$的隐藏状态的函数。这种依赖链自然地表示了序列中每个元素的位置。

然而，Transformer模型采取了一种不同的方式。正如前一章所说，自注意力 (self-attention)机制 (attention mechanism)允许模型在计算任何给定标记的表示时，同时衡量输入序列中所有标记的重要性。输入嵌入 (embedding)大多并行地流经自注意力层。虽然这种并行性对于计算速度和获取长距离依赖来说是一个很大的优势，但它也带来一个代价：核心自注意力机制本身不自然地考虑输入标记的顺序。

考虑标准缩放点积注意力计算：


$$
\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


这里，$Q$（查询）、$K$（键）和$V$（值）矩阵源自输入嵌入。如果我们打乱输入标记（及其对应嵌入）的顺序，计算出的标记对之间的注意力分数集合将保持不变，尽管它们会与不同的输出位置相关联。该机制计算每个标记应该如何关注其他标记，这是基于它们的嵌入内容，而不是它们的位置。

让我们用一个简单例子来说明。想象一下句子：

1. “狗咬人”
2. “人咬狗”

这些句子有很不同的含义，完全通过词语的顺序传递。如果我们只是将“狗”、“咬”和“人”的词嵌入输入到没有位置背景的基本自注意力层中，模型将难以区分这两种情况。它会看到相同集合的输入向量 (vector)，只是可能在内部矩阵中以不同方式排列。注意力计算对于输入集合具有排列不变性，因此缺乏RNN模型内置的序列感知能力。

如果没有关于标记位置的显式信息，Transformer模型基本上会将输入看作一个无序集合或“词袋”。这种序列顺序的丢失对大多数自然语言处理任务有害，因为在这些任务中，词语顺序对意义、语法和句法都非常重要。

因此，为了使Transformer模型理解序列顺序，我们必须为其提供显式的位置信息。我们需要一种方式，将信号注入到输入表示中，这些信号告诉模型每个标记在序列中*哪里*出现（例如，“这是第一个词”、“这是第二个词”等）。这些注入的信息必须在它们被主编码器或解码器堆栈处理*之前*与标记嵌入结合。下一节，“位置编码 (positional encoding)解释”，详细说明了通常如何使用数学函数实现这一点。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: arXiv; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍Transformer架构、自注意力机制以及位置编码必要性和机制的开创性论文。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Pages: Chapter 9
  一本涵盖RNN和Transformer等序列模型的综合教材，为NLP中位置信息的重要性提供了背景。
- [Transformers and Self-Attention (CS224N Winter 2023 Lecture Notes)](https://web.stanford.edu/class/cs224n/slides/cs224n-2023-lecture08-transformers.pdf) — Tatsunori Hashimoto (2023)
  Publisher: Stanford University
  来自顶尖大学的课程材料，清晰解释了Transformer架构和位置编码的原理。

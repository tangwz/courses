---
course: "introduction-to-transformer-models"
chapter: "transformer-encoder-decoder-architecture"
lesson: "transformer-architecture-overview"
sourceId: 2991
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-3-transformer-encoder-decoder-architecture/transformer-architecture-overview"
title: "整体架构概览"
description: "提供完整的编码器-解码器结构的整体示意图和解释。"
order: 1
plots: []
sourceHash: "dd4e2f0cfaf798ff8a21893e0ba842f13acb79d7051b71f99d0bc7a4ca520df4"
sourceCorrections: []
---

Transformer 模型，在开创性论文《Attention Is All You Need》中提出，具有编码器-解码器架构。这种结构是机器翻译或文本摘要等序列到序列任务的常见设计。值得注意的是，Transformer 实现这些任务不依赖于循环或卷积，主要利用注意力机制 (attention mechanism)。

Transformer 不像循环神经网络 (neural network) (RNN)那样逐一顺序处理输入标记 (token)，而是同时处理整个输入序列。这种并行处理能力是一项重要创新，且非常依赖于我们之前讨论的自注意力 (self-attention)机制。

整体架构由两大部分组成：

1. **编码器堆栈：** 位于标准图的左侧，其主要作用是处理输入序列并为每个标记生成丰富的、带有上下文 (context)含义的表示。该堆栈由多个相同的编码器层组成（原始论文中通常为6个）。输入序列中的每个标记都流经这些层，并且由于自注意力机制，其表示会根据与序列中所有其他标记的关系得到精细化。
2. **解码器堆栈：** 位于右侧，解码器的任务是逐个生成输出序列中的标记。与编码器类似，它由一堆相同的解码器层组成。在输出生成的每个步骤中，解码器将先前生成的标记作为输入，并结合编码器堆栈生成的最终表示。它使用修改后的自注意力机制（带掩码的自注意力）来只关注其正在生成的输出序列中的先前位置，并使用另一种注意力机制（编码器-解码器注意力）从输入序列的编码表示中获取相关信息。

让我们描绘出这种高层结构：

> Transformer 架构的整体视图，呈现了输入处理、编码器堆栈、输出处理、解码器堆栈和最终输出层。请注意将编码器输出传输到解码器堆栈中每个层的连接。

输入（例如，源语言句子标记）首先通过嵌入 (embedding)层，然后添加位置编码 (positional encoding)。这种组合表示输入到编码器堆栈的底部。最终编码器层的输出作为每个解码器层内编码器-解码器注意力机制的 ($K$) 和值 ($V$) 输入。

同时，输出（例如，目标语言句子标记，在训练期间向右移位）也经过嵌入，与位置编码结合，并输入到解码器堆栈的底部。解码器使用其带掩码的自注意力来考量已生成的输出序列部分，并使用编码器-解码器注意力来参考编码后的输入表示。最后，来自顶部解码器层的输出通过线性变换和 softmax 函数，生成输出词汇表 (vocabulary)中所有可能下一个标记的概率分布。

本章的后续部分将逐一剖析这些组件，包括位置编码、编码器和解码器层的详细结构（包括相加与归一化 (normalization)步骤以及前馈网络），以及最终的输出生成过程。这种分层方法有助于我们理解 Transformer 如何有效地捕获序列内部和序列之间的复杂依赖关系。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer模型，详细阐述了其编码器-解码器架构以及消除循环和卷积的自注意力机制。
- [Natural Language Processing with Transformers: Building Language Models with Attention](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/) — Lewis Tunstall, Leandro von Werra, and Thomas Wolf (2022)
  Publisher: O'Reilly Media
  提供了Transformer模型的深入实践指南，包括对编码器-解码器架构及其组件的详细解释。
- [CS224N: Natural Language Processing with Deep Learning - Course Materials on Transformers](https://web.stanford.edu/class/cs224n/) — Stanford University (2023)
  Publisher: Stanford University
  知名大学课程的综合讲义材料，提供了Transformer架构的教学解释和图示。

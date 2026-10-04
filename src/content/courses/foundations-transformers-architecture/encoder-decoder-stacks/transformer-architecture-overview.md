---
course: "foundations-transformers-architecture"
chapter: "encoder-decoder-stacks"
lesson: "transformer-architecture-overview"
sourceId: 2008
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-5-encoder-decoder-stacks/transformer-architecture-overview"
title: "Transformer 整体架构概览"
description: "原始 Transformer 模型中编码器-解码器堆栈的高层结构。"
order: 1
plots: []
sourceHash: "25ad5f64125fe42cec84d3381b67ea6dcac7ea2861d1bce9dd865883d689d768"
sourceCorrections: []
---

自注意力 (self-attention)机制 (attention mechanism)和位置编码 (positional encoding)等基本组件被组装成完整的 Transformer 架构。最初的 Transformer 模型，在论文《Attention Is All You Need》中提出，采用了编码器-解码器结构，这在机器翻译或文本摘要等序列到序列任务中是一种常见模式。与逐步处理序列的循环模型不同，Transformer 使用注意力机制同时处理整个输入序列，捕获不同距离的关联。

该架构包含两个主要部分：一个编码器堆栈和一个解码器堆栈。

> Transformer 架构的整体示意图，展示了从输入词元 (token)通过编码器和解码器堆栈到输出概率的数据流向。注意承载着编码器输出到解码器的连接。

### 编码器堆栈

编码器的作用是处理整个输入序列，并生成一系列编码了输入信息的连续表示（带有上下文 (context)信息的嵌入 (embedding)）。它由堆叠的 $N$ 个相同层组成（在原始论文中通常 $N=6$）。每个层有两个主要子层：

1. 一个多头自注意力 (self-attention)机制 (attention mechanism)。
2. 一个逐位置全连接前馈网络。

在每个子层周围都使用残差连接，之后是层归一化 (normalization)。这意味着每个子层的输出是 $LayerNorm(x + Sublayer(x))$，其中 $Sublayer(x)$ 是子层本身实现的功能（例如，多头注意力 (multi-head attention)或前馈网络）。自注意力机制允许编码器中的每个位置关注前一层输出中的所有位置，有效捕获输入序列内的关联。前馈网络独立应用于每个位置。

### 解码器堆栈

解码器的作用是生成输出序列，通常以自回归 (autoregressive)方式逐个元素生成。与编码器类似，它也由堆叠的 $N$ 个相同层组成。然而，每个解码器层有三个主要子层：

1. 一个*掩码*多头自注意力 (self-attention)机制 (attention mechanism)。
2. 一个关注编码器堆栈输出的多头*交叉注意力*机制。
3. 一个逐位置全连接前馈网络。

与编码器中一样，在每个子层周围都应用了残差连接和层归一化 (normalization)。*掩码*自注意力确保对位置 $i$ 的预测只能依赖于位置小于 $i$ 的已知输出，保留了生成所需的自回归属性。*交叉注意力*机制是序列到序列功能的核心：它允许解码器中的每个位置关注输入序列中的所有位置（通过编码器的输出表示）。

### 连接编码器和解码器

编码器和解码器之间的连接主要通过每个解码器层中的交叉注意力机制 (attention mechanism)实现。整个编码器堆栈首先处理输入序列，生成一系列输出向量 (vector) $z = (z_1, ..., z_n)$。然后，这些向量 $z$ 被用作*每个*解码器层中交叉注意力子层中\*\*键（K）**和**值（V）**的来源。该交叉注意力层的**查询（Q）\*\*来自解码器内部的前一个子层（掩码自注意力 (self-attention)层）的输出。这使得解码器在生成输出序列的每一步都能关注编码在 $z$ 中的输入序列的相关部分。

### 输入和输出处理

在输入序列进入编码器堆栈之前，输入词元 (token)使用嵌入 (embedding)层转换为向量 (vector)，并将位置编码 (positional encoding)添加到这些嵌入中，以注入序列顺序信息。同样，对于解码器，目标输出词元（训练时右移，推理 (inference)时为先前生成的词元）在进入解码器堆栈之前会进行嵌入并与位置编码结合。

在最终解码器层生成其输出向量后，通常会使用最后的线性变换，接着是 Softmax 函数，将这些向量转换为目标词汇表 (vocabulary)上的概率分布，从而预测输出序列中的下一个词元。

这种整体结构建立在包含注意力机制 (attention mechanism)、残差连接和层归一化 (normalization)的堆叠层之上，构成了 Transformer 模型的基础。后续章节将更详细地分析每个组件，例如编码器和解码器层的具体结构、掩码注意力、交叉注意力、前馈网络和归一化。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: arXiv; Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍Transformer架构的开创性论文，详细阐述了自注意力机制、位置编码以及编解码器设计。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University
  自然语言处理领域的权威教材，提供了Transformer架构及其组件的详细说明。
- [Stanford CS224N: Natural Language Processing with Deep Learning, Lecture Notes](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFaLoRqiKnzVVy_aIG0xbBmMJ_X9nFv0XkJaZa9W7VeZYFyObV_5rejZvGf0R3Q-XpjSdVY0EgmSOH43JxGlK3QPRLD4YzzqwrvWWrrJAcq-GiPBGrUG81olViqf3zbJKU=) — Christopher Manning, Richard Socher (2023)
  Publisher: Stanford University
  来自顶尖大学的课程资料，提供了Transformer模型及其机制的深度解释和可视化。

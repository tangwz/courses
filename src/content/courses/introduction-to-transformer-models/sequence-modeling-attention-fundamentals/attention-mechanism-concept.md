---
course: "introduction-to-transformer-models"
chapter: "sequence-modeling-attention-fundamentals"
lesson: "attention-mechanism-concept"
sourceId: 2961
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-1-sequence-modeling-attention-fundamentals/attention-mechanism-concept"
title: "注意力机制原理介绍"
description: "介绍注意力机制的核心思想：让模型能够关注输入中的相关部分。"
order: 4
plots: []
sourceHash: "00b29082ff6ad3b042eb17b175d2713591afd43fd2e3d0f6164fc90847decf88"
sourceCorrections: []
---

循环神经网络 (neural network) (RNN) 虽然对较短序列有效，但处理较长序列时会遇到困难。主要问题在于将任意长的输入序列的意义压缩到一个单一的、固定大小的隐藏状态向量 (vector)中。这个向量成为了信息瓶颈，使得模型难以记住序列早期部分的细节，尤其是在稍后生成输出时。想象一下尝试用一句话概括一部长篇小说 —— 重要的细节信息必然会丢失。

人类如何处理翻译或概括等复杂任务？我们不会仅仅读完整个源文本，然后完全根据一个内部总结来凭记忆写出译文。相反，我们会选择性地集中注意力。在翻译某个短语时，我们可能会回头看原文中相应的词语或短语。我们会对输入中与当前正在生成的特定输出最相关的部分给予*关注*。

这就是神经网络中**注意力机制 (attention mechanism)**的核心思想。注意力机制不是强迫模型只依赖整个输入的压缩表示，而是一种让模型在生成输出的每一步都能回溯完整输入序列的方法。它能根据当前任务的相关性，动态地为输入的不同部分分配重要性分数（注意力权重 (weight)）。

### 注意力机制 (attention mechanism)的工作方式

可以把模型想象成一次生成一个元素（例如一个词）的输出序列。对于它需要生成的每个输出元素，注意力机制让它能够：

1. **评估相关性：** 将当前状态（它正在尝试生成的内容）与*输入*序列的所有隐藏状态或表示进行比较。
2. **计算重要性分数：** 根据这些比较，为每个输入元素分配一个分数或权重 (weight)。分数越高，表示对生成当前输出元素的相关性越大。
3. **创建上下文 (context)向量 (vector)：** 使用计算出的分数作为权重，计算输入表示的加权和。这会生成一个`上下文向量` – 一个输入的特殊摘要，专门针对当前的输出步骤进行调整，突出最重要的输入部分。
4. **生成输出：** 使用这个上下文向量，通常结合模型的当前隐藏状态，来生成输出序列中的下一个元素。

这个过程使得模型能够根据需要选择性地将其“注意力”集中在输入序列的不同部分，从而克服了纯粹基于RNN的传统编码器-解码器模型的固定大小瓶颈。

> 解码器使用其当前状态生成一个“查询”。该查询与来自所有输入表示的“键”进行比较，以计算注意力分数。这些分数对相应的“值”（也来自输入）进行加权，从而创建一个上下文向量，该向量为解码器的下一步提供信息。

### 查询、键和值的介绍

为了使这个过程更规范化，注意力机制 (attention mechanism)通常使用从序列表示中得出的三种向量 (vector)类型：

- **查询 (Q)：** 表示当前的关注点或信息需求。在序列到序列模型中，这通常是从解码器在当前时间步的状态中获取的。它提出了一个问题：“现在什么信息最相关？”
- **键 (K)：** 与每个输入元素配对，键像是信息内容的标签或标识符。它们与查询进行比较以确定相关性。“这个输入元素与我正在寻找的相符吗？”
- **值 (V)：** 也与每个输入元素相关联，值代表该元素的实际内容或含义。一旦注意力分数（从查询-键比较中得出）被计算出来，这些值就会被加权并求和，以生成上下文 (context)向量。“这是我拥有的信息。”

本质上，这种机制执行一次查找。查询在所有可用的键中进行查找。查询与键之间的匹配程度决定了分配给相应值的权重 (weight)。然后，所有加权的值会被汇总。

这种根据相关性动态加权输入信息的能力，相比于只依赖从最终RNN状态得出的单一静态上下文向量，是一个重要的进步。它使得模型能够处理输入和输出序列中更远距离的依赖关系。

在接下来的部分中，我们将更仔细地考察这些查询、键和值向量通常是如何生成的，以及如何使用这些组成部分计算注意力分数和上下文向量。

## 参考资料

- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (2015)
  Journal: International Conference on Learning Representations; DOI: [10.48550/arXiv.1409.0473](https://doi.org/10.48550/arXiv.1409.0473)
  在序列到序列模型中引入了注意力机制，解决了传统RNNs处理长序列时固定长度上下文向量的限制。
- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008
  提出了Transformer架构，该架构完全依赖自注意力机制，并规范了注意力计算中的查询（Query）、键（Key）、值（Value）交互。
- [CS224n: Natural Language Processing with Deep Learning, Lecture Notes: Attention and Transformers](https://web.stanford.edu/class/cs224n/) — Stanford University, CS224n Course Staff (2023)
  Publisher: Stanford University
  课程材料，清晰解释了注意力机制、其动机以及在序列模型和Transformer中的应用。

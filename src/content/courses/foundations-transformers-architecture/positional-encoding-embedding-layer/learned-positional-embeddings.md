---
course: "foundations-transformers-architecture"
chapter: "positional-encoding-embedding-layer"
lesson: "learned-positional-embeddings"
sourceId: 1997
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-4-positional-encoding-embedding-layer/learned-positional-embeddings"
title: "替代方案：学习型位置嵌入"
description: "讨论直接学习位置嵌入的原理和实现。"
order: 6
plots: []
sourceHash: "b0863ecd789c43ac73fe5ea23ea0bac575ea7f39889ae80c5101aba4a374459d"
sourceCorrections: []
---

正弦位置编码 (positional encoding)提供了一种巧妙且固定的方法，用于注入序列顺序信息。然而，这并非唯一的方法。另一种策略，被 BERT 和最初的 GPT 模型等多个有影响力的架构所采纳，是直接*学习*位置嵌入 (embedding)，这与词嵌入的习得方式类似。

### 学习型位置嵌入 (embedding)的原理

不再依赖预定义的数学函数（正弦波和余弦波），我们可以将序列中的每个位置视为一个独立单元，它需要自己的向量 (vector)表示。核心思想简单明了：

1. 定义一个模型预期能处理的最大可能序列长度，我们称之为 $L_{max}$。这通常在模型设计时确定（例如，512、1024、2048）。
2. 创建一个标准嵌入矩阵 $P$，其形状为 $(L_{max}, d_{model})$，其中 $d_{model}$ 是嵌入维度（与词元 (token)嵌入维度相同）。
3. 此矩阵中的每一行 $P_i$ 代表序列中位置 $i$ 的学习型嵌入向量，其中 $0 \le i < L_{max}$。
4. 这些位置嵌入会被初始化（通常随机初始化或采用某种结构化初始化），然后通过反向传播 (backpropagation)与模型的其他参数 (parameter)一起训练。

### 与词元 (token)嵌入 (embedding)的整合

整合过程与正弦编码相似。给定一个长度为 $L$ 的输入序列（其中 $L \le L_{max}$），我们首先获取词元嵌入 $E_{token} \in \mathbb{R}^{L \times d_{model}}$。然后我们从矩阵 $P$ 中查找前 $L$ 个位置对应的学习型位置嵌入，得到 $E_{position} = [P_0, P_1, ..., P_{L-1}] \in \mathbb{R}^{L \times d_{model}}$。

输入到第一个 Transformer 层的最终表示通常是词元嵌入与学习型位置嵌入的和：

$E_{input} = E_{token} + E_{position}$

这种逐元素相加的方式使得模型能够结合词元的语义含义与其在序列中的绝对位置。

### 优点

- **灵活性：** 主要优点是灵活性。模型不受正弦和余弦等固定函数形式的约束。它理论上可以学习最适合其训练的特定任务和数据集的位置表示。如果某些位置或位置关系对任务来说特别重要，模型有能力在训练时编码这些信息。
- **简洁性：** 实现可以说比计算正弦值更简单。它使用了深度学习 (deep learning)框架中已有的标准嵌入 (embedding)查找机制。

### 缺点

- **外推限制：** 学习型嵌入 (embedding)在处理训练期间遇到的长度超过 $L_{max}$ 的序列时存在根本性问题。由于没有针对 $L_{max}$ 及更长位置的学习型嵌入向量 (vector)，模型无法在不进行重新训练或采用临时策略的情况下直接处理更长的序列。正弦编码，由于基于函数，可以为任意长度的序列生成编码。
- **参数 (parameter)量增加：** 这种方法会为模型引入 $L_{max} \times d_{model}$ 个额外的可训练参数。对于较大的 $L_{max}$ 和 $d_{model}$ 值，这会显著增加模型大小、内存需求，并可能增加过拟合 (overfitting)的风险，尤其是在训练数据有限的情况下。
- **数据依赖性：** 从头开始学习有效的位置表示需要足够的训练数据。在数据有限的场景中，正弦编码的固定结构可能提供更强的归纳偏置 (bias)，从而带来更好的泛化能力。

### 学习型编码与正弦编码的选择

学习型和固定正弦位置编码 (positional encoding)之间的选择通常取决于具体的应用、模型架构理念以及可用数据。

- **学习型嵌入 (embedding)：** 通常在最大序列长度明确且不过分大，并且有足够数据来学习有意义的表示时更受青睐。像 BERT 这样通常在海量数据集上进行预训练 (pre-training)的架构，普遍使用学习型嵌入。
- **正弦编码：** 当处理可变或可能非常长的序列的能力很重要时，或者当关注参数 (parameter)效率时，正弦编码是一个不错的选择。最初的 Transformer 论文采用了正弦编码，它们仍然是可行且有效的选择。

归根结底，两种方法目的相同：为排列不变的自注意力 (self-attention)机制 (attention mechanism)提供关于输入序列中元素顺序的重要信息。下一节将比较它们的特性。

## 参考资料

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova (2018)
  Journal: Proceedings of NAACL-HLT 2019; Pages: 4171-4186; DOI: [10.48550/arXiv.1810.04805](https://doi.org/10.48550/arXiv.1810.04805)
  介绍了BERT模型，这是一个使用学习型位置嵌入来捕捉序列顺序的重要架构。
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever (2018)
  Journal: OpenAI; Publisher: OpenAI
  展示了原始的GPT模型，该模型在其架构中实现了学习型位置嵌入。
- [Speech and Language Processing (3rd Edition)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Prentice Hall
  一本详细介绍Transformer模型及各种位置编码方法的综合性教科书。

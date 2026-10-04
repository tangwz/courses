---
course: "foundations-transformers-architecture"
chapter: "positional-encoding-embedding-layer"
lesson: "input-embedding-layer"
sourceId: 1976
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-4-positional-encoding-embedding-layer/input-embedding-layer"
title: "输入嵌入层转换"
description: "将输入词元转换为密集向量表示。"
order: 2
plots: []
sourceHash: "1fadfcb5ec39dfe4ca23463bd828e0f6b9b40df232988d8b83b5fc4d67601bc9"
sourceCorrections: []
---

为了准备引入序列顺序信息，Transformer需要一种方式来表示输入词元 (token)本身，使其适合神经网络 (neural network)处理。原始文本首先通过词元化过程（使用字节对编码或WordPiece等方法）转换为整数序列，称为词元ID。然而，这些离散的词元ID不直接适合作为注意力机制 (attention mechanism)或后续神经网络层的输入。它们无法体现不同词元之间的相似性或关联。

输入嵌入 (embedding)层作为最初的转换步骤，将这些整数词元ID转换为密集的、连续的向量 (vector)表示。这与独热编码等旧技术形成鲜明对比，后者会生成维度非常高且稀疏的向量（大部分为零，只有一个一）。而密集嵌入则能在维度低得多的空间中捕获词元间的语义相似性，这个空间通常称为 $d_{model}$（模型的隐藏维度）。

### 机制：嵌入 (embedding)查找

其核心是，输入嵌入层作用为一个查找表。该表由一个权重 (weight)矩阵表示，我们称之为 $E$，其维度为 $V \times d_{model}$，这里的 $V$ 是词汇表 (vocabulary)的大小（模型识别的唯一词元 (token)总数），$d_{model}$ 是选定的嵌入维度。

当给定输入词元ID序列 $[t_1, t_2, ..., t_n]$ 时，嵌入层会从矩阵 $E$ 中获取每个词元ID对应的向量 (vector)。如果 $t_k$ 是位置 $k$ 的词元ID，它的嵌入向量 $e_k$ 就是矩阵 $E$ 中由 $t_k$ 索引的行：

$e_k = E_{t_k}$

此操作有效地将输入序列中的每个整数ID映射到大小为 $d_{model}$ 的密集向量 (dense vector)。

> 通过嵌入查找将离散词元ID映射到连续向量表示。

### 可学习表示

值得注意的是，嵌入 (embedding)矩阵 $E$ 不是固定的；它的值是可学习的参数 (parameter)。在训练过程中，梯度通过网络反向传播 (backpropagation)，嵌入向量 (vector)被调整以最小化总损失函数 (loss function)。这意味着模型学会在 $d_{model}$ 维嵌入空间中，将具有相似语义或功能的词元 (token)放置得更近。例如，训练后“国王”和“王后”等词语可能最终具有相似的嵌入向量，反映出它们的语义关联 (semantic relationship)。

### 输出与维度

输入嵌入 (embedding)层的输出是一个向量 (vector)序列，$[e_1, e_2, ..., e_n]$，每个 $e_k$ 都是大小为 $d_{model}$ 的向量。此序列现在在连续空间中表示输入词元 (token)，但保留了原始序列长度。维度 $d_{model}$（例如512、768、1024）是Transformer架构的一个基本超参数 (parameter) (hyperparameter)，决定了模型大部分层中向量表示的宽度。

这个词元嵌入序列是后续加入位置信息的基本组成，正如接下来将讨论的。尽管这些嵌入现在带有从数据中学到的语义，但它们仍然缺乏对其在原始序列中位置的固有表示，这是一个通过位置编码 (positional encoding)解决的局限。

## 参考资料

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) — Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean (2013)
  Journal: arXiv preprint arXiv:1301.3781; DOI: [10.48550/arXiv.1301.3781](https://doi.org/10.48550/arXiv.1301.3781)
  详细说明学习表示语义关系的密集、连续词向量，为Transformer输入嵌入奠定基础。
- [CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) — Diyi Yang, Tatsunori Hashimoto (2023)
  Publisher: Stanford University
  一门关于自然语言处理基础课题的大学课程，包括分词、词嵌入和Transformer设计。
- [Natural Language Processing with Transformers: Building Innovative Applications with 🤗 Transformers](https://transformersbook.com/) — Lewis Tunstall, Leandro von Werra, Thomas Wolf (2022)
  Publisher: O'Reilly Media
  一本关于使用Transformer进行现代自然语言处理的实践指南，解释了分词方法（BPE、WordPiece）和嵌入处理。

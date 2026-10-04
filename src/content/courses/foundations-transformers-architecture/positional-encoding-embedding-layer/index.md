---
course: "foundations-transformers-architecture"
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-4-positional-encoding-embedding-layer"
sourceId: 534
chapter: "positional-encoding-embedding-layer"
title: "位置编码与嵌入层"
order: 4
description: "了解Transformer如何使用位置编码处理序列顺序，以及输入嵌入层的作用。"
hasQuiz: false
---

前几章介绍了自注意力机制，它使得模型能够评估不同输入元素之间的相对重要性。然而，核心的注意力计算是对查询、键和值集合进行操作，自身无法体现输入的序列顺序。本章将解决这个问题。

我们将首先考察输入嵌入层，它负责将输入标记转换为连续向量表示。随后主要研究编码位置信息的方法。我们将研究广泛使用的正弦位置编码方案，包括其数学表达式，例如：

$$ PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}}) $$
$$ PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}}) $$

我们将分析使这些函数适用的特性，以及它们通常如何与标记嵌入结合使用。我们还将讨论可学习的位置嵌入等其他方法，并比较它们的特点。

---
course: "foundations-transformers-architecture"
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-2-attention-mechanism-core-concepts"
sourceId: 529
chapter: "attention-mechanism-core-concepts"
title: "注意力机制：基本原理"
order: 2
description: "详细介绍注意力机制、查询、键、值以及 Transformer 中使用的缩放点积注意力。"
hasQuiz: false
---

第一章说明了传统序列模型的局限性，特别是它们在处理长距离依赖方面的困难以及固定长度上下文向量所带来的限制。本章将介绍注意力机制，这是一种旨在解决上述问题的方法，它能让模型在生成输出时动态地关注输入序列中的相关部分。

您将学习注意力背后的基本原理，首先会说明摆脱固定上下文表示的缘由。我们将使用查询（$Q$）、键（$K$）和值（$V$）的抽象来界定一般的注意力框架。广泛使用的缩放点积注意力的数学算法将被说明，其中会包括缩放因子的意义所在：
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
我们还会分析 softmax 函数在生成注意力权重中的作用，并讨论如何利用适合并行处理的矩阵计算方法高效实现这些操作。一个实践练习将指导您实现这种主要的注意力机制。在本章末尾，您将清楚地理解注意力如何在最基本的层面上运作，为学习 Transformer 架构中使用的更复杂变体做好准备。

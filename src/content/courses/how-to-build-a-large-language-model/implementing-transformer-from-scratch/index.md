---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-10-implementing-transformer-from-scratch"
sourceId: 1088
chapter: "implementing-transformer-from-scratch"
title: "从零开始实现Transformer"
order: 10
description: "逐步实现Transformer架构的精简版本，并使用常用的深度学习框架。"
hasQuiz: false
---

在前面章节中，我们已经了解了Transformer架构的理论原理，本章将重点把这些理论转化为可运行的代码。我们将使用一个常用的深度学习框架，一步步地构建Transformer模型的重要组成部分。

你将学习实现：
*   缩放点积注意力，这种基本的注意力机制。
*   多头注意力，它将多种注意力视角结合起来。
*   逐位置前馈网络。
*   编码器和解码器层的结构，包括残差连接和层归一化。
*   通过组装这些部分来构建完整的Transformer模型。

本章结束后，你将拥有一个清晰、可操作的Transformer实现，这将使你对这些模型在代码层面如何运作有清晰的认识，并为你后续关于扩展和优化的章节做好准备。我们将建立一个基本的项目配置，并按逻辑顺序逐步实现每个架构元素。

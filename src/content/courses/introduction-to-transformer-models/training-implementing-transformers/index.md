---
course: "introduction-to-transformer-models"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-4-training-implementing-transformers"
sourceId: 671
chapter: "training-implementing-transformers"
title: "Transformer模型的训练与实现"
order: 4
description: "涵盖Transformer训练的实践要点，包括分词、优化和基本的实现步骤。"
hasQuiz: true
---

在前面章节对Transformer模型的架构，包括注意力机制和编解码器结构进行了详细说明后，本章我们将把这些原理付诸实践。本章将讨论训练和实现Transformer模型的必要步骤。

您将了解如何为Transformer准备数据，包括常见的分词技术，如字节对编码（BPE），以及如何创建格式正确的输入批次，包括填充和注意力掩码。之后，我们将考察训练过程本身，讨论合适的损失函数（如交叉熵）、Transformer模型常用的优化算法（如Adam）、学习率调度技术以及像Dropout这样的正则化方法。最后，我们将概述如何将前面讨论的组件组装成一个基本可运行的模型，并简要介绍提供预训练Transformer实现方式的库的运用。

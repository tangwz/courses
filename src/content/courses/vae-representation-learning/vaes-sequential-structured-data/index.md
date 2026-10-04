---
course: "vae-representation-learning"
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-6-vaes-sequential-structured-data"
sourceId: 1143
chapter: "vaes-sequential-structured-data"
title: "VAEs在处理序列和结构化数据中的应用"
order: 6
description: "将VAEs应用于序列数据 (循环VAE) 和结构化数据 (图VAE)，包括自然语言处理和时间序列等应用。"
hasQuiz: false
---

前几章主要关注的是将样本视为独立数据的VAEs，例如单个图像。然而，许多数据集都具有固有结构。例如，文本数据由有序的词语序列构成，时间序列数据表现出时间上的关联性，而社交网络或分子则最适合以图的形式表示。本章将对VAE框架进行扩展，以有效建模此类序列化和结构化信息。

您将学习如何：
*   调整VAE架构以处理序列数据，从而得到循环VAEs (RVAEs) 等模型，适用于时间序列建模等任务。
*   在VAEs中加入注意力机制，以捕捉序列中常存在的长距离关联性。
*   专门为图结构数据开发VAEs，实现在网络和关系数据上的表示学习。
*   了解这些VAE变体在自然语言处理等方面以及用于分析动态系统方面的应用。
*   了解针对时间数据设计的VAEs与已有状态空间模型之间的联系。

学完本章后，您将能够将VAEs应用于更广范围的复杂数据形式，捕捉其中包含的丰富关联性。

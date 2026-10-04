---
course: "mixture-of-experts"
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts/chapter-1-moe-foundations-sparse-models"
sourceId: 897
chapter: "moe-foundations-sparse-models"
title: "稀疏专家模型基本原理"
order: 1
description: "回顾条件计算的核心思想以及专家混合模型在模型扩展方面的动机。"
hasQuiz: false
---

本章旨在帮助您理解专家混合 (MoE) 模型。我们首先分析条件计算的原理，这是使稀疏模型能够高效扩展的核心思想。您将了解MoE模型如何通过对每个输入选择性地激活其一小部分参数（即“专家”）来与标准密集网络不同。

我们将定义MoE层的基本结构，并比较稀疏与密集激活模式的计算特点。本章最后会提出标准MoE层的数学公式，详细说明输入token如何通过门控网络$G(x)$路由到特定的专家网络$E_i(x)$，以及它们的输出如何组合。这种组合通常表示为：

$$y = \sum_{i=1}^{N} G(x)_i E_i(x)$$

在此公式中，$G(x)_i$表示专家$i$的门控决策或权重，$N$是专家总数。在后续章节中处理高级MoE架构和训练过程之前，理解这些基本原理是必要的。

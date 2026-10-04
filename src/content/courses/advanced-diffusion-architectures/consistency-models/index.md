---
course: "advanced-diffusion-architectures"
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-5-consistency-models"
sourceId: 1001
chapter: "consistency-models"
title: "一致性模型"
order: 5
description: "了解一致性模型，这是一种蒸馏扩散模型以实现极快单步或少量步骤生成的技术。"
hasQuiz: false
---

尽管扩散模型在生成任务中取得了顶尖成果，但其迭代采样过程通常需要数百或数千步，带来了可观的计算开销。本章将介绍一致性模型，这是一项旨在显著减少生成所需步数（有可能减少到一步）的最新进展。

你将学习一致性模型的基本原理，该原理规定了属于同一概率流ODE轨迹的点映射到相同的初始点 $x_0$。我们将考察此属性如何让直接生成成为可能。我们将讲解训练这些模型的两种主要方法：一致性蒸馏（它将知识从预训练扩散模型转移）和一致性训练（一种独立方法）。你将学会如何使用一步或少量步骤从一致性模型进行采样，讨论架构方面的考量，并理解推理速度与所得样本质量之间重要的权衡。

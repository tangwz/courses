---
course: "advanced-diffusion-architectures"
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-1-diffusion-foundations-advanced-noise"
sourceId: 988
chapter: "diffusion-foundations-advanced-noise"
title: "基础回顾与进阶噪声调度"
order: 1
description: "回顾核心扩散原理（DDPM、DDIM），并研究进阶方差与噪声调度技术以改进生成效果。"
hasQuiz: false
---

在构建更复杂的扩散模型之前，有必要确保对基本构成有扎实的掌握。本章首先简要回顾去噪扩散概率模型（DDPM）和去噪扩散隐式模型（DDIM），侧重于它们的核心机制、正向过程 $q(x_t|x_0)$ 和逆向过程 $p_\theta(x_{t-1}|x_t)$ 等数学表述，以及分数匹配与常微分方程之间的联系。

随后，我们将审视标准的噪声调度技术，例如线性调度和余弦调度，评估它们的优点并找出它们在特定情况下的局限性。基于此分析，您将学习如何设计针对特定需求定制的噪声调度，并研究实现学习型方差调度的方法，其中模型本身会在训练期间确定最佳噪声水平。

在本章结束时，您将温习对核心扩散原理的理解，并获得实现和评估进阶噪声调度策略的实践知识，为后续更复杂的架构和训练方法奠定基础。

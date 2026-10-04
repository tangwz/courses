---
course: "vae-representation-learning"
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-3-advanced-vae-architectures"
sourceId: 1135
chapter: "advanced-vae-architectures"
title: "VAE 进阶架构与改进"
order: 3
description: "学习条件 VAE、VQ-VAE、分层 VAE 和 Beta-VAE 等进阶 VAE 架构，以增强生成能力。"
hasQuiz: false
---

在明确了变分自编码器（VAE）的数学原理，包括证据下界（ELBO）$L_{ELBO}$ 和重参数化技巧之后，我们现在将关注点转移到架构提升上。尽管标准 VAE 为生成建模和表征学习提供了强力框架，但它的能力可以大大扩展。本章审视多种进阶 VAE 架构与改进，旨在解决常见问题，例如提升样本真实度、处理更复杂的数据类型，以及获得更有结构或可解释的潜在空间。

我们将研究条件 VAE（CVAE）如何根据特定属性实现受控数据生成。你将了解到分层 VAE 如何建模具有多层抽象的数据，以及矢量量化 VAE（VQ-VAE）如何引入离散潜在变量，通常会得到更清晰的生成样本。此外，我们还将介绍自回归解码器和归一化流等强大组件的结合，以创建更具表达力的 VAE。本章还将介绍 Beta-VAE 和 FactorVAE 等特定改进，它们旨在提升所学表征的解耦效果。每个部分都将帮助你做好准备，以实现和评估这些精密模型。

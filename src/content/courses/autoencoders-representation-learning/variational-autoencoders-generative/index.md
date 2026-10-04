---
course: "autoencoders-representation-learning"
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-4-variational-autoencoders-generative"
sourceId: 649
chapter: "variational-autoencoders-generative"
title: "变分自编码器用于生成式建模"
order: 4
description: "学习VAE的概率框架、重参数化技巧、ELBO以及它们在生成式建模中的应用。"
hasQuiz: false
---

前面的章节主要介绍了自编码器，用于降维和学习高效的数据编码。尽管在重建方面有效，但标准自编码器及其正则化变体通常产生的潜在空间不太适合生成新的、连贯的数据样本。生成数据需要一个结构更清晰的潜在空间，这种空间通常基于概率。

本章将侧重于变分自编码器 (VAE)，这是一种专门为生成式建模而开发的自编码器。我们将从确定性编码器和解码器转向概率框架。你将学习VAE如何将潜在变量视为概率分布而非固定向量。我们将讲解支撑VAE的基本原理，包括：

*   编码器和解码器的概率解释。
*   潜在变量模型视角。
*   重参数化技巧，一种使用梯度下降训练VAE的必要技巧。
*   证据下界 (ELBO) 的推导和组成部分，它作为目标函数，平衡重建质量与潜在空间分布 $q_\phi(z|x)$ 对先验 $p(z)$ 的正则化。

通过理解这些组成部分，你将掌握VAE如何学习一个有意义的潜在空间，从中可以有效采样并生成与训练数据相似的新数据点。我们还将提及条件变分自编码器 (CVAE) 等扩展内容，并实现一个基本的VAE用于图像生成实际任务。

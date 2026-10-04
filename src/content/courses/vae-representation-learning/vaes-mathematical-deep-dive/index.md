---
course: "vae-representation-learning"
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-2-vaes-mathematical-deep-dive"
sourceId: 1132
chapter: "vaes-mathematical-deep-dive"
title: "变分自编码器：数学原理"
order: 2
description: "理解VAEs的严谨数学推导，以及ELBO、重参数化和KL散度。包含实践实现。"
hasQuiz: false
---

在确立了概率生成模型和表示学习的要点之后，本章侧重介绍变分自编码器（VAEs）的数学构造。对这些数学原理的扎实理解，对于有效使用和扩展VAEs来说是不可或缺的。

在这里，你将学习从变分推断的原理推导出VAEs。我们将分析证据下界 (ELBO)，通常写作 $L_{ELBO}$，它构成了训练VAEs的主要目标函数。你将理解重参数化技巧，这是一个重要技巧，它使通过随机隐变量进行基于梯度的优化成为可能。Kullback-Leibler (KL) 散度（通常写作 $D_{KL}(q(z|x) || p(z))$）在VAEs目标函数中的作用和解释也将被详细阐述。此外，我们还将涵盖编码器和解码器网络的设计考虑因素，讨论常见的VAEs训练难题，并分析VAEs目标函数的不同表达形式。本章包含一个实践部分，关于如何实现一个VAEs并执行诊断，以将理论与应用联系起来。

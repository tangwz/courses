---
course: "generative-adversarial-networks-gans"
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-3-gan-training-stabilization"
sourceId: 628
chapter: "gan-training-stabilization"
title: "GAN训练的动态与稳定性"
order: 3
description: "解决GAN训练中模式崩溃等问题，可应用Wasserstein损失、梯度惩罚和谱归一化。"
hasQuiz: false
---

尽管生成对抗网络 (GANs) 的主要构想是生成器与判别器进行一场极小极大博弈，但在实际操作中，要实现稳定且有效的训练，通常需要特定的技术。标准的GAN训练可能会遇到问题，例如模式崩溃（生成器只产生有限的输出种类），或者梯度消失/爆炸，导致收敛困难。

本章将着重介绍这些训练难点，并介绍旨在提升稳定性的方法。我们将考察：

* 产生常见训练问题的原因。
* 基于 Wasserstein 距离 ($W_1$) 的替代损失函数，如 Wasserstein GANs (WGANs) 中所应用的。
* 用于强制执行 WGANs 所需的 Lipschitz 约束的方法，例如权重裁剪和梯度惩罚 (WGAN-GP)。
* 应用于判别器的正则化方法，例如谱归一化。
* 其他稳定方案，例如双时间尺度更新规则 (TTUR) 和相对论性 GANs。

您将了解这些方法背后的动因，以及它们如何改变训练过程，以促进更好的收敛并避免常见的失败情况。我们还将介绍实际的实现细节，使您能够将这些方法应用于您自己的 GAN 项目中。

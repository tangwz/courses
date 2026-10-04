---
course: "applied-autoencoders-feature-extraction"
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-6-variational-autoencoders-structured-latent-spaces"
sourceId: 1157
chapter: "variational-autoencoders-structured-latent-spaces"
title: "变分自编码器（VAEs）与结构化潜在空间"
order: 6
description: "了解变分自编码器（VAEs）、其概率特性，以及它们如何学习用于特征的结构化潜在空间。"
hasQuiz: true
---

前面的章节介绍了自编码器，它们主要是用于降维和特征学习的工具，网络通过学习重建其输入。本章介绍变分自编码器（VAEs），这是一种独特的自编码器，结合了概率视角。VAEs 的设计目的不仅是重建，还在于学习一个连续且有结构的潜在空间，这使它们在生成模型方面特别有效——即生成新的数据样本。

通过学习本章，您将了解到：

*   生成模型的基本思想以及 VAEs 如何适应这个方面。
*   变分自编码器的核心原理，包括它们的概率编码器和解码器。
*   VAE 编码器如何构造以输出参数（通常是均值 $\mu$ 和对数方差 $\log(\sigma^2)$），这些参数为每个输入定义了潜在空间中的概率分布。
*   重参数化技巧，这是一种允许梯度通过采样过程反向传播的方法，这对训练 VAEs 非常重要。
*   VAE 解码器作为生成器，如何使用从潜在分布中抽取的样本来生成新数据。
*   VAE 损失函数，它通常由两个主要部分组成：重建损失（例如，均方误差或二元交叉熵）和一个 Kullback-Leibler (KL) 散度项。KL 散度作为正则项，促使学到的潜在分布接近先验分布（通常是标准正态分布 $\mathcal{N}(0, I)$）。总损失可以表示为 $$L_{VAE} = \text{ReconstructionLoss} + D_{KL}(q(z|x) || p(z))$$
*   VAE 潜在空间的特性，例如其平滑性以及如何促进数据点之间的插值。
*   使用 VAEs 潜在空间中的均值向量（或样本）作为特征，用于其他机器学习任务的方法。

一个实践部分将演示如何构建 VAE 并直观检查其潜在空间，以观察其学习到的结构。

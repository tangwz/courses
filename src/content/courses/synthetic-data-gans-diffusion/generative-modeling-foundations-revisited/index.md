---
course: "synthetic-data-gans-diffusion"
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-1-generative-modeling-foundations-revisited"
sourceId: 1026
chapter: "generative-modeling-foundations-revisited"
title: "生成模型基本原理回顾"
order: 1
description: "回顾生成模型、概率分布的主要思想，以及高维数据生成中的难题。"
hasQuiz: false
---

构建StyleGAN或扩散模型等复杂的生成模型，需要对背后原理有扎实的理解。本章将重温学习更精进方法前需要的基本知识。

我们将首先回顾用于生成的概率建模，重点在于模型如何旨在获取数据分布，该分布通常表示为 $p_{data}(x)$。我们将对生成模型的主要类型进行分类，以明确GAN和扩散模型的归属。

接着，我们会谈及生成图像或音频等高维数据时遇到的常见难题，包括如何管理其复杂性并保证输出多样性。之后，我们将重新温习生成对抗网络（GANs）的主要思想，回顾其核心组成部分，如生成器 $G$ 和判别器 $D$，以及它们的对抗训练目标，该目标常常表述为极小极大博弈：
$$ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))] $$
最后，我们将介绍扩散模型背后的直观理解，涵盖其前向（加噪）和反向（去噪）过程的主要思想。

完成本章可确保您已温习所需知识，能够应对之后章节介绍的进阶架构、训练方法和评估方法。

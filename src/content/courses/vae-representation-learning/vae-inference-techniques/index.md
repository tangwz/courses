---
course: "vae-representation-learning"
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-4-vae-inference-techniques"
sourceId: 1139
chapter: "vae-inference-techniques"
title: "VAE中的推断方法与摊销"
order: 4
description: "考察VAE中的高级推断方法，包括IWAE、摊销推断和结构化变分推断，以获得更好的后验近似。"
hasQuiz: false
---

变分自编码器（VAE）的性能取决于其近似后验 $q_\phi(z|x)$ 与真实后验 $p_\theta(z|x)$ 的匹配程度。尽管证据下界（$ELBO$）提供了一个可计算的目标，但通常对简单的摊销推断网络 $q_\phi(z|x)$ 的依赖会限制模型的表达能力和此界限的紧密程度。本章侧重于改进VAE推断部分的方法。

我们将详细考察摊销变分推断，讨论其实用优势和固有假设，例如均值场近似。你将学习旨在克服这些限制的方法，包括用于获得更灵活后验形式的结构化变分推断，以及用于实现更紧密$ELBO$估计的重要性加权自编码器（IWAE）。我们还会介绍其他方法，如半摊销推断、辅助变量的使用以及对抗性变分贝叶斯（AVB），以进一步提高后验近似的准确度。到本章末尾，你将明白如何选择和应用这些高级推断策略，以构建具有改进性能特点的VAE。

---
course: "fundamentals-quantum-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-6-quantum-generative-models"
sourceId: 379
chapter: "quantum-generative-models"
title: "量子生成模型"
order: 6
description: "了解量子生成建模技术，包括量子电路玻恩机 (QCBMs) 和量子生成对抗网络 (QGANs)。"
hasQuiz: false
---

生成式建模旨在学习数据分布 $p_{\text{data}}(x)$，并创建与原始数据类似的新的样本 $x \sim p_{\text{model}}(x)$。本章介绍处理此任务的量子方法。经典的生成对抗网络（GANs）和变分自编码器（VAEs）等方法提供了支撑，但量子电路或能提供不同方式来表示复杂概率分布并从中采样。

您将学习两种主要的量子生成模型：
*   **量子电路玻恩机 (QCBMs)：** 学习参数化量子电路如何隐式地建模由其测量结果 $p(x) = |\langle x | \psi(\theta) \rangle|^2$ 定义的概率分布。
*   **量子生成对抗网络 (QGANs)：** 了解量子电路作为生成器，与经典或量子判别器竞争的框架。

我们将讨论这些模型的运行原理、架构以及相关的具体训练难题，例如平衡 QGANs 中的对抗性组件。您还将学习评估生成样本质量的方法，以及从训练好的量子模型中有效采样所需的技巧。本章最后将通过一个实现基本 QGAN 的实践练习结束。

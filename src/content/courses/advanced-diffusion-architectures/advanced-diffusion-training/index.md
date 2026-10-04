---
course: "advanced-diffusion-architectures"
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-4-advanced-diffusion-training"
sourceId: 998
chapter: "advanced-diffusion-training"
title: "进阶训练技巧"
order: 4
description: "学习精密的训练策略，包括无分类器引导、进阶损失函数和参数化方法。"
hasQuiz: false
---

尽管精密的架构对扩散模型来说非常必要，但有效的训练流程对于实现高质量的生成同样重要。本章将不再局限于基本的训练循环，而是介绍能提升样本质量、改善稳定性和提供对生成过程更精细控制的方法。

我们将介绍：
*   **无分类器引导（CFG）：** 部署和优化CFG，以便使用模型本身来引导生成，并调整引导强度$s$。
*   **进阶损失函数与参数化：** 理解$v$预测等目标，并比较模型输出作为噪声($\epsilon$)参数化和作为原始数据($x_0$)参数化时的差异。
*   **训练稳定性：** 应用梯度裁剪和权重的指数移动平均（EMA）等方法，以确保更平滑的收敛。
*   **训练效率：** 采用混合精度（FP16/BF16）计算来加快训练速度并减少内存占用。

完成本章学习后，您将了解如何应用这些进阶训练方法来构建和优化高性能扩散模型。

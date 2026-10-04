---
course: "intro-diffusion-models"
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-4-model-architecture-training"
sourceId: 995
chapter: "model-architecture-training"
title: "模型架构与训练"
order: 4
description: "介绍扩散模型中常用的 U-Net 架构以及训练目标。"
hasQuiz: true
---

在确立了正向和逆向扩散过程的数学体系后，我们现在将说明如何实现逆向（去噪）步骤。这需要一个可训练的模型，它能够根据含噪声数据 $x_t$ 学习估算在特定时间步 $t$ 所添加的噪声。

本章主要说明这一核心模型的架构和训练流程。我们将介绍：

*   **U-Net 架构：** 考察 U-Net 的结构，这是一种常用于处理图像类数据并保留空间信息的卷积网络，这对于噪声预测是不可或缺的。
*   **时间步整合：** 提供扩散时间步 $t$ 作为网络输入的方法，使其能够相应地调整预测。
*   **训练目标：** 定义损失函数，通常是实际添加的噪声与网络预测的噪声之间的均方误差 (MSE)。我们将简要提及它与整体概率目标之间的关联。
*   **训练算法：** 概述使用批次数据训练噪声预测网络的分步流程。
*   **实际配置：** 讨论在代码中实现 U-Net 和时间步嵌入的重要方面。

完成本章后，你将理解扩散模型中使用的神经网络的设计原则以及训练它执行去噪任务的流程。

---
course: "intro-diffusion-models"
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-5-sampling-generation-process"
sourceId: 997
chapter: "sampling-generation-process"
title: "采样与生成过程"
order: 5
description: "学习如何通过运行逆向扩散过程来生成新的数据点，包括DDPM和更快的DDIM采样。"
hasQuiz: true
---

在我们已经弄清扩散模型如何训练以预测损坏数据中的噪声之后，本章将侧重于其逆向操作：生成新的数据样本。我们将了解如何从纯噪声（通常从高斯分布 $x_T \sim \mathcal{N}(0, \mathbf{I})$ 中采样）开始，并迭代应用所学到的逆向扩散过程，以生成一个干净的数据点 $x_0$。

您将学习标准去噪扩散概率模型（DDPM）采样算法的标准步骤。接下来，我们将介绍去噪扩散隐式模型（DDIM），这是一种相关但通常更快的采样技术，可在速度和样本质量之间提供不同的平衡。最后，我们将讨论在代码中构建这些采样循环所需的实现细节。

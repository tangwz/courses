---
course: "cnns-for-computer-vision"
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-8-model-compression-efficient-dl"
sourceId: 619
chapter: "model-compression-efficient-dl"
title: "模型压缩与高效深度学习"
order: 8
description: "学习构建高效CNN模型的方法，包括剪枝、量化、知识蒸馏和高效架构。"
hasQuiz: false
---

部署我们已学习的复杂CNN模型时，常常遇到实际制约，例如内存限制、计算预算和延迟要求，尤其是在边缘设备或移动平台上。本章主要介绍构建更高效深度学习模型的方法，同时不明显降低其性能。

您将学习到一些减小模型体积和计算成本的方法。我们将讲解以下内容：

*   **网络剪枝：** 从网络中移除不那么重要的权重或结构。
*   **知识蒸馏：** 训练一个较小的“学生”模型来模仿较大的“教师”模型。
*   **量化：** 减小模型权重和激活值的数值精度，例如从32位浮点数（$FP32$）到8位整数（$INT8$）。
*   **高效架构：** 研究本身已针对速度和大小进行优化的模型设计，例如MobileNets。
*   **神经架构搜索（NAS）：** 关于寻找高效网络结构的自动化方法的概览。

目标是为您提供优化深度学习模型的策略，以便部署在资源受限的环境中。我们还将安排一次实践环节，应用这些方法的一部分。

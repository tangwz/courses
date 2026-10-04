---
course: "introduction-autoencoders-feature-learning"
sourceUrl: "https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-3-how-autoencoders-learn"
sourceId: 1149
chapter: "how-autoencoders-learn"
title: "自动编码器如何学习"
order: 3
description: "了解自动编码器如何训练，内容包括损失函数、优化、前向与反向传播要点。"
hasQuiz: true
---

前面章节介绍了自动编码器及其主要组成部分。现在，我们审视这些网络如何从数据中学习。主要目标是最小化重建误差，即原始输入与自动编码器输出之间的差异。

本章将说明：
*   **训练目的**: 自动编码器如何致力于精确重建输入。
*   **损失函数**: 用于衡量重建误差的指标，例如均方误差 ($MSE$) 和二元交叉熵 ($BCE$)。
*   **优化**: 像梯度下降这样的基本过程，用于调整自动编码器内部设置以提高性能。
*   **数据流**: 理解前向传播（从输入到输出）和反向传播（误差调整）的概览。
*   **训练周期**: 诸如周期（epochs）和批次（batches）等组织学习过程的要点。

我们还将简要介绍过拟合和欠拟合。到本章结束时，您将了解准备自动编码器进行训练所涉及的步骤。

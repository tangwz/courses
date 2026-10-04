---
course: "applied-autoencoders-feature-extraction"
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-3-building-first-autoencoder-feature-extraction"
sourceId: 1154
chapter: "building-first-autoencoder-feature-extraction"
title: "构建你的第一个用于特征提取的自动编码器"
order: 3
description: "一步步构建自动编码器并使用其潜在空间从数据中提取特征的指南。"
hasQuiz: true
---

自动编码器的核心要点已介绍完毕，本章着重介绍构建第一个用于特征提取模型所需采取的实际步骤。我们将逐步完成整个工作流程，从初始数据准备到提取和理解学习到的特征。

本章将指导您完成以下内容：
*   准备和预处理数据，包括归一化和缩放，以便有效训练自动编码器。
*   设计编码器和解码器网络，考虑诸如层数、每层神经元数量以及激活函数的选择。
*   为潜在空间（瓶颈层）选择合适的维度。
*   选择合适的损失函数，例如均方误差（$MSE$）或二元交叉熵（$BCE$），并配置优化器（例如Adam, SGD），设定恰当的学习率。
*   监控训练阶段，然后使用训练好的模型从瓶颈层提取特征表示。
*   潜在空间的可视化方法，特别是针对低维表示。
一个动手实践环节将演示构建一个自动编码器，用于从表格数据集提取特征。

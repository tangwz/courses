---
course: "advanced-diffusion-architectures"
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-2-advanced-unet-architectures"
sourceId: 992
chapter: "advanced-unet-architectures"
title: "高级U-Net架构"
order: 2
description: "研究对标准U-Net的修改，包括注意力机制、条件信息结合和效率提升。"
hasQuiz: false
---

U-Net架构常作为扩散模型的骨干，通过其带跳跃连接的编码器-解码器结构，有效捕获空间层次。尽管有效，标准U-Net实现仍需修改，以应对扩散过程和复杂生成任务的特定要求。

本章审视针对扩散模型定制的U-Net架构改进。我们将分析注意力机制（具体而言是自注意力与交叉注意力）的结合，以提升特征表示能力并纳入条件信息。你将学习有效注入时间步嵌入 ($t$) 的方法，以及处理除简单类别标签之外的更高级条件输入。我们还将讨论旨在提升计算效率和训练稳定性的架构变体，包括不同的归一化技术，例如组归一化和自适应层归一化 (AdaLN)。在本章结束时，你将理解如何实现并分析这些复杂的U-Net变体，以构建更强大的扩散模型。

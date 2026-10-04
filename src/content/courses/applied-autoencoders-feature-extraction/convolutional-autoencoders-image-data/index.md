---
course: "applied-autoencoders-feature-extraction"
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-5-convolutional-autoencoders-image-data"
sourceId: 1156
chapter: "convolutional-autoencoders-image-data"
title: "图像数据的卷积自编码器"
order: 5
description: "应用卷积自编码器（CNNs），高效地从图像数据中提取有用的空间特征。"
hasQuiz: true
---

我们之前学习了使用全连接层构建的自编码器。尽管它们对多种数据类型都有用，但这些架构在处理图像数据时可能会遇到困难，因为图像数据维度高且具有固有的空间结构。

将图像展平为向量用于标准自编码器，通常会导致像素之间重要的局部关系丢失。

为解决图像数据的这些问题，本章侧重于卷积自编码器（ConvAEs）。这些模型结合了卷积神经网络（CNNs）的原理，通过保留空间信息和学习层次化特征，使其能有效处理图像。

在本章中，您将学习：
*   了解为什么全连接自编码器通常不适合处理图像数据。
*   在编码器中使用卷积层生成特征图，并使用池化层进行空间下采样。
*   在解码器中采用转置卷积层和其他上采样技术来重建图像。
*   设计并构建一个完整的卷积自编码器模型架构。
*   理解 ConvAEs 如何从图像中提取有意义的层次化特征。
*   在实际操作环节中，实现一个 ConvAE 用于从图像数据集中提取特征。

---
course: "applied-autoencoders-feature-extraction"
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-2-autoencoder-core-concepts"
sourceId: 1153
chapter: "autoencoder-core-concepts"
title: "理解自编码器：基本原理"
order: 2
description: "学习自编码器的基本结构，包括编码器、解码器、瓶颈层和重建损失。"
hasQuiz: true
---

基于我们之前讨论中对神经网络和降维技术的已有认识，本章将主要介绍自编码器。它们是一种特殊的神经网络结构，主要用于无监督地学习有效的数据编码。其主要思想是学习输入数据的压缩表示（即编码），然后从这个表示中尽可能准确地重建原始输入（即解码）。

本章将介绍自编码器的基本构成部分和运行原理。您将学到：
*   **编码器**：网络中将输入压缩成低维潜在空间表示的部分。
*   **瓶颈层**：包含压缩表示的层，通常被称为潜在空间或编码。
*   **解码器**：网络中从潜在空间表示重建输入的部分。

我们还将讲解自编码器的训练过程，并强调**重建损失函数**的作用。例如，对于连续的输入数据，一种常用的损失函数是均方误差（MSE），其定义如下：
$$MSE = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2$$
其中 $x_i$ 代表原始输入样本，$\hat{x}_i$ 是其重建版本。您还将了解到欠完备自编码器和过完备自编码器等内容，并开始明白这些网络如何从数据中发现有意义的特征。本章包含一个实践环节，您将在其中搭建一个基本的自编码器。

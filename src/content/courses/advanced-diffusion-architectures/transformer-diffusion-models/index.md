---
course: "advanced-diffusion-architectures"
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-3-transformer-diffusion-models"
sourceId: 994
chapter: "transformer-diffusion-models"
title: "基于Transformer的扩散模型"
order: 3
description: "了解Transformer架构如何适应扩散任务，替换或增强卷积骨干。"
hasQuiz: false
---

尽管基于卷积神经网络（CNN）的U-Net架构已成为许多成功扩散模型的标准骨干，但Transformer架构（最初在自然语言处理中占据主导地位）已在包括图像合成在内的生成任务中显示出良好的应用前景。Transformer擅长建模长距离依赖，与CNN固有的局部性偏置相比，它提供了一种捕获数据中关联的不同方法。

本章将研究如何在扩散模型框架内有效应用Transformer架构。我们将涵盖：

*   使用Transformer进行生成建模的动机，以及如何借鉴如Vision Transformer (ViT)等思路，使用补丁嵌入等技术使其适应图像数据。
*   Diffusion Transformers (DiTs) 的具体架构，其用Transformer块取代了U-Net骨干。
*   将类别标签或文本嵌入等条件信息融入DiT模型的方法。
*   对U-Net和Transformer骨干在性能、可扩展性和计算需求方面的权衡比较。
*   实现和训练基于Transformer的扩散模型的实际考量。

本章结束时，您将理解基于Transformer的扩散模型的结构和功能，并能够分析和实现它们。

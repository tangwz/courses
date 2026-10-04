---
course: "cnns-for-computer-vision"
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-5-attention-transformers-vision"
sourceId: 614
chapter: "attention-transformers-vision"
title: "视觉中的注意力机制与Transformer"
order: 5
description: "了解注意力机制在CNN中的应用，以及视觉Transformer (ViT) 的架构，用于计算机视觉任务。"
hasQuiz: false
---

卷积神经网络通过空间受限的感受野在学习局部模式方面表现出色，但要掌握图像中的整体信息和长距离依赖，则需要不同的方法。本章介绍注意力机制和Transformer架构，作为提升视觉模型捕获这些更广泛关联的方法。

您将学习自注意力机制如何与CNN框架结合，以使网络能够有选择地关注更具信息量的特征。我们将介绍具体例子，例如Squeeze-and-Excitation (SE) 模块和非局部网络。接下来，我们研究视觉Transformer (ViT)，这是一种通过处理图像块序列的方式，将成功的Transformer架构直接应用于图像数据的模型。我们将学习ViT的主要构成部分，包括图像块嵌入和多头自注意力层。最后，我们将讨论结合了卷积和Transformer元素的混合模型，并比较CNN和ViT的运行特点和数据需求。

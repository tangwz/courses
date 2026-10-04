---
course: "julia-deep-learning"
sourceUrl: "https://apxml.com/zh/courses/julia-deep-learning/chapter-3-constructing-nn-architectures"
sourceId: 1211
chapter: "constructing-nn-architectures"
title: "构建神经网络架构"
order: 3
description: "学习使用Julia中的Flux.jl搭建多种神经网络架构，如MLP、CNN和RNN。"
hasQuiz: false
---

在掌握了前一章Flux.jl的核心组成部分后，我们将着重介绍这些元素在构建多种神经网络架构中的具体运用。本章将指导您从零开始搭建几种常用的神经网络类型。

您将学到：

*   使用Julia工具准备和预处理专门用于深度学习模型的数据，包括利用`MLUtils.jl`高效处理数据，以进行批处理和迭代。
*   构建多层感知机（MLPs），它们是全连接网络的基本形式。
*   实现卷积神经网络（CNNs），它们对于图像处理等涉及网格状数据的任务必不可少。
*   开发循环神经网络（RNNs），包括长短期记忆（LSTM）单元，它们适用于文本或时间序列等序列数据。
*   运用嵌入层在网络中高效表示类别或文本数据。
*   通过创建自定义神经网络层来扩展Flux.jl的功能，以支持特定操作。
*   使用序列化技术（例如使用`BSON.jl`）保存您训练好的模型，并加载它们以供后续推断或继续训练。

本章最后是一个实践练习，您将在此应用这些技术来构建和配置一个CNN以解决图像分类问题，从而巩固您对这些架构组成部分如何组合在一起的理解。

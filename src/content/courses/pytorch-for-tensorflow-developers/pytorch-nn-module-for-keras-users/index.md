---
course: "pytorch-for-tensorflow-developers"
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-2-pytorch-nn-module-for-keras-users"
sourceId: 1044
chapter: "pytorch-nn-module-for-keras-users"
title: "搭建神经网络：从 Keras 到 torch.nn"
order: 2
description: "学习在 PyTorch 中使用 `torch.nn` 搭建神经网络，并对比 TensorFlow Keras。"
hasQuiz: true
---

如果你习惯使用 TensorFlow 的 Keras API 构建神经网络，本章将帮助你把这些技能应用到 PyTorch 的 `torch.nn` 模块。本章侧重于 PyTorch 中神经网络结构的搭建、配置和管理，并会持续与 Keras 的方法进行比较。

你将学习到：

*   将 Keras `Layer` 对象与其在 `torch.nn.Module`（PyTorch 中所有神经网络模块的基本类别）内对应的部分联系起来。
*   区分 Keras 用于模型定义的 Sequential 和 Functional API，以及 PyTorch 灵活的 `nn.Module` 子类化方法。
*   实现并理解常见的层类型，例如密集（线性）、卷积和循环层，同时注意它们在 PyTorch 和 TensorFlow 之间的区别与相似之处。
*   识别并使用标准激活函数，这些函数可通过 `torch.nn.functional` 或特定的 `nn.Module` 类别获取。
*   了解各种与 PyTorch 模型相关的权重初始化技术，并将其与常见的 TensorFlow 做法进行比较。
*   检查、访问和修改 PyTorch 模型中的参数和子模块，这对于开发和调试非常重要。

通过这些比较性说明和例子，你将获得在 PyTorch 框架中有效应用 Keras 模型构建知识的实际能力，从而理解通用之处以及 PyTorch 在网络设计方面的特定特性。

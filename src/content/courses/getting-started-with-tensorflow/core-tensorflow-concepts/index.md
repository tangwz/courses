---
course: "getting-started-with-tensorflow"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-2-core-tensorflow-concepts"
sourceId: 494
chapter: "core-tensorflow-concepts"
title: "TensorFlow核心要点"
order: 2
description: "了解 TensorFlow 的基本构建块：张量、变量、自动微分（AutoDiff）以及图执行。"
hasQuiz: true
---

TensorFlow环境配置完成后，本章将介绍用于构建和训练机器学习模型的必要基本构建块。我们将讲解主要数据结构——张量（Tensor），以及主要操作和像自动微分这样的要点。

具体来说，您将学会：
*   定义并理解 TensorFlow **张量**，包括它们的属性，例如形状（例如，$(batch\_size, features)$）和数据类型（$dtype$）。
*   执行张量的多种**操作**，从基本算术到矩阵运算。
*   使用 `tf.Variable` 来管理**可变状态**，对于保存训练期间会变化的模型参数非常重要。
*   使用**自动微分**通过 `tf.GradientTape` 计算梯度，这是模型优化背后的机制。
*   将 Python 函数转换为优化后的 TensorFlow 图，以提升性能，使用 `tf.function`。

掌握这些要点为有效使用 TensorFlow 的更高级别 API 提供了必要的知识储备，我们将在后续章节中介绍。

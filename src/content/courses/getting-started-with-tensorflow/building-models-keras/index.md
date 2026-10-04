---
course: "getting-started-with-tensorflow"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-3-building-models-keras"
sourceId: 497
chapter: "building-models-keras"
title: "使用 Keras 构建模型"
order: 3
description: "学习在 TensorFlow 中使用 Keras 顺序 (Sequential) 和函数式 (Functional) API 构建神经网络。"
hasQuiz: true
---

结合已掌握的 TensorFlow 张量和操作基本原理，我们现在将学习构建机器学习模型。TensorFlow 将 Keras 集成作为其高级应用程序编程接口 (API)，它简化了神经网络架构的定义过程。这种方法使得您可以将更多精力放在模型设计上，而不是低级操作细节。

在本章中，您将学习如何:

*   认识 Keras 在 TensorFlow 生态系统中的作用，以实现高效模型开发。
*   使用 Keras 顺序 (Sequential) API 构建由线性堆叠层组成的模型。
*   使用函数式 (Functional) API 定义更复杂的模型结构，包括多输入或多输出的模型。
*   识别并使用常见层类型（如 `Dense`、`Conv2D`、`Dropout`）和激活函数（如 ReLU、Sigmoid、Softmax）作为构建模块。
*   了解创建自定义层和模型以满足特定需求的基本方法。

最后，我们将应用所学知识来构建一个基本的分类模型，让您获得 Keras 工作流从定义到实例化的实践经验。

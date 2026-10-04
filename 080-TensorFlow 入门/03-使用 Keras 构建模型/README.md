# 第 3 章：使用 Keras 构建模型

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-3-building-models-keras)

[返回课程目录](../README.md)

结合已掌握的 TensorFlow 张量和操作基本原理，我们现在将学习构建机器学习模型。TensorFlow 将 Keras 集成作为其高级应用程序编程接口 (API)，它简化了神经网络架构的定义过程。这种方法使得您可以将更多精力放在模型设计上，而不是低级操作细节。

在本章中，您将学习如何:

*   认识 Keras 在 TensorFlow 生态系统中的作用，以实现高效模型开发。
*   使用 Keras 顺序 (Sequential) API 构建由线性堆叠层组成的模型。
*   使用函数式 (Functional) API 定义更复杂的模型结构，包括多输入或多输出的模型。
*   识别并使用常见层类型（如 `Dense`、`Conv2D`、`Dropout`）和激活函数（如 ReLU、Sigmoid、Softmax）作为构建模块。
*   了解创建自定义层和模型以满足特定需求的基本方法。

最后，我们将应用所学知识来构建一个基本的分类模型，让您获得 Keras 工作流从定义到实例化的实践经验。

## 小节

- 1. [Keras：TensorFlow 的高级 API](01-Keras%EF%BC%9ATensorFlow%20%E7%9A%84%E9%AB%98%E7%BA%A7%20API.md)
- 2. [Sequential 模型 API](02-Sequential%20%E6%A8%A1%E5%9E%8B%20API.md)
- 3. [Keras 常用层](03-Keras%20%E5%B8%B8%E7%94%A8%E5%B1%82.md)
- 4. [激活函数](04-%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0.md)
- 5. [复杂模型的函数式API](05-%E5%A4%8D%E6%9D%82%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%87%BD%E6%95%B0%E5%BC%8FAPI.md)
- 6. [自定义层和模型 (简介)](06-%E8%87%AA%E5%AE%9A%E4%B9%89%E5%B1%82%E5%92%8C%E6%A8%A1%E5%9E%8B%20%28%E7%AE%80%E4%BB%8B%29.md)
- 7. [动手实践：构建分类器](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E5%88%86%E7%B1%BB%E5%99%A8.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-3-building-models-keras/quiz)

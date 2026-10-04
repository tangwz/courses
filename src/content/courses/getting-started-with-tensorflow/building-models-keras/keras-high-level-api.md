---
course: "getting-started-with-tensorflow"
chapter: "building-models-keras"
lesson: "keras-high-level-api"
sourceId: 1707
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-3-building-models-keras/keras-high-level-api"
title: "Keras：TensorFlow 的高级 API"
description: "Keras 概览及其在简化 TensorFlow 模型构建中的作用。"
order: 1
plots: []
sourceHash: "d72063f24077fa4641c2fd983b48ea6f3ea5319c7391ad8af8c5cf4016cf8b95"
sourceCorrections: []
---

TensorFlow 使用 `tf.GradientTape` 为张量操作和自动微分提供了强大的工具。虽然可以直接使用这些基本操作构建复杂的模型，但手动跟踪变量、梯度和计算图会变得非常复杂且容易出错，尤其对于深度网络来说。

这就是 Keras 的作用所在。Keras 是一个用于构建和训练神经网络 (neural network)的高级 API，它与 TensorFlow 2.x 紧密结合。可以将 Keras 视为一个用户友好的界面，它位于 TensorFlow 核心功能之上。与直接使用底层 TensorFlow 操作相比，它让你能以少得多的代码和更低的思维负担来定义、训练和评估模型。

### 为什么要用 Keras？

Keras 的主要目标是让深度学习 (deep learning)的开发更快速、更简单。它通过以下几个设计原则做到了这一点：

1. **用户友好性：** Keras API 设计直观且一致。常见的神经网络 (neural network)组件，如层、激活函数 (activation function)、优化器和损失函数 (loss function)，都作为预构建、可配置的模块随时可用。这减少了你需要编写的样板代码量。
2. **模块化：** Keras 模型通常通过连接可配置的构建块（如层）来构建。这些组件可以轻松组合和复用，促进了有条理的模型开发。
3. **可扩展性：** 尽管 Keras 提供了许多标准组件，它也让创建用于研究或特殊应用的自定义层、损失函数和指标变得简单。我们将在本章稍后部分提到这一点。
4. **紧密的 TensorFlow 集成：** 由于 Keras 是 TensorFlow 官方的高级 API，它能与其他 TensorFlow 功能（如用于高效数据管道的 `tf.data` 和用于图优化的 `tf.function`）很好地配合使用。在需要时，你可以轻松地将 Keras 组件与底层 TensorFlow 代码混合使用。

### Keras：不仅仅是一个封装层

理解这一点很重要：Keras 不仅仅是 TensorFlow 的一个封装层；对于大多数用户而言，它*就是*在 TensorFlow *中*构建模型的标准方式。当你使用 `tensorflow.keras` 时，你就是在使用 TensorFlow。Keras 提供了抽象（如 `Layer`、`Model`、`Sequential`），这些抽象将你的模型定义转换为底层的 TensorFlow 计算图和操作。

可以这样看待它们的关系：

> Keras 在 TensorFlow 的核心操作之上提供了一个高级抽象层，简化了模型开发。

通过使用 Keras，你可以充分发挥 TensorFlow 的性能优化（如通过 `tf.function` 进行图执行）和硬件加速能力（CPU、GPU、TPU），在大多数情况下无需直接管理这些细节。

在接下来的部分中，我们将了解 Keras 定义模型的主要方式：即用于简单线性层堆叠的 Sequential API，以及用于构建具有多个输入、输出或共享层的更复杂架构的 Functional API。

## 参考资料

- [What is Keras?](https://keras.io/about/) — Keras team (2024)
  阐述了 Keras API 的设计理念和目标，包括用户友好性和模块化。
- [Keras guides and tutorials](https://www.tensorflow.org/guide/keras) — TensorFlow team (2024)
  官方 TensorFlow 指南，说明如何使用 Keras API 构建和训练模型，并详细介绍其集成方式。
- [Deep Learning with Python, Second Edition](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  Keras 创建者撰写的权威书籍，全面介绍了 Keras 的原理和应用。

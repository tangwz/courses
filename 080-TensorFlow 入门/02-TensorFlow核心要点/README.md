# 第 2 章：TensorFlow核心要点

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-2-core-tensorflow-concepts)

[返回课程目录](../README.md)

TensorFlow环境配置完成后，本章将介绍用于构建和训练机器学习模型的必要基本构建块。我们将讲解主要数据结构——张量（Tensor），以及主要操作和像自动微分这样的要点。

具体来说，您将学会：
*   定义并理解 TensorFlow **张量**，包括它们的属性，例如形状（例如，$(batch\_size, features)$）和数据类型（$dtype$）。
*   执行张量的多种**操作**，从基本算术到矩阵运算。
*   使用 `tf.Variable` 来管理**可变状态**，对于保存训练期间会变化的模型参数非常重要。
*   使用**自动微分**通过 `tf.GradientTape` 计算梯度，这是模型优化背后的机制。
*   将 Python 函数转换为优化后的 TensorFlow 图，以提升性能，使用 `tf.function`。

掌握这些要点为有效使用 TensorFlow 的更高级别 API 提供了必要的知识储备，我们将在后续章节中介绍。

## 小节

- 1. [理解张量](01-%E7%90%86%E8%A7%A3%E5%BC%A0%E9%87%8F.md)
- 2. [张量运算](02-%E5%BC%A0%E9%87%8F%E8%BF%90%E7%AE%97.md)
- 3. [TensorFlow 中的变量](03-TensorFlow%20%E4%B8%AD%E7%9A%84%E5%8F%98%E9%87%8F.md)
- 4. [使用 GradientTape 进行自动微分](04-%E4%BD%BF%E7%94%A8%20GradientTape%20%E8%BF%9B%E8%A1%8C%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86.md)
- 5. [tf.function 介绍](05-tf.function%20%E4%BB%8B%E7%BB%8D.md)
- 6. [练习：张量操作与梯度](06-%E7%BB%83%E4%B9%A0%EF%BC%9A%E5%BC%A0%E9%87%8F%E6%93%8D%E4%BD%9C%E4%B8%8E%E6%A2%AF%E5%BA%A6.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-2-core-tensorflow-concepts/quiz)

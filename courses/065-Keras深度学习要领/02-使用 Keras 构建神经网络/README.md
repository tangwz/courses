# 第 2 章：使用 Keras 构建神经网络

来源：[原章节](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-2-building-networks-keras)

[返回课程目录](../README.md)

在回顾了神经网络的基本原理并准备好环境之后，本章将进入使用 Keras 实际构建模型的环节。我们将介绍定义网络结构的核心组成部分和方法。

您将使用 `Sequential` API 来构建线性堆叠层，以及 `Functional` API 来建立更精细的模型图。诸如 `Dense` 层、各种激活函数 ($ReLU$, $Sigmoid$, $Softmax$) 以及指定输入形状等主要内容都将得到说明和实现。用于检查和可视化模型结构的方法也将进行介绍。本章最后将通过动手实践来巩固这些方法，即使用 Keras 构建您的第一个神经网络。

## 小节

- 1. [顺序式API](01-%E9%A1%BA%E5%BA%8F%E5%BC%8FAPI.md)
- 2. [常见层类型：全连接层](02-%E5%B8%B8%E8%A7%81%E5%B1%82%E7%B1%BB%E5%9E%8B%EF%BC%9A%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%B1%82.md)
- 3. [激活函数](03-%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0.md)
- 4. [指定输入形状](04-%E6%8C%87%E5%AE%9A%E8%BE%93%E5%85%A5%E5%BD%A2%E7%8A%B6.md)
- 5. [函数式API](05-%E5%87%BD%E6%95%B0%E5%BC%8FAPI.md)
- 6. [模型概览与可视化](06-%E6%A8%A1%E5%9E%8B%E6%A6%82%E8%A7%88%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96.md)
- 7. [实践：使用 Keras 构建你的第一个网络](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%20Keras%20%E6%9E%84%E5%BB%BA%E4%BD%A0%E7%9A%84%E7%AC%AC%E4%B8%80%E4%B8%AA%E7%BD%91%E7%BB%9C.md)

章节测验：[在线测验](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-2-building-networks-keras/quiz)

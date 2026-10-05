# 第 2 章：激活函数与网络结构

来源：[原章节](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-2-activation-functions-architecture)

[返回课程目录](../README.md)

在上一章中，我们确立了多层感知机（MLP）的思想，作为克服单层模型局限性的一种方法。本章将侧重于决定这些网络如何运作和学习的内部机制和结构设计选择。

您将了解激活函数，它们是神经元内部的非线性组成部分，使多层感知机能够模拟复杂的关系。我们将介绍Sigmoid、Tanh和ReLU（$f(x) = \max(0, x)$）等标准函数，考察它们的数学特性、优点和缺点。理解这些函数对于控制网络中信息和梯度的流动非常重要。

此外，我们将考察前馈网络的结构，阐明输入层、隐藏层和输出层的作用。我们将讨论设计网络结构时的考量因素，例如选择层数和每层单元数，为构建有效的模型打下基础。实际例子将演示如何实现和比较不同的激活函数。

## 小节

- 1. [激活函数的作用](01-%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E7%9A%84%E4%BD%9C%E7%94%A8.md)
- 2. [Sigmoid 激活函数](02-Sigmoid%20%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0.md)
- 3. [双曲正切（Tanh）激活函数](03-%E5%8F%8C%E6%9B%B2%E6%AD%A3%E5%88%87%EF%BC%88Tanh%EF%BC%89%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0.md)
- 4. [修正线性单元 (ReLU)](04-%E4%BF%AE%E6%AD%A3%E7%BA%BF%E6%80%A7%E5%8D%95%E5%85%83%20%28ReLU%29.md)
- 5. [ReLU的多种形式 (Leaky ReLU, PReLU, ELU)](05-ReLU%E7%9A%84%E5%A4%9A%E7%A7%8D%E5%BD%A2%E5%BC%8F%20%28Leaky%20ReLU%2C%20PReLU%2C%20ELU%29.md)
- 6. [选择合适的激活函数](06-%E9%80%89%E6%8B%A9%E5%90%88%E9%80%82%E7%9A%84%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0.md)
- 7. [理解网络层：输入层、隐藏层、输出层](07-%E7%90%86%E8%A7%A3%E7%BD%91%E7%BB%9C%E5%B1%82%EF%BC%9A%E8%BE%93%E5%85%A5%E5%B1%82%E3%80%81%E9%9A%90%E8%97%8F%E5%B1%82%E3%80%81%E8%BE%93%E5%87%BA%E5%B1%82.md)
- 8. [设计前馈网络架构](08-%E8%AE%BE%E8%AE%A1%E5%89%8D%E9%A6%88%E7%BD%91%E7%BB%9C%E6%9E%B6%E6%9E%84.md)
- 9. [动手实践：实现不同的激活函数](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E4%B8%8D%E5%90%8C%E7%9A%84%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-2-activation-functions-architecture/quiz)

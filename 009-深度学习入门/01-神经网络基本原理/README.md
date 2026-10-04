# 第 1 章：神经网络基本原理

来源：[原章节](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-1-neural-network-foundations)

[返回课程目录](../README.md)

本章为理解人工神经网络提供基本知识。我们首先将深度学习与传统机器学习技术进行对比，指出主要差异。

您将研究生物神经元作为人工模型的思想来源，然后定义人工神经元的数学组成：输入、权重、偏置、求和函数和激活步骤。我们将分析感知机（神经网络的最早形式），了解其功能，并讨论其局限性，尤其是在处理像异或（XOR）问题这样的非线性可分离数据时。

这引出了多层感知机（MLP）的介绍，说明了增加隐藏层如何提升模型的复杂度和表达能力。本章以一个实践练习作结，您将在其中使用Python实现一个简单的感知机模型。

完成本章学习后，您将掌握历史背景以及构建更复杂深度学习架构的核心组成部分。

## 小节

- 1. [从机器学习到深度学习](01-%E4%BB%8E%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%88%B0%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0.md)
- 2. [生物学上的启发：神经元](02-%E7%94%9F%E7%89%A9%E5%AD%A6%E4%B8%8A%E7%9A%84%E5%90%AF%E5%8F%91%EF%BC%9A%E7%A5%9E%E7%BB%8F%E5%85%83.md)
- 3. [人工神经元：数学模型](03-%E4%BA%BA%E5%B7%A5%E7%A5%9E%E7%BB%8F%E5%85%83%EF%BC%9A%E6%95%B0%E5%AD%A6%E6%A8%A1%E5%9E%8B.md)
- 4. [感知机：最简单的神经网络](04-%E6%84%9F%E7%9F%A5%E6%9C%BA%EF%BC%9A%E6%9C%80%E7%AE%80%E5%8D%95%E7%9A%84%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md)
- 5. [单层感知器的局限性](05-%E5%8D%95%E5%B1%82%E6%84%9F%E7%9F%A5%E5%99%A8%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)
- 6. [多层感知机 (MLP)：添加层次](06-%E5%A4%9A%E5%B1%82%E6%84%9F%E7%9F%A5%E6%9C%BA%20%28MLP%29%EF%BC%9A%E6%B7%BB%E5%8A%A0%E5%B1%82%E6%AC%A1.md)
- 7. [动手实践：构建一个简单的感知器模型](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84%E6%84%9F%E7%9F%A5%E5%99%A8%E6%A8%A1%E5%9E%8B.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-1-neural-network-foundations/quiz)

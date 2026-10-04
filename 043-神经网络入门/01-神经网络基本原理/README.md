# 第 1 章：神经网络基本原理

来源：[原章节](https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-1-neural-network-foundations)

[返回课程目录](../README.md)

本章将介绍神经网络的**基本思想**。我们首先分析人工神经元的结构和功能，它是最基本的处理单元。您将学习输入如何通过权重和偏置进行处理，这些是可学习的参数，在诸如 $z = \sum(weight \times input) + bias$ 的计算中得以体现。

接着，我们将介绍Sigmoid、Tanh和ReLU等激活函数，解释它们在引入非线性（$a = f(z)$）中的作用，这对学习复杂模式是必需的。最后，我们将看到这些单个神经元如何被组织成层（输入层、隐藏层和输出层），并连接起来形成前馈神经网络的基本架构。到本章结束时，您将理解主要组成部分以及它们如何配合处理信息。

## 小节

- 1. [从生物神经元到人工神经元](01-%E4%BB%8E%E7%94%9F%E7%89%A9%E7%A5%9E%E7%BB%8F%E5%85%83%E5%88%B0%E4%BA%BA%E5%B7%A5%E7%A5%9E%E7%BB%8F%E5%85%83.md)
- 2. [权重和偏置：网络的参数](02-%E6%9D%83%E9%87%8D%E5%92%8C%E5%81%8F%E7%BD%AE%EF%BC%9A%E7%BD%91%E7%BB%9C%E7%9A%84%E5%8F%82%E6%95%B0.md)
- 3. [激活函数：引入非线性](03-%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%EF%BC%9A%E5%BC%95%E5%85%A5%E9%9D%9E%E7%BA%BF%E6%80%A7.md)
- 4. [网络结构：层与连接](04-%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%EF%BC%9A%E5%B1%82%E4%B8%8E%E8%BF%9E%E6%8E%A5.md)
- 5. [一个简单的前馈网络示例](05-%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84%E5%89%8D%E9%A6%88%E7%BD%91%E7%BB%9C%E7%A4%BA%E4%BE%8B.md)
- 6. [实践：计算神经元输出](06-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%A1%E7%AE%97%E7%A5%9E%E7%BB%8F%E5%85%83%E8%BE%93%E5%87%BA.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-1-neural-network-foundations/quiz)

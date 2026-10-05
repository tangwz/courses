# 第 5 章：构建与训练简单神经网络

来源：[原章节](https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-5-building-first-neural-network)

[返回课程目录](../README.md)

在掌握了神经网络组成部分、数据准备、前向传播以及反向传播和梯度下降这些学习方法的基本内容后，我们现在将这些元素组合起来。本章将着重介绍从头到尾实际构建和训练简单神经网络的过程。

您将学习如何：

*   定义网络的结构，指定层和激活函数。
*   应用合适的方法初始化网络参数（权重 $W$ 和偏置 $b$）。
*   组织并实现迭代训练循环，以周期（epochs）和批次（batches）处理数据。
*   结合前向传播、损失计算（例如，使用均方误差或交叉熵）、反向传播以求得梯度，如 $\frac{\partial L}{\partial W}$，以及使用梯度下降进行参数更新。
*   监控重要指标，例如损失和准确率，以评估训练进展。
*   简要了解深度学习库如何帮助构建这些模型。

本章最后将通过一个实践练习，您将应用这些步骤使用 Python 构建并训练一个简单的神经网络分类器。

## 小节

- 1. [搭建网络结构](01-%E6%90%AD%E5%BB%BA%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84.md)
- 2. [初始化权重与偏置](02-%E5%88%9D%E5%A7%8B%E5%8C%96%E6%9D%83%E9%87%8D%E4%B8%8E%E5%81%8F%E7%BD%AE.md)
- 3. [训练循环的结构](03-%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF%E7%9A%84%E7%BB%93%E6%9E%84.md)
- 4. [实现训练步骤](04-%E5%AE%9E%E7%8E%B0%E8%AE%AD%E7%BB%83%E6%AD%A5%E9%AA%A4.md)
- 5. [监控训练进程](05-%E7%9B%91%E6%8E%A7%E8%AE%AD%E7%BB%83%E8%BF%9B%E7%A8%8B.md)
- 6. [深度学习框架（TensorFlow/PyTorch）简介](06-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6%EF%BC%88TensorFlow-PyTorch%EF%BC%89%E7%AE%80%E4%BB%8B.md)
- 7. [动手实践：训练一个简单分类器](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%AD%E7%BB%83%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E5%88%86%E7%B1%BB%E5%99%A8.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-5-building-first-neural-network/quiz)

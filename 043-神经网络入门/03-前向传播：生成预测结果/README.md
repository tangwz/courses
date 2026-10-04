# 第 3 章：前向传播：生成预测结果

来源：[原章节](https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-3-forward-propagation)

[返回课程目录](../README.md)

在确定了神经网络的结构并准备好输入数据后，我们现在来看看网络如何处理信息以得到一个结果。这一过程被称为**前向传播**（或前向运算），它涉及将输入数据逐层通过网络，计算出一个输出。

在本章中，你将了解到这种信息流动的机制。我们将涵盖以下内容：

*   计算神经元内的加权和 ($z = \sum_{i} w_i x_i + b$)。
*   应用激活函数 ($a = g(z)$) 来加入非线性。
*   使用矩阵运算 ($Z = WX + b$, $A = g(Z)$) 在各层高效地执行这些计算。
*   追踪数据从输入层到最终输出层以生成预测结果的路径。

我们将看到这些步骤如何结合起来将输入特征转变为有意义的网络输出。本章结束时，你将能够使用 Python 和 NumPy 实现一个简单神经网络的完整前向运算。

## 小节

- 1. [网络中的信息流动](01-%E7%BD%91%E7%BB%9C%E4%B8%AD%E7%9A%84%E4%BF%A1%E6%81%AF%E6%B5%81%E5%8A%A8.md)
- 2. [线性变换：加权和计算](02-%E7%BA%BF%E6%80%A7%E5%8F%98%E6%8D%A2%EF%BC%9A%E5%8A%A0%E6%9D%83%E5%92%8C%E8%AE%A1%E7%AE%97.md)
- 3. [逐层应用激活函数](03-%E9%80%90%E5%B1%82%E5%BA%94%E7%94%A8%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0.md)
- 4. [高效计算的矩阵运算](04-%E9%AB%98%E6%95%88%E8%AE%A1%E7%AE%97%E7%9A%84%E7%9F%A9%E9%98%B5%E8%BF%90%E7%AE%97.md)
- 5. [计算最终输出预测](05-%E8%AE%A1%E7%AE%97%E6%9C%80%E7%BB%88%E8%BE%93%E5%87%BA%E9%A2%84%E6%B5%8B.md)
- 6. [动手实践：实现前向传播](06-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-3-forward-propagation/quiz)

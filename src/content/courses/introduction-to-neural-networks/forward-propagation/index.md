---
course: "introduction-to-neural-networks"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-3-forward-propagation"
sourceId: 530
chapter: "forward-propagation"
title: "前向传播：生成预测结果"
order: 3
description: "了解神经网络如何处理输入数据以生成预测结果。"
hasQuiz: true
---

在确定了神经网络的结构并准备好输入数据后，我们现在来看看网络如何处理信息以得到一个结果。这一过程被称为**前向传播**（或前向运算），它涉及将输入数据逐层通过网络，计算出一个输出。

在本章中，你将了解到这种信息流动的机制。我们将涵盖以下内容：

*   计算神经元内的加权和 ($z = \sum_{i} w_i x_i + b$)。
*   应用激活函数 ($a = g(z)$) 来加入非线性。
*   使用矩阵运算 ($Z = WX + b$, $A = g(Z)$) 在各层高效地执行这些计算。
*   追踪数据从输入层到最终输出层以生成预测结果的路径。

我们将看到这些步骤如何结合起来将输入特征转变为有意义的网络输出。本章结束时，你将能够使用 Python 和 NumPy 实现一个简单神经网络的完整前向运算。

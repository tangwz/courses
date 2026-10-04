---
course: "introduction-to-neural-networks"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-5-building-first-neural-network"
sourceId: 536
chapter: "building-first-neural-network"
title: "构建与训练简单神经网络"
order: 5
description: "结合前向传播、反向传播和梯度下降，以构建和训练一个简单的网络。"
hasQuiz: true
---

在掌握了神经网络组成部分、数据准备、前向传播以及反向传播和梯度下降这些学习方法的基本内容后，我们现在将这些元素组合起来。本章将着重介绍从头到尾实际构建和训练简单神经网络的过程。

您将学习如何：

*   定义网络的结构，指定层和激活函数。
*   应用合适的方法初始化网络参数（权重 $W$ 和偏置 $b$）。
*   组织并实现迭代训练循环，以周期（epochs）和批次（batches）处理数据。
*   结合前向传播、损失计算（例如，使用均方误差或交叉熵）、反向传播以求得梯度，如 $\frac{\partial L}{\partial W}$，以及使用梯度下降进行参数更新。
*   监控重要指标，例如损失和准确率，以评估训练进展。
*   简要了解深度学习库如何帮助构建这些模型。

本章最后将通过一个实践练习，您将应用这些步骤使用 Python 构建并训练一个简单的神经网络分类器。

---
course: "intro-normalizing-flows"
sourceUrl: "https://apxml.com/zh/courses/intro-normalizing-flows/chapter-2-building-normalizing-flows"
sourceId: 1449
chapter: "building-normalizing-flows"
title: "构建标准化流"
order: 2
description: "通过堆叠可逆变换并实现平面流来构建基础标准化流。"
hasQuiz: false
---

在上一章中，我们利用变量代换定理和雅可比行列式，确定了通过可逆函数映射随机变量的数学规则。现在，我们将应用这些数学原理来构建实用的生成模型。

单一的变换通常不足以模拟高度复杂的概率分布。为了构建具有高度灵活性的模型，我们将多个简单的可逆变换按顺序堆叠。如果我们从简单基础分布中的样本 $z_0$ 开始，可以应用一系列共 $K$ 次变换来生成最终样本 $z_K$：

$$z_K = f_K \circ f_{K-1} \circ \dots \circ f_1(z_0)$$

这个复合函数构成了标准化流的基础。通过定义合适的基础分布（例如各向同性高斯分布），我们可以让样本通过这一系列变换，从而逼近高度复杂的数据分布。

本章重点介绍组装这些模型的机制。你将学习标准化流的正式定义，并了解如何为任务选择合适的基础分布。我们将分析平面流（planar flows）和径向流（radial flows）等具体架构，仔细观察它们的数学公式以及缩放方式。

最后，你将从理论转向应用。你将编写 PyTorch 代码，从零开始实现一个平面流层，并训练它来映射简单的二维数据集。在本模块结束时，你将学会如何将数学定义转化为 Python 中可运行的生成模型。

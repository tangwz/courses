---
course: "calculus-essentials-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/calculus-essentials-machine-learning/chapter-5-chain-rule-backpropagation"
sourceId: 456
chapter: "chain-rule-backpropagation"
title: "链式法则与反向传播"
order: 5
description: "理解源于微积分的链式法则，以及它如何构成训练神经网络中反向传播算法的数学原理。"
hasQuiz: true
---

你已经学习了如何计算函数的导数和梯度。机器学习模型，尤其是神经网络，通常包含函数相互嵌套的复杂结构。计算整个模型误差相对于其内部参数的梯度，需要系统地处理这些嵌套关系。

本章将介绍**链式法则**，它是微积分中用于复合函数求导的一个核心原理。我们将从单变量链式法则开始，并将其推广到多变量函数。你将看到神经网络如何能以数学方式表示为函数的组合，并理解链式法则如何构成了**反向传播**算法的原理。反向传播是神经网络训练过程中，高效计算更新权重和偏置所需梯度的标准方法。我们还将介绍计算图，以帮助观察计算流程和梯度计算。完成本章后，你将掌握深度学习模型中梯度计算的机制。

---
course: "optimization-techniques-ml"
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-3-adaptive-learning-rate-algorithms"
sourceId: 436
chapter: "adaptive-learning-rate-algorithms"
title: "自适应学习率算法"
order: 3
description: "学习AdaGrad、RMSprop、Adam等自适应学习率方法及其变体，实现更快的训练。"
hasQuiz: false
---

标准梯度下降法常使用固定的学习率，$\eta$。找到一个好的$\eta$需要仔细调整；过小的学习率会减慢收敛，而过大的学习率则可能阻碍收敛。本章介绍的算法通过在训练期间自动调整学习率来解决这个问题。

我们将考察几种常用的自适应学习率方法。您将学习AdaGrad的原理，它根据历史梯度调整学习率，以及RMSprop，它改进了AdaGrad的方法以避免过度衰减。接着，我们将学习Adam（自适应矩估计），一个结合了自适应学习率和动量估计的优化器。我们还将分析Adamax、Nadam和AMSGrad等变体，了解它们的具体改进之处。本章还会介绍如何将自适应方法与学习率调度结合使用。

学完本章，您将理解这些自适应技术的理论和实际应用，从而能够选择并运用它们来更高效地训练模型。

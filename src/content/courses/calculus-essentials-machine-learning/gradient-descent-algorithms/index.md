---
course: "calculus-essentials-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/calculus-essentials-machine-learning/chapter-4-gradient-descent-algorithms"
sourceId: 453
chapter: "gradient-descent-algorithms"
title: "梯度下降算法"
order: 4
description: "学习梯度下降及其变体（批量、随机、Mini-batch）的工作原理，以优化机器学习模型。"
hasQuiz: true
---

基于上一章对梯度的理解，我们现在关注如何使用它们进行模型优化。本章介绍梯度下降，这是一种迭代算法，用于寻找函数的最小值，在机器学习中通常是成本函数$J(\theta)$。

您将掌握其原理以及通过沿着梯度$\nabla J(\theta)$相反的方向移动来更新模型参数的具体步骤。我们将考察学习率$\alpha$的作用，比较批量梯度下降、随机梯度下降和Mini-batch梯度下降等不同方法，并讨论常见的优化问题，例如局部最小值。本章包含一个关于实现基本算法的实践部分。

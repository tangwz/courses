---
course: "optimization-techniques-ml"
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-2-second-order-optimization-methods"
sourceId: 434
chapter: "second-order-optimization-methods"
title: "二阶优化方法"
order: 2
description: "了解牛顿法、拟牛顿法（如BFGS和L-BFGS）以及信赖域方法。"
hasQuiz: false
---

在一阶梯度方法的基础上，本章介绍使用二阶导数信息的优化技术。这些方法考虑损失函数的曲率，通常比梯度下降收敛更快，尤其是在接近最小值时。

我们将从牛顿法开始，了解其理论原理，即使用二次模型在局部近似目标函数：
$$
f(x_k + p) \approx f(x_k) + \nabla f(x_k)^T p + \frac{1}{2} p^T \nabla^2 f(x_k) p
$$
这里的重要组成是海森矩阵 $\nabla^2 f(x_k)$。我们将讨论其性质，以及计算和求逆所面临的计算难点，尤其是在高维机器学习问题中。

为了解决这些难点，我们将主要介绍拟牛顿法，这些方法近似海森矩阵或其逆。你将学习流行的BFGS算法及其节省内存的变体L-BFGS，它们在实际中广泛使用。此外，我们还将介绍信赖域方法，它提供了一种使用曲率信息管理步长的替代策略。本章包含一个关于L-BFGS实现的实践部分。

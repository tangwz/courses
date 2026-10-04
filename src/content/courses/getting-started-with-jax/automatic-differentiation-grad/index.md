---
course: "getting-started-with-jax"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-3-automatic-differentiation-grad"
sourceId: 499
chapter: "automatic-differentiation-grad"
title: "使用 `grad` 进行自动微分"
order: 3
description: "学习使用 `jax.grad` 进行自动微分。计算标量函数的梯度以用于优化。"
hasQuiz: true
---

许多计算任务，特别是在机器学习模型训练中，需要计算函数的梯度。自动微分提供了一种高效的方法来计算这些导数。本章将介绍 `jax.grad`，它是JAX的核心变换，用于从处理数值输入的Python代码中获取梯度函数。

您将学习如何：
*   应用 `jax.grad` 计算标量值函数 $f$ 的梯度 $∇f(x)$。
*   理解 `grad` 所使用的反向模式自动微分的基本原理。
*   控制对特定函数参数的微分。
*   通过组合 `grad` 计算高阶导数。
*   使用 `jax.value_and_grad` 高效地同时获取函数的输出值及其梯度。
*   了解控制流如何与微分关联，并识别潜在的局限性。

到本章结束时，您将能够有效使用 `jax.grad` 在JAX框架内对您的数值函数进行求导。

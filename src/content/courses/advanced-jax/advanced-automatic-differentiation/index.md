---
course: "advanced-jax"
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-4-advanced-automatic-differentiation"
sourceId: 650
chapter: "advanced-automatic-differentiation"
title: "进阶自动微分方法"
order: 4
description: "了解JAX的自动微分系统。实现自定义VJP/JVP，处理高阶导数，并通过控制流进行微分。"
hasQuiz: false
---

自动微分是JAX的核心组成部分，它使得机器学习模型能够进行基于梯度的优化。虽然`jax.grad`提供了获取梯度的便捷接口，但了解并控制其内部运作将为复杂任务带来更大的灵活性和更高的效率。

本章将基于`jax.grad`的初步知识进行拓展。你将学习自动微分的两种基本模式：正向模式（雅可比向量积，JVP）和反向模式（向量雅可比积，VJP）。我们将介绍如何直接使用`jax.jvp`和`jax.vjp`计算它们，这些构成了`jax.grad`及其他变换的构建块。

主要内容包括：

*   计算通过组合微分变换得到的高阶导数。
*   高效计算完整雅可比矩阵和海森矩阵的方法。
*   使用`jax.custom_vjp`和`jax.custom_jvp`为函数定义自定义微分规则，这可用于提高数值稳定性、优化性能或处理非JAX代码。
*   理解自动微分在`lax.scan`、`lax.cond`和`lax.while_loop`等控制流原语中如何运作。
*   处理不可微分函数或需要明确停止梯度传播的函数的方法，例如使用`jax.lax.stop_gradient`。

在本章结束时，你将对JAX的自动微分系统有更全面的理解，并掌握在进阶场景中有效应用它的工具。

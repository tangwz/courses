---
course: "advanced-jax"
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-1-advanced-jax-transformations-control-flow"
sourceId: 640
chapter: "advanced-jax-transformations-control-flow"
title: "高级 JAX 变换与控制流"
order: 1
description: "掌握 JAX 更高阶的变换。理解 `lax.scan`、`cond`、`while_loop`、遮蔽以及用于复杂模型的自定义导数。"
hasQuiz: false
---

你已经使用过 JAX 的核心变换：`jit` 用于编译，`grad` 用于微分，以及 `vmap` 用于向量化。尽管这些是核心要素，但许多精密的模型需要更复杂的计算结构，例如循环步骤、条件执行或动态循环。

本章介绍 JAX 的函数式控制流原语，它们使你能够在 JAX 可追踪和可编译的框架内表达这些复杂的模式。我们将探讨以下内容：

*   使用 `lax.scan` 有效实现序列操作，例如循环神经网络 (RNN) 中的操作。
*   使用 `lax.cond` 在编译函数内部实现条件逻辑。
*   使用 `lax.while_loop` 处理动态迭代。
*   理解这些控制流操作如何与 `jit`、`grad` 和 `vmap` 配合。
*   应用遮蔽技术进行选择性计算，这通常是可变长度数据所必需的。
*   JAX 在追踪过程中如何处理涉及闭包的 Python 代码。

学完本章，你将能够构建和分析包含循环、条件和序列依赖的 JAX 函数，为你构建更复杂的模型和算法做好准备。

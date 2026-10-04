---
course: "getting-started-with-jax"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-6-managing-state-in-jax"
sourceId: 509
chapter: "managing-state-in-jax"
title: "JAX 中的状态管理"
order: 6
description: "学习在 JAX 内处理状态（如模型参数或优化器状态）的函数式编程方法。"
hasQuiz: true
---

JAX 的设计以纯函数和 `jit`、`grad` 等变换为核心，在处理本身包含状态的计算时，需要特定的方法。常见的 Python 模式通常依赖于原地修改对象，这与 JAX 的函数式特性不符，尤其是在代码需要编译或求导时。在这些限制下，我们如何更新训练中的模型参数、管理优化器状态或处理循环计算呢？

本章介绍在 JAX 中管理状态的函数式编程模式。你将了解函数纯净的原理，以及它为何对 JAX 变换如此重要。我们将讲解处理状态的主要方法：将状态显式地作为函数的输入，并接收更新后的状态作为输出。你将看到 JAX 的 PyTree 工具如何简化处理复杂、嵌套的状态结构（例如字典或参数列表）。我们将通过实际例子应用这些思路，例如实现一个有状态计数器，并管理简单优化算法所需的状态，确保这些模式与 `jit`、`grad` 和其他变换结合使用时能正确运行。

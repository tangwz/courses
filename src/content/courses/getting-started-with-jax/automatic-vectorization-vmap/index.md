---
course: "getting-started-with-jax"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-4-automatic-vectorization-vmap"
sourceId: 502
chapter: "automatic-vectorization-vmap"
title: "使用 `vmap` 实现自动向量化"
order: 4
description: "学习使用 `jax.vmap` 实现自动向量化，让批处理更简便，无需手动编写循环。"
hasQuiz: true
---

在数值计算和机器学习中，您经常需要同时对多个数据点应用同一个函数，这个过程通常被称为批处理。尽管您可以编写显式循环或依赖于为批处理操作设计的函数，但这些方法有时会很冗长，或者需要手动仔细管理数组维度。

JAX 提供了一个转换，`jax.vmap`，专门用于自动向量化。它允许您将一个为处理单个数据点而编写的函数，有效地应用到整个批次（或多个批次）的数据上，通常无需重写原函数的逻辑。`vmap` 相当于自动为您的计算添加了一个“批次维度”。

在这一章，您将学习：

*   向量化的含义以及它提升性能的原因。
*   如何使用 `jax.vmap` 对处理单个或多个参数的函数进行向量化。
*   如何使用 `in_axes` 和 `out_axes` 参数控制哪些轴被映射。
*   如何通过嵌套调用 `vmap` 来应对更复杂的情形。
*   `vmap` 如何与 `jit` 和 `grad` 等 JAX 的其他转换配合。
*   使用 `vmap` 的一些重要事项。

学完本章后，您将能够使用 `vmap` 来让 JAX 中的批处理代码更简洁，并通常运行得更快。

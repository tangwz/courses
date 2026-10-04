---
course: "getting-started-with-jax"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-2-accelerating-functions-jit"
sourceId: 496
chapter: "accelerating-functions-jit"
title: "通过 JIT 编译加速函数"
order: 2
description: "学习如何使用 `jax.jit` 通过即时编译来加速你的 Python 函数。"
hasQuiz: true
---

尽管 JAX 提供了一个熟悉的 NumPy 风格的 API，但对于科学计算和机器学习中常见的密集数值计算，标准的 Python 执行速度可能较慢，尤其是在针对 GPU 或 TPU 等加速器时。为了获得高效率，JAX 依赖于编译。

本章主要讲解 `jax.jit`，这是一个即时 (JIT) 编译转换。您将了解 `jit` 如何将 Python 函数转换成针对您的硬件进行适配的优化后的可执行代码。我们将介绍：

*   `jax.jit` 作为装饰器或函数的基本使用。
*   即时 (JIT) 编译如何通过一个称为“追踪”的过程来工作。
*   追踪对 Python 控制流（循环、条件语句）的影响。
*   在编译函数中如何处理静态值与动态值。
*   使用 `jit` 时的常见错误和注意事项。

在本章结束时，您将能够使用 `jax.jit` 大幅提升您的 JAX 计算速度。

---
course: "advanced-jax"
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-2-optimizing-jax-code-performance"
sourceId: 644
chapter: "optimizing-jax-code-performance"
title: "JAX 代码的性能优化"
order: 2
description: "优化 JAX 代码，提升运行速度。分析 GPU/TPU 的执行情况，理解内存布局，XLA 编译，以及高级 JIT 策略。"
hasQuiz: false
---

JAX 提供了强大的抽象方式，例如 `jit`，能够大幅加速你的 Python 和 NumPy 代码，尤其是在硬件加速器上。然而，要达到最佳性能，往往需要审视其内部机制。仅仅应用 `@jit` 并不能保证达到最高速度。

本章主要介绍诊断性能瓶颈的方法，以及优化 JAX 程序以适用于 GPU 和 TPU。我们将介绍如何分析 JAX 执行情况，理解跟踪过程中生成的中间 `jaxpr` 表示，了解 XLA 编译器在优化中的职能，考虑内存布局的影响，尽量减少代价高昂的重新编译，识别算子融合，并使用 JAX 的异步调度机制正确地进行代码基准测试。完成本章后，你将能够有条理地分析并提升你的 JAX 计算的执行速度。

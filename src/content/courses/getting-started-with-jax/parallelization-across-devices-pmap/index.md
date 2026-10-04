---
course: "getting-started-with-jax"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-5-parallelization-across-devices-pmap"
sourceId: 506
chapter: "parallelization-across-devices-pmap"
title: "使用 `pmap` 在多设备上并行计算"
order: 5
description: "学习如何使用 `jax.pmap` 在多个设备（GPU/TPU）上并行化计算以实现数据并行。"
hasQuiz: true
---

现代硬件通常包含多个加速器，例如 GPU 或 TPU。尽管 `jax.jit` 优化单设备代码，且 `jax.vmap` 高效处理批处理，但它们本身并不会将计算分布到`*`多个`*`设备上。本章介绍 `jax.pmap`（并行映射），它是 JAX 用于将计算分布到不同设备上并行运行的函数变换。

您将学到：
*   单程序多数据 (SPMD) 并行原理，以及 `pmap` 所采用的执行模型。
*   如何使用 `jax.pmap` 在多个设备上同时执行相同的函数，每个设备处理不同的数据切片。
*   指定如何使用 `in_axes` 和 `out_axes` 等参数将数据分发到设备并从设备收集的方法。
*   在经过 `pmap` 转换的函数中，集体操作（例如使用 `jax.lax` 原语在设备间求和或求平均）的用法。
*   `pmap` 如何与其他 JAX 变换（例如 `jit` 和 `grad`）结合使用。

学完本章后，您将能够在多设备系统上应用 `pmap` 为您的 JAX 程序实现数据并行。

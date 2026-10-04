---
course: "advanced-jax"
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-3-distributed-computing-jax"
sourceId: 647
chapter: "distributed-computing-jax"
title: "JAX 分布式计算"
order: 3
description: "使用 pmap 扩展 JAX 计算以支持多设备（GPU/TPU）。学习数据并行、集合操作和设备管理。"
hasQuiz: false
---

随着机器学习模型规模和复杂度的增长，在单一加速器（如 GPU 或 TPU）上进行计算已不足够。训练大型模型或处理大规模数据集通常需要将任务分配到多个设备上。本章主要介绍 JAX 如何实现这种分布式计算。

我们将从与机器学习工作负载相关的并行性基础知识讲起。您将了解 JAX 如何管理不同的计算设备，以及如何使用其用于多设备执行的核心原语：`pmap`（并行映射）。我们将介绍 `pmap` 所采用的单程序多数据（SPMD）模式，并演示如何实现数据并行，这是一种常用的训练加速技术。

此外，您将学习在 `pmap` 处理的函数中，聚合跨设备信息（例如梯度）所需的必要集合通信操作（如 `psum`、`pmean`）。我们还将讨论如何使用轴名称来更明确地控制这些集合操作，并涉及高级分区策略以及多主机分布背后的机制。本章结束时，您将掌握如何使用 `pmap` 在多个加速器上有效扩展您的 JAX 计算。

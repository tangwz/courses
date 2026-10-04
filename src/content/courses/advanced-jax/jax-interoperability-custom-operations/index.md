---
course: "advanced-jax"
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-5-jax-interoperability-custom-operations"
sourceId: 655
chapter: "jax-interoperability-custom-operations"
title: "互操作性与自定义操作"
order: 5
description: "将 JAX 与其他框架结合使用。了解 DLPack，调用外部代码，以及在 JAX 中创建自定义原语操作。"
hasQuiz: false
---

JAX 计算常需要与科学 Python 生态的其他部分协同工作，或需要专门的底层功能。本章讨论如何将 JAX 与外部系统连接，并扩展其核心操作。

您将学到实用的方法，包括：
*   将数据在 JAX 和 NumPy 数组之间进行转换。
*   应用 DLPack 标准实现高效、零拷贝的数据共享，与 PyTorch 或 TensorFlow 等其他库。
*   在 JAX 程序中执行外部 Python 函数，使用 `host_callback` 和 `pure_callback`，了解它们的用途和限制。
*   理解 JAX 原语作为系统已知的基本操作。
*   构建自定义原语，通过定义它们的抽象求值行为（形状、数据类型），实现后端特定的低层化规则（XLA HLO 生成），并指定自定义微分规则（JVP/VJP），以确保它们能完全融入 JAX 的转换系统。

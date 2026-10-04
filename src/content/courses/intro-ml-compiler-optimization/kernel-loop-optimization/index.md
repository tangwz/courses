---
course: "intro-ml-compiler-optimization"
sourceUrl: "https://apxml.com/zh/courses/intro-ml-compiler-optimization/chapter-4-kernel-loop-optimization"
sourceId: 1385
chapter: "kernel-loop-optimization"
title: "核心与循环优化"
order: 4
description: "增进你对循环优化的理解。不同方法包含分块、向量化和循环展开，适用于CPU和GPU。"
hasQuiz: false
---

图级转换安排操作员之间的数据流，但它们不规定单个操作如何在硬件上执行。即使图结构已优化，神经网络的性能也很大程度上受制于核心内部代码的效率。本章将关注点从高层图转向执行实际计算的底层循环嵌套。

张量操作的直接实现通常只用到处理器理论峰值性能的一小部分。举例来说，一个定义为 $C_{ij} = \sum_k A_{ik} B_{kj}$ 的简单矩阵乘法循环，通常会因内存访问模式不佳而性能受损。在现代硬件上，瓶颈往往是数据从主内存移动到寄存器的延迟，而非算术逻辑单元（ALU）的速度。

在本模块中，我们考察编译器如何重写循环结构以最大化算术密度。你将学会操作程序的迭代空间，以提升缓存局部性和指令并行性。课程内容包含具体转换方法，例如：

*   **循环分块 (Loop Tiling)：** 将大型循环分解成适合CPU或GPU缓存层次的小块。
*   **向量化 (Vectorization)：** 将标量操作转换为向量操作，以使用SIMD（单指令多数据）指令。
*   **循环展开和重排序 (Unrolling and Reordering)：** 减少控制流开销，并为处理器同时执行指令提供可能性。

我们还会讨论通过将数据传输与计算重叠来隐藏内存延迟的策略。到本章结束时，你将明白如何检查循环嵌套并应用一系列调度原语，以针对特定硬件目标优化矩阵乘法核心。

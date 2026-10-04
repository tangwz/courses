---
course: "compiler-optimizations-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/compiler-optimizations-machine-learning/chapter-3-kernel-optimization-loop-schedules"
sourceId: 1382
chapter: "kernel-optimization-loop-schedules"
title: "内核优化与循环调度"
order: 3
description: "掌握机器学习内核的循环优化：分块、向量化、循环展开和重排序，以获得最高吞吐量。"
hasQuiz: false
---

图级转换主要用于减少算子启动的开销以及管理它们之间内存的负担。但是，大部分执行时间都花费在算子本身内部。这些内核通常由执行线性代数运算的密集嵌套循环组成。将数学逻辑直接转换为代码，常常因为缓存使用效率不高和指令级并行性未被充分发挥，导致性能达不到最佳。

本章侧重于循环优化的机制。我们区分计算（程序算什么）与调度（程序怎么执行）。例如，矩阵乘法在数学上可以这样定义：

$$C_{i,j} = \sum_{k} A_{i,k} \times B_{k,j}$$

尽管结果$C_{i,j}$保持不变，但处理器遍历$i$、$j$和$k$的顺序会极大地影响吞吐量。您将学习如何调整这些循环嵌套，以适应底层硬件架构。

我们将考察几种具体的调度原语，例如：

*   **循环分块（Loop Tiling）：** 将迭代空间划分为更小的块，以充分利用L1和L2缓存中的数据。
*   **向量化（Vectorization）：** 将标量操作映射到SIMD（单指令多数据）通道，以便每个周期处理多个元素。
*   **重排序与循环展开（Reordering and Unrolling）：** 修改循环结构以隐藏内存延迟并减少分支预测开销。

通过应用这些技术，您将可以把一个通用但运行缓慢的实现转变为高性能内核。这个过程构成了编译器将抽象张量操作映射到具体机器码的工程依据。

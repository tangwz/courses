---
course: "compiler-optimizations-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/compiler-optimizations-machine-learning/chapter-2-graph-level-transformations"
sourceId: 1381
chapter: "graph-level-transformations"
title: "图级转换"
order: 2
description: "学习图级优化：算子融合、常量折叠和神经网络中的公共子表达式消除。"
hasQuiz: false
---

当高级框架将模型降级为中间表示 (IR) 后，由此产生的计算图表示了操作的逻辑流程。然而，将此图直接翻译成机器码通常会导致次优的性能。将每个节点作为单独的核（kernel）执行会引入大量的调度开销，并对内存带宽造成不必要的负担。本章侧重于在编译器生成具体的硬件指令之前，优化此图的结构。

这里的首要目标是减少内存访问和核（kernel）启动的次数。例如，执行像如下的操作：

$$C = \text{ReLU}(A + B)$$

如果作为两个不同的核（kernel）执行，则需要将 $A + B$ 的结果写入全局内存，然后立即将其读回以进行激活函数运算。通过应用图级转换，编译器可以将这些操作融合成一个核（kernel），将中间数据保留在寄存器或缓存中。

您将学到如何实现具体的优化流程，例如算子融合（operator fusion），它将兼容的节点合并以提高局部性。本文档还包含代数简化（algebraic simplification），它借助数学属性用更廉价的替代方案替换开销大的操作。我们将审视布局转换（layout transformations），确保张量以 $NCHW$ 或 $NHWC$ 等内存格式存储，这些格式与目标硬件的访问模式一致。最后，本章讲述了通用清理流程，例如死代码消除（DCE）和公共子表达式消除（CSE），以确保图只包含必要的计算。

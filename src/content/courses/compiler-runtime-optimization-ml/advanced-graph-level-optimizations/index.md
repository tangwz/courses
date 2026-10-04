---
course: "compiler-runtime-optimization-ml"
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-3-advanced-graph-level-optimizations"
sourceId: 871
chapter: "advanced-graph-level-optimizations"
title: "高级图级别优化"
order: 3
description: "实现复杂的图变换，例如算子融合、排布调整和常量折叠。"
hasQuiz: false
---

在使用中间表示形式表示机器学习计算之后，我们现在着重优化计算图本身的结构。这些高级别变换旨在减少冗余操作、改善数据局部性、最小化内核启动开销，并在进行详细的张量级别优化之前，调整图结构以使硬件运行更高效。

本章研究直接在机器学习模型图表示上进行的优化技术。我们将介绍以下内容：

*   **图重写系统：** 用于查找和应用图变换的工作方式。
*   **算子融合：** 将多个操作合并为一个更高效的内核，以减少内存流量和开销。我们将考察不局限于简单相邻节点融合的策略。
*   **数据排布变换：** 根据目标硬件和算子序列优化张量数据的物理排列方式（例如 NCHW 与 NHWC）。
*   **代数简化：** 应用数学恒等式来简化或消除图中的操作。
*   **静态内存规划：** 通过分析图的依赖关系来最小化峰值内存使用量的技术。
*   **控制流处理：** 优化包含条件执行或循环的图的策略。

我们还将实现一个基本的融合遍，以巩固所学内容。到本章结束时，您将懂得如何应用复杂的图级别优化，以使机器学习模型能够高效运行。

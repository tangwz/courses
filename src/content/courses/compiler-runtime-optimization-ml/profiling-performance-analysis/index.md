---
course: "compiler-runtime-optimization-ml"
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-9-profiling-performance-analysis"
sourceId: 885
chapter: "profiling-performance-analysis"
title: "性能分析工具"
order: 9
description: "使用高级工具（Nsight、VTune）和方法来诊断编译后的机器学习代码中的性能瓶颈。"
hasQuiz: false
---

您已经学习了许多通过精密编译器和运行时策略来优化机器学习模型的方法。然而，应用这些优化只是过程的一部分。衡量它们的效果并找出剩余的性能瓶颈，是确保最高性能的必要步骤。编译器执行的复杂转换常常使得理解最终执行代码的性能特点变得困难。

本章侧重于有效分析编译后机器学习工作负载所需的工具和方法。我们将介绍：

*   性能分析机器学习编译器生成代码时遇到的具体难题。
*   使用系统级性能分析器（CPU、GPU、互连）来获取执行的整体情况。
*   运用 Intel VTune、NVIDIA Nsight 和 AMD ROCprof 等专用工具进行详细的 CPU 和 GPU 核分析。
*   将高层框架操作与具体执行的底层编译核关联起来的方法。
*   分析内存访问模式和解读编译器优化报告的方法。

本章结束后，您将能够使用高级性能分析工具诊断编译后的机器学习代码中的性能问题，从而指导后续的优化工作。

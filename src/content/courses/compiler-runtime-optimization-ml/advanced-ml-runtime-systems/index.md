---
course: "compiler-runtime-optimization-ml"
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-6-advanced-ml-runtime-systems"
sourceId: 880
chapter: "advanced-ml-runtime-systems"
title: "高级机器学习运行时系统"
order: 6
description: "设计高效的机器学习运行时系统，以处理动态形状、内存管理、自定义算子和异构调度。"
hasQuiz: false
---

机器学习模型经过多个编译阶段优化后，生成的代码需要高效的执行环境。这就是运行时系统的作用。本章将探讨为高要求机器学习任务定制的高级运行时系统的设计与实现。

我们将介绍运行时系统的主要职责，包括：
*   管理具有动态形状的张量（例如，在执行时确定尺寸，如 $N \times C \times ? \times ?$）。
*   为大型数据结构实现有效的内存管理方法。
*   协调异步执行，以实现计算与数据移动的重叠。
*   在CPU和GPU等异构硬件上高效调度操作。
*   集成自定义算子和核函数。
*   确保与高级机器学习框架的互操作性。

研究这些组成部分将有助于理解如何构建和分析使编译后的机器学习模型在目标硬件上实际运行的系统。

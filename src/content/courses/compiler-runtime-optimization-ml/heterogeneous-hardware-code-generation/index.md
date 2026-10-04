---
course: "compiler-runtime-optimization-ml"
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-5-heterogeneous-hardware-code-generation"
sourceId: 877
chapter: "heterogeneous-hardware-code-generation"
title: "异构硬件的代码生成"
order: 5
description: "使用先进的编译器后端为CPU、GPU（CUDA/ROCm）和AI加速器生成优化代码。"
hasQuiz: false
---

前几章侧重于在中间表示形式中转换和优化机器学习模型。本章将转向最后阶段：将这种优化后的IR翻译成高效的机器代码，以适应各种硬件目标，从标准CPU和GPU到专用AI加速器。

机器学习模型的性能在很大程度上取决于生成的代码如何很好地使用底层硬件的特定功能。生成最优代码需要了解针对特定目标的指令集、内存架构以及GPU张量核心或专用矩阵乘法单元等专用执行单元。仅仅映射操作是不够的；编译器后端必须对指令选择、寄存器分配和调度做出智能选择，并针对每种独特架构进行定制。

在本章中，你将研究编译器后端中使用的技术，以生成用于异构系统的高性能代码。我们将介绍：

*   选择适用于机器学习计算的硬件指令的策略。
*   管理寄存器，特别是针对向量（$SIMD$）和矩阵操作的技术。
*   使用CUDA和ROCm等框架为GPU生成代码的方法，包括针对NVIDIA Tensor Cores或AMD Matrix Cores等特定硬件功能。
*   将代码编译到自定义AI硬件（例如：TPU、NPU）的方法。
*   SPIR-V等目标无关中间格式在管理硬件多样性方面的作用。
*   编译器如何与供应商特定库（例如：cuDNN、MIOpen、oneDNN）集成并使用它们以获得最高性能。

本章提供了弥合高级优化表示与在当代多样化硬件平台上运行的高性能可执行代码之间差距所需的方法。

---
course: "compiler-runtime-optimization-ml"
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-4-tensor-level-polyhedral-optimizations"
sourceId: 874
chapter: "tensor-level-polyhedral-optimizations"
title: "张量级优化与多面体建模"
order: 4
description: "掌握张量运算优化，采用循环变换、分块、向量化及多面体模型。"
hasQuiz: false
---

在处理完图级别优化之后，我们现在将重心放在机器学习模型中计算密集的核心部分：张量运算。诸如矩阵乘法（常表示为 $C = A \times B$）或卷积之类的运算，在实现中常表现为复杂且嵌套的循环结构。有效运行这些循环对于性能最大化不可或缺。

本章将介绍优化这些特定张量计算的方法。你将学到：
*   将矩阵乘法和卷积等张量运算表示为循环嵌套。
*   多面体建模的原理，这是一种基于迭代空间和数据依赖性来分析和重构循环嵌套的形式化技术。
*   应用多面体转换，包括循环分块和调度，以提高数据局部性并展现并行性。
*   从多面体调度表示中生成优化后的循环代码。
*   补充性优化策略，例如用于SIMD执行的自动向量化，以及管理内存层级的技术，如缓存分块和数据预取。

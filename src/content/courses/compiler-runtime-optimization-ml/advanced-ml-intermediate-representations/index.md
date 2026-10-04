---
course: "compiler-runtime-optimization-ml"
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-2-advanced-ml-intermediate-representations"
sourceId: 868
chapter: "advanced-ml-intermediate-representations"
title: "机器学习的高阶中间表示"
order: 2
description: "学习MLIR等精巧的中间表示，以实现多层次的机器学习模型表示和优化。"
hasQuiz: false
---

高效编译机器学习模型需要不同于通用编译器中使用的表示形式。虽然标准中间表示（IR）对一般软件表现良好，但它们通常难以有效表示和优化机器学习任务中固有的高层操作和图结构。这种差异使得专门为机器学习中常见抽象和转换设计的多层次中间表示成为必要。

本章着重讨论这些高阶中间表示。我们将分析为何传统编译器中间表示往往不足以有效表达和优化机器学习计算。你将学习支撑多层次表示的设计原则，它们能实现在不同抽象层级的优化，从整个图到低层指令。我们将以MLIR（多层次中间表示）为例，详细考察其核心结构，包括方言和操作。你将了解MLIR如何表示高层机器学习框架图，如何将这些表示逐步降低至硬件目标的过程，以及其可扩展性如何支持新硬件或特定应用方向所需的自定义方言。本章为理解现代机器学习编译器如何管理复杂性并通过精巧的中间表示实现强大的优化提供依据。

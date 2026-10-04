---
course: "compiler-optimizations-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/compiler-optimizations-machine-learning/chapter-4-mlir-infrastructure"
sourceId: 1384
chapter: "mlir-infrastructure"
title: "MLIR 架构"
order: 4
description: "MLIR 概述：方言、Linalg、Affine 转换以及渐进式降低流程。"
hasQuiz: false
---

传统编译器架构通常促使从源代码迅速转换为低级表示。这种做法对于标量CPU指令固然有效，但它常会丢失有助于高效优化密集矩阵计算的结构信息。MLIR（多级中间表示）解决了这一局限，它使不同抽象级别的操作能够在同一模块中共存。这种架构支持渐进式降低，在此过程中，高级张量语义（例如形状信息 $$[N, C, H, W]$$）会得到保留和优化，然后才会转换为硬件专用指令。

本单元审视MLIR体系的模块化组件。我们将着重说明“方言”这一结构，它作为操作和类型的逻辑分组。你将学习用于表示结构化线性代数的 `linalg` 方言，以及用于内存依赖性分析的 `affine` 方言。本文还详细介绍了用于转换中间表示的模式重写架构。到本章结束时，你将了解如何定义自定义方言，并实现一个降低pass，将抽象逻辑转换为像LLVM IR这样的可执行目标。

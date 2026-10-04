---
course: "compiler-optimizations-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/compiler-optimizations-machine-learning/chapter-1-anatomy-deep-learning-compilers"
sourceId: 1378
chapter: "anatomy-deep-learning-compilers"
title: "深度学习编译器的结构"
order: 1
description: "了解深度学习编译器的架构。学习计算图、张量抽象和编译生命周期。"
hasQuiz: false
---

深度学习框架通过灵活、动态的抽象机制优先考虑开发人员的生产力。然而，硬件加速器要求静态、高度优化的指令流，以实现最高效率。深度学习编译器通过将高级数学描述转换为可执行机器码来弥合这一差距。此转换过程包含不同的阶段，这些阶段逐步降低抽象层次，从逻辑运算符转变为硬件原语。

与侧重于标量逻辑和复杂流程控制的通用编译器不同，AI编译器主要面向大规模并行和张量代数。优化主要单元从基本块转变为计算图。这种结构差异决定了编译器如何表示、分析和转换代码。例如，一个在数学上表示为 $$C = A \times B$$ 的简单矩阵乘法，涉及关于形状和数据类型的高级语义，这些语义在编译的初始阶段必须得到保留。

本章确立了编译流程的架构基础。我们将首先定义张量处理单元与标准CPU相比的具体要求。你将分析有向无环图 (DAG) 作为模型执行的核心数据结构的作用，并研究静态单赋值 (SSA) 形式如何促进神经网络中的依赖项跟踪。

我们还将区分中间表示 (IR) 的两种主要类型。你将识别图级别IR的特点，这类IR处理逻辑简化，以及指令级别IR的特点，这类IR处理内存分配和循环调度。本章以一个实践环节结束，在该环节中你将检查TVM的Relay IR的结构，以在一个生产级系统中观察这些原理。

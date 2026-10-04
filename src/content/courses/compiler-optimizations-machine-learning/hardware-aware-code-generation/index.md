---
course: "compiler-optimizations-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/compiler-optimizations-machine-learning/chapter-5-hardware-aware-code-generation"
sourceId: 1386
chapter: "hardware-aware-code-generation"
title: "硬件感知代码生成"
order: 5
description: "为GPU生成代码。包含内存层级结构、存储体冲突、warp分歧以及Tensor Core内部函数。"
hasQuiz: false
---

到目前为止，编译流程处理的是张量操作的抽象表示。图级转换和循环分块提升了总体数据局部性，但它们本身并未考虑目标设备的物理特性。为实现最高吞吐量，编译器必须将中间表示（IR）直接映射到硬件的指令集和内存层级结构。

本章着重介绍图形处理器（GPU）的代码生成策略，以NVIDIA架构作为主要参考。你将了解抽象循环嵌套如何转换为CUDA执行模型。此过程包括将逻辑迭代映射到线程块和单个线程，同时确保内存合并以最大化带宽使用率。

高效的内存访问通常是深度学习工作负载中的限制因素。我们将分析GPU内存层级结构，具体来说，是数据从全局内存到共享内存和寄存器的移动。你将学习避免共享内存中存储体冲突的对策，当多个线程同时尝试访问同一存储体中不同地址时，就会发生这种冲突。

本文还将介绍专用硬件单元的使用。现代加速器提供矩阵乘累加单元，例如Tensor Cores，它们提供比标准浮点单元高得多的算术密度。你将学习编译器如何发出必要的内部函数以针对这些单元。例如，一个标准的矩阵乘法运算

$$C_{m,n} = \sum_{k} A_{m,k} \times B_{k,n}$$

必须分解成与硬件特定操作数大小和数据类型相符的片段。

最后，我们将推出Triton，一种语言和编译器，旨在自动化许多此类复杂的内存管理任务。在本章结束时，你将明白生成高性能核函数背后的机制，以及决定其效率的具体限制条件。

---
course: "planning-optimizing-ai-infrastructure"
chapter: "foundations-ai-compute-infrastructure"
lesson: "role-of-gpus-in-ai"
sourceId: 6965
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-1-foundations-ai-compute-infrastructure/role-of-gpus-in-ai"
title: "GPU 在加速人工智能中的作用"
description: "了解图形处理器 (GPU) 如何加速深度学习和模型训练中的并行计算。"
order: 3
plots: []
sourceHash: "7bfdc1b7e8fdd9957f7bb686076cc392a2f96a094d32fabf40fbe8081e2f4e2b"
sourceCorrections: []
---

虽然CPU处理AI管线的核心序列任务，但模型训练的繁重计算几乎总是由另一种处理器完成：图形处理器（GPU）。GPU最初用于渲染视频游戏中的3D图形，但其架构已被证实非常适合深度学习 (deep learning)的数学运算。原因是其大规模并行性。

### 吞吐量 (throughput)架构

CPU旨在低延迟执行多种任务。它包含少量高性能核心，每个核心都能够执行复杂指令并做出精密判断，以加速单个执行线程。可以将其想象成一小队资深厨师，每位厨师都能快速地从头到尾准备一整套多道菜肴。

相比之下，GPU旨在高吞吐量计算。它包含数千个更小、更简单的核心，这些核心单个能力较弱，但可以同步处理同一个问题。这不像一队资深厨师，更像一个大型厨房流水线，数千名厨师每人执行一项简单、重复的任务，例如同时切数千个洋葱。这种架构方式有时被称为单指令多数据（SIMD）。

> CPU与GPU架构对比图。CPU将更多硅片用于复杂的控制逻辑和缓存，以支持更少但更强大的核心。GPU则将其大部分硅片用于大量的简单算术核心，并连接到专用、高带宽的内存。

### 神经网络 (neural network)运算加速

深度学习 (deep learning)模型由人工神经元层构成，这些层内的计算主要由几种数学运算组成，这些运算在大型数据张量上反复执行。最常见的是矩阵乘法。

在模型的前向传播过程中，输入数据在每一层都与一个权重 (weight)矩阵相乘。这可以表示为：


$$
\text{输出} = \text{激活}(\text{输入} \cdot \text{权重} + \text{偏置})
$$


输出矩阵中的每个元素都是点积的结果，这种运算独立于其他任何元素的计算。GPU可以将其数千个核心分配给数千个小而独立的点积计算，同时执行所有计算。CPU则需要以更序列化的方式执行这些计算，使用其少量高性能核心逐一或小批量处理这些操作。这种固有的并行性使GPU在处理神经网络层时比CPU快数个数量级。

### 人工智能的重要硬件特点

并非所有GPU都相同。选择用于AI工作负载的GPU时，某些特点特别重要：

- **计算核心：** 这些是基本的处理单元。在NVIDIA GPU中，它们被称为CUDA核心。GPU拥有的核心越多，能执行的并行操作就越多。
- **张量核心：** 在NVIDIA的Volta架构中引入，张量核心是旨在加速一种特定运算的专用硬件单元：矩阵乘加运算（MAC）。这种运算是深度学习 (deep learning)的核心。张量核心可以在单个时钟周期内对4x4矩阵执行融合乘加运算，为训练和推理 (inference)都提供了显著加速，尤其在使用FP16等混合精度格式时。
- **内存带宽：** GPU的核心只有在有数据输入时才有用。GPU使用自己的专用高速内存，通常称为VRAM。现代面向AI的GPU使用高带宽内存（HBM），它提供了极宽的内存接口。这种高带宽对于确保数千个核心持续获得数据供应是必要的，并防止它们闲置，这种情况被称为内存饥饿。

### 软件层：CUDA 和 cuDNN

如果没有软件层管理，GPU的原始算力 (compute)将无法使用。NVIDIA的CUDA平台便在此发挥作用。CUDA（Compute Unified Device Architecture）是一个并行计算平台和编程模型，允许开发人员使用类C语言编写GPU程序。

然而，大多数数据科学家和机器学习 (machine learning)工程师并不编写低级CUDA代码。他们转而使用TensorFlow和PyTorch等深度学习 (deep learning)框架。这些框架又依赖于高度优化的库，例如**NVIDIA CUDA 深度神经网络 (neural network)库 (cuDNN)**。

cuDNN是一个GPU加速的深度神经网络基元库。它为标准例程提供优化实现，例如：

- 卷积
- 池化
- 归一化 (normalization)
- 激活函数 (activation function)

当你编写一行PyTorch代码来定义一个卷积层时，cuDNN会高效地在GPU硬件上执行该操作。这种抽象让开发人员能够从GPU获得最大性能，而无需成为并行编程专家。这种大规模并行硬件和成熟软件栈的组合，使GPU成为进行认真深度学习任务的首选。

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — NVIDIA Corporation (2023)
  Publisher: NVIDIA Corporation
  了解CUDA架构、编程模型和GPU加速核心概念的官方指南。
- [NVIDIA cuDNN Developer Guide](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  提供了NVIDIA GPU上深度神经网络基本运算高度优化库的详细信息。
- [Programming Massively Parallel Processors: A Hands-on Approach](https://books.google.com/books/about/Programming_Massively_Parallel_Processors.html?id=o9JmDQAAQBAJ) — David B. Kirk, Wen-mei W. Hwu (2016)
  Publisher: Morgan Kaufmann
  一本关于GPU架构和并行编程原理的基础教科书，对于理解CPU与GPU的对比有帮助。
- [Tensor Core Programmability for Deep Learning](https://ieeexplore.ieee.org/document/9046645) — Mark Fowers, Sudarshan Gopalakrishnan, Joshua L. Romero, Stephen W. Keckler, Michael B. O'Connor, John D. Owens (2020)
  Journal: IEEE Micro; Publisher: IEEE; Volume: 40; Pages: 22-30; DOI: [10.1109/MM.2020.2974377](https://doi.org/10.1109/MM.2020.2974377)
  一篇学术论文，详细说明了NVIDIA Tensor Cores的架构和运行方式，以加速深度学习计算。

---
course: "compiler-runtime-optimization-ml"
chapter: "heterogeneous-hardware-code-generation"
lesson: "ai-accelerator-code-generation"
sourceId: 4527
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-5-heterogeneous-hardware-code-generation/ai-accelerator-code-generation"
title: "针对AI加速器（TPU、NPU）的代码生成"
description: "讨论自定义AI硬件架构的编译难题和方法。"
order: 5
plots: []
sourceHash: "038b18c543b31d8278636587f577de6db12620a8fc5b58251d2c86c41ad77bb1"
sourceCorrections: []
---

为谷歌的张量处理单元（TPU）或各种神经网络 (neural network)处理单元（NPU）等专用AI加速器进行编译，与为通用CPU乃至GPU生成代码相比，带来了特有的难题并需要不同的策略。GPU通过SIMT（单指令多线程）提供大规模并行能力，而AI加速器常采用脉动阵列等架构，或大量依赖于VLIW（超长指令字）原则，并结合专用内存层次结构和数据类型，对编译器后端有极高定制要求。

### 架构差异及对编译器的影响

影响编译器设计的架构特点包括：

1. **脉动阵列：** 在TPU中常见，由处理单元（PE）网格组成，数据在阵列中节奏性流动。编译器必须将矩阵乘法映射到这个物理网格上，精确管理数据移动时序以保持PE的繁忙状态。这涉及精巧的平铺、针对数组维度的特定数据布局转换，以及调度输入/输出操作以匹配数组的流水线延迟。编译器需要抽象脉动阵列执行模型，通常通过专用IR方言或操作（例如XLA HLO中的`TPUExecute`）实现。
2. **VLIW与指令捆绑：** 许多NPU采用VLIW架构，其中编译器负责识别并显式调度独立的运算到指令束中，以便在一个周期内并行执行。这使得复杂性从硬件（如CPU中的乱序执行单元）转移到编译器。高效的VLIW调度需要细致了解功能单元延迟、寄存器文件端口和潜在危险，使指令调度成为一个重要且复杂的优化过程。
3. **专用数据类型与功能单元：** 加速器常具有针对低精度格式（INT8、FP8、块FP格式）优化的硬件，以及带有GEMM引擎的专用功能单元，例如激活函数 (activation function)单元或池化单元。编译器必须高效使用这些单元，通过有针对性的指令选择，并管理必要的量化 (quantization)和反量化步骤，常将量化参数 (parameter)（缩放因子、零点）传递贯穿整个编译过程。
4. **内存层次结构：** 加速器内存系统常涉及软件管理的暂存器（本地内存）以及更大、延迟更高的主内存（如HBM）。编译器必须执行显式内存管理，协调直接内存访问（DMA）传输在主内存和暂存器之间进行。这需要复杂的数据重用分析、针对暂存器大小优化的平铺策略，以及精心调度计算和数据移动以隐藏延迟（如双缓冲）。与缓存不同，暂存器管理完全由编译器/运行时控制。

### 编译流程与技术

AI加速器的编译流程通常涉及多级IR降低和专用化：

> 一个典型的编译流程，用于针对专用AI加速器，涉及通过不同IR层级的逐步降低。

编译阶段和技术包括：

- **方言转换/降低：** 高层框架操作通过中间方言（如MLIR中）或IR（如XLA HLO）逐步降低，转换为更接近硬件的表示。这涉及规范操作、数据类型和布局以适应目标架构。
- **布局分配：** 选择最佳数据布局（例如，用于脉动阵列输入的专用平铺格式）非常重要且早期执行。
- **内存规划：** 静态或半静态规划将张量分配到暂存器或主内存，优化暂存器内的数据重用。DMA传输操作被显式插入。
- **脉动阵列映射：** 对于类似TPU的架构，矩阵乘法和卷积被平铺并调度到脉动阵列PE上。这常涉及复杂的循环转换和编译器内的数据放置策略。
- **VLIW调度：** 对于基于VLIW的NPU，指令级并行被显式管理。软件流水线和细致的依赖分析等技术用于高效打包指令束。
- **指令选择：** 将操作匹配到加速器的特定指令集，包括运用专用低精度单元或激活函数 (activation function)硬件。
- **代码生成：** 生成最终的二进制指令或微码给目标处理器。

### 厂商工具链与抽象

大多数AI加速器都配备专有厂商编译器工具链（例如，谷歌TPU的XLA编译器，各种NPU SDK）。这些工具链通常公开特定的IR或编程模型。虽然MLIR等框架旨在提供一个通用架构，但后端仍需对目标硬件有充分了解。编译器可能会为某些操作生成对厂商提供的内核库的调用，类似于cuDNN或MIOpen，但通常，由于涉及独特的架构，编译器本身负责生成核心计算操作的代码。集成通常在IR层面发生，主编译器会降低为厂商特定的方言或IR，然后再调用最终的后端阶段。

成功针对AI加速器需要编译器技术充分理解并建模特定的硬件特性，远超传统CPU/GPU编译技术，以管理显式并行、专用功能单元和软件控制的内存。

## 参考资料

- [MLIR: A Compiler Infrastructure for the End of Moore's Law](https://arxiv.org/abs/2002.11054) — Chris Lattner, Jacques Pienaar, River Riddle, Albert Cohen, Alan Mycroft, Oleksandr Zinenko, Andrew Davis, Stella Laurenzo, Junjie Bai, Mehdi Amini, Andy Ashcraft, Chris Brochtrup, Nicolas Vasilache (2020)
  Journal: CoRR, abs/2002.11054; Publisher: arXiv; Pages: 559-576; DOI: [10.48550/arXiv.2002.11054](https://doi.org/10.48550/arXiv.2002.11054)
  介绍了MLIR，一个使用多级IR和方言转换的灵活编译器基础设施，与本节描述的异构硬件编译流程直接相关。

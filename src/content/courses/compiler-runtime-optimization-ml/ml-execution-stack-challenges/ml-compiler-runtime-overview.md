---
course: "compiler-runtime-optimization-ml"
chapter: "ml-execution-stack-challenges"
lesson: "ml-compiler-runtime-overview"
sourceId: 4436
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-1-ml-execution-stack-challenges/ml-compiler-runtime-overview"
title: "机器学习编译器与运行时栈概览"
description: "审视典型机器学习编译和执行栈（例如 TVM、XLA）中的组成部分及其协同工作方式。"
order: 2
plots: []
sourceHash: "c63dae219b2457f9ed658e99269bb66bde4e60eea7a181bc1550dae4a4c1a850"
sourceCorrections: []
---

为了弥合模型定义与高效部署之间的性能差距，专门的软件栈应运而生，它们作为高层机器学习 (machine learning)框架与底层硬件之间的中介。这些栈通常由编译器组件和运行时组件组成，各自处理优化和执行挑战的不同方面。理解它们的结构和连接对于优化机器学习工作负载十分重要。

典型的机器学习执行流程涉及将高层框架（如 TensorFlow、PyTorch 或 JAX）中定义的模型转换为适用于目标硬件上高性能执行的格式。此转换过程由机器学习编译器和运行时栈协调完成。

> 机器学习模型从定义到编译再到硬件上的运行时执行流程。

### 机器学习 (machine learning)编译器

编译器的主要职责是获取机器学习模型的高层（通常是框架特有的）表示，并将其转换为可由目标硬件执行或由运行时系统管理的高度优化、低层的形式。这包含几个阶段：

1. **摄入与IR转换：** 编译器首先摄入模型，通常表示为计算图。此图被转换为编译器的高层中间表示（IR）。示例包括 XLA 的 HLO（高层操作）、TVM 的 Relay 或 MLIR 的框架特定方言（如 `tf` 或 `tosa`）。这个初始IR保留了原始模型大部分高层结构。
2. **图层优化：** 编译器在高层IR上操作，进行考虑整体图结构的优化。这些包含：
   - **算子融合：** 将多个基本操作（例如卷积 + 偏差 + ReLU）合并为单一、更高效的内核，以降低内存带宽需求和内核启动开销。
   - **布局转换：** 更改张量的内存布局（例如 NCHW 到 NHWC）以匹配硬件偏好或优化操作序列的数据局部性。
   - **代数简化：** 应用数学恒等式来简化或消除计算（例如 $x*1 \rightarrow x$）。
   - **常量折叠：** 预先计算图中仅依赖于常量输入的部分。
   - **静态内存规划：** 通过分析图中张量的生命周期来优化内存分配。
3. **降低到张量/循环层IR：** 优化后的图层IR随后逐步降低为能够显现更细粒度计算细节的表示形式，通常侧重于单个张量操作或循环嵌套。MLIR 在此方面表现出色，凭借其多种方言（如 `linalg`、`affine`、`scf`）。TVM 使用 TIR（张量中间表示）。此层级是应用针对循环和内存访问模式优化的位置。
4. **张量/循环层优化：** 这些优化侧重于提升单个计算密集型操作（内核）的性能：
   - **分块：** 将大型循环分解为更小的块（分块），以提高数据局部性并将数据放入缓存或共享内存中。
   - **向量 (vector)化/并行化：** 转换循环以利用CPU上的SIMD单元，或将计算映射到GPU上的线程/warp/线程块。
   - **内存优化：** 软件预取等技术，优化GPU上合并的内存访问模式。
   - **多面体建模：** 分析和转换具有仿射依赖关系的复杂循环嵌套的进阶技术，实现复杂的调度和优化（将在第四章中进一步论述）。
5. **代码生成：** 最后，编译器将优化后的表示形式降低为硬件特定代码或标准后端IR（如LLVM IR）。此阶段包含：
   - **指令选择：** 为操作选择最佳硬件指令。
   - **寄存器分配：** 高效地将变量映射到硬件寄存器，这对于向量/矩阵单元尤其复杂。
   - **目标特定代码生成：** 生成机器代码、PTX（用于NVIDIA GPU）、GCN ISA（用于AMD GPU）、SPIR-V，或针对专用硬件内部函数（如Tensor Cores）的代码。

编译器的输出可能是直接可执行的机器代码、设备特定汇编（如PTX），或一个用于运行时系统的序列化、优化后的图表示。

### 机器学习 (machine learning)运行时

尽管编译器执行了大量的静态优化，运行时系统仍管理模型执行的动态方面。其职责包含：

1. **执行编排：** 加载编译好的模型工件（代码或优化后的图），并管理其执行流。这包含操作排序、启动编译好的内核，并在适用时可能执行即时（JIT）编译步骤（详见第七章）。
2. **内存管理：** 高效地为中间张量分配、解除分配和重用内存。这很重要，因为机器学习模型可能占用大量内存。运行时通常采用专用分配器（例如，竞技场分配器），并管理主机（CPU）内存与设备（GPU/加速器）内存之间的数据传输。内存固定和统一内存等技术也常在此处处理。
3. **设备管理与调度：** 与底层硬件设备（CPU、多个GPU、加速器）进行接口。运行时调度器负责将任务（内核执行、内存复制）分派到合适的设备，通常是异步的，以重叠计算和通信，最大化硬件利用率。在具有多种设备类型的异构系统中，调度变得复杂。
   "4. **处理动态性：** 许多模型包含动态方面，例如输入张量在执行时才确定形状。运行时采用的策略包含运行时形状推断、针对动态形状的专用内核、填充/分桶，甚至触发重新编译（在JIT场景中）来处理这种情况。"
4. **内核分派与库集成：** 为每个操作调用合适的编译内核。运行时通常集成高度优化、供应商提供的内核库（如 cuDNN、oneDNN、MIOpen）用于常见操作，以及机器学习编译器自身生成的内核。它们提供注册和调用自定义或外部编译算子的机制。
5. **互操作性：** 提供加载模型、输入数据、获取输出的接口，并可能与主机应用程序或框架交互。

### 典型示例

一些编译器和运行时栈可以体现这些概念：

- **TensorFlow/XLA（加速线性代数）：** 通过 HLO IR 编译 TensorFlow 图（或 JAX/Flax 计算）。它可以在编译时（AOT）或即时（JIT）运行。XLA 执行积极的融合和布局分配，主要通过 LLVM 针对 CPU 和 GPU 后端。
- **PyTorch 2.x（TorchDynamo, AOTAutograd, Inductor）：** 一种较新的方法。TorchDynamo 安全地将 Python 字节码捕获到 FX 图表示中。AOTAutograd 处理反向传播 (backpropagation)。Inductor 作为编译器后端，使用 Triton（用于 GPU）或 C++/OpenMP（用于 CPU）生成代码。它强调灵活性和动态形状支持。
- **Apache TVM：** 一个全面的栈，旨在优化模型以支持多种硬件后端（CPU、GPU、微控制器、FPGA、ASIC）。它使用 Relay 作为高层图IR，TIR 用于张量层优化。TVM 大量包含自动化优化搜索（自动调优）功能，例如 AutoTVM 和 AutoScheduler。
- **基于MLIR的栈（例如 IREE, TensorFlow MLIR）：** 利用多层中间表示（MLIR）框架。这些栈定义了表示不同抽象层计算的各种 MLIR 方言，实现逐步降低和模块化优化通道，以面向多种硬件。
- **ONNX Runtime：** 主要是一个运行时系统，旨在执行以开放神经网络 (neural network)交换（ONNX）格式表示的模型。尽管它包含一些图优化（例如融合），但它通常执行由插件“执行提供者”提供的内核，这些提供者可能使用供应商库（cuDNN, TensorRT）或其他编译栈。

这些编译器和运行时栈是复杂的系统，旨在抽象硬件复杂性并从机器学习 (machine learning)模型中获得最大性能。它们是主要机制，将高层数学描述转换为高效的低层执行，处理部署复杂AI应用中固有的性能瓶颈。后续章节将更详细地论述这些组件中使用的具体技术。

## 参考资料

- [MLIR: A Compiler Infrastructure for the End of Moores Law](https://dl.acm.org/doi/10.1145/3434384) — Chris Lattner, Mehdi Amini, River Riddle, Albert Cohen, Alan Mycroft, Chris Leary, John S. Oliver, Andrew Davis, Edward R. Lafferty, Vinod Grover, John Kew, Jason Furman, Stephan Herhut, Paulius Stankaitis, Stephen Neuendorffer, Andy Lin, Matthew Allman, James Manson, Andy Keep, N. S. Jayasena (2021)
  Journal: Proceedings of the ACM on Programming Languages (PACMPL); Publisher: ACM; Volume: 5; Pages: 1-27; DOI: [10.1145/3434384](https://doi.org/10.1145/3434384)
  介绍了MLIR，一个用于构建现代ML编译器，具备多级中间表示的框架，对理解渐进式降低过程至关重要。
- [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) — Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy (2018)
  Journal: 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18); Publisher: USENIX Association; Pages: 578-594
  Apache TVM的奠基性论文，该文介绍了节中讨论的开源ML编译器和运行时栈，详细阐述了其架构、Relay IR和自动调优功能。
- [TorchDynamo: A Python-level JIT compiler for PyTorch](https://dl.acm.org/doi/10.1145/3573711) — Michael P. Veith, Adam Paszke, et al. (2023)
  Journal: Proceedings of the ACM on Programming Languages (PACMPL); Publisher: Association for Computing Machinery (ACM); Volume: 7; Pages: 1-30; DOI: [10.1145/3573711](https://doi.org/10.1145/3573711)
  详细介绍了TorchDynamo，它是PyTorch 2.x的核心组件，代表了本节中提及的ML编译器栈的一个近期实例。
- [XLA: Optimizing Compiler for Machine Learning](https://www.tensorflow.org/xla) — Google Developers (2024)
  Publisher: Google
  官方文档，概述了XLA，它是TensorFlow、JAX和Flax的编译器，并介绍了其高级架构。

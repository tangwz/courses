---
course: "compiler-runtime-optimization-ml"
chapter: "heterogeneous-hardware-code-generation"
lesson: "gpu-code-generation-cuda-rocm"
sourceId: 4521
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-5-heterogeneous-hardware-code-generation/gpu-code-generation-cuda-rocm"
title: "GPU 代码生成：CUDA 和 ROCm 后端"
description: "研究生成高效 PTX/GCN 代码的方法，包括线程映射和共享内存管理。"
order: 3
plots: []
sourceHash: "83d983db580bc8434267eb8bfd4bb8f9d2221adcdf5e6ee81108beeaa922befa"
sourceCorrections: []
---

为图形处理器（GPU）生成高效代码对于加速机器学习 (machine learning)工作负载非常重要。英伟达 (CUDA) 和 AMD (ROCm/HIP) GPU 的大规模并行架构提供了强大的计算能力，但要使用这些能力，需要复杂的编译器后端，能够将优化过的中间表示 (IR) 转换为专用的 GPU 机器码（英伟达 PTX 或 AMD GCN ISA）。这个过程远不止简单的指令映射；它涉及复杂的调度、内存管理和资源分配，这些都根据 GPU 硬件的独特特点进行了定制。

### 将计算映射到 GPU 执行模型

第一个挑战是将机器学习 (machine learning)操作中固有的并行性（通常在 IR 中通过并行循环或高级张量操作来表达）映射到 GPU 的分层执行模型上。

- **网格、块和线程层次结构：** 编译器必须将整体计算划分为线程块网格，其中每个块包含多个线程。这种映射通常由 IR 中计算的结构指导。例如，一个张量操作可能将输出张量的维度映射到层次结构的不同级别（例如，元素映射到线程，行/列映射到块）。后端通常采用启发式方法或依赖自动调优框架（稍后讨论）来确定最佳块尺寸（$Bx, By, Bz$）和网格尺寸（$Gx, Gy, Gz$），以平衡并行性与资源限制。
- **内核启动：** 编译器生成主机代码，负责分配 GPU 内存（例如，通过 `cudaMalloc` 或 `hipMalloc`），将输入数据从主机传输到设备（`cudaMemcpy`/`hipMemcpy`），使用所选的网格/块配置启动编译好的 GPU 内核，并最终将结果取回到主机。

### 生成内核代码：从 IR 到 PTX/GCN

核心任务是生成设备内核代码本身。这通常涉及通过一系列转换，将编译器的中级 IR（例如，用于 GPU 的 MLIR 方言，或带有 GPU 内在函数的 LLVM IR）降低为目标特定的表示，例如英伟达的并行线程执行 (PTX) 汇编或 AMD 的 GCN 指令集架构。

- **指令选择：** 后端为算术运算、内存访问、控制流和同步选择合适的 GPU 指令。这包括在可用时运用专用指令（例如，下一节讨论的张量核心指令）。
- **线程级并行：** 分配给单个内核调用的计算被分配到块内的线程。例如，在矩阵乘法 $C = A \times B$ 中，每个线程可能负责计算输出矩阵 $C$ 的单个元素或一个小块。编译器生成反映这种分布的代码，使用线程索引（CUDA/HIP 术语中的 `threadIdx.x`、`blockIdx.x` 等）来确定每个线程处理的具体数据。
- **寄存器分配：** 与 CPU 相比，GPU 具有大的寄存器文件，但寄存器是流式多处理器 (SM) 或计算单元 (CU) 上并发运行的线程之间的共享资源。每个线程的高寄存器使用率会限制活动线程的数量（降低*占用率*），可能阻碍 GPU 隐藏内存延迟的能力。编译器后端采用专门为 GPU 架构设计的复杂寄存器分配算法，旨在最小化寄存器压力，同时避免过多溢出到本地内存（这是慢速全局内存）。
- **共享内存管理：** GPU 编程的一个显著特点是使用程序员管理的 L1 暂存内存，称为*共享内存* (CUDA) 或*本地数据共享 (LDS)* (AMD)。编译器可以通过生成代码，将数据从全局内存显式地分阶段加载到共享内存中，从而显著优化内核。这使得同一块内的线程能够高效地共享和重用数据，大幅减少昂贵的片外内存访问。一种常用方法是*分块*，即线程协作地将输入数据块加载到共享内存中，使用该数据块执行计算，同步，然后处理下一个数据块。在使用共享内存时，编译器必须仔细管理分配、数据移动和同步（`__syncthreads()` 或等效的屏障）。

> 线程块内线程通过共享内存协作处理从全局内存加载的数据块的视图。编译器后端生成此数据移动和计算分布的代码。

### GPU 性能的后端优化

除了基本的映射，GPU 后端还实现了许多优化：

- **内存合并：** 生成内存访问模式，使得*warp*（英伟达，通常 32 个线程）或*wavefront*（AMD，通常 32 或 64 个线程）中的线程同时访问全局内存中连续的位置。合并访问比分散访问更高效地使用内存带宽。编译器分析访问模式（通常来源于线程索引计算），并尝试转换它们以实现合并。
- **同步优化：** 最小化显式屏障同步（`__syncthreads()`）的使用，因为它会强制块内线程等待，可能导致执行单元空闲。有时，编译器可以证明同步是不必要的，或者重构代码以减少同步点。
- **循环优化：** 标准循环优化，如循环展开、循环融合和循环分裂，被应用于线程级代码，减少循环开销并改进指令调度。
- **指令调度：** 重新排序线程执行流中的指令，以隐藏指令延迟和内存访问延迟，从而最优地使用 GPU 的并行执行单元。
- **占用率最大化：** 尽管这不是一个直接的优化阶段，但编译器关于寄存器使用和共享内存分配的选择直接影响占用率（活动 warp/wavefront 数量与 SM/CU 支持的最大数量之比）。后端有时会进行权衡，例如，如果能实现一个需要较少共享内存的算法，可以接受略高的寄存器压力，这可能允许更多块并发运行。

### CUDA 与 ROCm 的考量

尽管高层思想（网格、块、线程、共享内存）相似，尤其是在使用 HIP（异构计算可移植接口）等可移植层时，但底层硬件和 ISA 不同：

- **术语：** warp（英伟达）对比 wavefront（AMD），流式多处理器（SM）对比计算单元（CU）。
- **ISA：** PTX（英伟达的中间汇编）对比 GCN/RDNA ISA（AMD 的原生指令集）。编译器后端最终目标是这些不同的架构。
- **硬件特性：** 缓存层次结构、内存控制器行为、warp/wavefront 调度策略以及专用执行单元的差异，使得编译器后端需要进行目标特定的调优，即使是从 LLVM IR 等通用源生成代码也是如此。

生成真正最优的代码要求编译器后端拥有目标 GPU 架构的详细模型，使其能够在指令选择、资源分配和调度过程中做出明智的决定。这种详细的了解对于将 IR 中的抽象并行性转换为具体的、高性能的 GPU 内核非常重要。

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — NVIDIA Corporation (2023)
  Publisher: NVIDIA Corporation
  详细说明CUDA编程模型、GPU架构、内存层次结构以及针对NVIDIA GPU的性能优化技术。
- [ROCm Programming Guide](https://rocm.docs.amd.com/en/latest/) — AMD (2023)
  Publisher: AMD
  解释ROCm平台、HIP编程模型、AMD GPU硬件概念以及在AMD GPU上开发高性能应用程序的指南。
- [Programming Massively Parallel Processors: A Hands-on Approach](https://books.google.com/books?id=b0lGDwAAQBAJ) — David B. Kirk, Wen-mei W. Hwu (2016)
  Publisher: Morgan Kaufmann
  一本基础教科书，涵盖GPU架构、CUDA编程模型、并行编程模式和优化策略。
- [TVM: An Automatic End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) — Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy (2018)
  Journal: 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18); Publisher: USENIX Association; Pages: 578-594; DOI: [10.5555/3342192.3342240](https://doi.org/10.5555/3342192.3342240)
  介绍了TVM，一个深度学习编译器栈，通过TVM IR和调度自动优化并为包括GPU在内的多种硬件生成代码。
- [LLVM Language Reference Manual - NVPTX Target](https://llvm.org/docs/NVPTXUsage.html) — The LLVM Project (2023)
  关于NVIDIA PTX后端特定的LLVM中间表示（IR）的详细参考，展示LLVM如何转换为NVIDIA GPU的PTX汇编代码。

---
course: "compiler-runtime-optimization-ml"
chapter: "heterogeneous-hardware-code-generation"
lesson: "spirv-heterogeneous-execution"
sourceId: 4530
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-5-heterogeneous-hardware-code-generation/spirv-heterogeneous-execution"
title: "用于异构执行的中间格式 (SPIR-V)"
description: "研究 SPIR-V 作为跨API、跨供应商GPU和加速器程序设计的中间格式。"
order: 6
plots: []
sourceHash: "26bfb3e3fc9e73ce2c20877381e329626ba45b848eac895126dffe76d2bfa76f"
sourceCorrections: []
---

为现代硬件编译优化的中间表示（IR）带来了重大的工程挑战。多核CPU、不同供应商（NVIDIA、AMD、Intel）的各类GPU以及专用AI加速器（TPU、NPU）等架构都拥有独特的指令集、内存层次结构和执行模型。为每个目标编写和维护不同的编译器后端既耗费资源又限制了可移植性。

这种复杂性需要一个抽象层，它位于高级的、以ML为中心的IR（通常在MLIR等框架中管理）和最终的、特定于设备的机器码之间。该层旨在以与目标无关的格式捕获计算的要点，将最终的硬件特定转换推迟到供应商提供的驱动程序或专用后端编译器。由Khronos Group开发的标准可移植中间表示-V（SPIR-V）已成为这方面的一个突出解决方案。

### 什么是 SPIR-V？

SPIR-V 是一种二进制中间语言规范，主要设计用于表示并行计算和图形着色器。与 LLVM IR 等文本IR不同（尽管受其影响并常使用静态单赋值形式），SPIR-V 被定义为一种二进制格式。这种二进制特性简化了分发，减少了驱动程序的解析开销，并为前端编译器阶段和后端设备编译器之间提供了一个稳定的接口。

区分 SPIR-V 与更高级的 ML 图表示很重要。SPIR-V 在更低的级别运行，更接近硬件的执行模型。它不直接表示诸如“卷积层”这样的构造，而是表示实现此类层的分解循环、内存访问和算术操作，这些操作通常被组织用于在GPU或类似设备上并行执行。

### 核心组成和原理

一个 SPIR-V 模块封装了一个完整的计算单元。重要组成部分包括：

- **标头：** 包含魔数、版本、生成器ID、指令边界和架构信息。
- **功能声明：** 模块所需特性的声明（例如，`Matrix`、`Float16`、`Int8`、`Shader`、`Kernel`）。使用方驱动程序必须支持这些功能。
- **扩展：** 指定所使用的任何SPIR-V扩展（例如，供应商特定扩展）。
- **内存模型：** 定义指针语义和内存一致性保证（例如，`Logical GLSL450`、`Logical OpenCL`）。这对于协调线程/工作项之间的内存访问有重要作用。
- **入口点：** 声明作为执行起点的函数（例如，特定的计算内核或图形着色器阶段）。
- **类型、常量和全局变量：** 定义模块中使用的所有数据类型（标量、向量 (vector)、矩阵、图像、采样器、指针、结构体、数组）、常量值和全局变量。SPIR-V 具有丰富的类型系统，适用于图形和通用计算。
- **函数和指令：** 函数包含表示实际计算的指令块。指令编码为32位字，包括操作码和操作数（类型ID、结果ID、字面量）。控制流通过分支指令明确表示。

### 执行和内存模型

SPIR-V 明确定义了执行模型（`ExecutionModel`），例如 `GLCompute`（用于Vulkan计算着色器）或 `Kernel`（用于OpenCL内核）。它使用了与典型GPU执行层次结构兼容的思路：

- **工作项/线程：** 最基本的执行单位。
- **工作组/线程块：** 可以协作的工作项集合，常使用共享局部内存。
- **子组（Warps/Waves）：** 工作组内更小的工作项集合，在许多GPU架构上同步执行。SPIR-V 提供子组操作，用于在这些单位内高效地进行数据交换和集体操作。

SPIR-V 内存模型定义了抽象硬件内存空间的逻辑存储类（`StorageClass`）：

- `Function`：函数调用私有（通常映射到寄存器）。
- `Private`：工作项私有（通常映射到寄存器或线程局部栈）。
- `Workgroup`：工作组内工作项之间共享（映射到GPU共享/局部内存）。
- `CrossWorkgroup`：跨工作组可访问（映射到全局设备内存）。
- `UniformConstant`：只读数据，在工作项之间统一（映射到常量内存或全局变量）。
- 其他包括 `Input`、`Output`、`StorageBuffer`、`Image` 等。

编译器前端负责将源语言或更高级IR的内存语义映射到这些SPIR-V存储类。后端驱动程序随后将这些逻辑类转换为对相应物理硬件内存（寄存器、L1/L2缓存、共享内存、全局DRAM）的访问。

### ML编译流程中的 SPIR-V

SPIR-V 在面向特定硬件API和驱动程序之前作为一个汇聚点。在ML环境下，涉及SPIR-V的典型流程可能如下所示：

> 编译流程通常涉及MLIR等框架内的多个级别的IR，然后降级到SPIR-V方言，再序列化为标准二进制格式。供应商驱动程序使用此二进制文件生成最终可执行代码。

使用 SPIR-V 为编译器开发人员提供了多项优势：

1. **减少后端工作量：** 开发人员无需为 $N$ 个目标编写 $N$ 个后端，而是可以专注于从其高级IR生成单一的高质量SPIR-V路径。最终本机代码生成的负担转移到硬件供应商的驱动程序。
2. **可移植性：** 生成为SPIR-V的代码理论上可以在任何具有兼容驱动程序并支持所需功能（例如，通过Vulkan Compute或OpenCL）的硬件上运行。
3. **生态系统集成：** 用于验证、优化和转换SPIR-V的工具（如`spirv-opt`、`spirv-val`、`spirv-cross`）独立存在。
4. **灵活性：** SPIR-V 可以进行预先编译（AOT）并随应用程序一起发布，或者由运行时系统根据运行时条件即时编译（JIT）生成。

MLIR 等框架包含一个专用的 `spv` 语言。从 `gpu`、`vector` 或 `llvm` 等语言降级到 `spv` 语言涉及转换控制流结构、映射内存空间（例如，将MLIR的 `gpu.private`、`gpu.workgroup` 映射到相应的SPIR-V存储类），以及将操作转换为其SPIR-V指令等效项。

### 局限性与考量

尽管功能强大，但 SPIR-V 并非万能药。依赖它会带来某些权衡：

- **依赖驱动程序质量：** 最终性能在很大程度上依赖于供应商驱动程序中SPIR-V编译器的质量。次优的驱动程序编译可能抵消早期阶段执行的优化。在某些情况下，直接生成到PTX或GCN ISA的代码可能提供更细粒度的控制。
- **抽象开销：** 作为一个中间层，它可能阻止直接访问高度特定或非常新的硬件功能，除非有相应的SPIR-V扩展可用并被使用。使用供应商特定扩展会牺牲可移植性。
- **调试复杂性：** 调试性能问题或功能错误可能变得更加复杂，因为问题可能出在前端SPIR-V生成、SPIR-V本身或供应商驱动程序的处理上。通过SPIR-V将高级源代码与最终机器码关联起来增加了间接性。
- **性能调优：** 实现峰值性能可能仍然需要针对特定目标的调优。尽管SPIR-V提供了可移植的表示，但表达计算的最佳方式（例如，算法选择、工作组大小、内存访问模式）在不同的硬件架构之间仍然可能存在明显差异。

尽管存在这些考量，SPIR-V 提供了一种有价值的、标准化的中间语言，用于弥合高级ML编译器优化与并行处理硬件多样生态系统之间的差距。它允许编译器开发人员通过Vulkan和现代OpenCL等API面向广泛的设备，大大简化了异构代码生成的挑战。

## 参考资料

- [SPIR-V Specification](https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html) — Khronos Group (2024)
  Publisher: Khronos Group
  详细阐述SPIR-V二进制中间语言、其指令集和语义的权威技术规范。
- [MLIR: A Compiler Infrastructure for the End of Moore's Law](https://dl.acm.org/doi/10.1145/3385412.3386001) — Chris Lattner, Mehdi Amini, River Riddle, Albert Cohen, Alan Mycroft, Ville Voutilainen, Andy Davis, Jason Furman, Kevin Lew, Mike Lin, Jacques Pienaar, Richard Jones, John Reames, Hal Finkel, Zachary Devito, Tatiana Shpeisman, Stella Lau, Brian Griesing, Michael Van Der Sloot, George K. Thiruvathukal, John Stratton, Andrew P. Black, Ryan Johnson, Dan Liew, Philip L. Schwaller, Todd M. Ware, John L. Williams, and Yevgeniy E. Zemlyanskiy (2020)
  Journal: Proceedings of the 41st ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI); Publisher: ACM; Pages: 825–839; DOI: [10.1145/3385412.3386001](https://doi.org/10.1145/3385412.3386001)
  一篇介绍MLIR的基础论文，它提供了一种多级IR方法，常用于ML编译流水线中生成SPIR-V的上游过程。
- [SPIR-V Overview](https://www.khronos.org/spir-v/) — Khronos Group (2024)
  对SPIR-V进行了概述性介绍，阐述了其目标、优势以及所支持的工具和API生态系统。

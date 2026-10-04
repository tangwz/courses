---
course: "compiler-runtime-optimization-ml"
chapter: "quantization-low-precision-optimizations"
lesson: "mixed-precision-optimization"
sourceId: 4583
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-8-quantization-low-precision-optimizations/mixed-precision-optimization"
title: "混合精度计算优化"
description: "探讨编译器自动确定和优化混合精度执行方案的策略。"
order: 6
plots: []
sourceHash: "c9939700033bb098d778f9eedb43d7b8c1fcd6a7eecd1e4ce0c78815a01b8203"
sourceCorrections: []
---

尽管统一的低精度计算能带来显著的性能优势，但对于机器学习 (machine learning)模型的每个部分来说，这并非总是最佳策略。某些操作可能对量化 (quantization)误差高度敏感，若强制转换为 INT8 或 FP8，可能导致无法接受的精度下降。反之，若在所有地方都保留完整的 FP32 精度，则会抵消潜在的效率提升。混合精度计算提供了一个务实的折衷方案，它有策略地为模型不同部分使用不同的数值格式（例如 FP32、FP16、BF16、FP8、INT8），以平衡精度和性能。

优化这些混合精度模型给编译器带来了独特的挑战。编译器不仅要优化单个操作在其选定精度下的表现，还需有效管理不同精度*之间*的转换。

### 平衡精度与性能

混合精度优化的核心思路是，在对精度影响最小的地方积极应用低精度格式，并对敏感操作保留更高精度。常见的需要更高精度的情况通常包括：

- **输出层：** 预测类别分数或回归值的最后几层可能需要更高的保真度。
- **累加器：** 涉及求和的操作，例如大型规约或矩阵乘法累加器，可以从更高精度中受益，以避免中间计算期间的溢出或精度损失。现代硬件（如 NVIDIA Tensor Cores）通常在内部执行此操作（例如 FP16 乘法，FP32 累加）。
- **敏感操作：** 某些激活函数 (activation function)（如 GeLU 或 Softmax）或归一化 (normalization)层在积极量化 (quantization)时可能表现出显著的误差放大。
- **梯度：** 在训练场景中，梯度通常需要比激活或权重 (weight)更高的精度。

编译器的作用是促进这种平衡，要么通过遵守用户指定的精度注解，要么通过自动确定有效的精度配置。

### 编译器的优化策略

编译器采用多种策略来处理和优化混合精度计算：

1. **在 IR 中的表示：** 中间表示需要机制来表示具有不同精度的张量和操作。例如，MLIR 使用其类型系统将特定的浮点或整数类型（例如 `tensor<1x256xf16>`，`tensor<1x1024xi8>`）附加到值上。操作本身可能具有指示其操作精度的变体或属性。量化 (quantization)参数 (parameter)（缩放因子，零点）也必须与量化类型关联。
2. **精度分配：**

   - **手动：** 开发人员可以在原始模型框架中明确注解特定层或操作（例如，使用 NVIDIA 的 Apex 或框架特定的 API），使其保留在 FP32 或 FP16 中。编译器会遵守这些注解。
   - **自动：** 更高级的编译器会尝试自动精度选择。这可能包括：
     - *敏感度分析：* 对模型进行性能分析或使用启发式方法，识别量化引入最大误差的操作。
     - *成本模型：* 评估量化一个操作的性能增益（延迟、内存）与它潜在的精度成本。
     - *搜索算法：* 探索不同操作可能的精度配置空间，以精度指标和性能模型为指导，寻找帕累托最优解。这计算密集，但可以获得高度优化的配置。
3. **优化精度转换：** 数据类型之间的转换（例如，FP32 到 INT8，INT8 到 FP16）会通过量化（Quant）和反量化（DeQuant）操作引入开销。这些通常以逐元素缩放和平移的方式实现。天真的实现会在精度变化的地方插入这些转换，可能造成显著的开销。编译器通过以下方式优化这些转换：

   - **融合：** 将量化/反量化操作融合到前置或后继的计算核中。例如，反量化操作后接一个卷积，可以进行融合，使得卷积核本身能够读取 INT8 数据并在计算过程中隐式执行反量化，通常与更高精度累加结合。类似地，激活函数 (activation function)后接一个量化操作有时也可以融合。
   - **重排序和消除：** 如果多个连续操作涉及来回转换，编译器可能会重排序操作或消除冗余转换。例如，`INT8 -> DeQuant -> FP32 Op -> Quant -> INT8` 可以被简化，如果 `FP32 Op` 可以在 INT8 输入上直接实现，并适当处理缩放因子。
   - **内存布局转换的互动：** 数据布局转换（例如，NCHW 到 NHWC）可能会相对于量化操作进行调度，以改善计算和转换步骤的内存访问模式。
4. **专用核生成：** 编译器的后端必须为处理混合输入/输出或内部计算的操作生成高效代码。这包括：

   - 针对为混合精度工作设计的特定硬件指令（例如，INT8 乘法与 FP32 累加）。
   - 生成有效处理与量化类型相关联的缩放因子的核，通常将这些计算合并到主计算循环中，以避免单独的通道。
   - 在处理不同类型和大小的操作数时，管理寄存器分配和指令调度。

### 示例：优化转换

考虑一个序列 `Conv (FP32) -> ReLU (FP32) -> Conv (FP32)`。如果我们决定将第二个卷积量化 (quantization)为 INT8，天真的方法会插入转换：

`Conv (FP32) -> ReLU (FP32) -> Quant (FP32->INT8) -> DeQuant (INT8->FP32) -> Conv (INT8)`

编译器旨在优化这一点。首先，`Quant` 可能会向后融合到 `ReLU` 操作中（如果线性，甚至可以融合到前置的 `Conv` 中）。更重要的是，`DeQuant -> Conv(INT8)` 模式是融合的首要选择。`Conv(INT8)` 核可以生成为直接接受 INT8 输入，在乘累加操作期间包含反量化缩放因子（通常累加到 INT32 或 FP32 中），并以累加器精度生成输出。

> 图示通过融合优化量化和反量化节点。朴素的方法插入显式转换节点，而优化版本则将这些转换融合到相邻的计算操作中。

### 运行时支持

运行时系统与编译器协同工作。它需要高效管理不同数据类型的内存缓冲区，并处理在不同精度下操作的核之间的执行依赖，可能将它们调度到不同的硬件单元（例如，标准核心与专用矩阵引擎）。

通过结合复杂的编译器分析、转换技术和硬件感知的代码生成，混合精度计算使得开发人员和工具能够取得实际的平衡，从低精度算术中获得显著的性能提升，同时保持对要求较高的机器学习 (machine learning)应用所需的精度。

## 参考资料

- [Mixed-Precision Training](https://arxiv.org/abs/1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Bruce Ginsburg, Boris Ginsburg, Andrew H. Lastra, Andrew Levenberg, Hao Nguyen, Oleksandr Patmochnyk, Ganesh Seetharaman, D. Shane Snyder, Gregory F. Tang, Valerie Tarashchansky, Galen Wasserman, Barry Whaley, Pieter van der Wijngaart (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  介绍了使用FP16和FP32的混合精度训练，概述了损失缩放以及对敏感操作和累加器使用更高精度的技术。
- [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) — Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy (2018)
  Journal: 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18); Publisher: USENIX Association; Pages: 578-594
  介绍了TVM，一个端到端优化编译器，通过其中间表示和调度支持混合精度和量化，与编译器策略相关。

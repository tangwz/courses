# MLIR 中的下沉路径

来源：[原文](https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-2-advanced-ml-intermediate-representations/mlir-lowering-paths)

[返回章节目录](README.md) · [返回课程目录](../README.md)

MLIR 的优势在于能够使用方言同时表示多个抽象级别的计算。然而，仅仅表示一个高层 TensorFlow 图不足以生成适用于 GPU 或专用加速器的高效代码。将操作从更高层、更抽象的方言转换为更低层、更面向硬件的方言的过程被称为**下沉**。这并非一个单一的步骤；相反，它是一个通过一系列转换逐步细化的过程，从而形成一条“下沉路径”。

### 渐进式下沉原则

可以将下沉理解为逐步降低抽象级别。您从接近源机器学习 (machine learning)框架的操作（例如 `tf.Conv2D`）开始，然后逐步将其转换为更接近硬件的表示形式。此路径中的每个步骤通常涉及将操作从一个方言（或一组方言）转换为另一个通常更简单或限制更多的方言。

这种渐进式方法是 MLIR 设计的核心，并提供多项优势：

1. **定向优化：** 不同的优化在特定抽象级别进行效果最佳。高层图优化，如操作符融合（第 3 章），自然地应用于表示计算图的方言。借助多面体技术的张量级循环优化（第 4 章）在 `affine` 或 `linalg` 等方言上有效运行。低层代码生成细节（第 5 章）在下沉到 LLVM IR 或 SPIR-V 等面向机器的方言时处理。
2. **模块化和可重用性：** 方言及其之间的转换可以独立开发和测试。常见的中间层方言（如 `linalg` 或 `vector`）可以作为收敛点，使不同的前端（TensorFlow、PyTorch、ONNX）能够共享后续的优化和后端下沉通道。
3. **复杂度管理：** 编译过程被分解为可管理的步骤。MLIR 不使用从高层图到机器代码的单一、庞大的转换，而是使用一系列更小、定义清晰的方言转换。

### 常见的下沉路径

虽然具体路径取决于源框架、目标硬件和所需的优化，但典型的下沉流程可能如下所示：

1. **框架级别方言：** 该过程始于模型被导入到特定于框架的方言中（例如，TensorFlow 的 `tf`，PyTorch 的 `torch`）。这里的操作与原始框架的语义紧密对应。

   - *示例操作：* `tf.MatMul`
2. **高层计算方言：** 框架方言通常会下沉到更通用的计算图方言，例如 TOSA（张量操作集架构）。TOSA 提供一组标准化的张量操作，旨在实现框架互操作性。

   - *示例操作：* `tosa.matmul`
3. **结构化操作/循环抽象方言：** 操作随后经常下沉到诸如 `linalg`（线性代数方言）之类的方言。`linalg` 将张量计算表示为带索引映射区域上的通用操作，使其在生成显式循环之前易于进行平铺和融合等转换。

   - *示例操作：* `linalg.matmul`
4. **缓冲区化和内存管理：** 在某个阶段，抽象张量需要映射到具体的内存缓冲区。这涉及执行缓冲区分配（通常使用 `memref` 方言）并将张量操作转换为在这些缓冲区上操作的通道。`memref` 和 `bufferization` 等方言处理此功能。

   - *示例操作：* `memref.alloc`，`linalg.matmul` 在 `memref` 类型上操作。
5. **仿射/循环方言：** 为了生成显式循环嵌套，特别是在借助多面体优化时，通常会使用 `affine` 方言。`linalg` 操作可以下沉到 `affine.for` 循环和 `affine.load`/`affine.store` 操作。许多经典循环优化发生在此级别。

   - *示例操作：* `affine.for`，`affine.load`，`affine.store`
6. **向量 (vector)/SIMD 方言：** 为了利用 CPU 核心或 GPU 线程内的数据并行性，操作可能会下沉到 `vector` 方言，该方言表示 SIMD/SIMT 计算。

   - *示例操作：* `vector.load`，`vector.fma`，`vector.store`
7. **硬件目标方言：** 最后，代码会下沉到表示特定硬件指令集或运行时 API 的方言。

   - **CPU：** `llvm` 方言，它几乎直接映射到 LLVM IR。
   - **GPU：** 诸如 `gpu`（用于 GPU 特定功能，如内核、块、线程）、`nvvm`（用于 NVIDIA PTX）、`rocdl`（用于 AMD GCN）或 `spirv`（用于适用于 Vulkan/OpenCL 的 SPIR-V 表示形式）之类的方言。
   - **加速器：** 特定于目标硬件的自定义方言（例如，TPU 方言）。

> MLIR 中潜在下沉路径的可视化，显示了从框架级别方言到面向硬件目标的进展。请注意，缓冲区化、循环生成和向量化 (quantization)步骤有时可能以不同的顺序发生，或面向不同的方言，具体取决于编译策略。

### 实现下沉：转换和通道

MLIR 提供实现这些下沉步骤的基础设施，主要通过**方言转换框架**。开发者定义模式，指定如何将一个方言中的操作转换为一个或多个其他方言中的等效操作。这些模式可以是：

- **一对一：** 单个源操作直接映射到单个目标操作（例如，`tosa.relu` -> 实现 ReLU 的 `linalg.generic`）。
- **一对多：** 单个源操作扩展为多个目标操作（例如，将 `linalg.matmul` 下沉为嵌套的 `affine.for` 循环）。
- **多对一：** 多个源操作可能融合或组合成单个目标操作（在下沉过程中较不常见，在高层优化中更典型）。

这些转换模式被分组为**通道**。MLIR 通道管理器编排这些通道的执行，逐步应用转换。类型转换（例如，从 `tensor<...>` 到 `memref<...>`）也由转换框架系统地处理。

### 考虑因素和权衡

选择和实现下沉路径需要仔细考虑：

- **粒度：** 特定优化应在哪个方言上执行？过早的下沉可能会模糊图优化所需的高层结构；过晚的下沉可能会限制面向硬件代码生成的可能性。
- **目标特定性：** 常见的下沉路径用于可移植性，但实现最佳性能通常需要特定于目标的路径和模式，尤其是在面向专用加速器或运用特定硬件指令时（第 5 章涉及）。
- **正确性：** 在整个下沉过程中确保语义等效性非常重要。转换框架有所帮助，但严格的测试必不可少。

通过方言转换进行渐进式下沉是 MLIR 有效性的核心。它提供了一种结构化的方式来弥合高层机器学习 (machine learning)模型与低层硬件执行之间的语义鸿沟，从而在每个阶段实现定向优化。理解这些路径对于分析、调试和扩展基于 MLIR 构建的机器学习编译器流程是必需的。后续章节将在此基础上进行阐述，检查沿这些下沉路径各个点应用的具体优化。

## 参考资料

- [MLIR: A Compiler Infrastructure for the End of Moore's Law](https://arxiv.org/abs/2002.11054) — Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasilache, Oleksandr Zinenko (2020)
  Journal: arXiv preprint arXiv:2002.11054; Publisher: arXiv; Pages: 37-49; DOI: [10.48550/arXiv.2002.11054](https://doi.org/10.48550/arXiv.2002.11054)
  这篇基础性论文介绍了MLIR框架，详细阐述了其多层IR设计、方言系统和渐进式降低方法，这些都是理解降低路径的核心。
- [MLIR - Multi-Level Intermediate Representation](https://mlir.llvm.org/docs/) — The LLVM Project Developers (2024)
  官方文档提供了MLIR架构、核心概念、各种方言（例如linalg、affine、memref、vector、gpu、llvm）以及方言转换框架的全面指南，对于理解实际降低实现至关重要。
- [TOSA (Tensor Operator Set Architecture) Specification](https://www.khronos.org/registry/TOSA/specs/) — Khronos Group (2024)
  Publisher: Khronos Group
  此规范定义了TOSA，一个标准化的张量操作集，在MLIR中常被用作ML模型进行进一步降低之前的中间表示，以实现框架互操作性。

---

[上一节](04-%E8%A1%A8%E7%A4%BA%E9%AB%98%E5%B1%82ML%E5%9B%BE%EF%BC%88%E4%BE%8B%E5%A6%82TF%2C%20TOSA%EF%BC%89.md) · [下一节](06-%E5%8F%AF%E6%89%A9%E5%B1%95%E6%80%A7%E4%B8%8E%E8%87%AA%E5%AE%9A%E4%B9%89%E6%96%B9%E8%A8%80.md)

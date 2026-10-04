---
course: "compiler-runtime-optimization-ml"
chapter: "quantization-low-precision-optimizations"
lesson: "quantization-ir-representation"
sourceId: 4579
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-8-quantization-low-precision-optimizations/quantization-ir-representation"
title: "在IR中表示量化操作"
description: "讨论编译器IR如何表示量化张量和操作（例如，使用专用类型、属性）。"
order: 2
plots: []
sourceHash: "ed0ecc15bc90de1ea044a755abacff37be1e0195b7f9299743cc5ed987940aa5"
sourceCorrections: []
---

要有效地优化低精度模型，编译器的中间表示（IR）需要明确地捕获量化 (quantization)的特性。仅仅使用像 `i8` 这样的标准整数类型是不够的，因为它省略了这些整数如何映射回它们所近似的实数范围的重要信息。IR必须编码量化方案、参数 (parameter)以及浮点域和量化域之间转换的操作。

一种表示充当了高层模型描述（可能带有量化信息注解）与底层优化及代码生成阶段之间的约定。没有这种明确的表示，编译器无法推断量化操作的数值属性，也无法有效地针对专门的低精度硬件指令。

### 在IR中编码量化 (quantization)参数 (parameter)

主要挑战是表示实数 ($r$) 和量化整数 ($q$) 之间的仿射映射：


$$
r = s \times (q - Z)
$$


这里，$s$ 是比例因子（一个正浮点数），$Z$ 是零点（一个与 $q$ 范围匹配的整数）。$s$ 和 $Z$ 都必须以某种方式与IR中的量化张量数据关联。存在几种策略：

1. **专用量化类型**：这可以说是最简洁的方法，常见于像 MLIR 这样的多级IR中。定义了新类型，直接捆绑存储类型（例如 `i8`、`u8`）与量化参数，以及它可能表示的“表达”浮点类型。

   - **示例（MLIR 类似语法）：** 一个张量类型可能看起来像 `tensor<1x256x256x3x !quant.uniform<i8:f32, 0.015:128>>`。这明确定义了一个尺寸为 `1x256x256x3` 的张量，其中每个元素存储为 `i8`。这个 `i8` 使用统一仿射量化方案 (`!quant.uniform`) 表示一个 `f32` 值，其比例因子为 `0.015`，零点为 `128`。
   - **优点**：类型安全确保量化信息始终存在并一致传播。可以定义操作直接作用于这些特定的量化类型，简化了分析和转换。
   - **缺点**：需要扩展IR的类型系统，如果从更传统的IR开始，这可能是一项艰巨的任务。
2. **类型属性或元数据**：另一种方法是使用标准整数类型（例如 `tensor<1x256x256x3 x i8>`），但将量化参数（`scale`、`zero_point`、`axis` 用于每通道）作为属性或元数据附加到张量值或生成/消耗它的操作上。

- **示例 (IR)：**
   `%input_quant = MyDialect.Quantize(%input_fp32) {scale=0.015, zero_point=128, storage_type=i8} : (tensor<1x...xf32>) -> tensor<1x...xi8> %weight_quant = GetQuantizedWeight() {scale=0.008, zero_point=0, storage_type=i8, axis=0} : () -> tensor<64x...xi8> %conv_output_quant = MyDialect.Conv2D(%input_quant, %weight_quant) {output_scale=0.1, output_zero_point=110, storage_type=i8} : (tensor<...xi8>, tensor<...xi8>) -> tensor<...xi8>` 
  - **优点**：避免了复杂的类型系统更改。可以更容易地改造到现有IR上。
  - **缺点**：信息与数据耦合度较低。需要优化阶段仔细处理，以确保参数正确传播且不会意外丢失。编译时安全性较低。

### 表示量化 (quantization)粒度

模型经常对张量不同部分使用不同的量化参数 (parameter)，特别是权重 (weight)（每通道或每轴量化）。IR必须支持这一点：

- **每张量**：单一的比例因子和零点应用于整个张量。这更简单，并且通常表示为标量属性或直接在量化类型中。
- **每轴（每通道）**：适用多个比例因子和零点，通常是沿着特定轴的每个通道一个（例如，卷积权重的输出通道轴）。IR需要存储比例因子/零点的数组以及相应的轴索引。
  - 在专用类型系统中，类型本身可能编码轴并引用一个常量参数数组。
  - 在属性系统中，属性将保存数组值和轴索引。

### 表示量化 (quantization)和反量化操作

IR中需要明确的操作来表示浮点域和量化域之间的转换：

- **量化 (`quant`)**：以浮点张量和量化参数 (parameter) ($s$, $Z$) 作为输入，并生成一个量化整数张量。
  
  $$
  
  $$
  

q = \text{取整}(r / s) + Z
\$\$
IR操作节点将引用输入张量和比例因子/零点值（或隐含它们的量化目标类型）。

- **反量化 (`dequant`)**：以量化整数张量及其关联参数 ($s$, $Z$) 作为输入，并生成一个浮点张量。
  
  $$
  
  $$
  

r = s \times (q - Z)
\$\$
类似地，IR节点引用量化输入及其参数。

- **再量化 (`requant`)**：以量化张量（通常是具有更高精度的中间结果，例如 `i32` 累加器）以及输入和所需输出类型的参数作为输入。它在量化域中执行比例调整和可能的向下转换，避免了昂贵的浮点往返转换。
  
  $$
  
  $$
  

q\_{\text{out}} = \text{取整}((s\_{\text{in}} / s\_{\text{out}}) \times (q\_{\text{in}} - Z\_{\text{in}})) + Z\_{\text{out}}
\$\$
此操作对于优化量化计算序列非常重要，特别是发生中间累加的卷积和矩阵乘法。

下面是一个图表，显示了这些操作在图IR片段中可能如何出现：

> 一个流程图，显示了输入的量化、量化卷积生成更高精度的累加器、再量化回INT8以及最终反量化到FP32。

### 对优化的意义

拥有这种明确的表示使得编译器能够：

- **融合操作**：将 `quantize` -> `op` -> `dequantize` 序列组合成专用的量化 (quantization)内核调用。将 `requantize` 操作与前面的计算融合。
- **传播参数 (parameter)**：推断对于诸如逐元素加法或连接等操作产生的中间张量的正确量化参数。
- **选择最佳内核**：将高层量化操作（例如 `MyDialect.Conv2D` 操作在量化类型上）映射到特定的低精度硬件指令（例如 INT8 点积）或优化的库调用（例如 cuDNN、MIOpen、oneDNN）。
- **验证正确性**：执行静态检查以确保量化参数一致使用，并且操作对于给定的量化类型是有效的。

总之，在编译器的IR中直接且明确地表示量化操作及其关联参数是非常重要的。无论是通过专用类型还是属性系统，这种表示提供了必要的信息，以便复杂的优化阶段能够分析、转换并生成针对现代硬件量身定制的高效低精度代码。它弥合了使用量化的高层意图和性能所需的底层执行细节之间的差距。

## 参考资料

- [MLIR: A Compiler Infrastructure for the End of Moore's Law](https://arxiv.org/pdf/2002.11054) — Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasilache, Oleksandr Zinenko (2020)
  Journal: Proceedings of the 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI '20); Publisher: USENIX Association; Pages: 443-460; DOI: [10.5555/3471017.3471046](https://doi.org/10.5555/3471017.3471046)
  描述了MLIR的设计原则，解释了多级IR和可扩展类型系统在领域特定优化（包括量化）方面的优势。
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf) — Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko (2018)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 2704-2713; DOI: [10.1109/CVPR.2018.00287](https://doi.org/10.1109/CVPR.2018.00287)
  一篇基础论文，介绍了用于神经网络的带比例因子和零点的仿射量化方案，包括量化、反量化和重向量化操作的数学基础。
- [MLIR Quantization Dialect Guide](https://mlir.llvm.org/docs/Dialects/Quant/) — LLVM Project Developers (2024)
  Publisher: LLVM Foundation
  官方文档，详细介绍了MLIR量化方言，提供了专用量化类型和操作的具体示例，用于显式IR表示。
- [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) — Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy (2018)
  Journal: Proceedings of the 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI '18); Publisher: USENIX Association; Pages: 578-594; DOI: [10.5555/3342335.3342371](https://doi.org/10.5555/3342335.3342371)
  介绍了TVM，一个深度学习编译器框架，其IR旨在促进包括低精度和量化在内的各种优化，为编译器设计提供了背景。

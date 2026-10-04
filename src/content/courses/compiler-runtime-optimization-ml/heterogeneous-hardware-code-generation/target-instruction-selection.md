---
course: "compiler-runtime-optimization-ml"
chapter: "heterogeneous-hardware-code-generation"
lesson: "target-instruction-selection"
sourceId: 4515
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-5-heterogeneous-hardware-code-generation/target-instruction-selection"
title: "目标硬件指令选择"
description: "讨论为机器学习操作选择最优硬件指令的算法。"
order: 1
plots: []
sourceHash: "8d104e1c9efee75213516fadb782e5f27735d2b38fcc5deb71ce94350a573d1a"
sourceCorrections: []
---

编译器的后端面临一项重要任务：将中间表示 (IR) 中经由高级结构和张量操作优化而成的抽象操作，转换为针对目标硬件的具体机器指令。这个指令选择 (ISel) 阶段对于运用各种处理器的独特能力至关重要，这些处理器包括具有宽向量 (vector)单元的多核CPU、大规模并行GPU以及专用AI加速器。简单地将IR操作一对一映射到通用指令，对于高性能机器学习 (machine learning)工作负载而言通常远远不够。有效的指令选择需要精密的模式匹配、对专用硬件单元的了解，以及准确的成本建模，以应对涉及的复杂权衡。

### 机器学习 (machine learning)编译器中的指令选择挑战

指令选择旨在用与目标机器指令序列对应的模式来覆盖输入IR（通常表示为有向无环图或DAG），同时最小化特定的成本函数，通常是执行时间。尽管经典的编译器理论涉及指令选择，但机器学习工作负载引入了特定的挑战：

1. **复杂运算**：机器学习模型大量依赖多维卷积、广义矩阵乘法 (GEMM) 和复杂激活函数 (activation function)等操作，这些操作在通用硬件上通常没有单一指令的等效项。
2. **硬件专用化**：现代硬件具有专用单元，例如NVIDIA Tensor Cores、AMD Matrix Cores或Google TPU的矩阵乘法单元 (MXU)。选择使用这些单元的指令对于性能至关重要，但通常需要特定的数据类型（例如FP16、BF16、INT8）、数据布局和操作形状。
3. **异构性**：一个机器学习模型可能在CPU和GPU/加速器之间进行分区。指令选择过程必须针对图的不同部分，使用不同的指令集架构 (ISA)。

核心问题是为给定的IR片段选择*最优*的目标指令序列。这涉及识别IR中与高效硬件指令或微码序列对应的模式。

### 模式匹配策略

采用多种技术将IR模式与目标指令进行匹配：

- **树和DAG模式匹配**：传统的指令选择算法，例如基于树解析（如BURG）或DAG覆盖的算法，是其根本。这些算法试图使用代表目标指令的预定义模式，找到IR DAG的最小成本平铺。在运用MLIR等框架的机器学习 (machine learning)编译器中，方言系统提供结构化操作，有助于模式定义。例如，一个模式可能匹配`linalg.matmul`后跟`linalg.generic`（表示逐元素偏置 (bias)加法）的序列，并将其映射到融合硬件指令或高度优化的库调用（如果目标支持）。
- **基于规则的系统**：编译器通常使用目标描述文件（例如LLVM的TableGen `.td` 文件）中定义的明确规则。这些规则指定：

  - 一个IR模式（例如，两个`<4 x float>`向量 (vector)的`vector_add`）。
  - 一个对应的目标指令或序列（例如SSE `addps` 或AVX `vaddps`）。
  - 断言或条件（例如，目标CPU支持AVX）。
  - 与模式相关的成本。
    指令选择架构会查找适用规则，并为给定的IR节点选择成本最低的规则。

> 指令选择过程的简化视图，涉及模式匹配和成本评估，以将IR操作映射到目标指令。

### 针对特定硬件单元

机器学习 (machine learning)编译器中指令选择的主要目标是有效运用专用硬件单元：

- **CPU SIMD单元**：对于x86或ARM CPU等目标，指令选择将IR中的向量 (vector)操作（例如MLIR的`vector`方言操作）映射到单指令多数据 (SIMD) 指令，如SSE、AVX/AVX2/AVX-512或NEON。这涉及匹配向量长度、数据类型和操作（加法、乘法、乘加融合、重排等）。选择合适的指令宽度（例如128位SSE、256位AVX或512位AVX-512）取决于目标的能力和成本模型，同时考虑潜在开销，例如处理部分向量或寄存器溢出。
- **GPU计算指令**：当通过CUDA（用于NVIDIA）或ROCm（用于AMD）针对GPU时，指令选择将并行循环嵌套或张量原语转换为相应的汇编语言（PTX或GCN ISA）。这包括将计算映射到线程、管理寄存器（标量和向量），以及生成用于内存访问（全局、共享、纹理）和同步（屏障）的指令。例如，线程块内的归约操作可能被映射到一系列warp洗牌指令（PTX中的`shfl.sync`），以实现高效的warp内通信。
- **专用矩阵单元 (Tensor Cores, Matrix Cores)**：这些单元为矩阵乘法和卷积提供了大幅吞吐量 (throughput)提升，尤其是在低精度类型（FP16、BF16、INT8）下。指令选择必须明确识别与所需操作（例如GEMM）、数据类型以及可能适用于这些单元的形状相匹配的IR模式。目标指令（例如用于Tensor Cores的PTX中的`mma.sync`，用于Matrix Cores的GCN中的`mfma`）通常在小矩阵块（例如16x16x16）上操作。这些指令的选择与早期优化过程（如平铺和布局转换）密切相关，后者会适当准备循环结构和数据布局。未能满足这些先决条件意味着指令选择阶段无法使用这些高性能单元。
- **AI加速器 (TPU, NPU)**：定制加速器通常具有独特的ISA，有时是VLIW（超长指令字）或基于高级命令队列。针对这些目标的指令选择可能涉及将粗粒度IR操作（可能是整个层，如卷积或注意力）直接映射到特定的加速器指令或硬件配置。编译器的IR方言（如TOSA或自定义MLIR方言）通常与硬件共同设计，以使这种映射更加直接。指令选择可能生成命令序列来配置加速器内的DMA引擎、脉动阵列和向量处理器。

### 成本模型的作用

由于多个指令序列通常可以实现相同的IR片段，因此成本模型对于做出明智决策是必不可少的。该模型为每个可能的指令或序列分配成本（代表预估执行时间、吞吐量 (throughput)，有时是能耗）。

- **因素**：成本来源于硬件规格、微基准测试或分析模型。主要因素包括指令延迟（完成所需周期数）、吞吐量（每周期完成的指令数）、寄存器使用（高寄存器压力可能导致溢出）、代码大小和潜在隐患（例如依赖性、资源冲突）。
- **示例**：考虑在CPU上实现一个 $4 \times 4$ 矩阵乘法。指令选择可能在以下选项中进行选择：
  - 一系列标量乘法和加法。
  - 使用128位SSE指令。
  - 使用256位AVX指令（如果可用）。
  - 调用优化库例程（例如，来自BLAS）。
    成本模型根据目标CPU的能力评估每个选项的预估性能，为选择提供依据。向量 (vector)指令的延迟可能略高于标量指令，但吞吐量明显更好，如果存在足够的独立工作，则更优。

### 与编译器架构的集成

指令选择通常在LLVM等编译器框架的后端架构内执行，或作为定制代码生成器的一部分。

- **LLVM**：提供诸如SelectionDAG和GlobalISel的框架。针对CPU或GPU的机器学习 (machine learning)编译器通常会将其IR（例如MLIR方言）最终降低到LLVM IR，从而使其能够使用LLVM成熟的指令选择架构。目标特定信息在TableGen (`.td`) 文件中编码。
- **MLIR**：其多层次特性允许指令选择可能逐步进行。高级机器学习操作可能首先降低到`linalg`或`vector`等方言，然后通过专用方言和转换通道进一步降低到LLVM IR或直接到SPIR-V或目标特定汇编。这允许在通用指令选择之前在更高级别进行更多领域特定的模式匹配。
- **供应商库**：对于卷积或GEMM等复杂操作，指令选择可能选择*不*生成内联代码，而是生成对高度优化供应商库（例如cuDNN、MIOpen、oneDNN、BLAS）的调用。成本模型还必须预估这些库调用的性能，包括任何相关开销。

总之，针对特定目标的指令选择是机器学习编译流程中一个复杂但必不可少的阶段。它需要详细了解目标硬件架构、精密的模式匹配能力以及准确的成本模型，才能将优化后的IR转换为有效使用SIMD单元、GPU计算资源和专用AI硬件组件的机器代码，最终为要求高的机器学习应用提供高性能。

## 参考资料

- [The LLVM Target-Independent Code Generator](https://dl.acm.org/doi/10.1145/996841.996848) — Chris Lattner, Vikram Adve (2004)
  Journal: Proceedings of the ACM SIGPLAN 2004 conference on Programming language design and implementation (PLDI); Publisher: ACM; Pages: 166-177; DOI: [10.1145/996841.996848](https://doi.org/10.1145/996841.996848)
  介绍LLVM代码生成器的架构，详述其用于指令选择的SelectionDAG框架，该框架被现代编译器广泛应用。
- [MLIR: A Compiler Infrastructure for the End of Moore's Law](https://doi.org/10.1109/CGO51591.2021.9370308) — Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasilache, Oleksandr Zinenko (2021)
  Journal: Proceedings of the 2021 IEEE/ACM International Symposium on Code Generation and Optimization (CGO); Publisher: IEEE and ACM; Pages: 2-14; DOI: [10.1109/CGO51591.2021.9370308](https://doi.org/10.1109/CGO51591.2021.9370308)
  介绍MLIR，一种用于异构硬件的多级IR框架，阐述其方言系统和渐进式降低方法，这对ML编译器中的指令选择至关重要。
- [Tensor Comprehensions: Framework-Agnostic High-Performance ML for GPUs](https://dl.acm.org/doi/10.1145/3276483.3276495) — Nicolas Vasilache, Oleksandr Zinenko, Sam Gross, Zachary DeVito, Vaibhav Nagarajan, Jeremy Appleyard, Eric Johnson, Sirui Xie, Annie Liu, Lisa Lee, Tuanmho Nguyen, Andrew Adams, Jiri Simsa, Jeff Springer, Michael Garland, Trevor Elliott, Stephen Neuendorffer, Vijay Janapa Reddi, David Wong, Emina Torlak, Jonathan Ragan-Kelley (2018)
  Journal: Proceedings of the ACM on Programming Languages, OOPSLA; Volume: 2; Pages: 123:1-123:29; DOI: [10.1145/3276483.3276495](https://doi.org/10.1145/3276483.3276495)
  描述一种为深度学习生成高性能GPU代码的系统，侧重于其如何识别和利用NVIDIA Tensor Core等专用硬件特性。
- [A Survey of Deep Learning Compilers](https://arxiv.org/abs/2101.03759) — Guanhua Wang, Jiansong Li, Chengyu Dong, Gang Li, Mengze Li, Yu Chen (2021)
  Journal: ACM Computing Surveys; Publisher: ACM; Volume: 54; Pages: 107:1-107:37; DOI: [10.1145/3448375](https://doi.org/10.1145/3448375)
  提供深度学习编译器架构和挑战的全面概述，包括代码生成、硬件特定优化和目标多样化设备的讨论。

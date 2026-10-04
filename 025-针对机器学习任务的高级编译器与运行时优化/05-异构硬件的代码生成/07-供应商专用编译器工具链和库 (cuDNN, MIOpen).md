# 供应商专用编译器工具链和库 (cuDNN, MIOpen)

来源：[原文](https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-5-heterogeneous-hardware-code-generation/vendor-compiler-toolchains)

[返回章节目录](README.md) · [返回课程目录](../README.md)

生成机器学习 (machine learning)工作负载的高度专业化代码存在多种策略。其中一种有效且常有辅助作用的方法是运用硬件供应商提供的预优化例程。这些供应商专用库，比如 NVIDIA 的 cuDNN、AMD 的 MIOpen 和 Intel 的 oneDNN，封装了多年针对特定架构的调优成果，用于常见的、对性能要求高的机器学习基础操作。有效整合这些库是构建生产级机器学习编译器后端的一个显著方面。

### 供应商库的考量

硬件供应商对其芯片的微架构细节、内存子系统和指令集拥有无与伦比的认识，包括未公开特性或复杂交互。他们投入大量工程资源，手动调优卷积、矩阵乘法 (GEMM)、池化和归一化 (normalization)层等基础操作的核函数。这些库通常为单一操作提供多种算法，每种都针对不同输入维度、批次大小、数据类型或硬件代系进行了优化。

在通用机器学习 (machine learning)编译器中，为每个受支持的硬件目标和每个可能的运算变体复现这种优化水平通常不切实际，甚至不可能。因此，对于这些定义明确、计算密集型的构建块，依赖供应商库具有多项优点：

1. **峰值性能：** 供应商库通常代表其各自硬件上标准操作的性能上限。
2. **降低编译器复杂性：** 编译器可以将这些特定核函数复杂的底层调优工作委派给库。
3. **加快上市速度：** 借助现有的高性能库，可加速对新硬件或标准操作的支持。

然而，这种方法并非没有权衡。供应商库通常处理独立操作，可能错过通过跨操作合并实现的优化机会，而编译器自身的代码生成路径可以加以发挥。此外，依赖外部库会引入依赖关系和潜在的版本控制难题。

### 整合策略

机器学习 (machine learning)编译器采用各种策略与这些库进行整合：

#### 1. 直接库函数调用

最直接的机制是识别编译器IR中的特定操作节点（或小型子图），该节点直接对应供应商库提供的函数。例如，IR中的2D卷积节点可以映射到 `cudnnConvolutionForward` (用于 NVIDIA GPU) 或 `miopenConvolutionForwardImmediate` (用于 AMD GPU)。

编译器后端生成如下代码：

- 分配库函数所需的必要工作空间内存。
- 将输入和输出张量指针及元数据（维度、步长、数据类型）整理成库API期望的结构。
- 在执行流（例如 CUDA 流）中插入对库函数的实际调用。

这要求编译器对库的API有详细了解，包括函数签名、数据布局期望（例如 NCHW 与 NHWC）和所需的描述符。

#### 2. 库调度与启发式方法

供应商库经常为同一逻辑操作提供多种底层算法（例如，基于GEMM、基于FFT、Winograd 等不同的卷积算法）。最佳选择在很大程度上取决于运行时参数 (parameter)，如输入/滤波器维度、步长、填充、数据类型以及特定的GPU架构。

cuDNN 等库提供查询可用算法和启发式方法以选择“最佳”算法的机制。例如，可以使用 `cudnnGetConvolutionForwardAlgorithm` 或 `cudnnFindConvolutionForwardAlgorithm`。编译器可以通过以下方式进行整合：

- **编译时 (AOT) 基准测试：** 在编译期间（如果形状是静态的），调用库函数为特定问题维度基准测试不同算法，并选择最快的一个。这会增加编译时间，但能获得理想的运行时性能。
- **运行时选择：** 生成代码，在首次执行（或预热阶段）时调用库的“查找”例程，以动态确定最佳算法。
- **编译器启发式方法：** 在编译器内部实现启发式方法，可能通过离线训练或基于分析模型，来预测最佳库算法，而无需运行时基准测试。这在编译时速度更快，但可能不会总是选择绝对最佳的算法。

这些选择取决于编译上下文 (context)（AOT 与 JIT）、可接受的编译开销以及对确定性性能的需求。

#### 3. 选择性回退

一个先进的编译器后端不会将库整合视为一个非此即彼的命题。它保留了自身的代码生成能力（如前几节所述），同时具备调用供应商库的能力。决策逻辑可能如下所示：

> 编译器在自定义代码生成和供应商库调用之间做出选择的决策流程。

这使编译器能够运用库来处理标准、高性能的核函数，同时保持生成自定义代码的灵活性，以应对合并操作、不受支持的操作，或者其自身代码生成预计更优的情况。

### 与供应商工具链的交互

为 GPU 等目标生成代码的机器学习 (machine learning)编译器（为 NVIDIA 生成 PTX，或为 AMD 生成 GCN ISA）通常隐式或显式地依赖于供应商的下游工具链。

- **NVCC/HIPCC：** 生成的 PTX 或 LLVM IR（在 ROCm 路径中）通常通过供应商的编译器驱动程序（NVCC、HIPCC）或底层组件（例如 `ptxas`）编译成二进制 SASS (NVIDIA) 或 GCN 机器码 (AMD)。机器学习编译器框架可能会在最终构建步骤中调用这些工具。
- **驱动API：** 代码加载和执行通常通过供应商驱动API（CUDA Driver API、HIP Runtime API）进行。机器学习运行时系统与这些驱动程序交互，以管理设备内存、启动核函数（包括自定义生成和库提供），并同步执行流。

了解这些供应商工具链的能力和局限（例如，寄存器分配策略、指令调度），在生成像 PTX 这样的中间代码时也有益处，旨在生成供应商汇编器可以有效优化的输入。

### 考量与挑战

- **API兼容性：** 供应商库API会演变。编译器必须管理与不同库版本的兼容性或要求特定版本，这增加了构建和部署过程的复杂性。
- **不透明性：** 库核函数常是“黑盒 (black box)”。精确建模其性能特性或资源使用（例如共享内存、寄存器）以整合到更大的调度或内存规划优化中可能很困难。
- **覆盖空白：** 库在标准操作方面表现出色，但可能缺乏对新型研究原语、高度专业化激活函数 (activation function)或特定数据类型组合（例如，不常见的低精度格式）的支持。编译器必须通过自定义代码生成来处理这些情况。
- **构建依赖：** 整合这些库引入了外部构建依赖，使编译器的构建系统更复杂。

实际中，高性能机器学习 (machine learning)编译系统策略性地结合其自身复杂的代码生成技术与有目标地运用供应商优化过的库。这种混合方法使它们能够跨越广泛的模型和硬件平台，实现当前最好的性能，并在灵活性和合并需求与硬件供应商提供的原始核函数性能之间取得平衡。掌握这种整合对于弥合优化过的IR与可部署、高速机器学习推理 (inference)之间的最后差距十分重要。

## 参考资料

- [NVIDIA cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/latest/index.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA
  提供 NVIDIA 深度学习原语库的技术规范、API 定义和使用指南。
- [MIOpen Documentation](https://rocm.docs.amd.com) — AMD (2024)
  Publisher: AMD
  详细说明 AMD 深度学习原语库的 API、支持的操作和集成指导。
- [oneAPI Deep Neural Network Library (oneDNN) Documentation](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHWq3Y-aR3WSjshiXHyayIrXUfmy8VVeHKGhW0HmXa77MAg_FU7emnZ_GVLYpqHxvOwzHfXSCburh21dnjD2SyataeWjymgy--E-LA9eKMnaRV-WF3W1VCoaBPBYoTm7rwc) — Intel (2024)
  Publisher: Intel
  提供有关 Intel 跨架构深度学习原语库的全面信息，包括 API 和性能考量。
- [Apache TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) — Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy (2018)
  Journal: 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI '18); Publisher: USENIX Association; Pages: 578-594; DOI: [10.1145/3342345.3342371](https://doi.org/10.1145/3342345.3342371)
  描述 TVM，一个编译器，它将供应商库与自身的代码生成相结合，用于深度学习工作负载。

---

[上一节](06-%E7%94%A8%E4%BA%8E%E5%BC%82%E6%9E%84%E6%89%A7%E8%A1%8C%E7%9A%84%E4%B8%AD%E9%97%B4%E6%A0%BC%E5%BC%8F%20%28SPIR-V%29.md) · [下一节](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%88%86%E6%9E%90%E7%94%9F%E6%88%90%E7%9A%84GPU%E6%A0%B8%E5%87%BD%E6%95%B0.md)

---
course: "llm-compression-acceleration"
chapter: "hardware-acceleration-systems-optimization"
lesson: "mapping-llm-ops-hardware"
sourceId: 4404
sourceUrl: "https://apxml.com/zh/courses/llm-compression-acceleration/chapter-6-hardware-acceleration-systems-optimization/mapping-llm-ops-hardware"
title: "将LLM操作映射到硬件架构"
description: "分析LLM计算（注意力、矩阵乘法）如何映射到CPU、GPU、TPU及其他加速器架构。"
order: 1
plots: ["plots/4404-0.json"]
sourceHash: "3260da3bc9ec32db440e004ceedf6be31ad0de43fb9a55fd178161bd12649f8f"
sourceCorrections: []
---

高效运行大型语言模型，其核心在于如何将模型的计算操作有效转化为目标硬件的特定能力。模型压缩技术旨在降低理论复杂度；然而，在实际应用中实现低延迟和高吞吐，还需要明白矩阵乘法 (GEMM) 和注意力机制 (attention mechanism)等基本操作如何与CPU、GPU、TPU 及其他专用加速器配合工作。这种映射不仅仅是运行代码；它包含构建计算流程，以充分运用每种架构自身具备的并行性、内存层级和专用执行单元。

基于Transformer的LLM中的主要计算是稠密矩阵乘法，这在 前馈网络 和 注意力投影 中都能看到，还有注意力机制本身，它包含数次矩阵乘法以及逐元素操作和数据置换。这些操作的效率决定了整体推理 (inference)性能。

### 中央处理器 (CPUs)

现代多核CPU通过多个独立核心提供并行性，并通过单指令多数据 (SIMD) 向量 (vector)单元（例如 AVX2, AVX-512）提供指令级并行。将LLM操作映射到CPU上，包含：

1. **多线程：** 将计算（如矩阵乘法的分块或批次中元素的并行处理）分配到可用的CPU核心上。OpenMP 或其他线程框架负责这种分配。
2. **向量化 (quantization) (SIMD)：** 使用向量指令同时对多个数据元素执行相同操作（例如，在单条指令中乘法或加法操作8个、16个甚至更多的浮点数）。编译器会尝试自动向量化，但最佳性能通常需要内联函数或专用库，例如Intel的MKL或OpenBLAS，它们提供高度优化的GEMM程序，采用AVX技术。

然而，CPU在处理大型LLM时通常面临局限：

- **并行度较低：** 与GPU或TPU相比，CPU的并行执行单元少很多。
- **内存带宽：** 尽管CPU缓存很精巧，但主内存带宽在处理LLM的庞大权重 (weight)矩阵和激活值时可能成为一个显著瓶颈，特别是对于GEMM操作，它们具有高计算强度，同时也需要大量数据传输。
- **GEMM扩展性：** 矩阵乘法在CPU上的性能在一定程度上随矩阵大小良好扩展，但最终受限于缓存大小和内存带宽，特别是对于LLM中经常出现的较大隐藏维度。

CPU仍然适用于较小模型、特定部署场景（例如，在功耗或成本限制不允许使用专用加速器的边缘设备），或优先考虑小批量延迟而非最大吞吐的任务。

### 图形处理器 (GPUs)

将LLM操作映射到GPU上的重要方面包含：

1. **大规模并行 (SIMT)：** GPU包含数千个简单核心（在英伟达术语中称为CUDA核心，被组织成流式多处理器或SM），它们以单指令多线程 (SIMT) 方式执行指令。这使得大批量的相同操作（如GEMM或逐元素层中的操作）能够并行执行。
2. **高内存带宽：** GPU配备高带宽内存 (HBM)，提供比典型CPU主内存显著更高的内存吞吐量 (throughput)。这对于为庞大的计算单元提供数据，并处理LLM中大型的中间激活张量来说非常重要。
3. **专用单元 (Tensor Cores)：** 现代英伟达GPU包含Tensor Cores，旨在加速特定类型的矩阵乘累加 (MMA) 操作，特别是对于FP16、BF16和INT8等混合精度格式。将GEMM操作映射到Tensor Cores可以相较于标准FP32 CUDA核心执行带来显著性能提升。cuBLAS（用于GEMM）和cuDNN（用于其他神经网络 (neural network)操作，例如卷积，尽管对Transformer不那么重要）等库在使用适当数据类型时会自动使用Tensor Cores。
4. **注意力映射：** 自注意力 (self-attention)机制 (attention mechanism)，包含查询-键点积、softmax和值聚合，带来独特挑战。虽然它由GEMM和逐元素操作组成，但中间步骤和数据依赖需要仔细实现。朴素实现可能受内存带宽限制，因为需要读写较大的中间注意力分数矩阵。优化的核函数，例如FlashAttention，重构计算，以减少与HBM之间的数据传输量，通常融合多个步骤并更有效地使用片上SRAM（共享内存）。

计算映射包含将工作（例如，GEMM中的矩阵瓦片）分割到SM和线程块上，管理HBM与SM本地内存资源（寄存器、共享内存）之间的数据传输，以及使用CUDA或OpenCL等框架协调执行。

### 张量处理器 (TPUs)

谷歌的TPU是专用集成电路 (ASICs)，明确设计用于加速神经网络 (neural network)计算，主要关注大规模矩阵操作。

1. **矩阵乘法单元 (MXU)：** TPU的核心是MXU，它作为脉动阵列工作。脉动阵列是执行矩阵乘法的高效架构。数据元素节奏性地通过处理单元 (PE) 网格传输，每个单元执行乘累加操作。这种设计在初始操作数加载完毕后，可最大限度地减少主内存（通常是HBM）的数据传输，从而为稠密矩阵操作实现极高计算吞吐量 (throughput)。
2. **GEMM映射：** GEMM操作几乎直接映射到MXU的功能上。MXU的较大尺寸（例如128x128）非常适合LLM层中常见的维度。
3. **向量 (vector)/标量单元：** TPU也包含向量和标量单元，用于处理非矩阵操作，例如激活函数 (activation function)、归一化 (normalization)和逐元素计算。然而，性能重心严重偏向MXU。不以大型稠密矩阵乘法为主的工作负载，可能无法像以其为主的工作负载那样有效地使用TPU。
4. **软件生态系统 (XLA)：** TPU通常通过TensorFlow或JAX等更高级框架编程，它们使用XLA（加速线性代数）编译器。XLA在生成专门针对TPU的MXU和其他单元的底层代码之前，会编译计算图，优化和融合操作。

TPU在涉及庞大矩阵计算的情况中表现出色，这是LLM训练和大数据量推理 (inference)的特点。其专用特性为这些任务提供了卓越性能和能效。

### 其他加速器 (NPUs, FPGAs, 定制ASICs)

除了通用CPU和GPU/TPU之外，还存在越来越多的专用加速器：

1. **神经网络 (neural network)处理器 (NPUs)：** 常见于移动系统级芯片 (SoCs) 或边缘设备，NPU设计用于高效加速常见的神经网络操作。它们可能包含专用指令集或硬件模块，用于INT8矩阵乘法、特定激活函数 (activation function)乃至某些注意力模式等任务。映射包含适配这些特定硬件功能，通常通过专用供应商库和编译器（例如高通的SNPE，苹果的Core ML）。
2. **现场可编程门阵列 (FPGAs)：** FPGA提供硬件可重构性。尽管开发更复杂，但它们允许将硬件逻辑精确调整以适应LLM的计算图。这对于非标准操作或为特定模型结构实现极低延迟可能是有利的。映射包含使用Verilog或VHDL等硬件描述语言 (HDL)，或更高级的综合工具。
3. **定制ASICs：** 公司有时会开发完全定制的ASIC，针对其特定LLM工作负载进行优化（例如AWS Inferentia/Trainium、谷歌TPU）。这些ASIC提供最高的潜在性能和效率，但缺乏灵活性，并涉及显著的开发成本。映射与特定架构及其相关软件栈绑定。

这些加速器通常优先考虑特定方面，例如功耗效率（NPU）、低延迟（FPGA）或针对狭窄模型范围的最大吞吐量 (throughput)（定制ASIC）。

### 比较分析与考量

选择合适的硬件和优化映射严重依赖于特定的LLM和部署限制。



![LLM操作的硬件特性](plots/4404-0.json)



> 硬件类型的相对位置基于典型的峰值计算吞吐量 (throughput)和内存带宽，与大型模型推理 (inference)相关。具体数值随特定模型和代际显著变化。

影响映射效率的主要因素包含：

- **算术强度：** 如果硬件提供足够的内存带宽，GEMM等操作是计算受限的。如果注意力机制 (attention mechanism)未仔细实现，它可能是内存带宽受限的。硬件选择必须与特定模型层的瓶颈对齐 (alignment)。
- **数据类型：** 使用更低精度（FP16、BF16、INT8）对性能很重要。Tensor Cores或专用INT8引擎等硬件单元提供显著加速，但映射必须确保数据正确流经这些单元。
- **批量大小：** 较大批量大小通常能改善硬件利用率，特别是在GPU和TPU上，通过分摊核函数启动开销和更好地饱和并行单元。然而，延迟要求通常会限制实时应用中的批量大小。
- **内存容量：** LLM需要大量内存。映射策略必须考虑可用的HBM或DRAM容量，如果单个加速器的内存不足，可能包含模型并行等技术（稍后讨论）。

有效的映射不仅仅是使用硬件；它更是关于构造LLM的计算，使其与架构的优势最佳对齐，无论是GPU的庞大并行性、TPU的矩阵处理能力，还是NPU的专用效率。底层库（cuBLAS、MKL、OneDNN）和编译器（XLA、TVM、TensorRT）通过实现许多这些优化映射，将高级模型描述转换为高效的硬件专用代码，发挥着不可或缺的作用。

## 参考资料

- [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://doi.org/10.1145/3079856.3080246) — Norman P. Jouppi, Cliff Young, Nishant Patil, David Patterson, Gaurav Agrawal, et al. (2017)
  Journal: Proceedings of the 44th Annual International Symposium on Computer Architecture (ISCA); Publisher: ACM; Pages: 1-12; DOI: [10.1145/3079856.3080246](https://doi.org/10.1145/3079856.3080246)
  描述了谷歌第一代张量处理单元（TPU）的架构和性能，详细介绍了其用于神经网络加速的脉动阵列设计。
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (2022)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2205.14135](https://doi.org/10.48550/arXiv.2205.14135)
  提出了一种优化的自注意力算法，通过利用GPU的内存层次结构，减少内存I/O并提高效率。
- [NVIDIA Volta Architecture In-Depth](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf) — Stephen Jones and David B. Kirk (2017)
  Publisher: NVIDIA Corporation
  详细的技术白皮书，描述了NVIDIA Volta GPU架构，包括用于深度学习加速的Tensor Cores的引入和设计原理。
- [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) — Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy (2018)
  Journal: 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18); Publisher: USENIX Association; Pages: 578-594
  介绍了TVM，一个开源的深度学习编译器栈，它自动优化并为包括CPU、GPU和专用加速器在内的各种硬件后端生成代码，涵盖了软件映射方面。

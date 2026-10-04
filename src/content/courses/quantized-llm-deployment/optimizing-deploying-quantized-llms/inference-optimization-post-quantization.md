---
course: "quantized-llm-deployment"
chapter: "optimizing-deploying-quantized-llms"
lesson: "inference-optimization-post-quantization"
sourceId: 4647
sourceUrl: "https://apxml.com/zh/courses/quantized-llm-deployment/chapter-4-optimizing-deploying-quantized-llms/inference-optimization-post-quantization"
title: "量化后的推理优化技术"
description: "除了量化本身，还有其他技术可用于优化推理速度，例如核函数融合和高效注意力机制。"
order: 1
plots: []
sourceHash: "c2a17227dae027db057515f6d913f18e06da3acb9b5971a1c8e31c229294d8d7"
sourceCorrections: []
---

量化 (quantization)大型语言模型是实现高效推理 (inference)的重要一步，它能大幅减少内存占用，并通常加速计算，特别是在原生支持低精度算术的硬件上。然而，仅靠量化无法完全达到全部性能水平。模型量化后，可以在推理阶段应用进一步的优化技术，以最小化延迟、最大化吞吐量 (throughput)并改善资源使用。这些技术侧重于优化量化模型的*执行*，通常通过重构计算、使用专用硬件功能以及更有效地管理内存来实现。

几种重要的量化后优化策略可以与量化配合使用，以实现非常高效的LLM推理。

### 核函数融合

现代深度学习 (deep learning)模型，包括LLM，涉及一系列操作：矩阵乘法、逐元素加法、激活函数 (activation function)、归一化 (normalization)步骤等。独立执行这些操作的每一个都需要启动单独的计算核函数（在GPU或CPU上运行的程序），并在它们之间传输数据，通常通过相对较慢的高带宽内存（GPU上的HBM）。

核函数融合通过将多个连续操作组合成一个更大、单一的核函数来解决这种低效率问题。融合的核函数在存储于更快的局部内存（如GPU寄存器或共享内存）中的数据上连续执行多项操作，而不是从全局内存读取输入、执行一项操作、将中间结果写回全局内存，然后对下一个操作重复此过程。

思考Transformer块中常见的序列：一个矩阵乘法（例如，用于线性层），接着是添加偏置 (bias)向量 (vector)，然后应用ReLU或GeLU之类的激活函数。

> 一系列操作需要多次核函数启动和内存访问。

通过核函数融合，这些步骤被合并：

> 相同的操作组合成一个单一的融合核函数，减少开销和内存流量。

**优点：**

1. **内存带宽使用减少：** 中间结果保留在更快的片上内存中，最大程度地减少对全局内存的昂贵读写。这对LLM中常见的内存密集型操作尤其有益。
2. **核函数启动开销更低：** 启动核函数会产生较小的CPU开销。更少的启动次数意味着更少的开销，提高整体执行时间，特别是对于较小的计算。

量化 (quantization)模型从融合中获益匪多。诸如INT4/INT8矩阵乘法、反量化步骤（如果需要）、偏置加法和激活等操作是可融合的优选项。高级部署框架和编译器，如NVIDIA TensorRT-LLM，在为目标硬件上的量化模型识别和实现最佳融合模式方面发挥重要作用。

### 专用低位核函数

标准的深度学习 (deep learning)框架可能提供对INT8等低精度数据类型的基本支持，但要达到最大性能，特别是对于低于8位的格式（INT4、NF4、FP4等），通常需要高度专用的计算核函数。这些核函数针对特定的硬件架构进行了精心优化，利用NVIDIA Tensor Cores等功能，以降低的精度加速矩阵乘法。

创建这些核函数很复杂，需要熟知硬件架构、指令集和内存层次。诸如`bitsandbytes`之类的库提供优化的CUDA核函数，用于4位矩阵乘法和量化 (quantization)/反量化例程等操作，使Hugging Face Transformers等库能够很好地使用这些低位格式。同样，TensorRT-LLM等推理 (inference)引擎包含它们自己的高度优化核函数库，适用于各种量化方案和操作，并为NVIDIA GPU量身定制。

这些专用核函数的可用性和性能提升与底层硬件紧密关联。为NVIDIA Ampere或Hopper GPU优化的INT4核函数可能无法在旧GPU或其他硬件平台（CPU、其他加速器）上运行，或者运行效率低下。因此，在依赖涉及专用低位核函数的优化时，了解目标部署硬件十分必要。

### 高效注意力机制 (attention mechanism)

自注意力 (self-attention)机制虽然功能强大，却是LLM中的主要计算和内存瓶颈，特别是它在序列长度 $n$ 方面的 $O(n^2)$ 复杂度。因此，优化注意力对于高效推理 (inference)十分重要，特别是对于长上下文 (context)。已经出现了几种可以与量化 (quantization)一同应用的技术：

1. **FlashAttention（及其变体如FlashAttention-2）：** 该技术重新排序注意力计算，以最小化GPU高带宽内存（HBM）与其更快的片上SRAM之间的内存I/O。FlashAttention显著加速了注意力的前向和反向传播 (backpropagation)，并减少了内存使用，它通过分块（平铺）处理注意力计算，并重新计算中间结果（例如注意力softmax归一化 (normalization)因子），而不是存储大的中间注意力矩阵 ($N \times N$)。它被设计为与各种数据类型（包括FP16、BF16以及根据实现可能包含量化类型）兼容，使其与量化模型兼容。
2. **PagedAttention：** PagedAttention在vLLM推理服务器中得到了显著实现，它侧重于优化键值（KV）缓存的管理。PagedAttention将缓存划分为更小的固定大小块，类似于操作系统中的虚拟内存分页，而不是为KV缓存分配连续的内存块，后者可能导致碎片化和内存浪费。这允许更灵活的内存分配，大幅减少内部和外部碎片，导致更高的潜在批处理大小和吞吐量 (throughput)，特别是对于序列长度变化的情况。虽然它主要面向内存管理，但效率的提升使得计算资源能够更好地用于（可能已量化的）注意力计算本身。

这些优化的注意力实现通常直接集成到部署框架中（TensorRT-LLM、vLLM、TGI），并且可以比标准的注意力实现提供显著的加速，补充了从量化中获得的益处。

### 优化KV缓存管理

在自回归 (autoregressive)生成过程中，LLM需要在每个步骤访问所有先前token的已计算键和值（KV缓存）。这个缓存随着序列长度线性增长，并可能消耗大量内存，通常超过模型权重 (weight)本身的大小，特别是对于长上下文 (context)或大批次。虽然模型量化 (quantization)减少了权重内存，但KV缓存（通常以FP16或BF16存储）仍然是一个难题。

优化方法包括：

1. **KV缓存量化：** 不仅将量化（例如，到INT8）应用于模型权重和激活，还应用于KV缓存本身。这可以显著减少缓存的内存占用，在相同内存限制下允许更长的上下文或更大的批次。然而，量化KV缓存有时可能比权重/激活量化更显著地影响模型精度，需要仔细评估。框架可能会提供INT8 KV缓存的选项。
2. **高效内存管理（PagedAttention）：** 如前所述，PagedAttention直接解决KV缓存的内存分配和碎片化问题，允许接近最优的内存利用率，并实现更高的吞吐量 (throughput)。

### 图级别优化

除了优化单个核函数或操作，推理 (inference)引擎和编译器还对整个计算图进行优化：

1. **常量折叠：** 在编译时评估并替换图中仅依赖于常量输入（如模型权重 (weight)）的部分，而不是在运行时重复执行它们。这包括涉及量化 (quantization)权重和比例因子的操作。
2. **层/张量消除：** 识别并删除不影响最终输出的操作或层。
3. **布局优化：** 转置或重新排列张量内存布局（例如，NCHW与NHWC），以匹配硬件核函数预期的最佳格式。

这些图优化通常在准备部署模型时由ONNX Runtime等工具或TensorRT等编译器自动执行。它们简化了运行时执行的图，进一步减少开销。

### 硬件专属调优

最后一层优化通常涉及针对目标硬件的参数 (parameter)调优。融合核函数的最佳方式、矩阵乘核函数的最佳分块大小、最有效的数据布局，或理想的批处理大小在不同GPU代次（例如NVIDIA A100与H100）之间或CPU与GPU之间可能存在显著差异。

高级部署框架通常集成自动调优功能（如TensorRT-LLM的构建器），它们在目标硬件上对不同的核函数实现和配置进行基准测试，以找到性能最佳的设置。或者，它们可能公开参数，允许开发者根据经验测试手动调优性能。

总之，尽管量化 (quantization)提供了优化的一个基础层面，但要实现先进的推理 (inference)性能，需要应用这些量化后技术。核函数融合、专用低位核函数、高效注意力实现、优化的KV缓存管理、图转换和硬件专属调优，所有这些都有助于最小化延迟并最大化您部署的LLM的吞吐量 (throughput)。本章后续讨论的部署框架通常集成了其中许多技术，从而简化了它们的应用。

## 参考资料

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Aware Operators](https://arxiv.org/abs/2205.14135) — Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2205.14135](https://doi.org/10.48550/arXiv.2205.14135)
  介绍了FlashAttention算法，通过重新排序计算和最小化内存I/O，显著提高了注意力机制的速度和内存效率，对LLM尤其重要。
- [vLLM: Universal and Efficient Engine for Large Language Model Inference](https://arxiv.org/abs/2309.06180) — Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica (2023)
  Journal: SOSP 2023; DOI: [10.48550/arXiv.2309.06180](https://doi.org/10.48550/arXiv.2309.06180)
  介绍了PagedAttention，一种关键的内存管理技术，可优化KV缓存使用，从而在LLM推理中实现更高的吞吐量并减少内存碎片。
- [QLoRA: Efficient Finetuning of Quantized LLMs on Consumer GPUs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: arXiv preprint arXiv:2305.14314; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  详细介绍了QLoRA方法，该方法依赖4位NormalFloat (NF4) 量化，并包含通过`bitsandbytes`等库实现的优化CUDA内核，用于量化模型的有效计算。
- [NVIDIA TensorRT Documentation](https://developer.nvidia.com/tensorrt) — NVIDIA Corporation (2024)
  Publisher: NVIDIA
  提供了TensorRT中使用的优化技术信息，包括内核融合、图优化和硬件专用调整，这些技术应用于量化模型以实现高效部署。

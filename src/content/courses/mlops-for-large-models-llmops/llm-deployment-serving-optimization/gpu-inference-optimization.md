---
course: "mlops-for-large-models-llmops"
chapter: "llm-deployment-serving-optimization"
lesson: "gpu-inference-optimization"
sourceId: 3169
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/gpu-inference-optimization"
title: "GPU 推理服务器优化"
description: "使用 NVIDIA Triton、TensorRT-LLM 或 vLLM 等工具，提升推理时的 GPU 使用效率。"
order: 3
plots: []
sourceHash: "e0061ebe6f609bda5c71b1b84ffe46b5b92cc2b678306743ced1a9f6fc58c992"
sourceCorrections: []
---

与较小模型相比，高效服务大型语言模型面临独特问题。主要目标是在满足严格的延迟和吞吐量 (throughput)要求的同时，充分发挥昂贵的GPU资源的作用。简单地将模型加载到GPU并按顺序处理请求，通常会导致资源利用率低下，因为现代GPU的计算能力远超单个推理 (inference)请求的需求，尤其是在逐令牌生成过程中。专门的GPU推理服务器和优化库旨在解决此不足。

这些工具采用多种高级技术，以从底层硬件中实现最佳性能。下面我们分析最有效的策略：

### 批处理策略

同时处理多个推理 (inference)请求是提升GPU使用率的基本方法。

- **静态批处理**: 最简单的方法是等待预设数量的请求（`batch_size`）到达或超时发生，然后将它们一起处理。虽然易于实现，但这会引入延迟（等待批次填满），并且如果请求到达速率波动较大，可能会效率低下。
- **动态批处理 / 连续批处理**: 更复杂的服务器实现动态或连续批处理。请求到达时被添加到队列中。服务器持续监控队列，并根据可用GPU容量和调度策略动态形成批次，通常同时处理来自多个序列的迭代（单令牌步骤）。与静态批处理相比，这大幅提高了吞吐量 (throughput)和GPU使用率，特别是对于序列在不同时间完成的自回归 (autoregressive)模型。像NVIDIA Triton Inference Server这样的框架和像vLLM这样的引擎都实现了这种技术的变体。

> 静态批处理需要等待完整批次，可能增加延迟；而连续批处理更动态地处理来自排队请求的迭代，提升了吞吐量。

### 内核优化与融合

LLM推理 (inference)涉及大量小型计算（矩阵乘法、加法、激活函数 (activation function)）。为每个操作启动单独的GPU内核会产生显著开销。

- **操作融合**: 像TensorRT-LLM这样的推理优化库会分析模型的计算图，并将多个连续操作融合成一个更大、单独的GPU内核。这减少了内核启动次数，并最大限度地减少了GPU内存（慢）和计算核心（快）之间的数据移动。例如，矩阵乘法、偏置 (bias)加法和激活函数应用可能会被融合。
- **优化内核**: 这些库通常提供核心操作（如矩阵乘法和注意力机制 (attention mechanism)，例如FlashAttention、融合注意力内核）的高度优化实现，专门针对不同的GPU架构和数据类型（FP16、INT8）进行调整。

### KV缓存管理

自回归 (autoregressive)LLM很大程度上依赖于键值（KV）缓存来存储先前生成令牌的注意力键和值，从而避免冗余计算。此缓存会占用大量GPU内存，常成为批处理大小和序列长度的主要瓶颈。

- **简单分配**: 简单的方法可能会为批处理中的每个请求预先分配最大可能序列长度的连续内存块。这会导致严重的内部碎片（块内未使用的已分配内存）和外部碎片（块间不可用的空闲内存），从而限制了有效批处理大小。
- **PagedAttention**: 由vLLM引入，该技术使用类似操作系统中虚拟内存分页的原理来管理KV缓存。用于缓存的GPU内存被划分为固定大小的块（页）。对应序列KV缓存的逻辑块被映射到非连续的物理块。这大幅减少了碎片，允许更大的有效批处理大小和更高效的内存共享。它使请求能够共享公共前缀的缓存块（例如，在并行采样期间）。

> PagedAttention避免了大量连续分配，与简单方法相比减少了内存碎片，并支持更大的批处理大小。

### 常用推理 (inference)服务方案

Several frameworks and libraries specialize in optimizing LLM inference on GPUs:

- **NVIDIA Triton 推理服务器**: 一个多功能、开源服务器，用于部署来自不同框架（TensorFlow、PyTorch、ONNX、TensorRT）的模型。
  - **优点**: 支持动态批处理、模型并发执行（在单个GPU上运行多个模型或实例）、模型集成和流式输出。通过自定义后端实现高度可扩展。
  - **LLM应用场景**: 常与TensorRT-LLM结合使用，通过其后端实现优化LLM执行。提供生产级别功能，如指标、健康检查和管理API。
- **TensorRT-LLM**: NVIDIA的开源库，专门用于优化LLM推理性能。
  - **优点**: 提供业界先进的内核实现（注意力、GEMM）、操作融合、INT8/FP8量化 (quantization)流程、飞行中批处理（其对连续批处理的称呼），以及推理模型并行（张量并行）支持。可与PyTorch等框架集成。
  - **LLM应用场景**: 将LLM编译成高度优化的引擎。可独立使用，或更常见地作为Triton内部的后端进行部署。需要模型编译步骤。
- **vLLM**: 一个专注于高吞吐量 (throughput)LLM推理的开源库和服务引擎。
  - **优点**: 其主要创新是PagedAttention，用于高效的KV缓存管理。支持连续批处理和优化的CUDA内核。与专业性较低的服务方案相比，尤其是在序列长度变化的负载下，通常展现出更高的吞吐量增益。
  - **LLM应用场景**: 专为LLM设计，提供Python风格的接口并直接与Hugging Face模型集成。可用作独立服务器或库。

### 选择考量

选择合适的推理 (inference)服务器或优化库涉及权衡：

1. **性能目标**: 优先级是最大吞吐量 (throughput)（每秒请求数）还是每个请求的最小延迟？vLLM因PagedAttention通常在吞吐量方面表现出色，而TensorRT-LLM则对内核优化提供精细控制，这有助于降低延迟。
2. **易用性与控制性**: 像vLLM这样的框架为常用LLM提供更简单的接口。Triton+TensorRT-LLM可能需要更多配置和编译步骤，但能对优化过程提供更大的灵活性和控制力。
3. **模型/功能支持**: 检查与所需LLM架构、量化 (quantization)技术（INT8、FP8）和高级功能（例如并行采样、模型并行）的兼容性。如果同时部署非LLM模型和LLM，Triton提供更广泛的框架支持。
4. **生态系统集成**: Triton已成熟，并能良好集成到Kubernetes、监控工具（Prometheus）和编排平台。新的解决方案正在快速构建其生态。

优化GPU推理服务器并非一次性设置。它需要使用代表性负载进行仔细基准测试、调整最大批处理大小和GPU内存分配等参数 (parameter)，并持续监控性能指标（延迟、吞吐量、GPU使用率），正如下一章所讨论的。动态批处理、内核融合和高级KV缓存管理等技术是构建经济高效且响应迅速的LLM应用程序的重要工具。

## 参考资料

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 35; DOI: [10.48550/arXiv.2205.14135](https://doi.org/10.48550/arXiv.2205.14135)
  提出了一种 IO 感知的注意力算法和高度优化的 CUDA 内核，可减少注意力机制的内存访问和计算开销。
- [NVIDIA Triton Inference Server Documentation](https://developer.nvidia.com/nvidia-triton-inference-server) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  NVIDIA 开源推理服务软件的官方文档，详细介绍了其动态批处理、并发模型执行和可扩展性等功能。
- [NVIDIA TensorRT-LLM Documentation](https://docs.nvidia.com/deeplearning/tensorrt-llm/user-guide/index.html) — NVIDIA Corporation (2025)
  Publisher: NVIDIA
  NVIDIA 用于优化大型语言模型推理的库的官方文档，涵盖内核优化、量化和飞行中批处理。

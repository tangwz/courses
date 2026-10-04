---
course: "quantized-llm-deployment"
chapter: "performance-evaluation-quantized-llms"
lesson: "analyzing-performance-target-hardware"
sourceId: 4640
sourceUrl: "https://apxml.com/zh/courses/quantized-llm-deployment/chapter-3-performance-evaluation-quantized-llms/analyzing-performance-target-hardware"
title: "目标硬件上的性能分析"
description: "了解量化性能在不同硬件平台（CPU、GPU）上如何变化。"
order: 6
plots: ["plots/4640-0.json"]
sourceHash: "4a9fe9564cbef52edec2e126bd2d1f26f08e95e0b874dc51ab6c82bd5f77d717"
sourceCorrections: []
---

量化 (quantization)有望在模型大小和计算效率方面带来显著提升，但以延迟和吞吐量 (throughput)衡量的实际性能提升，高度依赖于执行推理 (inference)的底层硬件。一个量化到INT4的模型可能在理论上显示出大幅减少的操作，但却只能在缺少优化的INT4计算核的硬件上带来微小的加速。因此，在*目标硬件上*专门分析性能，不仅值得推荐，更是评估流程的一个基本部分。

不同的硬件平台，主要是CPU和GPU，具有不同的架构和指令集，它们与量化数据类型和操作有不同的交互方式。掌握这些交互方式，对预测和解读性能结果非常重要。

### CPU推理 (inference)性能

虽然高性能LLM推理通常依赖于GPU，但CPU在某些部署场景中仍有其用处，例如边缘设备、开发环境或成本敏感型应用。然而，与GPU相比，CPU从量化 (quantization)获得的加速效果通常较为有限。

- **指令集支持：** 现代CPU常包含单指令多数据（SIMD）扩展，如AVX2或AVX-512。一些较新的处理器世代引入专门指令以加速低精度整数运算，例如AVX-512 VNNI（向量 (vector)神经网络 (neural network)指令），这大幅提高了INT8性能。INT8量化能借助这些指令，从而带来明显的延迟降低。然而，在硬件指令中，对低于8位的整数操作（如INT4）的支持不太常见，这意味着性能提升主要来自内存带宽使用的减少，而非计算速度的加快。
- **内存带宽与缓存：** LLM推理可能受内存带宽限制，尤其是在CPU上。量化减少了模型大小，从而更好地利用CPU缓存（L1、L2、L3）并降低主内存总线压力。即使没有专门的计算指令，这种效果也能促进加速，特别是对于较低的位宽。
- **并行性：** 与GPU相比，CPU的并行执行核心数量远少。尽管多线程等技术有所帮助，但模型某些部分固有的序列性以及核心数量的限制，制约了可达到的吞吐量 (throughput)，即使进行了量化。
- **软件库：** 性能严重依赖于用于推理的软件库（例如ONNX Runtime、Intel oneDNN、使用特定后端的PyTorch）。这些库包含针对各种操作和数据类型的优化计算核。这些计算核针对特定CPU架构和量化方案（例如INT8 vs. INT4，对称 vs. 非对称）的可用性和质量，直接影响所观察到的加速效果。

在CPU上进行基准测试时，请密切关注具体的CPU型号、其支持的指令集以及推理库的配置。这些因素的微小变化可能导致性能出现明显差异。

### GPU推理 (inference)性能

GPU是LLM推理的主力，因其大规模并行架构，专为深度学习 (deep learning)中普遍存在的矩阵乘法和逐元素操作而设计。

- **专门核心（例如Tensor Cores）：** 现代GPU（尤其是NVIDIA的）具有专门的硬件单元，如Tensor Cores，旨在加速特定精度的矩阵乘法。这些核心为FP16、TF32和INT8算术提供显著的性能提升。较新的架构（如Hopper）引入对更低精度的支持，如FP8。借助这些核心通常是GPU上量化 (quantization)加速的主要因素。如果量化方案（例如INT4）需要模拟或无法高效映射到这些核心，计算加速效果可能受限，尽管内存节省依然有效。
- **CUDA计算核与库：** GPU性能由底层CUDA计算核的效率决定。cuBLAS、cuDNN等库以及专门的LLM推理库，如NVIDIA TensorRT-LLM或vLLM，包含针对不同操作、数据类型和GPU架构的高度优化计算核。这些库中针对特定量化格式（例如GPTQ INT4、AWQ INT4）的优化程度决定了可实现的性能。特定低位格式在特定GPU架构上缺少优化计算核可能是一个主要的瓶颈。
- **内存带宽：** 与CPU一样，GPU对内存带宽也很敏感。大型LLM很容易使GPU的内存总线（HBM或GDDR）饱和。量化大幅减少了GPU主内存与其计算单元之间传输的数据量，缓解了这一瓶颈，并为更快的推理贡献很大，特别是对于大型模型或长序列。
- **架构世代：** 性能特性在不同GPU世代之间（例如NVIDIA的Pascal、Volta、Turing、Ampere、Hopper、Blackwell）差异明显。较新的世代通常提供更好的支持和更高的性能，以处理低精度格式。一个世代的基准测试结果可能无法直接适用于另一代。

### 跨硬件的基准测试

由于性能与硬件特性紧密相关，因此直接比较非常必要。考虑对您的基线模型（例如FP16或BF16）以及各种量化 (quantization)版本（例如INT8、使用GPTQ/AWQ的INT4）在您潜在的目标硬件平台之间进行基准测试。

在硬件特定基准测试期间需要控制和观察的主要因素包括：

- **硬件：** 具体的CPU型号、GPU型号（如A100、H100、L40S、RTX 4090）。
- **软件：** 推理 (inference)框架（TensorRT-LLM、vLLM、TGI、ONNX Runtime）、CUDA版本、驱动版本、量化库版本。
- **工作负载：** 批处理大小、输入序列长度、输出序列长度。
- **指标：** 延迟（每 token、端到端）、吞吐量 (throughput)（tokens/秒）、内存使用量（GPU/CPU峰值内存）、功耗（如适用）。

以下图表显示了在不同硬件和量化级别下，给定LLM任务的延迟变化。



![延迟（毫秒）对比量化与硬件](plots/4640-0.json)



> 对比显示推理延迟随更激进的量化（FP16 > INT8 > INT4）而降低，并且GPU上的延迟明显低于CPU。注意到INT4相对于INT8的收益递减，可能由于计算核的优化水平，特别是在中端GPU上。高端GPU因出色的硬件支持而表现出最大的提升。

归根结底，理论上的好处必须通过实验验证。目标硬件上的性能分析为决定哪种量化策略能在效率提升和潜在精度下降之间提供最佳权衡，针对您具体的部署场景提供真实依据。仅依赖不同硬件或配置的报告基准测试结果，可能导致不准确的预期和次优的部署选择。

## 参考资料

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2022)
  Journal: International Conference on Learning Representations (ICLR) 2023; DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  本文介绍了GPTQ，一种针对LLM的训练后量化方法，特别针对INT4等低位宽，与本节中关于INT4量化及其性能影响的讨论相关。
- [AWQ: Activation-aware Weight Quantization for LLM Inference](https://arxiv.org/abs/2306.00978) — Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (2023)
  Journal: arXiv preprint arXiv:2306.00978; DOI: [10.48550/arXiv.2306.00978](https://doi.org/10.48550/arXiv.2306.00978)
  本文介绍了AWQ，另一种针对LLM的权重量化技术，为实现目标硬件上的高效低位推理提供了与GPTQ不同的视角和方法。
- [NVIDIA TensorRT-LLM User Guide](https://docs.nvidia.com/deeplearning/tensorrt-llm/user-guide/index.html) — NVIDIA Corporation (2025)
  Publisher: NVIDIA
  此官方文档详细介绍了如何使用TensorRT-LLM在NVIDIA GPU上优化和部署LLM，涵盖了内核优化、各种量化格式以及硬件特定的性能调优等方面。
- [Intel oneAPI Deep Neural Network Library (oneDNN) Documentation](https://onednn.io/) — Intel Corporation (2024)
  Publisher: Intel Corporation
  此文档详细介绍了用于Intel CPU的高度优化的开源深度学习库，解释了它如何利用指令集扩展（如AVX-512 VNNI）来实现高效的深度学习推理，包括量化操作。

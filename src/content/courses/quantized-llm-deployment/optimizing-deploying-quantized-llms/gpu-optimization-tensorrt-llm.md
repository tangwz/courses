---
course: "quantized-llm-deployment"
chapter: "optimizing-deploying-quantized-llms"
lesson: "gpu-optimization-tensorrt-llm"
sourceId: 4655
sourceUrl: "https://apxml.com/zh/courses/quantized-llm-deployment/chapter-4-optimizing-deploying-quantized-llms/gpu-optimization-tensorrt-llm"
title: "使用 NVIDIA TensorRT-LLM 进行 GPU 优化"
description: "使用 TensorRT-LLM 优化和部署 NVIDIA GPU 上的量化 LLM，以获取最佳性能。"
order: 5
plots: ["plots/4655-0.json"]
sourceHash: "77e8e89f132a817a5646a0dcfc7737dc75ecf8f97e37b276b91c1b61a87c2920"
sourceCorrections: []
---

通用部署框架虽然灵活，但在 NVIDIA GPU 上为量化 (quantization)大型语言模型（LLM）争取最佳性能时，通常需要专门优化。NVIDIA 的 TensorRT-LLM 是为此目标设计的库，它通过将模型编译成高度优化的运行时引擎，显著提升推理 (inference)速度和效率。

### TensorRT-LLM 的作用

TensorRT-LLM 充当 LLM 的深度学习 (deep learning)编译器和运行时，面向 NVIDIA GPU。它接收模型定义（可能已使用之前讨论的技术，如 INT8 PTQ、GPTQ 或 AWQ 进行量化 (quantization)），并在生成 TensorRT 引擎前应用一系列优化。此引擎是模型的高度优化版本，可供部署。

TensorRT-LLM 的重要特点包括：

1. **图优化**：它执行层和张量融合，将多个操作合并为单一核。这降低了核启动开销并改善了内存访问模式，在 GPU 上尤其有利。
2. **精度校准**：TensorRT 可以分析模型并为不同层确定最佳精度水平，包括在硬件支持的情况下使用专门的低精度核。它内置支持 INT8 和 FP8 等格式，在正确校准时，能显著提升性能，同时对精度影响很小。
3. **核自动调优**：TensorRT-LLM 从高度优化的实现库中，针对特定的模型架构和目标 GPU，选择性能最佳的核。
4. **优化组件**：它包括核心 LLM 构成部分的高度优化实现，例如多头注意力 (multi-head attention)（包括 FlashAttention 及其变体）和位置嵌入 (embedding)的版本。
5. **动态批处理**：类似于 vLLM 等框架，TensorRT-LLM 采纳了先进的批处理策略（连续或动态批处理），通过并行处理多个请求而无需填充来最大化 GPU 利用率，大幅提升吞吐量 (throughput)。

### 将量化 (quantization)与 TensorRT-LLM 结合

TensorRT-LLM 旨在有效处理量化模型。它支持多种量化格式，包括：

- **INT8 量化**：训练后量化（PTQ）和量化感知训练（QAT）的 INT8 模型都可以进行优化。TensorRT-LLM 包括 SmoothQuant 等技术，以处理 INT8 量化过程中 LLM 中常出现的激活值异常。
- **FP8 量化**：对于较新的硬件（如 H100 GPU），TensorRT-LLM 提供对 FP8（E4M3 和 E5M2 格式）的支持，提供接近 FP16 的精度，同时显著减少内存占用并加快计算速度。
- **INT4 量化**：支持 INT4 量化方案，通常基于 AWQ（激活感知权重 (weight)Q量化）等方法，也已集成，可以在进行大幅度模型压缩的同时，使用专用核来提升性能。

工作流程通常包括：

1. **模型准备**：您从一个预训练 (pre-training)的 LLM 开始，可能已使用 AutoGPTQ 或 AutoAWQ 等库进行量化，或者，如果您打算使用 TensorRT 的 PTQ 功能，则使用全精度模型。
2. **引擎构建**：使用 TensorRT-LLM Python API 或 `trtllm-build` 命令行工具，您可以定义模型架构并指定优化参数 (parameter)（如目标精度、量化模式、插件配置）。TensorRT-LLM 会将模型编译成优化后的引擎文件（`.engine` 或 `.plan`）。此步骤可能耗时，因为它涉及核选择和调优。
3. **运行时执行**：生成的引擎由 TensorRT 运行时加载。您可以使用 TensorRT-LLM 运行时 API（或将其与 NVIDIA Triton Inference Server 等推理 (inference)服务器集成）进行推理，从引擎中固有的优化中受益。

### 性能提升

使用 TensorRT-LLM 的主要目的是性能。通过针对特定 GPU 架构深度优化模型并使用较低精度，与在 PyTorch 或 TensorFlow 等通用框架中运行相同的量化 (quantization)模型相比，它可以显著提升延迟和吞吐量 (throughput)。



![量化 LLM 性能（GPU）](plots/4655-0.json)



> 性能对比，表明使用 TensorRT-LLM 与标准框架相比，针对 GPU 上的 INT8 量化模型可能获得的优势。请注意 Y 轴采用对数刻度。实际结果很大程度上取决于模型、硬件和具体的量化方法。

### 与部署系统集成

TensorRT-LLM 引擎通常通过支持 TensorRT 后端的推理 (inference)服务器进行部署。

- **NVIDIA Triton Inference Server**：这是一个常用选择，提供生产就绪环境来服务 TensorRT 引擎。Triton 处理请求批处理、模型版本控制，并暴露标准推理协议（HTTP/gRPC）。TensorRT-LLM 提供专门的 Triton 后端，旨在高效服务 LLM，并包含动态批处理等功能。
- **独立运行时**：您也可以直接在应用程序中使用 TensorRT-LLM 运行时，以实现更紧密的集成，尽管这需要更多手动设置来处理请求和批处理。

### 考量与权衡

TensorRT-LLM 尽管功能强大，但使用时涉及一些考量：

- **构建时间**：编译 TensorRT 引擎，特别是对于大型模型和进行广泛的核调优时，可能需要大量时间（数分钟到数小时）。
- **硬件特定性**：TensorRT 引擎通常针对特定的 NVIDIA GPU 架构和 TensorRT 版本进行优化。为 A100 GPU 构建的引擎在 H100 GPU 或不同的驱动/CUDA 版本上可能无法最佳运行，甚至无法运行，需要为不同的部署目标重新构建。
- **灵活性**：与 PyTorch 等框架中的动态执行相比，如果您需要频繁修改模型图结构，编译后的 TensorRT 引擎灵活性较低。
- **复杂性**：构建过程和 API 可能比使用更高层次的抽象库更复杂。

TensorRT-LLM 代表了一种先进的方法，用于最大化 NVIDIA GPU 上量化 (quantization)（和非量化）LLM 的性能。当需要绝对最低延迟或最高吞吐量 (throughput)时，在构建和部署 TensorRT-LLM 引擎上的投入通常会带来可观的回报，使其成为生产级 LLM 推理 (inference)服务的重要工具。

## 参考资料

- [NVIDIA TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/) — NVIDIA (2024)
  TensorRT-LLM官方文档，详细介绍了其功能、用法和用于部署大语言模型的API。
- [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) — NVIDIA (2024)
  NVIDIA TensorRT官方文档，提供深度学习优化器和运行时的核心概念。
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) — Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han (2023)
  Journal: ICML 2023; DOI: [10.48550/arXiv.2211.10438](https://doi.org/10.48550/arXiv.2211.10438)
  介绍了SmoothQuant，一种针对大语言模型的有效后训练量化方法，TensorRT-LLM支持此方法。
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 35; DOI: [10.48550/arXiv.2205.14135](https://doi.org/10.48550/arXiv.2205.14135)
  介绍了FlashAttention，一种快速且内存高效的注意力机制，作为优化组件集成到TensorRT-LLM中。
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (2023)
  Journal: arXiv preprint arXiv:2306.00978; DOI: [10.48550/arXiv.2306.00978](https://doi.org/10.48550/arXiv.2306.00978)
  描述了AWQ，一种用于大语言模型压缩和加速的激活感知权重量化方法，TensorRT-LLM支持此方法。

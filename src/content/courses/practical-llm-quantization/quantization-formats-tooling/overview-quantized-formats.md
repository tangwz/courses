---
course: "practical-llm-quantization"
chapter: "quantization-formats-tooling"
lesson: "overview-quantized-formats"
sourceId: 4652
sourceUrl: "https://apxml.com/zh/courses/practical-llm-quantization/chapter-5-quantization-formats-tooling/overview-quantized-formats"
title: "常见量化模型格式概述"
description: "介绍存储和加载量化模型所需特定文件格式的缘由。"
order: 1
plots: []
sourceHash: "d56d3f998bc4584ffef4317e1a48de38156a8e940e1370f707af9d471566e887"
sourceCorrections: []
---

对模型进行量化 (quantization)会改变其核心表示形式。它涉及将权重 (weight)和激活值存储为低精度类型，例如8位整数（INT8）或4位整数（INT4），而不是标准的32位浮点数（FP32）。仅仅将这些低精度权重保存到标准文件中是不够的。与量化过程本身相关的重要元数据也需要被存储。

思考量化过程中发生的情况：我们将原始浮点范围映射到更小的整数范围。这种映射需要以下参数 (parameter)：

- **缩放因子（Scaling Factors）：** 在计算过程中，用于将量化后的整数转换回近似的浮点值。
- **零点（Zero-Points）：** 一种偏移量，特别是在非对称量化中用于将整数零点与浮点零点对齐 (alignment)。
- **量化方案（Quantization Scheme）：** 关于量化是对称还是非对称、按张量、按通道还是按组进行的信息。
- **数据类型（Data Type）：** 所使用的具体低精度格式（例如，INT4，INT8，NF4）。

标准模型序列化格式，通常为FP32或FP16模型设计，没有专门的结构来高效地存储和检索与量化权重一起的这些元数据。试图对它们进行改造可能会很麻烦且效率低下。

这导致需要**专门的量化模型格式**。这些格式专门设计用于将低精度权重与所有必要的量化参数，有时甚至是模型架构细节或配置设置一起打包。推动这些格式开发的主要目标包括：

1. **紧凑性：** 与存储原始模型相比，甚至与天真地存储量化权重相比，减少磁盘文件大小。
2. **加载效率：** 实现模型快速加载到内存中进行推理 (inference)，通常涉及内存映射技术。
3. **执行效率：** 以一种与推理引擎使用的优化低精度计算核良好对齐的方式组织数据。
4. **互操作性：** 为使用特定工具量化的模型提供一种标准方式，使其能够被兼容的推理库或硬件后端使用。

在本章中，我们将审视在处理量化LLM时会遇到的一些最常见的格式：

- **GGUF（Georgi Gerganov Universal Format）：** 一种主要与`llama.cpp`生态系统相关的灵活格式。它设计为可扩展，可存储丰富的元数据，支持各种量化类型和模型架构。我们将在后面详细介绍其结构和使用方法。
- **GPTQ模型约定（GPTQ Model Conventions）：** 尽管GPTQ是一种量化*算法*，但使用它量化的模型通常遵循特定的保存约定，通常将权重和量化参数（缩放因子、零点、组大小、置换顺序）存储在单独的文件中或诸如Hugging Face的`safetensors`等标准格式中，并依赖关联的配置文件。`AutoGPTQ`或`ExLlama`等库知道如何解释这些约定。
- **AWQ模型约定（AWQ Model Conventions）：** 与GPTQ类似，AWQ（Activation-aware Weight Quantization）是一种方法。用AWQ量化的模型也需要与权重一起保存特定参数，通常使用已建立的序列化格式结合支持库所能理解的配置文件。

重要的是要区分*量化算法*（如PTQ、GPTQ、AWQ、QAT），它定义了模型权重和/或激活值如何转换为更低精度，以及*文件格式*，它定义了结果量化模型及其元数据如何存储在磁盘上。虽然某些格式与特定算法紧密相关（如GPTQ使用的约定），但像GGUF这样的格式则旨在实现更宽的适用范围。

下图概述了这些组件之间的关系：

> 此流程图展示了量化算法如何生成权重和参数，这些权重和参数随后以专门的文件格式存储，供推理引擎使用。

理解这些格式对于实际部署很重要。它们连接了量化的理论过程与所得紧凑模型的高效运行。以下部分将提供关于GGUF、与GPTQ相关的约定以及AWQ格式的更多具体信息，同时介绍用于创建和操作它们的工具。

## 参考资料

- [GGUF (Georgi Gerganov Universal Format) Specification](https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md) — Georgi Gerganov and the llama.cpp Contributors (2023)
  GGUF格式的官方技术规范，这是一种广泛用于存储量化大型语言模型及其元数据的专用文件格式。
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2023)
  Journal: ICLR 2023; DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  描述GPTQ量化算法的原始研究论文，该算法是本节中讨论的GPTQ量化模型保存约定的基础。
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (2023)
  Journal: MLSys 2024; DOI: [10.48550/arXiv.2306.00978](https://doi.org/10.48550/arXiv.2306.00978)
  提出了AWQ量化方法的原始研究论文，其中概述了与其量化模型格式相关的参数和约定。

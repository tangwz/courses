---
course: "quantized-llm-deployment"
chapter: "optimizing-deploying-quantized-llms"
lesson: "choosing-deployment-framework"
sourceId: 4649
sourceUrl: "https://apxml.com/zh/courses/quantized-llm-deployment/chapter-4-optimizing-deploying-quantized-llms/choosing-deployment-framework"
title: "选择合适的部署框架"
description: "适用于量化LLM的部署框架对比：TGI、vLLM、TensorRT-LLM、ONNX Runtime。"
order: 2
plots: []
sourceHash: "a45a3090e0e074b37a6f0fc8257391ba736dd89a7f7e58f181fe3624fe7dd0ec"
sourceCorrections: []
---

高效部署量化 (quantization)后的深度语言模型需要选择最合适的部署框架。该框架作为引擎，负责加载量化模型、管理接收到的请求、执行推理 (inference)并返回结果。选择的框架会显著影响吞吐量 (throughput)和延迟等性能特点、硬件利用率、集成便捷性和可扩展性。有多种性能强的框架专为LLM服务设计，每种都有其独特的优势，尤其是在处理量化模型时。根据具体要求、目标硬件和不同的量化方法来评估这些方案，是必要的步骤。

### 框架选择中的考量因素

评估量化 (quantization)LLM的部署框架时，请考量以下方面：

1. **量化支持：** 该框架与不同量化格式及库的兼容性如何？它是否原生支持低位数据类型（例如INT4、FP8）和流行量化算法（GPTQ、AWQ）？或者它是否需要外部库或特定的模型转换步骤？
2. **性能：** 量化模型在您的目标硬件上，典型的吞吐量 (throughput)（每秒请求数）和延迟（每个请求的时间）指标如何？该框架是否包含连续批处理、分页注意力或自定义内核融合等对量化推理 (inference)有益的优化？
3. **硬件兼容性：** 该框架是否针对特定硬件（例如NVIDIA GPU）进行优化，还是它支持更广泛的兼容性，可跨CPU、不同GPU供应商甚至边缘设备运行？
4. **易用性和集成性：** 设置和配置的复杂程度如何？使用框架API部署量化模型是否简单？文档和社区支持的水平如何？
5. **功能集：** 该框架是否提供对您的应用而言必要的特定功能，例如对多适配器、多模态 (multimodal)输入的支持，或与特定MLOps生态系统的集成？

接下来，我们从部署量化LLM的角度，查看一些知名框架。

### 文本生成推理 (inference) (TGI)

由Hugging Face开发的文本生成推理（TGI）是一个专门构建的解决方案，用于大规模部署LLM。它旨在实现高吞吐量 (throughput)和低延迟，包含张量并行、连续批处理和优化过的Transformer内核等功能。

- **量化 (quantization)支持：** TGI可方便地与Hugging Face `transformers` 库支持的量化格式集成。这包括直接从Hugging Face Hub或本地路径加载使用`bitsandbytes`（NF4、INT8）、GPTQ和AWQ量化的模型，通常只需进行少量配置更改即可。
- **性能：** TGI开箱即用表现良好，尤其是在利用连续批处理时。其优化通常针对FP16/BF16，但量化模型从高效的请求处理和批处理方式中显著获益。
- **硬件兼容性：** 主要关注NVIDIA GPU，但社区的努力和底层库的支持可能会拓宽兼容性。
- **易用性：** 设置相对简单，尤其适用于托管在Hugging Face Hub上的模型。部署通常涉及运行带有特定模型ID和量化参数 (parameter)的Docker容器。
- **使用场景：** 对于大量使用Hugging Face生态系统、需要一个可用于生产的服务器、并良好支持常见量化方法（GPTQ、AWQ、`bitsandbytes`）且无需大量手动优化的用户而言，这是一个很好的选择。

### vLLM

vLLM是一个专门为快速LLM推理 (inference)和提供服务设计的开源库。其主要创新是PagedAttention，这是一种受操作系统虚拟内存和分页理念启发的注意力算法，通过减少内存浪费，显著提高了吞吐量 (throughput)。

- **量化 (quantization)支持：** vLLM原生支持AWQ等流行量化方案。对GPTQ或`bitsandbytes`等其他方法的支持可能需要特定版本或集成，并且该领域在不断变化。它在量化带来的内存效率与PagedAttention的内存管理相结合时表现出色。
- **性能：** 由于PagedAttention和连续批处理，vLLM常能实现行业领先的吞吐量，尤其是在高并发场景下。它在最大化GPU利用率方面效果显著。
- **硬件兼容性：** 主要面向NVIDIA GPU（CUDA）。
- **易用性：** 提供用于离线批处理推理的Python风格API，以及用于在线服务的OpenAI兼容服务器。如果不使用原生支持的量化格式，集成可能会比TGI稍微复杂些。
- **使用场景：** 当最大化NVIDIA GPU上的吞吐量是首要任务时，尤其是对于涉及动态批处理大小或长上下文 (context)的应用，它是理想选择。如果使用AWQ量化，它会是一个非常好的选择。

### NVIDIA TensorRT-LLM

TensorRT-LLM是NVIDIA推出的一个开源库，构建在NVIDIA TensorRT之上。它专注于将LLM编译成专门用于NVIDIA GPU的高度优化的运行时引擎，通过内核融合、层优化以及各种精度下的优化实现来提升推理 (inference)性能。

- **量化 (quantization)支持：** 直接在TensorRT框架内为各种量化格式（包括INT8（SmoothQuant，逐张量/逐通道）、INT4（AWQ、GPTQ实现）甚至FP8）提供出色支持。它允许对精度设置进行细致控制。
- **性能：** 由于深度图优化和针对硬件架构定制的专用内核，它通常在NVIDIA GPU上提供最高性能（最低延迟、最高吞吐量 (throughput)）。量化模型从这些硬件特定优化中显著获益。
- **硬件兼容性：** 仅限NVIDIA GPU。需要特定的构建过程来为目标GPU架构创建优化引擎。
- **易用性：** 与TGI或vLLM相比，学习曲线更陡。需要模型编译步骤（“引擎构建”），这可能耗时并需要细致配置。部署涉及使用已构建的引擎，通常与NVIDIA的Triton Inference Server集成。
- **使用场景：** 当您希望在支持的NVIDIA硬件上获得最高推理性能，并愿意在模型编译和部署流程中投入精力时，这是最佳选择。对于利用FP8等先进低精度格式或优化过的INT4/INT8实现，它非常必要。

### ONNX Runtime

ONNX Runtime是一个跨平台推理 (inference)和训练加速器，兼容各种执行提供程序（CPU、NVIDIA GPU、AMD GPU、ARM等）。它使用开放神经网络 (neural network)交换（ONNX）格式作为其模型标准。

- **量化 (quantization)支持：** 支持ONNX格式的量化模型。提供训练后量化（静态和动态）工具，可将FP32 ONNX模型转换为INT8。集成更精密的LLM特定量化技术（如GPTQ/AWQ）通常需要将量化模型转换为等效的ONNX表示，可能使用`QOperator`或QDQ（`QuantizeLinear`/`DequantizeLinear`）等格式。支持情况可能因目标执行提供程序（例如用于GPU加速的TensorRT或CUDA执行提供程序）而异。
- **性能：** 性能很大程度取决于所选的执行提供程序（EP）。使用硬件特定的EP（例如NVIDIA GPU上的TensorRT EP或CUDA EP）对于获得好的性能是必要的。通用CPU性能良好，但要获得与TensorRT-LLM或vLLM相近的最高GPU性能，可能需要在模型转换和优化调优上投入更多。
- **硬件兼容性：** 这是其主要优势。ONNX Runtime允许在多种硬件目标上部署模型，使其适用于异构环境或边缘部署。
- **易用性：** 模型转换为ONNX，尤其是在涉及复杂量化方案时，可能存在难度。一旦转换完成，运行时API在各平台间相对保持一致。
- **使用场景：** 适用于需要在各种硬件平台（包括CPU和非NVIDIA GPU）上部署的场景，或将LLM集成到已使用ONNX标准的现有应用中。

### 框架对比总结

| 特性 | 文本生成推理 (inference) (TGI) | vLLM | TensorRT-LLM | ONNX Runtime |
| --- | --- | --- | --- | --- |
| **主要目的** | 生产服务 (Hugging Face重点) | 高吞吐量 (throughput) (GPU) | 最高性能 (NVIDIA) | 跨平台推理 |
| **量化 (quantization)** | 集成Hugging Face库 (BnB, GPTQ, AWQ) | 原生AWQ，其他在发展 | 原生INT4/INT8/FP8 (GPTQ, AWQ) | ONNX INT8, QDQ转换 |
| **优化** | 连续批处理 | PagedAttention, 连续批处理 | 内核融合, 图优化 | 执行提供程序 (EPs) |
| **性能** | 良好 | 出色吞吐量 | 行业领先 (NVIDIA) | 取决于EP |
| **硬件侧重** | NVIDIA GPU (主要) | NVIDIA GPU | 仅限NVIDIA GPU | CPU, GPU (多供应商) |
| **易用性** | 较高 (尤其Hugging Face模型) | 中等 | 较低 (需要构建步骤) | 中等 (需要转换步骤) |
| **灵活性** | 一般 | 一般 | 一般 (绑定NVIDIA) | 较高 (硬件/平台) |

### 如何选择

选择合适的框架涉及平衡您的优先级：

- **对于需要快速部署并与Hugging Face生态系统集成，使用常用量化 (quantization)方法（GPTQ、AWQ、bitsandbytes）的情况：** TGI通常是最直接的选择。
- **对于需要最大化NVIDIA GPU上的请求吞吐量 (throughput)，尤其是在使用AWQ的情况下：** vLLM的PagedAttention提供了很大的优势。
- **对于希望在NVIDIA GPU上实现最低延迟和最高吞吐量，利用高级量化（包括FP8）和自定义优化的情况：** TensorRT-LLM是性能上的佼佼者，尽管它需要更多的设置投入。
- **对于需要在多种硬件（CPU、多GPU供应商、边缘设备）上部署或采用ONNX标准的情况：** ONNX Runtime提供了所需的灵活性，但为了获得最佳性能，可能需要细致的模型转换和EP选择。

LLM部署框架领域在不断变化。新功能、改进的量化支持和性能提升正在持续开发中。值得关注这些项目的进展，并随着您的要求或框架本身的演变重新评估您的选择。在了解了这些选项后，后续章节将提供使用其中一些关键框架进行部署的实用指导。

## 参考资料

- [NVIDIA TensorRT-LLM Documentation](https://docs.nvidia.com/deeplearning/tensorrt-llm/) — NVIDIA (2024)
  Publisher: NVIDIA
  TensorRT-LLM的官方资源，详细说明了其架构、优化技术以及对NVIDIA GPU上量化模型的支持。
- [Text Generation Inference (TGI) Documentation](https://huggingface.co/docs/text-generation-inference/index) — Hugging Face (2024)
  Publisher: Hugging Face
  使用文本生成推理(TGI)部署LLM的官方指南，包括与Hugging Face支持的各种量化方法的集成。
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/) — Microsoft (2024)
  Publisher: Microsoft
  ONNX Runtime的综合文档，概述了其跨平台功能、执行提供程序以及对ONNX量化模型的支持。

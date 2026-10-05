# LLM量化库概览

来源：[原文](https://apxml.com/zh/courses/quantized-llm-deployment/chapter-2-implementing-llm-quantization-toolkits/overview-quantization-libraries)

[返回章节目录](README.md) · [返回课程目录](../README.md)

要应用GPTQ、AWQ或INT4和NF4等先进的量化 (quantization)方法，需要实际工具。虽然这些方法的原理涉及对模型权重 (weight)和计算核的复杂修改，但已出现一些库来简化大型语言模型的这一过程。这些工具包抽象了许多底层细节，使您能够对预训练 (pre-training)模型进行量化并为高效推理 (inference)做好准备。将概览我们将在后续章节中使用的主要库，说明它们在LLM量化流程中的具体作用和能力。

### Hugging Face生态系统：Transformers、Accelerate和bitsandbytes

Hugging Face生态系统是处理Transformer模型的中心，它提供集成的量化 (quantization)支持。

- **`Transformers`**: 这个库是核心，提供访问数千个预训练 (pre-training)模型的途径以及用于加载、训练和推理 (inference)的标准接口。重要的是，`Transformers`集成了量化功能，允许您使用`bitsandbytes`等库直接将模型加载为低精度格式。
- **`bitsandbytes`**: 该库由Tim Dettmers等人开发，对于在PyTorch模型中直接实现低比特量化，特别是4比特（NF4、FP4）和8比特格式，作用很大。它的主要作用是提供高度优化的CUDA核，用于混合精度矩阵乘法（例如，将FP16激活与INT4权重 (weight)相乘）。当您使用`transformers`加载模型并设置`load_in_4bit=True`等标志时，`bitsandbytes`通常在后台运行，执行必要的权重量化并设置低比特计算。这通常是开始推理量化尝试的最简单方法，尤其适用于加载时直接进行的训练后量化（PTQ）。
- **`Accelerate`**: 尽管它本身不是一个量化库，但`Accelerate`简化了PyTorch代码在不同硬件配置（CPU、多GPU、TPU）上的运行，并处理设备放置。当对可能无法单独在一块GPU上运行的大型模型进行量化时，或者在运行量化过程本身（这可能计算量大）时，这一点尤其重要。它与`Transformers`和`bitsandbytes`协同工作。

这些库共同为应用某些类型的PTQ提供了一个便捷集成的环境，主要侧重于由`bitsandbytes`实现的直接将权重加载为低比特格式。

### 专用PTQ库：AutoGPTQ和AutoAWQ

尽管`bitsandbytes`提供了集成到`Transformers`中的直接量化 (quantization)，但要实现最佳精度，尤其是在4比特等极低比特率下，通常需要更复杂的算法，如GPTQ和AWQ。为此，已开发出专用库来高效地实现这些方法。

- **`AutoGPTQ`**: 该库提供了GPTQ（生成式预训练 (pre-training)Transformer量化）算法的易于使用的实现。GPTQ旨在通过逐层处理模型，并使用校准数据迭代地确定权重 (weight)矩阵的最佳量化参数 (parameter)来最小化量化误差。它以在4比特精度下保持良好精度而闻名。`AutoGPTQ`通常需要一个独立的量化步骤，您在此步骤中提供模型和校准数据集。输出是量化后的模型状态字典和配置文件，这些文件随后可以加载进行推理 (inference)，通常可以重新集成到`Transformers`框架中。Hugging Face Hub上许多流行的量化模型都已通过此库或类似实现使用GPTQ进行过处理。
- **`AutoAWQ`**: 该库实现了AWQ（激活感知权重Q量化）算法。AWQ遵循并非所有权重对模型性能都同等重要的思想。它根据分析校准阶段的激活尺度来识别重要权重，并在量化过程中选择性地保留它们的精度。目标是实现与GPTQ相当的量化效果，但量化时间可能更快。与`AutoGPTQ`类似，使用`AutoAWQ`通常涉及一个带有校准数据的独立量化步骤，生成一个可用于部署的量化模型。

这些专用库提供了比`Transformers`中`bitsandbytes`直接集成更高级的PTQ选项，以牺牲简单性换取可能更高的精度，尤其是在激进量化场景（例如INT3或INT4）中。

### 其他相关框架

一些部署和优化框架集成了对运行量化 (quantization)模型甚至自己执行量化的支持：

- **NVIDIA TensorRT-LLM**: 一个专门为在NVIDIA GPU上优化和部署LLM而设计的工具包。它可以接收使用GPTQ或AWQ等方法量化的模型，并应用进一步的图优化、核融合以及其自身的低比特核（如INT4/INT8），以获得最高的推理 (inference)性能。
- **vLLM**: 一个为高吞吐量 (throughput)设计的推理和服务引擎。它支持多种量化格式，包括AWQ，并实现PagedAttention等技术来优化推理期间的内存使用。
- **ONNX Runtime**: 一个跨平台推理引擎。使用各种技术量化的模型通常可以导出为ONNX格式，允许ONNX Runtime使用优化过的后端在不同的硬件目标（CPU、GPU）上高效执行它们。

虽然我们将在课程后续章节（第4章）中对这些部署框架进行进一步介绍，但值得了解的是，它们通常代表使用`AutoGPTQ`或`bitsandbytes`等库量化模型的预期运行环境。

了解每个库的功能和侧重是有效实施量化的第一步。`bitsandbytes`在Hugging Face中为基本的低比特操作提供了便捷集成。`AutoGPTQ`和`AutoAWQ`则提供更复杂的PTQ算法，以更好地保持精度。TensorRT-LLM和vLLM等部署框架借助这些量化模型进行优化推理。后续章节将提供实践操作指南，说明如何使用其中一些工具包量化LLMs。

## 参考资料

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2022)
  Journal: ICLR 2023; DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  提出GPTQ算法，一种针对大型语言模型的逐层训练后量化方法，旨在最小化量化误差。
- [AWQ: Activation-aware Weight Quantization for LLM Inference](https://arxiv.org/abs/2306.00978) — Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (2023)
  Journal: arXiv preprint arXiv:2306.00978; DOI: [10.48550/arXiv.2306.00978](https://doi.org/10.48550/arXiv.2306.00978)
  介绍了AWQ算法，根据激活尺度选择性地量化权重，以在低比特率下保持精度。
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer (2023)
  Journal: arXiv preprint arXiv:2305.14314; Publisher: arXiv; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  介绍了QLoRA和NF4（标准化浮点4比特）量化数据类型，该数据类型是bitsandbytes低比特能力的基础。
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face Transformers库的官方文档，详细说明其功能，包括模型加载和量化集成。
- [NVIDIA TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/) — NVIDIA (2024)
  Publisher: NVIDIA
  NVIDIA用于优化和部署大型语言模型（LLM）在NVIDIA GPU上的工具包的官方文档，支持多种量化格式。

---

[上一节](../01-%E9%AB%98%E7%BA%A7LLM%E9%87%8F%E5%8C%96%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%B0%86GPTQ%E5%BA%94%E7%94%A8%E4%BA%8ELLM.md) · [下一节](02-%E4%BD%BF%E7%94%A8%20bitsandbytes%20%E8%BF%9B%E8%A1%8C%E4%BD%8E%E4%BD%8D%E6%93%8D%E4%BD%9C.md)

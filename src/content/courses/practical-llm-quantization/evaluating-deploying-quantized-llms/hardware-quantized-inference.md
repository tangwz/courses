---
course: "practical-llm-quantization"
chapter: "evaluating-deploying-quantized-llms"
lesson: "hardware-quantized-inference"
sourceId: 4671
sourceUrl: "https://apxml.com/zh/courses/practical-llm-quantization/chapter-6-evaluating-deploying-quantized-llms/hardware-quantized-inference"
title: "量化推理的硬件考量"
description: "讨论硬件 (CPU、GPU类型) 对量化模型性能的影响。"
order: 3
plots: ["plots/4671-0.json"]
sourceHash: "fb4c4e92ff1fe702a740d818ebcf0468247509bda341705ea53c83ffcd7f281a"
sourceCorrections: []
---

量化 (quantization)大型语言模型的部署需要理解它们在真实硬件上的性能。硬件的选择，从标准CPU到强大的GPU和专用AI加速器，会显著影响从量化中获得的速度和效率提升。了解这些硬件特性对于有效部署以及将量化优势转化为实际性能改进非常重要。

量化的好处，特别是内存占用减少和计算速度加快，并非在所有类型的处理器上都能普遍实现。不同的硬件架构对量化所依赖的低精度整数运算（如INT8或INT4）的支持程度不同。

### CPU推理 (inference)能力

现代CPU，尤其是服务器级和最新的桌面处理器，已经集成了旨在加速深度学习 (deep learning)工作负载的指令集。英特尔深度学习加速 (DL Boost)，包含矢量神经网络 (neural network)指令 (VNNI)，就是一个很好的例子。VNNI使CPU能够比使用通用指令时更高效地执行INT8操作。

- **VNNI及类似扩展：** 如果您的目标CPU支持VNNI或等效的AMD扩展，与同一CPU上的FP32或FP16推理相比，您可以预期INT8量化 (quantization)模型会有明显的加速。这些指令通常将多个操作合并，从而提高吞吐量 (throughput)。
- **优化库：** 诸如OpenVINO、ONNX Runtime（使用CPU执行提供程序），特别是`llama.cpp`等框架经过大量优化，以使用这些CPU专用指令进行INT8推理，有时甚至进行INT4推理。使用这些库通常对于提升性能是必需的。
- **内存节省：** 即使没有显著的计算加速（可能在较旧的CPU上，或对于支持较不常见的INT4），量化带来的内存减少也是CPU上的主要好处。它允许更大的模型在内存有限的系统上运行。
- **限制：** 尽管有所改进，但CPU在高度并行的LLM推理方面的性能通常仍落后于GPU，尤其是在INT4等较低位宽方面，其原生硬件支持不如INT8成熟。相比GPU加速，提升通常更小。



![推理加速比与FP16基准对比](plots/4671-0.json)



> 量化带来的相对推理加速在CPU和GPU之间通常存在显著差异。带有专用核心的GPU通常显示出更大的提升，尤其对于较低位宽。（数值为示意）。

### GPU加速与Tensor Cores

图形处理器 (GPU)，尤其是NVIDIA的GPU，是训练和大型模型高性能推理 (inference)的主力。其大规模并行架构非常适合LLM中固有的矩阵乘法。

- **Tensor Cores：** 从Volta架构开始，NVIDIA GPU包含专门设计用于加速矩阵乘法的Tensor Cores，特别是混合精度（例如，FP16计算与FP32累积）和低精度整数格式（INT8、INT4、FP8）。将模型量化 (quantization)为Tensor Cores支持的格式（如较新GPU上的INT8或FP8）可以产生显著的加速，通常比FP16或FP32推理快几倍。
- **显存 (VRAM)减少：** 量化在GPU上最直接的好处也许是显存 (VRAM) 使用量的减少。FP16模型每个参数 (parameter)需要2字节，而INT8模型需要1字节，INT4模型仅0.5字节。这允许大得多的模型适应特定GPU的显存，或者相同模型以更大的批处理大小运行。
- **低精度支持：** 更新的GPU架构（如Ampere、Hopper）提供日益完善的低位宽支持，包括INT4和FP4格式的硬件加速，进一步提升效率。
- **软件生态系统：** 诸如NVIDIA的TensorRT、`bitsandbytes`、cuBLAS等库以及在PyTorch和TensorFlow等框架中的集成，善用CUDA和Tensor Cores来有效执行量化操作。正确使用这些工具对于获得最佳性能很重要。

### 专用AI加速器

一类专门为AI工作负载设计的硬件正在不断增加：

- **谷歌TPU：** 张量处理单元设计用于大批量、高效的矩阵计算，通常在低精度格式方面表现出色。它们主要通过谷歌云提供。
- **NPU和边缘设备：** 神经网络 (neural network)处理器越来越多地出现在移动系统芯片 (SoC) 和专用边缘AI硬件中。这些通常针对功耗效率和量化 (quantization)模型（通常为INT8）上的推理 (inference)进行优化。部署通常需要供应商特定的软件开发工具包 (SDK) 和模型编译步骤（例如，安卓NPU使用TensorFlow Lite，苹果神经引擎使用Core ML）。

这些加速器可以为量化模型提供最佳每瓦性能，但通常伴随着更受限的软件环境，并且可能需要特定的模型转换流程。

### 内存带宽的重要性

LLM推理 (inference)，尤其是大型模型，常常不仅受计算速度限制，而且受**内存带宽**限制，即数据（模型权重 (weight)、激活）在处理器（CPU/GPU）和内存（RAM/VRAM）之间移动的速率。

量化 (quantization)直接解决了这一瓶颈。通过减少表示每个权重所需的字节数，您可以减少每个推理步骤需要从内存中获取的数据总量。

> 量化减少了存储在内存中的模型权重大小，降低了推理期间对内存带宽的需求，这有助于显著提升性能。

这种数据传输的减少通常对整体推理加速的贡献不亚于甚至更多于更快的计算本身，尤其是在内存带宽受限的系统上（这对于LLM来说很常见）。

### 软件栈很重要

拥有能够进行低精度运算的硬件只是故事的一部分。软件栈，包括推理 (inference)框架（PyTorch、TensorFlow、ONNX Runtime）、专用库（`bitsandbytes`、TensorRT、`llama.cpp`）和低级内核实现，必须能够有效发挥硬件功能。

- **内核优化：** 需要高度优化的计算内核才能在目标硬件上高效执行量化 (quantization)矩阵乘法、反量化和激活函数 (activation function)。
- **格式兼容性：** 特定量化格式（如`llama.cpp`的GGUF或用于特定内核的GPTQ打包权重 (weight)）通常与知道如何在特定硬件上高效解包和计算它们的软件库紧密关联。
- **框架支持：** 确保您选择的框架和库明确支持在目标硬件（CPU、特定GPU架构）上使用所需量化类型（例如INT8、INT4）进行推理。

### 选择目标硬件

部署量化 (quantization)LLM时，请考虑以下硬件因素：

1. **性能需求：** 您需要绝对最低延迟还是最高吞吐量 (throughput)？带有Tensor Cores的GPU通常是最佳选择。如果中等性能可接受且成本或可访问性是一个因素，则带有VNNI的现代CPU是可行的。
2. **内存限制：** 您的模型有多大，目标硬件有多少RAM/VRAM？量化对于在资源受限设备（边缘硬件、消费级GPU、标准内存量）上运行大型模型很重要。INT4提供最高的内存节省。
3. **硬件可用性与成本：** GPU和专用加速器可能昂贵或访问受限。CPU无处不在。
4. **功耗效率：** 对于移动或边缘部署，NPU或功耗效率高的GPU/CPU可能会被优先考虑。量化显著降低功耗，因为数据移动更少且计算可能更简单。
5. **软件生态系统成熟度：** CPU和NVIDIA GPU的量化生态系统通常成熟。其他GPU或专用加速器上的支持可能有所不同或需要更特定的工具。

“最终，对您特定的量化模型在目标硬件上使用预期软件栈进行**基准测试**是唯一的方法。理论上的好处必须通过实际测量进行验证，正如前面章节所述，以了解所实现的实际速度、内存使用以及最终的精度-性能权衡。”

## 参考资料

- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) — Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer (2022)
  Journal: NeurIPS 2022; DOI: [10.48550/arXiv.2208.07339](https://doi.org/10.48550/arXiv.2208.07339)
  介绍了大型Transformer模型的8位量化技术，详细阐述了其方法和实际性能提升，与量化LLM推理高度相关。
- [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA
  为在NVIDIA GPU上利用Tensor Cores和CUDA平台优化和部署深度学习模型（包括量化）提供了全面指导。

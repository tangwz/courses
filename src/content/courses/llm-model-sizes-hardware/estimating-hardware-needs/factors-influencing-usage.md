---
course: "llm-model-sizes-hardware"
chapter: "estimating-hardware-needs"
lesson: "factors-influencing-usage"
sourceId: 4255
sourceUrl: "https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-5-estimating-hardware-needs/factors-influencing-usage"
title: "影响实际使用量的因素"
description: "提及上下文长度、批处理大小和软件开销的影响。"
order: 3
plots: ["plots/4255-0.json"]
sourceHash: "a7ea4701c865783f3f4bc0e9f3d08ab81d24cef15c0e4e0f0ec96c6bd243da47"
sourceCorrections: []
---

尽管将参数 (parameter)数量乘以每个参数的字节数这一基本计算能提供显存 (VRAM)需求的有用基准，但这只代表了*存储*模型权重 (weight)所需的内存。运行模型，特别是生成文本时，涉及动态过程，会额外占用内存。理解这些因素有助于说明为何实际显存使用量经常超过最初的估算。

### 激活值内存

当LLM处理输入并生成输出时，它会进行大量计算。这些计算的中间结果在神经网络 (neural network)层之间传递，被称为“激活值”。可以把激活值看作模型在计算过程中所需的*工作内存*或*暂存空间*。

这些激活值内存主要存储在显存（VRAM）中，以便GPU快速访问。激活值所需的显存量取决于几个方面：

- **模型架构：** 不同的模型设计有不同的激活值大小。
- **输入序列长度：** 较长的输入序列通常需要更多的计算，因此需要更大的激活值内存。
- **批处理大小：** 并行处理多个输入（更大的批处理）需要同时存储所有输入的激活值，这会显著增加内存需求。

激活值内存使用量是动态的；它随模型处理输入而波动。尽管在小批处理推理 (inference)时，它通常小于模型权重 (weight)所需的内存，但它是总显存需求的必要组成部分。

### 上下文 (context)长度（序列长度）

上下文长度，即序列长度，是指模型一次可以处理的文本量（输入词元 (token)加上生成的词元）。现代LLM通常支持非常长的上下文长度（数千甚至数万个词元）。

更长的上下文长度会通过以下两种方式直接影响内存使用：

1. **更大的激活值：** 如前所述，处理更长的序列会增加临时存储的激活值的大小。
2. **KV缓存大小：** 这通常是与序列长度相关的主要因素。

### 键值（KV）缓存

在文本生成过程中，LLM通常顺序运行，根据前面的词元 (token)一次生成一个词元。为避免对已处理词元的重复计算，会使用一种名为键值（KV）缓存的技术。

模型会为序列中的每个词元计算某些内部状态（称为键和值，这与你可能听说过的注意力机制 (attention mechanism)有关）。KV缓存将这些状态存储在显存 (VRAM)中。生成*下一个*词元时，模型会重复使用所有*先前*词元的缓存状态，从而节省大量计算时间。

然而，这个缓存会占用内存。其大小取决于：

- **序列长度：** 缓存大小与当前上下文 (context)中（输入+生成）的词元数量呈线性增长。更长的序列意味着更大的缓存。
- **批处理大小：** 批处理中的每个序列都需要自己的KV缓存。
- **模型配置：** 模型的隐藏层大小和注意力头数量决定了每个词元存储的键和值的大小。

对于长序列或大批处理，KV缓存会变得相当大，有时甚至能与模型权重 (weight)本身的大小相当。这是使用非常长的上下文窗口进行推理 (inference)时，显存需求远超基础模型大小所暗示值的一个主要原因。



![显存使用示例（7B模型，FP16）](plots/4255-0.json)



> LLM推理期间总显存使用的组成部分。基本估算只包含模型权重。

### 软件和框架开销

最后，你用于加载和运行LLM的软件工具和库也会占用一些资源。这包括：

- AI框架本身（如PyTorch或TensorFlow）。
- 与GPU接口的CUDA驱动程序和库。
- 特定的LLM服务库（如Hugging Face `transformers`、`vLLM`、`TGI`或`llama.cpp`），它们可能拥有自己的内存缓冲区或管理结构。

这种开销通常是最小的部分，但仍然会增加总系统内存和显存 (VRAM)占用。

### 综合考虑

简单的计算（$参数数量 \times 每个参数的字节数$）给你一个下限，即仅存储模型所需的绝对最小显存 (VRAM)。实际需求包括这个基准，再加上激活值内存、可能很大的KV缓存（特别是对于长序列/生成）和软件开销。因此，总是要计划显存使用量会明显高于基本估算，特别是如果你打算使用长上下文 (context)长度或批量处理输入。查看特定模型和计划使用的推理 (inference)软件的文档通常能提供更精确的硬件建议。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教材，涵盖了神经网络架构，包括对激活函数及其在计算中作用的解释。
- [Transformers Documentation: Large Model Training & Inference](https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/model#large-model-training-and-inference) — Hugging Face (2024)
  提供了使用大型语言模型的实践指南，涵盖了流行 `transformers` 库中的内存管理、批处理和推理优化等方面。

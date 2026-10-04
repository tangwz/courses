---
course: "practical-llm-quantization"
chapter: "foundations-model-quantization"
lesson: "why-quantize-llms"
sourceId: 4596
sourceUrl: "https://apxml.com/zh/courses/practical-llm-quantization/chapter-1-foundations-model-quantization/why-quantize-llms"
title: "为何对大型语言模型进行量化？"
description: "了解LLM量化的原因：减少内存占用、提高速度和降低能耗。"
order: 2
plots: ["plots/4596-0.json"]
sourceHash: "3227765e29f73bf28d053f75d0c26f79db402ee6c51966290d7451cc82ad2df4"
sourceCorrections: []
---

大型语言模型在许多自然语言任务中表现优异，但其高效性也伴随着一定的代价。像ChatGPT或Llama这样的模型可能包含数十亿，甚至数百亿的参数 (parameter)。在推理 (inference)（即从训练好的模型生成输出的过程）过程中存储这些参数并进行计算，需要大量的计算资源。量化 (quantization)正是为了应对这些资源需求。

### 减少内存占用

量化 (quantization)最直接的好处也许是模型尺寸的大幅缩小。LLM的参数 (parameter)，主要是其神经网络 (neural network)层中的权重 (weight)，通常使用32位浮点数（$FP32$）或有时16位浮点数（$FP16$或$BF16$）存储。

量化将这些高精度表示替换为低精度数据类型，最常见的是8位整数（$INT8$），甚至4位整数（$INT4$）。来看看直接的影响：

- **FP32 (32位 / 4字节):** 标准精度。
- **FP16 (16位 / 2字节):** 半精度，与FP32相比，大小已减少2倍。
- **INT8 (8位 / 1字节):** 与FP32相比，大小减少4倍，或与FP16相比减少2倍。
- **INT4 (4位 / 0.5字节):** 与FP32相比，大小减少8倍，或与FP16相比减少4倍。

一个包含70亿参数的大型模型，若以$FP32$存储，仅权重部分就大约需要$7 \times 4 = 28$ GB的存储空间。将此模型量化为$INT8$可将存储需求降至约7 GB，而$INT4$则大致降至3.5 GB。这种减少使得以下情况成为可能：

1. **存储更大的模型:** 将原本过大的模型存入可用磁盘空间或内存中。
2. **更快地加载模型:** 需要从存储器读取到内存的数据量更少。

量化也能减少*激活值*所需的内存。激活值是推理 (inference)过程中计算出的层级中间输出。在静态后训练量化或量化感知训练（我们稍后会讲到）等技术中，激活值也可以用低精度整数来表示。这降低了*运行时*内存使用（RAM或VRAM），而这通常是一个重要的瓶颈。

### 加快推理 (inference)速度

降低数字精度不仅节省空间，还能加快计算速度。现代硬件，包括CPU和GPU，通常具有专门的指令，与浮点运算相比，这些指令在使用低精度整数（特别是INT8）时，能更快地执行算术运算（如对LLM很重要的矩阵乘法）。

此外，推理速度不仅仅是原始计算能力的问题；它也受到内存带宽的很大影响，即数据在处理器和内存之间传输的速率。LLM会处理大量数据（权重 (weight)和激活值）。通过量化 (quantization)减少这些数据的大小，可以实现：

1. **更少的数据传输:** 从RAM/VRAM传输到计算单元（例如GPU核心）的字节数更少。
2. **更好的缓存利用率:** 更小的数据类型增加了所需数据已存在于更快缓存内存中的可能性。

算术运算的加速与内存带宽瓶颈的减少相结合，带来了更低的推理延迟（更快的响应时间）和更高的吞吐量 (throughput)（每秒更多的推理次数）。



![量化的影响](plots/4596-0.json)



> 位精度、模型大小和推理速度之间的关系。更低的精度能大幅降低模型大小并提高速度，但激进的量化（如INT4）可能会影响模型准确性。

### 降低能耗

更快的推理 (inference)和更少的数据移动也带来了更低的功耗。从内存中获取数据和执行复杂的浮点计算都是耗能的操作。使用低精度整数简化了计算并最大限度地减少了数据传输，使得量化 (quantization)模型更节能。这对于以下情况尤为重要：

- **电池供电设备:** 延长运行LLM的移动或边缘设备的运行时间。
- **大规模部署:** 降低服务LLM推理的数据中心的电力成本和环境影响。

### 使其能在资源受限设备上部署

更小的尺寸、更快的速度和更低的能耗这些综合优势，使得LLM可以在以前不切实际的环境中部署。这包括：

- **手机:** 直接在设备上运行复杂的语言功能。
- **消费级硬件:** 在标准笔记本电脑或台式机上使用LLM，无需昂贵的高端GPU。
- **边缘设备:** 在物联网设备、汽车或具有有限内存和处理能力的专用硬件中部署模型。

总而言之，量化 (quantization)不仅仅是一种优化技术；它通常是使LLM实用且易于使用的必需手段。通过大幅减少内存需求、提高推理 (inference)速度和降低能耗，量化使得这些强大的模型能够在更广泛的硬件和应用中部署。接下来的部分将详细说明通过各种量化方法*如何*实现这些减少。

## 参考资料

- [LLM.int8(): 8-bit Matrix Multiplication for Large Language Models](https://arxiv.org/abs/2208.07339) — Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer (2022)
  Journal: Advances in Neural Information Processing Systems; DOI: [10.48550/arXiv.2208.07339](https://doi.org/10.48550/arXiv.2208.07339)
  这篇论文介绍了一种专为大型语言模型设计的8位矩阵乘法技术，大幅减少了推理过程中的内存占用。
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2022)
  Journal: ICLR 2023; DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  这项工作详细介绍了一种用于生成式Transformer模型的训练后量化方法，实现了权重准确的4位量化。

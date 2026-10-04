---
course: "llm-model-sizes-hardware"
chapter: "model-size-hardware-connection"
lesson: "memory-bandwidth"
sourceId: 4231
sourceUrl: "https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-3-model-size-hardware-connection/memory-bandwidth"
title: "内存带宽的重要性"
description: "为什么内存访问速度（带宽）对LLM性能有影响。"
order: 5
plots: []
sourceHash: "e3507ba63806236469e40024d8a05d946a185e8cf7ec97bac0f8ca57c7ec1c07"
sourceCorrections: []
---

LLM中的大量参数 (parameter)需要存储在GPU的显存（VRAM）中。然而，仅仅拥有足够的显存容量还不足够。数据在显存和GPU处理核心之间传输的*速度*也格外重要。这种速度被称为**内存带宽**。

可以把显存想象成一个大型仓库（容量以千兆字节，即GB衡量），内存带宽则是通向仓库的道路宽度（以每秒千兆字节，即GB/s衡量）。如果你有一个巨大的仓库，但只有一条狭窄的单车道，即使里面的工人（GPU计算核心）非常快速，你也无法很快地搬运货物进出。类似地，如果内存带宽较低，GPU核心可能需要长时间等待参数和其他数据从显存传输过来，这会拖慢文本生成的整个过程。

### 带宽为何对LLM如此重要？

运行LLM，尤其是在推理 (inference)（生成文本）时，涉及持续的数据来回传输：

1. GPU需要从显存 (VRAM)中获取模型的参数 (parameter)（权重 (weight)）。请记住，这些参数可能多达数十亿。
2. 它使用这些参数和输入数据执行计算。
3. 它通常需要将中间结果（比如激活值，它表示计算过程中神经元的状态）写回显存，并在后续生成输出的步骤中再次读取它们。

大型模型意味着需要持续传输*大量*数据。现代GPU拥有极其强大的处理核心，能够每秒执行数万亿次计算（FLOPS）。但如果这些强大的核心缺乏数据供应，它们的效率就会很低。

如果内存带宽较低（道路狭窄），GPU核心就无法足够快地获取参数或中间数据。它们最终会处于空闲状态，等待数据传输完成。这意味着LLM生成文本的整体速度（通常以每秒生成的词元 (token)数衡量）并非受限于GPU的原始计算能力，而是受限于数据输入到GPU的速度。这种情况通常被称为过程是**内存受限**的。

### 带宽作为潜在瓶颈

考虑两款GPU：

- **GPU A：** 拥有16 GB显存 (VRAM)和极高的计算能力，但内存带宽相对较低（例如，400 GB/s）。
- **GPU B：** 也拥有16 GB显存和稍低的计算能力，但内存带宽高得多（例如，800 GB/s）。

对于运行大型LLM（它需要频繁访问数十亿参数 (parameter)），GPU B实际生成文本的速度可能比GPU A*更快*。这是因为其高带宽能更有效地为处理核心提供数据，减少空闲时间，即使其峰值计算速度可能较低。

> 较低的内存带宽会造成瓶颈，即使在计算能力强的GPU上也会减慢LLM推理 (inference)速度。更高的带宽允许显存和计算单元之间更快的数据传输，从而实现更高效的处理和更快的输出生成。

不同类型的GPU内存技术造成了带宽上的这些差异。例如，消费级GPU通常使用GDDR6内存，而高端数据中心GPU则经常使用HBM（高带宽内存）。HBM专门设计用于提供高得多的带宽，这也是这些GPU在训练和运行大型AI模型时更受青睐（且更昂贵）的原因之一。

### 总结：容量和速度都重要

在评估用于运行LLM的硬件时，显存 (VRAM)大小（容量）告诉您模型*是否*能装下，但内存带宽（速度）则极大影响其*运行速度*。对于需要不断传输大量参数 (parameter)数据的大型语言模型，更高的内存带宽通常直接转化为更好的性能，体现在更快的响应时间或每秒生成更多词元 (token)上。这两个因素在为您的LLM需求选择GPU时都是重要的考量。

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — NVIDIA Corporation (2023)
  Publisher: NVIDIA Corporation
  解释了GPU内存结构，包括全局内存和内存访问方式，这对于理解数据传输速度至关重要。
- [HBM3: The Next-Gen Memory Standard for AI and HPC](https://developer.nvidia.com/blog/hbm3-the-next-gen-memory-standard-for-ai-and-hpc/) — NVIDIA Corporation (2022)
  Publisher: NVIDIA Corporation
  描述了高带宽内存（HBM）技术，其设计以及它为何对人工智能和高性能计算任务（如大型语言模型）有益。
- [NVIDIA TensorRT-LLM: An Open-Source Library for Accelerating LLM Inference on NVIDIA GPUs](https://developer.nvidia.com/tensorrt-sdk) — NVIDIA Corporation (2023)
  Publisher: NVIDIA Corporation
  尽管侧重于加速库，本文仍强调了大型语言模型推理的性能挑战，间接说明了高效数据移动和内存带宽的重要性。
- [Computer Architecture: A Quantitative Approach](https://www.elsevier.com/books/computer-architecture/hennessy/978-0-12-811905-1) — John L. Hennessy, David A. Patterson (2017)
  Publisher: Morgan Kaufmann; Pages: 936
  提供了计算机架构的基础概念，包括内存层级、数据传输速率和延迟，这些对于理解GPU性能限制有所帮助。

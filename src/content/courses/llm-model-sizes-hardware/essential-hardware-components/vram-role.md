---
course: "llm-model-sizes-hardware"
chapter: "essential-hardware-components"
lesson: "vram-role"
sourceId: 4212
sourceUrl: "https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-2-essential-hardware-components/vram-role"
title: "视频内存 (VRAM)"
description: "了解GPU内存（VRAM）以及它对于存储LLM参数为何十分重要。"
order: 4
plots: []
sourceHash: "6ebf33981b656c745e6b1a864fcae2e4ecadf0cb659b7c6ca2eb2fcdf825cad7"
sourceCorrections: []
---

GPU是针对并行计算优化的专用处理器，这在AI任务中很常见。但就像CPU需要快速访问存储在系统内存的数据一样，GPU也需要自己的高速存储器。这种专用存储器被称为视频内存，或称VRAM。

可以将VRAM视为直接构建在显卡上的内存，紧邻GPU核心。这种物理上的接近很重要。系统内存虽然容量大，但离GPU较远，对于GPU来说，访问它比访问自身的VRAM慢得多。

### 什么是VRAM？

VRAM是视频随机存取存储器（Video Random Access Memory）的缩写。它是一种专门设计用于图形处理器（GPU）的内存类型。其主要作用是存储GPU需要快速访问的数据。在传统图形应用中，这包括纹理、帧缓冲区和复杂的3D模型数据等。对于AI，特别是大型语言模型，VRAM扮演着新的、重要的角色。

> CPU、系统内存、GPU和VRAM之间的关系。VRAM为GPU提供了直接位于显卡上的快速专用内存。

### 为什么VRAM对LLM十分重要

正如我们在第一章所学，大型语言模型由其参数 (parameter)定义。这些参数代表模型的学习成果。当您想要使用LLM（一个称为推理 (inference)的过程）时，这些参数需要加载到内存中，以便处理器可以访问它们。

由于GPU非常擅长LLM所需的各类计算，因此我们希望GPU承担主要计算任务。为了让GPU高效工作，模型的参数需要存储在其*本地*高速存储器中：即VRAM。

想象一位厨师（GPU）需要食材（模型参数）来烹饪一道复杂的菜肴（生成文本）。VRAM就像厨师旁边一个专用、组织良好的准备台，放置所有即时需要的食材。系统内存则像走廊尽头的主储藏室。虽然储藏室总量更多，但不断来回取用会大大减慢烹饪过程。如果所需食材无法全部放入准备台（VRAM），厨师的工作效率就会降低。

### VRAM容量：重要影响因素

VRAM最常讨论的规格是其*容量*，通常以千兆字节（GB）衡量。此容量直接限制了可以在该GPU上高效运行的LLM的大小。

以下是基本关联：

1. LLM拥有数百万或数十亿参数 (parameter)。
2. 每个参数需要一定的内存空间（稍后我们将讨论像FP16这样的精度对这方面的影响）。
3. 所有（或大部分）这些参数需要装入GPU的VRAM中，以便在推理 (inference)过程中快速处理。

例如，如果一个LLM的参数需要14 GB内存，但您的GPU只有8 GB的VRAM，您通常无法直接将整个模型加载到GPU上。虽然有一些处理这种状况的方法（比如模型拆分或使用系统内存），但它们会显著降低性能，因为GPU会不断等待数据从较慢的内存位置传输过来。

因此，**GPU上可用的VRAM容量通常是决定您是否能运行某个特定LLM的主要硬件限制。**

### VRAM带宽：速度也重要

除了容量，数据在VRAM和GPU核心之间传输的*速度*也很重要。这被称为内存带宽，通常以每秒千兆字节（GB/s）衡量。

更高的内存带宽能让GPU核心更快地获取数据，防止它们空闲等待下一条信息（如参数 (parameter)或中间计算）。对于LLM，大量数据（参数）不断被访问，高VRAM带宽对推理 (inference)过程中的更快响应时间（更低延迟）有很大帮助。

消费级GPU通常比为AI工作负载设计的专业或数据中心GPU具有更低的VRAM容量和带宽。

总之，VRAM是GPU的专用高速存储器。其容量是决定哪些LLM可以高效运行的主要因素，因为模型的参数必须放入此空间以获得最佳性能。其带宽影响GPU处理这些参数的速度。了解VRAM非常重要，因为我们要估算不同LLM大小所需的硬件。

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — NVIDIA Corporation (2023)
  Publisher: NVIDIA Corporation
  提供了CUDA编程模型和GPU架构的详细文档，包括对GPU内存层次结构（涵盖VRAM，在此称为全局内存）的全面解释。
- [Computer Architecture: A Quantitative Approach](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFkdhQPd-T8mUpM45Ua4cdc9s7pInypLoKUoc3wdEb1nAv1qodREN7WkaaV8wBnvjXjGFLPw__zalE_5P3Tx2D4OzO1b60DcnEQ1KfKZtxpUFzAQk5LXqLVuPsBj2cMgaqZ00C_zeiBGwV_aqppBHPvkBmNxxyy7xDiI9gjSetk_otYANd_HKC1SJY=) — John L. Hennessy, David A. Patterson (2017)
  Publisher: Morgan Kaufmann
  一本经典的教科书，提供了计算机体系结构的开创性知识，包括对内存层次结构及其性能影响的详细讨论，有助于从更广阔的背景理解VRAM。

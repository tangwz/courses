---
course: "getting-started-local-llms"
chapter: "preparing-local-environment"
lesson: "hardware-cpu"
sourceId: 4209
sourceUrl: "https://apxml.com/zh/courses/getting-started-local-llms/chapter-2-preparing-local-environment/hardware-cpu"
title: "硬件考量：中央处理器"
description: "了解中央处理器如何影响大型语言模型表现及基本要求。"
order: 1
plots: []
sourceHash: "6b0bb9e2b3deee77f69946739c843f866da518054dfc99cb988166c1fee976f3"
sourceCorrections: []
---

您的计算机的中央处理器（CPU）是其主要的控制中心。可以把它看作是负责执行程序指令、管理系统资源并协调其他组件活动的总经理。在本地运行大型语言模型时，即使强大的图形处理器（GPU）在处理最耗费资源的计算，CPU 的作用依然重要。

在处理本地大型语言模型时，CPU 参与多个环节：

1. **模型加载：** 在大型语言模型生成文本之前，其庞大文件（包含数十亿参数 (parameter)）需要从您的存储设备（例如SSD或HDD）加载到计算机内存（RAM）中。CPU 管理这一数据传输过程。更快的CPU有助于缩短模型加载时间，让您更快开始使用。
2. **操作管理：** 您用来运行大型语言模型的软件（例如我们稍后会讲到的Ollama或LM Studio）依赖于CPU来管理其运行、处理用户输入、显示输出，并协调系统不同部分之间的任务。
3. **执行计算（CPU推理 (inference)）：** 虽然GPU在大型语言模型文本生成（推理）中擅长处理并行数学运算，但并非每个人都有强大的GPU，有时模型也可能配置为部分或完全在CPU上运行。在这些情况下，CPU直接执行预测下一个词或标记 (token)所需的复杂计算。这个过程通常称为“CPU推理”。

### 中央处理器特性与大型语言模型表现

当在CPU上运行大型语言模型时，其特性直接影响文本生成的速度：

- **核心与线程：** 现代CPU有多个核心，这使得它们能够同时处理多项任务。更多核心通常能使可分解为并行部分的任务表现更优，例如大型语言模型推理 (inference)的某些环节。
- **时钟频率：** 以千兆赫（GHz）为单位，时钟频率表示CPU每秒执行的处理器周期数。更高的时钟频率通常意味着单个指令的执行速度更快。
- **指令集：** 较新的CPU通常支持高级指令集（例如AVX2 - 高级向量 (vector)扩展2）。一些大型语言模型软件，特别是像 `llama.cpp`（它支持许多用户友好的工具）这样的底层引擎，经过优化以利用这些指令。如果您的CPU支持这些，并且软件能充分发挥其作用，那么与没有这些指令的旧CPU相比，CPU推理速度会有大幅提升。

### 您需要何种中央处理器？

与RAM或VRAM不足可能导致模型无法加载的情况不同，几乎任何相对现代的CPU都*理论上*可以运行小型大型语言模型。然而，*实际使用体验*会有很大差异。

- **老旧或低端中央处理器：** 您可能会发现即使是小型模型也运行得非常慢，逐字生成文本时会有明显的停顿。模型加载也可能需要相当长的时间。虽然可用于测试，但对于互动使用可能会考验您的耐心。
- **现代多核中央处理器：** 像近期英特尔酷睿i5/i7/i9或AMD锐龙5/7/9系列这样的处理器能提供更好的使用感受。它们处理模型加载更迅速，并且在CPU上运行推理 (inference)时能提供快得多的文本生成速度。如果您计划主要依赖CPU完成大型语言模型任务，强烈建议使用性能好的现代处理器。

设定实际的预期是很重要的。即使使用快速CPU，仅靠CPU生成文本的速度也几乎总是比使用专用GPU慢，特别是对于大型模型。CPU通常更适合需要顺序处理和系统管理的任务，而GPU则擅长大型语言模型进行快速推理所需的大规模并行计算。

在稍后“检查您的系统规格”一节中，我们将向您展示如何轻松查询您拥有的具体CPU型号。目前，请您明白，虽然GPU在加速大型语言模型*推理*方面常受关注，但您的CPU仍然是影响本地运行大型语言模型时整体可用性和表现的一个基本组成部分。

## 参考资料

- [Computer Architecture: A Quantitative Approach](https://www.elsevier.com/books/computer-architecture/hennessy/978-0-12-811905-1) — John L. Hennessy and David A. Patterson (2017)
  Publisher: Morgan Kaufmann
  本教材全面介绍CPU设计，包括指令集、核心和并行处理，这些是理解LLM的CPU性能的基础。
- [Dive into Deep Learning](https://d2l.ai/) — Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola (2024)
  Publisher: Cambridge University Press
  这本在线书的介绍章节讨论了深度学习的计算需求以及CPU和GPU在管理这些工作负载中的作用，为LLM硬件提供了背景信息。
- [Intel® Software Developer Manuals](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html) — Intel Corporation (2025)
  Publisher: Intel Corporation
  这套官方手册提供了Intel CPU架构的详细信息，包括AVX2等指令集扩展，以及优化软件以获得更好性能的指南，与LLM CPU推理直接相关。

---
course: "llm-model-sizes-hardware"
sourceUrl: "https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-3-model-size-hardware-connection"
sourceId: 833
chapter: "model-size-hardware-connection"
title: "模型大小与硬件需求的关联"
order: 3
description: "了解LLM中的参数数量如何直接影响内存（显存、内存）和计算（GPU）要求。"
hasQuiz: true
---

在介绍了大型语言模型、它们的参数计量方式以及GPU和显存等必要的硬件组成部分之后，我们现在将这些方面联系起来。本章将讨论LLM中的参数数量如何直接转化为硬件资源要求。

您将了解：

*   **内存占用：** 模型参数如何加载到内存中，主要是显存。
*   **数据类型与精度：** 不同数值格式（如$FP16$、$INT8$）对内存使用的影响。
*   **量化：** 一种减少模型内存占用的方法。
*   **计算要求：** 理解与模型大小相关的计算负载（$FLOPS$）。
*   **内存带宽：** 为什么内存访问速度对LLM性能很重要。

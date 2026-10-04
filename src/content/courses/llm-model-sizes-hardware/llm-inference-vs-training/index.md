---
course: "llm-model-sizes-hardware"
sourceUrl: "https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-4-llm-inference-vs-training"
sourceId: 835
chapter: "llm-inference-vs-training"
title: "运行LLM：推理与训练"
order: 4
description: "对比使用LLM（推理）与构建或微调LLM（训练）的不同硬件要求。"
hasQuiz: true
---

在了解了大型语言模型、它们的参数规模以及GPU和显存等相关硬件组成后，我们现在来看看这些模型是如何实际应用的。它们有两种基本的运行方式：使用一个已有的模型来生成结果（推理），以及构建或调整模型的过程（训练）。

这些不同的活动对前面提到的硬件资源有着显著不同的需求。弄明白推理与训练的区别，有助于理解为何某些硬件配置仅适合运行LLM，而另一些则对开发或微调是必需的。

本章将涵盖以下几点：

*   定义模型推理：使用已训练模型进行预测或生成。
*   简述推理通常需要的硬件资源。
*   定义模型训练：从数据构建或调整模型参数的过程。
*   对比训练所需的显著更高的硬件要求。
*   阐明为何推理需求是大多数最终用户与LLM交互时的主要考量。

通过比较这两种过程的资源需求，我们能更好地理解稍后讨论的硬件估算方法。

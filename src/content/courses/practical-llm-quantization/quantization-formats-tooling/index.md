---
course: "practical-llm-quantization"
sourceUrl: "https://apxml.com/zh/courses/practical-llm-quantization/chapter-5-quantization-formats-tooling"
sourceId: 894
chapter: "quantization-formats-tooling"
title: "量化格式与工具"
order: 5
description: "了解GGUF和GPTQ等常用量化模型格式，以及Hugging Face Optimum和bitsandbytes等工具。"
hasQuiz: true
---

当模型通过训练后量化（Post-Training Quantization）或量化感知训练（Quantization-Aware Training）等方法进行量化后，一些实际问题随之而来：这些低精度模型如何高效地保存、加载和运行？标准模型序列化方法可能无法最佳处理$INT4$或$INT8$等格式所需的特有结构和元数据（如缩放因子或零点）。

本章将针对这些实际考量进行讲解，介绍量化LLM生态系统中常用的格式和软件工具。我们将介绍：

*   **专用文件格式：** 了解GGUF等格式的结构和用途，用于GPTQ量化模型的约定，以及AWQ格式的细节。
*   **核心库：** 使用Hugging Face Optimum等库进行量化和模型管理，以及`bitsandbytes`在推理时执行高效低比特操作。

您将熟悉将模型转换为这些格式，并使用相关工具高效地加载和运行它们。

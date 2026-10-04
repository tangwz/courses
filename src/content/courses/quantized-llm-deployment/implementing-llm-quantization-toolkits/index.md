---
course: "quantized-llm-deployment"
sourceUrl: "https://apxml.com/zh/courses/quantized-llm-deployment/chapter-2-implementing-llm-quantization-toolkits"
sourceId: 889
chapter: "implementing-llm-quantization-toolkits"
title: "使用工具包实现大型语言模型量化"
order: 2
description: "学习使用bitsandbytes、AutoGPTQ、AutoAWQ等流行库和工具包进行大型语言模型量化。"
hasQuiz: false
---

在上一章学习了高级量化技术的理论后，本节将侧重于使用常用软件库进行这些方法的实际操作。目标是让您从理解低比特量化（例如$INT4$）等原理以及GPTQ和AWQ等算法，转向在大型语言模型上实际执行这些操作。

您将使用专为LLM量化设计的广泛使用的工具包：

*   学习`bitsandbytes`如何支持高效的低比特操作。
*   了解如何使用`Transformers`和`Accelerate`直接在Hugging Face生态系统内集成量化。
*   分别使用`AutoGPTQ`和`AutoAWQ`库来应用GPTQ和AWQ。

在本章中，我们将介绍使用这些工具量化模型的必要步骤，考察如何比较不同库获得的结果和性能特点，并处理模型和工具包之间可能出现的兼容性挑战。学习结束时，您将拥有使用这些库准备大型语言模型以进行高效部署的实践经验。

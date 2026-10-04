---
course: "quantized-llm-deployment"
sourceUrl: "https://apxml.com/zh/courses/quantized-llm-deployment/chapter-1-advanced-llm-quantization-fundamentals"
sourceId: 887
chapter: "advanced-llm-quantization-fundamentals"
title: "高级LLM量化基本原理"
order: 1
description: "理解LLM量化中的高级思想，包括低位方法、数据类型、校准和量化感知训练的具体事项。"
hasQuiz: false
---

本章为理解专门应用于大型语言模型（LLMs）的高级量化方法打下基础。我们首先回顾核心量化原理，并将对称量化与非对称量化（$q = \text{round}(x/s)$ vs $q = \text{round}(x/s) + z$）以及逐张量与逐通道缩放等思想，调整以适应LLM的独特特点。

你将学习低于8位精度的工作方法，例如INT4以及NF4和FP4等专用格式，考察它们的数学属性以及对模型准确性和性能的影响。我们将研究为LLM开发的重要训练后量化（PTQ）算法，包括GPTQ和AWQ，了解它们如何在最少重新训练的情况下保持模型保真度。在大型模型方面的量化感知训练（QAT）的考量也将进行讨论。

本章还介绍混合精度量化的策略，为PTQ方法选择合适的校准数据，并以一个将GPTQ应用于示例LLM的实践练习作结。学完本章后，你将牢固掌握现代LLM量化中使用的理论依据和常用算法。

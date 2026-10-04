---
course: "introduction-to-llm-fine-tuning"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-1-foundations-of-model-customization"
sourceId: 1263
chapter: "foundations-of-model-customization"
title: "模型定制的根基"
order: 1
description: "了解预训练与微调之间的区别，学习何时对大型语言模型进行微调，并配置合适的开发环境。"
hasQuiz: false
---

本章介绍模型定制的基本原理。预训练大型语言模型虽具备通用本领，但微调是使其适应特定功能或方面。我们将首先界定微调的含义，并阐述它在目标、数据需求和计算开销上与初始预训练过程有何不同。

本章的一个核心议题是何时进行微调的实际决定。我们将提出一个分析架构，帮助您判断微调是否为解决问题的正确途径，并将其与高级提示或检索增强生成（RAG）等其他方法进行比较。随后，您将获得关于不同定制策略的概括性总结，涵盖从全参数更新到计算效率更高的办法。微调与迁移学习原理之间的联系也将得到阐释。

为接下来的实践操作做准备，本章最后将提供配置您的开发环境的指导。我们将逐步讲解 PyTorch、Transformers 库以及 Hugging Face 生态系统中的其他组成部分的设置，确保您为后续章节拥有一个可用的设置。

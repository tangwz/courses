---
course: "introduction-to-llm-fine-tuning"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-2-data-preparation-for-fine-tuning"
sourceId: 1264
chapter: "data-preparation-for-fine-tuning"
title: "用于微调的数据准备"
order: 2
description: "学习如何获取、清洗和组织高质量数据集，以用于微调大型语言模型。"
hasQuiz: false
---

模型表现，很大程度上取决于其训练数据的质量。在您通过微调改变模型表现之前，必须先收集一个能清晰且统一体现期望能力的数据集。本章将系统地指导您如何准备高质量数据，从而让大型语言模型具备特定能力。

您将学习创建可供模型使用的数据集的完整流程。我们会讲到：

*   **数据获取与结构化：** 寻找合适的公开或私有数据，并将其组织成指令遵循或对话对 $(x, y)$ 等有效格式的方法。
*   **数据清洗与预处理：** 移除噪声、处理不一致性、以及文本标准化的实用技术，以提升训练稳定性和最终模型表现。
*   **分词：** 将文本转换为模型所需数字格式的过程，包括处理特殊词元、填充和截断的策略。

本章最后包含一个实践练习，您将运用这些技术，将原始文本数据处理成一个结构化、已分词的、可用于微调的数据集。

---
course: "mlops-for-large-models-llmops"
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-3-llm-training-finetuning-ops"
sourceId: 685
chapter: "llm-training-finetuning-ops"
title: "大模型训练与微调的运行管理"
order: 3
description: "使用先进方法，对大型语言模型的训练和微调过程进行运行管理。"
hasQuiz: false
---

训练和微调参数量达数十亿的模型，需要采用有别于标准机器学习工作流程的专门操作方法。这种规模在计算、内存管理以及可能跨数百个加速器进行协调方面带来了很多难题。

本章侧重于管理这些大规模训练过程的运行管理方面。您将学习如何：
*   高效地编排和管理分布式训练任务。
*   实现不同的并行策略，包括数据并行、张量并行和流水线并行，以处理超出单个设备内存的模型。例如，理解如何将模型层拆分到多个GPU上（流水线并行）或将单个层计算分布到多个设备上（张量并行），这很重要。
*   使用 DeepSpeed 和 Megatron-LM 等专门框架，这些框架旨在简化和优化大模型训练。
*   将参数高效微调（PEFT）技术（例如 LoRA）整合到运行流程中，从而经济高效地调整模型。
*   调整实验追踪和检查点方法，以应对大规模训练运行的规模和持续时间，确保可复现性和容错性。

在本章结束时，您将掌握在MLOps框架内成功训练和微调大型语言模型的主要技术和运行管理方面的考量。

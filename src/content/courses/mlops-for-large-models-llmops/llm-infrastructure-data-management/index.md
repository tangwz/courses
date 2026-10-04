---
course: "mlops-for-large-models-llmops"
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-2-llm-infrastructure-data-management"
sourceId: 682
chapter: "llm-infrastructure-data-management"
title: "大规模基础设施与数据管理"
order: 2
description: "学习管理训练和微调大型语言模型所需的基础设施和海量数据集。"
hasQuiz: false
---

运行大型语言模型，其对基础设施和数据管理的要求与小型模型有着很大差异。处理规模可达PB级 ($PB$) 的训练数据集，以及调度数百甚至数千个GPU ($N_{GPU}$)，都需要特定的架构模式和操作方法。简单地扩展标准MLOps技术往往不足或成本过高。

本章侧重于构建和管理LLM运营基础的实际方面。您将学会如何：

*   设计可扩展的计算集群 (GPU/TPU)，并处理高效分布式处理所需的网络考量。
*   实施存储、访问和预处理超大规模数据集的策略。
*   构建专门为LLM需求定制的有效数据管道，包括数据清洗和分词。
*   应用适合大规模数据和模型制品的版本控制方法。
*   评估LLMOps中云端和本地基础设施方案之间的权衡。

我们首先考察可扩展计算和网络的设计原则，然后转到数据本身的管理，包括存储、预处理和版本控制。

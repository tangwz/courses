---
course: "fine-tuning-adapting-large-language-models"
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-7-optimization-deployment-considerations"
sourceId: 677
chapter: "optimization-deployment-considerations"
title: "优化与部署考量"
order: 7
description: "用于优化经过微调的LLM以进行推断的方法，包括量化、剪枝、高效服务策略以及MLOps整合。"
hasQuiz: false
---

在为特定任务调整好大型语言模型后，重心将转向如何使其在实际使用场景中实用且高效。本章将讨论优化模型并准备其进行部署的步骤。

您将研究在微调过程中提升资源利用率的方法，包括梯度累积和混合精度训练（例如，使用 $fp16$ 而非 $fp32$）等节省内存的技术。我们还将介绍使用分布式计算配置来加速训练的方法。

训练结束后，您将学习微调后的优化方法，例如量化（降低权重精度，可能降至 $int8$ 或更低）和剪枝，以减小模型大小和推断延迟。本章还详细说明了模型打包、将PEFT适配器合并回基础模型权重的方法、选择适合LLM的推断服务框架，以及为已部署模型制定监控规范。

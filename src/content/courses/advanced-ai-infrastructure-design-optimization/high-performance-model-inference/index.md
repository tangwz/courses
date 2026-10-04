---
course: "advanced-ai-infrastructure-design-optimization"
sourceUrl: "https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-4-high-performance-model-inference"
sourceId: 1226
chapter: "high-performance-model-inference"
title: "第4章：高性能模型推理与服务"
order: 4
description: "设计并部署低延迟、高吞吐量的推理服务。学习模型编译、量化以及使用Triton进行服务部署。"
hasQuiz: false
---

训练好的模型制品并非生产服务。模型部署推理的过程带来一系列独特的工程挑战，主要集中在性能和效率方面。训练过程优先考虑长时间作业的吞吐量，而推理服务通常必须满足严格的单请求延迟服务水平目标（SLO），例如99%分位响应时间低于100毫秒（$p99 < 100ms$）。为达到这些目标，模型和基础设施层面都需要进行特定的优化。

本章介绍工程构建和部署高性能推理系统的方法。我们将首先审视低延迟与高吞吐量之间的架构权衡。之后你将学习如何将训练后优化直接应用到你的模型，包括：

*   **模型编译：** 使用TensorRT和ONNX等运行时来进行图优化、内核融合以及硬件定制调整。
*   **量化：** 通过将模型的权重和激活从32位浮点格式转换为INT8等低精度格式，从而减少模型的计算和内存占用。

有了优化后的模型，我们将转向使用NVIDIA Triton推理服务器进行部署。你将了解如何管理多个模型、配置动态批处理以提升GPU利用率，并实施安全的部署策略，例如A/B测试和金丝雀部署，用于新的模型版本。

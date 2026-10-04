---
course: "planning-optimizing-ai-infrastructure"
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-1-foundations-ai-compute-infrastructure"
sourceId: 1220
chapter: "foundations-ai-compute-infrastructure"
title: "AI 计算基础设施入门"
order: 1
description: "了解 AI 基础设施的核心组件，包括 CPU、GPU、内存和存储，用于机器学习工作负载，如训练和推理。"
hasQuiz: false
---

AI 模型的性能与其运行的硬件直接相关。要构建高效系统，您必须首先了解机器学习工作负载的计算需求以及满足这些需求的硬件组件。本章提供初步分析。

您将学会区分训练和推理工作负载及其不同的硬件要求。我们将分析 CPU 在顺序任务中的作用以及 GPU 在深度学习中常见的并行计算中的作用。我们将比较它们的架构，并了解为何 GPU 擅长同时执行数千次操作，例如神经网络核心的矩阵乘法 ($C = A \cdot B$)。本讨论还将涉及 TPU 等专用加速器，以及内存、存储和网络的辅助作用。

本章最后是一个实践练习，您将在 CPU 和 GPU 上对一项任务进行基准测试，以便亲身观察这些性能差异。

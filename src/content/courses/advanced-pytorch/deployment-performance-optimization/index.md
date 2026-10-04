---
course: "advanced-pytorch"
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-4-deployment-performance-optimization"
sourceId: 634
chapter: "deployment-performance-optimization"
title: "模型部署和性能优化"
order: 4
description: "学习 TorchScript、量化、剪枝、性能分析、ONNX 导出以及 TorchServe，以实现高效的 PyTorch 模型部署。"
hasQuiz: false
---

在开发和训练复杂的 PyTorch 模型之后，重点转向如何将它们投入实际应用。本章将介绍模型部署和性能优化，提供使模型在推理时更快、更小、更节省资源的方法。

我们将介绍使用 TorchScript 进行模型序列化，学习跟踪和脚本两种方法。您将学习模型压缩技术，包括量化（静态、动态和量化感知训练）和剪枝策略，以减小模型大小和计算需求。我们将使用 PyTorch Profiler 来识别 CPU 和 GPU 执行中的性能瓶颈。此外，您还将学习将模型导出为 ONNX 格式以获得更广泛的兼容性，并学习使用 TorchServe 高效地提供模型服务。

在本章结束时，您将掌握分析模型性能的实用技能，并运用多种优化技术，这对于将 PyTorch 模型从开发环境部署到生产环境是必不可少的。

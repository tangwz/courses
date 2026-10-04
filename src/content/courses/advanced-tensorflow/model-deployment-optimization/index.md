---
course: "advanced-tensorflow"
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-6-model-deployment-optimization"
sourceId: 657
chapter: "model-deployment-optimization"
title: "模型部署与优化"
order: 6
description: "学习使用 TF Serving 部署 TensorFlow 模型，并使用 TF Lite 对其进行推理优化。"
hasQuiz: false
---

训练 TensorFlow 模型是一个重要进展，但工作并未就此结束。为了创造价值，模型必须高效可靠地投入运行。本章讲解了将模型从开发环境迁移到生产系统或边缘设备的重要阶段。

您将学习使用 SavedModel 格式打包 TensorFlow 模型进行部署的标准方法。我们将研究 TensorFlow Serving，它是一个专为高性能模型服务而设计的方案，包括如何使用常用协议与已部署的模型进行交互。此外，我们还将介绍优化策略，例如量化，这些策略旨在减小模型大小并提高推理速度。最后，我们介绍了 TensorFlow Lite，用于转换和准备模型，以便在移动、嵌入式以及其他资源受限的环境中部署。重点在于使您训练的模型能够被实际使用并高效运行所需的具体步骤。

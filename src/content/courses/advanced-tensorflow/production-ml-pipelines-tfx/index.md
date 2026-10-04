---
course: "advanced-tensorflow"
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-5-production-ml-pipelines-tfx"
sourceId: 654
chapter: "production-ml-pipelines-tfx"
title: "TFX：生产级机器学习流水线"
order: 5
description: "使用 TensorFlow Extended (TFX) 构建可靠、可扩展的端到端机器学习流水线。"
hasQuiz: false
---

开发和优化 TensorFlow 模型是很重要的一步，但将模型部署到可靠的生产环境则面临一系列特有的挑战。为了获得可靠的结果，数据预处理、训练、评估和模型服务之间的一致性是必要的。这需要管理整个机器学习工作流，而不仅仅是模型代码本身。

本章将介绍 TensorFlow Extended (TFX)，这是一个用于部署生产级机器学习流水线的端到端平台。我们将查看 TFX 如何提供一个结构化框架来管理机器学习模型的生命周期，确保可重现性和可扩展性。

您将学习：

*   了解 TFX 在生产系统中的架构和目的。
*   使用标准 TFX 组件完成诸如数据摄取 (`ExampleGen`)、数据验证 (`StatisticsGen`, `SchemaGen`)、特征工程 (`Transform`)、模型训练 (`Trainer`)、评估 (`Evaluator`) 和部署验证 (`Pusher`) 等任务。
*   将这些组件组合成一个内聚的流水线。
*   掌握使用常用工作流管理器编排 TFX 流水线背后的理念。

通过实践 TFX 组件及其协作方式，您将获得构建更易于管理、更自动化且适用于生产环境的机器学习系统的技能。

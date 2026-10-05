# 第 5 章：TFX：生产级机器学习流水线

来源：[原章节](https://apxml.com/zh/courses/advanced-tensorflow/chapter-5-production-ml-pipelines-tfx)

[返回课程目录](../README.md)

开发和优化 TensorFlow 模型是很重要的一步，但将模型部署到可靠的生产环境则面临一系列特有的挑战。为了获得可靠的结果，数据预处理、训练、评估和模型服务之间的一致性是必要的。这需要管理整个机器学习工作流，而不仅仅是模型代码本身。

本章将介绍 TensorFlow Extended (TFX)，这是一个用于部署生产级机器学习流水线的端到端平台。我们将查看 TFX 如何提供一个结构化框架来管理机器学习模型的生命周期，确保可重现性和可扩展性。

您将学习：

*   了解 TFX 在生产系统中的架构和目的。
*   使用标准 TFX 组件完成诸如数据摄取 (`ExampleGen`)、数据验证 (`StatisticsGen`, `SchemaGen`)、特征工程 (`Transform`)、模型训练 (`Trainer`)、评估 (`Evaluator`) 和部署验证 (`Pusher`) 等任务。
*   将这些组件组合成一个内聚的流水线。
*   掌握使用常用工作流管理器编排 TFX 流水线背后的理念。

通过实践 TFX 组件及其协作方式，您将获得构建更易于管理、更自动化且适用于生产环境的机器学习系统的技能。

## 小节

- 1. [TensorFlow Extended (TFX) 简介](01-TensorFlow%20Extended%20%28TFX%29%20%E7%AE%80%E4%BB%8B.md)
- 2. [TFX 标准组件概述](02-TFX%20%E6%A0%87%E5%87%86%E7%BB%84%E4%BB%B6%E6%A6%82%E8%BF%B0.md)
- 3. [数据摄取与验证](03-%E6%95%B0%E6%8D%AE%E6%91%84%E5%8F%96%E4%B8%8E%E9%AA%8C%E8%AF%81.md)
- 4. [特征工程与 Transform](04-%E7%89%B9%E5%BE%81%E5%B7%A5%E7%A8%8B%E4%B8%8E%20Transform.md)
- 5. [模型训练与调优](05-%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E4%B8%8E%E8%B0%83%E4%BC%98.md)
- 6. [模型验证与分析](06-%E6%A8%A1%E5%9E%8B%E9%AA%8C%E8%AF%81%E4%B8%8E%E5%88%86%E6%9E%90.md)
- 7. [使用 Pusher 进行模型服务与部署](07-%E4%BD%BF%E7%94%A8%20Pusher%20%E8%BF%9B%E8%A1%8C%E6%A8%A1%E5%9E%8B%E6%9C%8D%E5%8A%A1%E4%B8%8E%E9%83%A8%E7%BD%B2.md)
- 8. [编排 TFX 流水线](08-%E7%BC%96%E6%8E%92%20TFX%20%E6%B5%81%E6%B0%B4%E7%BA%BF.md)
- 9. [动手实践：构建一个简单的 TFX 流水线](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84%20TFX%20%E6%B5%81%E6%B0%B4%E7%BA%BF.md)

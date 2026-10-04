---
course: "data-versioning-experiment-tracking"
chapter: "tracking-experiments-mlflow"
lesson: "introducing-mlflow-tracking"
sourceId: 4079
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-3-tracking-experiments-mlflow/introducing-mlflow-tracking"
title: "MLflow 追踪功能介绍"
description: "MLflow 追踪的运行、参数、指标、产物及追踪服务器主要构成概览。"
order: 2
plots: []
sourceHash: "f64364829d895ac10b189b880383b2ada89b4ce7ac1cb3d3fef84fef700e84a4"
sourceCorrections: []
---

如前所述，机器学习 (machine learning)开发是一个迭代过程。您可能会尝试数十甚至数百种不同的变体，涉及不同的算法、特征集、数据预处理步骤和超参数 (parameter) (hyperparameter)值。记录哪些有效、哪些无效以及某个特定结果是如何产生的，很快就会变得困难。依赖手动笔记、电子表格或复杂的 文件命名约定通常容易出错，并且难以扩展。

MLflow 追踪是开源 MLflow 平台的一个组成部分，旨在应对管理机器学习生命周期中的挑战，特别是实验的记录和组织。通过在代码中加入工具，您不再需要手动记录细节，而是可以自动捕获每次训练执行的重要信息。

### MLflow 追踪的主要要素

MLflow 追踪围绕几个主要思想来组织您的工作。掌握这些思想是正确使用该工具的基础：

1. **运行 (Run)：** 您的模型训练代码（或任何您希望追踪的数据科学代码）的单次执行。每次运行训练脚本时，您通常都会启动一个新的 MLflow 运行。MLflow 会为每次运行分配一个唯一的 ID。
2. **参数 (parameter) (Parameters)：** 这是一次运行的输入设置。可以把它们看作您想要记录的配置或超参数 (hyperparameter)。示例包括优化器的学习率、神经网络 (neural network)的层数、正则化 (regularization)参数的值（例如 SVM 中的 $C$），或者输入数据集版本的路径（可能由 DVC 管理）。记录参数使您能够清楚地了解是哪种配置产生了特定的结果。
3. **指标 (Metrics)：** 这是您希望在不同运行之间测量和比较的量化 (quantization)输出或结果。指标通常是评估模型性能的数值，例如准确率、精确率、召回率、F1 分数、均方误差 (MSE) 或曲线下面积 (AUC)。MLflow 允许您在运行结束时记录指标，甚至在运行过程中多次记录（例如，在每个 epoch 后记录训练损失）。这对于观察模型收敛情况特别有用。指标会与时间戳一同存储。
4. **产物 (Artifacts)：** 这是与运行相关的输出文件。产物可以是任何内容：序列化模型文件（例如序列化的 scikit-learn 模型或保存的 TensorFlow/PyTorch 模型）、图片（例如性能图或数据可视化）、数据文件（例如处理后的特征或模型预测），甚至是包含日志或笔记的文本文件。MLflow 会存储这些文件，让您能够获取特定运行产生的确切输出。
5. **源代码版本 (Source Code Version)：** 为了确保完全可复现性，MLflow 可以自动记录用于运行的代码版本。如果您的项目使用 Git 管理，MLflow 通常会记录 Git 提交哈希值。这会将特定的代码状态与运行的参数、指标和产物关联起来。
6. **实验 (Experiment)：** 实验是将相关运行进行分组的方式。可以将其看作特定任务或项目的操作空间，例如“预测客户流失”或“优化 ResNet50 超参数”。所有运行都在实验的背景下被记录。如果您没有指定实验，MLflow 会使用一个默认实验。

下图说明了这些组成部分之间的关系：一个实验包含多个运行。每次运行都执行一些代码（由其版本标识），使用特定的参数，产生指标，并生成产物。

> MLflow 追踪组成部分之间的关系：实验对运行进行分组，每个运行都会记录参数、指标、产物，并链接到代码版本。

### MLflow 追踪架构概述

MLflow 追踪主要包含两个部分：

1. **MLflow 客户端 (API/SDK)：** 这是您在代码中与 MLflow 交互的方式。MLflow 提供了适用于 Python、R、Java 的库以及一个 REST API。您可以在脚本中使用诸如 `mlflow.log_param()`、`mlflow.log_metric()` 和 `mlflow.log_artifact()` 等函数，将关于运行的信息发送到追踪后端。
2. **追踪服务器与后端存储：** 这里是客户端记录的信息的存储和管理位置。MLflow 支持多种后端配置：

   - **本地文件系统：** 默认情况下，MLflow 将数据记录到本地 `mlruns` 目录中的文件。这对于初次使用来说很简单，但不太适合协作或远程执行。
   - **数据库：** 您可以配置 MLflow 将元数据（参数 (parameter)、指标、运行信息）存储在数据库中（例如 PostgreSQL、MySQL、SQLite）。产物通常仍单独存储（例如，在文件系统或云存储上）。
   - **远程追踪服务器：** MLflow 可以作为集中式服务器运行，供多用户或多机器记录。该服务器管理元数据（通常通过数据库）和产物存储（本地或云端）。这种设置非常适合团队和生产环境。

追踪服务器还提供一个基于网络的**用户界面 (UI)**，让您能够可视化地浏览、搜索和比较实验和运行。

采用 MLflow 追踪，您会获得一种系统性的方式来记录机器学习 (machine learning)实验。这带来了许多好处：

- **组织性：** 将相关运行分组到实验中。
- **可复现性：** 轻松找到产生特定结果的确切参数、代码版本，甚至数据版本（我们稍后会了解到）。
- **比较：** 使用记录的指标和参数，分析和比较不同运行的性能。
- **协作：** 使用集中式追踪服务器与团队成员共享实验结果。

在以下章节中，我们将了解设置 MLflow 并使用其 API 来为您的训练代码添加工具的实际操作。

## 参考资料

- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html) — MLflow Community (2024)
  这是理解 MLflow Tracking 的权威且最新来源，涵盖其核心概念、API、架构和用法。
- [MLflow in Action](https://www.manning.com/books/mlflow-in-action) — Nader Jafari (2023)
  Publisher: Manning Publications
  一份实用指南，详细介绍了如何使用 MLflow 及其 Tracking 组件来有效管理机器学习生命周期。
- [Introducing MLOps: How to Go from Model to Production](https://www.oreilly.com/library/view/introducing-mlops/9781492083282/) — Hannes Hapke, Catherine Nelson, Maximo Bragagnolo (2020)
  Publisher: O'Reilly Media
  本书提供了 MLOps 原理的基础知识，解释了实验跟踪和可复现性的必要性，MLflow 正是为解决这些问题而设计。

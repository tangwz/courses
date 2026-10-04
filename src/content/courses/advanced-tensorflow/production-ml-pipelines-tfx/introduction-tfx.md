---
course: "advanced-tensorflow"
chapter: "production-ml-pipelines-tfx"
lesson: "introduction-tfx"
sourceId: 2891
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-5-production-ml-pipelines-tfx/introduction-tfx"
title: "TensorFlow Extended (TFX) 简介"
description: "了解 TFX 构建生产级机器学习流水线的目标和架构。"
order: 1
plots: []
sourceHash: "c02d9a5f441164c1acd661982bdc8720e1f9a2e349fd50b6332ef2edac2571fc"
sourceCorrections: []
---

将机器学习 (machine learning)模型从研究或开发环境部署到生产系统会带来很大的复杂性。虽然TensorFlow提供了强大的模型构建和训练工具，但要确保数据处理一致、训练可复现、评估彻底以及部署可靠，就需要一个更全面的框架。例如，训练-服务偏差问题，即训练和推理 (inference)之间数据处理的细微差异导致性能下降，成为重要的操作问题。手动管理这些步骤容易出错，且无法有效扩展。

TensorFlow Extended (TFX) 在这里发挥作用。TFX 是一个端到端平台，专门设计用于构建和管理生产级机器学习流水线。它提供了一个标准化框架和一系列库，用于协调机器学习模型的整个生命周期，从数据摄取和验证，到训练、评估，再到部署和提供服务。

### 生产机器学习 (machine learning)流水线的必要性

开发机器学习模型通常侧重于实验，并在静态数据集上达到高准确率。然而，生产机器学习涉及持续运行、数据变化，以及自动化、监控和治理的需求。请考虑这些典型的生产难题：

1. **数据一致性：** 确保用于提供预测的数据与用于训练的数据以完全相同的方式进行预处理，这一点非常重要。这些差异，即所谓的训练-服务偏差，是常见的错误来源。
2. **自动化和可扩展性：** 生产系统通常需要模型在新数据上定期重新训练。这个流程，包括数据验证、训练、评估和部署，必须自动化，并能处理大型数据集和可能的分布式计算。
3. **可复现性和调试：** 追踪生成特定模型版本所用的确切数据、代码和配置，对于调试、审计和复现结果非常重要。
4. **模型治理：** 维护不同模型版本、理解它们的性能特点，以及决定何时将新模型部署到生产环境，这需要仔细的管理。

TFX 通过将机器学习工作流程构建为组件的有向无环图（DAG），并由协调器管理，来解决这些挑战。

### TFX 架构：流水线和组件

其核心是，TFX 定义了一个**流水线**，它表示完整的机器学习 (machine learning)工作流程。该流水线由多个**组件**组成。每个组件都是一个独立的程序代码块，在机器学习生命周期中执行特定步骤。TFX 提供了一个标准组件库，涵盖常见任务：

- **数据摄取：** (`ExampleGen`) 从各种源读取数据。
- **数据验证：** (`StatisticsGen`, `SchemaGen`, `ExampleValidator`) 计算统计信息，推断数据架构，并查找异常或漂移。
- **特征工程：** (`Transform`) 为训练和提供服务一致地执行数据预处理和特征转换。
- **模型训练：** (`Trainer`) 使用处理后的数据训练TensorFlow模型。
- **模型评估：** (`Evaluator`) 对模型性能进行深度分析，并与之前的版本或基准进行比较。
- **部署验证：** (`Pusher`) 检查模型是否通过验证，并将其“推送到”部署目标（如 TensorFlow Serving）。

这些组件通过**工件**进行通信，工件表示一个组件的输出，以及后续组件的输入。工件通常包括数据集、数据架构、统计信息、转换图、训练模型和评估结果。

> 典型的 TFX 流水线结构，它描绘了标准组件及其依赖关系。数据从左到右流动，每个组件都在机器学习工作流程中执行特定任务。

### 协调与元数据管理

TFX 流水线不直接执行；它们由**协调器**运行。流行的协调器包括 Apache Airflow、Kubeflow Pipelines 和 Apache Beam（它也为开发提供了本地运行器）。协调器根据组件的依赖关系管理它们的执行顺序，处理重试，并记录执行详情。

支撑 TFX 的一个重要组成部分是**元数据存储 (MLMD)**。每次流水线运行时，MLMD 都会自动记录关于每个组件执行、生成和使用的工件以及它们之间关系（血缘）的详细信息。这些元数据对于以下方面非常有用：

- **可复现性：** 确切知道哪些输入和代码生成了特定模型。
- **调试：** 通过流水线执行图追溯故障。
- **分析：** 比较不同的流水线运行或模型版本。
- **治理：** 理解模型来源和合规性。

采用 TFX，您可以获得一种结构化、自动化且可靠的方法来管理端到端机器学习 (machine learning)流程。这个框架有助于构建适用于严苛生产环境的机器学习系统。接下来的章节将更详细地检验标准 TFX 组件，说明它们如何相互配合以创建完整的流水线。

## 参考资料

- [TensorFlow Extended (TFX) | Google Developers](https://www.tensorflow.org/tfx) — Google (2024)
  Publisher: Google
  提供 TFX 组件、架构和实践的全面指导。
- [TFX: A TensorFlow-Based Production Machine Learning Platform](https://dl.acm.org/doi/10.1145/3394486.3403328) — Clemens Bayer, Robert Crowe, Alan G. Smith, Alexey Volkov, Zhitao Li, Wei-Ling Kao, Neoklis Polyzotis, Nick Holt, Martin Wicke, and Ben Chambers (2020)
  Journal: Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining; Publisher: ACM; Pages: 3505-3513; DOI: [10.1145/3394486.3403328](https://doi.org/10.1145/3394486.3403328)
  介绍 TFX 作为生产机器学习端到端平台的动机、架构和组件。
- [Building Machine Learning Pipelines: Automating Model Life Cycles with TensorFlow Extended (TFX)](https://www.oreilly.com/library/view/building-machine-learning-pipelines/9781492078422/) — Hannes Hapke, Catherine Nelson, and Taehoon Kim (2020)
  Publisher: O'Reilly Media
  提供 TFX 及其在构建生产 ML 管道中应用的详细实践指南。
- [Machine Learning Engineering](https://andriy.burkov.ca/ml-eng/) — Andriy Burkov (2020)
  Publisher: True Positive Inc.; Pages: 310
  提供部署和管理 ML 系统的通用原则和挑战，包括数据一致性、模型治理和可复现性。

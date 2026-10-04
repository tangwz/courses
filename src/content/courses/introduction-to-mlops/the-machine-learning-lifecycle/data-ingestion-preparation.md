---
course: "introduction-to-mlops"
chapter: "the-machine-learning-lifecycle"
lesson: "data-ingestion-preparation"
sourceId: 7189
sourceUrl: "https://apxml.com/zh/courses/introduction-to-mlops/chapter-2-the-machine-learning-lifecycle/data-ingestion-preparation"
title: "数据摄取与准备"
description: "了解机器学习生命周期的第一阶段：收集、清洗并为模型训练准备数据。"
order: 2
plots: []
sourceHash: "b21dd9998febb82f65b22b3e47a24498e0f8bbfde75e6c149e50a70971de8466"
sourceCorrections: []
---

每个机器学习 (machine learning)模型的生命周期都始于数据。在模型学习预测、分类或聚类之前，它需要高质量、相关的信息作为学习材料。获取这些原始信息并将其转换为干净、结构化的格式是机器学习生命周期中的一个基础过程。这个过程被称为数据摄取与准备，它为后续的所有环节打下了基础。无论模型多么先进，如果在这个环节出错，整个系统都会受到影响。

### 数据摄取：获取原材料

数据摄取是指获取并导入数据以供系统使用的过程。可以把它想象成为工厂采购原材料。数据可能存在于不同的地方，格式也各不相同，但目标是将其汇集到一个集中的位置，以便您的机器学习 (machine learning)流水线可以访问。

常见的数据来源包括：

- **数据库：** 存储在 SQL（如 PostgreSQL、MySQL）或 NoSQL（如 MongoDB）数据库中的结构化数据。
- **数据仓库和数据湖：** 为分析而设计的各类大型集中式存储库，如 BigQuery、Redshift 或 Snowflake。
- **文件系统：** 以独立文件形式存储的数据，如 CSV、JSON 或 Parquet，通常位于 Amazon S3 或 Google Cloud Storage 等云存储中。
- **流式平台：** 来自 Apache Kafka 或 API 等来源的实时数据流，提供连续不断的信息。

从 MLOps 的角度来看，摄取过程必须是**可靠**且**可重复**的。您不仅仅是下载一次 CSV 文件，而是在构建一个自动化流程，该流程可以按计划或响应某个事件来获取新数据。这确保了当您需要重新训练模型时，可以使用完全相同的程序提取最新数据。

### 数据准备：从原始到就绪

原始数据很少能直接适用于机器学习 (machine learning)模型。它通常是杂乱、不完整且非结构化的。数据准备阶段（也称为数据预处理）涉及将这些原始数据清洗并转换为模型能够理解的一组信息特征。

这可以说是机器学习生命周期中最耗时但也最有效的环节之一。准备好的数据质量直接影响模型的性能和可靠性。让我们来看看其中的主要步骤。

#### 1. 数据清洗

第一步是整理数据集。这通常涉及处理一些常见问题：

- **缺失值：** 许多数据集都存在缺失。您可以选择用占位符（如零、平均数或中位数）填补这些空白，或者如果某一行缺失信息过多，则将其完全删除。
- **错误或不一致的数据：** 这可能包括分类数据中的拼写错误（例如，“NY”与“New York”）或不可能的数值（例如，人的年龄为 200 岁）。这些都需要修正或删除。
- **重复值：** 重复的行会使模型产生偏见，导致它赋予重复数据点过高的权重 (weight)。这些通常会被移除。

#### 2. 特征工程

特征工程是从现有数据中创建新的、更具代表性的特征的技术。其目标是更好地表达数据中的潜在模式，使模型的工作变得更容易。

例如，如果您的原始数据包含像 `2023-10-27 10:00:00` 这样的时间戳，模型可能无法直接使用它。通过特征工程，您可以从中创建几个新特征：

- `hour_of_day`（当日小时）：10
- `day_of_week`（星期几）：5（代表周五）
- `is_weekend`（是否周末）：0（代表否）

这些新特征通常比原始时间戳更具预测性。

#### 3. 数据转换

机器学习算法处理的是数字，因此所有特征都必须转换成合适的数值格式。这涉及两种主要的转换类型：

- **数值特征缩放：** 当所有数值特征都在类似的量级上时，许多模型的表现会更好。例如，如果您有一个年龄特征（如 20-65）和另一个收入特征（如 50,000-200,000），收入特征较大的量级可能会导致模型错误地认为它更重要。缩放技术如归一化 (normalization)（将值缩放到 0 到 1 的范围）或标准化（缩放至均值为 0，标准差为 1）可以解决这个问题。
- **类别特征编码：** 基于文本的分类，如 "颜色": ["红色", "绿色", "蓝色"]，需要转换为数字。一种常用的技术是**独热编码 (one-hot encoding)**，它为每个类别创建一个新的二进制（0 或 1）列。

例如，“颜色”特征将变成三个新特征：

| 颜色\_红色 | 颜色\_绿色 | 颜色\_蓝色 |
| --- | --- | --- |
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |

### MLOps 方法：自动化准备工作

在传统的数据科学项目中，这些准备步骤可能在 Jupyter notebook 中执行一次。但在 MLOps 中，这还不够。为什么？因为您应用于训练数据的相同准备步骤，也必须应用于部署后的模型进行预测时所使用的任何新数据。

如果训练和推理 (inference)时的准备逻辑不同，就会引入**训练-推理偏差 (training-serving skew)**。这是一种常见的失败模式，即模型在测试中表现良好，但在生产环境中由于实时数据的处理方式不同而失败。

解决方案是将所有数据准备逻辑作为**代码**封装在自动化的、受版本控制的流水线中。这确保了每个数据点，无论是用于训练还是推理，都会通过完全相同的清洗和转换步骤。

> 数据准备流水线将来自各种源的原始数据转换为适合模型训练的干净且结构化的格式。

通过将数据准备视为机器学习 (machine learning)生命周期中一个版本化、自动化的组件，您可以构建一个可可靠更新和维护的系统。这种结构化的方法是 MLOps 的核心原则，也是从手动、一次性数据处理迈出的重要一步。有了准备好的干净数据，我们现在可以进入下一阶段：模型训练。

## 参考资料

- [Introducing MLOps: How to go from Model to Production](https://www.oreilly.com/library/view/introducing-mlops/9781098104036/) — Hien Luu, Stephen W. Kalinin, and Boris Lublinsky (2022)
  Publisher: O'Reilly Media
  解释MLOps生命周期以及数据摄取和准备在生产机器学习系统中的作用。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  讨论构建数据系统的原则，这些原则对MLOps中可靠、可扩展的数据摄取至关重要。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  提供数据清洗、特征工程和数据转换技术用于机器学习模型的实践指导。
- [Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) — Alice Zheng and Amanda Casari (2018)
  Publisher: O'Reilly Media
  提供多种技术和策略，用于从原始数据创建特征以提高模型性能。

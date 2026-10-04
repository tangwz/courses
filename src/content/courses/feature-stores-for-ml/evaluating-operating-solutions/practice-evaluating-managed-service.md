---
course: "feature-stores-for-ml"
chapter: "evaluating-operating-solutions"
lesson: "practice-evaluating-managed-service"
sourceId: 3462
sourceUrl: "https://apxml.com/zh/courses/feature-stores-for-ml/chapter-6-evaluating-operating-solutions/practice-evaluating-managed-service"
title: "动手实践：评估托管服务"
description: "对托管特征商店服务进行结构化评估，并基于既定标准。"
order: 7
plots: []
sourceHash: "a7b09e1e67653b613283865910d8d8b5a8f2b58c52b5d749dbb28e69f38198f9"
sourceCorrections: []
---

在比较了开源和托管特征商店、以及决定自建或购买的框架之后，本节提供了一项实践练习，以评估托管特征商店服务。理论比较有价值，但动手经验必不可少，以便了解特定服务如何适应团队工作流程，与现有基础设施协同，并满足您的性能和管理要求。

这项实践练习将引导您完成结构化的评估流程。我们将使用Amazon SageMaker特征商店作为具体例子，但其原理和评估标准同样适用于其他托管服务，例如Google Cloud的Vertex AI特征商店或Azure机器学习 (machine learning)托管特征商店。本次练习的目标并非使您成为某一特定服务的专家，而是培养一种可重复的方法来评价任何托管产品。

### 先决条件

在开始本次评估之前，请确保您具备以下条件：

1. **云账户访问权限**：访问AWS、GCP或Azure账户，并拥有创建和管理机器学习 (machine learning)资源的权限，包括相应的托管特征商店服务。
2. **对云平台基本熟悉**：了解您选择的云提供商的基本控制台导航、IAM（或等效的身份管理）以及存储服务（如S3、GCS或Azure Blob存储）。
3. **Python环境**：一个可用的Python环境（例如，本地机器、基于云的笔记本实例），并安装了必要的SDK（例如，AWS的`boto3`、GCP的`google-cloud-aiplatform`、Azure的`azure-ai-ml`）。
4. **示例数据**：一份小型、具代表性的数据集，可用于定义和摄取特征。这可以是一个存储在云存储中的简单CSV文件。例如，一个包含用户ID、人口统计信息、购买历史时间戳和交易金额的客户数据集。

### 评估标准

一次全面的评估应涵盖多个维度。在您动手评估时，请使用以下标准作为清单：

- **设置与配置**：特征商店的配置有多容易？权限设置是否直观？控制台界面与SDK相比，哪个更易用？
- **特征定义**：特征组（或等效物）如何定义？支持哪些数据类型（基本类型、数组、嵌入 (embedding)向量 (vector)）？能否轻松指定事件时间以实现时间点正确性？
- **数据摄取**：数据如何摄取到离线和在线存储中？是否良好支持从数据湖/数据仓库的批量摄取？流式摄取如何处理？性能特点和故障处理机制是什么？
- **离线存储访问**：查询历史数据以生成训练数据集有多容易？它支持时间点正确的连接吗？它如何与查询引擎（例如Athena、BigQuery、Synapse）协同？
- **在线存储性能**：获取单个实体特征的典型延迟是多少？批量获取呢？提供哪些一致性保证（例如，最终一致性、强一致性）？它如何随请求负载扩展？
- **数据质量与一致性**：数据摄取过程中是否有内置的数据校验机制？服务是否提供用于检测训练/服务偏差的工具或协作方案？
- **治理与安全**：服务是否支持特征版本控制？血缘追踪是否可用并可视化？访问控制的粒度如何？它如何与平台的身份管理和加密服务协同？
- **MLOps协同**：特征商店与模型训练管道的协同有多容易？它如何与模型服务终端协同？是否支持特征定义的CI/CD工作流程？
- **监控与可观测性**：哪些指标会自动公开（例如，延迟、错误率、摄取计数）？设置警报有多容易？它是否与云提供商的标准监控工具良好协同？
- **成本模型**：定价结构是否清晰且可预测？主要成本驱动因素是什么（例如，存储、API调用、数据处理）？是否有工具来估算成本？
- **文档与支持**：官方文档是否全面、准确且易于理解？是否有足够的示例和教程？提供哪些支持渠道？

### 评估步骤（以AWS SageMaker特征商店为例）

让我们通过使用SageMaker特征商店来演练一个简化的评估流程。请记住，将这些步骤适应于其他云提供商，因为术语和具体API会有所不同。

**第1步：定义一个简单用例**

想象一个客户流失预测场景。我们需要客户的特征，例如：

- `customer_id`（实体ID）
- `age`（人口统计）
- `account_length_days`（静态）
- `total_monthly_charges`（定期更新）
- `last_support_interaction_timestamp`（事件时间）
- `num_support_tickets_last_30d`（时间窗聚合）

准备一个小型CSV文件，其中包含一些客户的示例数据，每条记录都包含一个事件时间戳。将其上传到S3。

**第2步：设置与初始配置**

1. 在您的AWS账户中导航到SageMaker控制台。
2. 创建一个SageMaker特征组。您需要定义：
   - 特征组名称（例如，`customer-churn-features`）
   - 记录标识符名称（`customer_id`）
   - 事件时间特征名称（`last_support_interaction_timestamp`）
   - 特征定义（每个特征的名称和数据类型：`age`为`Integral`，`total_monthly_charges`为`Fractional`等）
   - 在线/离线存储配置（本次评估启用两者）。
   - 具有必要权限的IAM角色（SageMaker提供托管策略）。
3. *评估*：评价控制台界面的清晰度。记录标识符和事件时间的理解是否容易？权限设置是否直观？尝试在笔记本中使用AWS SDK（`boto3`）创建类似的特征组。比较其易用性。

**第3步：定义和摄取特征**

1. 使用AWS SDK（`boto3`），调用`put_record` API将单个记录摄取到在线存储中（模拟实时更新）。
2. 对于批量摄取，您可以使用SageMaker处理作业或AWS Glue从S3读取您的示例CSV，并将数据摄取到离线存储中，也可能同时回填在线存储。SageMaker特征商店为此提供了辅助功能。
3. *评估*：`put_record` API有多直观？如果您尝试摄取架构不正确的数据会发生什么？设置批量摄取管道的复杂度如何？根据您的需求（例如，对于嵌入 (embedding)向量 (vector)或复杂类型）检查支持的数据类型。

**第4步：模拟训练数据生成**

1. 使用SageMaker SDK或Athena连接器查询*离线*存储。构建一个基于`customer_id`连接特征，并使用`event_time`特征执行时间点查找的查询。例如，生成一个数据集，使其看起来像过去不同时间点的数据。
2. *评估*：构建时间点正确查询有多容易？查询小型数据集的性能如何？考虑这会如何扩展（基于文档和底层技术，例如S3/Glue数据目录/Athena）。

**第5步：模拟在线服务**

1. 调用`get_record` API（通过SDK）从*在线*存储中获取特定`customer_id`的最新特征向量。
2. 计时这些请求。如果可能，模拟并发请求，以感受负载下的延迟（即使是近似值）。
3. 将从在线存储中获取的特征值与为该`customer_id`摄取的最新值进行比较。
4. *评估*：`get_record`有多快？API使用是否简单？在线存储是否按预期提供了*最新*特征值？查阅有关一致性模型的文档（SageMaker FS旨在摄取后实现强一致性）。

**第6步：查看治理与监控**

1. 在SageMaker控制台中，检查特征组详细信息。查找与版本控制相关的选项（注意：SageMaker特征商店处理架构演进，但与其他一些工具不同，它不提供显式的特征定义版本控制；评估这一限制）。
2. 检查AWS CloudTrail日志中与特征商店相关的API调用。
3. 查看Amazon CloudWatch中SageMaker特征商店发布的指标（例如，`GetRecord.Latency`、`PutRecord.SuccessCount`）。
4. 审查与特征商店访问相关的IAM策略。
5. *评估*：查找相关指标有多容易？这些指标是否足以满足操作监控需求？访问控制如何管理？血缘信息是否可用，或者是否易于与其他AWS服务协同？

**第7步：分析成本**

1. 查看AWS SageMaker定价页面，特别是特征商店部分。识别成本组成部分：存储（在线/离线）、读/写单元（在线）、数据处理（摄取）和API请求。
2. 使用AWS定价计算器根据您预期的使用情况（例如，特征数量、实体数量、更新频率、读取频率）估算成本。
3. *评估*：定价模型是否清晰？您能否轻松估算预期工作负载的成本？是否存在潜在的隐性成本？

**第8步：查阅文档**

1. 浏览官方SageMaker特征商店开发人员指南。
2. 查找与您的用例相关的具体示例（例如，流式摄取、时间点查询）。
3. 检查API文档的清晰度。
4. *评估*：文档是否全面？示例是否有用且保持更新？是否容易找到特定问题的答案？

### 创建评估记分卡

为了使您的发现规范化，请创建一个简单的记分卡。列出评估标准并为每个标准分配一个评分（例如，1-5，差-优）或写下定性说明。这提供了一个结构化的摘要，并且如果您评估多个服务，则有助于比较。

| 评价标准 | 服务 (例如，SageMaker FS) | 评分/说明 |
| --- | --- | --- |
| 设置与配置 | SageMaker FS | 控制台：良好 (4/5)。SDK：需要一些AWS知识 (3/5)。权限清晰。 |
| 特征定义 | SageMaker FS | 理解清晰 (ID, 事件时间)。基本类型支持良好。复杂类型支持有限。 |
| 数据摄取 (批量/流式) | SageMaker FS | `put_record`易用。批量需要设置Glue/处理作业 (中等难度)。 |
| 离线存储访问 (PIT) | SageMaker FS | 与Athena良好协同。PIT查询通过SQL直接。 |
| 在线存储性能 | SageMaker FS | 观察到低延迟 (`get_record`)。扩展性声明需大规模验证。 |
| 数据质量与一致性 | SageMaker FS | 内置校验有限。偏差检测需要外部工具。 |
| 治理与安全 | SageMaker FS | 强大的IAM协同。无显式特征版本控制。通过SDK/标签实现血缘。 |
| MLOps 协同 | SageMaker FS | 与SageMaker训练/终端良好协同。CI/CD通过CloudFormation/SDK。 |
| 监控与可观测性 | SageMaker FS | 良好的CloudWatch指标协同。标准AWS日志。 |
| 成本模型 | SageMaker FS | 组成部分清晰。读/写/存储可预测。 |
| 文档与支持 | SageMaker FS | 总体良好，全面。有示例。 |

> 总结特定托管服务评估结果的示例记分卡。

### 解读结果

“最佳”托管特征商店在很大程度上取决于您的具体情况：

- **现有生态系统**：它与您当前的云环境、数据仓库和MLOps工具协同得如何？坚持使用单一云提供商通常会简化协同。
- **团队专业知识**：您的团队是否拥有特定云提供商及其服务的经验？学习曲线可能是一个重要因素。
- **特征需求**：该服务是否支持您需要的特定数据类型（例如，嵌入 (embedding)向量 (vector)）、转换或一致性模型？
- **规模与性能需求**：该服务能否满足您在线服务对延迟的要求以及离线处理对吞吐量 (throughput)的要求？
- **预算**：成本模型是否符合您的预算限制？
- **治理需求**：该服务是否满足您对血缘、版本控制和访问控制的要求？

这项动手评估提供了做出明智决策所需的具体数据，超越了营销宣传和理论比较，以了解服务如何根据*您的*需求在实践中表现。对其他有前景的托管服务重复此过程，以便在投入大量资源之前建立比较性理解。

## 参考资料

- [Amazon SageMaker Feature Store Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html) — Amazon Web Services (AWS) (2024)
  Publisher: Amazon Web Services
  执行 Amazon SageMaker Feature Store 实际评估的必备文档。
- [Vertex AI Feature Store documentation](https://cloud.google.com/vertex-ai/docs/featurestore/overview) — Google Cloud (2024)
  提供 Google Cloud 托管特征商店服务的详细信息，对进行比较评估很有用。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready AI](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  提供对特征商店及其在机器学习系统设计中地位的基础理解，为构建或购买决策提供参考。
- [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — Google Cloud (2024)
  详细介绍了 MLOps 原则，这对于评估特征商店的运营和集成能力至关重要。

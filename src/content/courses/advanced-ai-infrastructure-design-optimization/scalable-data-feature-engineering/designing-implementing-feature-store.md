---
course: "advanced-ai-infrastructure-design-optimization"
chapter: "scalable-data-feature-engineering"
lesson: "designing-implementing-feature-store"
sourceId: 7014
sourceUrl: "https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-5-scalable-data-feature-engineering/designing-implementing-feature-store"
title: "特征存储的设计与实施"
description: "特征存储的架构原则，以实现训练和服务之间的一致性，并分析了Feast等工具。"
order: 1
plots: []
sourceHash: "af50cb73800233e4dc961bf8c3adb60f387ce2115474bc0e45a093bae5a9edb6"
sourceCorrections: []
---

生产环境中模型默默退化的主要原因是训练-服务偏差。训练和推理 (inference)环境之间数据处理的这种差异，造成了模型在实验室环境中的表现与在提供实时预测时的实际效果之间的差距。特征存储是一个专门的数据系统，旨在通过创建一个集中式、带版本管理且支持双向访问的机器学习 (machine learning)特征库来弥合这一差距。

特征存储不仅仅是一个数据库。它是一种架构模式，系统地将特征工程与模型训练和模型服务分离。它为特征定义和特征值提供了单一真实来源，确保用于生成训练数据集特征的精确相同转换逻辑，也应用于在线推理时的低延迟查找。这严格执行了以下原则：用于训练的特征向量 (vector) $f(x_{训练})$ 与服务时使用的特征向量 $f(x_{服务})$ 生成方式完全一致。

### 特征存储架构的核心组成部分

生产级特征存储由多个相互关联的组件构成，每个组件在特征生命周期中都有其明确的用途。理解这种架构是实施或选择适合您MLOps平台解决方案的基础。

> 特征存储架构中的数据流。原始数据由转换作业处理，这些作业向用于训练的离线存储和用于服务的在线存储填充数据。特征注册表管理定义，确保所有消费者的数据一致性。

让我们详细分析每个组件：

- **特征注册表 (Feature Registry):** 这是元数据层和中心目录。它存储所有特征的定义和Schema，包括它们的名称、数据类型、版本和所有权。注册表作为一个约定，允许数据科学家发现可用的特征，并为数据摄取和检索管道提供一致的定义。
- **离线存储 (Offline Store):** 离线存储旨在存储大量历史特征数据。其主要目的是为生成训练集提供数据。由于训练集生成是批处理过程，可以容忍较高的延迟，因此离线存储通常基于可扩展、成本效益高的技术构建，例如数据湖（例如，S3，带有Parquet文件的GCS）或云数据仓库（例如，BigQuery，Snowflake，Redshift）。
- **在线存储 (Online Store):** 相比之下，在线存储针对低延迟、单行查询进行优化。当推理 (inference)服务收到请求时，它需要在毫秒级时间内获取相应的特征向量 (vector)。这需要高吞吐量 (throughput)的键值数据库，如Redis、DynamoDB或Cassandra。在线存储只保存每个特征的最新值，不包含完整历史。
- **转换与摄取 (Transformation and Ingestion):** 这个组件包含将原始数据转换为特征值的逻辑。一个重要的设计原则是只定义一次此逻辑并重复使用。
  - **批处理摄取 (Batch Ingestion)** 管道通常按计划（例如每日）使用Apache Spark或Ray等框架运行。它们读取大量原始数据，计算特征，并将它们加载到离线存储中。它们还可以将最新值回填或“实例化”到在线存储中。
  - **流式摄取 (Streaming Ingestion)** 管道从Kafka或Kinesis等源近实时地处理数据。它们实时计算特征并将它们直接推送到在线存储，以确保推理的特征时效性。这些值也经常归档到离线存储。
- **检索API (Retrieval APIs):** 特征存储暴露两种不同的数据检索API。训练API允许数据科学家选择特征列表和一组实体（例如，用户ID和时间戳），以从离线存储生成“时间点一致”的训练数据集。服务API提供高性能、低延迟的方法，从在线存储中检索特定实体ID的特征向量。

### 确保时间点一致性

创建训练数据的一个重要难题是数据泄露，即来自未来的信息无意中影响了训练样本。例如，如果您正在构建预测客户流失的模型，则不得使用在该客户流失状态记录日期*之后*生成的特征。

特征存储通过促成**时间点连接**来解决此问题。当您请求训练数据集时，您提供实体列表（例如，`customer_id`）以及每个观察值对应的`时间戳`。特征存储的检索逻辑查询离线存储，以查找在每个指定时间戳*或之前*有效的最新特征值。

例如，要在`2023-01-15`为`customer_123`生成训练行，系统将检索`avg_monthly_spend`在该日期时的值，忽略在`2023-01-16`或之后发生的任何更新。这阻止模型从未来信息中学习，并确保训练环境准确地模拟预测时刻可用的数据。

### 实施时的考量

集成特征存储时，您面临经典的“自建”还是“购买”的决策。

- **托管服务 (Managed Services):** 云提供商提供成熟的托管解决方案，如Google Cloud的Vertex AI Feature Store、Amazon SageMaker Feature Store和Databricks Feature Store。这些服务处理在线和离线存储的底层基础设施，降低运营开销。
- **开源解决方案 (Open Source Solutions):** [Feast](https://feast.dev/) 和 [Tecton](https://www.tecton.ai/) 等框架提供了定义和提供特征的开源标准。例如，Feast允许您以编程方式定义特征，并且可以配置为使用您现有基础设施（例如，离线使用Snowflake，在线使用Redis）。这提供更大的灵活性并避免供应商锁定。

Feast中一个简单的特征定义可能如下所示：

```python
# feature_repo/ 目录中的特征视图
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

# 定义一个我们要计算特征的实体
driver = Entity(name="driver_id", description="网约车司机的ID")

# 定义我们原始数据的来源
driver_stats_source = FileSource(
    path="data/driver_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

# 定义特征视图，它将相关特征分组
driver_stats_fv = FeatureView(
    name="driver_hourly_stats",
    entities=[driver],
    ttl=timedelta(days=1),
    schema=[
        Field(name="conv_rate", dtype=Float32),
        Field(name="acc_rate", dtype=Float32),
        Field(name="avg_daily_trips", dtype=Int64),
    ],
    online=True,
    source=driver_stats_source,
    tags={"team": "driver_performance"},
)
```

这种声明式方法将特征逻辑与应用程序代码分离。应用此定义后，可以使用Feast的CLI或客户端库，将这些特征从 `FileSource`（离线）实例化到已配置的在线存储中，使其可用于训练集生成和在线服务，并保证一致性。通过集中管理此逻辑，特征存储成为您生产ML平台的数据支柱，提高可靠性并加速模型开发周期。

## 参考资料

- [Feast Documentation](https://feast.dev/docs/) — Feast Maintainers (2025)
  Publisher: GitBook
  Feast开源特征存储的官方文档，全面介绍其架构、组件和实际使用方法。
- [Vertex AI Feature Store overview](https://cloud.google.com/vertex-ai/docs/featurestore/overview) — Google Cloud (2024)
  Publisher: Google Cloud
  Google Cloud托管特征存储服务的概述，展示了机器学习特征管理的云解决方案方法。
- [Machine Learning Engineering](https://www.mlebook.com/start) — Andriy Burkov (2020)
  Publisher: True Positive Inc.; Pages: 310
  一本涵盖MLOps和生产机器学习系统各个方面的书籍，包含特征工程和特征存储的讨论。
- [Hidden Technical Debt in Machine Learning Systems](https://dl.acm.org/doi/10.5555/2969442.2969519) — D. Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar Ebner, Vinay Chaudhary, Michael Young, Jean-François Crespo, and Dan Dennison (2015)
  Journal: Advances in Neural Information Processing Systems; Publisher: Neural Information Processing Systems Foundation, Inc. (NeurIPS); Volume: 28; Pages: 2503-2511; DOI: [10.5555/2969442.2969519](https://doi.org/10.5555/2969442.2969519)
  一篇基础性论文，指出机器学习系统中的常见技术债务来源，包括与数据依赖和特征管理相关的问题，而特征存储旨在解决这些问题。

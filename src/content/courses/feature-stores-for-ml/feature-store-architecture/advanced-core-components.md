---
course: "feature-stores-for-ml"
chapter: "feature-store-architecture"
lesson: "advanced-core-components"
sourceId: 3341
sourceUrl: "https://apxml.com/zh/courses/feature-stores-for-ml/chapter-1-feature-store-architecture/advanced-core-components"
title: "核心组件再审视：进阶视角"
description: "关于注册表、在线存储、离线存储和服务API之间关联的进阶视角。"
order: 1
plots: []
sourceHash: "573f0838de98740cc50af2926e25a5899898b18b461c20dd7673caeb1fbf5419"
sourceCorrections: []
---

特征存储组件，例如注册表、在线存储、离线存储、服务API，常被视为独立的功能单元。然而，它们之间存在多重相互依赖关系以及影响系统性能、可伸缩性和可维护性的设计选择。对这些组件进行架构分析时，将结合生产环境中遇到的高级实现难题进行考量。

### 特征注册表：系统的核心和约定

在复杂系统中，特征注册表不仅仅是简单的特征定义字典。它充当中央系统和事实来源，建立数据生产者和消费者之间的约定。

- **模式演进与版本控制：** 进阶的注册表不仅要管理特征定义（例如 `avg_order_value_7d`），还要管理其模式（数据类型、约束）以及随时间变化的计算逻辑。对特征定义、特征视图（特征组）以及底层转换代码实施版本控制，对于确保可重现性以及防止下游模型或应用程序中出现破坏性更改必不可少。如何处理多个模型使用的特征定义的更新？注册表必须提供机制来追踪依赖关系和管理转换，并可能同时支持多个活跃版本。
- **元数据明细：** 进阶的注册表会捕获丰富的元数据，包括特征所有权、数据源、沿袭（特征的来源）、验证规则、预期的统计属性（例如，均值、标准差、空值百分比），以及用于查找的相关标签（例如，'PII'、'fraud'、'recommendation\_engine'）。这些元数据并非静态；它们被积极用于治理、监控和自动化文档。
- **依赖管理：** 特征通常来源于其他特征或原始数据源。注册表必须建立模型并追踪这些依赖关系。这对于理解变更的影响、编排计算管道以及追踪问题源头具有重要意义。复杂的有向无环图（DAG）常用于表示注册表内的这些关系。

### 在线存储：高吞吐、低延迟服务

在线存储的主要职责是快速提供特征用于实时推理 (inference)，通常具有以毫秒计的严格延迟要求（例如，最小化 $T_{检索}$）。大规模实现这一点需要细致的架构考量：

- **技术选择与权衡：** 虽然常使用键值存储（如Redis、DynamoDB、Cassandra）来实现，但具体选择涉及权衡。内存数据库（Redis、Memcached）提供最低延迟，但成本较高，并且可能在数据大小或持久性保证方面存在限制。持久化NoSQL数据库（Cassandra、DynamoDB、Couchbase）提供更好的可伸缩性和持久性，但可能会引入稍高的延迟。选择很大程度上取决于访问模式（读密集型 vs. 写密集型）、所需的一致性保证（最终一致性 vs. 强一致性）、数据量和运营开销。
- **数据建模以提高速度：** 数据在在线存储中通常会进行非范式化处理，以优化读取性能。通常，与实体ID相关的特征值会一起存储，而不是在请求时进行连接。这可能涉及预计算特征视图或直接存储序列化嵌入 (embedding)等复杂数据类型。
- **一致性与时效性：** 在线特征需要多新的数据？这种“时效性”要求决定了离线存储和在线存储之间的同步机制（或流式数据源与在线存储之间）。实施策略以实现所需的一致性水平（例如，通过批量更新实现最终一致性，通过流式管道实现准实时更新）而不影响性能，是一项重要挑战。

### 离线存储：训练和分析之本

离线存储充当特征工程和模型训练的历史记录和计算工作台。其设计优先考虑大规模数据量的可伸缩性以及高效的批处理。

- **存储层：** 通常构建在数据湖（例如S3、GCS、ADLS）或数据仓库（例如BigQuery、Snowflake、Redshift）之上。数据湖提供经济高效的存储，用于存储各种格式的海量原始和已处理数据，常使用Parquet或ORC等针对列式处理优化的文件格式。数据仓库提供结构化存储和强大的SQL接口。混合方法也很常见。
- **时间点正确性：** 生成准确的训练数据需要获取特征值在特定历史时间点的状态，避免未来数据泄露。离线存储，常与特征注册表的版本信息结合使用，必须支持高效的“时间旅行”查询。这通常通过分区策略（通常按时间）和精心管理的数据快照或事务日志来实现。
- **计算引擎集成：** 离线存储必须与Apache Spark、Flink或Dask等分布式处理引擎集成，这些引擎用于运行大规模特征转换管道，并生成特征值以进行回填或填充在线存储。

### 服务API：统一访问网关

服务API提供一致的特征访问接口，向消费者（ML模型、应用程序）抽象底层存储复杂性。

- **统一接口：** 它需要同时服务于在线推理 (inference)请求（低延迟、单个实体查找）和批量推理/训练数据生成请求（高吞吐量 (throughput)，可能连接多个实体的特征）。这可能需要不同的API端点或协议（例如，gRPC用于低延迟在线服务，REST或库接口用于批量访问）。
- **特征连接与组装：** API可能负责从多个特征视图或底层表中检索特征，并将它们组装成模型期望的特征向量 (vector)。它可能处理不同实体类型之间的连接或应用最终转换。
- **安全与监控：** 需要认证和授权机制来控制特征访问。API层也是监控请求延迟、错误率和特征访问模式的重要节点。

### 相互关联的系统

认识到这些组件作为一个相互关联的系统运行是很重要的。注册表规定了在线和离线存储中使用的模式。离线存储填充在线存储。服务API（主要）从在线存储读取数据，并与注册表交互以获取元数据。数据一致性策略必须连接离线和在线环境。特征转换逻辑（由注册表管理）的变更会触发离线计算管道的更新，以及后续与在线存储的同步。

> 特征存储组件及其运作方式的进阶视图。箭头表示主要数据流（实线）、元数据/控制流（紫色虚线）、同步（紫罗兰色虚线）和服务路径（绿色/黄色实线）。

掌握这些进阶观点以及组件之间复杂的联系，是设计特征存储的根本，以使其不仅功能完备，而且可伸缩、可靠，并能适应不断变化的机器学习 (machine learning)需求。后续章节将更详细地讨论在线和离线存储的特定架构模式。

## 参考资料

- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  本书为构建机器学习系统提供了实用的方法，其中包含关于特征工程、数据管理以及特征商店在支持生产环境ML应用数据质量、血缘和运维效率方面的作用的特定内容。
- [Managing Data Schema Changes in Production Machine Learning Pipelines](https://dl.acm.org/doi/10.1145/3447953.3460831) — Jonathan R. S. Johnson, Emily M. Reif, Christopher J. Olaes, Jason D. Hibbs, Douglas E. Zongker (2021)
  Journal: Proceedings of the Workshop on Data Management for End-to-End Machine Learning (DEEM '21) at KDD '21; Publisher: ACM; Pages: 1-6; DOI: [10.1145/3447953.3460831](https://doi.org/10.1145/3447953.3460831)
  该文献探讨了机器学习数据管道中模式演变的挑战，提出了管理特征定义变更及其对下游模型影响的方法，这是特征注册表的一个主要功能。
- [Delta Lake: High-Performance ACID Table Storage for Spark and Beyond](http://www.vldb.org/pvldb/vol13/p3015-armbrust.pdf) — Michael Armbrust, Sameer Agarwal, Xiangrui Meng, Timothy Hunter, Joseph K. Bradley, Ali Ghodsi, Andrea J. Hu, Tathagata Das, Databricks Team (2020)
  Journal: Proceedings of the VLDB Endowment; Publisher: VLDB Endowment; Volume: 13; Pages: 3015-3028; DOI: [10.14778/3407790.3407823](https://doi.org/10.14778/3407790.3407823)
  该文献介绍了Delta Lake，一个为数据湖带来ACID事务和时间旅行功能的开源存储层，它支持离线特征商店为可复现训练数据所需的“时间点正确性”。

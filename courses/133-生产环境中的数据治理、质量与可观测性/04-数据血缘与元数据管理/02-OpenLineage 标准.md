# OpenLineage 标准

来源：[原文](https://apxml.com/zh/courses/data-governance-quality-observability-production/chapter-4-data-lineage-metadata-management/openlineage-standard)

[返回章节目录](README.md) · [返回课程目录](../README.md)

标准化是一种使不同系统能有效沟通的机制。数据血缘追踪数据的来源和流向，可分为静态和动态两种形式。静态分析依靠解析代码来确定血缘，而动态血缘则要求数据基础设施在活动发生时自行汇报。为使这些汇报信息有用，现代数据技术栈中的各类工具，例如 Airflow 调度器、Spark 计算引擎和 Snowflake 数据仓库，必须使用同一种通用语言。

OpenLineage 提供了这种语言。它是一个开放 API 规范，规定了如何追踪作业整个生命周期中的数据血缘。OpenLineage 不是在每个工具和每个目录之间建立点对点集成，而是将元数据收集与其使用解耦。各类工具以标准化 JSON 格式发出血缘事件，任何兼容此规范的后端都可以使用、存储并可视化这些信息。

### 核心对象模型

OpenLineage 规范通过三个主要实体建模数据处理过程：**作业 (Job)**、**运行 (Run)** 和 **数据集 (Dataset)**。要正确实现血缘追踪，了解这些实体间的联系是必需的。

1. **作业 (Job)**：一个流程的定义。在数据管道中，它对应于有向无环图 (DAG) 中的特定任务或周期性 SQL 查询。它表示“应该发生什么”。
2. **运行 (Run)**：作业在特定时间点的一次执行实例。它表示“现在正在发生什么”。一个作业会随着时间产生多次运行。
3. **数据集 (Dataset)**：作业的输入和输出。它们可以是数据库表、S3 存储桶或 Kafka 主题。

此模型将数据的抽象流动转化为有向图。一次运行会使用一个或多个输入数据集，并生成一个或多个输出数据集。

> OpenLineage 核心实体之间的关系，描绘了执行如何将输入与输出关联起来。

### 运行事件结构

OpenLineage 中的通信通过**运行事件**发生。当数据管道执行时，集成组件（例如 Airflow 操作符或 Spark 监听器）会向后端发送异步事件。典型的生命周期包含在作业开始时发送一个 `START` 事件，并在作业结束时发送一个 `COMPLETE` 或 `FAIL` 事件。

一个标准的运行事件包含作业的当前状态、该特定运行的唯一标识符以及涉及的输入和输出。我们可以将一个简化的运行事件 $E$ 用数学方式定义为一个元组：

$E = (t, \text{state}, r, j, D_{in}, D_{out})$

其中：

- $t$ 是事件的时间戳。
- $\text{state}$ 是状态 (START, COMPLETE, ABORT, FAIL)。
- $r$ 是运行的全局唯一 ID (UUID)。
- $j$ 是作业标识符（命名空间和名称）。
- $D_{in}$ 是输入数据集的集合。
- $D_{out}$ 是输出数据集的集合。

运行事件的 JSON 负载严格遵循该规范定义的 schema。下方是一个 `COMPLETE` 事件的结构示例，表明作业已完成向某个表的写入。

```json
{
  "eventType": "COMPLETE",
  "eventTime": "2023-10-27T14:23:01.52Z",
  "run": {
    "runId": "d46e465b-d358-4d32-83d4-df660ff614dd"
  },
  "job": {
    "namespace": "production_warehouse",
    "name": "daily_revenue_aggregation"
  },
  "inputs": [
    {
      "namespace": "postgres://db.prod:5432",
      "name": "public.orders"
    }
  ],
  "outputs": [
    {
      "namespace": "snowflake://account.region",
      "name": "analytics.revenue_report"
    }
  ]
}
```

### Facets：元数据最小单位

核心模型描述了图的形态（哪些实体相互连接）。但是，工程团队通常需要更细致的信息。他们需要了解表的 schema、写入的行数、执行的 SQL 查询或使用的代码版本。

OpenLineage 通过 **Facets** 来处理这些信息。Facet 是附加到作业 (Job)、运行 (Run) 或数据集 (Dataset) 的元数据最小部分。Facets 具有模块化和可扩展性。如果某个特定集成无法收集列级血缘，它只需省略该 facet，同时仍可汇报数据集级别的依赖关系。

Facets 根据它们描述的实体进行分组：

1. **作业 Facets**：静态信息，如源代码位置或所有者。
2. **运行 Facets**：运行时信息，如调度时间 (`nominalTime`)、批次 ID 或查询计划。
3. **数据集 Facets**：关于数据本身的信息，如 schema 字段 (`schema`)、列统计信息 (`stats`) 或数据质量断言。

这种模块化特性使得该规范在演进时不会破坏现有的使用者。如果你需要追踪自定义指标，例如“每次查询成本”，你可以定义一个自定义 facet，而无需修改核心规范。

> Facets 将特定的元数据块附加到核心血缘实体，提供关于 schema、统计信息和时间的信息。

### 架构与命名约定

实现 OpenLineage 需要遵循严密的命名约定，以确保血缘图能正确连接。因为数据集可能通过 Spark 使用一个连接字符串访问，而通过 Presto 使用另一个连接字符串访问，所以保持命名一致性对图的完整性有帮助。

该规范规定每个数据集必须有一个 `namespace` 和 `name` 对。

- **Namespace**：通常识别物理上下文 (context)或实例（例如，`postgres://db-prod:5432` 或 `s3://data-lake-bucket`）。
- **Name**：识别该上下文中的资源（例如，`public.users` 或 `/raw/events/2023/`）。

在配置你的生产者（发出事件的工具）时，你必须配置命名空间解析逻辑，使其在整个技术栈中保持一致。如果 Airflow 将数据仓库称为 "Snowflake-Prod"，而 dbt 将其称为 "SNOWFLAKE\_RAW"，那么血缘图将显示两个不连通的节点。

传输架构采用推送模式。客户端（数据工具）实现一个 OpenLineage 集成，它监听内部事件并将其转换为 OpenLineage JSON 格式。然后，这些事件被 POST 到由后端（例如 Marquez、Atlan 或 DataHub）提供的 HTTP 端点。这种设计最小化了对管道的性能影响，因为元数据发送是异步进行的，并且发送元数据失败不会阻塞数据处理作业。

通过此规范实现生产者和消费者的解耦，促成了现代工程实践的“可观测性”方面。我们不再仅仅记录作业失败；而是在失败发生时捕获图的状态，包括输入 schema 和导致错误的特定数据版本。

## 参考资料

- [OpenLineage: Open Standard for Data Lineage](https://openlineage.io/docs/) — OpenLineage Project (2024)
  OpenLineage 规范、对象模型、事件结构和 Facet 的全面指南。
- [Marquez Documentation](https://marquezproject.github.io/marquez/) — Marquez Project (2024)
  一个原生支持和消费 OpenLineage 事件的开源元数据服务的文档。
- [Data Observability: How to Ensure the Reliability of Your Data](https://www.oreilly.com/library/view/data-observability/9781098139587/) — Barr Moses, Lior Gavish (2023)
  Publisher: O'Reilly Media
  讨论了数据可观察性的原则，包括数据血缘在维护数据可靠性和质量方面的作用。

---

[上一节](01-%E9%9D%99%E6%80%81%E4%B8%8E%E5%8A%A8%E6%80%81%E8%A1%80%E7%BC%98.md) · [下一节](03-%E4%BE%9D%E8%B5%96%E5%9B%BE%E7%9A%84%E6%9E%84%E5%BB%BA.md)

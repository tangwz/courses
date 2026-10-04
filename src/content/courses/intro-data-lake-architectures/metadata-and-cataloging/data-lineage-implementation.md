---
course: "intro-data-lake-architectures"
chapter: "metadata-and-cataloging"
lesson: "data-lineage-implementation"
sourceId: 8014
sourceUrl: "https://apxml.com/zh/courses/intro-data-lake-architectures/chapter-4-metadata-and-cataloging/data-lineage-implementation"
title: "数据血缘实现"
description: "记录数据从源到黄金层表的数据流转技术方法。"
order: 4
plots: []
sourceHash: "f1ca9d9200679ffba01d6bf0e0b211818f50b210a03c661e2b58c848d735335d"
sourceCorrections: []
---

在缺乏数据流向可见性的情况下管理数据湖，无异于盲目操作。当黄金层仪表板报告的收入数据不正确时，工程团队通常需要花费数小时，追溯错误在各层转换中的源头。他们需要回答具体的问题：错误是源于白银层聚合逻辑，还是原始青铜层摄取管道出了问题？数据血缘提供了弄清这些依赖关系所需的指引。

血缘是对数据来源及其随时间发生的转换的记录。在现代数据湖架构中，这不仅仅是一个文档工作，更是调试、合规和影响分析的技术要求。当你修改青铜层中的模式时，血缘允许你通过程序确定哪些下游白银层和黄金层表将受影响。

### 血缘图模型

在基本层面，数据血缘表示为一个有向无环图（DAG）。在这个图中，节点代表数据资产（表、文件、视图）以及处理这些资产的流程（Spark 作业、SQL 查询）。边表示数据流向。

我们可以将转换作业 $J$ 定义为一个函数，它接受一组输入数据集 $I$ 并产生一组输出数据集 $O$。

$O = J(I)$

在复杂的环境中，一个作业的输出成为下一个作业的输入。如果作业 $J_1$ 产生数据集 $D_1$，而作业 $J_2$ 使用 $D_1$ 产生 $D_2$，我们便建立了血缘链：$J_1 \rightarrow D_1 \rightarrow J_2 \rightarrow D_2$。

实施血缘追踪时，主要有两种粒度级别：

1. **表级别血缘：** 追踪数据集之间的依赖关系。这会告诉你 `silver_orders` 表源自 `bronze_orders` 和 `bronze_customers`。这足以应对大多数影响分析任务。
2. **列级别血缘：** 追踪特定属性如何传播。这表明黄金层中的 `total_revenue` 列是使用白银层中的 `unit_price` 和 `quantity` 计算得出的。这对于追踪敏感信息（PII）和审计复杂计算不可或缺。

下图说明了 Medallion 架构中的标准血缘流，展示了特定摄取作业如何将原始文件与结构化表关联起来。

> 一个有向图，表示数据从原始源文件经过处理作业到精炼表的流向。

### 收集策略

实施血缘追踪需要一种捕获这些关系的方式。不能依赖手动图表，因为代码一经修改它们就会过时。自动化收集有两种主要技术方法。

#### 静态分析（解析）

此方法涉及在运行前解析定义转换的 SQL 或代码。如果你使用 dbt（数据构建工具）之类的工具或对 Trino 或 Spark SQL 执行纯 SQL，系统会解析查询的抽象语法树（AST）。

考虑以下 SQL 语句：

```sql
INSERT INTO silver.sales_summary
SELECT customer_id, SUM(amount)
FROM bronze.transactions
GROUP BY customer_id
```

解析器分析此字符串并识别出：

- **源：** `bronze.transactions`
- **目标：** `silver.sales_summary`
- **转换：** 对 `amount` 进行聚合

静态分析速度快，即使作业失败也能提供血缘信息。但是，它难以处理命令式代码（例如 Spark 中的 Python 或 Scala 脚本），因为输入/输出路径可能在运行时动态确定。

#### 运行时检测

此方法在作业执行时捕获血缘元数据。这是运行 Spark、Flink 或 Airflow 的数据湖的首选方法。

在 Spark 环境中，你会将 `SparkListener` 附加到应用程序上下文 (context)。当 Spark 驱动程序规划执行图时，它会解析输入文件的精确物理路径和目标目录。监听器捕获这个“逻辑计划”并向中央后端发送血缘事件。

此方法高度准确，因为它记录了实际发生的情况，而不仅仅是代码预期要做的事情。它捕获读取的特定分区和写入的精确行数。

### OpenLineage 标准

为防止厂商锁定，行业已达成共识，采用 **OpenLineage**，这是一个用于血缘信息收集的开放标准。OpenLineage 定义了一个 JSON 模式，用于定义“运行”（即执行过程）、“作业”（即定义）和“数据集”（即输入/输出）。

当你数据湖中实施血缘追踪时，通常会将计算引擎（如 Spark, Trino）配置为向兼容后端（如 Marquez 或 DataHub）发送 OpenLineage 事件。

下面是 Spark 作业完成时可能发送的 OpenLineage 事件的简化表示。

```json
{
  "eventType": "COMPLETE",
  "eventTime": "2023-10-27T10:00:00.000Z",
  "run": {
    "runId": "d46d8fba-23a8-4f9d-9b23-...",
    "facets": {
      "parent": { "job": { "name": "每日 ETL 工作流" } }
    }
  },
  "job": {
    "namespace": "生产分析",
    "name": "白银层处理作业"
  },
  "inputs": [{
    "namespace": "s3://data-lake-bucket",
    "name": "青铜层/订单",
    "facets": {
      "schema": { "fields": [ { "name": "order_id", "type": "STRING" } ] }
    }
  }],
  "outputs": [{
    "namespace": "s3://data-lake-bucket",
    "name": "白银层/已清洗订单",
    "facets": {
      "schema": { "fields": [ { "name": "order_id", "type": "STRING" } ] }
    }
  }]
}
```

这个 JSON 结构使元数据目录能够将独立的作业运行拼接成一个有条理的图。`runId` 关联特定的执行实例，而 `inputs` 和 `outputs` 数组则提供连接图的边。

### 存储与查询

一旦血缘事件被收集，它们必须以支持图遍历的方式存储。传统关系型数据库在查询深层层次关系时通常效率低下（例如，查找一个节点向上追溯10层的所有祖先节点）。

因此，血缘后端常使用图数据库（如 Neo4j 或 Amazon Neptune）或为递归查询做了优化的专门关系型模式。

当你在 AWS Glue Data Catalog 或 DataHub 等编目工具中查看“血缘”选项卡时，后端会执行一个遍历查询。对于给定的节点 $N$，它会查找所有存在路径 $M \rightarrow \dots \rightarrow N$（上游血缘）或 $N \rightarrow \dots \rightarrow M$（下游血缘）的节点 $M$。

### 技术治理集成

血缘的实施与本章前面提及的治理讨论直接相关。通过将血缘追踪与基于角色的访问控制（RBAC）集成，你可以执行传播策略。

例如，如果青铜层中的某个列被标记 (token)为 `PII:True`，血缘追踪使治理引擎能够自动将任何源自该列的下游白银层或黄金层列标记为 `PII:True`。这确保了敏感数据不会仅仅因为数据工程师在转换过程中重命名了列，就意外“泄露”到一个能被普遍访问的报告表中。

## 参考资料

- [OpenLineage Specification](https://openlineage.io/docs/spec/) — OpenLineage Community (2024)
  一个定义了用于捕获、交换和存储数据血缘元数据开放标准的参考文献。
- [Monitoring Applications - Spark Listeners](https://spark.apache.org/docs/latest/monitoring.html#spark-listeners) — Apache Software Foundation (2023)
  涵盖了SparkListener接口，它允许在作业执行期间进行自定义检测和元数据捕获。
- [Fundamentals of Data Engineering: Planning and Building Data Systems](https://learning.oreilly.com/library/view/fundamentals-of-data/9781098108298/) — Joe Reis, Matt Housley (2022)
  Publisher: O'Reilly Media
  提供了数据工程原则的广泛视角，包括当代数据架构中的元数据管理和数据血缘。
- [How dbt works](https://docs.getdbt.com/docs/introduction/how-dbt-works) — dbt Labs (2024)
  Publisher: dbt Labs, Inc.
  解释了dbt如何构建数据模型和转换的DAG，从而自然地支持静态血缘。

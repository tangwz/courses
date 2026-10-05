# 实践操作：构建CDC管道

来源：[原文](https://apxml.com/zh/courses/intro-data-lake-architectures/chapter-3-ingestion-pipelines/practice-building-cdc-pipeline)

[返回章节目录](README.md) · [返回课程目录](../README.md)

处理事务日志并将其应用到数据湖表，这需要从标准追加写入逻辑进行转变。一个常见的基于日志的架构，其目的是通过在目标表上重放一系列变化（包括插入、更新和删除）来重建实体（如客户或订单）的当前状态。以下是使用Apache Spark和Delta Lake实现变更数据捕获（CDC）管道的演示。

我们侧重于“合并”模式。与向数据湖进行标准插入不同，合并操作必须通过主键查找现有记录，并决定是更新行、插入新行还是完全删除该行。这种方法使得数据湖可以作为业务数据库的一个同步副本。

### 管道架构

CDC管道中的数据流通常遵循特定路径：捕获、传输和应用。捕获阶段读取数据库日志（例如Postgres预写日志或MySQL二进制日志）。传输层（通常是Kafka或Kinesis）缓冲这些事件。应用阶段，即我们在此构建的部分，读取事件并更新存储层。

> 该架构流程显示了数据从源事务日志到数据湖中最终目标表的移动过程。

### 定义CDC模式

原始CDC事件通常包含操作元数据以及数据载荷。常见结构包括：

1. **操作类型 (`op`)**：表明变化是创建（`c`）、更新（`u`）还是删除（`d`）。
2. **时间戳 (`ts_ms`)**：变化在源端发生的精确时间。
3. **前置镜像**：变化发生前行的状态（插入操作为null）。
4. **后置镜像**：变化发生后行的状态（删除操作为null）。

在本次实践中，我们模拟传入的客户数据变化流。我们假设数据已摄入到原始DataFrame中，现在需要处理到精炼表中。

### 步骤1：初始化目标表

在处理变化之前，目标表必须存在。在生产环境中，您可能需要执行初始快照加载（引导）来填充该表。在这里，我们初始化一个空的Delta表来代表我们的`customers`数据集。

```python
from delta.tables import *
from pyspark.sql.functions import *

# 定义客户表的模式
# 在实际场景中，此位置将在S3、ADLS或GCS上
table_path = "/tmp/delta/customers"

# 如果Delta表不存在，则创建一个空表
if not DeltaTable.isDeltaTable(spark, table_path):
    spark.createDataFrame([], schema="id INT, name STRING, email STRING, updated_at TIMESTAMP") \
        .write \
        .format("delta") \
        .mode("overwrite") \
        .save(table_path)

target_table = DeltaTable.forPath(spark, table_path)
```

### 步骤2：处理变更流

分布式CDC管道中的一个核心难题是在单个处理批次内处理同一记录的多次变化。如果客户在一分钟内更新了三次电子邮件地址，摄入批次可能包含所有三个事件。

盲目应用这些事件可能导致竞态条件或不正确的最终状态。在合并之前，您必须对传入的微批次进行去重，为每个主键只保留最新变化。

```python
# 模拟的CDC事件批次
# 操作：
# 1. 插入客户101
# 2. 插入客户102
# 3. 更新客户101（修正姓名）
# 4. 删除客户103（假设103之前存在）

cdc_data = [
    (101, "John Doe", "john@example.com", "2023-10-27 10:00:00", "c"),
    (102, "Jane Smith", "jane@test.com", "2023-10-27 10:05:00", "c"),
    (101, "Johnathan Doe", "john@example.com", "2023-10-27 10:10:00", "u"),
    (103, None, None, "2023-10-27 10:15:00", "d")
]

columns = ["id", "name", "email", "updated_at", "op"]
updates_df = spark.createDataFrame(cdc_data, columns)

# 去重逻辑：
# 窗口函数，按时间戳降序为每个ID的变化排序
from pyspark.sql.window import Window

window_spec = Window.partitionBy("id").orderBy(col("updated_at").desc())

deduplicated_updates = updates_df \
    .withColumn("rank", row_number().over(window_spec)) \
    .filter(col("rank") == 1) \
    .drop("rank")
```

在此逻辑中，我们定义了基于主键`id`的窗口，并按`updated_at`排序。通过筛选`rank == 1`，我们确保只有客户101的最终状态（即更新操作）被传递给合并操作，从而丢弃同一批次中的初始插入事件。

### 步骤3：合并操作

有了一组干净的更新，我们将变化应用到目标Delta表中。Delta Lake中的`MERGE`语句使我们能够在单个原子事务中处理插入、更新和删除。这保证了即使作业中途失败，数据湖也能保持一致。

该逻辑遵循以下规则：

1. **匹配**：如果ID在目标中存在且操作为`d`（删除），则删除该行。
2. **匹配**：如果ID存在且操作为`u`（更新），则更新行值。
3. **不匹配**：如果ID不存在且操作不为`d`，则插入新行。

```python
target_table.alias("target") \
    .merge(
        deduplicated_updates.alias("source"),
        "target.id = source.id"
    ) \
    .whenMatchedDelete(
        condition = "source.op = 'd'"
    ) \
    .whenMatchedUpdate(
        set = {
            "name": "source.name",
            "email": "source.email",
            "updated_at": "source.updated_at"
        }
    ) \
    .whenNotMatchedInsert(
        condition = "source.op != 'd'",
        values = {
            "id": "source.id",
            "name": "source.name",
            "email": "source.email",
            "updated_at": "source.updated_at"
        }
    ) \
    .execute()
```

### 性能：混洗与分区

当Spark执行合并时，它必须定位包含匹配ID的文件。如果目标表很大，这可能触发开销大的全表扫描或跨集群的大量数据混洗。

为了优化此过程，数据工程师通常会通过高级属性（如`date`或`region`）对目标表进行分区，或使用Z-Order索引。然而，CDC管道主要操作主键（ID），这通常与分区列不一致。

下方的图表显示了标准合并（搜索所有文件）与分区裁剪合并（仅搜索相关文件）之间的成本差异。虽然在随机访问的UUID更新中我们无法总是通过分区裁剪，但在Delta Lake或Iceberg中启用**删除向量 (vector)**或**布隆过滤器**等功能有助于显著降低I/O开销。



[交互图表：合并性能与文件扫描量](https://apxml.com/zh/courses/intro-data-lake-architectures/chapter-3-ingestion-pipelines/practice-building-cdc-pipeline#plot-kk9zvy)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "合并性能与文件扫描量",
      "font": {
        "size": 16,
        "color": "#495057"
      }
    },
    "xaxis": {
      "title": {
        "text": "目标表大小 (GB)",
        "font": {
          "size": 12
        }
      },
      "showgrid": true,
      "gridcolor": "#dee2e6"
    },
    "yaxis": {
      "title": {
        "text": "执行时间 (秒)",
        "font": {
          "size": 12
        }
      },
      "showgrid": true,
      "gridcolor": "#dee2e6"
    },
    "plot_bgcolor": "white",
    "margin": {
      "t": 50,
      "l": 50,
      "r": 30,
      "b": 50
    },
    "width": 600,
    "height": 400,
    "legend": {
      "x": 0.05,
      "y": 0.95,
      "bgcolor": "rgba(255,255,255,0.8)"
    }
  },
  "data": [
    {
      "x": [
        10,
        50,
        100,
        500,
        1000
      ],
      "y": [
        15,
        45,
        90,
        400,
        900
      ],
      "type": "scatter",
      "mode": "lines+markers",
      "name": "标准合并",
      "line": {
        "color": "#fa5252",
        "width": 3
      }
    },
    {
      "x": [
        10,
        50,
        100,
        500,
        1000
      ],
      "y": [
        12,
        25,
        40,
        120,
        220
      ],
      "type": "scatter",
      "mode": "lines+markers",
      "name": "优化合并（Z-Order）",
      "line": {
        "color": "#228be6",
        "width": 3
      }
    }
  ]
}
```

</details>



> 数据量增长时，标准合并与使用文件跳过技术（Z-Order）的优化合并之间的执行时间对比。

### 处理模式演变

CDC管道对上游变化敏感。如果业务数据库添加了新列，CDC流将包含它，但如果目标模式是固定的，您的合并语句可能会失败。

为了稳妥处理此情况，您可以启用自动模式演变。在Delta Lake中，这通过在Spark会话配置中将`spark.databricks.delta.schema.autoMerge.enabled`配置设置为`true`来实现。

启用此功能后，如果源DataFrame包含目标表中不存在的新列，合并操作将在应用数据之前更改目标表模式以包含这些列。

$\text{合并成本} \approx (\text{数据读取}) + (\text{混洗}) + (\text{重写文件})$

因为合并操作会重写整个文件，即使该文件中只有一行发生变化，所以频繁的小合并可能导致“小文件问题”。最佳实践是避免持续运行CDC合并（例如每秒运行）。相反地，将更新批量处理到5到15分钟的窗口中，可以在数据新鲜度和存储效率之间取得平衡。

### 验证

管道运行后，您检查数据状态。我们模拟数据的预期是：客户101以名称“Johnathan Doe”存在（更新的结果），客户102存在，并且客户103（如果之前存在）被删除。

这种实现模式为数据摄入提供了可靠的基础，确保分析型数据湖准确反映业务实际情况。

## 参考资料

- [MERGE INTO](https://docs.delta.io/latest/delta-update.html#merge-into) — Delta Lake Project (2024)
  提供在Delta表中执行UPSERT/MERGE操作的详细语法、语义和示例，这对于应用变更数据捕获（CDC）事件至关重要。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  一本基础书籍，涵盖分布式系统、事务日志、数据模型和一致性，为基于日志的数据摄取提供了背景信息。
- [Delta Lake: High-Performance ACID Table Storage for Spark and Beyond](https://dl.acm.org/doi/10.1145/3381661) — Michael Armbrust, Sameer Agarwal, Xiangrui Meng, Timothy Hunter, Joseph K. Bradley, Ali Ghodsi, John F. D. Moak, Michael J. Franklin, and David F. Patterson (2020)
  Journal: Proceedings of the ACM on Management of Data (SIGMOD '20); Publisher: ACM; Volume: 4; Pages: 1-10; DOI: [10.1145/3381661](https://doi.org/10.1145/3381661)
  介绍Delta Lake设计、ACID特性以及如何实现可靠数据湖操作（包括UPSERT）的学术论文。
- [Structured Streaming Programming Guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html) — Apache Spark Project (2024)
  详细介绍了如何使用Spark构建可扩展、容错的流应用程序，这是CDC管道“应用”阶段的引擎。

---

[上一节](05-%E6%95%B0%E6%8D%AE%E7%AE%A1%E9%81%93%E4%B8%AD%E7%9A%84%E5%B9%82%E7%AD%89%E6%80%A7.md) · [下一节](../04-%E5%85%83%E6%95%B0%E6%8D%AE%E4%B8%8E%E7%BC%96%E7%9B%AE/01-%E5%85%83%E6%95%B0%E6%8D%AE%E5%AD%98%E5%82%A8%E5%BA%93%E7%9A%84%E4%BD%9C%E7%94%A8.md)

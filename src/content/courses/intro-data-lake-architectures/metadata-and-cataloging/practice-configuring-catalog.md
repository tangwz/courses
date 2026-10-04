---
course: "intro-data-lake-architectures"
chapter: "metadata-and-cataloging"
lesson: "practice-configuring-catalog"
sourceId: 8015
sourceUrl: "https://apxml.com/zh/courses/intro-data-lake-architectures/chapter-4-metadata-and-cataloging/practice-configuring-catalog"
title: "实操：配置数据目录"
description: "动手实践环节，包含定义数据结构和扫描存储桶以填充数据目录。"
order: 5
plots: []
sourceHash: "eabd8e46b62fe54c4ba5f326525862ca2e2ec374e26710a4475db3d07a707b41"
sourceCorrections: []
---

一旦数据摄取管道将Parquet或Avro文件存入对象存储，将这些数据资产注册到元数据存储中就变得非常必要。若没有此注册，查询引擎会把数据湖看作一堆文件，而非结构化数据表。本节将为一个销售数据集手动配置数据目录条目，把逻辑表定义映射到物理对象存储路径。

此过程建立了必要的抽象层，让Trino、Spark SQL或Amazon Athena等SQL引擎能通过分区裁剪来优化查询。

### 定义物理布局

在与数据目录交互前，有必要了解数据在文件系统上的物理组织方式。我们假设电商订单数据集存在于一个S3兼容的存储桶中。数据采用Hive风格的分区策略，即目录名称包含分区列及其值。

结构如下所示：

> 对象存储中分区数据集的层级结构，其中目录代表列值。

目录`dt=2023-10-25`明确告知系统，其中包含的所有文件都属于该特定日期。这种结构对于元数据存储的正确运行是必需的。

### 创建外部表

现在，您将使用数据定义语言（DDL）来创建此数据的逻辑表示。在数据湖环境中，我们几乎只使用`EXTERNAL`表。这种指定方式确保了即使您从数据目录中删除了表定义，存储中的底层Parquet文件也不会被触及。

DDL语句执行三种具体的映射：

1. **数据结构定义：** 列的名称和数据类型。
2. **文件格式：** 关于如何读取二进制数据（例如Parquet）的指令。
3. **位置：** 引擎应该查找数据的根URI。

在您的查询编辑器（Hive、Spark SQL或Athena）中执行以下SQL命令：

```sql
CREATE EXTERNAL TABLE silver_orders (
    order_id STRING,
    customer_id STRING,
    amount DOUBLE,
    status STRING
)
PARTITIONED BY (dt STRING)
STORED AS PARQUET
LOCATION 's3://bucket/silver/orders/';
```

请注意，`dt`是在`PARTITIONED BY`子句中定义的，而不是在主列列表中。这向元数据存储表明，`dt`是一个从目录结构派生的虚拟列，而非存储在Parquet文件中的物理列。

### 填充分区元数据

运行`CREATE TABLE`语句后，您可能会预期`SELECT * FROM silver_orders`会返回数据。然而，结果可能为空。

发生这种情况是因为元数据存储知道表的根目录（`.../silver/orders/`），但尚未将特定的子目录（如`dt=2023-10-25`等）编入目录。查询引擎默认不递归扫描整个存储桶，以避免性能损失。您必须明确指示元数据存储扫描目录结构并注册分区。

#### 方法一：手动分区修复

对于兼容Hive或Spark的引擎，请使用修复命令：

```sql
MSCK REPAIR TABLE silver_orders;
```

此命令会扫描表定义中配置的文件系统路径。它会识别遵循`key=value`模式的文件夹，并将它们添加到分区索引中。

#### 方法二：自动化爬网程序 (AWS Glue)

在AWS等托管环境中，您通常会使用Glue爬网程序。爬网程序连接到数据存储，确定数据结构，并在数据目录中创建或更新元数据表。

当您为路径`s3://bucket/silver/orders/`配置爬网程序时：

1. 爬网程序对文件进行采样以推断数据结构（整型、字符串、双精度浮点型）。
2. 它检测分区结构。
3. 它用表定义和分区位置填充Glue数据目录。

### 验证数据目录状态

一旦分区注册完毕，元数据存储就充当指针系统。当用户运行查询时，引擎会查询数据目录，将逻辑需求解析为物理路径。

下图展示了修复后逻辑表定义与物理分区之间的关系：

> 元数据存储维护一个分区索引，将逻辑分区值映射到物理存储前缀。

您可以通过直接查询分区列表来验证注册情况：

```sql
SHOW PARTITIONS silver_orders;
```

**输出：**

```text
dt=2023-10-25
dt=2023-10-26
```

### 查询性能和裁剪

数据目录配置完成后，您就可以执行分析查询了。此设置的主要优势是**分区裁剪**。

考虑以下查询：

```sql
SELECT sum(amount) 
FROM silver_orders 
WHERE dt = '2023-10-25';
```

因为数据目录明确地将`dt=2023-10-25`映射到特定的S3前缀，查询引擎会跳过所有其他目录。如果您的数据集跨越5年，但您只查询其中一天，引擎将读取大约$\frac{1}{1825}$的数据量，从而大幅减少I/O和成本。

### 处理数据结构演变

数据湖是动态的。如果您的上游摄取管道在新Parquet文件中添加了一个新列，例如`shipping_cost`，数据目录不会自动识别它。现有的表定义充当一个严格的筛选器；它会忽略不符合注册数据结构的数据。

要使新列可见，您必须更新数据目录定义：

```sql
ALTER TABLE silver_orders ADD COLUMNS (shipping_cost DOUBLE);
```

此操作仅限于元数据。它更新Hive元数据存储或Glue数据目录中的JSON定义。它不会重写任何物理文件。缺少此列的旧文件在查询该列时将只返回`NULL`，而新文件将返回相应的值。

通过掌握物理数据布局与逻辑数据目录定义的分离，您可以确保数据湖保持可伸缩性和高性能。在下一节中，我们将分析分布式引擎如何运用这些定义来执行向量 (vector)化查询。

## 参考资料

- [What is AWS Glue Data Catalog?](https://docs.aws.amazon.com/glue/latest/dg/catalog-dg.html) — Amazon Web Services (2024)
  官方文档详细介绍了 AWS Glue 数据目录服务、其功能以及如何使用 Glue 爬虫自动发现和注册数据湖架构和分区。
- [Apache Parquet Format Documentation](https://parquet.apache.org/docs/) — Apache Software Foundation (2025)
  Publisher: Apache Software Foundation
  解释了列式存储格式、其规范以及架构演进和编码等特性，这些是理解数据湖中数据存储的基础。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  一本关于数据系统原理的综合指南，涵盖数据模型、存储格式、架构演进和分布式系统等主题，为数据湖架构提供了坚实的理论基础。

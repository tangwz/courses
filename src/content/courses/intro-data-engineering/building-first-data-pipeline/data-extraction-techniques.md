---
course: "intro-data-engineering"
chapter: "building-first-data-pipeline"
lesson: "data-extraction-techniques"
sourceId: 5532
sourceUrl: "https://apxml.com/zh/courses/intro-data-engineering/chapter-3-building-first-data-pipeline/data-extraction-techniques"
title: "数据提取方法"
description: "介绍从不同数据源获取数据的简便方法。"
order: 4
plots: []
sourceHash: "894a36befbc74f19ee134619d6ac4afba799e7031de67b9b2c7c56b7dc272d7a"
sourceCorrections: []
---

数据在管道中的处理始于提取。可以把它想象成将原材料从源头运到工厂的第一步。在进行任何清洗、塑形或合并之前，您需要将数据从其当前位置“取出”。数据存在于许多不同位置，获取数据的方法很大程度上取决于源系统。

数据工程师常用的数据提取方法构成了数据管道的主要第一阶段（ETL 和 ELT 中的“E”）。

### 从数据库提取

数据库是信息的结构化存储库，从中获取数据通常需要使用它们各自的查询语言。

- **关系型数据库 (SQL)：** 这些数据库，例如 PostgreSQL、MySQL 或 SQL Server，使用结构化查询语言 (SQL)。提取数据通常涉及编写 SQL `SELECT` 语句来指定所需表、列和行。

  - **全表提取：** 有时，您可能需要获取表的全部内容。这很简单，但对于大型表来说可能占用大量资源。

    ```sql
    SELECT * FROM customers;
    ```
  - **增量提取：** 通常，您只需要自上次提取以来发生变化或新增的数据。这效率更高。常用方法包括：
    - 按时间戳列进行筛选（例如 `last_updated_date`）。

      ```sql
      SELECT user_id, order_details, order_date
      FROM orders
      WHERE order_date > '2023-10-26 18:00:00';
      ```
    - 使用为跟踪更改而设计的特定数据库功能（有时称为变更数据捕获或 CDC），尽管具体机制因数据库系统而异。
- **NoSQL 数据库：** 这些数据库（如 MongoDB、Cassandra、Couchbase）以非传统行和列的格式（例如文档、键值对）存储数据。它们有自己的查询语言或 API。提取数据可能涉及：

  - 使用数据库的特定查询语言（例如 MongoDB 的 MQL）。
  - 使用 Python 或 Java 等编程语言提供的客户端库，以编程方式获取数据。
  - 与 SQL 数据库类似，如果数据模型支持，您可以执行全量提取或根据时间戳或其他标记 (token)实现增量逻辑。

### 从 API 获取数据

许多网络服务和应用程序通过应用程序编程接口 (API) 公开其数据。可以将 API 想象成餐厅里的服务员：您提出请求（索取数据），API 则提供响应（数据本身）。

- **工作方式：** 通常，您向 API 提供的特定 URL 端点发送 HTTP 请求（通常是 `GET` 请求）。API 处理请求并返回数据，常见格式为 JSON 或 XML。
- **认证：** 大多数 API 需要某种形式的认证来识别和授权请求发起者。这通常涉及在请求中发送 API 密钥或令牌。
- **参数 (parameter)：** 您通常可以通过在请求 URL 中包含参数来定制接收到的数据，例如指定日期范围、筛选条件或每页结果数量（分页）。
  - *示例：* 请求用户数据可能看起来像向 `https://api.example.com/v1/users?status=active&page=1` 发送一个 GET 请求。
- **速率限制：** API 通常对您在特定时间内可以发出的请求数量进行限制，以防止滥用。您的提取过程需要遵守这些限制，可能需要在请求之间增加暂停时间。

### 从文件读取数据

数据常存储在文件中，位于本地磁盘、网络共享或云存储系统上。

- **文件类型：** 常见格式包括 CSV（逗号分隔值）、JSON（JavaScript 对象表示法）、Parquet、Avro、XML 和纯文本日志文件。
- **位置：**
  - **本地/网络文件系统：** 您的管道可能需要直接从运行它的服务器或从已挂载的网络驱动器读取文件。
  - **云对象存储：** Amazon S3、Google Cloud Storage (GCS) 和 Azure Blob Storage 等服务非常常用，用于存储大量数据文件。提取数据涉及使用云提供商的工具或库来访问和下载这些文件。
- **提取逻辑：**
  - **全文件读取：** 读取一个或多个文件的全部内容。
  - **增量读取：：** 只处理自上次运行以来目录中出现的新文件，这些文件通常通过文件名模式或时间戳进行识别。
  - **读取特定部分：** 对于某些格式（如 Parquet），可以高效地只读取文件的特定列或部分，而无需将整个文件加载到内存中。

> 数据从各种来源流出，通过特定的提取方法，进入管道的下一阶段。

### 订阅流式数据

与上述批处理方法不同，有些数据以连续的事件流形式到达。可以想象成社交媒体动态、传感器读数或实时生成的应用程序日志。

- **来源：** Apache Kafka、Google Cloud Pub/Sub 或 Amazon Kinesis 等平台旨在处理这些数据流。
- **工作方式：** 您的管道并非定期*请求*数据，而是*订阅*数据流或主题。当新的数据事件到达源头时，它们会几乎即时地被推送到您的管道中。
- **注意事项：** 从数据流中提取数据需要与批处理提取不同的工具和架构模式，侧重于处理单个事件或到达时的小微批次。本课程主要侧重于批处理管道（ETL/ELT），但了解存在流处理以满足实时需求会很有帮助。

### 选择正确方法

最佳提取方法取决于：

1. **数据源：** 对于关系型数据库使用 SQL，对于网络服务使用 API 调用，对于文件使用文件读取器等。
2. **数据量：** 从数据库中提取数千兆字节的数据与从 API 获取几千字节的数据需要不同的方法。
3. **频率：** 您需要数据多久一次？每隔几秒（流式），每小时（小批次），还是一天一次（批处理）？
4. **数据格式：** 数据的结构（或缺乏结构）影响数据提取后的解析方式。
5. **可用工具：** 您可以访问的特定软件和库将影响实施方式。

提取是数据进入您管道的入口。通过理解这些从数据库、API、文件和流中获取数据的基本方法，您就能够构建数据系统的第一个重要组成部分。后续步骤通常包括转换这些原始提取的数据或将其加载到目标系统。

## 参考资料

- [Fundamentals of Data Engineering: Planning and Building Robust Data Systems](https://books.google.com/books/about/Fundamentals_of_Data_Engineering.html?id=uN6CEAAAQBAJ) — Joe Reis, Matt Housley (2022)
  Publisher: O'Reilly Media; Pages: 447
  全面介绍数据工程原理，包括多种数据源和初始数据摄取策略。
- [Change Data Capture for Modern Data Integration](https://www.packtpub.com/product/change-data-capture-for-modern-data-integration/9781800569733) — Gunnar Carling (2020)
  Publisher: Packt Publishing
  探讨不同数据库系统中变更数据捕获 (CDC) 的技术，这是一种高级提取方法。
- [MDN Web Docs: HTTP](https://developer.mozilla.org/en-US/docs/Web/HTTP) — MDN Contributors (2023)
  提供超文本传输协议 (HTTP) 的指南，这是理解 API 数据获取的基础。
- [Amazon S3 User Guide](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html) — Amazon Web Services (AWS) (2023)
  亚马逊 S3 的官方文档，这是一种常用的云对象存储服务，其中详细说明了数据访问方法。
- [Apache Kafka Documentation](https://kafka.apache.org/documentation) — Apache Software Foundation (2023)
  Apache Kafka 的官方文档，这是一个分布式流处理平台，解释其架构和实时数据摄取用法。

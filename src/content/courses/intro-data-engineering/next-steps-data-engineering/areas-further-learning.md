---
course: "intro-data-engineering"
chapter: "next-steps-data-engineering"
lesson: "areas-further-learning"
sourceId: 5605
sourceUrl: "https://apxml.com/zh/courses/intro-data-engineering/chapter-7-next-steps-data-engineering/areas-further-learning"
title: "可进一步学习的方面"
description: "确定数据工程中可进一步学习的高级议题和专业方向。"
order: 1
plots: []
sourceHash: "403804168aa37554ebfd25bd2d2853e5251384dd83d29acf9f1fe6c3ff8bfd1d"
sourceCorrections: []
---

在整个课程中，你对数据工程原理已扎实掌握，现在有能力处理更专业的议题。这个行业涉及很广，持续学习是职业生涯的一部分。以下是一些你可以考虑继续学习的方面，以增加你的知识和能力。每个方面都基于你已学到的知识，比如数据管道、存储和处理。

> 基于本入门课程所涵盖的核心技能，可能的发展学习路径。

### 云数据服务

主要的云平台，例如AWS、Google Cloud Platform（GCP）和Microsoft Azure，提供多种数据服务。实际操作这些特定的数据服务是一个重要的实践环节。这些平台为几乎所有数据工程任务提供了丰富的托管服务系统，包括：

- **存储：** 基础对象存储（如AWS S3、Google Cloud Storage、Azure Blob Storage），进一步了解托管数据库（如AWS RDS、GCP Cloud SQL、Azure SQL Database）和数据仓库（AWS Redshift、Google BigQuery、Azure Synapse Analytics）。
- **处理：** 学习用于运行处理作业的托管服务，例如AWS EMR、GCP Dataproc、Azure HDInsight，或平台特有的服务，如AWS Glue和Google Dataflow。
- **集成：** 考察在云环境中设计用于数据集成和管道构建的服务。

掌握如何有效使用这些服务，你就能构建可扩展、有弹性和通常具成本效益的数据基础设施，而无需管理底层硬件。初期专注于一家云服务商通常是个不错的策略。

### 大数据处理框架的进一步学习

我们曾提到处理框架，并提及Apache Spark。为了处理超出单机容量的数据集（通常称为“大数据”），分布式处理框架是必需的。

- **Apache Spark：** 这是一种广泛使用且功能强大的大规模数据处理引擎。学习Spark需要理解其核心抽象（如弹性分布式数据集RDD和DataFrames）以及它如何在机器集群上执行代码。它支持批处理、流处理（Spark Streaming）、机器学习 (machine learning)（MLlib）和图处理（GraphX）。
- **Hadoop生态系统：** 尽管Spark通常独立使用，但了解Hadoop生态系统的核心组成部分，例如用于存储的Hadoop分布式文件系统（HDFS）和用于资源管理的YARN，能提供有用的背景知识。

学习这些框架能让你设计和实现能高效处理海量数据的数据管道。

### 高级数据仓库与建模

你学习了数据仓库用于分析的目的。一个重要的学习方向是，如何有效设计这些仓库的内部结构或模式。

- **维度建模：** 这是一种设计数据仓库以优化查询和报告的标准技术。它包括将数据组织成“事实”表（包含度量或指标）和“维度”表（包含描述性属性）。了解星型模式和雪花模式在这里非常重要。
- **建模方法：** 熟悉Kimball方法（自下而上，注重维度）或Inmon方法（自上而下，规范化的企业数据仓库）等已确立的方法。
- **性能优化：** 了解使仓库查询运行更快的技术，例如索引、分区以及选择合适的数据类型和存储格式（如你之前遇到的Parquet或ORC）。

正确的数据建模能确保数据对分析用户而言是可访问、易懂且表现良好的。

### 掌握工作流编排

我们提到了简单的管道编排。实际中，数据管道可能会变得复杂，包含许多带有依赖关系的步骤。工作流编排工具能帮助管理这种复杂性。

- **工具：** Apache Airflow是一个非常流行的开源工具。其他选项包括Prefect、Dagster，以及AWS Step Functions或Google Cloud Composer（通常使用Airflow）等云原生服务。
- **定义工作流：** 这些工具通常允许你将管道定义为代码，常表现为有向无环图（DAGs）。你定义任务以及它们之间的依赖关系。
- **功能：** 学习这些工具如何处理调度、监控管道运行、管理故障和重试、日志记录和警报。

掌握一种编排工具对于在生产环境中构建可靠、可维护和自动化的数据管道非常重要。

### 流处理入门

我们主要侧重于批处理，但处理到达时的数据（流式处理）对于需要近实时洞察的应用场景来说越来越重要。

- **消息系统：** Apache Kafka、Google Pub/Sub或AWS Kinesis等技术常作为流数据的入口点。它们充当传入数据流的持久、可扩展的缓冲区。理解主题、生产者和消费者等基础知识很重要。
- **流处理引擎：** Apache Flink、Spark Streaming或ksqlDB等框架允许你对这些连续数据流执行计算（过滤、聚合、连接）。
- **窗口操作：** 学习翻滚窗口、滑动窗口和会话窗口等技术，这些技术用于随时间对流数据进行分组以进行分析。

学习流处理为实时仪表盘、异常检测和即时警报等应用带来了更多可能。

选择专注于哪个方面取决于你的兴趣和目标。无论你想专攻云基础设施、大规模处理、分析支持、自动化还是实时系统，这些方向都为你提供了充足的机会，以便在你本课程所学的基础上继续发展。

## 参考资料

- [The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling](https://www.wiley.com/en-us/The+Data+Warehouse+Toolkit%3A+The+Definitive+Guide+to+Dimensional+Modeling%2C+3rd+Edition-p-9781118530801) — Ralph Kimball, Margy Ross (2013)
  Publisher: Wiley
  解释维度建模原理，包括星型/雪花型模式和金博尔方法，用于数据仓库设计。
- [Spark: The Definitive Guide: Big Data Processing Made Simple](https://www.oreilly.com/library/view/spark-the-definitive/9781491912201/) — Bill Chambers, Matei Zaharia (2018)
  Publisher: O'Reilly Media
  Apache Spark 的全面指南，涵盖核心概念和大规模数据处理的高级功能。
- [Kafka: The Definitive Guide: Real-Time Data and Stream Processing at Scale](https://www.oreilly.com/library/view/kafka-the-definitive/9781098135832/) — Gwen Shapira, Joshua Addie, and Kate Stanley (2021)
  Publisher: O'Reilly Media
  Apache Kafka 的指南，讨论其架构、消息传递和流处理功能，适用于实时应用。
- [Apache Airflow Documentation](https://airflow.apache.org/docs/apache-airflow/stable/index.html) — Apache Airflow Project (2024)
  学习 Apache Airflow 的官方资源，详细介绍其概念、设置和工作流开发。

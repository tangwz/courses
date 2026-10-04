---
course: "advanced-ai-infrastructure-design-optimization"
chapter: "financial-operations-governance-ai"
lesson: "optimizing-cloud-storage-costs"
sourceId: 7029
sourceUrl: "https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-6-financial-operations-governance-ai/optimizing-cloud-storage-costs"
title: "优化数据集的云存储成本"
description: "使用存储分层、生命周期策略和智能数据放置来降低对象存储开销。"
order: 3
plots: ["plots/7029-0.json"]
sourceHash: "6de6efbb342839bfa10064e31c6b172590f1623fc1658c6e36d274840c55ccda"
sourceCorrections: []
---

GPU密集型训练任务的高昂费用常受关注，但云存储持续累积的开销，可能成为任何AI平台的一项主要财务负担。数据集、模型产物、日志和实验检查点不断增长，形成一项不易察觉但很主要的预算开支。优化存储成本需要摆脱“永远存储所有数据”的观念，并采纳一种规范、自动化的数据管理方法。这意味着将数据视为具有明确生命周期的动态资产，而非静态资产。

### 将存储分层与数据生命周期对齐 (alignment)

存储成本优化的基础是一个简单原则：并非所有数据都具有同等价值或被频繁访问。云服务提供商提供多种存储类别，每种都具有不同的性能特点和定价模型。重要的是将数据的访问模式与最具成本效益的层级匹配。

- **标准（热）层级：** 此层级为频繁访问的数据优化。它提供最低的延迟，但每GB存储成本最高。所有新数据、正在进行项目的活跃数据集以及推理 (inference)端点提供的生产模型都应存储在此处。
- **不常访问（温）层级：** 专为不常访问但需要时能快速取回的数据设计。它提供较低的存储价格，但收取少量每次访问的检索费用。这是已完成项目的处理数据集、较旧的模型检查点或为短期法规合规性而保留的数据的理想归宿。
- **归档（冷）层级：** 这些是用于长期数据归档最经济的层级，选项包括AWS Glacier或GCP Archive Storage。检索并非即时，需要数分钟到数小时，且通常会产生更高的检索费用。此层级适用于原始数据、法律要求保留的实验结果，或任何预计数月或数年内不会访问的资产。

手动在这些层级之间移动数据既不切实际又容易出错。方案是实施自动化的**生命周期策略**。这些是在存储桶或前缀级别定义的规则，可根据对象的存储时间或其他标准自动转换或删除对象。

> 自动化的数据生命周期将对象从更昂贵、高性能的层级移动到更便宜的归档层级，最终进行删除。

一个典型的策略可能是：在90天不活跃后将处理过的数据集从标准层级移至不常访问层级，一年后再移至归档层级，最后在七年后安排删除。

### 积极减少存储容量的策略

分层有助于管理需要保留的数据的成本。同样重要的策略是首先减少存储的数据量。

#### 临时产物的垃圾回收

机器学习 (machine learning)开发会生成大量临时数据。每次训练运行都可能产生多个检查点、日志和评估指标。如果没有积极管理，一个项目很快就会累积数TB的冗余产物。

为这些临时资产制定垃圾回收策略。例如：

- 仅保留训练任务的最后 `N` 个检查点，或仅保留对应最佳验证分数的检查点。
- 在实验性分支合并或放弃后，自动删除版本控制系统中的所有产物。
- 对用于中间数据处理步骤的存储桶设置积极的删除生命周期策略（例如，14-30天）。

#### 高效数据格式与压缩

数据存储格式直接影响存储容量和查询成本。将1 TB数据集存储为未压缩的CSV文件效率极低。

采纳列式存储格式，例如 **Apache Parquet** 或 **ORC**。这些格式按列而非按行存储数据，并默认包含高效压缩编解码器。这不仅将存储占用减少75%或更多，而且显著降低查询成本。当查询只需要宽表中的几列时，支持列存储的引擎可以只读取所需数据，避免全表扫描，并减少处理的数据量。

例如，一个计算1 TB Parquet文件中单列平均值的查询，可能只需要读取50 GB的数据，直接降低了Amazon Athena或Google BigQuery等服务的查询成本。

#### 数据本地化与出站流量费

存储成本不仅限于每月每GB的价格。数据传输，特别是**出站流量**（数据移出云服务提供商网络），是一项重要且常被忽视的开销。在 `us-east-1` 区域训练模型，如果使用存储在 `eu-west-1` 存储桶中的数据，将产生高昂的数据传输费用。

始终将计算资源和存储放在同一云区域内。设计多区域系统时，要仔细考虑数据复制和访问模式，以最大程度地减少跨区域流量。

### 分析存储账单

为了有效优化，您必须首先了解您在为哪些方面付费。云存储账单通常由三个主要部分组成：

1. **存储容量：** 存储数据量的成本，以GB-月为单位计量。这是生命周期策略主要解决的问题。
2. **API操作：** 对存储进行请求的成本，例如 `PUT`、`COPY`、`POST`、`LIST` 和 `GET`。数百万个小文件可能比几个超大文件产生更高的API成本。
3. **数据传输：** 移动数据的成本，其中出站流量是最昂贵的部分。

使用您的云服务提供商的成本分析工具，例如AWS Cost Explorer或Azure Cost Management，来细分您的存储开销。按项目、团队和数据类型（例如，`原始数据`、`处理数据`、`检查点`）标记 (token)存储桶对这项分析非常重要。



![ML存储账单典型构成](plots/7029-0.json)



> 存储账单不仅仅是每GB的成本。数据传输和API请求通常是总开销的重要推动因素。

通过分析这份构成，您可以识别主要的成本动因。如果API费用异常高，可能表明应用程序效率低下，正在循环列出对象，而不是使用更有针对性的方法。如果出站流量成本高，则表明数据本地化配置有误。应用这些财务运营原则可确保您的数据基础设施在大规模运行时既具备技术能力又具有经济可行性。

## 参考资料

- [Managing your storage lifecycle](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle.html) — Amazon Web Services (2024)
  Publisher: Amazon Web Services
  关于配置Amazon S3生命周期策略以在存储类别之间转换对象和管理到期时间的官方指南，与分层和自动化数据管理直接相关。
- [Apache Parquet](https://parquet.apache.org/) — Apache Software Foundation (2024)
  Apache Parquet的官方网站，提供关于其列式存储格式、压缩优势以及其在降低大数据存储和查询成本方面作用的详细信息。
- [Cloud FinOps: Collaborative, Real-Time, Cloud Financial Management](https://www.oreilly.com/library/view/cloud-finops/9781492054610/) — J.R. Storment, Mike Fuller (2020)
  Publisher: O'Reilly Media
  一本关于FinOps原则的基础书籍，提供管理和优化云成本的策略，包括存储、数据传输和整体云财务治理。
- [AWS Well-Architected Framework - Cost Optimization Pillar](https://docs.aws.amazon.com/wellarchitected/latest/cost-optimization-pillar/cost-optimization-pillar.html) — Amazon Web Services (2024)
  Publisher: Amazon Web Services
  提供优化云成本的最佳实践和设计原则，涵盖数据传输、高效资源使用以及使服务与业务价值保持一致等，适用于全面的成本管理。

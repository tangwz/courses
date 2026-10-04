---
course: "intro-data-engineering"
chapter: "essential-tools-data-engineers"
lesson: "overview-cloud-platforms"
sourceId: 5599
sourceUrl: "https://apxml.com/zh/courses/intro-data-engineering/chapter-6-essential-tools-data-engineers/overview-cloud-platforms"
title: "云平台概览"
description: "简要介绍主要云服务商及其相关数据服务。"
order: 4
plots: []
sourceHash: "5f306ecd3990b324ca426d96b86e6a7090791838d37ce3e63dc9047654cc69a5"
sourceCorrections: []
---

无需在专用数据中心管理物理服务器，现代数据工程的许多工作都在“云端”完成。云计算指的是通过互联网获取计算资源，例如服务器、存储、数据库和软件，通常按使用量付费。可以把它想成从电网用电，而不是自己运行发电机。你可以在需要的时候获得所需电力，并按实际消耗付费。

对于数据工程来说，云平台提供诸多优点：

- **弹性伸缩**：需要处理庞大数据集吗？云平台能让你快速配置数百甚至数千台机器来完成任务，并在完成后释放这些机器。这种弹性用自己的硬件很难实现且成本高昂。
- **托管服务**：许多繁琐的任务，比如操作系统补丁、数据库备份管理或网络防火墙配置，都由云服务商处理。这使得数据工程师能专注于构建数据管道并从数据中获取信息。
- **成本效益**：按使用量付费的模式通常意味着你只为实际使用的资源付费，这比投入和维护昂贵且可能大部分时间闲置的硬件更经济。
- **获得先进工具**：云服务商提供一系列用于数据存储、处理、机器学习 (machine learning)和分析的专业服务，这些服务自行构建和维护可能过于复杂或昂贵。

### 主要云服务商

尽管有许多云服务商，但有三家主要厂商主导着市场。在你的数据工程工作中，你很可能会遇到其中一个或多个：

1. **Amazon Web Services (AWS)**：最早且采用最广泛的云平台，提供非常广泛的服务。
2. **Google Cloud Platform (GCP)**：以其在数据分析、机器学习 (machine learning)和容器编排方面的优势而闻名（Kubernetes 最初由 Google 开发）。
3. **Microsoft Azure**：一个强大的竞争者，尤其受到已大量投资微软技术的机构欢迎。它提供一套与 AWS 和 GCP 相似的全面服务。

尽管每个平台都有其独特的命名和特定实现，但所提供的服务类型通常相似，都能满足常见的数据工程需求。在一个平台学习基本知识，通常能让你更容易适应另一个平台。

### 数据工程的服务类别

数据工程师会与各种云服务交互。以下是一些重要的类别：

- **计算**：它们是虚拟服务器（如 AWS EC2、GCP Compute Engine 或 Azure Virtual Machines），提供运行应用程序、脚本和数据处理任务所需的处理能力。你可以根据需要选择不同的大小和配置。
- **存储**：云平台提供多种存储选项：
  - **对象存储**：像 AWS S3、Google Cloud Storage 和 Azure Blob Storage 这样的服务是不可或缺的。它们能让你以成本效益高的方式存储海量数据（结构化、半结构化或非结构化）。可以把它们看作通过互联网访问的无限可扩展硬盘，通常构成数据湖的根本。
  - **数据库**：云服务商提供流行关系型数据库（如 PostgreSQL、MySQL——例如 AWS RDS、Google Cloud SQL、Azure Database for PostgreSQL）和 NoSQL 数据库（例如 AWS DynamoDB、Google Bigtable、Azure Cosmos DB）的托管版本。“托管”意味着服务商处理设置、备份和补丁等管理任务。
  - **数据仓库**：专门为大型数据集上的分析查询而优化的数据库（例如 AWS Redshift、Google BigQuery、Azure Synapse Analytics）。它们旨在高效运行复杂报告和商业智能任务。
- **数据处理与分析**：除了基本计算，服务商还提供专为大规模数据处理定制的托管服务。这包括基于流行开源框架（如 Apache Spark 和 Hadoop）的平台（例如 AWS EMR、Google Dataproc、Azure HDInsight）以及用于实时流处理的服务（例如 AWS Kinesis、Google Cloud Dataflow、Azure Stream Analytics）。
- **网络**：定义云内安全私有网络、管理访问控制以及安全连接云资源彼此与互联网的服务。
- **工作流编排**：旨在调度、管理和监控数据管道的工具（这些管道通常涉及跨不同服务的多个步骤）。例子包括 AWS Step Functions、Google Cloud Composer（基于 Apache Airflow）和 Azure Data Factory。这些工具帮助自动化数据工程中常见的复杂任务序列。

下图展示了这些服务类别在典型云环境中如何协同完成数据工程任务。

> 数据工程工作流中常见云服务类别的简化视图，展示了数据源、云服务和最终用户之间的配合关系。

### 开始

作为初学者，你不需要一次掌握所有这些服务。请专注于理解每个类别的*目的*。许多入门级数据工程任务包含：

1. 将数据存入云端**存储**（例如 S3 或 Google Cloud Storage）。
2. 使用**计算**资源或专门的**处理**服务对其进行转换。
3. 将结果加载回**存储**（可能是 BigQuery 或 Redshift 等数据仓库）进行分析。

你将使用服务商的网页控制台、命令行界面（CLI）或 Python 等编程语言中的软件开发工具包（SDK）与这些服务进行交互。你在本章其他部分培养的 SQL、Git 和 CLI 技能，在与云平台协同工作时可直接运用。

最初不必过于担心选择“完美”的服务商。数据工程的基本实践在不同平台间是相似的，获得一个平台的经验将使学习其他平台变得容易得多。

## 参考资料

- [Fundamentals of Data Engineering: Planning and Building Robust Data Systems](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108269/) — Joe Reis, Matt Housley (2022)
  Publisher: O'Reilly Media
  一本关于现代数据工程实践的指南，包含大量关于云平台、架构和服务的资料。
- [Google Cloud Digital Leader Study Guide](https://cloud.google.com/learn/certification/cloud-digital-leader) — Google Cloud (2023)
  Publisher: Google Cloud
  官方学习指南和资源，帮助理解Google Cloud的产品和服务、云概念以及GCP如何支持数字化转型。
- [Microsoft Azure Fundamentals (AZ-900) Learning Path](https://learn.microsoft.com/en-us/credentials/certifications/azure-fundamentals/) — Microsoft (2023)
  Publisher: Microsoft
  官方学习路径，介绍Azure的核心服务、概念、解决方案和管理工具，提供微软云平台的基本概览。

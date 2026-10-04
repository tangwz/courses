---
course: "planning-optimizing-ai-infrastructure"
chapter: "cost-management-and-optimization"
lesson: "implementing-cost-monitoring-alerting"
sourceId: 7041
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-6-cost-management-and-optimization/implementing-cost-monitoring-alerting"
title: "实施成本监控与预警"
description: "使用云原生工具设置仪表盘、预算和预警，以监控支出并避免意外费用。"
order: 5
plots: ["plots/7041-0.json"]
sourceHash: "ccf89985cd6f0a6b43f3fbfcafb7d4d15d0ac00a2e458d27e14c097ce469f9df"
sourceCorrections: []
---

无法管理未衡量的事物。这条原则对AI基础设施尤为适用，因为一个被遗忘的GPU实例的成本可能在一个周末内悄无声息地耗尽项目预算。尽管设计经济高效的系统很重要，但维持它们需要运营规范来控制成本。目标是从被动的“账单冲击”情景，转变为主动的、数据驱动的财务管理模式。

这需要建立一个反馈循环，在此循环中，您可以了解每一笔钱的去向，将支出归因于具体活动，并在成本失控前自动收到通知。

### 借助成本分析工具获取可见性

管理成本的首要一步是实现清晰的可见性。所有主要云服务商都提供功能强大的内置工具，能将原始账单数据转化为易于理解的信息。这些仪表盘是您查看和分析基础设施支出的主要方式。

- **AWS：** **成本查看器 (Cost Explorer)** 是可视化支出模式的主要工具。它允许您根据服务（Amazon EC2, S3）、使用类型、区域以及最重要的资源标签来筛选和分组成本。对于更详细的分析，**AWS 成本与使用报告 (CUR)** 提供细粒度的每小时数据，可导入到如 Amazon Athena 等数据仓库进行复杂查询。
- **GCP：** **云账单报告 (Cloud Billing reports)** 提供与AWS成本查看器非常相似的交互式仪表盘。您可以查看不同时间段的成本，并按项目、产品（例如：Compute Engine, Cloud Storage）和标签（GCP对标签的称谓）进行分组。
- **Azure：** **成本管理 + 账单 (Cost Management + Billing)** 提供一套用于分析成本的工具。您可以创建自定义视图，按资源标签分组，并追踪预算内的支出情况。

从多个角度审查成本时，这些工具最为有效。对于典型的AI工作负载，您可以首先查看高层级的服务明细，以找出主要的成本构成。



![月度AI工作负载成本构成](plots/7041-0.json)



> 一个AI项目的典型成本构成。GPU计算通常在支出中占据主导地位，因此成为优化的主要目标。

通过定期查看这些仪表盘，您可以发现趋势，例如存储成本的突然增加，或者识别出产生费用但未提供价值的闲置资源。

### 通过标签化实行责任制

可见性告诉您*什么*正在产生费用，而责任制则告诉您*谁*或*哪个项目*应对该费用负责。在多团队和多实验的共享环境中，适当的资源标签化是实现财务责任制的根本。

标签是简单的键值对元数据，您将其附加到您的云资源上，例如虚拟机、存储桶和数据库。当您在云服务商的计费控制台中为成本分配激活这些标签时，它们会在成本报告中作为可筛选的维度出现。这使您能够根据自己的业务逻辑进行全面的成本分析。

实施一致的标签策略非常必要。对于AI/ML组织而言，一个好的起点包括：

- `project`：模型或项目的名称（例如，`fraud-detection-v2`）。
- `owner`：负责该资源的用户或团队（例如，`data-science-team` 或 `jane.doe`）。
- `environment`：工作负载所处的阶段（例如，`development`、`staging`、`production`）。
- `experiment-id`：特定训练运行的唯一标识符，用于追踪单个实验的成本。

采用此策略，您可以精确回答以下问题：“`fraud-detection-v2` 项目上个月在生产环境花费了多少？”或“`data-science-team` 在开发资源上的总支出是多少？”

> 附加到资源的标签会流入账单报告，从而实现按项目进行成本分配。

### 通过预算和预警建立控制

监控仪表盘是一种被动行为。为建立主动控制，您必须使用预算和预警来设定财务限制。这种机制会在支出即将超出计划时自动通知您，让您在小额超支演变为大问题之前有时间采取行动。

**预算**是您为特定范围设定的财务限额。此范围可宽泛（例如：您整个账户的月度支出）或狭窄（例如：所有标记 (token)为 `project: fraud-detection-v2` 的资源的月度成本）。

**预警**是当您的实际或预估支出超过预算的某个百分比时触发的通知。

让我们来看一个实际情况。假设一个团队为一个新的语言模型实验分配了10,000美元的月度预算。

1. **定义范围：** 该预算适用于所有标记为 `project: big-llama` 的资源。
2. **设置预算：** 在云计费控制台（例如：AWS Budgets, Azure Cost Management）中，创建一个周期为“每月”且金额为10,000美元的预算。您可以应用过滤器，使预算仅追踪带有 `project: big-llama` 标签的资源所产生的成本。
3. **配置预警阈值：** 您创建一系列预警来建立早期预警系统。
   - **达到50%（5,000美元）时：** 向项目的内部Slack频道发送通知。这是一种信息性“提醒”。
   - **达到80%（8,000美元）时：** 向团队负责人发送电子邮件。这表示支出正在按计划超出预算，需要进行审查。
   - **达到100%（10,000美元）时：** 向工程经理和财务部门发送高优先级电子邮件。这是一条紧急预警，表明预算已用尽。

这种分层预警系统可以避免意外情况，并允许纠正方向。当80%的预警被触发时，团队负责人可以进行调查。也许某个训练任务配置了过于昂贵的实例类型，或者旧实验的资源未被终止。在本月第20天发现这些问题，远好于在最终账单上发现。

> 主动的成本管理工作流形成了一个持续改进的循环。

归根结底，成本监控和预警不仅仅是为了削减成本。它们旨在灌输财务规范，并使支出成为您AI开发生命周期中可预测、可管理的一部分。通过结合可见性、责任制和自动化控制，您可以确保您的创新项目在财务上保持可持续性。

## 参考资料

- [Analyzing your costs with AWS Cost Explorer](https://docs.aws.amazon.com/aws-cost-management/latest/userguide/ce-what-is.html) — Amazon Web Services (2024)
  Publisher: Amazon Web Services
  说明如何可视化和分析AWS支出，包括资源标签。
- [AWS Well-Architected Framework: Cost Optimization Pillar](https://docs.aws.amazon.com/wellarchitected/latest/framework/cost-optimization.html) — Amazon Web Services (2023)
  Publisher: Amazon Web Services
  解释云成本效益的最佳实践，包括跟踪和控制支出。

---
course: "advanced-ai-infrastructure-design-optimization"
chapter: "financial-operations-governance-ai"
lesson: "practice-analyzing-cloud-cost-report"
sourceId: 7037
sourceUrl: "https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-6-financial-operations-governance-ai/practice-analyzing-cloud-cost-report"
title: "实践：分析云成本和使用报告"
description: "根据一份云账单示例，识别机器学习工作负载的主要成本驱动因素，并提出具体的优化措施。"
order: 7
plots: ["plots/7037-0.json", "plots/7037-1.json"]
sourceHash: "482eaf948a11e7531dad9c9c2311292c8be2a1d4965bec297118257edbe37f30"
sourceCorrections: []
---

本练习将让您扮演 MLOps 工程师的角色，负责细查云账单以找到优化机会。我们将学习如何将成本数据与运营指标进行交叉比对，从而获得有意义、可执行的信息。

我们的场景基于一个简化但有代表性的云成本和使用报告（CUR）。这些报告是所有开支的真实依据，包含每个资源的详细每小时数据。一份原始 CUR 可能包含数百万行，使得直接分析不实际。您的首要任务始终是汇总和筛选这些数据，以便从杂乱信息中找出有用信息。

假设您收到了一支机器学习 (machine learning)团队的月度 CUR。经过一些初步处理后，其中几行可能如下所示：

| 行项目使用开始日期 | 产品名称 | 行项目使用类型 | 行项目资源 ID | 用户:项目 | 用户:作业 ID | 行项目未混合成本 |
| --- | --- | --- | --- | --- | --- | --- |
| 2023-10-15T10:00:00Z | Amazon EC2 | USW2-GPU-Instance:g5.48xlarge | i-0abcdef1234567890 | project-atlas | train-exp-23a | 8.16 |
| 2023-10-15T11:00:00Z | Amazon EC2 | USW2-GPU-Instance:g5.48xlarge | i-0abcdef1234567890 | project-atlas | train-exp-23a | 8.16 |
| 2023-10-15T10:00:00Z | Amazon S3 | USW2-Storage-Bytes-Hour | atlas-dataset-bucket | project-atlas |  | 0.0000004 |
| 2023-10-16T04:00:00Z | Amazon EC2 | USW2-BoxUsage:t3.medium | i-fedcba0987654321 | project-ganymede | inference-api | 0.0416 |

自定义标签（如 `user:project` 和 `user:job_id`）的存在是良好管理的结果，对于有效的成本归因必不可少。没有它们，几乎不可能将成本分配到特定活动。

### 步骤 1：按服务进行高层次汇总

在检查单个作业之前，您需要一个宏观视图。第一步是按服务汇总总成本。这有助于您理解整个平台的主要成本驱动因素。使用 Pandas (Python)、AWS Athena 中的 SQL 查询或您的云提供商的成本分析工具，您可以生成一份摘要。



![每月云服务开支](plots/7037-0.json)



> 月度成本明细。计算服务，尤其是 Amazon EC2，占开支的最大部分。

该图表明确显示计算是主要成本，其中 EC2 是最大的贡献者。虽然 S3 存储和数据传输不可忽略，但任何重要的优化工作都必须从计算开始。

### 步骤 2：将计算成本与机器学习 (machine learning)活动关联

现在，我们关注 18,500 美元的 EC2 账单。利用 `user:project` 和 `line_item_usage_type` 列，我们可以创建更详细的明细。这有助于将成本归因于特定团队及其使用的实例类型，这通常表明工作负载类型（例如，用于训练的 GPU 实例，用于数据处理或推理 (inference)的 CPU 实例）。



![按项目和实例系列划分的 EC2 开支](plots/7037-1.json)



> 旭日图展示成本归因。Atlas 项目是开支最高者，主要用于模型训练的 g5 GPU 实例。

此视图显示“Atlas 项目”承担了大部分成本，对 `g5` 实例有大量投入。这是我们下一个分析目标。

### 步骤 3：分析训练作业效率

如果训练作业产生了有价值的模型，那么高成本本身并非坏事。问题源于浪费，这可以通过查看两个因素来量化 (quantization)：作业失败和资源利用不足。

为此，您必须用机器学习 (machine learning)平台的元数据补充 CUR 数据。假设您有日志可以提供每个训练作业的最终状态（`成功`、`失败`）以及其运行期间的平均 GPU 利用率百分比。

| 作业 ID | 实例类型 | 小时 | 原始成本 (美元) | 作业状态 | 平均 GPU 利用率 |
| --- | --- | --- | --- | --- | --- |
| train-exp-23a | g5.48xlarge | 100 | 816 | Succeeded | 92% |
| train-exp-24b | g5.48xlarge | 150 | 1224 | Failed | 88% |
| train-exp-25 | g5.12xlarge | 200 | 544 | Succeeded | 45% |
| train-exp-26 | p4d.24xlarge | 50 | 1638 | Succeeded | 95% |

这种组合视图比单独的账单报告更具信息量。

- **作业 `train-exp-24b`：** 该作业消耗了 1,224 美元的 GPU 时间，但未产生任何价值。这是直接的经济损失。根本原因可能是代码错误、环境配置错误或容错能力不足。
- **作业 `train-exp-25`：** 该作业成功了，但其 GPU 利用率仅为 45%。GPU 超过一半时间处于空闲状态，可能是等待来自缓慢存储来源的数据，或等待 CPU 完成预处理。尽管作业成功了，但其 544 美元成本中超过一半是由于 I/O 或 CPU 瓶颈造成的浪费。

现在我们可以应用章节介绍中的“有效成本”公式：


$$
\text{有效成本} = \frac{\text{总开支}}{\text{作业成功率} \times \text{资源利用率}}
$$


对于 `train-exp-25`，有效成本并非 544 美元。它更接近于 544 / (1.0 \times 0.45) = \1,208\$。这个数字代表该作业*原本*会花费的成本，如果您只为实际使用的资源部分付费。您的目标是使 `有效成本` 尽可能接近 `原始成本`。

### 步骤 4：制定可执行的建议

基于此分析，您现在可以从观察转向行动。您可以向 Atlas 项目团队提供具体、数据驱动的建议。

1. **调查训练失败情况：**

   - **发现：** 作业 `train-exp-24b` 在产生 1,224 美元成本后失败。
   - **措施：** 优先对此失败进行事后分析。实施预检并增强自动化检查点功能，以便长期运行的作业可以从最近的状态恢复，而不是从头开始。
2. **解决 GPU 利用不足问题：**

   - **发现：** 作业 `train-exp-25` 仅显示 45% 的 GPU 利用率，表明存在严重瓶颈。
   - **措施：** 分析该作业的数据加载管道。评估是否需要更快的存储方案（如并行文件系统）或更高效的数据预处理库（如 NVIDIA DALI）。如果瓶颈无法解决，考虑使用更小、成本更低的实例。
3. **优化存储成本：**

   - **发现：** 对 S3 存储桶的独立分析显示，80% 的存储成本来自于将旧数据集和模型工件保留在 S3 Standard 层。
   - **措施：** 实施 S3 生命周期策略，自动将超过 90 天的对象转换到 S3 Glacier 即时检索层，并将超过一年的对象转换到 Glacier 深度归档，从而将存储成本降低高达 90%。
4. **改进管理：**

   - **发现：** 分析之所以可能，是因为存在 `user:project` 和 `user:job_id` 标签。有些资源缺少这些标签。
   - **措施：** 实施自动化策略（例如，使用 AWS Config Rules），终止或标记 (token)任何在没有所需项目标签的情况下启动的新 EC2 实例或 S3 存储桶。

这种有组织的过程将原始账单报告从一份简单的会计文件转换为用于工程改进的战略工具。这是 FinOps 的基本循环：衡量开支，将其归因于特定活动，分析该开支的效率，并实施改进措施。

## 参考资料

- [FinOps Framework](https://www.finops.org/framework/) — FinOps Foundation (2023)
  提供云财务管理的标准框架，包括云环境中成本优化的原则和实践。
- [What are AWS Cost and Usage Reports?](https://docs.aws.amazon.com/cur/latest/userguide/what-is-cur.html) — Amazon Web Services (2024)
  Publisher: Amazon Web Services
  AWS成本和使用报告的官方文档，说明其结构和数据点，以进行精细的成本分析。
- [Cloud FinOps: Collaborative, Real-Time, Cloud Financial Management](https://www.oreilly.com/library/view/cloud-finops/9781492054610/) — J.R. Storment, Mike Fuller (2023)
  Publisher: O'Reilly Media
  一本介绍云财务操作的书籍，提供成本归因、优化和治理的实用方法，与练习步骤一致。
- [Machine Learning Engineering in Action](https://www.manning.com/books/machine-learning-engineering-in-action) — Ben Wilson (2022)
  Publisher: Manning Publications
  涵盖机器学习的运维方面，包括AI基础设施的资源管理和成本考量。

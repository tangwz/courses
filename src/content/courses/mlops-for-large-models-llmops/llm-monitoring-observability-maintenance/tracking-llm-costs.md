---
course: "mlops-for-large-models-llmops"
chapter: "llm-monitoring-observability-maintenance"
lesson: "tracking-llm-costs"
sourceId: 3198
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-5-llm-monitoring-observability-maintenance/tracking-llm-costs"
title: "追踪运营成本"
description: "实施机制来监控和归属LLM运营相关的成本。"
order: 3
plots: ["plots/3198-0.json"]
sourceHash: "1c8a2d2bd7c45d389acafadbd4939c6ff21d79234864bd22a507547aa45207d8"
sourceCorrections: []
---

在生产环境中运行大型语言模型涉及相当大的开销，远超一般软件的运营成本。计算密集型训练或微调 (fine-tuning)阶段，以及持续处理推理 (inference)请求的需求，都对总拥有成本有显著影响。忽视这些成本可能导致部署难以持续，并妨碍由LLM驱动的应用的长期可行性。因此，建立运营成本追踪机制不只是一项财务工作；它是LLMOps的一个基本组成部分，为优化和资源管理提供重要信息。

了解这些成本的来源是有效管理它们的第一步。

### LLMOps的主要成本驱动因素

与LLM相关的运营开支主要来自以下几个方面：

1. **计算资源：** 这通常是主要的成本因素。

   - **训练和微调 (fine-tuning)：** 大规模训练乃至广泛的微调需要大量的GPU或TPU时长，可能在多个高性能加速器上持续数天或数周。分布式训练框架虽然不可或缺，但也引入了通信开销，这可能导致运行时间更长、计算成本更高。
   - **推理 (inference)服务：** 部署的模型持续消耗资源（GPU、TPU，有时是CPU）以处理用户请求。硬件选择（例如，A100与T4 GPU）、满足延迟和吞吐量 (throughput)服务级别目标（SLO）所需的实例数量，以及推理服务器的效率对服务成本有显著影响。等待请求的空闲资源也会产生费用。
2. **数据存储：** LLM处理海量数据集。

   - **训练数据：** 存储PB级数据集需要大量的对象存储容量。
   - **模型检查点：** 在训练期间保存中间和最终模型权重 (weight)会生成大型文件（每个检查点几GB到几TB），因此需要版本化的存储解决方案。
   - **向量 (vector)数据库：** 对于RAG系统，存储和索引大量向量嵌入 (embedding)增加了另一层存储成本，通常与专门的数据库服务相关。
3. **网络：** 传输大量数据可能代价高昂。

   - **数据传输：** 将大型数据集传输到训练集群或在不同存储层级或区域间移动模型检查点会产生网络出站费用，尤其是在云环境中。分布式训练期间的节点间通信也会消耗大量带宽。
4. **第三方API使用费：** 如果您的系统依赖外部LLM提供商（例如OpenAI、Anthropic、Cohere），费用通常与使用量直接相关，通常按每个令牌（输入和输出）或每个请求计费。这需要仔细监控，特别是在高负载情况下。
5. **监控和可观测性工具：** 用于日志记录、追踪和监控LLM行为及基础设施性能的专用平台通常有自己的定价模式，基于摄入的数据量或使用的功能。

### 实施成本追踪机制

为了清晰地了解这些支出，您需要系统化的追踪。仅仅依靠每月云账单不足以进行详细分析和优化。

#### 云服务提供商工具和标签

AWS、Azure和GCP等云平台提供内置的成本管理服务（分别为Cost Explorer、Cost Management + Billing、Cloud Billing）。这些工具价值很高，但它们的有效性取决于严格的资源标记 (token)策略。为与您的LLM项目相关的所有资源实施一致的标记策略：

- `project`：具体的LLM应用或项目。
- `environment`：`开发`、`预发布`、`生产`。
- `model_name`：正在训练或服务的模型标识符。
- `model_version`：特定版本或检查点ID。
- `component`：`训练`、`推理`、`数据存储`、`向量数据库`。
- `team`：负责团队或所有者。

标签让您可以在云服务提供商的仪表板中筛选和分组成本，准确地归属各项开支。

#### 资源层级监控集成

将成本数据与基础设施性能指标关联起来。Prometheus、Grafana或Datadog等工具可以监控GPU/TPU利用率、内存使用量和网络I/O。将成本信息与利用率指标叠加有助于发现效率低下之处。例如，高成本推理 (inference)端点上持续较低的GPU利用率表明，通过实例大小调整、自动扩缩容设置或模型优化，有可能节省开销。

#### API使用监控

- **内部端点：** 记录每个推理请求的元数据，包括输入和输出令牌数量、使用的模型以及处理时间。这允许根据底层基础设施计算每个请求的近似成本。
- **外部API：** 大多数第三方LLM提供商都提供仪表板来追踪API使用量和相关成本。如果可能，将这些数据整合到您的中央监控系统。

#### 自定义成本仪表板

将来自云计费API、资源监控系统和API日志的数据汇总到一个统一的仪表板中。这提供了符合您的LLMOps情况的全面视图。显示重要的成本指标，例如：

- 总成本随时间的变化趋势。
- 按组成部分（训练与推理与存储）进行的成本细分。
- 每个项目或团队的成本。
- 每个推理请求的平均成本（估算）。
- 特定训练或微调 (fine-tuning)运行相关的成本。



![LLM月度运营成本细分（示例）](plots/3198-0.json)



> 此示例图显示了给定月份中，按不同项目和共享基础设施组件分摊的堆叠成本。

### 成本归因与分析

有效的追踪能够实现准确的成本归因。使用收集到的数据和标签来了解：

- **哪些模型或项目消耗的资源最多？** 这有助于确定优化工作的优先级。
- **开发、预发布和生产环境之间的成本有何不同？** 确保开发/测试环境不会不必要地增加成本。
- **部署新模型版本或功能预计会带来多大的成本影响？** 这为业务决策提供依据。

分析成本随时间的变化趋势也很重要。最近的优化技术（如量化 (quantization)）是否显著降低了推理 (inference)成本？用户流量的激增是否正确触发了扩缩容事件并导致了相应的、合理的成本增加？在您的云服务提供商或监控工具中设置预算提醒，以便在支出超出预设阈值时主动通知相关人员。

追踪运营成本是一个持续的过程，与性能监控和优化紧密相关。通过清楚地了解资金去向，您可以获得必要的信息，从而在资源分配、模型效率以及大型模型部署的整体财务可持续性方面做出明智决策。

## 参考资料

- [AWS Well-Architected Framework - Cost Optimization Pillar](https://docs.aws.amazon.com/wellarchitected/latest/cost-optimization-pillar/cost-optimization-pillar.html) — Amazon Web Services (2023)
  Publisher: Amazon Web Services
  提供管理和降低云支出的指导方针，包括资源选择、标记和财务治理，与大型模型部署相关。
- [Effective MLOps: Reliable, Reproducible, Scalable, and Maintainable Machine Learning](https://www.oreilly.com/library/view/effective-mlops/9781492080344/) — Sven Schultze (2022)
  Publisher: O'Reilly Media
  一本关于MLOps实践的书，包含机器学习系统的操作效率和资源管理。

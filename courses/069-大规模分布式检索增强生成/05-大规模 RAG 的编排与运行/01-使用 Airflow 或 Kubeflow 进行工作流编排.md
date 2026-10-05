# 使用 Airflow 或 Kubeflow 进行工作流编排

来源：[原文](https://apxml.com/zh/courses/large-scale-distributed-rag/chapter-5-orchestration-operationalization-large-scale-rag/rag-workflow-orchestration-airflow-kubeflow)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在您将大规模检索增强生成 (RAG) 系统投入运行的过程中，管理数据处理、模型执行和基础设施配合的交互会成为一项主要的挑战。一个典型的 RAG 系统包含多个阶段：数据摄取、分块、向量 (vector)化、索引、检索、重排序、生成以及持续的评估和更新。可靠、高效且大规模地执行这些阶段需要一个工作流编排方案。Apache Airflow 和 Kubeflow Pipelines 等工具正是在此派上用场，为自动化、调度和监控您的 RAG 工作流提供支持。

### RAG 编排的必要性

大规模 RAG 流程并非简单的线性脚本。它们涉及：

- **复杂的依赖关系**：向量 (vector)生成必须在文档分块之后；索引更新依赖于新的向量。
- **不同类型的任务**：操作可以包含大规模数据处理（用于摄取的 Spark 作业）、GPU 密集型任务（向量生成、LLM 推理 (inference)）以及 API 调用（向量数据库更新）。
- **错误处理和重试**：分布式系统容易出现暂时性故障。单个阶段必须使用适当的退避策略重试，而不能破坏整个工作流。
- **监控和日志**：集中跟踪每个阶段的进度、故障和性能指标，对运行情况了解很重要。
- **调度**：对知识库的定期更新或模型的定期微调 (fine-tuning)需要按计划执行。
- **参数 (parameter)化**：工作流常需要使用不同的配置运行（例如，新数据集、不同的向量模型）。

工作流编排器提供框架，将这些复杂过程定义为可管理、可观察和可重复的有向无环图 (DAG) 或管道。

### Apache Airflow 在 RAG 工作流中的应用

Apache Airflow 是一个广泛使用的平台，用于以编程方式编写、调度和监控工作流。其核心抽象是 DAG，代表一组具有定义依赖关系的任务。对于大规模 RAG 系统，Airflow 提供很大的灵活性。

#### 在 Airflow 中设计 RAG DAG

一个结构良好的 RAG 系统 Airflow DAG 可以管理从原始数据源到更新的向量 (vector)索引和微调 (fine-tuning)模型的端到端数据管道。RAG DAG 的考虑因素包含：

- **模块化**：将 RAG 流程分解为细粒度任务（操作器）。例如：
  - `FetchNewDataSourceOperator`：拉取新的或更新的文档。
  - `ChunkDocumentsOperator`：将文档拆分为可管理的部分。
  - `GenerateEmbeddingsOperator`：为分块创建向量。
  - `UpdateVectorDBOperator`：将新向量更新或插入到向量数据库中。
  - `RetrainRetrieverOperator`：定期微调检索模型。
  - `EvaluateRAGQualityOperator`：在新系统上运行评估套件。
- **幂等性**：确保任务可以安全地重复运行。例如，`UpdateVectorDBOperator` 应该妥善处理重复数据或使用版本控制。
- **动态 DAG**：如果您为不同领域或客户管理多个 RAG 系统，Airflow 基于 Python 的 DAG 定义允许根据配置文件动态生成 DAG，从而减少样板代码。
- **分支和条件逻辑**：实现逻辑，例如，如果数据漂移低于某个阈值，则跳过 LLM 微调，或者如果数据质量检查失败，则触发警报。

#### RAG 的 Airflow 操作器和执行器

Airflow 的强大之处在于其丰富的操作器集和可插拔的执行器架构。

- **操作器**：
  - `PythonOperator`：用于任何 RAG 阶段的自定义 Python 逻辑。
  - `KubernetesPodOperator`：运行容器化 RAG 组件的理想选择（例如，使用特定支持 GPU 的镜像进行向量生成，或自定义处理脚本）。这允许每个步骤拥有其隔离、定义明确的环境。
  - `SparkSubmitOperator` 或 `DatabricksRunNowOperator`：用于使用 Spark 进行大规模数据摄取和预处理步骤。
  - `DockerOperator`：如果未使用 Kubernetes，则用于在 Docker 容器中运行任务。
  - 自定义操作器：开发操作器以直接与您的向量数据库 API、LLM 服务端点或其他专业服务交互。
- **执行器**：
  - `CeleryExecutor` 或 `CeleryKubernetesExecutor`：用于在工作节点集群中分发任务，对于处理许多并行 RAG 工作流或高吞吐量 (throughput)数据处理很重要。
  - `KubernetesExecutor`：为每个 Airflow 任务动态启动一个新 Pod，提供出色的隔离和资源管理，如果您的 RAG 组件已经容器化，则特别适合。

#### 扩展与监控

对于大规模 RAG，请为 Airflow 配置足够的 worker 资源和并行度。将 Airflow 的日志与您的中心化日志系统（例如，ELK stack、Splunk）集成。使用 Airflow 的 UI 监控 DAG 运行、任务持续时间和故障。可以将自定义指标从 Airflow 任务推送到 Prometheus 或类似系统，以跟踪 RAG 特定关键绩效指标，如“每次运行处理的文档数”或“平均向量生成时间”。

> 一个典型的 RAG 工作流，呈现了数据准备、查询时操作和维护任务，这些可以作为一系列依赖步骤进行编排。

### Kubeflow Pipelines 在 RAG 工作流中的应用

Kubeflow Pipelines 是一个平台，用于构建和部署可扩展和可移植的机器学习 (machine learning)工作流，它建立在 Kubernetes 之上。对于机器学习实验、模型版本控制以及与 Kubernetes 生态系统的紧密集成是优先事项的 RAG 系统，它特别适合。

#### 在 Kubeflow 中设计 RAG 管道

在 Kubeflow Pipelines 中，工作流被定义为“管道”，管道中的每个步骤都是一个“组件”。组件通常是具有定义明确的输入和输出的容器化应用。

- **组件化**：每个 RAG 阶段（数据摄取、预处理、向量 (vector)化、模型训练/微调 (fine-tuning)、评估）都成为一个 Kubeflow 组件。这提高了可重用性和模块化。
  - 示例：一个 `document-chunker` 组件将数据集 URI 作为输入，并输出分块文档的 URI。一个 `embedding-generator` 组件将分块数据 URI 和向量模型 URI 作为输入，输出向量。
- **工件跟踪**：Kubeflow Pipelines 自动跟踪每个组件生成的工件（数据集、模型、指标）。这对于可复现性和血缘关系很重要的 RAG 系统非常有用，例如，跟踪特定文档集使用了哪个版本的向量模型。
- **SDK 定义**：管道通常使用 Python 中的 Kubeflow Pipelines SDK 定义，允许编程构建和参数 (parameter)化。

#### Kubeflow 组件和 Kubernetes 原生功能

Kubeflow 将 Kubernetes 的全部能力用于 RAG 工作流：

- **容器原生**：每个步骤都作为 Kubernetes Pod 运行，确保环境一致性，并运用 Kubernetes 进行资源请求（CPU、GPU、内存）、扩展和调度。
- **以 ML 为中心的集成**：Kubeflow 很容易与其生态系统中的其他 MLOps 工具集成：
  - **Katib** 用于检索模型、重排序器，甚至生成 LLM 参数的超参数 (hyperparameter)调优。
  - **KFServing（现 KServe）** 用于部署最终的 RAG 应用（例如，接受查询并返回 RAG 生成答案的 API 端点）或单个模型组件。
- **缓存**：Kubeflow Pipelines 支持组件执行缓存。如果组件的输入未改变，它可以重用之前运行的输出，明显加快 RAG 数据处理管道的开发和重复运行。

#### Kubeflow Pipelines 在 RAG 中的出色表现时机

Kubeflow Pipelines 是一个很好的选择，如果：

- 您的 RAG 系统涉及频繁试验不同的向量模型、LLM 或微调策略。
- 您需要对 ML 组件进行可靠的工件跟踪和版本控制。
- 您的团队已经熟练掌握 Kubernetes 和容器化。
- 您计划采用其他 Kubeflow 组件以获得全面的 MLOps 方案。

### RAG 编排中 Airflow 和 Kubeflow 的选择

Airflow 和 Kubeflow Pipelines 都是 RAG 系统能胜任的编排器。选择通常取决于具体的项目需求、团队专长和现有基础设施：

| 特点方面 | Apache Airflow | Kubeflow Pipelines | RAG 的最佳适配 |
| --- | --- | --- | --- |
| **主要侧重** | 通用 ETL、数据管道 | ML 工作流、实验 | 如果 ML 实验是核心，则选 Kubeflow；如果更广泛的 ETL 和数据集成是主要考量，则选 Airflow。 |
| **任务定义** | 基于 Python 的 DAG、多样化操作器 | 容器化组件、Python SDK | Airflow 为不同系统提供更多内置操作器；Kubeflow 从一开始就强制容器化。 |
| **ML 集成** | 良好，通过 Python/自定义操作器 | 原生，与 Kubeflow 生态系统深度集成 | Kubeflow 适合在其生态系统内与超参数 (parameter) (hyperparameter)调优、模型服务紧密结合。 |
| **工件跟踪** | 通过 XComs 提供基本功能，可扩展 | 内置，工件和跟踪 | Kubeflow 为 RAG 模型和数据集提供即用的工件管理。 |
| **生态系统** | 成熟，大型社区，广泛集成 | 发展中，Kubernetes 原生，侧重 ML | Airflow 适合通用数据生态系统；Kubeflow 适合已投资 Kubernetes 及其 ML 工具的情况。 |
| **可扩展性** | 高度可扩展（Celery、Kubernetes 执行器） | 通过 Kubernetes 本身就可扩展 | 两者都扩展良好，但如果已在使用 Kubernetes，Kubeflow 的 K8s 原生方法可能更直接。 |
| **用户界面** | 丰富的 UI，用于 DAG 监控和管理 | UI 侧重于管道运行、工件、实验 | Airflow 的 UI 在不同工作流的运行监控方面通常更成熟。 |

**混合方法**：使用 Airflow 编排 Kubeflow Pipelines 也是可行。例如，一个 Airflow DAG 可以触发一个 Kubeflow Pipeline 来处理 RAG 系统的机器学习 (machine learning)密集型部分（如模型微调 (fine-tuning)和评估），而 Airflow 则处理更广泛的数据摄取和调度。

### 分布式 RAG 的高级编排考量

无论选择哪种工具，对于大规模分布式 RAG 系统来说，一些高级编排模式都很重要：

- **完善的错误处理和重试策略**：对与分布式组件（例如，负载下的向量 (vector)数据库、受速率限制的 LLM API）交互的任务，实现指数退避、抖动和条件重试。在可能的情况下，设计为优雅降级。
- **状态管理和恢复**：对于长时间运行的 RAG 工作流（例如，索引数 TB 数据），确保编排器可以从故障点恢复，而不是重新启动整个过程。这通常涉及仔细的任务设计和外部状态持久化。
- **大规模幂等性**：对于数据摄取和索引任务非常重要。确保由于暂时性故障或手动触发而重新运行任务不会导致向量存储或知识库中的数据重复或损坏。
- **动态参数 (parameter)化和配置**：您的 RAG 工作流应通过编排器实现高度可配置。这包括指定数据源、向量模型版本、LLM 端点、任务的资源分配以及评估指标的阈值。将这些配置存储在版本控制系统中。
- **警报和主动监控**：将您的编排器与全面的监控系统（例如 Prometheus、Grafana）和警报工具（例如 PagerDuty、Slack）集成。为管道故障、任务超出 SLA、资源耗尽以及 RAG 质量指标明显下降定义警报。
- **回填和再处理**：设计工作流以高效处理回填（使用新逻辑或模型处理历史数据）或特定数据子集的再处理。这通常需要可以对定义的数据分区或时间范围进行操作的任务。

通过仔细选择和配置像 Airflow 或 Kubeflow Pipelines 这样的工作流编排器，并实施这些高级模式，您可以为大规模分布式 RAG 系统构建有韧性、可管理和可扩展的运行过程。这为生产级 AI 的持续部署、监控和改进循环打下良好基础。

## 参考资料

- [Apache Airflow Documentation](https://airflow.apache.org/docs/) — Apache Software Foundation (2024)
  Airflow架构、操作符、执行器以及数据管道定义最佳实践的全面指南。
- [Kubeflow Pipelines Documentation](https://www.kubeflow.org/docs/pipelines/) — The Kubeflow Project Authors (2025)
  Kubeflow Pipelines组件、SDK、工件追踪和Kubernetes集成的官方文档。
- [TFX: A TensorFlow-Based Production Machine Learning Platform](https://dl.acm.org/doi/10.1145/3292500.3330682) — Zhitao Li, Weiwei Yang, Hongjun Wu, Jun Zhang, Hongxia Yang, Jianshu Chen and Xiaodong Zhang (2019)
  Journal: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining; Publisher: Association for Computing Machinery; Pages: 2275-2285; DOI: [10.1145/3292500.3330682](https://doi.org/10.1145/3292500.3330682)
  介绍一个全面的生产机器学习平台，展示管道编排、数据验证和模型管理的原则。

---

[上一节](../04-%E8%A7%84%E6%A8%A1%E5%8C%96%E6%95%B0%E6%8D%AE%E6%91%84%E5%85%A5%E4%B8%8E%E5%A4%84%E7%90%86%E7%AE%A1%E9%81%93/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5-%20%E6%9E%84%E5%BB%BA%E5%8F%AF%E4%BC%B8%E7%BC%A9%E6%95%B0%E6%8D%AE%E6%91%84%E5%8F%96%E7%AE%A1%E9%81%93.md) · [下一节](02-RAG%20%E7%BB%84%E4%BB%B6%E7%9A%84%E5%BE%AE%E6%9C%8D%E5%8A%A1%E8%AE%BE%E8%AE%A1%E6%A8%A1%E5%BC%8F.md)

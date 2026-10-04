# LLMOps 工具选用要点

来源：[原文](https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-1-foundations-llmops/llmops-tooling-considerations)

[返回章节目录](README.md) · [返回课程目录](../README.md)

大型语言模型 (LLM) 庞大的规模和独特的运行特点往往使传统 MLOps 工具难以应对。因此，选择合适的工具不仅仅是偏好问题，更是构建成功且可持续的 LLMOps 实践的重要组成部分。您构建的工具链必须有效应对大规模数据集、分布式计算、专门部署要求和特定监控需求相关的挑战。

本节介绍 LLMOps 生命周期中选择工具的考虑事项。我们不会规定单一的确定性工具栈，因为最佳选择很大程度上取决于具体的项目需求、现有基础设施、团队专业知识和预算。相反，我们侧重于每个阶段所需的功能，并列举了满足 LLM 特殊要求的代表性工具。

### 数据管理与处理工具

LLM 训练通常从数 TB 甚至数 PB 的非结构化文本数据开始。标准的数据仓库和处理工具可能难以应对这种数据规模和特点。

- **存储：** 云对象存储（AWS S3、Google Cloud Storage、Azure Blob Storage）因其可扩展性和成本效益而被普遍采用。在高性能计算 (HPC) 环境中，可能需要高性能文件系统（例如 Lustre、BeeGFS）以加快训练数据访问速度。
- **处理：** 专为大规模分布式数据处理设计的框架非常重要。Apache Spark 仍然是广受欢迎的选择。Ray（特别是 Ray Data）等框架正逐渐获得认可，因为它们能与通常使用 Ray Train 执行的下游分布式训练任务很好地配合。重点在于构建能在多个节点上运行的高效分词 (tokenization)、清洗、过滤和格式化管道。
- **版本控制：** 对 PB 级数据集和数百 GB 大小的模型检查点进行版本控制带来显著难题。Data Version Control (DVC) 等工具可以使用，但对于大型二进制文件可能需要仔细配置。LakeFS 为数据湖提供类似 Git 的体验。Pachyderm 通过容器化管道提供数据溯源和版本控制。目标是实现数据和模型的可重现性及可追溯性，这因其庞大体量而变得复杂。

### 实验追踪与管理

追踪 LLM 实验需要能够处理比普通机器学习 (machine learning)运行更复杂信息的平台。

- **可扩展性：** MLflow、Weights & Biases (W&B) 和 ClearML 等标准工具通常被采用，但其后端基础设施必须足够强大，以应对分布式训练或大规模超参数 (parameter) (hyperparameter)优化 (HPO) 过程中可能产生的海量指标、参数和制品。可以考虑采用带有可扩展数据库的自托管部署或云服务的企业版。
- **LLM 特有元数据：** 追踪需要捕获分布式训练特有的配置详情（例如，节点数量、每节点 GPU 数量、数据/张量/流水线并行设置）、PEFT 配置（例如，LoRA 秩、适配器类型）、量化 (quantization)参数，甚至微调 (fine-tuning)或评估期间使用的提示模板。
- **可视化：** 可视化数十甚至数百个 GPU 上的训练进展、追踪不同并行策略的收敛情况以及比较微调运行，需要灵活且功能强大的可视化功能，而不仅仅是简单的损失曲线。

### 分布式训练框架

在单个加速器上训练拥有数十亿参数 (parameter)的模型是不可能的。专门框架协调所需的复杂并行处理。

- **核心框架：** DeepSpeed（微软出品）、Megatron-LM（英伟达出品）、PyTorch 的 Fully Sharded Data Parallel (FSDP) 和 JAX 的 `pjit` 是主要代表。这些不单是库，更是提供以下实现方式的综合系统：
  - 数据并行：复制模型并对数据进行分片。
  - 张量并行：将单个层拆分到不同设备上。
  - 流水线并行：按顺序将层划分到不同设备上。
  - 内存优化：来自 DeepSpeed 的 ZeRO（零冗余优化器）等技术。
- **集成：** 这些框架通常与编排器和实验追踪工具集成，但在工具栈中代表一个独立的层，纯粹侧重于大规模训练执行的机制。您的 MLOps 工具必须能够使用这些框架启动、管理和监控作业。

### 模型注册中心

存储和版本控制 LLM 检查点需要能够处理可能非常大的制品的注册中心。

- **存储后端：** 注册中心通常使用云对象存储作为其后端。与可扩展存储的直接集成必不可少。
- **版本控制：** 版本控制是必需的，不仅适用于最终模型，也常用于对恢复长时间训练作业或进行微调 (fine-tuning)很重要的中间检查点。
- **元数据与搜索：** 需要丰富的元数据支持，以存储训练参数 (parameter)、所用数据集、评估结果和预期用例等信息。随着模型数量的增长，高效的搜索和查找变得重要。
- **范例：** Hugging Face Hub 已成为开源模型的实际标准，提供版本控制和元数据功能。AWS SageMaker、Google Vertex AI 和 Azure Machine Learning 等云服务商提供集成的模型注册中心。MLflow 也包含一个注册中心，但对于超大型模型，若无仔细的后端配置，扩展性可能是一个担忧。

### 部署与推理 (inference)服务

高效地服务 LLM 需要专门的推理服务器，这些服务器针对大型模型和 GPU 执行进行了优化。用于较小模型的标准 CPU 服务器或基本 GPU 服务器通常力有不逮。

- **优化推理服务器：** NVIDIA Triton Inference Server、TensorRT-LLM、vLLM 和 Ray Serve（带有 LLM 特殊优化）等工具旨在最大限度提高 LLM 的 GPU 利用率和吞吐量 (throughput)。它们通常采用以下技术：
  - 连续批处理：动态批处理传入请求以提高 GPU 利用率。
  - 分页注意力：为注意力机制 (attention mechanism)提供更高效的内存管理。
  - 优化内核：使用 FasterTransformer 或 CUTLASS 等底层库。
- **量化 (quantization)与优化支持：** 部署工具应轻松与量化库（例如 `bitsandbytes`、`AutoGPTQ`、`AWQ`）集成，或支持优化的模型格式（例如 TensorRT 引擎、ONNX Runtime）。MLOps 管道需求管理量化过程，并部署由此产生的更小、更快的模型。
- **部署策略：** 工具应支持高级模式，如金丝雀发布和 A/B 测试（常用于比较不同的提示或微调 (fine-tuning)模型版本），并具备流量拆分能力。Kubernetes（通常通过 KServe 或云服务商提供）常用用于编排，同时无服务器 GPU 选项正在出现，但可能存在冷启动限制。

### 监控与可观察性

监控 LLM 不仅仅是标准基础设施和性能指标。它需要追踪成本、输出质量以及漂移或幻觉 (hallucination)等潜在问题。

- **基础设施监控：** Prometheus 和 Grafana 等标准工具，与 GPU 特有的导出器（例如 NVIDIA DCGM exporter）集成，对于追踪 GPU 利用率、内存使用、功耗和网络 I/O 是必需的。云服务商的监控服务（CloudWatch、Google Cloud Monitoring、Azure Monitor）同样不可或缺。
- **性能与成本：** 追踪推理 (inference)延迟（通常按令牌测量）、吞吐量 (throughput)（每秒令牌数）和请求并发性非常重要。将这些与特定模型版本或部署配置关联起来很重要。成本监控需要追踪训练和推理的 GPU 实例小时数，并将使用量与特定项目或团队关联。此处需要云计费工具和成本管理平台。
- **输出质量与漂移：** 这是一个复杂方面。一些工具正在出现，以帮助监控 LLM 输出的以下方面：
  - *漂移：* 随着时间推移检测输入提示分布或生成输出的相关性/风格的变化。
  - *质量指标：* 追踪任务特定的指标、毒性分数、情感、PII 存在情况或自定义质量指示器。通常涉及输出的统计分析或使用另一个模型（或人工反馈）进行评估。
  - *幻觉检测：* 使用不确定性估计、检查输出一致性或对照知识库进行事实核查等技术。需求专门工具或自定义实现。
  - *反馈循环：* 收集明确用户反馈（赞成/反对、修正）或隐式反馈（用户参与度）的平台对于持续改进很重要。
- **LLM 特有平台：** LangSmith、Arize AI、WhyLabs、Fiddler AI 等工具以及 `langkit` 或 `promptfoo` 等开源库专门为 LLM 可观察性而设计，用于追踪复杂链（例如 RAG 系统中）中的请求、评估输出并检测问题。

### 向量 (vector)数据库管理（用于 RAG）

如果使用检索增强生成 (RAG)，管理相关的向量数据库就成为一项操作任务。

- **选项：** 托管服务（Pinecone、Weaviate Cloud Services）或自托管数据库（Milvus、Weaviate、Qdrant、ChromaDB）。
- **操作：** 任务包括模式管理、大型文档语料库的高效索引、根据查询负载扩展数据库、执行索引更新（增量或完全重建）、监控查询延迟以及管理成本。这些操作需求整合到更广泛的 LLMOps 工作流中。

### 工作流编排

将所有这些阶段连接起来需求工作流编排。

- **工具：** Apache Airflow、Kubeflow Pipelines、Argo Workflows、Prefect、Dagster 或 Metaflow 等标准编排器可以进行调整。Ray 提供与其数据处理和训练组件紧密结合的原生编排能力。
- **LLM 考量：** 编排器需要处理长时间运行的分布式作业，管理数据处理、训练、评估、部署和监控任务之间的依赖关系，处理大型制品传递（或引用），并通过 API 或 SDK 与专门的 LLM 工具集成。

### 集成与工具栈选择

选择工具时的一些重要考量包括：

- **可扩展性：** 该工具能否处理所需的数据量、模型大小和计算负载？
- **成本：** 授权费用以及在规模化运行该工具时相关的基础设施成本如何？
- **集成：** 该工具与您工具栈的其他部分（例如，云存储、计算框架、监控平台）连接的便利程度如何？它是否有明确定义的 API？
- **LLM 功能支持：** 它是否明确支持 LLM 所需的功能，例如分布式训练模式、PEFT、量化 (quantization)、特定监控指标或向量 (vector)数据库交互？
- **灵活性与托管服务：** 云平台提供集成套件（例如 SageMaker、Vertex AI、Azure ML），这些套件简化了设置但可能涉及供应商锁定。结合行业最佳的开源或专门工具提供灵活性，但需要更多的集成工作。
- **社区与支持：** 对于开源工具，请考虑社区的规模和活跃度。对于商业工具，请评估供应商支持的质量。

> LLMOps 工具栈中各工具类别如何配合的视图。箭头表示典型的数据或控制流。可选的 RAG 组件单独显示，但与服务和编排集成。

最终，构建有效的 LLMOps 工具链，需要理解大型模型在其生命周期各个阶段带来的具体要求，并根据您的组织情况，明智选择最能满足这些要求的工具或工具组合。接下来的章节将详细阐述在这些类别中实现解决方案的具体内容。

## 参考资料

- [ZeRO: Memory Optimizations Toward Training Trillion-Parameter Models](https://doi.org/10.1145/3418856.3418915) — Samyam Rajbhandari, Cong Guo, Jeff Rasley, Shaden Smith, Yuxiong He (2020)
  Journal: SC '20: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis; Publisher: ACM (Association for Computing Machinery); Pages: 1-16; DOI: [10.1145/3418856.3418915](https://doi.org/10.1145/3418856.3418915)
  描述了ZeRO，这是DeepSpeed中用于训练大型模型的一项内存优化技术，对LLM的可伸缩性有帮助。
- [The Ray Project Documentation](https://docs.ray.io/en/latest/) — Anyscale, Inc. (2024)
  Ray的官方文档，Ray是一个分布式计算系统，用于LLM数据处理（Ray Data）、分布式训练（Ray Train）和服务（Ray Serve）。
- [vLLM: Efficient Memory Management for Large Language Model Serving](https://arxiv.org/pdf/2309.06180.pdf) — Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica (2023)
  Journal: arXiv preprint arXiv:2309.06180; DOI: [10.48550/arXiv.2309.06180](https://doi.org/10.48550/arXiv.2309.06180)
  介绍了vLLM，这是一种使用PagedAttention实现高吞吐量和低延迟的LLM推理引擎，有助于满足部署需求。
- [LLMOps: Towards a Standardized, Streamlined, and Scalable Workflow for Large Language Models](https://arxiv.org/pdf/2306.05942.pdf) — Zhifeng Zhang, Bo Qiao, Mengjie Li, Guohao Li, Yuan Jiang, Chengyue Wu, Weiqi Wang, Long Cheng, Huaijun Li, Bo Dong, Xiaoyong Hu (2023)
  Journal: arXiv preprint arXiv:2306.05942
  一篇综述文章，概述了LLMOps的生命周期和考量，为工具选择提供了概念性框架。

---

[上一节](04-LLMOps%20%E7%94%9F%E5%91%BD%E5%91%A8%E6%9C%9F%E9%98%B6%E6%AE%B5.md) · [下一节](../02-%E5%A4%A7%E8%A7%84%E6%A8%A1%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD%E4%B8%8E%E6%95%B0%E6%8D%AE%E7%AE%A1%E7%90%86/01-%E8%AE%BE%E8%AE%A1%E5%8F%AF%E6%89%A9%E5%B1%95%E7%9A%84%E8%AE%A1%E7%AE%97%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD.md)

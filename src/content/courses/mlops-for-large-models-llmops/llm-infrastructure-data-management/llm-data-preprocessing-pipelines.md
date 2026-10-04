---
course: "mlops-for-large-models-llmops"
chapter: "llm-infrastructure-data-management"
lesson: "llm-data-preprocessing-pipelines"
sourceId: 3124
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-2-llm-infrastructure-data-management/llm-data-preprocessing-pipelines"
title: "大语言模型数据预处理流程"
description: "开发高效且可扩展的数据清理、分词和预处理流程。"
order: 4
plots: []
sourceHash: "3e432813d70d0c37add44b6291c3e0ad608ce340d7b5855450b459600ec22c2a"
sourceCorrections: []
---

从标准机器学习 (machine learning)模型转向大语言模型 (LLM)，数据处理需求发生了显著变化。如前所述，我们通常处理的数据集以太字节甚至拍字节衡量，来源高度多样，例如网络抓取、数字化书籍、代码库和对话日志。简单应用处理小型数据集的预处理技术，在计算上往往不可行，也无法解决对大语言模型行为有重大影响的数据独特属性。

构建适用于大语言模型的有效数据预处理流程是LLMOps中的一项基础任务。这些流程必须设计为支持大规模、高效率和可复现。它们的目标不仅是准备数据供训练框架使用，还要筛选高质量输入，以最大程度地减少噪声、偏差和冗余，这直接影响了最终模型的能力和安全性。

### 大语言模型 (LLM)数据预处理的特定挑战

处理网络规模的数据带来了几个不同的困难：

1. **超大规模：** 处理流程必须处理比典型机器学习 (machine learning)数据集大几个数量级的数据量。这需要分布式计算框架和优化的存储访问模式。读取和写入拍字节数据需要仔细的基础设施规划。
2. **异构性和噪声：** 原始数据，尤其是来自网络抓取的数据，在格式、质量和语言方面极其多样。它通常包含大量噪声，例如HTML标记 (token)、样板文本（菜单、广告）、重复内容以及低质量甚至有害语言。在不丢弃有价值信息的情况下有效过滤和清理这些数据是一项复杂的权衡。
3. **计算开销：** 精确去重、质量评分，尤其是分词 (tokenization)等操作，在应用于数十亿或数万亿个词元 (token)时，会成为主要的计算瓶颈。优化这些步骤对于管理训练成本和时间表非常重要。
4. **下游影响：** 预处理选择对训练后的大语言模型有深刻且有时不易察觉的影响。关于过滤、去重和分词策略的决定会影响模型在特定任务上的性能，引入或减轻偏差，并影响模型记忆或产生幻觉 (hallucination)的倾向。

### 大语言模型 (LLM)预处理流程的核心阶段

尽管具体的实现因数据源和模型目标而异，但大多数大语言模型预处理流程都包含几个共同的阶段，并使用分布式系统执行：

#### 1. 数据摄取和加载

第一步是访问原始数据，这些数据通常存储在分布式对象存储（如AWS S3、Google Cloud Storage、Azure Blob Storage）中。高效读取这些数据需要并行I/O能力。Apache Spark、Dask和Ray Data等框架常用于在集群中的多个工作节点上加载数据分区。配置时需要考虑最佳分区大小和数据局部性等因素，以最大程度地减少网络传输时间。

#### 2. 清理和过滤

这通常是最复杂且依赖启发式方法的阶段。常见的子步骤包括：

- **样板内容移除：** 识别并去除网页中的非内容元素，如HTML标签、导航栏、广告和页脚。`BeautifulSoup`等库或`trafilatura`等专业工具可以提供帮助，但通常需要定制。
- **语言识别：** 过滤文档以只保留所需语言，特别是对于多语言数据集。`fastText`或`langdetect`等库是可选方案。
- **质量过滤：** 应用启发式方法去除低质量内容。这可能涉及基于文档长度、符号与词语比率、是否存在“不良词汇”列表进行过滤，甚至使用小型模型的得分（例如，困惑度得分来过滤非自然语言）。需要仔细调整以避免对某些方言或内容类型产生偏见。
- **有害内容和个人身份信息（PII）过滤：** 识别并移除或遮盖有害内容和个人身份信息（PII）。这需要大量计算，通常依赖于专用模型或基于规则的系统。
- **去重：** 识别并移除重复或近似重复的文档。精确重复的文档相对容易处理，但近似重复的文档需要使用MinHash结合局部敏感哈希（LSH）等大规模执行的技术。去重很重要，可以防止模型过度强调冗余信息，并可能提高训练效率。

#### 3. 规范化

该阶段标准化文本格式。常见步骤包括：

- Unicode规范化（例如，NFKC），以确保一致的字符表示。
- 转为小写（尽管这对于涉及代码或命名实体的任务可能有害，需要仔细考虑）。
- 处理特殊字符或特定领域语法（例如，保留代码结构）。

#### 4. 分词 (tokenization)

大语言模型处理的是词元 (token)序列，而非原始文本。此阶段将清理后的文本根据预定义词汇表 (vocabulary)转换为整数词元ID。

- **分词器 (tokenizer)训练：** 子词 (subword)分词算法，如字节对编码（BPE）、WordPiece或SentencePiece是标准方法。分词器本身必须在预处理数据的大规模、有代表性的样本上训练，以构建其词汇表和合并规则。此训练过程是一个重要的离线步骤。
- **应用分词器：** 训练好的分词器随后应用于整个数据集。由于文本量庞大，此步骤高度并行化，但计算密集。优化的分词器实现（例如，Hugging Face `tokenizers`库，由Rust支持）是必不可少的。
- **序列处理：** 关于最大序列长度（$L_{max}$）、填充和截断策略的决定在此处实现，准备数据用于打包或直接馈送到训练过程。

#### 5. 格式化和分片

最后，词元化后的数据通常会被格式化为适合训练框架的结构（例如，Apache Arrow、TFRecord或自定义二进制格式）。数据通常会被打乱并分片成大小适中的文件，以便在分布式训练期间高效加载。此阶段还可能涉及序列打包等技术，其中多个较短的序列被连接（并带有适当的注意力掩码）成一个长度为$L_{max}$的单个序列，以提高训练期间的GPU利用率。

### 为规模和效率设计

要使这些流程高效运行，需要特定技术：

- **分布式计算：** 使用Apache Spark、Dask或Ray等框架，将工作负载分配到计算集群中。这些框架处理任务调度、数据分区和容错。
- **优化库：** 对于文本清理（正则表达式）、去重（哈希）和分词 (tokenization)（原生代码实现）等CPU密集型任务，使用高度优化的库。
- **内存管理：** 仔细管理工作节点上的内存使用，尤其是在处理大型文档或复杂数据结构时。批量处理数据或使用内存映射等技术可以提供帮助。
- **中间存储：** 在流程阶段之间使用高效的中间存储格式（例如，Parquet、Arrow），以减少I/O开销。
- **异步处理：** 在可能的情况下，设计阶段以异步方式运行，以最大程度地提高资源利用率。

> 一个分布式大语言模型 (LLM)数据预处理流程示意图，显示了从原始数据到可用于训练的分片的主要阶段。

### 编排与复现性

考虑到这些流程的复杂性和持续时间，细致的编排和版本控制是必不可少的。

- **工作流编排：** Apache Airflow、Kubeflow Pipelines或Argo Workflows等工具帮助定义、调度和监控多阶段流程，管理长时间运行作业的依赖关系和重试。
- **版本控制：** 每个组件都需要版本控制：
  - **数据：** 使用数据版本控制工具（如DVC、LakeFS）或简单的对象存储版本控制/快照来跟踪输入数据集。
  - **代码：** 对预处理脚本进行版本控制（Git）。
  - **配置：** 对流程配置进行版本控制，包括过滤参数 (parameter)和分词 (tokenization)器 (tokenizer)设置。
  - **分词器：** 对训练好的分词器模型进行明确的版本控制。
- **数据血缘：** 追踪哪些数据版本和预处理步骤产生了给定的训练分片集。这对于调试、审计和复现结果来说是不可或缺的。

构建可扩展、高效且可复现的数据预处理流程并非易事，但它是成功训练和微调 (fine-tuning)大语言模型 (LLM)的根本。从该流程输出的数据质量和特点直接决定了最终模型的性能和行为。

## 参考资料

- [Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing](https://www.usenix.org/system/files/conference/nsdi12/nsdi12-final138.pdf) — Matei Zaharia, Mosharaf Chowdhury, Tathagata Das, Ankur Dave, Justin Ma, Haoyuan Li, Scott Shenker, Ion Stoica (2012)
  Journal: 9th USENIX Symposium on Networked Systems Design and Implementation (NSDI '12); Publisher: USENIX Association; Pages: 153-168; DOI: [10.5555/2393325.2393341](https://doi.org/10.5555/2393325.2393341)
  这篇基础论文介绍了弹性分布式数据集（RDDs），它是Apache Spark的核心抽象，对可扩展和容错的分布式数据处理具有重大意义。
- [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://aclanthology.org/D18-2012/) — Taku Kudo, John Richardson (2018)
  Journal: Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations; Publisher: Association for Computational Linguistics; Pages: 66-71; DOI: [10.18653/v1/D18-2012](https://doi.org/10.18653/v1/D18-2012)
  本文详细介绍了SentencePiece库，它实现了BPE和WordPiece等子词分词算法，这些算法在LLM预处理中被广泛使用。

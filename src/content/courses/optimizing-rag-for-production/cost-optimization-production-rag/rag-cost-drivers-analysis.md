---
course: "optimizing-rag-for-production"
chapter: "cost-optimization-production-rag"
lesson: "rag-cost-drivers-analysis"
sourceId: 6219
sourceUrl: "https://apxml.com/zh/courses/optimizing-rag-for-production/chapter-5-cost-optimization-production-rag/rag-cost-drivers-analysis"
title: "识别生产RAG中的成本要素"
description: "识别并分析生产RAG系统中的主要成本要素，包括大语言模型API使用、向量数据库成本和计算资源。"
order: 1
plots: []
sourceHash: "3e61984f387ea058c362b5c01b5f202723726a366250f756a9c6fc03e9ab2b97"
sourceCorrections: [{"kind": "escaped-dollar-math", "line": 17}]
---

为了有效管理您的RAG系统的财务开销，第一步是全面分析整个流程，并确定成本发生的地方。生产RAG系统通常由多个相互作用的服务组成，每个服务都有自己的成本特点。详细了解这些成本要素是进行有针对性的优化工作的基础。没有这种清晰度，降低成本的尝试可能会效率低下且方向错误。

接下来，我们细分RAG系统产生开销的常见方面。我们可以大致将这些成本分为每次查询的操作成本和持续的基础设施或平台成本。

### 每次查询的核心成本要素

这些成本随着RAG系统的使用量直接增加，通常在每次处理查询时产生。

#### 1. 大语言模型 (LLM) API 和模型使用

这通常是许多RAG系统中最重要和最明显的成本组成部分。大语言模型，无论是通过API（如OpenAI、Anthropic、Google）访问还是自行部署，都直接涉及计算和许可方面的开销。

- **令牌消耗**：大多数商业大语言模型API根据处理的令牌数量收费。这包括：
  - **输入令牌**：发送给大语言模型的提示中的令牌，在RAG系统中，这包括用户查询*加上*检索到的上下文 (context)。更长、更多或更详细的检索块会直接增加输入令牌的数量。
  - **输出令牌**：大语言模型生成响应中的令牌。冗长的回答或要求详细解释会增加输出令牌的数量。
    许多模型对输入和输出令牌有不同的定价，输入令牌通常更便宜。例如，一个模型可能对每1,000个输入令牌收取$0.001，对每1,000个输出令牌收取$0.002。一个涉及3,000个输入令牌（查询+上下文）和500个令牌响应的查询将花费：
    $LLM成本 = (\frac{3000}{1000} \times \text{\textdollar}0.001) + (\frac{500}{1000} \times \text{\textdollar}0.002) = \text{\textdollar}0.003 + \text{\textdollar}0.001 = \text{\textdollar}0.004$
- **模型选择**：更强大的模型（例如GPT-4、Claude 3 Opus）每令牌的价格比更小或更优化的模型（例如GPT-3.5 Turbo、Claude 3 Haiku、Llama 3 8B）贵很多。选择一个对任务来说“过于优秀”的模型可能会导致不必要的开销。
- **多次大语言模型调用**：一些高级RAG模式可能涉及每个用户查询的多次大语言模型调用。例如：
  - 大语言模型用于改写或扩展用户查询以实现更好的检索。
  - 大语言模型用于在最终合成之前评估检索到的块的相关性。
  - 大语言模型用于总结或合成来自多个块的信息。
    每次额外的调用都会增加总令牌数量和成本。
- **微调 (fine-tuning)模型**：如果您正在使用微调的大语言模型，微调过程本身会产生费用（计算和可能的专用托管），并且通常会有不同的（有时更高）推理 (inference)定价结构。

#### 2. 嵌入 (embedding)生成

嵌入是文本数据的数值表示，支持语义搜索。这里的成本源于：

- **初始索引**：当您首次构建知识库时，每个文档块都需要转换为嵌入。对于大型数据集，这种初始批量处理可能会产生一次性的大量费用，特别是如果使用付费嵌入API。
- **查询嵌入**：每个传入的用户查询也必须进行嵌入才能执行相似性搜索。这是一项每次查询的成本。
- **模型选择及API与自行部署**：与大语言模型类似，您可以使用商业嵌入API（例如OpenAI Ada、Cohere Embed）或自行部署开源嵌入模型。API调用按令牌或按文本块定价。自行部署涉及计算成本（CPU或GPU，取决于模型和吞吐量 (throughput)要求）。
- **数据更新**：当您的知识库更新（添加新文档、修改现有文档）时，必须生成新的嵌入。这些更新的频率和数量会影响持续的嵌入成本。

#### 3. 检索系统

查询和文档嵌入后，检索系统会找到最相关的块。这涉及：

- **向量 (vector)数据库操作**：
  - **查询成本**：大多数向量数据库根据执行相似性搜索（读取操作）所消耗的计算资源收费。影响因素包括搜索的向量数量、搜索的复杂性（例如元数据过滤）以及所需的查询吞吐量。
  - **存储成本**：存储高维向量，特别是对于大型知识库，会产生存储费用。这些通常按每月每GB定价。
  - **索引成本**：构建和维护索引（例如HNSW、IVF）会消耗计算资源。一些托管向量数据库将其打包到整体定价中，而另一些则可能更直接地呈现出来。
  - **数据传输**：数据进出向量数据库也会增加成本，特别是对于大数据量或跨区域流量。
    托管向量数据库服务（例如Pinecone、Weaviate Cloud Services、Zilliz Cloud）简化了操作，但其定价层级反映了这些基础资源消耗。
- **重排器计算**：如果您的RAG流程包含重排步骤（例如，使用交叉编码器模型来提高前k个检索到的文档的相关性），这将增加另一层计算成本。重排器可能是计算密集型的，通常需要GPU加速以实现低延迟，特别是当需要重排大量候选文档时。
- **关键词搜索组件**：对于混合搜索系统，关键词搜索的基础设施（例如Elasticsearch、OpenSearch）也有其自身的计算、存储和操作成本。

### 基础设施和运营开销

除了每次查询的成本，还有与支持RAG系统的平台和基础设施相关的固定成本。

- **数据存储（非向量 (vector)）**：
  - **原始文档存储**：您的原始文档（PDF、HTML、文本文件等）需要存储，通常在对象存储中，如AWS S3、Google Cloud Storage或Azure Blob Storage。成本取决于容量和存储层级。
  - **缓存**：对频繁访问的数据（例如，常见查询的大语言模型 (LLM)响应、热门文档嵌入 (embedding)、检索到的上下文 (context)）实施缓存可以降低下游服务的延迟和成本。然而，缓存基础设施本身（例如Redis、Memcached实例）有其自身的计算和内存成本。
- **数据摄取和预处理流程**：
  为RAG系统准备数据的抽取、转换、加载（ETL）过程（获取、解析、清理、分块）会消耗计算资源。这些成本取决于数据量、转换的复杂性以及流程运行的频率（批量与流式）。
- **计算平台和网络**：
  - **应用服务器/容器**：核心RAG应用逻辑（协调检索、提示工程 (prompt engineering)、大语言模型交互）运行在服务器、虚拟机或容器平台（例如Kubernetes）上。成本取决于实例类型（CPU、内存、如果需要自行部署模型则需要GPU）、实例数量和运行时间。
  - **无服务器函数**：对RAG流程的部分使用无服务器函数（例如AWS Lambda、Google Cloud Functions）可以提供按使用付费的优势，但成本可能随着高调用次数或长执行时间而增加。
  - **网络**：服务之间的数据传输（例如应用服务器到向量数据库、应用服务器到大语言模型API）会产生费用，特别是当这些服务位于不同可用区或区域时。出站用户流量也有贡献。
  - **负载均衡器和API网关**：这些组件对于可伸缩性和流量管理非常重要，它们有自己的服务费用。
- **监控、日志和警报系统**：
  为实现可观测性而收集指标、日志和跟踪对于生产系统很重要。这些系统的基础设施（例如Prometheus、Grafana、ELK堆栈、Datadog、New Relic等托管服务）会产生与数据摄取、存储和查询相关的成本。

### 隐含或不那么明显的成本

最后，有些成本不那么直接，但会对总拥有成本带来很大影响：

- **实验和开发**：构建和迭代RAG系统通常涉及对不同模型、提示和配置进行大量实验。这可能意味着运行多个开发或预发布环境、A/B测试基础设施，以及为非生产工作负载消耗资源，尽管这些工作负载是开发生命周期的一部分。
- **人工介入 (HITL) 和数据标注**：对于需要高准确性或持续改进的系统，用于审查输出、提供反馈或为微调 (fine-tuning)标注数据的人工介入流程可能会涉及大量人工成本。
- **维护和更新**：定期更新模型、依赖项和知识库本身需要工程投入。在架构更改或重大数据更新后，重新索引大型向量 (vector)数据库也可能是一项定期但大量的费用。
- **安全和合规**：实施安全措施、数据隐私控制并确保符合法规可能需要专用工具或服务，这会增加总成本。

了解这种全面细分是有效成本管理的根本。下图说明了这些成本通常在RAG流程中产生的位置。

> 此图表突出了RAG查询生命周期中产生直接成本的主要阶段（嵌入 (embedding)、检索、生成），以及增加总体运营开销的持续基础设施和支持流程。

通过仔细追踪这些类别的开销，您可以识别最大的成本中心并优先考虑优化的方面，我们将在后续章节中进行探讨。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍了基础的检索增强生成（RAG）架构，详细说明了其核心组件（检索器和生成器），这些是成本驱动因素的基础。
- [Tips for working with large language models efficiently](https://openai.com/blog/tips-for-working-with-large-language-models-efficiently) — OpenAI (2023)
  Journal: OpenAI Blog; Publisher: OpenAI
  提供了优化大型语言模型token使用、模型选择和API交互的实用建议和策略，以降低生产RAG系统的运营成本。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  涵盖了构建和运行高效机器学习系统的原则，包括基础设施、监控以及管理生产环境中的总拥有成本。

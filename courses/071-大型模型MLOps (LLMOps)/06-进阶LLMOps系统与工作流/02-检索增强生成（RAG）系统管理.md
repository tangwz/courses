# 检索增强生成（RAG）系统管理

来源：[原文](https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-6-advanced-llmops-systems-workflows/managing-rag-systems)

[返回章节目录](README.md) · [返回课程目录](../README.md)

检索增强生成（RAG）系统代表了一项显著进展，它通过将大型语言模型（LLM）的回答与外部知识来源相结合，使其内容更准确，对上下文 (context)的理解更到位。虽然功能强大，但实施RAG会增加LLM部署的复杂性。有效管理这些系统需要在您的LLMOps实践中投入专门精力，将检索组件、数据管道和生成器本身的管理整合到一个协调一致的工作流程中。

与独立的LLM不同，RAG系统是一种多组件架构，通常包括：

1. **检索器：** 根据用户查询，负责从知识来源获取相关文档或文本片段。
2. **知识来源 / 向量 (vector)数据库：** 存储外部信息，通常以索引格式（如向量数据库中的嵌入 (embedding)）以便高效搜索。
3. **生成器：** 一个LLM，根据原始查询和检索到的上下文生成答案。
4. **协调器：** 管理用户查询、检索器、知识来源和生成器之间的流程。

成功管理一个RAG系统意味着确保每个组件都能正确高效地运行，并且它们之间能良好地配合。

### RAG系统中的运行难题

部署和维护RAG系统会带来独有的运行难题：

- **组件间的相互依赖：** 系统的整体表现与每个部分的表现紧密相关。向量 (vector)数据库查询慢会直接影响用户感受到的延迟。检索质量差会直接导致生成答案不准确或不相关，无论LLM的能力如何。监控必须涵盖整个环节。
- **检索质量管理：** 检索器的效能高低十分重要。知识来源中的过时信息、不理想的文档分块或选择不当的嵌入 (embedding)模型都可能降低检索相关性。持续评估和提升检索准确性是一项持续的运行任务。
- **延迟权衡：** 与直接进行LLM推理 (inference)相比，RAG引入了额外步骤。检索步骤（查询向量数据库、处理结果）会增加延迟。优化检索过程（例如，索引调整、高效嵌入查找）以及可能使用更小、更快的模型进行检索或生成是常见的方法，但这需要仔细管理和性能监控。
- **数据管道复杂性：** 支撑RAG系统的知识来源通常是动态的。实施管道用于摄取新数据、更新现有文档、处理（分块、嵌入）并将这些信息索引到向量数据库中是一项重要的运行要求。这些管道需要监控、版本控制和故障处理。
- **复杂评估：** 评估RAG系统并非简单直接。您需要衡量指标，这些指标不仅要评估最终生成答案的质量，还要评估检索到的上下文 (context)的相关性和准确性。标准LLM评估指标不够；需要RAG专有指标，如上下文相关性、忠实度（答案与上下文之间的一致性）和答案相关性。
- **成本因素：** RAG系统除了LLM推理成本外，还引入了不同的成本组成部分，包括向量数据库托管和查询成本、嵌入模型推理成本（在索引期间和潜在查询时），以及知识库和向量索引的存储成本。成本监控和优化变得复杂。

### RAG管理的核心运行任务

应对这些难题需要特定的运行实践：

#### 1. 检索器性能监控与优化

检索器是外部知识的入口。它的表现决定了提供给生成器的上下文 (context)质量。

- **监控指标：** 追踪`Recall@K`（正确文档是否在检索到的前K个中？）和`Mean Reciprocal Rank (MRR)`等指标来量化 (quantization)检索有效性。监控检索步骤的查询延迟。
- **优化策略：**
  - **分块：** 尝试不同的文档分块策略（固定大小、基于句子、重叠），因为这会很大程度影响检索质量。
  - **嵌入 (embedding)模型：** 评估不同的嵌入模型。在特定领域数据上微调 (fine-tuning)嵌入模型可以显著提升相关性。将更新和部署新嵌入模型的过程纳入运行。
  - **索引：** 调整向量 (vector)数据库索引参数 (parameter)（例如，HNSW索引中的`ef_construction`、`ef_search`）以平衡速度和准确性。
  - **混合搜索：** 考虑将向量搜索与传统关键词搜索（BM25）结合使用，以增强可靠性。
- **故障处理：** 当检索器找不到相关文档或遇到错误时，实施回退策略。

#### 2. 数据摄取与知识库更新

知识来源的新鲜度和准确性非常重要。

- **自动化管道：** 构建自动化、可审计的数据处理和索引管道。使用工作流协调工具（例如Airflow、Kubeflow Pipelines、Prefect）来管理依赖、重试和监控。

  > 用于更新RAG知识库的自动化管道，包含清洗、分块、嵌入和索引步骤，之后更新向量数据库。
- **更新策略：** 在增量更新（添加/更新特定向量）和定期完全重新索引之间做出选择。增量更新对于小改动更快，但时间久了可能导致索引碎片化。完全重新索引能确保一致性，但需要更多资源和停机时间规划（例如，使用蓝绿索引部署）。
- **版本控制：** 对数据源、分块逻辑和嵌入模型进行版本控制，以确保可重现性并便于回滚。

#### 3. 端到端系统监控与评估

全面监控RAG系统，而不仅仅是其单独组件。

- **分布式追踪：** 在请求生命周期（API网关 -> 协调器 -> 检索器 -> 向量数据库 -> 生成器 -> 响应）中实施追踪。这有助于确定瓶颈和错误。OpenTelemetry等工具在此方面很有用。
- **RAG专有评估：** 定期使用为RAG设计的指标评估系统：
  - **上下文相关性：** 检索到的上下文与查询的相关程度如何？（通常需要人工评估或基于模型的评估）。
  - **忠实度：** 生成的答案是否准确反映了检索到的上下文中的信息，避免了基于上下文的“幻觉 (hallucination)”？
  - **答案相关性：** 最终答案在考虑检索到的上下文的情况下，对用户查询的解决程度如何？
- **反馈循环：** 收集明确的（例如，点赞/点踩）和隐含的（例如，用户点击来源）反馈，以确定需要改进的地方。使用此反馈来优化提示、微调生成器或嵌入模型，或调整检索策略。
- **A/B测试：** 在运行上支持不同RAG配置的A/B测试（例如，比较两种不同的嵌入模型、分块策略或生成器提示）。

#### 4. 将RAG运行整合到LLMOps中

RAG管理不应孤立存在；它必须与您更广泛的MLOps工具和流程整合。

- **RAG的CI/CD：** 将RAG组件纳入您的CI/CD管道。自动化部署更新的嵌入模型、检索器逻辑更改或新的生成器提示。作为部署过程的一部分运行自动化评估套件。
- **工件管理：** 使用MLOps平台（MLflow、Vertex AI、SageMaker）追踪RAG专有工件：嵌入模型、向量索引（或关于它们的元数据）、评估数据集、提示和实验结果。
- **警报：** 根据重要的RAG性能指标设置警报：检索指标下降、端到端延迟增加、向量数据库错误激增或评估分数下降。

在生产环境中管理RAG系统是一个活跃且持续的过程。它需要扩展标准LLMOps实践来处理检索、数据管理和生成组件之间独特的配合方式。通过建立强大的监控、评估、数据管道和集成策略，您可以确保您的RAG系统保持有效、可靠且具有成本效益。下一节将更仔细地查看向量数据库的具体运行需求，它是大多数现代RAG实施的基础组件。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b49323028b1c43916dd592652dd1600-Paper.pdf) — Patrick Lewis, Ethan Perez, Aleksa Gordić, Abdelrahman Mohamed, Omer Levy, Antoine Faucon, Fabrice Massa, Yacine Tarfahi, Alfonso Ferrer, Marc'Aurelio Ranzato, Nicolas Schunck, Koustuv Sinha, Michael S. G. Seltzer, Thomas Wolf (2020)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 33; Pages: 9459-9474; DOI: [10.55917/b4549f4b](https://doi.org/10.55917/b4549f4b)
  介绍了原始的RAG架构，展示了信息检索与生成式语言模型结合如何提升事实一致性和性能。
- [Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs](https://ieeexplore.ieee.org/document/8575003) — Yury Malkov and Dmitry Yashunin (2018)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: 43; Pages: 3589-3599; DOI: [10.1109/TPAMI.2018.2889473](https://doi.org/10.1109/TPAMI.2018.2889473)
  介绍了分层可导航小世界（HNSW）图算法，这是一种用于高维向量空间中高效且准确的近似最近邻搜索的方法。
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) — Nils Reimers and Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Pages: 3981–3990; DOI: [10.48550/arXiv.1908.10084](https://doi.org/10.48550/arXiv.1908.10084)
  描述了Sentence-BERT，一个用于生成语义有意义的句子嵌入的常用框架，这对于RAG系统中的高效检索至关重要。

---

[上一节](01-%E6%8F%90%E7%A4%BA%E5%B7%A5%E7%A8%8B%E7%9A%84%E8%BF%90%E8%A1%8C%E5%8C%96.md) · [下一节](03-%E5%90%91%E9%87%8F%E6%95%B0%E6%8D%AE%E5%BA%93%E6%93%8D%E4%BD%9C%E4%B8%8E%E7%AE%A1%E7%90%86.md)

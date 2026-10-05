# 大规模RAG系统评估指标

来源：[原文](https://apxml.com/zh/courses/large-scale-distributed-rag/chapter-1-scalable-rag-architectures-foundations/large-scale-rag-evaluation-metrics)

[返回章节目录](README.md) · [返回课程目录](../README.md)

评估在生产规模下运行的检索增强生成（RAG）系统的性能和有效性，需要一套比标准学术基准更全面的指标策略。虽然诸如准确率（precision）、召回率（recall）和ROUGE等指标能在组件层面提供有用信息，但大规模分布式RAG系统引入了与分布式计算、数据量、速度和操作稳定性相关的复杂问题。因此，有效的评估不仅要包含生成输出的质量，还要包括整个分布式架构的性能、可靠性和成本效益。在设计和改进大规模RAG系统时，此处讨论的指标将成为您应对分布式环境中固有权衡以及衡量架构决策影响的工具。

### 全面评估：指标类别

为了全面了解大规模RAG系统的行为表现，将指标按多个维度进行分类会很有益处。每个类别都针对系统性能和质量的不同方面，从而确保一个方面的改进不会无意中降低另一个方面的表现。

#### 1. 大规模检索系统性能

检索组件是任何RAG系统的主要支柱。在大规模运行时，其性能不仅在于找到相关文档，还在于能否在可能达TB或PB级的切分数据中快速高效地完成此操作。

- **查询延迟（检索特定）**：衡量从接收查询到返回一组候选文档ID或段落所花费的时间。这应使用百分位数（例如，P50、P90、P99、P99.9）进行追踪。在分布式系统中，尾部延迟（P99及以上）尤其重要，因为单个慢速分片可能成为整个检索过程的瓶颈。
- **吞吐量 (throughput)（检索特定）**：系统每秒可处理的检索查询数量（QPS）。这需要在各种负载条件下，并根据索引大小和查询复杂性进行评估。
- **分布式搜索质量**：
  - **Precision@K、Recall@K、平均倒数排名（MRR）、归一化 (normalization)折损累计增益（NDCG）**：这些标准指标仍然是基本要素。然而，它们的计算必须考虑索引的分布式特性。例如，您是在衡量全局Top-K还是从分片级结果聚合的Top-K？结果合并策略如何影响这些分数？
  - **分片级命中率**：查询中，由*每个参与分片*或*聚合器*返回的Top-K结果中至少找到一个相关文档的查询百分比。这有助于发现性能不佳的分片或数据分布问题。
- **索引新鲜度和覆盖率**：
  - **索引更新延迟**：文档创建或修改到其可供检索之间的时间。对于处理快速变化信息的系统，这是相关性的重要指标。
  - **覆盖率**：目标语料库中已成功索引并可搜索的部分。覆盖率不足可能导致信息遗漏和RAG输出质量下降。
- **缓存命中率（检索缓存）**：如果对频繁查询或热门文档嵌入 (embedding)采用缓存策略，监测缓存命中率对于了解其有效性以及对延迟和底层索引负载的影响非常重要。

#### 2. 分布式环境中的生成质量和LLM性能

LLM根据检索到的文档生成的最终输出质量非常重要。在大规模运行时，这包括确保在可能大量的LLM推理 (inference)实例之间保持一致性、忠实性和效率。

- **忠实性/归因**：生成文本基于并由检索到的文档支持的程度。在大规模运行时，随着可能从不同来源检索到更多文档，确保忠实性变得更具挑战性。通常需要自动化指标（例如，使用NLI模型或经过问答微调 (fine-tuning)的模型来验证与来源相关的声明），并辅以人工评估。
- **生成的相关性**：最终答案在考虑检索到的上下文 (context)后，对用户查询意图的解决程度。
- **流畅性和连贯性**：评估生成文本可读性和逻辑流程的标准NLP指标。
- **上下文窗口管理**：对于上下文窗口有限的LLM，衡量系统如何有效地从大量检索到的文档中总结或选择信息以适应LLM处理能力的指标。这可能包括衡量信息损失或优先级设定的有效性。
- **副本间一致性**：如果LLM推理是分布式的，对于由不同模型副本处理的相同或非常相似的输入，响应是否一致，尤其是在不同负载下？
- **大规模幻觉 (hallucination)率**：生成输出中无法验证或事实不正确陈述的频率。这需要仔细监测，因为检索到的信息量可能会无意中增加LLM产生细微误解的可能性。

#### 3. 端到端系统性能和效率

用户体验的是整个RAG系统。因此，端到端指标对于理解整体用户体验和操作效率非常重要。

- **端到端查询延迟**：从用户提交查询到收到最终生成响应的总时间。这通常是用户角度最主要的指标。同样，追踪P50、P90、P99和P99.9。

  
  
  [交互图表：端到端查询延迟与吞吐量](https://apxml.com/zh/courses/large-scale-distributed-rag/chapter-1-scalable-rag-architectures-foundations/large-scale-rag-evaluation-metrics#plot-1bw004a)
  
  <details>
  <summary>图表数据（JSON）</summary>
  
  ```json
  {
    "layout": {
      "title": "端到端查询延迟与吞吐量",
      "xaxis": {
        "title": "吞吐量（每秒查询数）"
      },
      "yaxis": {
        "title": "P99延迟（毫秒）"
      },
      "legend": {
        "title": "系统配置"
      }
    },
    "data": [
      {
        "type": "scatter",
        "mode": "lines+markers",
        "name": "基线",
        "x": [
          50,
          100,
          150,
          200,
          250
        ],
        "y": [
          350,
          450,
          600,
          900,
          1500
        ],
        "marker": {
          "color": "#4263eb"
        }
      },
      {
        "type": "scatter",
        "mode": "lines+markers",
        "name": "优化缓存",
        "x": [
          50,
          100,
          150,
          200,
          250,
          300
        ],
        "y": [
          200,
          280,
          380,
          550,
          800,
          1200
        ],
        "marker": {
          "color": "#12b886"
        }
      },
      {
        "type": "scatter",
        "mode": "lines+markers",
        "name": "扩容LLM推理",
        "x": [
          50,
          100,
          150,
          200,
          250,
          300,
          350
        ],
        "y": [
          300,
          380,
          480,
          600,
          750,
          950,
          1300
        ],
        "marker": {
          "color": "#f76707"
        }
      }
    ]
  }
  ```
  
  </details>
  
  

  > P99延迟与系统吞吐量对比不同RAG系统配置，显示性能权衡。
- **系统吞吐量（QPS/RPS）**：在保持可接受的延迟和质量的同时，整个RAG系统每秒可以成功处理的用户查询总数。
- **每次查询成本/每百万Token成本**：管理运营费用是必要的。这需要对各组件（向量 (vector)数据库、LLM推理、检索计算、数据管道）进行详细的成本归因。
- **资源利用率**：
  - **CPU/GPU利用率**：所有组件的平均和峰值利用率。低利用率表明资源浪费，而高利用率可能导致性能下降和不稳定。
  - **内存使用量**：对于内存中向量数据库和LLM服务器尤其重要。
  - **网络带宽**：监测服务之间的数据传输，这在分布式设置中可能成为瓶颈。
- **首个Token时间（TTFT）**和**最后一个Token时间（TTLT）**：对于流式响应很重要的应用程序，TTFT衡量响应速度，而TTLT（或Token生成率）衡量整个生成过程的速度。

#### 4. 运行稳定性和可靠性

大规模系统必须持续可用。运行指标有助于量化 (quantization)这一点。

- **运行时间/可用性**：通常由服务水平目标（SLO）定义，并以百分比衡量（例如，99.9%、99.99%）。这应针对整个系统及其重要子组件进行追踪。
- **错误率**：导致错误的请求百分比，按错误类型和组件分类。错误率的突然飙升可能表明存在系统性问题。
- **平均恢复时间（MTTR）**：系统从故障中恢复的速度。这是其弹性能力的直接衡量。
- **平均故障间隔时间（MTBF）**：组件或系统在发生故障前正确运行的平均时间。
- **可扩展性和弹性指标**：
  - **扩容/缩容时间**：系统响应负载变化而添加或移除资源所需的时间。
  - **资源分配效率**：自动扩缩机制如何良好地匹配资源供应与实际需求，从而避免过度供应（成本）或供应不足（性能问题）。

#### 5. 数据管道健康状况

RAG系统知识库的新鲜度和质量取决于上游数据摄取和处理管道。

- **数据摄取速率**：单位时间内处理的新数据或更新数据的量（例如，文档数/小时，GB/小时）。
- **嵌入生成速度**：为新数据生成嵌入的速度。如果未能有效并行化，这可能成为瓶颈。
- **端到端数据新鲜度**：从数据在其源头可用到其在RAG系统响应中得到体现的总时间。这是一个综合指标，受摄取、处理、嵌入和索引阶段的影响。
- **管道错误率**：在数据获取、解析、分块、嵌入或索引过程中发生的错误。
- **数据质量指标（预索引）**：评估文档在索引*之前*质量的指标，例如文档长度、样板文本的存在或语言检测置信度。质量差的输入数据必然导致RAG性能不佳。

### 分布式环境中的指标解读

仅仅收集这些指标是不够的。对于大规模RAG系统，解读时必须考虑架构的分布式特性：

- **聚合挑战**：如果分布具有长尾，平均延迟等指标可能产生误导。始终优先使用百分位数和直方图。从多个可能独立的组件聚合错误率或质量分数需要仔细思考，才能得出有意义的整体系统状态。
- **服务间依赖**：一个微服务（例如向量 (vector)数据库）的性能下降可能会连锁反应，影响端到端延迟或生成质量。分布式追踪对于查明此类依赖和瓶颈非常有价值。
- **异构性**：不同的组件（例如，不同数据类型的检索器、多个LLM）可能具有截然不同的性能特征。指标仪表板应允许分解，以便了解各个组件的贡献。
- **基线建立**：在各种负载条件下建立全面的基线是必要的。没有清晰的基线，很难评估变更的影响或发现退步。

通过认真追踪和分析这广泛的指标集，您可以全面认识大规模RAG系统的行为，发现需要优化的方面，确保高可用性，并最终提供高质量、高成本效益的解决方案。后续章节将研究优化RAG管道各个部分的技术，而这些指标将指导您评估这些工作的成效。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) — P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, D. Kiela (2020)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 33; Pages: 9459-9474; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  本文介绍了RAG范式，为理解检索和生成组件如何相互作用提供了核心框架。它是评估RAG系统的基础。
- [A Survey of Hallucination in Large Language Models: Principles, Taxonomy, and Challenges](https://dl.acm.org/doi/10.1145/3618497) — Ziwei Ji, Nayeon Lee, Rita Singh, Eric P. Xing (2023)
  Journal: ACM Computing Surveys; Publisher: Association for Computing Machinery (ACM); Volume: 56; Pages: 1-38; DOI: [10.1145/3618497](https://doi.org/10.1145/3618497)
  一项综合性调查，对LLM中幻觉的检测和缓解方法进行了分类和讨论。这与衡量RAG系统输出中的忠实度和幻觉率直接相关。
- [Site Reliability Engineering: How Google Runs Production Systems](https://sre.google/sre-book/table-of-contents/) — Betsy Beyer, Chris Jones, Jennifer Petoff, Niall Richard Murphy (2017)
  Publisher: O'Reilly Media
  本书提供了运行大规模分布式系统的基本原则和实践，包括对延迟、吞吐量、错误率和其他对生产规模RAG系统至关重要的操作指标的详细讨论。
- [Benchmarking Large Language Models for Retrieval-Augmented Generation](https://arxiv.org/pdf/2309.01431) — Junzhang Shi, Kaiyu Huang, Shibo Hao, Ziyuan Zeng, Xiaofei Sun, Wenge Rong, Jianxin Li, Yexin Li (2023)
  Journal: arXiv preprint; Publisher: arXiv
  本文提出了一个RAG基准测试框架，涉及评估的各个方面，包括检索效率、生成质量和效率，为定义评估指标提供了直接的参考。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  本书介绍了构建和运行生产就绪型ML系统的工程原则。它讨论了成本效率、资源利用率、可扩展性和MLOps，这些对大规模RAG系统的性能和操作指标至关重要。

---

[上一节](04-%E5%88%86%E5%B8%83%E5%BC%8FRAG%E7%B3%BB%E7%BB%9F%E7%9A%84%E6%9E%B6%E6%9E%84%E6%A8%A1%E5%BC%8F.md) · [下一节](06-%E9%AB%98%E5%8F%AF%E7%94%A8%E6%80%A7%E4%B8%8E%E5%AE%B9%E9%94%99%E6%80%A7%E8%AE%BE%E8%AE%A1.md)

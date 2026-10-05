# 针对特定应用需求的调整（RAG 与 语义搜索）

来源：[原文](https://apxml.com/zh/courses/advanced-vector-search-llms/chapter-5-advanced-tuning-evaluation/tuning-for-application-needs)

[返回章节目录](README.md) · [返回课程目录](../README.md)

有效地调整向量 (vector)搜索系统需要根据其服务的具体应用进行调整。尽管基础算法和参数 (parameter)普遍适用，但*重点*会根据您构建的是通用语义搜索引擎还是检索增强生成（RAG）系统的检索组件而显著变化。理解这些差异，对于获得与用户需求和系统目标一致的最佳性能，是必不可少的。

### 目标对比：语义搜索与RAG

我们来明确各自的目标：

- **通用语义搜索：** 其主要目标通常是**信息获取和浏览**。用户通常寻找与查询概念相关的文件或信息，心中可能没有唯一的“正确”答案。他们可能会浏览多个结果。因此，获取多样化的相关项集合（在更大的`k`范围内具有更高的召回率）可能与前几个结果的准确性同等重要。用户可能为了更全面的结果而容忍稍高的延迟。
- **检索增强生成（RAG）：** 目标非常明确：获取简洁、事实性的上下文 (context)，供大型语言模型（LLM）用于生成针对直接查询的准确且相关的回答。大型语言模型的输出质量直接取决于所获取上下文的准确性。不相关或误导性的获取块会显著降低最终答案的质量。在这里，**Precision@k**（特别是对于小的 $k$，例如 $k=3, 5$）非常重要。低延迟也通常是优先考虑的事项，以确保大型语言模型提供快速响应的用户体验。召回率的重要性主要在于确保*最佳*上下文包含在检索到的少量 $k$ 文档中。

### 调整重点与参数 (parameter)设置

这些不同的目标直接影响您如何调整索引参数和评估性能：

**1. 召回率与准确性权衡（例如，`efSearch`、`nprobe`）：**

- **语义搜索调整：** 您可以将HNSW的`efSearch`或IVF的`nprobe`等参数配置在其合理范围的较高一端。这会增加索引内的搜索范围，通常可以提高召回率（总体上找到更多相关项），但代价是增加查询延迟。评估将侧重于召回率@10、召回率@20等指标，以及可能在更大结果集上的平均准确率（MAP）或归一化 (normalization)折现累积增益（NDCG）在更大结果集上的表现。
- **RAG调整：** 重点转向最大化提供给大型语言模型的前 $k$ 个结果的准确性。您可能会将`efSearch`或`nprobe`调整到较低的值，以在最小化延迟的同时提供出色的Precision@3或Precision@5。如果能保证提供给生成模型的上下文 (context)质量并加快交互速度，牺牲排名列表中靠后的一些召回率通常是可接受的权衡。



[交互图表：调整目标：RAG 与 语义搜索](https://apxml.com/zh/courses/advanced-vector-search-llms/chapter-5-advanced-tuning-evaluation/tuning-for-application-needs#plot-jpwacq)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "调整目标：RAG 与 语义搜索"
    },
    "xaxis": {
      "title": {
        "text": "查询延迟 (毫秒)"
      }
    },
    "yaxis": {
      "title": {
        "text": "Precision@3"
      }
    },
    "legend": {
      "title": {
        "text": "应用类型"
      }
    }
  },
  "data": [
    {
      "type": "scatter",
      "mode": "markers",
      "name": "RAG 系统",
      "x": [
        10,
        15,
        20,
        25,
        30
      ],
      "y": [
        0.95,
        0.92,
        0.88,
        0.85,
        0.82
      ],
      "marker": {
        "color": "#228be6",
        "size": 10
      }
    },
    {
      "type": "scatter",
      "mode": "markers",
      "name": "语义搜索",
      "x": [
        25,
        35,
        45,
        55,
        65
      ],
      "y": [
        0.86,
        0.83,
        0.8,
        0.77,
        0.75
      ],
      "marker": {
        "color": "#f59f00",
        "size": 10
      }
    }
  ]
}
```

</details>



> 插图显示了RAG（优先考虑低k值下的高准确性，通常接受较低的延迟设置）与语义搜索（可能容忍更高的延迟以获得更广泛的召回率，从而产生不同的准确性/延迟权衡）的典型目标区域。

**2. 量化 (quantization)与索引选择：**

- **语义搜索调整：** 乘积量化（PQ）等技术可以显著减少内存占用，并可能加快大型数据集上的扫描速度。如果总体召回率和信息获取搜索的关联性保持良好，细粒度准确性的一些损失可能是可接受的。
- **RAG调整：** 量化对*最靠前*结果准确性的影响必须仔细审查。来自PQ的有损压缩可能无意中将一个高度相关的文档从前 $k$ 个结果中替换掉。根据应用的敏感性和资源可用性，如果为大型语言模型提供上下文时保持最大准确性是绝对优先的事项，您可能会选择不那么激进的量化、标量量化（SQ），甚至不进行量化。

**3. 获取结果的数量（`k`）：**

- **语义搜索调整：** 系统通常配置为返回更多结果（例如，$k=10$ 到 $k=100+$），以方便用户浏览。调整和评估会考虑在此更广泛范围内的性能表现。
- **RAG调整：** $k$ 的值通常较小（例如，$k=3$ 到 $k=10$），由下游大型语言模型的上下文窗口限制和处理能力决定。调整工作会集中精力优化此特定小 $k$ 值内的关联性。

### 评估策略

您的评估方法也应反映应用需求：

- **语义搜索评估：** 使用Precision@k、Recall@k、MAP和NDCG等标准信息检索（IR）指标，在不同 $k$ 值下进行评估。离线对照真实数据集进行评估很常见。分析查询日志和进行A/B测试以衡量用户参与度（例如，点击率、会话时长）可提供有价值的在线验证。
- **RAG评估：** 尽管Precision@k和Recall@k（对于小 $k$ 值）是必要的中间指标，但最终衡量的是大型语言模型输出的质量。这需要端到端评估。专为RAG设计的框架和指标，例如上下文 (context)关联性（获取的上下文是否与查询相关？）、上下文忠实度（大型语言模型响应是否准确反映了获取的上下文？）以及答案关联性（最终答案是否正确回答了用户的查询？），在评估过程中变得非常重要。离线评估通常涉及对照真实数据评估所获取上下文本身，而在线评估则着眼于生成答案的质量。

### 混合搜索的考量

在实施结合向量 (vector)和关键词（例如BM25）获取的混合搜索时，融合机制（如倒数排名融合 - RRF）的调整也取决于应用：

- **语义搜索：** 融合可能旨在实现平衡贡献，让用户从精确关键词匹配和更广泛的语义相似性中获益。
- **RAG：** 融合策略可能会被加权，以更倾向于提供精确、事实性片段的结果，可能对特定实体或术语的关键词匹配赋予更大影响，同时仍使用向量相似性来捕捉关联性。目标仍然是为大型语言模型获取最佳上下文 (context)，无论该上下文由哪种获取方法获得。

总而言之，摆脱通用默认设置并调整您的向量搜索系统，需要清楚了解其作用。无论是为了语义搜索中所需的广泛信息获取，还是RAG中不可或缺的高准确性上下文获取，将您的参数 (parameter)选择、调整过程和评估指标与具体应用要求保持一致，对于成功来说都非常重要。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: NeurIPS; Volume: 33; Pages: 9459-9474; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍了检索增强生成（RAG）框架，提供了对RAG目标及其检索组件精度要求的理解基础。
- [RAGAs: Automated Evaluation of Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2309.15217) — Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert (2023)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2309.15217](https://doi.org/10.48550/arXiv.2309.15217)
  提出了一个用于自动化评估RAG系统的框架，这对于理解和实施RAG的端到端评估策略非常相关。
- [Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs](https://doi.org/10.1109/TPAMI.2018.2889473) — Yu. A. Malkov, D. A. Yashunin (2020)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE Computer Society; Volume: 42; Pages: 824-836; DOI: [10.1109/TPAMI.2018.2889473](https://doi.org/10.1109/TPAMI.2018.2889473)
  详细介绍了HNSW算法，这是一种流行的向量索引方法。这有助于理解`efSearch`等参数以及近似最近邻搜索中的核心召回率/精确度权衡。
- [An Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/) — Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  一本经典的教材，涵盖了信息检索的基本概念，包括Precision@k、Recall@k、MAP和NDCG等评估指标，对于语义搜索评估至关重要。

---

[上一节](06-%E5%9C%A8%E7%BA%BF%E4%B8%8E%E7%A6%BB%E7%BA%BF%E8%AF%84%E4%BC%B0%E6%96%B9%E6%B3%95.md) · [下一节](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%85%A8%E9%9D%A2%E6%80%A7%E8%83%BD%E8%AF%84%E4%BC%B0.md)

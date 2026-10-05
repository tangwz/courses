# 评估 RAG 性能指标

来源：[原文](https://apxml.com/zh/courses/python-llm-workflows/chapter-7-building-rag-systems/evaluating-rag-performance)

[返回章节目录](README.md) · [返回课程目录](../README.md)

评估检索增强生成 (RAG) 流水线的性能是了解其有效性的主要步骤。评估 RAG 系统不仅仅是检查最终答案是否合理；它还需要评估检索信息的质量以及基于该信息生成回复的准确性。与可能仅关注输出文本的标准大型语言模型 (LLM) 评估不同，RAG 评估会剖析其整个过程。

我们通常将 RAG 评估分为两个主要阶段：评估**检索器**和评估**生成器**（基于检索到的上下文 (context)），然后进行端到端评估。

### 评估检索部分

检索器的任务是从您的知识库中找到与用户查询相关的文档块（或“上下文 (context)”）。如果检索器未能获取正确信息，无论生成器多么优秀，都无法产生准确、有依据的答案。衡量检索器的有效性通常涉及使用信息检索领域的标准指标，前提是您拥有一个将查询映射到相关文档 ID 的真实数据集。

以下是一些常见的检索指标：

- **上下文精确度（Context Precision）**：这衡量检索到的文档中实际相关的比例。如果您的检索器返回 $K$ 份文档，其中 $R$ 份是相关的（根据您的真实情况），则精确度为 $R/K$。它回答了：“系统向我展示的文档中，有多少是有用的？”当您想尽量减少呈现给 LLM 的不相关结果时，高精确度是很重要的。
- **上下文召回率（Context Recall）**：这衡量数据集中*所有*相关文档中成功检索到的比例。如果您的整个数据集中某个查询共有 $T$ 份相关文档，而您的检索器在其前 $K$ 个结果中找到了 $R$ 份，则召回率为 $R/T$。它回答了：“系统是否找到了大多数可用的相关文档？”当不遗漏相关信息非常重要时，高召回率是很重要的。
- **命中率（Hit Rate）**：这是一个更简单的指标，它检查在前 $K$ 个结果中是否检索到*至少一份*相关文档。它是每个查询的二元衡量（是/否），通常对多个查询求平均。对于快速检查系统是否检索到*任何*有用的信息很有用。
- **平均倒数排名（Mean Reciprocal Rank, MRR）**：此指标评估*第一份*相关文档的排名有多高。对于单个查询，倒数排名是 $1 / rank$，其中 $rank$ 是排名最高的相关文档的位置。如果没有检索到相关文档，倒数排名为 0。MRR 是所有查询的这些倒数排名的平均值。当用户主要关注快速找到唯一最佳答案时，它尤其有用。
  
  $$
  MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}
  $$
  
  其中 $|Q|$ 是查询总数，$rank_i$ 是查询 $i$ 的第一份相关文档的排名。

创建真实情况（知道哪些文档与给定查询相关）可能耗时耗力，通常需要人工标注。

### 评估生成部分

一旦检索器提供了上下文 (context)，生成器的任务就是*仅*基于此上下文和原始查询合成一个答案。这里的评估侧重于生成文本相对于检索文档的质量和忠实性。

重要的指标包括：

- **忠实性（Faithfulness 或 Groundedness）**：这可以说是最重要的 RAG 特定指标之一。它衡量生成的答案是否与检索到的上下文在事实上一致，并避免引入上下文中不存在的信息（幻觉 (hallucination)）。衡量忠实性可能存在挑战。方法包括：
  - 使用另一个强大的 LLM（“LLM 作为评判者”）将答案与上下文进行比较并评分一致性。
  - 将答案分解为单独的陈述，并使用自然语言推理 (inference) (NLI) 模型或特定的事实核查工具对照上下文验证每个陈述。
  - 人工评估。
- **答案相关性（Answer Relevance）**：此指标评估生成的答案是否直接回应了原始用户查询。一个答案可能忠实于所提供的上下文，但如果检索到的上下文本身与查询不相关，则该答案仍然可能不相关。与忠实性一样，这通常使用 LLM 作为评判者的方法或人工评估来衡量。请注意其细微差别：忠实性检查与*上下文*的一致性，而相关性检查与*查询*的对齐 (alignment)程度。
- **答案正确性（Answer Correctness）**：衡量最终答案中的信息是否事实正确。如果检索到的上下文被假定为真相的来源，这可能与忠实性有显著重叠。然而，如果外部验证可能，正确性可以独立评估。通常需要人工判断。

虽然像 BLEU 或 ROUGE 这样的标准 NLP 指标可以衡量生成答案与参考答案（如果可用）之间的文本相似度，但它们对于 RAG 评估通常不足。它们不直接衡量对所提供上下文的忠实性或事实准确性，而这些对于可靠的 RAG 系统非常重要。

### 端到端评估

最终，您需要评估整个 RAG 系统的性能，从查询输入到最终答案输出。这种整体视图捕捉了检索器和生成器之间的相互影响。

- **问答基准（Question Answering Benchmarks）**：调整现有问答数据集（如 Natural Questions、TriviaQA），其中答案预计在提供的文档中找到，这可以作为端到端测试。您将查询通过 RAG 系统运行，并使用精确匹配（EM）或 F1 分数（衡量词语重叠度）等指标将生成的答案与真实答案进行比较。
- **人工评估（Human Evaluation）**：这仍然是一种非常有价值的方法，尤其适用于语气、帮助性以及整体用户满意度等方面。可以要求评估者根据相关性、忠实性、清晰度和正确性等标准对输出进行评分。虽然主观且成本高昂，但它提供了自动化指标可能遗漏的见解。
- **LLM 作为评判者（LLM-as-Judge）**：如前所述，使用复杂的 LLM 根据查询和上下文 (context)，依据预定义标准（如忠实性、相关性、连贯性）评估最终答案，是人工评估的一种可扩展替代方案。然而，请注意潜在的偏见以及评判者 LLM 本身的可靠性。

以下图表展示了两种不同 RAG 系统配置在多个指标上的评估分数：



[交互图表：RAG 系统评估比较](https://apxml.com/zh/courses/python-llm-workflows/chapter-7-building-rag-systems/evaluating-rag-performance#plot-jnu7qz)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "bar",
      "x": [
        0.85,
        0.75,
        0.9,
        0.8
      ],
      "y": [
        "Context Precision",
        "Context Recall",
        "Faithfulness",
        "Answer Relevance"
      ],
      "name": "系统 A",
      "orientation": "h",
      "marker": {
        "color": "#4263eb"
      }
    },
    {
      "type": "bar",
      "x": [
        0.92,
        0.68,
        0.82,
        0.88
      ],
      "y": [
        "Context Precision",
        "Context Recall",
        "Faithfulness",
        "Answer Relevance"
      ],
      "name": "系统 B",
      "orientation": "h",
      "marker": {
        "color": "#12b886"
      }
    }
  ],
  "layout": {
    "title": "RAG 系统评估比较",
    "xaxis_title": "分数 (0 到 1)",
    "yaxis_title": "指标",
    "margin": {
      "l": 120,
      "r": 40,
      "t": 60,
      "b": 40
    },
    "barmode": "group",
    "legend": {
      "traceorder": "normal"
    }
  }
}
```

</details>



> 使用多项指标比较 RAG 系统。系统 A 具有更好的召回率和忠实性，而系统 B 实现了更高的精确度和答案相关性。选择哪一个取决于应用的优先级。

### 实际考量

评估 RAG 系统是一个持续的过程。它并非系统构建后的一次性检查，而应贯穿整个开发过程。

- **框架支持**：像 LangSmith、TruLens 和 Ragas 这样的工具正在出现以简化 RAG 评估，提供了内置指标和日志记录功能。第 9 章讨论了其中一些评估框架。
- **迭代**：使用评估结果找出瓶颈。是检索器出错了？还是生成器产生了幻觉 (hallucination)？根据指标反馈迭代地改进组件。
- **明确您的需求**：最重要的指标在很大程度上取决于您的具体应用。一个优先考虑安全的聊天机器人可能会将忠实性放在首位，而一个文档摘要工具可能会平衡召回率和相关性。在开始评估之前，请明确定义您的成功标准。

通过系统地评估检索和生成两个方面以及端到端性能，您可以构建更可靠、更高效的 RAG 应用。

## 参考资料

- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/) — Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  全面介绍了信息检索核心指标，如准确率、召回率和MRR，这些指标对于评估RAG系统的检索组件至关重要。
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert (2023)
  Journal: arXiv preprint arXiv:2309.15217; DOI: [10.48550/arXiv.2309.15217](https://doi.org/10.48550/arXiv.2309.15217)
  介绍了一个专门用于评估RAG系统的框架，详细说明了用于评估忠实度、答案相关性和上下文质量的自动化指标，与本节内容直接相关。
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica (2023)
  Journal: NeurIPS 2023 Datasets and Benchmarks Track; DOI: [10.48550/arXiv.2306.05685](https://doi.org/10.48550/arXiv.2306.05685)
  研究了使用大型语言模型作为生成任务自动评估器（LLM-as-a-judge）的有效性和局限性，该方法与评估RAG输出的忠实度和答案相关性直接相关。

---

[上一节](05-%E6%9E%84%E5%BB%BA%20RAG%20%E6%B5%81%E7%A8%8B.md) · [下一节](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84RAG%E5%BA%94%E7%94%A8.md)

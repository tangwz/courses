---
course: "python-llm-workflows"
chapter: "testing-evaluating-llm-apps"
lesson: "llm-evaluation-strategies"
sourceId: 5215
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-9-testing-evaluating-llm-apps/llm-evaluation-strategies"
title: "评估策略：指标与人工反馈"
description: "了解评估大型语言模型输出质量的定量指标和定性方法。"
order: 4
plots: []
sourceHash: "c1752f1613ce1b9c06099138b83a7bdc4bad16d9c88a36b40bb84757ccb5a7b7"
sourceCorrections: []
---

评估大型语言模型 (LLM) 应用的表现，不仅仅是检查代码是否无误运行。由于大型语言模型生成类似人类的文本，它们的输出通常是概率性的，即使对于相同的输入也可能不同。这种固有的可变性意味着传统的确定性测试不足够。我们需要方法来评估生成内容的*质量*、*准确性*和*实用性*，这通常需要结合自动化计算和人工判断。

### 定量指标

自动化指标为大型语言模型表现的某些方面提供了一种可扩展的方式，以获得客观的数值分数。它们在开发过程中特别有用，用于比较不同的模型版本、提示策略或检索方法。但是，请记住，这些指标通常只是真实质量的近似值，并且存在局限性。

#### 信息检索指标（特别是针对RAG）

当您的应用需要检索信息来为大型语言模型提供依据时（例如在检索增强生成 (RAG)中），您可以使用标准的信息检索指标来评估检索步骤本身：

- **精确率：** 在检索到的文档中，有多少是相关的？
  $精确率 = \frac{检索到的相关文档数量}{检索到的文档总数}$
- **召回率：** 在所有可能的相关文档中，系统检索到了多少？
  $召回率 = \frac{检索到的相关文档数量}{相关文档总数}$
- **F1分数：** 精确率和召回率的调和平均值，提供一个平衡两者的单一分数。
  $F1 = 2 \times \frac{精确率 \times 召回率}{精确率 + 召回率}$
- **平均倒数排名（MRR）：** 衡量在多个查询中检索到的*第一个*相关文档的平均排名。当您主要关心快速找到至少一个好结果时非常有用。
- **归一化 (normalization)折现累积增益（NDCG）：** 评估检索文档排名的质量，对在列表中靠前出现的相关文档给予更高的分数。

这些指标需要真实数据，这意味着您的测试查询需要预先定义的相关文档集合。

#### 文本相似度指标

这些指标将大型语言模型生成的文本与一个或多个参考文本进行比较（通常是人工编写的优质答案示例）。

- **BLEU（双语评估替补）：** 衡量生成文本与参考文本之间N-gram（词序列）的重叠度。它会惩罚过短的输出。主要用于机器翻译。
- **ROUGE（面向召回的摘要评估替补）：** 衡量基于N-gram（ROUGE-N）、最长公共子序列（ROUGE-L）或跳跃双词组的重叠度。常用于评估摘要。
- **METEOR（显式排序翻译评估指标）：** 考虑精确单词匹配、词干匹配和同义词匹配，对生成文本和参考文本进行对齐 (alignment)。它包含对错误词序的惩罚。

虽然易于计算，这些词汇重叠指标通常无法捕捉语义。一个输出可能使用不同的词但表达相同的意思（低分），或者使用相似的词但毫无意义（高分）。

#### 语义相似度指标

为了解决词汇重叠的局限性，语义相似度指标使用文本嵌入 (embedding)（捕捉含义的数值表示）。

- **余弦相似度：** 计算生成文本和参考文本的嵌入（使用如Sentence-BERT的模型）。这些嵌入向量 (vector)之间的余弦相似度（$cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$）衡量它们在含义上的对齐程度。得分越接近1表示语义相似度越高。

这种方法更善于理解释义和同义词，但仍不能保证事实准确性或连贯性。

#### 特定任务指标

根据您的应用目标，您可以使用更直接的指标：

- **准确率：** 对于分类任务（例如情感分析、意图识别），预测的正确百分比是多少？
- **精确匹配（EM）：** 对于问答，生成的答案是否与参考答案完全匹配？这非常严格。
- **事实一致性：** 生成的文本是否与已知事实或提供的源文档中的信息相矛盾（与RAG相关）？衡量这一点通常需要更精细的技术或人工审查。

### 人工评估

尽管自动化指标有用，人工判断仍然是评估大型语言模型输出质量许多方面的黄金标准，这些方面是指标难以处理的：

- **流畅性和连贯性：** 文本阅读起来自然吗？逻辑结构清晰吗？
- **事实准确性和幻觉 (hallucination)：** 信息正确吗？模型会编造事实吗？
- **相关性和实用性：** 回应是否真正解决了用户的查询或目标？它有用吗？
- **语气和风格：** 输出是否符合所需的角色或风格指南？
- **安全性和偏见：** 输出是否包含有害、有偏见或不适当的内容？
- **指令遵循：** 大型语言模型是否遵守了提示中的特定限制或指令？

收集人工反馈的常见方法包括：

- **李克特量表：** 评估者根据数值量表（例如1-5）对“准确性”、“流畅性”或“实用性”等特定属性的回应进行评分。清晰的评分标准对于一致性至关重要。
- **成对比较：** 评估者会看到两个回应（例如，来自模型A vs 模型B，或更改前 vs 更改后），并被要求根据特定标准选择哪个更好。这通常比分配绝对分数更容易、更可靠。
- **排名：** 评估者将多个回应从最好到最差进行排名。
- **注解和错误分析：** 评估者在生成的文本中识别并分类特定错误（例如，标记 (token)事实错误、语法错误、拒绝回答）。这为改进提供了详细的定性反馈。

> 结合自动化指标以在开发过程中实现速度和规模，并结合人工评估以进行质量评估和验证的集成评估流程。

人工评估是资源密集型的（时间和成本），并且可能受到主观性的影响。制定清晰的指导方针、培训评估者以及衡量评估者间一致性（不同评估者应用指导方针的一致程度）是确保结果可靠的重要步骤。

### 结合指标与人工反馈

最有效的评估策略通常结合使用定量指标和人工反馈：

1. **使用指标进行大规模评估：** 在开发过程中使用自动化指标，以便快速获得更改反馈、对不同提示或模型进行A/B测试，并监控回归。它们帮助您快速迭代。
2. **使用人工进行深度和真实性验证：** 依靠人工评估来评估指标无法捕捉的方面、理解输出为何好或坏、识别错误、验证您的自动化指标是否与实际感知质量相关，并设定质量基准。
3. **迭代优化：** 利用人工评估的洞察来优化您的提示、微调 (fine-tuning)模型、改进检索策略，甚至有可能开发出更好、定制的自动化指标，这些指标能更准确地反映您特定应用所关注的质量。

通过周密地结合这些方法，您可以全面了解大型语言模型应用的表现，并系统地提升其质量和可靠性。后面讨论的框架可以帮助您将这些评估策略付诸实践。

## 参考资料

- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/) — Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  一本信息检索基础教材，包含精确率、召回率、F1分数、MRR和NDCG等标准评估指标。
- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/) — Kishore Papineni, Salim Roukos, Todd Ward, Wei-Jing Zhu (2002)
  Journal: Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics; Publisher: Association for Computational Linguistics; Pages: 311-318; DOI: [10.3115/1073083.1073135](https://doi.org/10.3115/1073083.1073135)
  介绍了BLEU，一种广泛用于通过与参考译文比较来评估机器翻译文本质量的指标。
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/) — Nils Reimers and Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Publisher: Association for Computational Linguistics; Pages: 3982-3992; DOI: [10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)
  展示了Sentence-BERT，一种BERT的改进，生成有语义意义的句子嵌入，适用于语义相似性等任务。
- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, et al. (2023)
  Journal: Transactions on Machine Learning Research (TMLR); DOI: [10.48550/arXiv.2211.09110](https://doi.org/10.48550/arXiv.2211.09110)
  提出了一个评估语言模型的广泛框架，涵盖多样的指标、场景和人类偏好，解决了传统指标的局限性。

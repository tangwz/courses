---
course: "mlops-for-large-models-llmops"
chapter: "llm-monitoring-observability-maintenance"
lesson: "hallucination-detection-techniques"
sourceId: 3207
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-5-llm-monitoring-observability-maintenance/hallucination-detection-techniques"
title: "幻觉检测方法"
description: "了解 LLM 输出中识别潜在幻觉的技术方法和指标。"
order: 6
plots: []
sourceHash: "a592ad50a78275be1bd99d5983edbc4c9ab630d6956b8e8d6205ee9807632679"
sourceCorrections: []
---

大型语言模型，尽管能力强大，但会生成看似合理但事实不符、与所提供上下文 (context)不相关或无意义的输出。这种常被称为“幻觉 (hallucination)”的现象，对依赖 LLM 生成内容以确保准确性和可靠性的应用程序构成重大风险。与简单的错误或语法问题不同，幻觉可能不易察觉，需要超越标准输出验证的专门检测技术。因此，在生产环境中监控幻觉是维护 LLM 质量的一个独特而主要的方面，直接影响用户信任和应用程序的实用性。

检测幻觉有挑战性，因为它们常模仿正确输出的风格和流畅度。像困惑度或 BLEU 分数这样的简单指标（通常用于序列生成任务）是不足的，因为它们无法衡量事实依据或上下文一致性。有效的幻觉检测需要仔细查看输出内容和模型的生成过程。

以下是在 LLMOps 流程中采用的几种识别潜在幻觉的技术方法：

### 不确定性量化 (quantization)

一种方法是考察模型在生成过程中的内部状态，以评估其置信度。其想法是，模型在生成推测性或捏造信息时，可能会表现出较低的置信度。常见技术包括：

- **Token 概率分析：** 检查生成 Token 的概率分布。包含许多低概率 Token 的序列，或模型对其选择的 Token 相比其他备选项赋予较低概率的情况，可能表明不确定性。计算生成序列的负对数似然或熵可以提供一个分数。对于 Token 序列 $t_1, t_2, ..., t_N$，序列的对数概率为 $\sum_{i=1}^{N} \log P(t_i | t_1, ..., t_{i-1})$。较低的分数（更负）可能表明生成时置信度较低。类似地，在特定生成步骤中，概率分布的熵较高（$S = -\sum_{j} p_j \log p_j$，其中在给定步骤中，$p_j$ 是词汇表 (vocabulary)中第 $j$ 个 Token 的概率），表明模型对下一个 Token 的确定性较低。
- **通过采样进行语义一致性检查：** 使用核采样或温度缩放 ($T > 0$) 为同一提示生成多个输出。如果生成的输出在事实主张或意义上差异很大，这表明模型缺乏稳定、有事实依据的响应。衡量这些样本之间的语义相似性或一致性可以作为置信度的替代指标。
- **蒙特卡洛 Dropout：** 对同一输入在推理 (inference)过程中多次应用 Dropout。结果输出的方差（无论是在嵌入 (embedding)层面还是在生成文本层面）可以作为不确定性衡量标准。

尽管计算效率高，但不确定性指标有局限性。模型可能高度自信但仍可能出错，特别是如果其训练数据包含偏见或广泛的错误信息。因此，不确定性通常应作为多个信号之一使用，而非一个决定性的幻觉 (hallucination)检测器。

### 外部知识验证

这种方法尝试对照可信的外部知识来源，验证 LLM 输出中提出的事实主张。这尤其适用于需要事实准确性的任务。流程通常包括：

1. **声明提取：** 识别生成文本中可验证的事实声明。这本身可能是一个复杂的自然语言处理任务，常使用专门模型或基于规则的系统。
2. **知识源查询：** 基于提取的声明，查询数据库、知识图谱（如 Wikidata）、企业数据存储或执行网络搜索。
3. **证据比较：** 比较从外部来源检索到的信息与 LLM 的声明。这可能涉及使用自然语言推理 (inference) (NLI) 模型来判断蕴含、矛盾或中立关系。
4. **评分/标记 (token)：** 分配事实分数或标记与验证信息相矛盾的输出。

> 用于检测幻觉 (hallucination)的外部知识验证的典型流程。

这种方法的有效性在很大程度上依赖于外部知识源的质量和覆盖范围，以及声明提取和比较步骤的准确性。它也带来了与查询外部 API 或数据库相关的延迟和潜在成本。这种方法是许多检索增强生成（RAG）系统的基本组成部分，其中对照检索到的上下文 (context)进行验证是非常契合的。

### 内部一致性与逻辑检查

有时，幻觉 (hallucination)表现为生成输出本身的矛盾或违反基本逻辑原则。技术包括：

- **自相矛盾检测：** 分析生成文本中相互矛盾的声明。这通常需要指代消解和关系提取技术。
- **逻辑推理 (inference)验证：** 对于涉及推理步骤的输出（例如数学计算、演绎论证），验证推理过程的有效性。
- **约束检查：** 确保输出符合域相关的预定义规则或约束（例如，生成的配置文件必须遵循特定架构）。

这些方法适用于较长的输出或预期一致性的对话上下文 (context)。

### 基于模型的幻觉 (hallucination)检测

另一种方法是训练一个独立的机器学习 (machine learning)模型，专门用于将文本片段分类为可能的幻觉。这种“幻觉检测器”模型可以是：

- 一个在事实性文本和幻觉文本示例上训练的微调 (fine-tuning)分类模型（例如，基于 BERT 或 T5）。
- 一个预测输入提示/上下文 (context)与生成输出之间蕴含关系（蕴含、矛盾、中立）的模型。矛盾或中立的输出可能会被标记 (token)。

创建必要的标注训练数据（特定于您的领域和模型的幻觉示例）通常是这种方法最大的挑战。然而，一旦训练完成，与外部验证相比，这些模型在推理 (inference)时可以相对较快。

### 将检测集成到 LLMOps 流程中

有效的幻觉 (hallucination)检测不仅仅是选择一种技术；而是将其操作性地集成：

- **采样：** 在每一次生成上运行计算成本高的检查（如外部验证）通常是不可行的。因此，实施采样策略（随机、基于不确定性分数的分层，或侧重于高风险用例）是必要的。
- **异步处理：** 异步运行检测检查可以防止它们给面向用户的响应时间增加明显的延迟。结果可以记录下来供后续分析或用于触发警报。
- **阈值设置和警报：** 定义不确定性分数、事实分数或矛盾标记 (token)的明确阈值，以触发警报或自动化操作（例如，需要人工审查）。
- **反馈循环：** 将被检测到的幻觉用作有价值的数据点。这种反馈可以指导提示调整，找出模型中需要微调 (fine-tuning)的弱点，或指出用于验证或 RAG 的外部知识源中的不足。

没有单一技术能保证完美的幻觉检测。通常，结合多种方法（例如，使用不确定性量化 (quantization)进行广泛筛选，对高不确定性或关键输出进行外部验证）提供更全面的策略。持续监控和调整这些技术是维护可靠且值得信赖的 LLM 部署的重要组成部分。

## 参考资料

- [Survey of Hallucination in Large Language Models](https://dl.acm.org/doi/10.1145/3613694) — Ziqi Ji, Nayeon Lee, Pang Wei Koh, Shangyu Tong, Raghuveer Thirukovalluru, Chauncey Wang, Faisal Radwan, Sohee Park, Sana Malik, Xiang Ren, H. Andrew Schwartz (2023)
  Journal: ACM Computing Surveys (CSUR); Publisher: Association for Computing Machinery (ACM); Volume: 56; Pages: 1-38; DOI: [10.1145/3613694](https://doi.org/10.1145/3613694)
  全面概述大型语言模型中幻觉的起因、检测和缓解策略。
- [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://proceedings.mlr.press/v48/gal16.html) — Yarin Gal and Zoubin Ghahramani (2016)
  Journal: Proceedings of the 33rd International Conference on Machine Learning (ICML); Publisher: PMLR; Volume: 48; Pages: 1050-1059; DOI: [10.48550/arXiv.1506.02142](https://doi.org/10.48550/arXiv.1506.02142)
  介绍了蒙特卡洛Dropout作为深度学习模型中量化不确定性的实用方法，与不确定性量化部分直接相关。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 33; Pages: 9459-9474; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍了检索增强生成（RAG），一种将大型语言模型与外部知识关联起来的基础方法，是外部验证的核心技术。
- [Measuring and Improving Factual Consistency Using Natural Language Inference](https://aclanthology.org/D19-1406.pdf) — Sean Welleck, Samira Paranjape, Jason Baldridge, Yejin Choi (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Publisher: Association for Computational Linguistics; Pages: 3959-3969; DOI: [10.18653/v1/D19-1406](https://doi.org/10.18653/v1/D19-1406)
  讨论了如何使用自然语言推理（NLI）来衡量和提高生成文本的事实一致性，直接适用于证据比较。

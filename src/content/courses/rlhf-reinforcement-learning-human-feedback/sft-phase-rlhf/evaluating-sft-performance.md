---
course: "rlhf-reinforcement-learning-human-feedback"
chapter: "sft-phase-rlhf"
lesson: "evaluating-sft-performance"
sourceId: 5109
sourceUrl: "https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-2-sft-phase-rlhf/evaluating-sft-performance"
title: "评估 SFT 模型表现"
description: "评估 SFT 模型质量的衡量指标与方法，在奖励建模之前进行。"
order: 4
plots: ["plots/5109-0.json"]
sourceHash: "8154d6125221e304e7fd6c67553d88b6f2220d4cdce4c45910f1584e7167d579"
sourceCorrections: []
---

在进入奖励建模之前，必须首先评估监督式微调 (fine-tuning)（SFT）模型的质量和表现。SFT 模型是通过用精心准备的提示-响应对数据集对基础语言模型进行微调而开发的。这项评估是一个主要检查点：SFT 过程是否成功使模型适应训练数据中体现的期望风格、格式和指令遵循能力？表现不佳的 SFT 模型将为后续的强化学习 (reinforcement learning)阶段提供薄弱的支撑，可能阻碍整个对齐 (alignment)过程。

几点重要问题包括：

- 模型是否准确遵循 SFT 数据集中类似提示的指令？
- 模型是否采用期望的语调、风格和输出格式？
- 微调过程是否引入了不理想的行为，例如重复性增加、内容平淡或拒绝回答合理提示？
- 与基础预训练 (pre-training)模型相比，模型在目标任务上的表现是否明显更好？

回答这些问题需要结合定量指标和定性人工评估。

### 评估数据集

就像标准机器学习 (machine learning)实践中一样，评估应在 SFT 训练期间未使用的留出数据集（验证集或测试集）上进行。理想情况下，此数据集应包含代表你所微调 (fine-tuning)的任务和风格的提示。使用训练集中的提示只会衡量记忆能力，而非泛化能力。

### 自动化指标

虽然自动化指标通常难以全面反映语言质量和对齐 (alignment)情况，但它们可以提供有用的信息，尤其是在识别训练问题方面。

#### 困惑度

困惑度是语言建模中的标准衡量指标，量化 (quantization)了概率模型预测样本的好坏程度。它的计算方式是评估数据集的平均负对数似然的指数，根据模型。数学上，对于一个词元 (token)序列 $W = w_1, w_2, ..., w_N$，困惑度（PPL）是：


$$
PPL(W) = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \log p(w_i | w_1, ..., w_{i-1}) \right)
$$


较低的困惑度分数通常表示模型在预测评估集中的词元序列时更确信和准确。在 SFT 评估中，你通常计算留出数据集中提示-响应对的*响应*部分的困惑度。

尽管训练期间验证集上的困惑度下降表明模型正在学习数据分布，但这并非质量或指令遵循的直接衡量指标。模型可能通过生成与 SFT 数据统计特征相符的重复或通用文本来达到低困惑度，但未能生成有帮助或具体的响应。然而，验证困惑度*显著*高或*持续增加*通常是表明训练问题或严重过拟合 (overfitting)的警示。

#### 序列重叠指标（ROUGE, BLEU）

像 ROUGE (Recall-Oriented Understudy for Gisting Evaluation) 和 BLEU (Bilingual Evaluation Understudy) 这样的指标衡量生成的响应与一个或多个参考响应之间的重叠。

- **ROUGE:** 常用于摘要任务，ROUGE 变体（ROUGE-1, ROUGE-2, ROUGE-L）分别衡量一元词、二元词和最长公共子序列的重叠。
- **BLEU:** 常用于机器翻译，BLEU 衡量 n-gram 精度，对过短的生成进行惩罚。

如果你的 SFT 任务涉及生成应与特定参考密切匹配的文本（例如，封闭式问答、特定格式任务），这些指标会很有用。然而，对于可能存在多个有效且不同的响应的开放式生成任务，ROUGE 和 BLEU 分数可能产生误导。高重叠度不一定意味着高质量，低重叠度也不一定意味着低质量。谨慎使用它们，主要作为基本检查或在预期有强烈参考重叠的任务中使用。

#### 损失曲线分析

监测训练和验证损失曲线是基本操作。理想情况下，两种损失都应下降并趋于稳定。训练损失和验证损失之间存在大差距表明过拟合，即模型记住了训练数据但泛化能力不佳。验证损失中的尖峰或不稳定性可能指向超参数 (parameter) (hyperparameter)问题，例如学习率或批量大小。



![SFT 损失曲线](plots/5109-0.json)



> 典型的 SFT 损失曲线显示训练和验证损失都在下降，且差距较小，表明泛化能力良好。

### 基于模型的评估

一种日益常用的技术是使用另一个强大的大型语言模型（例如 GPT-4, Claude）作为自动化评估器。过程通常如下：

1. 将评估集中的提示提供给 SFT 模型以生成响应。
2. 将相同的提示、SFT 模型的响应以及可能的参考响应提供给评估器大型语言模型。
3. 指示评估器大型语言模型根据特定标准评估 SFT 响应（例如，“请从 1-5 分评价此响应的有用性，” “响应是否遵循了提示中的所有指令？”，“格式是否正确？”）。

这种方法比人工评估提供更好的可扩展性，并且能够评估超出简单文本重叠的方面。然而，它依赖评估模型的自身能力和潜在偏见，结果有时可能不一致，或需要对评估器本身进行细致的提示工程 (prompt engineering)。

### 人工评估

尽管存在成本和可扩展性方面的挑战，直接人工评估仍是评估 SFT 所针对的大型语言模型行为方面的衡量标准。

#### 定性评估

这涉及人工评审员检查模型根据评估提示生成的一些输出样本。评审员会关注：

- **指令遵循：** 模型是否遵循提示中的具体限制和要求？
- **风格和语调：** 输出是否符合 SFT 数据中体现的期望风格（例如，正式、对话式、代码式）？
- **事实准确性：** （如果适用）模型的说法是否正确？
- **有用性和相关性：** 响应是否有用并直接回应提示？
- **安全性和无害性：** 模型是否避免生成有害、带有偏见或不当的内容？
- **失败模式：** 是否存在重复、幻觉 (hallucination)、语无伦次或过于通用的响应等迹象？

评审员通常提供书面反馈以及好的和坏的输出示例。

#### 定量评估

人工评估也可以结构化以得出定量分数：

- **李克特量表：** 评分者根据预定义的量表（例如，1-5）对响应在有用性、清晰度、格式遵循等维度上进行评分。
- **成对比较：** 评分者会看到一个提示和两个响应（例如，一个来自 SFT 模型，一个来自基础模型，或两个来自 SFT 模型的变体），并被要求选择哪个更好，可能还需要解释原因。这构成了奖励建模中后续使用的偏好数据的依据，但也可以在此处用于将 SFT 模型与其基线进行比较。

清晰的标注指南和评分者培训对于确保人工评估结果的一致性和可靠性很重要。

> 使用留出评估数据集评估 SFT 模型的不同方法概述。

### 结果解读

SFT 评估的目标不一定是通用基准测试上达到先进水平，而是确认是否为下一个 RLHF 阶段做好准备。关注以下几点：

- **明显改进：** SFT 模型应明确表现出比基础预训练 (pre-training)模型更好的指令遵循和目标风格采用能力，特别是经人工评估确认后。
- **无灾难性遗忘：** 模型应保留其通用语言能力，并且在流畅性或连贯性方面不出现严重退步。
- **可接受的失败模式：** 尽管不太可能达到完美表现，但任何失败模式（如偶尔重复或轻微格式错误）都应被理解并被视为强化学习 (reinforcement learning)调整的起点。严重问题可能需要重新审视 SFT 数据集的准备或训练过程。

最终，评估 SFT 模型是为了建立信心。一个表现良好的 SFT 模型，通过自动化指标和（理想情况下）人工评审的结合确认后，能为 PPO 微调 (fine-tuning)阶段提供更强的起始策略。这使得强化学习过程能够更有效地专注于优化奖励模型捕捉到的人类偏好，而不是从头开始艰难学习基本的指令遵循或风格一致性。

## 参考资料

- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University
  这是一本标准NLP教材，涵盖了自然语言处理中的概念，包括语言模型、困惑度计算以及使用BLEU和ROUGE等指标评估生成式语言模型的方法。这是第四版草稿。
- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/) — Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu (2002)
  Journal: Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics; Publisher: Association for Computational Linguistics; Pages: 311-318; DOI: [10.3115/1073083.1073135](https://doi.org/10.3115/1073083.1073135)
  这篇有影响力的论文介绍了BLEU，一种广泛使用的自动指标，通过与参考译文比较来评估机器生成文本的质量。它与需要高精度匹配参考文本的任务相关。
- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — Chin-Yew Lin (2004)
  Journal: Text Summarization Branches Out; Publisher: Association for Computational Linguistics; Pages: 74-81; DOI: [10.3115/1621300.1621317](https://doi.org/10.3115/1621300.1621317)
  这篇论文介绍了ROUGE，一套用于评估自动摘要及其他文本生成任务的指标。它通过N-gram和最长公共子序列测量与参考摘要的重叠程度。
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica (2023)
  Journal: NeurIPS 2023 Datasets and Benchmarks Track; DOI: [10.48550/arXiv.2306.05685](https://doi.org/10.48550/arXiv.2306.05685)
  这项研究探讨了使用大型语言模型作为评估器（“LLM即评委”）评估其他大型语言模型的可靠性。它为这种模型评估方法引入了基准和方案。

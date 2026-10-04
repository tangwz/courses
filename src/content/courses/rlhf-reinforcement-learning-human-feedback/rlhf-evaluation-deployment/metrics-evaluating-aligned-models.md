---
course: "rlhf-reinforcement-learning-human-feedback"
chapter: "rlhf-evaluation-deployment"
lesson: "metrics-evaluating-aligned-models"
sourceId: 5200
sourceUrl: "https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-7-rlhf-evaluation-deployment/metrics-evaluating-aligned-models"
title: "对齐模型的评估指标"
description: "超越标准NLP指标：衡量有益性、无害性、诚实性。"
order: 1
plots: []
sourceHash: "2f03e031fb90fca1486403264e7e9bb3c4a709c5dde530fc13792d8cd94b649b"
sourceCorrections: []
---

评估经过RLHF微调 (fine-tuning)的模型，需要关注传统NLP指标以外的方面。虽然困惑度（perplexity）、BLEU或ROUGE等指标可以衡量流畅性或语义相似度，但在衡量与有益性、诚实性和无害性相关的复杂人类偏好对齐 (alignment)方面存在不足。评估一个经过RLHF微调的模型，需要一套为这些对齐目标定制的专门指标。

正如章节引言中所述，实现对齐意味着确保模型的行为符合人类意图和价值观。标准指标无法衡量一个回应，即使它流畅且相关，是否实际有益、真实或安全。因此，我们需要专门设计用来检验这些方面的指标。

### 标准NLP指标的局限性

考虑一个旨在提供帮助而微调 (fine-tuning)的模型。一个回应可能通过包含相似关键词而在与参考答案对比时获得高ROUGE分数，但却未能提供可操作的建议或误解了用户的根本需求。同样，困惑度衡量语言模型概率分布下的流畅性，但不评估事实准确性或安全性。当目标是对齐 (alignment)时，仅仅依赖这些指标可能会产生误导。

### 衡量“HHH”：有益性、诚实性和无害性

一个广泛采用的LLM对齐 (alignment)评估框架以三个支柱为核心：有益性、诚实性和无害性 (HHH)。指标通常被开发出来以针对这些方面中的一个或多个。

#### 有益性指标

有益性是指模型在帮助用户达成目标、准确完整地回答问题以及提供有用信息方面的表现。

- **人类偏好分数：** 尽管在人类评估协议部分将进一步讨论，但成对比较或评分量表（例如，衡量满意度的李克特量表）的汇总结果通常被提炼成量化 (quantization)分数。相对于基线模型更高的平均评分或胜率，可以作为感知有益性的直接指标。
- **任务成功率：** 对于LLM辅助完成具体任务（例如代码生成、信息检索、总结）的特定方面或基准测试，任务完成率或准确率可以作为衡量有益性的替代指标。
- **埃洛等级分：** 模型可以基于成对的人类判断，使用埃洛等级分系统进行排名，从而提供相对于其他模型的有益性相对衡量。
- **奖励模型 (RM) 分数：** 经过训练的奖励模型本身分配的分数可以作为衡量有益性的替代指标，其定义源于偏好数据。然而，由于潜在的校准问题以及奖励作弊的风险（即策略在不真正提升有益性的情况下优化RM分数），这需要谨慎解读。

#### 诚实性和真实性指标

诚实性涉及模型提供事实准确的信息、避免编造（“幻觉 (hallucination)”）、在适当时候承认不确定性，并在适用时引用来源。

- **问答基准测试准确率：** 专门的数据集，例如`TruthfulQA`，旨在衡量模型是否避免生成模仿常见人类误解的错误陈述。性能通常通过真实回答问题的百分比来衡量，通常区分生成真实陈述和在多项选择环境中正确识别真实陈述。
- **校准指标：** 评估模型的置信度分数（例如分配给令牌的概率）与其实际准确性的一致程度。一个校准良好的诚实模型在可能出错时应表达较低的置信度。预期校准误差（ECE）等指标可以为此目的进行调整。
- **幻觉率：** 通过人工审查或自动化检查（例如将生成的摘要与源文档进行比较）来衡量，这量化了模型输出中事实不准确或编造信息的频率。

#### 无害性指标

无害性侧重于模型避免生成有害、有偏见、歧视性、不安全或不道德内容的倾向。

- **毒性检测分数：** 使用外部分类器（如Perspective API或在`ToxiGen`等数据集上训练的专门模型）来评估模型输出的毒性水平。指标通常包括平均毒性分数或超过特定阈值被标记 (token)为有毒的输出百分比。
- **安全基准测试表现：** 评估模型对旨在引出有害内容的提示（例如RealToxicityPrompts的子集、自定义红队提示）的回应。重要指标包括：
  - **拒绝率：** 模型正确拒绝提供实质性答案的不当或有害提示的百分比。
  - **有害内容生成率：** 模型生成有害内容的提示百分比，通常按类型分类（例如仇恨言论、歧视、非法建议）。
- **偏见指标：** 使用像`QA偏见基准 (BBQ)`这样的基准，或分析不同人口群体之间的性能差异（例如使用`Winogender模式`），以量化模型输出中反映的社会偏见。

> 此图示意了HHH对齐支柱与用于评估它们的指标类型之间的关系。

### 使用奖励模型分数和KL散度

尽管并非真实对齐 (alignment)的直接衡量，但在RLHF过程中生成的另外两个量也提供有用信息：

1. **奖励模型分数：** 如前所述，评估期间奖励模型（RM）分配给策略模型输出的平均分数，提供了关于策略如何优化所学偏好目标的视角。高RM分数表明策略的行为方式符合RM预测的人类偏好。在训练期间追踪这个分数非常重要，但由于策略利用RM漏洞（奖励作弊）的风险，应与其他指标一同解读。
2. **KL散度：** RLHF微调 (fine-tuning)策略 ($\pi_{RL}$) 与初始SFT策略 ($\pi_{SFT}$) 之间的库尔巴克-莱布勒（KL）散度衡量了策略在RL微调期间的变化程度。它通常在PPO训练中用作惩罚项 ($KL(\pi_{RL} || \pi_{SFT})$)，以阻止策略与SFT模型偏离过大，从而保持语言质量和通用能力。虽然主要是一个约束，但*最终*的KL散度值可以作为指标报告。极高的KL可能表明行为发生显著变化（可能有利于对齐，但对能力有风险），而极低的KL可能表明RL调优影响很小。解读KL散度需要了解训练设置和其他评估结果。

### 聚合评估

通常，单一指标不足以进行评估。评估对齐 (alignment)通常涉及一套涵盖HHH方面的指标，以及传统的能力基准测试。`语言模型整体评估 (HELM)` 框架或 `EleutherAI语言模型评估工具` 等工具提供了运行跨越许多指标和数据集的标准化评估的基础设施，从而提供模型性能和对齐的更完整视图。这些自动化工具将在后续章节中进一步讨论。

选择合适的指标集取决于您的模型的具体对齐目标，以及其部署场景相关的潜在风险。通常，自动化指标和人工评估的组合对于全面评估来说是必要的。

## 参考资料

- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) — Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2204.05862; DOI: [10.48550/arXiv.2204.05862](https://doi.org/10.48550/arXiv.2204.05862)
  一篇基础性论文，介绍了用于评估通过RLHF训练的大型语言模型的HHH框架，并详细阐述了相关的人工反馈和模型评估方法。

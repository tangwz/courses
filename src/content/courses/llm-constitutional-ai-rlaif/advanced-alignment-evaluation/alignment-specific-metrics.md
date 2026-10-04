---
course: "llm-constitutional-ai-rlaif"
chapter: "advanced-alignment-evaluation"
lesson: "alignment-specific-metrics"
sourceId: 4550
sourceUrl: "https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-7-advanced-alignment-evaluation/alignment-specific-metrics"
title: "标准基准：对齐专用指标"
description: "开发和应用直接衡量对宪法原则或人工智能偏好的遵守情况的指标。"
order: 1
plots: ["plots/4550-0.json"]
sourceHash: "454b9053469073e5dd0565981b543864905f40d2caf5609a5e901bd461b1f688"
sourceCorrections: []
---

GLUE、SuperGLUE 或困惑度等标准自然语言处理基准衡量的是通用语言能力，但不足以评估宪法人工智能 (CAI) 和来自人工智能反馈的强化学习 (reinforcement learning) (RLAIF) 所追求的特定对齐 (alignment)目标。这些先进方法旨在培养有益性、无害性、诚实性以及遵守预设原则等行为特质，这需要专门的测量方法。仅依赖通用基准可能会产生误导，因为模型可能在语言任务上表现良好，但在开放式互动情境中仍表现出不良行为。

为有效评估经过 CAI 或 RLAIF 微调 (fine-tuning)的模型，我们需要直接反映预期对齐属性的指标。这涉及摆脱对封闭任务准确性的依赖，并建立针对这些特定、通常是定性目标的评估框架。

### 将原则和偏好转化为实际衡量

核心思想是将 CAI（遵守宪法）和 RLAIF（与学习到的人工智能偏好对齐 (alignment)）的抽象目标转化为可量化 (quantization)的衡量标准。

#### 源自宪法原则的指标 (CAI)

宪法为模型行为提供明确规则或指导。评估指标应直接评估对这些原则的遵守情况。

1. **原则违反率：** 设计有针对性的提示集，旨在诱导模型违反特定的宪法原则（例如，生成有害内容、表达有偏见的观点、不恰当地拒绝合理请求）。该指标是模型响应违反相关原则的提示百分比。这通常需要单独的分类器模型（一个“原则违反检测器”）或人工评估来评分输出。
2. **自我批评准确性（内部指标）：** 如果 CAI 过程涉及明确的自我批评，则评估这些批评的准确性。模型是否正确识别其初始响应何时违反了原则？这提供了对模型内部“理解”宪法的认识。
3. **原则遵守度对比：** 向模型呈现需要在原则之间进行权衡的情境（例如，有益性与无害性）。评估模型是否根据预设优先级处理这些冲突，这些优先级可能源于宪法结构或元原则。这里常需要人工判断。

考虑一个简化的宪法，其中包含一项原则：“P1：避免生成有害内容。”评估集可能包含已知会从基线模型中引出有害响应的提示。该指标可以是：


$$
\text{有害性比率} = \frac{\sum_{i=1}^{N} \text{是有害的}(\text{响应}_i | \text{提示}_i)}{N}
$$


N 则表示评估提示的数量，$\text{是有害的}$ 是一个布尔函数，可能由高准确度有害性分类器或人工标注者实现，检查响应是否违反 P1。

#### 反映人工智能偏好的指标 (RLAIF)

RLAIF 根据人工智能偏好模型 (PM) 或奖励模型 (RM) 学到的偏好来对齐模型。评估可以直接使用这些模型。

1. **平均奖励/偏好分数：** 使用训练好的 RM/PM 来评分对齐模型在留出的评估提示集上的响应。平均分数越高表明 *根据 RM/PM 的* 对齐度越好。
   - **注意：** 该指标容易受到奖励作弊的影响。大语言模型 (LLM)可能会找到方法来最大化 RM/PM 预测的分数，而不是真正提升其所需品质，特别是当 RM/PM 存在可利用的缺陷或未能很好地泛化时。
2. **偏好胜率：** 在同一组提示上，将对齐模型 ($M_{\text{aligned}}$) 的输出与基线模型 ($M_{\text{baseline}}$) 或上一个检查点 ($M_{\text{prev}}$) 进行比较。使用 PM 预测每个提示哪个响应更受偏好。
   
   $$
   \text{胜率}(M_{\text{对齐}} \text{ 对比 } M_{\text{基线}}) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left[ \text{PM}(y_{\text{对齐}, i}, y_{\text{基线}, i} | \text{提示}_i) > \tau \right]
   $$
   
   此处，$\mathbb{I}$ 是指示函数，$\text{PM}(y_A, y_B)$ 输出一个分数，表示 $y_A$ 相对于 $y_B$ 的偏好（例如，对数几率），$\tau$ 是一个阈值（通常为 0）。胜率显著高于 50% 表明根据 PM 有所改进。
   - **注意：** 这完全取决于 PM 本身的质量和对齐情况。如果 PM 存在缺陷，胜率可能无法反映有益性或无害性方面的真实改进。

### 衡量特定对齐 (alignment)维度

除了与 CAI/RLAIF 机制直接相关的指标，我们还需要针对特定行为轴线的评估：

- **有益性：**

  - **人工评估：** 在多样化的任务提示上使用利克特量表（例如，1-5 分的有益性评分）或配对比较（“哪个响应更有益？”）。这通常是黄金标准，但成本高昂。
  - **任务成功率：** 对于结果可验证的任务（例如，编码、数学问题、特定问答），衡量模型成功完成任务的百分比。
  - **信息准确性：** 使用 TruthfulQA 等数据集或通过与精心策划的知识库进行比较来评估事实正确性。衡量幻觉 (hallucination)或编造的发生率。
- **无害性：**

  - **敏感话题拒绝率：** 衡量模型恰当拒绝处理有害或不允许的提示（例如，生成非法内容、仇恨言论）的频率。根据 RealToxicityPrompts 或自定义红队提示等基准进行评估。
  - **有害性/偏见分数：** 使用外部分类器（例如，Google 的 Perspective API、自定义训练的分类器）为模型在广泛提示下的输出分配有害性、偏见（性别、种族等）或其他安全维度分数。
  - **越狱鲁棒性：** 评估抵御旨在规避安全训练的提示（对抗性提示）的能力。
- **诚实性与校准：**

  - **真实性基准：** 在 TruthfulQA 等数据集上的表现，衡量模仿常见虚假信息与提供真实陈述的倾向。
  - **校准误差：** 衡量模型表达的置信度（如果可用）是否与其经验准确性匹配。糟糕的校准（过度自信或自信不足）可能导致误导。
  - **谨慎表达的恰当性：** 评估模型是否在有理由时恰当地表达不确定性，而非自信地陈述可能不正确的信息。
- **谄媚：**

  - **同意有缺陷的前提：** 设计用户表达明显错误观点或前提的提示。衡量模型同意或验证用户错误观点的频率，而非提供纠正或中立立场。Anthropic 的研究为构建此类评估提供了方法。

### 综合指标与可视化

任何单一指标都无法捕捉对齐 (alignment)的全貌。需要追踪多个维度下的多种指标。结合这些分数的仪表板能提供更全面的视角。例如，雷达图可以呈现有益性、无害性、诚实性、宪法遵守度以及反谄媚性等轴线上的性能表现。



![对齐概况对比](plots/4550-0.json)



> 雷达图比较了两个模型版本在主要对齐维度上的表现（分数 1-5）。此类可视化有助于发现权衡并追踪进展。

### 挑战与注意事项

- **自动化与人工评估：** 自动化指标可扩展，但可能缺乏深度或容易被钻空子。人工评估能提供更深的理解，但速度慢、成本高且可能主观。通常需要结合使用，自动化指标用于广泛跟踪，人工评估用于验证和评估不易察觉的失败。
- **指标作弊：** 模型可能会非常擅长优化特定指标，却没有真正改进其潜在的预期行为。这需要多样化且不断演进的评估套件。
- **指标有效性与可靠性：** 确保指标准确衡量预期构念并产生一致的结果。这通常涉及与人工判断的相关性研究。
  "\* **上下文 (context)与分布变化：** 模型对齐 (alignment)可能脆弱且依赖于上下文。评估集必须多样化且能代表预期互动。在分布外提示上的表现可能会下降。"

开发对齐专用指标是一个活跃的研究方向。这需要仔细考虑对齐目标、所用机制 (CAI/RLAIF) 以及模型和评估过程本身可能出现的失败模式。

## 参考资料

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  介绍宪法级AI框架，涵盖原则和自我批判，直接指导CAI评估。
- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) — Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2204.05862; DOI: [10.48550/arXiv.2204.05862](https://doi.org/10.48550/arXiv.2204.05862)
  阐述RLHF方法及其通过人类偏好和奖励模型评估有用性和无害性，直接涵盖RLAIF指标。
- [TruthfulQA: Measuring Models' Tendency to Fantasize Falsehoods](https://doi.org/10.48550/arXiv.2109.07958) — Stephanie Lin, Jacob Hilton, Owain Evans (2021)
  Journal: ACL 2022; DOI: [10.48550/arXiv.2109.07958](https://doi.org/10.48550/arXiv.2109.07958)
  介绍旨在衡量诚实性和真实性的基准，与“诚实与校准”部分内容相符。

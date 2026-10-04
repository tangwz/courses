---
course: "llm-compression-acceleration"
chapter: "knowledge-distillation-large-models"
lesson: "challenges-distilling-generative-models"
sourceId: 4371
sourceUrl: "https://apxml.com/zh/courses/llm-compression-acceleration/chapter-4-knowledge-distillation-large-models/challenges-distilling-generative-models"
title: "生成模型蒸馏的难题"
description: "讨论蒸馏生成能力（如序列生成和连贯性）时的具体困难。"
order: 6
plots: ["plots/4371-0.json"]
sourceHash: "adb9559c4ba24b401ca1f614b86dbdd5b604812d17d9502a88a998ec45ddabd9"
sourceCorrections: []
---

知识蒸馏 (knowledge distillation)是模型压缩的一种有效方法。但是，将其直接应用于生成模型，特别是大型自回归 (autoregressive)语言模型时，会带来一系列独特的复杂挑战。这些挑战在分类或自然语言理解（NLU）任务中通常不会遇到。生成的序列化、开放性特点从根本上改变了知识传递的运作方式。本文将分析这些主要的障碍。

### 暴露偏差与错误传播

传统的知识蒸馏 (knowledge distillation)通常涉及使用教师强制方法训练学生模型。这意味着在给定前面的真实序列 $y_{<t}$ 的条件下，预测下一个token $y_t$，或者在某些知识蒸馏变体中，基于教师模型之前的输出进行预测。损失函数 (loss function)通常会最小化学生模型和教师模型在给定此理想上下文 (context)时，对*下一个*token预测分布之间的散度：


$$
L_{token-KD} = \sum_{t=1}^{T} D_{KL}(p_{teacher}(y_t | y_{<t}) || p_{student}(y_t | y_{<t}))
$$


然而，在推理 (inference)过程中，学生模型是自回归 (autoregressive)运行的：它在步骤 $t$ 的输入是其*自身*在步骤 $t-1$ 生成的输出，表示为 $\hat{y}_{t-1}$。这导致了训练条件（可访问真实数据或教师上下文）与推理条件（仅可访问可能不完美的自生成上下文）之间的不匹配。这种现象被称为**暴露偏差**。

结果是**错误传播**。如果学生模型在步骤 $k$ 生成了一个次优的token $\hat{y}_k$，这个错误会影响步骤 $k+1$ 的预测，可能导致进一步的偏离。初始的小错误在生成过程中可能会累积，导致学生模型的输出序列与其训练时基于token级别预测准确性所预期的质量或连贯性大幅偏离。缓解这种情况通常需要考虑序列级别的蒸馏目标（例如，优化序列似然或使用基于强化学习 (reinforcement learning)的奖励），这会带来其自身的优化复杂性。

> 错误传播的图示。训练时，学生模型预测Token 3的损失可能基于参考Token 2。然而，在推理时，学生模型会基于其自身之前生成的Token 2来预测Token 3，这可能包含错误，从而导致偏离。

### 掌握长期依赖关系与全局连贯性

生成模型因其能够在长序列中生成连贯、上下文 (context)相关的文本而受到重视。这需要捕捉复杂的长期依赖关系，保持一致的风格或角色，并遵守从训练数据中隐式学到的事实约束。

简单的token级别知识蒸馏 (knowledge distillation)，若仅仅专注于匹配下一个token的预测概率，通常难以有效传递这些全局属性。学生模型可能擅长在给定前面上下文的情况下预测紧邻的下一个token，但却无法在跨段落或文档时保持连贯性或一致性。教师模型内部状态和注意力模式中包含的细致、高层次知识（正是这些知识实现了连贯生成），可能无法通过仅仅最小化输出层的KL散度而充分获得。涉及中间表示匹配或注意力图迁移的技术旨在解决这个问题，但对齐 (alignment)潜在不同架构（教师模型与学生模型）之间的表示仍然是一项不简单的任务。

### 模式崩溃与多样性丧失

大型教师语言模型通常与温度缩放、Top-K或核采样等采样策略结合使用，可以生成多样化和富有创造性的输出。它们隐式地对可能序列的复杂分布进行建模。

知识蒸馏 (knowledge distillation)中的一个常见问题是，优化学生模型以匹配教师模型的平均预测（软标签）可能会无意中抑制这种多样性。学生模型可能会学会偏好高概率的“安全”token，导致重复或通用的输出。这类似于**模式崩溃**，即学生模型仅学习捕获教师模型输出分布的主导模式，从而失去原始的广泛性。保留生成多样性需要更高级的知识蒸馏技术，例如在训练期间从教师模型的分布中采样，或采用旨在匹配平均预测分布特性的目标函数（例如，使用对抗训练或矩匹配），这增加了复杂性。



![图示：蒸馏对输出多样性的影响](plots/4371-0.json)



> 模式崩溃的图示。教师模型呈现更宽泛、可能多峰的输出分布（蓝色）。传统的知识蒸馏可能导致学生模型（红色）将概率质量集中在最主要的模式周围，从而减少输出多样性。

### 生成忠实度评估的困难

评估生成模型的蒸馏成功与否，比判别任务远更具挑战性。像困惑度这样的标准指标衡量模型的平均不确定性，但并不总是与人类对质量的判断有很好的相关性。N-gram重叠度指标（BLEU、ROUGE）对摘要或翻译等任务很有用，但在开放式生成中未能捕捉创造性、连贯性或事实准确性。

我们如何确定学生模型是否真正掌握了教师模型的*生成特性*？仅仅将学生模型的输出与教师模型在给定提示下的具体输出进行比较可能会产生误导，因为可能存在多种多样但有效的响应。评估生成内容的*分布*在统计上要求高且通常不切实际。缺乏直接、可靠的评估指标使得有效调整蒸馏过程并客观比较生成任务的不同知识蒸馏 (knowledge distillation)策略变得困难。人工评估通常仍然必要，但成本高且耗时。

### 架构不匹配

尽管知识蒸馏 (knowledge distillation)通常涉及将模型蒸馏到相同架构的较小版本，但教师模型和学生模型之间差异很大的架构差异（例如，从Transformer到非Transformer，或层数、隐藏层大小或注意力机制 (attention mechanism)差异很大的模型）带来了重大挑战，特别是对于生成任务。当基本组成部分不同时，传递与序列处理、注意力模式和内部状态管理相关的知识会变得复杂。对齐 (alignment)中间层进行特征匹配需要仔细的、通常是启发式的映射策略，这些策略可能无法有效传递支配生成过程的隐式知识。

总之，蒸馏生成模型需要超越简单的token级别模仿。解决暴露偏差、保留长期连贯性和多样性、发展合适的评估方法以及处理架构差异是当前研究和工程中的活跃方向。成功通常取决于采用更复杂的序列级别目标，仔细管理训练与推理 (inference)的不匹配，并可能整合中间特征匹配或强化学习 (reinforcement learning)等技术。

## 参考资料

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Geoffrey Hinton, Oriol Vinyals, Jeff Dean (2015)
  Journal: arXiv preprint arXiv:1503.02531; DOI: [10.48550/arXiv.1503.02531](https://doi.org/10.48550/arXiv.1503.02531)
  介绍知识蒸馏的奠基性论文，为软标签提供了基础，这与多样性和模式崩溃的讨论相关。
- [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://proceedings.neurips.cc/paper/2015/file/338bc86e220749970dc3079b552bbfe7-abs.html) — Samy Bengio, Oriol Vinyals, Navdeep Jaitly, Noam Shazeer (2015)
  Journal: Advances in Neural Information Processing Systems; Publisher: Neural Information Processing Systems Foundation; Volume: 28; Pages: 1171-1179
  介绍计划采样技术，通过在训练期间逐渐使用模型自身的输出来缓解序列生成模型中的曝光偏差。
- [Sequence-level training with recurrent neural networks](https://arxiv.org/abs/1511.06732) — Marc'Aurelio Ranzato, Sumit Chopra, Michael Auli, Wojciech Zaremba (2015)
  Journal: arXiv preprint arXiv:1511.06732; DOI: [10.48550/arXiv.1511.06732](https://doi.org/10.48550/arXiv.1511.06732)
  探讨使用序列级强化学习训练序列生成模型，直接解决误差传播和训练-推理不匹配问题。

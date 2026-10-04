---
course: "llm-constitutional-ai-rlaif"
chapter: "optimization-scalability-cai-rlaif"
lesson: "distillation-aligned-models"
sourceId: 4574
sourceUrl: "https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-8-optimization-scalability-cai-rlaif/distillation-aligned-models"
title: "对齐模型的模型蒸馏"
description: "将大型模型的对齐能力迁移到更小、更高效模型的技术。"
order: 5
plots: []
sourceHash: "d5544879167e0fb282f59d366e06d61b6897e938f21a5830f9e15dc93ae12fc9"
sourceCorrections: []
---

宪法式AI (CAI) 和 基于AI反馈的强化学习 (reinforcement learning) (RLAIF) 使我们能构建功能强大且对齐 (alignment)的大型语言模型 (LLMs)。然而，由此产生的模型通常计算资源消耗大，使得部署成本高昂并限制了它们的可及性。训练这些模型需要大量资源，尤其是在反馈生成和强化学习优化方面。模型蒸馏提供了一个有吸引力的策略，通过将能力（包括学到的对齐特性）从大型、强大的“教师”模型迁移到更小、更高效的“学生”模型，从而降低这些部署成本。

知识蒸馏 (knowledge distillation) (KD) 的核心思想是训练学生模型，不仅基于真实标签（如果可用），还要模仿大型教师模型的输出行为或内部计算。当应用于对齐模型时，目标扩大到重现任务表现，以保留通过CAI或RLAIF灌输的安全和道德行为。

### 对齐 (alignment)迁移的蒸馏技术

有几种蒸馏策略可以适用于迁移对齐特性：

1. **输出分布匹配 (软标签)：** 这是最常用的蒸馏技术。我们训练学生模型，不是基于硬标签（例如，教师预测的下一个单一词元 (token)），而是让它匹配教师对整个词汇表 (vocabulary)分配的概率分布。这通常通过最小化学生模型和教师模型输出分布之间的Kullback-Leibler (KL) 散度来实现。

   教师模型的概率（软目标）通常使用softmax函数中的温度缩放参数 (parameter) ($T$) 生成：

   
   $$
   p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
   $$
   

   $z_i$ 为教师模型产生的logit。更高的温度 ($T > 1$) 会软化概率分布，根据教师模型提供更多关于不同词元的相对概率的信息。学生模型使用相同的温度进行训练，蒸馏损失通常与（如果适用）硬标签上的标准交叉熵损失结合使用：

   
   $$
   L_{\text{蒸馏}} = \alpha L_{\text{KL}}(p_{\text{教师}}, p_{\text{学生}}) + (1 - \alpha) L_{\text{CE}}(y_{\text{真实}}, p_{\text{学生}})
   $$
   

   对于对齐蒸馏，关注点通常仅在于模仿对齐教师模型的输出 ($\alpha = 1$)，尤其是在根据旨在测试对齐的提示生成文本时。
2. **中间表示匹配：** 对齐可能不只存在于最终输出层。复杂的推理 (inference)或对精微宪法原则的遵守可能编码在教师模型的中间激活中。存在训练学生模型模仿这些内部表示的技术，通常通过添加损失项来最小化选定教师模型和学生模型隐藏状态之间的差异（例如，均方误差）。由于模型架构和层维度的差异，这可能具有挑战性，通常需要学习的投影层来将学生表示映射到教师模型空间。
3. **蒸馏辅助模型（偏好/奖励）：** 在RLAIF场景下，偏好模型或派生奖励模型体现了期望对齐的重要方面。将*这些*模型蒸馏成更小的对应模型可以带来显著的效率提升。例如，一个蒸馏后的奖励模型可以用于更便宜的强化学习 (reinforcement learning)训练迭代，或用于在边缘设备上高效进行强化学习。蒸馏过程将涉及训练一个更小的学生模型，使其在给定相同输入（例如，提示和响应对）的情况下，预测与大型教师模型相同的偏好分数或奖励值。
4. **对齐策略蒸馏：** 这直接关注于迁移通过CAI/RLAIF学到的行为策略。学生模型在一个提示数据集上进行训练（可能包括用于对齐训练或红队攻击的提示），并优化以生成与教师模型的对齐输出匹配的响应，通常使用上述的输出分布匹配技术。

### 对齐 (alignment)忠实度：核心难题

虽然蒸馏可以大幅降低模型大小和推理 (inference)成本，但主要关注点是保持*对齐忠实度*。教师模型学到的安全性、有用性和原则遵守有多少能成功迁移到学生模型？

- **能力差距：** 更小的模型本身能力较小。它们可能难以复制大型教师模型进行的复杂推理或权衡，尤其是在面对困难或对抗性提示时。这可能导致对齐性能下降，学生模型在教师模型能正确处理的场景中失败。
- **数据选择：** 用于蒸馏的数据选择非常重要。仅使用标准指令遵循数据可能无法迁移对齐的鲁棒性。通常需要包含具有挑战性的提示、红队攻击示例，以及专门涵盖原始对齐过程中目标宪法原则或偏好的数据。
- **评估是不可省略的：** 仅仅测量困惑度或在标准NLP基准上的表现是不够的。蒸馏后的学生模型*必须*使用与应用于原始教师模型相同的先进对齐指标、红队攻击策略和定性分析（如第7章所述）进行严格评估。这对于量化 (quantization)任何对齐损失并确保蒸馏模型满足安全要求非常重要。

> 迁移对齐的基本知识蒸馏 (knowledge distillation)设置。输入提示同时馈送给大型教师模型和更小的学生模型。学生模型通过最小化损失函数 (loss function)（如KL散度），将其输出分布与教师模型的软化输出分布进行比较来训练。

### 实施考量

蒸馏框架在Hugging Face的`transformers`或TensorFlow/PyTorch等标准机器学习 (machine learning)库中随时可用。该过程通常包括：

1. 加载预训练 (pre-training)（且对齐 (alignment)）的教师模型。
2. 初始化更小的学生模型（可能从较小架构的预训练权重 (weight)或随机初始化）。
3. 定义蒸馏数据集（精心策划的提示）。
4. 设置训练循环，计算每个批次的教师和学生模型输出。
5. 计算蒸馏损失（例如，带温度缩放的KL散度）。
6. 使用优化器更新学生模型的参数 (parameter)。

蒸馏的计算成本主要由通过教师模型的前向传播来生成软目标所驱动。虽然数量可观，但这通常远低于原始对齐训练的计算需求（尤其是RLAIF的强化学习 (reinforcement learning)阶段）。

### 与其他优化方法的交互

蒸馏经常与其他优化技术结合使用，例如量化 (quantization)（降低权重 (weight)的数值精度）和剪枝（移除冗余权重）。常见的工作流程是首先将知识蒸馏 (knowledge distillation)到更小的架构中，然后对所产生的学生模型应用量化或剪枝，以获得进一步的效率提升。然而，评估对对齐 (alignment)的累积影响很重要，因为每个优化步骤都带有降低学到的安全特性的风险。

“总之，模型蒸馏是使强大的对齐模型实用化部署的不可或缺的技术。通过将学到的对齐从大型教师模型迁移到更小的学生模型，我们可以大幅降低计算成本。然而，这个过程需要仔细执行、周全的数据选择和严格评估，以确保重要的对齐特性在最终的优化模型中得以保留。”

## 参考资料

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Geoffrey Hinton, Oriol Vinyals, Jeff Dean (2015)
  Journal: NIPS 2014 Deep Learning Workshop; DOI: [10.48550/arXiv.1503.02531](https://doi.org/10.48550/arXiv.1503.02531)
  这篇基础论文介绍了知识蒸馏，解释了如何使用软目标和温度缩放来训练一个小型学生模型以模仿大型教师模型。
- [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) — Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf (2019)
  Journal: 5th Workshop on Energy Efficient Machine Learning and Cognitive Computing - NeurIPS 2019; DOI: [10.48550/arXiv.1910.01108](https://doi.org/10.48550/arXiv.1910.01108)
  介绍了知识蒸馏在预训练Transformer语言模型中的实际应用，展示了输出分布匹配和中间层匹配等技术。

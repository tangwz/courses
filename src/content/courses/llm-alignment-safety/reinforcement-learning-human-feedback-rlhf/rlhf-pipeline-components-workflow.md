---
course: "llm-alignment-safety"
chapter: "reinforcement-learning-human-feedback-rlhf"
lesson: "rlhf-pipeline-components-workflow"
sourceId: 3119
sourceUrl: "https://apxml.com/zh/courses/llm-alignment-safety/chapter-2-reinforcement-learning-human-feedback-rlhf/rlhf-pipeline-components-workflow"
title: "RLHF流程：组成部分与工作原理"
description: "分解RLHF流程的核心组成部分和工作原理。"
order: 1
plots: []
sourceHash: "21f4c58622304f8adf995baecf20103f17029928b806d0f214f6ebc1cf6ac925"
sourceCorrections: []
---

人类反馈强化学习 (reinforcement learning)（RLHF）通过将人类判断直接融入训练过程，提供了一种结构化方法，引导大型语言模型（LLM）实现乐于助人、诚实和无害等预期行为。尽管其思路直接，但实际实施涉及一个多阶段流程，它结合了监督学习 (supervised learning)、偏好建模和强化学习的组成部分。理解这一工作原理，方能有效应用RLHF。

典型的RLHF流程如下图所示，包含三个主要环节，尽管存在一些不同：

1. **模型准备（可选但常见）：** 从一个基础预训练 (pre-training)LLM开始，并可选地使用监督方法对其进行微调 (fine-tuning)。
2. **奖励模型训练：** 建立一个模型，能够根据人类偏好为LLM输出评分。
3. **RL策略微调：** 使用奖励模型通过强化学习进一步微调LLM，以最大化偏好输出。

> RLHF流程各环节及其关联的概述。SFT常被使用，但并非必需。SFT模型在RL微调期间常作为参考策略（$\pi_{ref}$）。

下面我们更详细地说明每个环节。

### 环节1：模型准备（基础模型和可选SFT）

该过程通常从一个能力较强的预训练 (pre-training)LLM开始。这个模型从大量文本数据中习得了通用语言模式，但可能没有专门适应遵循指令或表现出如乐于助人等预期行为。

通常会执行一个中间步骤，称为监督微调 (fine-tuning)（SFT）。在这个环节中，预训练模型在一个较小、高质量的输入提示和预期输出（示例）数据集上进行微调。这个数据集的目的是展示目标行为（例如，按指令提供帮助，特定的对话风格）。

SFT环节旨在使模型适应它可能遇到的提示分布，并教会它预期回应的基本格式和风格。该环节产生的模型，我们称之为$\pi_{SFT}$，通常作为后续RLHF环节的起点。具体来说，它成为RL微调的初始策略，以及RL训练期间用于正则化 (regularization)的参考模型（$\pi_{ref}$）。SFT虽然有益，但并非RLHF的严格必需；可以从基础预训练模型直接开始奖励建模和RL微调。

### 环节2：奖励模型训练

RLHF的核心思路是学习人类偏好。这是通过训练一个单独的模型，即*奖励模型*（$r_\theta$），来预测人类会偏好哪一个LLM输出。

1. **数据收集：**

   - 选择一组提示（$x$）（通常从SFT中使用的或预期部署时会遇到的分布中选取）。
   - 当前LLM（通常是SFT模型$\pi_{SFT}$）为每个提示$x$生成两个或更多回应（$y_1, y_2, ...$）。
   - 人类标注员会看到提示$x$以及一对回应（例如，$y_i, y_j$），并被要求选择他们偏好哪个回应（或者表示平局，或者两者都不可接受）。常见做法侧重于成对比较。
   - 这一过程会产生一个偏好元组数据集：$(x, y_w, y_l)$，其中$y_w$是针对提示$x$的偏好（“胜出”）回应，而$y_l$是较不偏好（“落败”）的回应。
2. **模型架构：** 奖励模型通常与正在训练的LLM共享相同的架构（或者使用基础预训练 (pre-training)架构）。然而，其最终层被修改，以输出一个表示预测奖励（偏好分数）的单一标量值，而不是令牌概率。
3. **训练目标：** 奖励模型$r_\theta(x, y)$的训练目标是，使其为偏好回应$y_w$分配比不偏好回应$y_l$更高的分数。一种常见方法是使用源自Bradley-Terry模型的损失函数 (loss function)，该模型对$y_w$优于$y_l$的概率进行建模：
   $P(y_w \succ y_l | x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$
   其中$\sigma$是sigmoid函数。模型通过最小化数据集中人类偏好标签的负对数似然进行训练：
   $\mathcal{L}(\theta) = - \mathbb{E}_{(x, y_w, y_l) \sim D} [\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))]$
   这个损失函数促使奖励模型$r_\theta$在胜出回应和落败回应之间输出更大的分数差。

这个奖励模型的质量和校准是影响整个RLHF过程成功的要紧因素。本章后面我们将讨论奖励建模中的难点。

### 环节3：RL策略微调 (fine-tuning)

有了假定能捕捉人类偏好的已训练奖励模型$r_\theta$，最后环节使用强化学习 (reinforcement learning)来微调LLM策略$\pi_\phi$（初始为$\pi_{SFT}$），以最大化$r_\theta$分配的奖励。

1. **RL设置：**

   - **策略：** LLM本身，$\pi_\phi(y|x)$，由$\phi$参数 (parameter)化。
   - **动作空间：** LLM可以生成的令牌词汇表 (vocabulary)。
   - **观察空间：** 给定输入提示$x$后，目前已生成的令牌序列。
   - **奖励函数：** 由已训练的奖励模型$r_\theta(x, y)$提供，应用于给定提示$x$的最终生成序列$y$。
2. **优化过程：** 一种RL算法，最常见的是近端策略优化（PPO），用于更新策略参数$\phi$。该过程通常包括：

   - 从数据集中采样提示$x$（通常是SFT提示分布）。
   - 使用当前策略$\pi_\phi(y|x)$为每个提示$x$生成回应$y$。
   - 使用奖励模型计算提示-回应对$(x, y)$的奖励：$r = r_\theta(x, y)$。
   - 使用PPO算法更新策略参数$\phi$，以增加生成高奖励回应的可能性。
3. **带KL惩罚的PPO目标：** RLHF的一个要紧方面是防止策略$\pi_\phi$与原始SFT模型$\pi_{SFT}$（如果跳过SFT，则为基础预训练 (pre-training)模型）偏离过大。单纯为了已训练的奖励$r_\theta$进行过度优化，可能导致生成无意义或重复的、但恰好得分很高的文本（这种现象称为“奖励作弊”），或导致通用语言能力灾难性遗忘。为缓解此问题，会在奖励中添加一个基于当前策略$\pi_\phi$与参考策略$\pi_{ref}$（通常是$\pi_{SFT}$）之间Kullback-Leibler（KL）散度的惩罚项。RLHF中PPO优化的目标通常是：
   $\text{目标}(\phi) = \mathbb{E}_{(x,y) \sim D_{\pi_\phi}} [r_\theta(x, y) - \beta \text{KL}(\pi_\phi(y|x) || \pi_{ref}(y|x))]$
   其中，$D_{\pi_\phi}$是通过采样提示$x$并生成回应$y \sim \pi_\phi(y|x)$得到的提示-回应对的分布。系数$\beta$控制KL惩罚的强度。这一项促使策略$\pi_\phi$在最大化$r_\theta$预测奖励的同时，保持与参考模型$\pi_{ref}$行为相对接近，从而保持语言流畅性并缓解奖励作弊。

这个环节的产出是最终对齐 (alignment)的LLM，$\pi_{\phi^*}$，它经过调整，能生成根据已训练奖励模型偏好的回应，并兼顾保持连贯性以及与预训练和SFT阶段学到的初始能力保持接近的需求。

这个三阶段流程（SFT -> 奖励建模 -> RL微调）构成了许多成功RLHF实现的核心。后续章节将说明每个环节的具体细节，以及相关难点和变体，特别是奖励建模和PPO优化。

## 参考资料

- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) — Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, Jared Kaplan (2022)
  Journal: arXiv; DOI: [10.48550/arXiv.2204.05862](https://doi.org/10.48550/arXiv.2204.05862)
  详述了Anthropic公司如何使用RLHF训练大型语言模型以实现其助益性和无害性。
- [Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano (2020)
  Journal: Advances in Neural Information Processing Systems; Volume: 33; DOI: [10.48550/arXiv.2009.01325](https://doi.org/10.48550/arXiv.2009.01325)
  早期有影响力的工作，展示了RLHF在语言生成（摘要）方面的应用，为后续的大型语言模型应用奠定了基础。
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; Publisher: arXiv; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  引入近端策略优化（PPO）算法的原始研究论文，PPO是RLHF流程中的一个核心组成部分。

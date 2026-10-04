---
course: "rlhf-reinforcement-learning-human-feedback"
chapter: "rlhf-foundations-alignment"
lesson: "rl-principles-refresher"
sourceId: 5089
sourceUrl: "https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-1-rlhf-foundations-alignment/rl-principles-refresher"
title: "强化学习原理回顾"
description: "对马尔可夫决策过程、策略梯度和PPO核心内容的集中回顾，这些对RLHF很实用。"
order: 3
plots: []
sourceHash: "2a89841a2db137cbf2c8e6351521218cbfff3265821d268a491eae443ba9c4db"
sourceCorrections: []
---

单独的监督微调 (fine-tuning)常常难以完全实现大型语言模型所需的全部行为范围。正是在这种情况下，强化学习 (reinforcement learning)（RL）发挥作用，提供了一个框架，可以根据表示人类偏好的学习信号来优化大型语言模型。通常需要对强化学习（RL）有一定了解。此处提供对RLHF最相关的原理的集中回顾，特别是支撑近端策略优化（PPO）的那些原理。

### 将大型语言模型生成看作强化学习 (reinforcement learning)问题

强化学习的根本是关于智能体在一个环境中学习做一系列决策，以最大化累积奖励。我们可以使用标准强化学习形式，特别是马尔可夫决策过程（MDP），来构建大型语言模型生成文本的任务：

- **状态（$s$）**：当前的情境，通常包括初始提示和目前已生成的令牌序列。
- **动作（$a$）**：大型语言模型要生成的下一个令牌。动作空间是语言模型的词汇表 (vocabulary)。
- **策略（$\pi_\theta(a|s)$）**：大型语言模型本身，由$\theta$参数 (parameter)化。它根据当前状态（提示+之前令牌）定义了下一个可能令牌（动作）的概率分布。
- **奖励（$R(s, a)$）**：这是从人类反馈中学习到的重要组成部分。生成一个序列（响应）后，奖励信号会反映该序列与人类偏好的一致程度。这通常只在序列结束时，根据奖励模型（RM）的输出来分配，我们将在第3章详细介绍。
- **转移（$P(s'|s, a)$）**：在文本生成中，转移通常是确定性的。给定当前状态$s$（例如：“Translate 'hello' to French:”）和选定的动作$a$（例如令牌“Bonjour”），下一个状态$s'$简单地是它们的拼接（例如：“Translate 'hello' to French: Bonjour”）。

在这种强化学习设定中，目标是调整大型语言模型的参数$\theta$，以找到一个策略$\pi_\theta$，使生成序列的预期累积奖励（通常称为预期回报）最大化：


$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t) \right]
$$


这里，$\tau = (s_0, a_0, s_1, a_1, ...)$表示一个完整的轨迹（例如，一个完整的生成响应），$\pi_\theta$决定了生成该轨迹的概率，$T$是序列的长度，$\gamma$是折扣因子（在有限时间范围的文本生成中通常设为1）。

### 策略梯度与稳定性需求

策略梯度方法通过对目标函数$J(\theta)$进行梯度上升来直接优化策略参数 (parameter)$\theta$。策略梯度定理的常见形式如下：


$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \hat{A}_t \right]
$$


在此，$\hat{A}_t$是时间步$t$时*优势函数*的估计。优势$A(s, a) = Q(s, a) - V(s)$表示在状态$s$中采取动作$a$相比于当前策略下的平均动作要好多少，这由价值函数$V(s)$估算。直观来说，这会增加导致高于预期奖励的动作的概率，并降低导致低于预期奖励的动作的概率。

虽然有效，但基本策略梯度方法可能存在高方差和不稳定性问题，特别是在处理像大型语言模型中那样高维的参数空间时。由特别高或低的奖励样本驱动的单个大梯度更新，可能会剧烈改变策略，可能导致性能急剧下降。这种不稳定性促使了像PPO这样更精巧算法的出现。

### 近端策略优化 (PPO)

PPO因其相对简单、稳定和良好的实证表现，已成为RLHF流程中微调 (fine-tuning)大型语言模型的标准强化学习 (reinforcement learning)算法。它通过限制策略在每个更新步骤中可以改变的程度，解决了香草策略梯度的不稳定性问题。

PPO优化了一个*替代目标函数*，它包含一种机制来阻止大的策略更新。最常见的变体使用了*裁剪*目标：


$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{裁剪}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$


我们来分解一下：

- $\mathbb{E}_t [...]$ 表示对一批收集到的经验（时间步）取平均。
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$是当前策略$\pi_\theta$（正在优化）与旧策略$\pi_{\theta_{old}}$（用于收集数据）之间的概率比。它衡量了策略改变的程度。
- $\hat{A}_t$是时间步$t$的估计优势，通常使用广义优势估计（GAE）计算，这有助于减少方差。这要求在学习策略（“行动者”）的同时学习一个*价值函数*$V(s)$（“评论者”）。
- $\text{裁剪}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)$ 限制比率 $r_t(\theta)$ 保持在区间 $[1 - \epsilon, 1 + \epsilon]$ 内。超参数 (parameter) (hyperparameter)$\epsilon$（例如0.2）定义了裁剪范围。
- $\min(...)$ 取以下两项中的最小值：
  1. 未裁剪目标 $r_t(\theta) \hat{A}_t$。
  2. 裁剪目标 $\text{裁剪}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t$。

这种裁剪的作用是限制当策略比率 $r_t(\theta)$ 超出 $[1-\epsilon, 1+\epsilon]$ 边界时动作的影响力。如果优势$\hat{A}_t$为正（表示该动作优于平均水平），那么当策略变化过大（$r_t > 1+\epsilon$）时，目标会被从上方裁剪，从而防止更新过于激进。如果优势$\hat{A}_t$为负（表示该动作差于平均水平），那么当策略变化过大（$r_t < 1-\epsilon$）时，目标会被从下方裁剪，从而防止该动作的概率过度大幅下降。这使得新策略与旧策略保持接近，保证了更稳定的学习。

### RLHF中的KL散度约束

虽然PPO的裁剪目标本身就鼓励较小的更新，但RLHF的实际操作中通常会添加一个显式惩罚项，它基于当前策略$\pi_\theta$和参考策略$\pi_{ref}$之间的Kullback-Leibler (KL) 散度。通常，$\pi_{ref}$是初始的SFT模型。

此KL惩罚通常直接纳入PPO训练中使用的奖励信号：


$$
R_{total}(s, a) = R_{RM}(s, a) - \beta \, D_{KL}(\pi_\theta(\cdot|s) || \pi_{ref}(\cdot|s))
$$


或者，在实践中更常见的是逐令牌奖励：


$$
R_{token}(s_t, a_t) = R_{RM, t} - \beta \log \frac{\pi_\theta(a_t|s_t)}{\pi_{ref}(a_t|s_t)}
$$


这里：

- $R_{RM}$是奖励模型获得的奖励。
- $\beta$是一个控制KL惩罚项强度的超参数 (parameter) (hyperparameter)。
- KL项逐令牌地惩罚策略$\pi_\theta$过度偏离参考策略$\pi_{ref}$。

**为何此KL惩罚在RLHF中如此重要？**

1. **能力保持：** 它防止大型语言模型在优化人类偏好奖励$R_{RM}$时，偏离原始SFT模型中固有的通用语言建模能力和知识。没有它，模型可能会学习生成重复或无意义的文本，这些文本碰巧在RM上得分很高（“奖励作弊”），但实际质量很低。
2. **稳定性：** 类似于PPO中的裁剪，它作为一个正则化 (regularization)项，确保训练更新更平滑、更稳定。

我们将在第4章讨论PPO算法中此KL惩罚的实际实现和调整。

### RLHF的整合要点

本次回顾强调了RLHF中强化的主要组成部分：

1. 我们将文本生成构建为MDP问题，其中大型语言模型充当策略。
2. 目标是最大化来自人类偏好（通过奖励模型）的奖励信号。
3. 采用PPO来稳定更新大型语言模型的策略参数 (parameter)。
4. 引入了相对于初始SFT模型的KL散度惩罚，以保持语言质量并防止灾难性遗忘或奖励作弊。

理解强化学习 (reinforcement learning)的这些根本原理，特别是PPO和KL约束背后的机制及原理，对于有效实施和解决RLHF流程中的强化学习微调 (fine-tuning)阶段的问题非常重要，我们将在后续章节中详细阐述。

## 参考资料

- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  全面介绍强化学习基础知识，包括马尔可夫决策过程、策略梯度方法和价值函数。对于打下基础至关重要。
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍了近端策略优化 (PPO) 算法，详细说明了其剪裁代理目标和用于稳定策略优化的优点，这是 RLHF 的核心。
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: arXiv preprint arXiv:2203.02155; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  描述了人类反馈强化学习 (RLHF) 流水线，具体阐述了如何使用带有 KL 散度惩罚的 PPO 来使语言模型与人类偏好对齐。

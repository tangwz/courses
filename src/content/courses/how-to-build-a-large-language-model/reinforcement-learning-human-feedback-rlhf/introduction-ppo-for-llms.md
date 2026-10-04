---
course: "how-to-build-a-large-language-model"
chapter: "reinforcement-learning-human-feedback-rlhf"
lesson: "introduction-ppo-for-llms"
sourceId: 6086
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-26-reinforcement-learning-human-feedback-rlhf/introduction-ppo-for-llms"
title: "近端策略优化 (PPO) 介绍"
description: "了解在 RLHF 中常用于 RL 微调的 PPO 算法。"
order: 4
plots: []
sourceHash: "fde572b234d4450c1393dfdfb4b05e5ed1cd3ae4e1e4936f770f31775e8a6aca"
sourceCorrections: []
---

监督微调 (fine-tuning) (SFT) 模型通常需要使用人类偏好进行进一步优化。这些偏好通常通过奖励模型 (RM) 获取。然而，简单的监督更新不能直接处理来自此类奖励模型的标量奖励信号。因此，强化学习 (reinforcement learning) (RL) 算法被用于此优化过程。在各种强化学习方法中，近端策略优化 (PPO) 已成为在强化学习人类反馈 (RLHF) 流程中微调大型语言模型的一种流行且有效的方法。

那么，PPO 是什么，为何它在此处适用呢？PPO 属于强化学习中的策略梯度方法。策略梯度方法的核心思想是直接调整策略（在本例中是语言模型 $\pi_\theta$）的参数 (parameter) $\theta$，以使预期奖励最大化。我们估计预期奖励的梯度并朝该方向前进。然而，简单的策略梯度实现可能不稳定。一次更新步长如果对策略的改变过于剧烈，可能导致性能大幅下降，并可能难以恢复。这对于大型复杂模型（如 LLM）来说风险尤其高。

PPO 通过限制策略在每个更新步骤中的变化幅度来解决这个稳定性问题。它通过一个特定的目标函数来实现这一点，该函数阻止策略大幅偏离之前的策略，同时仍鼓励基于奖励信号的改进。

### PPO 目标函数

PPO 的核心是优化一个替代目标函数。最常见的变体使用裁剪目标。我们先定义一些术语：

- **策略 $\pi_\theta(a_t|s_t)$：** 我们的语言模型，由 $\theta$ 参数 (parameter)化。给定状态 $s_t$（输入提示和先前生成的令牌），它输出下一个可能的令牌 $a_t$ 的概率分布。
- **旧策略 $\pi_{\theta_{old}}(a_t|s_t)$：** 当前更新迭代之前的策略。我们使用此策略采样轨迹（令牌序列）。
- **优势函数 $\hat{A}_t$：** 估计在状态 $s_t$ 中执行动作 $a_t$ 相对于当前策略下平均动作的好处程度。它通常使用来自我们 RM 的奖励和一个学习到的价值函数 $V(s_t)$（评论员）来计算，常使用广义优势估计 (GAE) 等技术。正优势值表明所采取的动作好于预期，而负优势值则表明它更差。
- **概率比 $r_t(\theta)$：** 这衡量了在新策略 $\pi_\theta$ 和旧策略 $\pi_{\theta_{old}}$ 之间，在状态 $s_t$ 中执行动作 $a_t$ 的概率变化。
  
  $$
  r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
  $$
  

裁剪后的替代目标函数，通常表示为 $L^{CLIP}(\theta)$，公式如下：


$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) ]
$$


我们来分解一下：

1. **$\hat{\mathbb{E}}_t [...]$：** 这表示我们正在对从与环境的交互中（即生成文本序列并从 RM 获取奖励）收集的一批时间步取平均值。
2. **$r_t(\theta) \hat{A}_t$：** 这是标准的策略梯度目标。如果优势 $\hat{A}_t$ 为正，我们希望增加执行动作 $a_t$ 的概率，因此我们增加 $r_t(\theta)$。如果 $\hat{A}_t$ 为负，我们希望降低概率，因此我们降低 $r_t(\theta)$。
3. **$\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)$：** 此函数将概率比 $r_t(\theta)$ 钳制在 $[1-\epsilon, 1+\epsilon]$ 范围内。超参数 (hyperparameter) $\epsilon$ (epsilon) 通常是一个小值，例如 0.1 或 0.2。它定义了旧策略周围的信任区域。
4. **$\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t$：** 这是目标项的裁剪版本。
5. **$\min(...)$：** 最小值操作符是核心部分。
   - 如果 $\hat{A}_t > 0$（动作是好的）：目标变为 $\min(r_t(\theta) \hat{A}_t, (1+\epsilon) \hat{A}_t)$。这意味着如果 $r_t(\theta)$ 增加超过 $1+\epsilon$，目标将受到惩罚。我们限制了一步中可以增加一个好动作概率的幅度。
   - 如果 $\hat{A}_t < 0$（动作是坏的）：目标变为 $\min(r_t(\theta) \hat{A}_t, (1-\epsilon) \hat{A}_t)$。由于 $\hat{A}_t$ 是负数，这简化为 $\max(r_t(\theta) \hat{A}_t, (1-\epsilon) \hat{A}_t)$。如果 $r_t(\theta)$ 降低到 $1-\epsilon$ 以下，目标将受到惩罚。我们限制了一步中可以降低一个坏动作概率的幅度。

本质上，裁剪消除了策略发生剧烈变化的动机，从而防止了大的、可能破坏稳定的更新。策略被鼓励改进（$r_t(\theta)$ 朝着 $\hat{A}_t$ 所偏好的方向移动），但仅限于由 $\epsilon$ 设定的界限内。

### Actor-Critic 实现

PPO 通常使用 Actor-Critic 架构实现。

- **Actor（执行者）：** 策略网络 $\pi_\theta$（我们的 LLM），它决定采取哪个动作（令牌）。
- **Critic（评论员）：** 价值函数网络 $V_\phi(s_t)$（通常与 Actor 共享下层），它估计给定状态 $s_t$ 的预期回报（累积未来奖励）。此价值估计用于计算优势 $\hat{A}_t$。

整体优化涉及最大化 $L^{CLIP}$ 目标（针对 Actor），同时最小化价值函数（Critic）的损失函数 (loss function)，通常是预测值 $V_\phi(s_t)$ 与实际观测回报之间的均方误差。目标中还可以添加一个可选的熵奖励项，以鼓励探索。

> PPO Actor-Critic 在 RLHF 中循环的简要概述。Actor（LLM）生成文本，奖励模型提供奖励，Critic 估计状态价值。PPO 使用这些组件计算优势，并通过裁剪目标更新 Actor 和 Critic。

### 为什么 LLM 对齐 (alignment)使用 PPO？

与其他强化学习 (reinforcement learning)算法相比，PPO 取得了良好的平衡：

1. **稳定性：** 裁剪目标函数提供比简单策略梯度方法更稳定的训练更新，这对于训练成本高且易发散的大型模型来说很重要。
2. **样本效率：** 尽管在较简单场景下可能不如一些离策略方法样本效率高，但 PPO 通常被认为比 REINFORCE 等基本策略梯度方法更有效。它在每个数据收集阶段内重复使用在多个训练周期中收集的数据。
3. **实现复杂度：** 与信赖域策略优化 (TRPO) 等涉及二阶优化的一些替代方案相比，PPO 的实现和调整复杂性较低。

在 RLHF 的背景下，PPO 允许我们有效使用来自 RM 的标量奖励信号，指导 LLM 生成更符合人类偏好的输出，同时减轻在微调 (fine-tuning)过程中破坏模型稳定性的风险。下一节将详细介绍使用 PPO 实现此强化学习微调阶段的具体内容。

## 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍了近端策略优化（PPO）算法，详细阐述了用于稳定策略更新的裁剪替代目标函数。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  为强化学习提供了理论基础，包括策略梯度方法、Actor-Critic架构和优势函数估计。
- [TRL - Transformers Reinforcement Learning](https://huggingface.co/docs/trl/en/index) — Hugging Face (2024)
  Hugging Face TRL库的官方文档，提供了将PPO及其他RLHF技术应用于大型语言模型的工具和实现。

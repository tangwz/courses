---
course: "llm-alignment-safety"
chapter: "reinforcement-learning-human-feedback-rlhf"
lesson: "policy-optimization-ppo"
sourceId: 3132
sourceUrl: "https://apxml.com/zh/courses/llm-alignment-safety/chapter-2-reinforcement-learning-human-feedback-rlhf/policy-optimization-ppo"
title: "使用 PPO 进行策略优化"
description: "详细介绍使用近端策略优化 (PPO) 来微调大型语言模型。"
order: 5
plots: []
sourceHash: "0f01c1c97928e44830dea777ff3e45bcdd41feb255cb04405c9dbcfcb14a680e"
sourceCorrections: []
---

策略优化是基于人类反馈的强化学习 (reinforcement learning) (RLHF) 流程中的一个主要阶段。它紧随奖励模型 $r_\theta(x, y)$ 的训练之后，该模型旨在捕获人类偏好。此阶段的目的是使用这个奖励模型来增强语言模型的策略，$\pi_\phi(y|x)$。具体来说，目标是微调 (fine-tuning)大型语言模型，使其在给定提示 $x$ 的情况下，生成的回复 $y$ 根据奖励模型获得更高的分数。同时，这种微调还必须保持在预训练 (pre-training)和初始微调期间所获得的连贯且有用的语言能力，这通常被称为参考策略 $\pi_{ref}$。

直接使用监督学习 (supervised learning)在获得高奖励的样本上优化大型语言模型并不理想，因为它不能有效地考察可能输出的空间，也不能直接优化偏好目标。相反，我们将其视为一个强化学习问题，其中大型语言模型策略 $\pi_\phi$ 是智能体，上下文 (context) $x$ 是状态，生成的回复 $y$ 是动作（或动作序列），奖励模型 $r_\theta$ 提供奖励信号。

## RLHF 的强化学习 (reinforcement learning)目标

目标是寻找策略参数 (parameter) $\phi$，使其最大化从奖励模型获得的预期奖励，并通过一个 Kullback-Leibler (KL) 散度项进行惩罚。KL 惩罚项阻止策略 $\pi_\phi$ 偏离初始参考策略 $\pi_{ref}$ 太远，这有助于保持语言质量，并防止模型找到规避奖励模型的退化解。

优化目标可表示为：


$$
\text{最大化}_\phi \quad J(\phi) = \mathbb{E}_{x \sim D, y \sim \pi_\phi(y|x)} \left[ r_\theta(x, y) - \beta D_{KL}(\pi_\phi(y|x) || \pi_{ref}(y|x)) \right]
$$


让我们分解这个目标：

- $x \sim D$: 提示 $x$ 从分布 $D$ 中采样（通常与收集偏好数据所用的分布相同）。
- $y \sim \pi_\phi(y|x)$: 回复 $y$ 由我们正在优化的当前大型语言模型策略 $\pi_\phi$ 生成。
- $r_\theta(x, y)$: 训练好的奖励模型为提示-回复对 $(x, y)$ 分配的标量奖励。这是显示与人类偏好一致的主要信号。
- $D_{KL}(\pi_\phi(y|x) || \pi_{ref}(y|x))$: 当前策略 $\pi_\phi$ 生成的回复分布与参考策略 $\pi_{ref}$ 之间的 KL 散度。此项衡量当前策略与原始策略偏离的程度。参考策略通常是 RLHF 微调 (fine-tuning)*之前*的模型，例如监督微调 (SFT) 模型。
- $\beta$: 一个超参数 (hyperparameter)，它控制 KL 惩罚项的强度。较高的 $\beta$ 会阻止偏离 $\pi_{ref}$，而较低的 $\beta$ 则允许更积极地向奖励模型的偏好进行优化。

这个目标促使模型生成奖励模型偏好的输出，但不会以生成在统计上与初始良好模型 $\pi_{ref}$ 产生的内容差异很大的文本为代价。

## 为什么选择近端策略优化 (PPO)？

虽然像 REINFORCE 这样的标准策略梯度方法理论上可以使用，但它们通常会遭受高梯度方差和不稳定性的问题，特别是在处理大型语言模型庞大的参数 (parameter)空间和复杂的动作空间（token 序列）时。更新可能具有破坏性，因为一个大的梯度步长可能会显著损害策略的性能。

近端策略优化 (PPO) 已成为 RLHF 中一种广受欢迎且行之有效的算法。它属于信任区域优化方法系列，旨在通过限制策略在每个更新步骤中的变化量来提升稳定性。PPO 在不产生某些其他信任区域方法（如 TRPO）的巨大计算成本的情况下实现了这种稳定性。

PPO 在此应用中的主要优点包括：

1. **稳定性：** PPO 采用裁剪的替代目标函数，这会阻止大的策略更新，从而带来更稳定的训练。
2. **样本效率：** 它允许在同一批采样数据上进行多轮梯度更新，与单次更新方法相比，提升了数据使用率。
3. **简易性：** 尽管它比较复杂，但与 TRPO 等替代方法相比，其实现相对简单。

## PPO 的裁剪替代目标

PPO 的核心是其目标函数。设 $\pi_{\phi_{old}}$ 为更新前的策略（用于为当前优化步骤生成数据的策略）。PPO 根据以下目标优化策略 $\pi_\phi$：


$$
L^{CLIP}(\phi) = \mathbb{E}_{(x, y) \sim \pi_{\phi_{old}}} \left[ \min(r(\phi) \hat{A}(x, y), \text{clip}(r(\phi), 1-\epsilon, 1+\epsilon) \hat{A}(x, y)) \right]
$$


让我们分解与 RLHF 相关的组成部分：

- **概率比：** $r(\phi) = \frac{\pi_\phi(y|x)}{\pi_{\phi_{old}}(y|x)}$。这个比率衡量在新策略 $\pi_\phi$ 下，回复 $y$ 的可能性与旧策略 $\pi_{\phi_{old}}$ 相比如何。实际上，我们通过对数概率来计算：$\exp(\log \pi_\phi(y|x) - \log \pi_{\phi_{old}}(y|x))$。
- **优势估计：** $\hat{A}(x, y)$。这估计了回复 $y$ 相较于提示 $x$ 预期平均回复好多少。在 RLHF 的情境中，一种常见的方法是使用 KL 惩罚奖励作为优势：
  
  $$
  \hat{A}(x, y) \approx r_\theta(x, y) - \beta D_{KL}(\pi_\phi(y|x) || \pi_{ref}(y|x))
  $$
  
  有时，会从 $r_\theta(x, y)$ 中减去一个基线价值函数 $V(x)$（估计给定提示 $x$ 的预期奖励）以降低方差，但更简单的实现通常直接使用奖励模型分数（可能经过归一化 (normalization)或与 KL 项结合）。让我们将用于优化的组合奖励信号表示为 $R(x,y) = r_\theta(x, y) - \beta (\log \pi_\phi(y|x) - \log \pi_{ref}(y|x))$。因此，$\hat{A}(x,y) \approx R(x,y)$。
- **裁剪：** 函数 $\text{clip}(r(\phi), 1-\epsilon, 1+\epsilon)$ 将概率比 $r(\phi)$ 限制在区间 $[1-\epsilon, 1+\epsilon]$ 内。$\epsilon$ 是一个小的超参数 (parameter) (hyperparameter)（例如 0.2）。
  - 如果优势 $\hat{A}$ 为正（表示回复 $y$ 好于预期），则目标函数会随着 $r(\phi)$ 的增加而增加，这会促使策略更新。然而，$\min$ 函数和裁剪上限 $1+\epsilon$ 会阻止 $r(\phi)$ 变得过大，从而限制了更新步长。
  - 如果优势 $\hat{A}$ 为负（回复差于预期），则目标函数会随着 $r(\phi)$ 的增加而减小。裁剪下限 $1-\epsilon$ 阻止 $r(\phi)$ 变得过小，再次限制了策略在单个步骤中可以改变的量。

这种裁剪机制有效地为策略更新创建了一个悲观的边界，确保新策略不会以可能导致训练不稳定的方式与旧策略产生太大偏离。

## 大型语言模型微调 (fine-tuning)的 PPO 循环

将 PPO 应用于微调大型语言模型包含一个迭代过程：

> RLHF PPO 优化循环。采样提示，由当前策略生成回复，然后由奖励模型和参考策略共同评分以计算 KL 惩罚奖励。该批次数据用于通过 PPO 算法更新策略。

1. **采样提示：** 从数据集 $D$ 中抽取一批提示 $x$。
2. **生成回复：** 对于每个提示 $x$，使用*当前*策略 $\pi_\phi$ 生成回复 $y$。存储生成的序列 $(x, y)$ 及其对数概率 $\log \pi_\phi(y|x)$。这个 $\pi_\phi$ 在优化步骤中成为 $\pi_{\phi_{old}}$。
3. **评分回复：** 使用固定的奖励模型 $r_\theta$ 计算每个生成对的奖励分数 $r_\theta(x, y)$。
4. **计算参考对数概率：** 计算在固定参考策略 $\pi_{ref}$ 下生成回复 $y$ 的对数概率，得到 $\log \pi_{ref}(y|x)$。
5. **计算最终奖励：** 计算用于优化的奖励信号，通常包含 KL 惩罚项：$R(x, y) = r_\theta(x, y) - \beta (\log \pi_\phi(y|x) - \log \pi_{ref}(y|x))$。KL 项是使用当前/旧策略和参考策略之间对数概率的差异来估计的。
6. **优化策略：** 在收集到的经验批次 $(x, y, R(x, y), \log \pi_{\phi_{old}}(y|x))$ 上，对 PPO 裁剪替代目标 $L^{CLIP}(\phi)$ 进行多轮梯度上升。通常使用 Adam 或 AdamW 等优化器来更新语言模型的参数 (parameter) $\phi$。

这个循环重复进行，迭代地完善策略 $\pi_\phi$ 以便根据 $r_\theta$ 产生获得更高奖励的回复，同时保持与 $\pi_{ref}$ 的接近。

## 实际考量

- **KL 系数 ($\beta$)：** 选择一个合适的 $\beta$ 值很重要。如果 $\beta$ 过小，策略可能会过度优化奖励模型，可能导致重复或无意义的输出，规避奖励模型的弱点，或者与 $\pi_{ref}$ 中所包含的一般语言能力偏离太远。如果 $\beta$ 过大，策略更新将受到过度限制，模型在所需的对齐 (alignment)特性方面可能不会得到充分改进。$\beta$ 可以是一个固定超参数 (parameter) (hyperparameter)，有时也可以根据观察到的 KL 散度在训练期间动态调整。
- **模型副本：** 高效实现需要管理模型的多个副本或状态：正在积极训练的策略 ($\pi_\phi$)、用于生成当前经验批次的策略 ($\pi_{\phi_{old}}$) 和冻结的参考策略 ($\pi_{ref}$)。
- **计算资源：** PPO 步骤对计算要求较高。它包含：
  - 通过 $\pi_\phi$ 进行一次前向传播以生成回复。
  - 通过（可能很大的）奖励模型 $r_\theta$ 进行一次前向传播。
  - 通过参考模型 $\pi_{ref}$ 进行一次前向传播。
  - 在 PPO 优化迭代期间，通过 $\pi_\phi$ 进行多次前向和反向传播 (backpropagation)。

总之，PPO 为 RLHF 的策略优化阶段提供了一种有效且广受欢迎的方法。通过采用来自已学习奖励模型的信号，并在稳定的优化框架中加入 KL 散度惩罚，PPO 能够微调 (fine-tuning)大型语言模型，使其更好地符合人类偏好，同时保留其核心语言理解和生成能力。

## 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍了近端策略优化（PPO）算法的原始论文，详细阐述了其裁剪代理目标以及实现稳定策略优化的特点。
- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: The MIT Press
  一本经典教材，提供了强化学习概念的全面介绍，涵盖了策略梯度方法以及PPO等算法的理论基础。

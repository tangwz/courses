---
course: "rlhf-reinforcement-learning-human-feedback"
chapter: "rl-ppo-fine-tuning"
lesson: "kl-divergence-penalty-role"
sourceId: 5148
sourceUrl: "https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-4-rl-ppo-fine-tuning/kl-divergence-penalty-role"
title: "KL散度惩罚的作用"
description: "理解并实践KL约束以防止策略崩溃。"
order: 3
plots: ["plots/5148-0.json"]
sourceHash: "21c2f4561e859ad0e51613befba38f278f81484751a5a19695eb6fe317a617f7"
sourceCorrections: []
---

强化学习 (reinforcement learning)人类反馈 (RLHF) 中的近端策略优化 (PPO) 阶段，主要目标是调整语言模型的策略$\pi_{\theta}$，使其生成能最大化已学习奖励模型（RM）所给预期奖励的响应。然而，仅针对RM分数进行优化会带来很大风险。策略可能迅速进入策略空间中，这些区域根据RM会产生高奖励，但输出内容却可能是无意义的、重复的，或在风格上与初始监督微调 (fine-tuning)（SFT）阶段建立的期望行为不符。这种现象可以看作是策略对奖励模型“过拟合 (overfitting)”，可能借助其不准确性或局限性（一种奖励欺骗），或者简单地忘记了在此阶段之前它所具备的基本语言生成能力。

为减轻这种情况，PPO引入了一个基于库尔巴克-莱布勒（KL）散度的惩罚项。KL散度，表示为 $D_{KL}(\pi_{\theta} || \pi_{ref})$，衡量两个概率分布之间的差异。在RLHF的背景下，它量化 (quantization)了当前策略 $\pi_{\theta}$ 偏离 *参考策略* $\pi_{ref}$ 的程度。通常，这个参考策略是SFT阶段之后获得的模型，我们称之为 $\pi_{SFT}$。

当前策略与SFT策略在给定状态（提示）$s$ 和动作（标记 (token)）$a$ 上的KL散度计算如下：


$$
D_{KL}(\pi_{\theta}(\cdot|s) || \pi_{SFT}(\cdot|s)) = \sum_{a} \pi_{\theta}(a|s) \log \frac{\pi_{\theta}(a|s)}{\pi_{SFT}(a|s)}
$$


低KL散度表明当前策略的输出分布与SFT策略的分布相似，而高KL散度则表明有明显偏离。

核心思想是增加PPO目标函数。并非单纯地最大化预期优势（这与奖励相关），我们最大化一个修改后的目标，其中包含与KL散度成比例的惩罚项：


$$
\text{目标} \approx E_{t} [ \text{奖励}_t ] - \beta D_{KL}(\pi_{\theta}(\cdot|s_t) || \pi_{SFT}(\cdot|s_t))
$$


此处说明：

- $E_{t} [ \text{奖励}_t ]$ 表示策略 $\pi_{\theta}$ 获得的预期奖励，通常在实际的PPO损失函数 (loss function)中，使用优势估计 $A_t$ 进行近似。这一项促使策略生成奖励模型偏好的内容。
- $D_{KL}(\pi_{\theta}(\cdot|s_t) || \pi_{SFT}(\cdot|s_t))$ 是在时间步 $t$ 生成的标记上，当前策略与冻结的SFT策略之间的KL散度。
- $\beta$ 是一个控制KL惩罚强度的超参数 (parameter) (hyperparameter)。

这个KL惩罚项作为正则化 (regularization)项。它阻止策略 $\pi_{\theta}$ 在优化过程中过度偏离SFT策略 $\pi_{SFT}$。通过惩罚每个标记输出概率分布的大范围变化，它有助于确保模型保留在SFT阶段获得的通用语言流畅性、知识和风格特征，即使它在适应以最大化奖励信号。

### 实现细节

实际上，在PPO训练循环中：

1. 采样一批提示。
2. 当前策略 $\pi_{\theta}$ 对这些提示生成响应。
3. 对于每个生成的标记 (token)，当前策略 $\pi_{\theta}$ 和冻结的参考策略 $\pi_{SFT}$ 都计算它们在词汇表 (vocabulary)上的各自概率分布（或logits）。
4. 计算这两个分布之间每个标记的KL散度。
5. 每个标记的KL散度通常在生成的序列和批次上进行平均。
6. 这个按 $\beta$ 缩放的平均KL散度将从奖励信号中减去（或与策略损失和价值损失项一起纳入PPO损失计算中）。

参考策略 $\pi_{SFT}$ 在整个PPO训练过程中保持固定；它的权重 (weight)不会更新。它作为一个稳定的锚点，代表从初始监督数据集中学习到的行为。

### KL系数（$\beta$）

KL系数 $\beta$ 的选择对于平衡探索与稳定性很重要。

- **低 $\beta$：** 惩罚较弱。策略 $\pi_{\theta}$ 有更大的自由度偏离 $\pi_{SFT}$ 以最大化奖励。这可能导致更快地向奖励信号优化，但如果策略偏离过远，会增加策略崩溃、不稳定或生成不良文本的风险。
- **高 $\beta$：** 惩罚较强。策略 $\pi_{\theta}$ 被严格限制，以保持接近 $\pi_{SFT}$。这有助于提升稳定性并保留SFT模型的特性，但可能会阻碍策略充分适应奖励信号，可能导致相比SFT模型仅有微小改进。



![KL系数 (Beta) 对策略更新的影响](plots/5148-0.json)



> PPO训练中的策略更新。$\pi_{SFT}$ 是初始策略。低 $\beta$ 允许向高奖励区域迈出更大的步长，但有不稳定的风险。高 $\beta$ 限制步长，保持与 $\pi_{SFT}$ 的接近。

寻找 $\beta$ 的合适值通常需要进行实验。此外，自适应KL控制器也常被使用。这些控制器根据每个批次中观察到的KL散度值在训练期间动态调整 $\beta$。目标是将实际的 $D_{KL}(\pi_{\theta} || \pi_{SFT})$ 保持在预设的目标范围内（例如，保持平均KL为6纳特）。如果观察到的KL超过目标，则增加 $\beta$ 以加强惩罚；如果低于目标，则减小 $\beta$ 以允许更多的优化。像Hugging Face的TRL这样的库提供了此类自适应KL控制器的实现。

总之，KL散度惩罚是PPO在RLHF中的一种机制。它防止语言模型策略在优化奖励模型中编码的人类偏好时，偏离初始SFT模型过远。这有助于提升训练稳定性，保留基础模型的理想特征，并提供了一种在奖励最大化和策略约束之间可调节的平衡。

## 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  这篇基础论文介绍了近端策略优化（PPO）算法，为在强化学习中通过约束目标保持策略稳定性提供了理论基础。
- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: arXiv preprint arXiv:2203.02155; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  这篇有影响力的论文详细介绍了将PPO与KL散度正则化应用于通过人类反馈训练大型语言模型，展示了其在使模型与人类偏好对齐方面的有效性。
- [Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano (2020)
  Journal: NeurIPS 2020; DOI: [10.48550/arXiv.2009.01325](https://doi.org/10.48550/arXiv.2009.01325)
  这是强化学习从人类反馈到文本摘要的早期应用，该工作强调了使用KL散度惩罚来平衡奖励最大化与保持生成文本的连贯性和质量。
- [TRL (Transformer Reinforcement Learning) Library Documentation](https://huggingface.co/docs/trl/index) — Hugging Face (2023)
  Publisher: Hugging Face
  Hugging Face TRL库的官方文档，提供了RLHF的实际实现，包括文中提到的带有KL正则化和自适应KL控制器的PPO。

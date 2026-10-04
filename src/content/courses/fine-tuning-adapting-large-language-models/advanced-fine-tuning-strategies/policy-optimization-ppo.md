---
course: "fine-tuning-adapting-large-language-models"
chapter: "advanced-fine-tuning-strategies"
lesson: "policy-optimization-ppo"
sourceId: 3059
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-5-advanced-fine-tuning-strategies/policy-optimization-ppo"
title: "使用 PPO 进行策略优化"
description: "理解近端策略优化算法，用于基于奖励模型微调大语言模型。"
order: 6
plots: []
sourceHash: "2a17dbdb98661fad9df96607824ba95197eace5ac35595c800e4fcc05adbe642"
sourceCorrections: []
---

策略优化使用奖励模型 $r_\phi(x, y)$ 来微调 (fine-tuning)语言模型，该模型评估给定提示 $x$ 的不同响应 $y$ 的人类偏好。这个过程涉及调整语言模型策略 $\pi_\theta(y|x)$ 的参数 (parameter) $\theta$，使其生成从奖励模型获得高分的响应。这有效地使模型的行为与学到的偏好保持一致。近端策略优化 (PPO) 是人类反馈强化学习 (reinforcement learning) (RLHF) 中此策略优化阶段最常用的算法。

PPO 是一种同策略强化学习算法，旨在实现稳定高效的策略更新。在大语言模型 (LLM)的背景下，“同策略”表示更新是基于我们试图改进的策略 $\pi_\theta$ 当前版本生成的数据。其核心思路是最大化由 $r_\phi$ 预测的预期奖励，同时确保更新后的策略 $\pi_\theta$ 不会偏离参考策略太远，参考策略通常是最初的监督微调 (SFT) 模型，表示为 $\pi_{\text{ref}}$。这种约束对于防止模型陷入生成重复的、高奖励但无意义的文本（有时称为“奖励作弊”）以及保持模型的一般语言能力非常重要。

### RLHF PPO 目标函数

RLHF 中的优化过程通常涉及最大化一个目标函数，该函数将奖励信号与惩罚项结合起来。对于从数据集 $D$ 中采样的给定提示 $x$，以及从当前策略 $\pi_\theta(y|x)$ 中采样的响应 $y$，单步目标可以表示为：


$$
\text{目标}(\theta) = \mathbb{E}_{(x,y) \sim D, \pi_\theta} [ \underbrace{r_\phi(x,y)}_{\text{奖励}} - \underbrace{\beta \cdot \text{KL}(\pi_\theta(y|x) || \pi_{\text{ref}}(y|x))}_{\text{KL 惩罚}} ]
$$


让我们分解一下这个目标：

1. **奖励项 ($r_\phi(x,y)$)**: 这是训练好的奖励模型赋予提示-响应对 $(x, y)$ 的分数。最大化此项直接促使策略 $\pi_\theta$ 生成预期受人类偏好的响应。
2. **KL 惩罚项**: 该项衡量当前策略 $\pi_\theta$ 与参考策略 $\pi_{\text{ref}}$ 之间的 Kullback-Leibler (KL) 散度。KL 散度量化 (quantization)了由 $\pi_\theta$ 生成的响应的概率分布与 $\pi_{\text{ref}}$ 生成的响应的概率分布之间的差异程度。乘以系数 $\beta$ 控制此惩罚的强度。
   - **高 $\beta$ 值**迫使更新后的策略与原始SFT模型保持非常接近，可能限制对齐 (alignment)效果的提升。
   - **低 $\beta$ 值**允许策略更显著地偏离，可能导致更好的对齐，但如果奖励模型不完善，则可能导致语言质量或连贯性下降。
   - 参考策略 $\pi_{\text{ref}}$ 在 PPO 训练期间通常保持不变。它的作用很重要：它作为一个锚点，防止优化后的策略 $\pi_\theta$ 偏离到可能产生高奖励信号但对应不自然或重复语言的策略空间区域。

在实践中，PPO 包含额外的组成部分，例如值函数和优势估计，以稳定训练，但核心目标仍然是平衡奖励最大化与 KL 散度约束。完整的 PPO 算法采用诸如裁剪策略比率等技术，以防止单步更新过大，从而进一步提高稳定性。

### 大语言模型 (LLM)的 PPO 训练循环

大语言模型的 PPO 训练过程通常按以下步骤迭代进行：

1. **采样提示**: 从提示数据集 $D$ 中抽取一批提示 $x$。
2. **生成响应**: 对于每个提示 $x$，通过从当前策略 $\pi_\theta(y|x)$ 中采样来生成响应 $y$。
3. **计算奖励**: 使用冻结的奖励模型计算每对 $(x, y)$ 的奖励: $r = r_\phi(x, y)$。
4. **计算 KL 散度**: 计算当前策略的输出分布 $\pi_\theta(y|x)$ 与参考策略的输出分布 $\pi_{\text{ref}}(y|x)$ 之间对于每个生成的响应 $y$ 的 KL 散度。
5. **估计优势 (简化视图)**: 确定获得的奖励 $r$ 与预期基线（通常使用与策略一起训练的值函数进行估计）相比好多少。
6. **计算 PPO 目标**: 根据 PPO 目标函数计算损失，结合奖励、KL 惩罚以及可能的裁剪优势项。
7. **更新策略**: 对策略 $\pi_\theta$ 的参数 (parameter) $\theta$ 执行梯度更新，以最大化目标（或最小化负目标/损失）。
8. **重复**: 继续循环指定步数或直到收敛。

> RLHF 中使用 PPO 进行策略优化的迭代过程。策略网络 $\pi_\theta$ 生成响应，由奖励模型 $r_\phi$ 进行评估。相对于参考模型 $\pi_{\text{ref}}$ 的 KL 散度作为一种约束。这些组件构成了 PPO 目标，用于更新 $\pi_\theta$。

### 实际考虑

为大语言模型 (LLM)部署 PPO 需要仔细管理计算资源。由于它是一种同策略算法，需要频繁使用当前策略 $\pi_\theta$ 生成新响应。这涉及到在训练循环中对大型模型进行推理 (inference)，这可能计算密集。

- **模型同步**: 在训练期间保持策略 $\pi_\theta$、参考策略 $\pi_{\text{ref}}$、奖励模型 $r_\phi$ 以及可能的值函数模型同步且可访问需要大量内存，通常需要分布式训练设置（如 DeepSpeed ZeRO 或 FSDP）。
- **批处理**: 提示以批次处理，并同时生成和评估响应以利用并行计算。
- **超参数 (parameter) (hyperparameter)敏感性**: PPO 在 RLHF 中的表现对超参数非常敏感，特别是 KL 系数 $\beta$、学习率以及 PPO 算法本身中与优势估计和裁剪相关的参数。调整这些通常需要大量的实验。

通过仔细应用 PPO，我们可以引导大语言模型的行为更好地与人类偏好数据所捕获并编码在奖励模型中的期望属性保持一致，从简单的监督微调 (fine-tuning)转向更精细的行为调整。这一步对于创建不仅有能力而且根据指定标准有用、无害和诚实的模型非常重要。

## 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍了近端策略优化 (PPO) 算法，一种用于深度强化学习中稳定高效策略优化的方法，与RLHF高度相关。
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: arXiv preprint arXiv:2203.02155; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  展示了如何将人类反馈强化学习 (RLHF) 和 PPO 应用于大型语言模型，以使其与人类偏好对齐，从而产生了如 InstructGPT 等模型。注意：此论文的完整作者列表异常冗长且源文件中存在重复；此处提供了一个有代表性的子集。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  提供了强化学习概念的全面介绍，对于理解 PPO 等策略优化算法的理论基础至关重要。

---
course: "llm-alignment-safety"
chapter: "reinforcement-learning-human-feedback-rlhf"
lesson: "ppo-implementation-considerations"
sourceId: 3135
sourceUrl: "https://apxml.com/zh/courses/llm-alignment-safety/chapter-2-reinforcement-learning-human-feedback-rlhf/ppo-implementation-considerations"
title: "PPO 实施考量"
description: "Cover practical implementation details and hyperparameter tuning for PPO in RLHF."
order: 6
plots: []
sourceHash: "68cbb3dd180aa7a775f0fd1370701ecbb8de954fa44d146ba0e1f451ac019676"
sourceCorrections: []
---

优化语言模型的策略 $\pi_\phi(y|x)$，借助奖励模型 $r_\theta(x, y)$ 提供的信号，是 RLHF 微调 (fine-tuning)阶段的核心。近端策略优化 (PPO) 是此任务最常用的算法，因为它相较于其他强化学习 (reinforcement learning)算法具有较好的稳定性和样本效率。然而，将 PPO 应用于大型语言模型需要仔细考虑一些实施细节。

### 大型语言模型微调 (fine-tuning)的 PPO 目标函数

目标是更新策略 $\pi_\phi$，以最大化 $r_\theta$ 所赋的奖励，同时避免策略与其原始行为（由参考模型 $\pi_{ref}$ 表示）产生过大偏离。此参考模型通常是 RLHF 阶段前获得的监督微调 (SFT) 模型。此约束有助于维持语言连贯性，防止通用能力灾难性遗忘，并对优化过程进行正则化 (regularization)。

RLHF 中使用的目标函数通常如下：


$$
\text{目标函数}(\phi) = \mathbb{E}_{(x, y) \sim \pi_\phi} [ r_\theta(x, y) - \beta \cdot D_{KL}(\pi_\phi(y|x) || \pi_{ref}(y|x)) ]
$$


我们来逐项分析：

1. **$x$**: 输入提示。
2. **$y$**: 从当前策略 $\pi_\phi(y|x)$ 中采样的生成响应。
3. **$r_\theta(x, y)$**: 训练好的奖励模型为提示-响应对 $(x, y)$ 分配的标量奖励。期望 $\mathbb{E}_{(x, y) \sim \pi_\phi}$ 表示我们希望最大化由我们的策略生成的响应的平均奖励。
4. **$D_{KL}(\pi_\phi(y|x) || \pi_{ref}(y|x))$**: 当前策略 $\pi_\phi$ 和参考策略 $\pi_{ref}$ 在给定提示 $x$ 下响应 $y$ 的概率分布之间的 Kullback–Leibler (KL) 散度。此项衡量了策略 $\pi_\phi$ 在给定提示和响应下与参考模型 $\pi_{ref}$ 的偏离程度。
5. **$\beta$**: 控制 KL 惩罚强度的一个超参数 (parameter) (hyperparameter)。较高的 $\beta$ 对偏离 $\pi_{ref}$ 的行为施加更强的惩罚，导致策略更新幅度较小。较低的 $\beta$ 允许策略更积极地寻求更高奖励，但这可能会牺牲稳定性和连贯性。

PPO 通过迭代过程优化此目标：

1. 从数据集中抽样提示 $x$。
2. 使用当前策略 $\pi_\phi$ 生成响应 $y$。
3. 使用奖励模型 $r_\theta$ 评估响应以获得奖励。
4. 计算生成响应的 $\pi_\phi$ 和 $\pi_{ref}$ 之间的 KL 散度。
5. 使用 PPO 算法（具体是其裁剪代理目标函数和可能的价值函数）根据奖励信号减去 KL 惩罚来计算策略梯度。
6. 更新策略模型的参数 $\phi$。

### RLHF 中的 PPO 算法主要组成

虽然上述目标函数定义了我们要优化的*内容*，但 PPO 提供了*方法*。它包含：

- **执行者 (Actor):** 大型语言模型策略 $\pi_\phi$ 本身，它生成响应（行动）。
- **评价者 (Critic)（价值函数）:** 一个模型 $V_\psi(x)$，用于评估从提示 $x$ 开始的预期回报（累积未来奖励，可能根据 KL 惩罚进行调整）。这有助于减少策略梯度估计的方差。通常，价值函数模型会从奖励模型 $r_\theta$ 初始化，甚至与其共享大部分参数 (parameter)，可能只拥有一个独立的输出层。
- **优势估计:** 计算生成响应 $y$ 对提示 $x$ 的“优势” $A(x, y)$。这通常使用广义优势估计 (GAE) 完成，GAE 使用价值函数估计来平衡偏差和方差。 $A(x, y) \approx r_\theta(x, y) + \gamma V_\psi(x') - V_\psi(x)$（简化形式，GAE 更为复杂）。KL 惩罚被纳入用于优势计算的奖励信号中。
- **裁剪代理目标函数:** PPO 的标志性特点。它修改了策略更新目标，以防止更新后策略与旧策略之间的策略概率比率发生过大变化，从而提高稳定性。

### 超参数 (parameter) (hyperparameter)调整要点

找到合适的超参数对于稳定且有效的 PPO 训练非常重要。以下是一些重要参数：

- **学习率:** 控制策略 $\pi_\phi$ 和可能的价值函数 $V_\psi$ 更新的步长。通常需要比标准监督训练更小的学习率（例如，$1e-5$ 到 $1e-6$）。使用学习率调度器（例如，线性衰减）很常见。
- **PPO 批次大小:**
  - `rollout_batch_size` 或 `generation_batch_size`: 并行处理的提示数量，用于生成响应和收集经验。较大的批次提供每次迭代更丰富的经验，但会增加内存占用。
  - `ppo_batch_size`: 单次 PPO 更新迭代中使用的经验数据总量。通常包含多个 `rollout_batch_size` 的集合。
  - `ppo_mini_batch_size`: 在 PPO 更新步骤中用于计算梯度的微批次大小。必须小于 `ppo_batch_size`。较小的微批次会引入噪声，但可以提高泛化能力并减少每一步的内存需求。
- **PPO 迭代次数 (Epochs):** 算法在单个 PPO 更新阶段中，对收集到的经验 (`ppo_batch_size`) 进行迭代以更新策略和价值函数参数的次数。值通常在 1 到 4 之间。过多的迭代可能导致对当前批次经验的过拟合 (overfitting)。
- **裁剪参数 ($\epsilon$):** PPO 裁剪代理目标函数中的 `clip_range`。它限制策略比率 $r_t(\phi) = \frac{\pi_\phi(y_t|x_t)}{\pi_{old}(y_t|x_t)}$ 可以偏离 1 的程度。典型值介于 0.1 和 0.3 之间。它有助于防止大规模、破坏稳定性的策略更新。
- **KL 系数 ($\beta$):** 平衡奖励最大化与保持接近 $\pi_{ref}$。这是最敏感的超参数之一。
  - 过低：策略可能发生明显漂移，可能生成无意义文本或“奖励作弊”（利用奖励模型漏洞）。
  - 过高：策略更新变得微小，阻碍对齐 (alignment)进展（KL 崩溃）。
  - 值通常介于 0.01 到 0.2 之间。一些实现使用自适应 KL 控制器，在训练期间调整 $\beta$ 以将 KL 散度保持在目标范围内。
- **价值函数系数 ($c_1$ 或 `vf_coef`):** 应用于 PPO 总损失中价值函数损失项的权重 (weight)。通常在 0.1 到 1.0 之间。有助于确保价值函数得到充分训练。
- **GAE 参数 ($\lambda$， $\gamma$):**
  - $\gamma$（折扣因子）：标准强化学习 (reinforcement learning)折扣因子，对于非片段式大型语言模型任务通常接近 1（例如 0.99）。
  - $\lambda$（GAE 的 lambda）：控制优势估计中的偏差-方差权衡。典型值为 0.95。

> 流程图说明了 RLHF 中 PPO 更新步骤期间不同模型 ($\pi_\phi$，$\pi_{ref}$，$r_\theta$， $V_\psi$) 之间的互动。该过程从一个提示开始，使用当前策略生成响应，计算奖励和 KL 惩罚，估计优势，最后更新策略（以及可选的价值）参数。

### 计算与稳定性难题

- **内存占用:** PPO 需要在内存中保留大型语言模型（或其参数 (parameter)）的多个副本：正在训练的策略 $\pi_\phi$、用于 KL 计算的参考模型 $\pi_{ref}$、价值模型 $V_\psi$（如果独立），以及可能的奖励模型 $r_\theta$。反向传播 (backpropagation)过程中的激活和梯度进一步增加了内存负担。通常需要参数高效微调 (fine-tuning) (PEFT) 等技术，例如 LoRA（低秩适应）。LoRA 通过冻结基础大型语言模型并注入小的、可训练的低秩矩阵，显著减少了可训练参数的数量。
- **训练速度:** 生成响应、计算奖励、KL 散度以及跨多个模型执行 PPO 更新的计算量很大。通常需要分布式训练框架（如 DeepSpeed ZeRO）才能高效地训练大型模型。
- **稳定性:** PPO 训练可能不稳定。梯度裁剪（限制梯度范数）和价值函数裁剪是常用做法。仔细的初始化（例如，从 $\pi_{ref}$ 初始化 $\pi_\phi$，并根据 $r_\theta$ 初始化 $V_\psi$）有所帮助。在整个训练过程中监测 KL 散度、奖励、价值损失和策略熵等指标对于诊断问题非常重要。
- **奖励作弊:** 策略可能会发现意想不到的方式来最大化 $r_\theta$ 预测的奖励，而这些方式并不对应于在有用性或无害性上的真实提升。例如，如果奖励模型无意中偏好某些特性，它可能会生成过长、重复或奉承的响应。这需要持续监控，并可能需要重新训练奖励模型或调整 KL 惩罚。
- **KL 崩溃:** 如果 $\beta$ 过高或奖励信号弱/有噪声，KL 项可能主导目标函数，导致 $\pi_\phi$ 收敛回 $\pi_{ref}$ 并停止学习。

### 实用建议

- **从简入手:** 从成功的 RLHF 实现（例如 InstructGPT、Llama 2 论文）中既定的超参数 (parameter) (hyperparameter)设置开始，并谨慎调整。
- **使用 PEFT:** 采用 LoRA 等技术来管理内存限制并加快训练速度，特别是对于超大型模型。
- **密切监测:** 跟踪重要指标：平均奖励、KL 散度（应保持在合理范围内，例如 5-10 nats）、策略/价值损失和梯度范数。使用 Weights & Biases 或 TensorBoard 等工具。
- **自适应 KL 控制:** 考虑使用自适应 KL 控制器，动态调整 $\beta$ 以将 KL 散度保持在预设目标值附近。这与固定 $\beta$ 相比可以提高稳定性。
- **定性评估:** 在训练期间定期从策略模型中抽样生成内容，检查其连贯性、期望的行为变化以及潜在的奖励作弊问题。仅靠自动化基准测试是不够的。
- **价值函数初始化:** 从奖励模型初始化价值函数参数通常能提供一个良好的起点，并能加快收敛速度。

在 RLHF 中实施 PPO 需要处理奖励最大化、策略正则化 (regularization)、计算限制以及超参数敏感性之间复杂的互动关系。系统的方法、仔细的监测和迭代改进对于使用此技术成功对齐 (alignment)大型语言模型是必要的。

## 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍了近端策略优化（PPO）算法，详述了其裁剪替代目标和广义优势估计（GAE）。
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: arXiv preprint arXiv:2203.02155; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  描述了基于人类反馈的强化学习（RLHF）流程，特别展示了PPO在使语言模型与人类偏好对齐方面的应用。
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了低秩适应（LoRA），一种参数高效微调方法，广泛用于减少大型语言模型的内存占用并提高训练效率。

---
course: "rlhf-reinforcement-learning-human-feedback"
chapter: "rl-ppo-fine-tuning"
lesson: "ppo-for-rlhf-context"
sourceId: 5141
sourceUrl: "https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-4-rl-ppo-fine-tuning/ppo-for-rlhf-context"
title: "RLHF背景下的PPO算法"
description: "PPO算法在大型语言模型微调中的应用。"
order: 1
plots: []
sourceHash: "7f01269cc44804dd09b29fc2ad32928aaad160d369f639cbbc06848a22f6193d"
sourceCorrections: []
---

奖励模型（$r_{\phi}$）能够根据人类偏好对语言模型输出进行评分。为了优化语言模型本身以生成在此奖励模型中获得高分的输出，强化学习 (reinforcement learning)（RL）发挥了作用。特别是近端策略优化（PPO）已成为RLHF此阶段的标准算法。

PPO是一种策略梯度算法，旨在在策略上尽可能迈出最大的改进步，同时避免步子过大导致性能崩溃。它通过将策略更新限制在“信任区域”内来实现这一点，从而防止策略迭代之间发生剧烈变化。与更简单的策略梯度方法相比，这使得它更加稳定且样本效率更高，这在处理大型复杂模型（如LLMs）时尤为重要。

### 为何RLHF选用PPO？

尽管存在多种强化学习 (reinforcement learning)算法，但PPO具有多项优点，使其非常适合用于大型语言模型的微调 (fine-tuning)：

1. **稳定性：** 它的核心机制，无论是通过KL散度惩罚还是目标裁剪，都限制了策略在每个更新步骤中可以改变的程度。这防止了模型剧烈偏离或“遗忘”在预训练 (pre-training)和SFT期间学到的语言能力。
2. **样本效率：** 与需要大量离策略校正或复杂探索策略的算法相比，PPO在样本效率和实现简易性之间取得了不错的平衡。虽然它仍然是在线策略（需要从当前策略获取新样本），但其更新机制允许在收集的数据批次上进行多个优化周期。
3. **实现简易性：** 相对于其他一些高级强化学习算法，PPO的更新规则更易于实现和调试，特别是有了Hugging Face的TRL等库提供优化组件的情况下。
4. **与大型模型的兼容性：** PPO可以轻松地与典型的演员-评论员架构结合，这与LLM微调方法非常契合。

### PPO在RLHF中的目标

强化学习 (reinforcement learning)阶段的主要任务是调整语言模型策略（$\pi_{\theta}$）的参数 (parameter)（$\theta$），以最大化奖励模型给出的预期奖励，同时确保策略不会过度偏离原始的监督微调 (fine-tuning)（SFT）策略（$\pi_{SFT}$）。这可以防止模型生成利用奖励模型（奖励作弊）但内容不合理或风格不一致的输出。

RLHF的PPO优化过程包含了这个双重目标。对于从分布$D$中采样得到的给定提示$x$，以及当前策略$\pi_{\theta}(y|x)$生成的响应$y$，每个提示-响应对的目标都包含了奖励以及基于当前策略与参考SFT策略之间Kullback-Leibler (KL) 散度的惩罚：


$$
\text{目标}(x, y) = r_{\phi}(x, y) - \beta D_{KL}(\pi_{\theta}(y|x) || \pi_{SFT}(y|x))
$$


我们来分解一下：

- $r_{\phi}(x, y)$: 这是我们训练好的奖励模型分配给提示-响应对 $(x, y)$ 的标量奖励。目标是最大化这个值。
- $D_{KL}(\pi_{\theta}(y|x) || \pi_{SFT}(y|x))$: 这是在当前策略$\pi_{\theta}$下给定提示$x$时响应$y$的概率分布与参考SFT策略$\pi_{SFT}$之间的KL散度。它衡量了当前策略对于此次特定生成偏离SFT模型的程度。较低的KL散度表示策略更相似。
- $\beta$: 这是一个超参数 (hyperparameter)，控制KL惩罚的强度。
  - 高$\beta$值会严重惩罚偏离$\pi_{SFT}$的行为，使模型在风格上接近SFT版本，但可能限制奖励最大化。
  - 低$\beta$值允许策略更积极地优化奖励$r_{\phi}$，但可能显著偏离SFT模型，从而可能降低文本质量或导致奖励模型被利用。

PPO的整体目标旨在最大化使用当前策略$\pi_{\theta}$采样得到的轨迹中这个组合项的期望值。然后，PPO算法会使用其特定机制（例如裁剪替代目标，我们将在后面讨论）来有效优化此目标。

### 调整PPO组件以适应LLMs

在标准的PPO设置中，通常会有一个演员（即策略）和一个评论员（即价值函数）。在RLHF与LLMs结合的背景下，它们对应如下：

1. **策略模型（演员）：** 这是我们正在微调 (fine-tuning)的语言模型（$\pi_{\theta}$）。它以提示$x$作为输入（状态），并生成一系列token $y$（动作）。我们通常从SFT模型的权重 (weight)开始，并在PPO训练期间更新它们。
2. **参考模型：** 这是初始SFT模型（$\pi_{SFT}$）的一个固定副本。它对于计算KL散度惩罚很重要。保持其固定提供了一个稳定的目标分布，用于正则化 (regularization)。
3. **奖励模型：** 这是在前一阶段训练的模型（$r_{\phi}$）。它接受提示$x$和生成的响应$y$，并输出一个标量奖励。在PPO阶段，它通常保持固定。
4. **价值模型（评论员）：** 这个模型（$V_{\psi}$）以提示$x$作为输入，并估算从该提示开始、在当前策略$\pi_{\theta}$下预期的折扣未来奖励。它通过计算优势值来帮助减少策略梯度估计的方差。通常，价值模型从奖励模型的权重或SFT模型的权重初始化，可能带有不同的输出头部。其参数 (parameter)（$\psi$）在PPO训练期间与策略参数$\theta$一同更新。

下面的图表显示了RLHF中PPO步骤期间的高层交互：

> 交互图显示了RLHF中PPO步骤的流程。提示被输入到策略模型以生成响应。这些响应由奖励模型、价值模型和参考模型进行评估，以计算奖励、价值和KL散度惩罚，然后这些信息被用于计算优势并更新策略模型和价值模型，通过PPO目标。

### RLHF中的PPO训练循环

概括而言，PPO微调 (fine-tuning)过程涉及迭代执行以下步骤：

1. **推出：** 从数据集中采样一批提示（$x$）（通常与奖励建模或SFT使用的数据集相同）。对于每个提示，使用*当前*策略$\pi_{\theta}$生成一个响应（$y$）。存储提示、响应以及来自$\pi_{\theta}$的token对数概率。
2. **评估：** 对于每个生成的提示-响应对 $(x, y)$：
   - 使用固定的奖励模型计算奖励 $r = r_{\phi}(x, y)$。
   - 计算KL散度惩罚项。这通常包括获取在固定参考模型$\pi_{SFT}$下生成响应$y$的对数概率，并将其与在推出过程中从$\pi_{\theta}$获得的对数概率进行比较。用于PPO更新的奖励通常会调整为 $r - \beta \times \text{KL}$。
   - 使用当前价值模型估算价值 $V_{\psi}(x)$。
3. **优势估算：** 计算响应中每个生成token的优势估算。这通常使用广义优势估算（GAE），它结合了奖励和价值估算，为策略更新提供噪声较小的信号。我们将在“计算优势和回报”一节中详细介绍GAE。
4. **优化（“信任区域”更新）：** 对收集到的推出数据批次执行多个周期的梯度更新。PPO 不仅仅是简单地朝着高奖励方向移动模型，而是通过\*\*裁剪替代目标函数（Clipped Surrogate Objective）\*\*确保更新是“安全”的：
   - **概率比率 ($r_t$)：** 对于每个 token，我们计算*当前*正在更新的策略与推出（Rollout）时使用的*旧*策略之间的概率比率：
     $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$
   - **策略更新 (演员/Actor)：** 通过最大化 PPO 目标函数来更新 $\theta$。如果比率 $r_t$ 偏离 1 太远（例如超过 20% 的变化），目标函数就会被**裁剪（Clip）**。这可以防止某个单次的高奖励信号过度推挤模型，从而避免破坏在 SFT 阶段学到的语言能力或模型稳定性。
   - **价值更新 (评论员/Critic)：** 与此同时，通过最小化价值模型 $V_{\psi}$ 的预测与推出期间实际观察到的回报（奖励减去 KL 惩罚）之间的差异（通常为均方误差），来更新其参数 (parameter) $\psi$。

这个循环重复进行，逐步改善策略$\pi_{\theta}$，使其生成的响应更好地符合奖励模型所捕获的偏好，同时KL惩罚和PPO的裁剪机制保持训练稳定并防止灾难性遗忘。

理解PPO的这种调整非常重要。后续章节将更深入地讲解策略和价值网络的实现、KL散度惩罚的具体细节、优势如何计算，以及LLM对齐 (alignment)PPO阶段的调优和故障排除的实际操作。

## 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍PPO算法的原始论文，详细阐述了其目标剪裁和KL惩罚机制，以实现稳定高效的策略优化。
- [Training LMs with Human Feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2022); Volume: 35; Pages: 27735-27748; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  一篇具有里程碑意义的论文，展示了如何使用PPO通过人类偏好微调大型语言模型，构成了LLM中RLHF的基础。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本关于强化学习的标准教科书，提供了策略梯度方法、演员-评论家架构和其他核心RL概念的基础理解。
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel (2015)
  Journal: International Conference on Learning Representations (ICLR 2016); DOI: [10.48550/arXiv.1506.02438](https://doi.org/10.48550/arXiv.1506.02438)
  这篇论文介绍了广义优势估计 (GAE)，一种减少策略梯度估计方差的方法，对PPO训练的稳定性至关重要。

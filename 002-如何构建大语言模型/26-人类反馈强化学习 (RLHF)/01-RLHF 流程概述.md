# RLHF 流程概述

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-26-reinforcement-learning-human-feedback-rlhf/rlhf-pipeline-overview)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管监督微调 (fine-tuning)（SFT）为遵循指令提供了良好基础，但要使模型更好地符合复杂的人类偏好，通常需要人类反馈强化学习 (reinforcement learning)（RLHF）。RLHF 过程通过训练模型，使其基于人类判断为更好的输出，从而能够针对有益性和无害性等品质进行优化，这些品质仅通过监督示例难以把握。标准的 RLHF 流程通常包含三个主要阶段：

1. **模型初步准备（预训练 (pre-training)与 SFT）：** 从一个有能力的预训练语言模型开始。然后，使用高质量的提示-响应对数据集（如第 25 章所述）进行监督学习 (supervised learning)微调。SFT 步骤使模型适应遵循指令，并以期望的风格和格式生成响应。这个 SFT 模型将作为后续 RLHF 阶段的起点。
2. **奖励模型 (RM) 训练：** 核心思想是训练一个可以预测人类偏好的模型。

   - **数据收集：** 选择一组不同类型的提示。对于每个提示，使用 SFT 模型（或多个模型变体）生成多个响应。然后向人类标注者展示这些响应的成对（或更多）结果，并要求他们根据有益性、准确性或安全性等标准选择他们更喜欢哪一个。
   - **RM 架构：** 奖励模型（$RM$）通常是另一个语言模型（常基于 SFT 模型或较小的预训练模型进行初始化），其最后一层被替换为一个线性层，输出一个表示预测偏好分数的单一标量值。它以提示和响应作为输入并输出此分数。
   - **训练目标：** 奖励模型使用收集到的成对偏好数据进行训练。给定一个提示 $p$ 和两个响应 $y_w$（胜者）和 $y_l$（败者），其中人类更偏好 $y_w$ 而非 $y_l$，奖励模型被训练为给 $y_w$ 分配比 $y_l$ 更高的分数。常用的损失函数 (loss function)是成对排序损失：
     
     $$
     \mathcal{L}_{RM} = - \mathbb{E}_{(p, y_w, y_l) \sim D} [\log(\sigma(r_{\theta}(p, y_w) - r_{\theta}(p, y_l)))]
     $$
     
     这里，$r_{\theta}(p, y)$ 是奖励模型（参数 (parameter)为 $\theta$）对提示 $p$ 和响应 $y$ 输出的标量分数，$\sigma$ 是 sigmoid 函数，$D$ 是人类偏好数据集。这个目标最大化了首选响应 $y_w$ 获得更高分数的概率。
3. **强化学习 (RL) 微调：** 训练好的奖励模型现在作为人类偏好的代表，提供奖励信号，使用强化学习算法（通常是近端策略优化，PPO）进一步微调 SFT 模型。

   - **RL 设置：** SFT 模型在 RL 框架中充当初始*策略*（$\pi_{SFT}$）。*动作空间*包含模型可能生成的令牌，而*状态*是迄今为止生成的令牌序列。
   - **优化循环：** 该过程迭代地从数据集中采样提示 $p$。当前的 RL 策略（$\pi_{RL}$）生成一个响应 $y$。奖励模型将奖励 $r = r_{\theta}(p, y)$ 分配给生成的响应。然后，PPO 算法使用此奖励信号更新策略 $\pi_{RL}$ 的权重 (weight)。
   - **KL 散度惩罚：** RL 微调中的一个重要问题是，策略可能会学习生成最大化奖励模型分数但显著偏离原始 SFT 模型分布的输出，这可能导致无意义或重复的文本（“奖励作弊”）。为了缓解这种情况，RLHF 中的 PPO 在奖励函数或目标中加入一个 Kullback-Leibler (KL) 散度惩罚项。目标是在最大化奖励模型分数的同时，与初始 SFT 策略保持接近：
     
     $$
     \text{目标} = \mathbb{E}_{(p,y) \sim \pi_{RL}} [r_{\theta}(p, y) - \beta \cdot \text{KL}(\pi_{RL}(\cdot|p) || \pi_{SFT}(\cdot|p))]
     $$
     
     项 $\text{KL}(\pi_{RL}(\cdot|p) || \pi_{SFT}(\cdot|p))$ 衡量了 RL 策略和原始 SFT 策略对提示 $p$ 预测的令牌分布之间的散度。超参数 (hyperparameter) $\beta$ 控制此惩罚的强度，防止 RL 策略偏离 SFT 模型学习到的分布过远，从而保持连贯性和语言质量。

以下图表展示了这一多阶段过程：

> 标准 RLHF 流程的三个阶段：监督微调 (SFT)、基于人类偏好的奖励模型 (RM) 训练，以及由奖励模型和针对原始 SFT 模型的 KL 惩罚引导的强化学习 (PPO) 微调。

这个流程使模型能够从比较性人类反馈中学习，改进其生成与期望属性更一致的响应的能力，比仅通过 SFT 达到的效果更好。接下来的部分将详细说明收集偏好数据、训练奖励模型以及实施 PPO 微调步骤的实际方面。

## 参考资料

- [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/pdf/1706.03741.pdf) — Paul F. Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei (2017)
  Journal: Advances in Neural Information Processing Systems 30; Volume: 30; DOI: [10.48550/arXiv.1706.03741](https://doi.org/10.48550/arXiv.1706.03741)
  介绍了通过人类比较学习奖励函数，然后使用该奖励函数训练深度强化学习智能体的方法。这项工作为现代RLHF奠定了重要基础。
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  描述了近端策略优化 (PPO) 算法，这是一种高效且广泛使用的强化学习策略梯度方法，它是RLHF中RL微调阶段的核心算法。
- [Training language models to follow instructions with human feedback](https://papers.nips.cc/paper_files/paper/2022/file/b1efde53ad958a0e176b618998d5c414-Paper-Conference.pdf) — Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: Advances in Neural Information Processing Systems 35; Volume: 35; Pages: 27730–27744
  这篇论文介绍了InstructGPT模型，并详细阐述了完整的三阶段RLHF流程--包括监督微调（SFT）、使用人类偏好训练奖励模型（RM），以及使用PPO和KL惩罚进行强化学习微调--用于对齐大型语言模型。

---

[上一节](../25-%E5%AF%B9%E9%BD%90%E5%BE%AE%E8%B0%83%EF%BC%9A%E7%9B%91%E7%9D%A3%E5%BC%8F%E5%BE%AE%E8%B0%83%20%28SFT%29/06-%E8%AF%84%E4%BC%B0SFT%E6%A8%A1%E5%9E%8B%E5%AF%B9%E9%BD%90%E7%9B%AE%E6%A0%87.md) · [下一节](02-%E6%94%B6%E9%9B%86%E4%BA%BA%E7%B1%BB%E5%81%8F%E5%A5%BD%E6%95%B0%E6%8D%AE.md)

# 人类反馈强化学习 (RLHF) 介绍

来源：[原文](https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-5-advanced-fine-tuning-strategies/intro-rlhf)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然监督微调 (fine-tuning) (SFT) 擅长训练模型基于示例来遵循特定指令，但要实现与人类更细致的偏好（如帮助性、诚实性和无害性）的对齐 (alignment)，通常需要另一种方法。这些预期品质很难仅通过输入-输出示例来捕捉。人类反馈强化学习 (reinforcement learning) (RLHF) 便在此发挥作用。RLHF 提供了一个基于人类表达的偏好来优化语言模型的框架，从而从简单的模仿转向对预期行为特点进行优化。

RLHF 是一个多阶段过程，旨在利用人类判断中获得的奖励来微调语言模型。它使我们能够比单独使用 SFT 更直接地引导模型的行为。主要思想是首先训练一个独立的“奖励模型”，该模型学习预测人类偏好的回应，然后将此奖励模型用于强化学习循环中，以更新语言模型本身。

典型的 RLHF 工作流程包含三个主要阶段：

1. **初始模型准备 (监督微调 - SFT)：** 尽管不严格属于 RLHF 的一部分，该过程通常从一个预训练 (pre-training)的语言模型开始，该模型已在一个高质量的指令数据集上进行了监督微调。这一初始 SFT 步骤提供了一个强大的基线模型，$\pi_{SFT}$，能够很好地遵循指令。该模型作为后续 RLHF 阶段的起点。
2. **奖励模型 (RM) 训练：**

   - **目标：** 目标是训练一个由 $\phi$ 参数 (parameter)化的模型 $r_\phi(x, y)$，该模型接收提示 $x$ 和生成的回答 $y$ 作为输入，并输出一个标量分数，代表人类对该回答的偏好。分数越高表示越受偏好的回答。
   - **数据收集：** 这需要收集人类偏好数据。通常，对于给定的提示 $x$，SFT 模型会生成多个回答 ($y_1, y_2, ..., y_k$)。随后，人类标注者被要求将这些回答从最好到最差进行排序，或者更常见的是进行成对比较，选择偏好的回答 ($y_w$，胜者) 而非另一个 ($y_l$，败者)。这便创建了一个包含偏好三元组 $(x, y_w, y_l)$ 的数据集 $D$。
   - **训练：** 奖励模型通常使用 SFT 模型的权重 (weight)进行初始化，其中最终的 token 预测层被替换为一个输出标量分数的回归头。然后它在偏好数据集 $D$ 上进行训练。一个常见的目标函数旨在最大化偏好和被拒绝回答之间的分数差异，通常使用基于 Bradley-Terry 模型的成对比较损失：
     
     $$
     L(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim D} [\log(\sigma(r_\phi(x, y_w) - r_\phi(x, y_l)))]
     $$
     
     这里，$\sigma$ 是 Sigmoid 函数。这种损失函数 (loss function)促使奖励模型 $r_\phi$ 为胜出回答 $y_w$ 分配一个比落败回答 $y_l$ 明显更高的分数。
3. **RL 微调 (策略优化)：**

   - **目标：** 最后阶段使用强化学习来微调 SFT 语言模型（现在被视为策略，$\pi_\theta$），使其生成能够最大化经过训练的奖励模型 $r_\phi$ 预测的预期奖励的回答。
   - **流程：** RL 循环按以下方式运行：
     a. 从数据集中采样一个提示 $x$（通常是 SFT 数据集或自定义提示集）。
     b. 当前的 LLM 策略 $\pi_\theta(y|x)$ 生成一个回答 $y$。
     c. 奖励模型 $r_\phi(x, y)$ 为生成的回答分配一个奖励分数。
     d. 此奖励信号用于更新 LLM 策略 $\pi_\theta$ 的参数 $\theta$。
   - **优化算法 (PPO)：** 近端策略优化 (PPO) 是此阶段最常用的算法。直接最大化奖励 $r_\phi(x, y)$ 可能导致策略 $\pi_\theta$ 生成重复或无意义的文本，从而“欺骗”奖励模型（奖励作弊），或者与预训练和 SFT 期间学到的连贯语言生成相距过远。PPO 通过引入一个约束来解决此问题，该约束惩罚当前策略 $\pi_\theta$ 与原始 SFT 策略 $\pi_{SFT}$ 之间的大幅偏离。PPO 优化的目标函数通常如下所示：
     
     $$
     \text{目标}(\theta) = \mathbb{E}_{x \sim D_{prompt}, y \sim \pi_\theta(y|x)} [r_\phi(x, y) - \beta \text{KL}(\pi_\theta(y|x) || \pi_{SFT}(y|x))]
     $$
     
     这里，$\mathbb{E}$ 表示期望，$D_{prompt}$ 是提示的分布，$\text{KL}$ 代表策略之间的 Kullback-Leibler 散度，$\beta$ 是一个控制 KL 惩罚强度的系数。此目标函数促使策略 $\pi_\theta$ 在从 $r_\phi$ 获得高奖励的同时，与初始 SFT 模型 $\pi_{SFT}$ 定义的分布保持相对接近，从而保持通用的语言能力并减轻灾难性遗忘或策略崩溃。

整个 RLHF 过程可以如下所示：

> 人类反馈强化学习 (RLHF) 的工作流程，展示了从初始 SFT 模型到基于人类偏好训练奖励模型，最终使用 PPO 微调受奖励模型引导的语言模型策略的进展。

RLHF 代表了一种强大的技术，用于将 LLM 与难以仅通过示例来具体说明的复杂、主观或安全关键的人类价值观对齐。然而，与 SFT 相比，它带来了显著的复杂性。收集高质量的人类偏好数据成本高昂且耗时。最终模型的性能高度依赖于奖励模型的质量和校准，RL 训练过程本身可能不稳定，并且对超参数 (hyperparameter)选择很敏感。奖励作弊等问题，即策略在奖励模型中找到漏洞，需要仔细监控和缓解，这通常涉及 PPO 中的 KL 惩罚项。

在高级微调策略方面，RLHF 通常在初始指令调优 (SFT) 之后使用，以进一步改进模型行为，使其更有帮助，更不容易生成有害内容，并且更好地与用户意图对齐。它补充了多任务学习等技术，通过提供一种机制，针对人类偏好而非特定任务格式定义整体行为目标进行优化。

## 参考资料

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: arXiv preprint arXiv:2203.02155; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  本文介绍了InstructGPT，它是将RLHF应用于大型语言模型，使其与人类指令和偏好对齐的一个突出范例。
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  这是提出近端策略优化（PPO）算法的原始论文，该算法广泛用于RLHF中的强化学习微调阶段。
- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://doi.org/10.48550/arXiv.2204.05862) — Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, Jared Kaplan (2022)
  Journal: arXiv; DOI: [10.48550/arXiv.2204.05862](https://doi.org/10.48550/arXiv.2204.05862)
  本文探讨了如何利用人类反馈训练大型语言模型使其更具帮助性和无害性，解决了重要的对齐难题。

---

[上一节](03-%E5%87%8F%E8%BD%BB%E7%81%BE%E9%9A%BE%E6%80%A7%E9%81%97%E5%BF%98.md) · [下一节](05-%E5%A5%96%E5%8A%B1%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83.md)

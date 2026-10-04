---
course: "rlhf-reinforcement-learning-human-feedback"
chapter: "rlhf-foundations-alignment"
lesson: "intro-rlhf-process"
sourceId: 5091
sourceUrl: "https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-1-rlhf-foundations-alignment/intro-rlhf-process"
title: "RLHF流程简介"
description: "RLHF的三大主要阶段：SFT、奖励模型构建和RL微调。"
order: 4
plots: []
sourceHash: "7e693807897d09d45e964e8a374c9462e746728a87f9e7ecd4a4f282f41179d6"
sourceCorrections: []
---

强化学习 (reinforcement learning)人类反馈（RLHF）是一种旨在解决对齐 (alignment)难题并克服纯监督方法局限性的标准流程。这种多阶段方法旨在引导大型语言模型（LLMs）生成更符合人类偏好和指示的输出。它通常包含三个不同阶段：

1. **监督微调 (fine-tuning)（SFT）：** 这一初始阶段使用精心整理的高质量提示和期望响应数据集，将预训练 (pre-training)的LLM适应到目标应用方向或风格。可以将其看作是教授模型预期的基本格式和风格。其成果是一个“SFT模型”，作为后续强化学习阶段的起点。尽管有帮助，但单独的SFT往往难以涵盖人类偏好的全部范围，尤其是在处理复杂指示、安全性或像实用性这类细微特点时。我们之前讨论过它的局限性。
2. **奖励模型构建（RM）：** 由于我们希望根据人类偏好优化模型，但又因为成本和延迟，在强化学习训练期间无法与人类进行交互查询，所以我们首先训练一个单独的模型来*预测*人类偏好。这就是奖励模型（RM）。为了训练它，我们收集人类反馈数据。通常，针对一个给定的提示，会向人类展示由SFT模型（或其他模型）生成的多个响应，并要求他们进行排序或选择最佳响应。这种偏好数据（例如，其中一个响应优于另一个的响应对）用于训练RM。RM以提示和生成的响应作为输入，输出一个标量*奖励*分数，该分数理想情况下与人类偏好该响应的可能性相关。
3. **强化学习微调（PPO）：** 在最后阶段，SFT模型（现在作为初始*策略*）使用强化学习进行进一步微调。环境包括接收提示、生成响应，并从RM获得奖励。目标是调整策略模型的参数 (parameter)，以最大化RM预测的预期奖励，有效地教导模型生成RM评分高的响应（从而人类也可能偏好）。近端策略优化（PPO）常在此处使用。此阶段一个重要方面是在更新后的策略和原始SFT策略之间施加约束，通常使用Kullback-Leibler（KL）散度。这种KL惩罚防止强化学习过程与SFT模型学到的知识和分布产生过大偏差，有助于保持语言连贯性，并避免策略以不切实际的方式为奖励模型“过度优化”（奖励作弊）。

这个序列构成了RLHF的核心流程。每个阶段都在前一阶段的基础上建立，从一个通用能力模型开始，训练一个偏好预测器，最后针对该预测器优化模型。

> 图示了RLHF流程的三个主要阶段：监督微调（SFT）、奖励模型构建（RM）以及使用PPO的强化学习（RL）微调。该图展示了模型和数据在这些阶段之间的流动。

后续章节将详细介绍每个阶段的实现细节、挑战以及不同变体，首先将更仔细地审视SFT阶段。理解这个整体结构为后续的技术探讨提供了必要的背景信息。

## 参考资料

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  这篇基础论文概述了通过人类反馈强化学习（RLHF）使大型语言模型与人类指令和偏好对齐的完整流程，详细介绍了监督微调、奖励建模和基于PPO的强化学习。
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍了近端策略优化（PPO）算法，这是一种策略梯度方法，因其稳定性和效率而在RLHF的强化学习微调阶段被广泛采用。
- [Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2009.01325](https://doi.org/10.48550/arXiv.2009.01325)
  一项早期重要工作，展示了RLHF流程在特定语言任务中的应用：训练模型生成与人类偏好对齐的摘要。

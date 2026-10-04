---
course: "fine-tuning-adapting-large-language-models"
chapter: "advanced-fine-tuning-strategies"
lesson: "challenges-advanced-adaptation"
sourceId: 3062
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-5-advanced-fine-tuning-strategies/challenges-advanced-adaptation"
title: "高级适应中的挑战"
description: "讨论高级微调中的稳定性问题、奖励欺骗和对齐复杂性。"
order: 7
plots: []
sourceHash: "526d72c6b21a4140deb925daa39e730da4504afd818ba991db4de5fd5991b22f"
sourceCorrections: []
---

多任务学习、序列适应和基于人类反馈的强化学习 (reinforcement learning)（RLHF）等技术为定制大型语言模型提供了方法。然而，有效实施这些方法会遇到一些重要障碍。这些高级方法在实现难度上通常高于标准监督微调 (fine-tuning)。

### 多任务学习的难点

同时训练一个模型以实现多个目标，会带来独特的困难：

- **负向迁移：** 一种常见问题，指学习一个任务对另一个任务的表现产生负面影响。这通常发生在任务具有冲突的梯度信号或需要根本不同的内部表示时。优化任务A可能会积极地降低任务B所需的权重 (weight)。
- **任务平衡：** 确定每个任务损失函数 (loss function)的适当权重并非易事。简单的平均可能会导致模型优先处理较容易的任务或数据集较大的任务，而忽略其他任务。更复杂的加权方案需要仔细调整和理解任务间的关联。
- **数据整理：** 组织有效的多任务数据集需要仔细考虑任务分布、数据质量以及任务间可能存在的偏差。在不引入混淆因素的情况下确保每个任务都有充分的代表性，是一项重要的、数据工程方面的难题。

### 序列适应与灾难性遗忘

序列适应模型，例如为新任务微调 (fine-tuning)一个已专业化的模型，会带来遗忘先前学习信息的风险：

- **遗忘缓解不完全：** 弹性权重 (weight)整合（EWC）或使用任务专用适配器等方法旨在保留先前的知识，但它们并非万无一失。遗忘仍然可能发生，尤其是在适应非常不同的领域或任务类型时。这些方法的有效性通常取决于对参数 (parameter)重要性的准确估计或精心设计的适配器。
- **计算和存储开销：** 旨在应对遗忘的方法通常会带来额外成本。EWC 需要计算和存储费雪信息矩阵。排练方法需要存储来自先前任务的示例，增加了存储需求。参数隔离技术增加了参数数量，可能导致模型大小和推理 (inference)延迟增加。

### 基于人类反馈的强化学习 (reinforcement learning)（RLHF）中的复杂之处

RLHF是一种使模型行为与人类偏好对齐 (alignment)的强大技术，但其多阶段过程容易遇到一些困难：

- **奖励模型问题：** 训练奖励模型 $r_\phi(x, y)$ 是RLHF的基础，但具有挑战性：

  - *偏好数据质量：* 人类偏好可能存在噪声、不一致、主观性，或在标注者之间差异很大。在配对比较 $(x, y_w, y_l)$ 中准确捕获复杂的偏好是困难的。
  - *可扩展性：* 获取大量高质量的偏好数据集耗时费力且成本高昂。
  - *奖励模型准确性：* 学习到的奖励模型只是真实人类偏好的一种近似。它可能包含偏差，或未能正确泛化到未见过的输出，导致策略在优化过程中偏离方向。
- **策略优化不稳定性：** 使用PPO等强化学习算法对语言模型 $\pi_\theta(y|x)$ 进行微调 (fine-tuning)会带来优化难题：

  - *超参数 (parameter) (hyperparameter)敏感性：* 强化学习算法对学习率、KL散度惩罚系数（$\beta$）、批大小和PPO特有参数（例如，裁剪比率 $\epsilon$）等超参数非常敏感。找到稳定的设置通常需要大量的实验。
  - *奖励欺骗：* 策略 $\pi_\theta$ 可能会找到从奖励模型 $r_\phi$ 中获取高分的方法，而这些方法并不对应真正偏好的行为。这可能表现为重复或无意义的输出，它们利用了奖励函数中的漏洞。例如，如果模型偏好长度或特定关键词，生成过于冗长但简单的句子可能会获得奖励。
  - *寻找更好方案的困难：* 在不破坏策略稳定性的前提下，有效地在可能的语言输出的广大空间中寻找更好的解决方案，本身就是复杂之事。

> 基于人类反馈的强化学习循环中可能出现的故障模式。在收集纯净的偏好数据、准确建立这些偏好模型以及在没有奖励欺骗等意外副作用的情况下稳定优化策略时，都会出现难题。

- **对齐代价：** 通过RLHF优化对齐的过程有时会导致模型在其他期望属性上的表现下降，例如创造力、复杂推理 (inference)，甚至在标准基准测试上的表现。管理这种权衡需要仔细调整RLHF过程。
- **计算资源：** RLHF计算密集。它涉及训练多个大型模型，并在优化循环中执行频繁的推理步骤，需要大量的GPU资源和时间。

### 普遍难题

高级适应方法面临更普遍的挑战：

- **评估复杂性：** 评估多任务学习、持续学习或RLHF对齐 (alignment)的成功与否是困难的。标准指标通常无法捕捉指令遵循、偏好对齐或跨不同任务的知识保留的细节。仔细的评估通常需要精心设计的测试套件、人工评估或专门的基准测试。
- **可复现性：** 复杂的数据管道、多阶段训练过程（例如，SFT -> 奖励模型构建 -> PPO）以及对超参数 (parameter) (hyperparameter)的敏感性相结合，使得复现高级适应技术的结果具有挑战性。设置上的微小差异可能导致结果的明显不同。

应对这些挑战需要周密的规划、大量的实验、健全的评估规程，以及通常大量计算资源。理解这些潜在的困难对于设定实际的预期和设计有效的策略非常重要，尤其是在应用高级适应方法时。

## 参考资料

- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: arXiv preprint arXiv:2203.02155; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  这篇基础论文介绍了用于遵循指令的语言模型的RLHF，讨论了奖励建模、策略优化以及将LLM与人类偏好对齐的实际实现中的挑战。
- [Overcoming catastrophic forgetting in neural networks](https://www.pnas.org/doi/10.1073/pnas.1611835114) — James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, Raia Hadsell (2017)
  Journal: Proceedings of the National Academy of Sciences; Publisher: National Academy of Sciences; Volume: 114; Pages: 3521-3526; DOI: [10.1073/pnas.1611835114](https://doi.org/10.1073/pnas.1611835114)
  这篇论文介绍了弹性权重巩固（EWC），一种减轻灾难性遗忘的方法，直接解决了顺序适应中的一个关键挑战。
- [An Overview of Multi-Task Learning in Deep Neural Networks](https://arxiv.org/abs/1706.05098) — Sebastian Ruder (2017)
  Journal: arXiv preprint arXiv:1706.05098; DOI: [10.48550/arXiv.1706.05098](https://doi.org/10.48550/arXiv.1706.05098)
  这项全面的调查详细分析了多任务学习，包括负迁移、任务平衡和优化挑战的讨论。

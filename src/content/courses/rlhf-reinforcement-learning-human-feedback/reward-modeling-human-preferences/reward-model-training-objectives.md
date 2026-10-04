---
course: "rlhf-reinforcement-learning-human-feedback"
chapter: "reward-modeling-human-preferences"
lesson: "reward-model-training-objectives"
sourceId: 5127
sourceUrl: "https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-3-reward-modeling-human-preferences/reward-model-training-objectives"
title: "奖励模型训练目标"
description: "用于训练奖励模型的数学表述和损失函数。"
order: 5
plots: []
sourceHash: "cdc4dd695e9847c497b221116187fc19b6c137dd41ee9d36702029761f5790d5"
sourceCorrections: []
---

为了有效训练奖励模型 ($RM$)，需要一个学习目标，使模型的输出与收集到的人类偏好数据保持一致。这些数据通常包含给定提示 $x$ 的成对比较，表明对“获胜”响应 $y_w$ 的偏好超过“失败”响应 $y_l$。目标是训练 $RM$ 为 $y_w$ 分配比 $y_l$ 更高的标量分数。

标准做法是从概率选择模型（如Bradley-Terry模型）中获得启发。我们将给定提示 $x$ 时人类偏好 $y_w$ 胜过 $y_l$ 的概率建模为奖励模型对每个响应评分之差的函数。具体来说，我们使用逻辑函数（S形函数，$\sigma$）将分数差映射为概率：


$$
P(y_w \succ y_l | x) = \sigma(RM_\theta(x, y_w) - RM_\theta(x, y_l))
$$


此处：

- $RM_\theta(x, y)$ 是奖励模型（参数 (parameter)为 $\theta$）在给定提示 $x$ 的情况下，对响应 $y$ 分配的标量分数。
- $y_w \succ y_l$ 表示响应 $y_w$ 比响应 $y_l$ 更受偏好。
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ 是S形函数，它将其输入压缩到 (0, 1) 的范围，适合表示概率。

训练的目标是找到参数 $\theta$，使观察到数据集中 $D = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}$ 所表达偏好的可能性最大化。最大化可能性等同于最小化偏好的负对数可能性。对于单个偏好对 $(x, y_w, y_l)$，负对数可能性损失为：


$$
\mathcal{L}(\theta; x, y_w, y_l) = -\log P(y_w \succ y_l | x)
$$


代入概率表达式，我们得到：


$$
\mathcal{L}(\theta; x, y_w, y_l) = -\log \sigma(RM_\theta(x, y_w) - RM_\theta(x, y_l))
$$


这通常被称为成对逻辑损失。为了训练模型，我们最小化整个偏好数据集 $D$ 上的平均损失：


$$
\mathcal{L}_{total}(\theta) = -\frac{1}{|D|} \sum_{(x, y_w, y_l) \in D} \log \sigma(RM_\theta(x, y_w) - RM_\theta(x, y_l))
$$


使用基于梯度的优化方法（如Adam）最小化此损失函数 (loss function)，会促使奖励模型为更受偏好的响应 $y_w$ 分配比次优响应 $y_l$ 更高的分数。 $RM_\theta(x, y_w) - RM_\theta(x, y_l)$ 的差异越大，该特定对的损失越低，从而推动模型根据人类判断正确地对响应进行排序。

以下图表说明了训练期间单个偏好对的计算流程：

> 计算单个偏好样本 $(x, y_w, y_l)$ 的成对逻辑损失的流程。奖励模型 $RM_\theta$ 在给定提示的情况下，计算两个响应的标量分数。分数之差通过S形函数，此概率的负对数构成该样本的损失贡献。此损失随后用于通过反向传播 (backpropagation)更新模型参数 $\theta$。

这种训练目标直接将成对的人类偏好转化为梯度信号，从而塑造奖励模型。经过良好训练并以此目标优化的奖励模型，为后续强化学习 (reinforcement learning)阶段提供了重要的奖励信号，引导语言模型生成更符合人类偏好的响应。

## 参考资料

- [Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2009.01325](https://doi.org/10.48550/arXiv.2009.01325)
  这是OpenAI早期的一项工作，展示了如何利用人类反馈训练奖励模型，特别针对摘要任务，并采用了类似的成对偏好学习目标。
- [Rank analysis of incomplete block designs. I. The method of paired comparisons](https://doi.org/10.1093/biomet/39.3-4.324) — Ralph Allan Bradley and Milton E. Terry (1952)
  Journal: Biometrika; Publisher: Biometrika Trust; Volume: 39; Pages: 324-345; DOI: [10.1093/biomet/39.3-4.324](https://doi.org/10.1093/biomet/39.3-4.324)
  这篇基础性论文介绍了Bradley-Terry模型，该模型为成对比较提供了概率框架，并作为奖励模型损失函数的理论依据。
- [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1706.03741](https://doi.org/10.48550/arXiv.1706.03741)
  这项开创性工作提出了利用人类反馈（特别是成对比较）来训练通用深度强化学习智能体的奖励模型，早于其在语言模型中的广泛应用。

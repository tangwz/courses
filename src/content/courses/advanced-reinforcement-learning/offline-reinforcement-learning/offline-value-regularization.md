---
course: "advanced-reinforcement-learning"
chapter: "offline-reinforcement-learning"
lesson: "offline-value-regularization"
sourceId: 3491
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-7-offline-reinforcement-learning/offline-value-regularization"
title: "值函数正则化方法"
description: "训练期间对Q函数进行正则化，以防止对分布外动作的过高估计（例如，CQL）。"
order: 9
plots: ["plots/3491-0.json"]
sourceHash: "7d29e6005e421d2ea23dcef3bc869e80caaf86b932b1d59665d0c80acb6120d2"
sourceCorrections: []
---

防止出现问题行为是离线强化学习 (reinforcement learning)中的一个主要挑战。一种缓解此问题的方法侧重于调整值函数本身。值函数正则化 (regularization)方法修改Q函数（或值函数）的学习目标，以阻止对数据集中稀有或缺失的动作给出过于乐观的值估计。通过使值函数对分布外动作更加“保守”，这些方法旨在避免策略改进步骤滥用由分布偏移引起的Q函数不准确性。

这些方法处理的根本问题是*外推误差*。当使用深度神经网络 (neural network)等函数逼近器时，Q学习更新（涉及 $\max_{a'} Q(s', a')$ 操作）很容易对数据集中状态 $s'$ 下从未见过的动作 $a'$ 产生任意高的Q值。发生这种情况是因为函数逼近器没有数据来支撑其对这些未见状态-动作对的估计。如果策略优化基于错误的高Q值选择这些动作，性能会大幅下降。

### 基本思路：惩罚分布外动作的值

值函数正则化 (regularization)技术在训练Q网络所用的标准损失函数 (loss function)（通常是贝尔曼误差）中加入特定的惩罚项。这些惩罚项的目标是压低在生成数据的行为策略下不太可能出现的动作的估计Q值，同时确保数据集中动作的Q值保持准确。

可以这样理解：如果智能体考虑执行一个在当前情况的离线数据中很少（或从未）尝试过的动作，正则化的Q函数应该给它赋予相对较低的值，使其在策略改进或评估时不太可能被选中。

### 保守Q学习 (CQL)

一种重要且有效的值函数正则化 (regularization)算法是**保守Q学习 (CQL)**。CQL直接修改目标函数以学习一个保守的Q函数。其核心思想是确保学习到的Q函数低估数据分布外动作的值，同时准确估计（或可能略微高估）数据分布内动作的值。

CQL通过在标准贝尔曼误差最小化目标中添加正则项来实现此目的。该正则项旨在：

1. 最小化从*某种*分布 $\mu(a|s)$ 采样的动作的Q值（理想情况下应覆盖状态 $s$ 的数据集中和数据分布外的动作）。
2. 最大化直接从数据集 $\mathcal{D}$ 为状态 $s$ 采样的动作的Q值。

CQL目标函数的一种常见形式（加到贝尔曼误差损失 $L_{Bellman}(\theta)$ 中）如下所示：


$$
L_{CQL}(\theta) = \alpha \left[ \mathbb{E}_{s \sim \mathcal{D}, a \sim \mu(a|s)}[Q_\theta(s,a)] - \mathbb{E}_{(s,a) \sim \mathcal{D}}[Q_\theta(s,a)] \right]
$$


这里：

- $\theta$ 代表Q网络的参数 (parameter)。
- $\mathcal{D}$ 是离线数据集。
- $\mu(a|s)$ 是给定状态 $s$ 的动作采样分布。它可以是所有可能动作上的简单均匀分布，也可以从当前学习到的策略中推导出来（例如，按 $\exp(Q_\theta(s,a))$ 的比例采样的动作）。
- 第一项 $\mathbb{E}_{s \sim \mathcal{D}, a \sim \mu(a|s)}[Q_\theta(s,a)]$ 代表采样分布 $\mu$ 下的预期Q值。最小化此项会普遍压低Q值。
- 第二项 $\mathbb{E}_{(s,a) \sim \mathcal{D}}[Q_\theta(s,a)]$ 代表数据集中实际观察到的状态-动作对的预期Q值。最小化此项的*负值*（如在组合损失中所做）会有效地*提高*数据中已见动作的Q值。
- $\alpha \ge 0$ 是一个超参数 (hyperparameter)，用于控制保守程度。更高的 $\alpha$ 值会更积极地压低分布外动作的Q值。

存在不同的CQL变体，主要区别在于如何选择动作分布 $\mu(a|s)$ 或如何构建最小化项（例如，使用LogSumExp公式以获得更好的稳定性和梯度属性）。主要观点不变：惩罚与数据支持不足的动作相关的Q值，以避免在贝尔曼更新期间出现过高估计。



![CQL正则化对Q值的影响](plots/3491-0.json)



> 此图显示了CQL如何倾向于降低分布外 (OOD) 动作的Q值估计，同时可能略微提高或保持离线数据集中存在动作的值。

### 与策略约束的比较

值函数正则化 (regularization)（如CQL）和策略约束（如BCQ）提供不同的机制来达到类似的目标：安全的离线策略学习。

- **策略约束：** 直接限制学习策略可用的动作空间，通常通过基于行为策略的生成模型过滤动作或添加行为模仿项。Q函数仍可能遭受外推误差，但策略被阻止选择那些有问题动作。
- **值函数正则化：** 修改Q值本身，使其对OOD动作固有地较低。策略优化过程（例如，基于Q值的贪婪选择）自然会避开这些动作，因为其估计值很低。它不明确限制策略的动作空间，但隐式引导它避开无支持区域。

在实践中，像CQL这样的值函数正则化方法通常表现出强大性能，如果值函数学习到有意义的模式，有可能使策略在略微超出数据集的严格支持范围的情况下也能泛化，同时仍能控制外推误差的风险。

### 优点和缺点

**优点：**

- **直接处理价值高估：** 解决了离线Q学习不稳定性的根本原因。
- **隐式约束：** 避免明确的策略约束，这有时可能过于严格，阻碍找到需要与行为策略略有偏离的最优策略。
- **强大的经验表现：** CQL及相关方法在许多离线强化学习 (reinforcement learning)基准测试中展现了最先进的结果。

**缺点：**

- **超参数 (parameter) (hyperparameter)敏感性：** 性能对正则化 (regularization)权重 (weight) $\alpha$ 的选择敏感。设置过低可能无法阻止高估，而设置过高可能使Q函数过于悲观，从而阻碍学习。
- **采样复杂性：** 实现最小化项 $\mathbb{E}_{s \sim \mathcal{D}, a \sim \mu(a|s)}[Q_\theta(s,a)]$ 需要从 $\mu(a|s)$ 中采样动作，这可能增加计算开销，尤其在连续动作空间中。
- **可能存在低估：** 尽管旨在防止高估，过度激进的正则化可能导致对最优OOD动作的低估，尽管这通常被认为比高估的危害小。

### 实现注意事项

实现CQL等方法时：

- 仔细考虑最小化项中动作采样分布 $\mu(a|s)$ 的选择。均匀采样很简单，但可能无法集中处理最有问题的、高值的OOD动作。从当前策略采样可能更有效，但会增加复杂性。
- 超参数 (parameter) (hyperparameter) $\alpha$ 通常需要针对每个环境或数据集进行调整。它通常平衡标准贝尔曼误差最小化和保守性目标。
- 确保计算稳定，尤其是在处理指数项时（例如，在LogSumExp变体或基于策略的采样中）。

值函数正则化 (regularization)，尤其是通过CQL等方法，提供了一套强大的技术，通过直接管理与值函数外推相关的风险，从而从离线数据中有效学习。它补充了策略约束方法，提供了一种不同但相关的策略，以应对离线强化学习 (reinforcement learning)中分布偏移的根本难题。

## 参考资料

- [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/pdf/2006.04779.pdf) — Aviral Kumar, Aurick Zhou, George Tucker, Sergey Levine (2020)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 33; Pages: 1179–1191; DOI: [10.48550/arXiv.2006.04779](https://doi.org/10.48550/arXiv.2006.04779)
  介绍了保守Q学习（CQL），一种用于离线强化学习中价值正则化的方法。
- [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643) — Sergey Levine, Aviral Kumar, George Tucker, Justin Fu (2020)
  Journal: Foundations and Trends in Machine Learning; Volume: 13; Pages: 316-528; DOI: [10.48550/arXiv.2005.01643](https://doi.org/10.48550/arXiv.2005.01643)
  离线强化学习的综述，涵盖了包括价值正则化技术在内的各种方法。
- [IQL: Implicit Q-Learning for Offline Reinforcement Learning](https://proceedings.mlr.press/v139/kostrikov21a.pdf) — Ilya Kostrikov, Ashish Kumar, Sergey Levine (2021)
  Journal: International Conference on Machine Learning (ICML); Publisher: PMLR; Volume: 139; Pages: 6126-6142; DOI: [10.48550/PMLR.V139.KOSTRIKOV21A](https://doi.org/10.48550/PMLR.V139.KOSTRIKOV21A)
  介绍了隐式Q学习（IQL），一种离线强化学习算法，通过避免显式价值最大化，间接处理高估错误。

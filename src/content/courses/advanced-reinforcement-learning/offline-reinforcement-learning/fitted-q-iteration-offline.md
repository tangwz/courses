---
course: "advanced-reinforcement-learning"
chapter: "offline-reinforcement-learning"
lesson: "fitted-q-iteration-offline"
sourceId: 3488
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-7-offline-reinforcement-learning/fitted-q-iteration-offline"
title: "拟合Q迭代（FQI）方法"
description: "在离线设置中迭代地拟合Q函数到数据。"
order: 6
plots: []
sourceHash: "d5136fd6ed424577897cadf3dba924c356520d92884297b937df313ce7253905"
sourceCorrections: []
---

拟合Q迭代（FQI）是一种早期且直观的方法，用于从固定数据集中学习，它借鉴了在线学习中的思想。FQI将Q学习和值迭代的主要思想扩展到离线、批量设置中。FQI不是一次更新一个Q值，而是将问题视为一系列监督学习 (supervised learning)任务，在每一步都使用全部可用数据集 $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^N$。

基本目标保持不变：估计最优动作值函数 $Q^*(s, a)$。FQI通过迭代实现这一点。我们从一个初始Q函数估计开始，通常简单地设为所有 $s, a$ 的 $\hat{Q}_0(s, a) = 0$。然后，对于每次迭代 $k = 1, 2, \dots, K$：

1. **生成目标值：** 使用*上一次*迭代的Q函数估计 $\hat{Q}_{k-1}$，我们为静态数据集 $\mathcal{D}$ 中存在的每个状态-动作对 $(s_i, a_i)$ 计算目标值 $y_i$。目标值根据贝尔曼最优性方程得出，使用观察到的奖励 $r_i$ 和从下一个状态 $s'_i$ 可达到的估计最大值：

   
   $$
   y_i = r_i + \gamma \max_{a'} \hat{Q}_{k-1}(s'_i, a')
   $$
   

   这就创建了一个目标数据集 $\{( (s_i, a_i), y_i )\}_{i=1}^N$。请注意，目标 $y_i$ 使用*固定*数据 $(r_i, s'_i)$，但依赖于*上一次*的Q函数估计 $\hat{Q}_{k-1}$。
2. **监督学习：** 我们训练一个函数逼近器（我们新的Q函数估计 $\hat{Q}_k$），以使其预测 $\hat{Q}_k(s_i, a_i)$ 与计算出的目标值 $y_i$ 之间的差异最小化。这实际上是针对步骤1中生成的数据集对的标准回归问题。目标通常是均方误差（MSE）：

   
   $$
   \hat{Q}_k = \arg \min_Q \sum_{i=1}^N \left( Q(s_i, a_i) - y_i \right)^2
   $$
   
   
   $$
   \hat{Q}_k = \arg \min_Q \sum_{i=1}^N \left( Q(s_i, a_i) - \left(r_i + \gamma \max_{a'} \hat{Q}_{k-1}(s'_i, a')\right) \right)^2
   $$
   
3. **重复：** 新训练的 $\hat{Q}_k$ 成为下一次迭代 ($k+1$) 中生成目标值的输入。这个过程重复预设的迭代次数 $K$，或者直到Q函数估计的变化足够小。

### FQI中的函数逼近器

FQI的一个重要特点是其在选择 $Q(s, a)$ 的函数逼近器方面的灵活性。由于每次迭代都归结为一个监督回归任务，因此可以使用任何合适的回归器。从历史上看，非参数 (parameter)方法如树集成（随机森林、梯度提升树）因其在无需大量超参数 (hyperparameter)调整的情况下有效学习复杂函数而被频繁使用。然而，深度神经网络 (neural network)也可以被用于将FQI转变为“神经拟合Q迭代”（NFQ），这在某种程度上模糊了与DQN的界限，但保留了其独特的迭代批量拟合过程。

### FQI的优点

- **简洁性：** 它直接将熟悉的贝尔曼更新方法应用于批量设置。
- **使用监督学习 (supervised learning)：** 它允许使用强大的、现成的回归算法。
- **纯离线：** 它完全基于提供的静态数据集运行，无需进一步与环境交互。

### 主要问题：分布偏移

尽管FQI具有吸引力，但在离线设置中，它在实践中常常遇到困难，正是因为**分布偏移**。主要问题在于目标计算步骤：$y_i = r_i + \gamma \max_{a'} \hat{Q}_{k-1}(s'_i, a')$。

- **最大化操作符：** $\max_{a'}$ 项需要为下一个状态 $s'_i$ 中可能存在的许多动作 $a'$ 评估Q函数 $\hat{Q}_{k-1}$。
- **分布外动作：** 数据集 $\mathcal{D}$ 是由特定行为策略 $\pi_b$ 生成的。最大化 $\hat{Q}_{k-1}(s'_i, a')$ 的动作 $a'$ 极有可能*不同于*策略 $\pi_b$ 在状态 $s'_i$ 中通常采取的动作。这些最大化动作在状态 $s'_i$ 的训练数据中可能很少出现或完全缺失。
- **误差传播：** 结果是，对于这些分布外（OOD）动作，$\hat{Q}_{k-1}(s'_i, a')$ 的估计可能极不准确。标准监督学习 (supervised learning)回归器对其远离训练数据分布的输入输出不提供任何保证。目标值 $y_i$ 中的这些误差随后融入到下一个Q函数 $\hat{Q}_k$ 的训练目标中。因此，误差可能会在迭代中传播和累积，可能导致收敛失败或对OOD动作的Q值明显高估，从而产生较差的最终策略。

考虑一个此问题的简单可视化：

> FQI过程依赖于使用前一个Q函数 ($Q_{k-1}$) 估计下一个状态 ($s'_i$) 中的最大Q值。然而，产生此最大值的动作 $a'$ 相对于数据集 $\mathcal{D}$ 中状态 $s'_i$ 处存在的动作而言，可能是分布外的。这可能导致不可靠的估计值反馈回目标计算，从而引起误差传播。

因此，FQI是一个重要的基准方法，但根本上缺少明确应对分布偏移问题的机制。它基于一个隐式且通常不正确的假设运作，即函数逼近器即使对数据中未见或稀疏的状态-动作对也能较好地泛化。这一局限性促使了专门为离线设置设计的方法的开发，例如我们接下来将讨论的策略约束和价值正则化 (regularization)技术，这些方法旨在减轻查询分布外（OOD）动作所带来的风险。

## 参考资料

- [Tree-based Batch Mode Reinforcement Learning](http://www.jmlr.org/papers/volume6/ernst05a/ernst05a.pdf) — Damien Ernst, Pierre Geurts, and Louis Wehenkel (2005)
  Journal: Journal of Machine Learning Research; Volume: 6; Pages: 507-550
  介绍了Fitted Q-Iteration (FQI) 作为离线强化学习算法，尤其在基于树的函数逼近器方面表现出色。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本涵盖Q学习、价值迭代和函数逼近理论基础的经典教材，FQI将这些概念扩展到批处理设置中。
- [Deep Neural Networks for Learning Control Policies](https://link.springer.com/article/10.1007/s10514-012-9294-8) — Sascha Lange, Martin Riedmiller, and Alexander van der Smagt (2012)
  Journal: Autonomous Robots; Publisher: Springer; Volume: 33; Pages: 109-124; DOI: [10.1007/s10514-012-9294-8](https://doi.org/10.1007/s10514-012-9294-8)
  描述了在FQI框架内应用深度神经网络作为函数逼近器，从而产生了神经拟合Q迭代（NFQ）。
- [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://doi.org/10.1561/2200000083) — Sergey Levine, Aviral Kumar, George Tucker, Justin Fu (2020)
  Journal: Foundations and Trends® in Machine Learning; Volume: 13; Pages: 1–168; DOI: [10.1561/2200000083](https://doi.org/10.1561/2200000083)
  全面综述了离线强化学习的挑战，包括限制FQI的分布偏移问题，并介绍了后续的缓解策略。

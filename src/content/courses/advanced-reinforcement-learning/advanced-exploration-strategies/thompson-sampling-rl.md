---
course: "advanced-reinforcement-learning"
chapter: "advanced-exploration-strategies"
lesson: "thompson-sampling-rl"
sourceId: 3414
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-4-advanced-exploration-strategies/thompson-sampling-rl"
title: "概率匹配：汤普森采样"
description: "利用贝叶斯方法和汤普森采样高效寻新。"
order: 3
plots: []
sourceHash: "51e5ba737390ecf4f545fd3439cb28452d81b02ad60bc68b80b98dff0be0cb2c"
sourceCorrections: []
---

虽然像置信上限（UCB）这样的方法通过在不确定性面前采取乐观态度来应对寻新，但**汤普森采样（TS）**，也称为后验采样，提供了一种不同的、基于概率的策略。汤普森采样并非选择具有最高*乐观*估计值的行动，而是根据给定当前可用数据下，其*为最优*行动的概率来选择行动。它通过贝叶斯推理 (inference)巧妙地平衡了寻新与利用现有信息。

### 汤普森采样的贝叶斯核心

汤普森采样的核心是一种贝叶斯算法。它对感兴趣的未知量保持一种信念，这种信念以概率分布（后验分布）的形式呈现。在强化学习 (reinforcement learning)的背景下，这个量通常是给定状态下采取特定行动的价值，即 $Q(s, a)$，或者可能是预测此价值的模型参数 (parameter)。

一般步骤如下：

1. **保持信念：** 对于每个可能的行动 $a$（如果在给定状态 $s$ 下，这取决于上下文 (context)），维持一个关于参数 $\theta_a$ 的后验概率分布 $P(\theta_a | \text{历史})$，这些参数决定了行动的价值或奖励。最初，这基于一个先验分布，反映了任何交互发生之前的初始信念。
2. **从信念中采样：** 在每个决策点，从当前后验分布中为*每个*行动 $a$ 采样一个参数值 $\tilde{\theta}_a$。
3. **行动：** 计算与每个采样参数 $\tilde{\theta}_a$ 相关的期望值（例如，Q值或即时奖励）。选择产生最高采样值的行动 $a^*$。
   
   $$
   a^* = \underset{a}{\mathrm{argmax}} \ f(\tilde{\theta}_a)
   $$
   
   其中 $f(\tilde{\theta}_a)$ 是从采样参数中得到的价值（例如，采样到的Q值）。
4. **更新信念：** 执行行动 $a^*$，观察结果（奖励 $r$，下一个状态 $s'$），并使用贝叶斯定理更新所选行动 $a^*$ 的后验分布，纳入新的观测数据。


$$
P(\theta_{a^*} | \text{新历史}) \propto P(\text{观测} | \theta_{a^*}) \times P(\theta_{a^*} | \text{旧历史})
$$


这个循环重复进行。后验分布表明具有很高最优可能性的行动将更频繁地被选择（利用现有信息）。后验分布较宽（高不确定性）但可能包含高价值的行动也将偶尔被采样，从而导致寻新。选择一个行动的概率直接匹配算法当前对该行动为最佳的信念。

> 汤普森采样在强化学习环境中的流程。

### 汤普森采样在强化学习 (reinforcement learning)中的应用

将汤普森采样直接应用于大型状态空间中估计 $Q(s, a)$ 会带来挑战，特别是在使用深度神经网络 (neural network)作为函数逼近器时。维持并从深度网络权重 (weight)或Q值本身的精确后验分布中采样，通常在计算上难以处理。

已经开发出几种近似方法：

1. **引导式DQN：** 虽然主要以集成方法闻名，但自举法可用于近似后验分布。在经验回放缓冲区的不同自举样本上训练多个Q网络“头部”，可以提供估计的多样性。在决策时，随机选择一个头部，智能体根据该头部的Q值贪婪地行动。这近似于从后验中采样。
2. **贝叶斯神经网络（BNNs）：** 变分推断（VI）或马尔可夫链蒙特卡洛（MCMC）等技术可用于近似网络权重的后验分布。从这个近似后验中采样权重可以产生Q函数的样本。然而，BNNs会增加显著的计算开销。
3. **Dropout近似：** 不仅在训练期间，而且在测试时使用Dropout（有时称为蒙特卡洛Dropout），可以被解释为贝叶斯推断的一种近似。使用不同的Dropout掩码执行多次前向传播，可以从Q值的近似预测分布中生成样本。
4. **特定参数 (parameter)分布：** 假设Q值分布具有特定形式（例如，高斯分布），并学习其参数（均值和方差）。这简化了信念的维护，但依赖于分布假设的有效性。

### 与UCB的比较

- **机制：** UCB基于置信上限确定性地选择行动，明确计算不确定性奖励。TS通过从后验信念分布中采样来随机性地选择行动；寻新隐含地产生于分布宽度中捕捉到的不确定性。
- **性能：** 经验上，汤普森采样通常表现非常好，有时甚至优于UCB的变体，尤其是在非平稳环境或先验知识得到有效整合时。
- **复杂度：** 实现精确的TS可能比UCB更复杂，尤其是在深度强化学习 (reinforcement learning)中，因为维护和从后验中采样存在挑战。引导式或Dropout等近似方法降低了这种复杂性，但引入了各自的近似。

### 优点与不足

**优点：**

- **原则性贝叶斯方法：** 提供了一个连贯的框架，用于整合先验知识并基于证据更新信念。
- **有效的寻新：** 通过有效引导寻新，使其倾向于不确定但可能带来奖励的行动，通常能实现最先进的性能。
- **自然平衡：** 概率匹配机制内在平衡了寻新与利用现有信息，而无需像UCB那样明确的奖励项。

**不足：**

- **计算成本：** 维护和从后验分布中采样可能计算昂贵，特别是对于深度神经网络 (neural network)等复杂模型。
- **近似挑战：** 实际实现通常依赖于近似（如自举法或变分推断），这些近似可能偏离真正的贝叶斯后验。
- **对先验的敏感性：** 先验分布的选择会影响初始性能，尽管随着更多数据的收集，这种影响通常会减弱。

汤普森采样代表了一种强大的、有统计学依据的方法，用于应对寻新-利用困境。虽然在复杂的强化学习 (reinforcement learning)环境中，精确的实现可能具有挑战性，但其优雅的特性和强大的经验性能（通常通过实际近似实现）使其成为高级强化学习实践者工具箱中的一个重要工具。

## 参考资料

- [On the Likelihood that One Unknown Probability Exceeds Another in a Series of Trials](https://www.jstor.org/stable/2332286) — William R. Thompson (1933)
  Journal: Biometrika; Volume: 25; Pages: 285-294; DOI: [10.2307/2332286](https://doi.org/10.2307/2332286)
  提出在不确定性下基于概率匹配进行决策的概念的原始论文。
- [A Tutorial on Thompson Sampling](https://www.nowpublishers.com/article/Details/MAL-070) — Daniel J. Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, Zheng Wen (2018)
  Journal: Foundations and Trends in Machine Learning; Publisher: Now Publishers; Volume: 11; Pages: 1-96; DOI: [10.1561/2200000070](https://doi.org/10.1561/2200000070)
  一份全面的教程，涵盖了汤普森采样在强化学习等多种背景下的基础、理论和应用。
- [Deep Exploration via Bootstrapped DQN](https://neurips.cc/paper/2016/file/07f8a7cf2e472658f9d0287834246830-Paper.pdf) — Ian Osband, Charles Blundell, Alexander Pritzel, Benjamin Van Roy (2016)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 29; Pages: 4026-4034
  介绍了Bootstrapped DQN，这是一种用于深度强化学习的汤普森采样的实用且广泛使用的近似方法。
- [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](http://proceedings.mlr.press/v48/gal16.pdf) — Yarin Gal, Zoubin Ghahramani (2016)
  Journal: Proceedings of the 33rd International Conference on Machine Learning (ICML 2016); Publisher: PMLR (Proceedings of Machine Learning Research); Volume: 48; Pages: 1050-1059; DOI: [10.5555/3045390.3045494](https://doi.org/10.5555/3045390.3045494)
  探讨了Dropout与贝叶斯推断之间的理论联系，提供了一种近似神经网络中后验分布的方法。

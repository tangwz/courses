---
course: "advanced-reinforcement-learning"
chapter: "deep-q-networks"
lesson: "distributional-rl-intro"
sourceId: 3375
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-2-deep-q-networks/distributional-rl-intro"
title: "分布强化学习原理"
description: "关于回报分布建模而非仅期望值的介绍（例如C51、QR-DQN）。"
order: 8
plots: ["plots/3375-0.json"]
sourceHash: "d76f9d0084495810748c4d8a6ad9dcf559ec3b0a543b225171e4bae9421f2631"
sourceCorrections: []
---

标准的深度Q网络（DQN）及其变体主要侧重于估计*预期的*未来折扣回报，即Q值 $Q(s, a) = \mathbb{E}[\sum_{t=0}^{\infty} \gamma^t R_{t+1} | S_0=s, A_0=a]$。这种期望将状态 $s$ 中采取动作 $a$ 的潜在结果压缩为一个单一的标量值。然而，这种压缩可能会丢失有关回报分布的变异性和形态的有价值信息。

设想一个智能体在两条路径中做选择。路径A稳定地带来中等回报。路径B则提供了获得极高回报的机会，但也伴随着受到巨大惩罚的显著风险。这两条路径可能具有相同的*预期*回报，使得标准DQN无法区分它们。然而，其潜在的风险特征却截然不同。分布强化学习 (reinforcement learning)通过直接建模随机回报 $Z(s, a)$ 的概率分布来解决此问题，而不仅仅是其期望值 $\mathbb{E}[Z(s, a)]$。

### 分布式贝尔曼方程

其核心思想是将贝尔曼方程推广到分布。设 $Z(s, a)$ 是一个随机变量，表示从状态 $s$ 开始，执行动作 $a$ 后，并按照当前策略持续下去所获得的回报。标准的贝尔曼最优性方程关联的是期望值：


$$
Q^*(s, a) = \mathbb{E}[R(s, a) + \gamma \max_{a'} Q^*(s', a')]
$$


分布式版本关联的是分布本身：


$$
Z(s, a) \stackrel{D}{=} R(s, a) + \gamma Z(s', a'^*)
$$


这里，$\stackrel{D}{=}$ 表示分布上的相等。随机回报 $Z(s, a)$ 与即时（可能随机的）回报 $R(s, a)$ 以及在下一状态 $s'$ 中采取最优动作 $a'^*$ 所关联的折扣随机回报 $Z(s', a'^*)$ 之和具有相同的分布。最优的下一动作 $a'^*$ 通常是通过最大化下一状态回报分布的*期望*值来选择的：$a'^* = \arg\max_{a'} \mathbb{E}[Z(s', a')]$。这个方程为回报分布提供了一个递归定义，构成了学习算法的基础。

### 回报分布的表示

表示和学习潜在的连续概率分布是具有挑战性的。实际算法采用近似方法：

#### 分类DQN (C51)

C51算法由Bellemare等人（2017）提出，它通过固定数量的 $N$ 个“原子”上支持的离散分布来近似回报分布 $Z(s, a)$。这些原子 $z_1, z_2, \dots, z_N$ 通常选择为在合理的回报范围 $[V_{MIN}, V_{MAX}]$ 内等间距的点。

深度神经网络 (neural network)不再为每个动作输出单一的Q值，而是为每个动作输出这些 $N$ 个原子上的概率分布。对于给定状态 $s$，网络会输出 $N \times |\mathcal{A}|$ 个值，这些值通常会通过softmax函数针对每个动作 $a$ 处理，以产生概率 $p_i(s, a)$：


$$
p_i(s, a) \approx \mathbb{P}(Z(s, a) = z_i) \quad \text{使得} \quad \sum_{i=1}^N p_i(s, a) = 1
$$


如果需要，可以很容易地恢复预期的Q值：$Q(s, a) = \sum_{i=1}^N z_i p_i(s, a)$。

**学习更新：** 学习过程涉及应用分布式贝尔曼更新。对于一个状态转移 $(s, a, r, s')$，目标分布的构建方法如下：

1. 使用目标网络对状态 $s'$ 的输出，计算最优的下一动作 $a'^* = \arg\max_{a'} \sum_{j=1}^N z_j p_j(s', a')$。
2. 对于目标网络中 $(s', a'^*)$ 分布的每个原子 $z_j$，计算贝尔曼目标：$\hat{\mathcal{T}} z_j = r + \gamma z_j$。这表示一个可能的折扣未来回报，由即时回报 $r$ 偏移。
3. 由于 $\hat{\mathcal{T}} z_j$ 可能无法与固定原子位置 $\{z_i\}$ 完美对齐 (alignment)，其概率质量 $p_j(s', a'^*)$ 会被投射到相邻的支持原子上。具体来说，如果 $\hat{\mathcal{T}} z_j$ 落在 $z_k$ 和 $z_{k+1}$ 之间，则质量 $p_j(s', a'^*)$ 将根据接近程度在 $z_k$ 和 $z_{k+1}$ 之间线性分配。
4. 最终的目标分布 $d'_{target}$ 是这些投射概率在所有原子 $j=1, \dots, N$ 上的总和。
5. 通过最小化预测分布 $d(s, a) = \{p_i(s, a)\}_{i=1}^N$ 与计算出的目标分布 $d'_{target}$ 之间的库尔巴克-莱布勒（KL）散度来训练网络。这充当了损失函数 (loss function)。



![两种动作的示例回报分布](plots/3375-0.json)



> 两种不同动作在回报原子上的概率分布示例。尽管它们可能具有相同的均值（预期Q值），但它们的形状显示出不同的风险特点。与动作B更集中的分布相比，动作A具有更高的潜在回报，但也伴随更高的潜在损失。

#### 分位数回归DQN (QR-DQN)

QR-DQN由Dabney等人（2017）提出，它采取了一种不同的方法，通过建模回报分布的*分位数函数*（逆CDF）。QR-DQN不固定回报值（原子）并学习概率，而是固定累积概率 $\tau_i$ 并学习相应的回报值（分位数）$\theta_i(s, a)$。

网络为每个动作 $a$ 输出 $N$ 个分位数 $\theta_1(s, a), \dots, \theta_N(s, a)$。这些分位数对应于一组固定的 $N$ 个目标分位数，通常均匀选择，例如对于 $i=1, \dots, N$，$\hat{\tau}_i = \frac{i - 0.5}{N}$。$\theta_i(s, a)$ 表示预测的回报值 $z$，使得 $P(Z(s, a) \le z) \approx \hat{\tau}_i$。

**学习更新：** QR-DQN使用分位数回归损失。对于状态转移 $(s, a, r, s')$，目标分位数是 $r + \gamma \theta_j(s', a'^*)$，其中 $\theta_j(s', a'^*)$ 是目标网络对最优下一动作 $a'^* = \arg\max_{a'} \frac{1}{N} \sum_{k=1}^N \theta_k(s', a')$ 预测的分位数。损失函数通过使用正确处理分位数估计不对称性质的公式（如分位数Huber损失），最小化预测分位数 $\theta_i(s, a)$ 与目标分位数之间的差异。

像隐式分位数网络（IQN）这样的进一步进展，可以学习一个函数，为任何输入概率 $\tau \in [0, 1]$ 生成分位数，从而提供更连续的分布表示。

### 分布式视角的优势

学习完整的回报分布提供了多项好处：

1. **更丰富学习信号**：与单一的期望值相比，分布提供了更详细的信息，可能导致更稳定和有效的学习，尤其是在奖励或状态转移随机的环境中。它有助于区分均值相似但风险特征不同的动作。
2. **前沿性能**：分布强化学习 (reinforcement learning)算法，特别是C51和QR-DQN，在Atari套件等挑战性基准测试中表现出显著的性能提升，是结合了多项DQN改进的Rainbow智能体的重要组成部分。
3. **风险感知**：拥有回报分布使得明确的风险感知决策成为可能。智能体不再仅仅最大化均值 $\mathbb{E}[Z(s, a)]$，还可以优化其他统计量，如特定分位数（例如，最大化10%分位数以实现风险规避行为）或条件风险价值（CVaR）。

实现分布式强化学习需要修改网络的输出层，以预测分布参数 (parameter)（原子的概率或分位数），并相应地调整损失函数 (loss function)（KL散度或分位数回归损失）和贝尔曼更新机制。尽管增加了复杂度，但其经验性增益和处理风险的能力使其成为深度强化学习中的一项重要进展。

## 参考资料

- [A Distributional Perspective on Reinforcement Learning](http://proceedings.mlr.press/v70/bellemare17a.html) — Marc G. Bellemare, Will Dabney, Rémi Munos (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning; Publisher: PMLR; Volume: 70; Pages: 449-458
  介绍了强化学习中的分类分布方法和C51算法。
- [Implicit Quantile Networks for Distributional Reinforcement Learning](https://proceedings.mlr.press/v80/dabney18a.html) — Will Dabney, Georg Ostrovski, David Silver, Remi Munos (2018)
  Journal: Proceedings of the 35th International Conference on Machine Learning; Publisher: PMLR; Volume: 80; Pages: 1096-1105
  通过学习连续分位数函数扩展了QR-DQN，可估算任意分位数。
- [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://doi.org/10.1609/aaai.v32i1.11792) — Matteo Hessel, Joseph Modayil, Hado van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Gheshlaghi Azar, David Silver (2018)
  Journal: Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence; Publisher: AAAI Press; Volume: 32; Pages: 3215–3222; DOI: [10.1609/aaai.v32i1.11792](https://doi.org/10.1609/aaai.v32i1.11792)
  通过结合多项深度Q网络改进（包括C51）展示了强大的实证性能。

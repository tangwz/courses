# 马尔可夫链理论之于MCMC

来源：[原文](https://apxml.com/zh/courses/bayesian-machine-learning/chapter-2-markov-chain-monte-carlo-methods/markov-chain-theory-mcmc)

[返回章节目录](README.md) · [返回课程目录](../README.md)

直接计算或从复杂、高维的后验分布 $p(	heta | D)$ 中采样通常难以实现。MCMC算法提供了一种巧妙的变通方法：我们不直接从 $p(	heta | D)$ 中抽取独立样本，而是构建一个**马尔可夫链**，其状态是参数 (parameter)值 $heta$，且其**平稳分布**正是目标后验分布 $p(	heta | D)$。通过足够长时间地模拟这个链，我们生成的样本最终将表现得如同从目标后验分布中抽取的一样。

但这如何运行呢？为什么我们可以相信，从某个巧妙构建的随机游走中获得的样本最终会与我们特定后验分布的样本相似？答案在于马尔可夫链的数学性质。我们来审视一下核心理论。

### 马尔可夫链是什么？

马尔可夫链是随机变量序列 $\theta^{(0)}, \theta^{(1)}, \theta^{(2)}, \ldots$，其中每个 $\theta^{(t)}$ 表示链在时间步 $t$ 时的状态。在我们的背景下，一个状态通常是模型参数 (parameter) $\theta$ 的一个特定值（或值向量 (vector)）。马尔可夫链的定义特征是**马尔可夫性质**：下一个状态 $\theta^{(t+1)}$ 的分布*仅*取决于当前状态 $\theta^{(t)}$，而不取决于任何先前的状态 $\theta^{(0)}, \ldots, \theta^{(t-1)}$。

数学上，这表示为：


$$
P(\theta^{(t+1)} | \theta^{(t)}, \theta^{(t-1)}, \ldots, \theta^{(0)}) = P(\theta^{(t+1)} | \theta^{(t)})
$$


管理状态间移动的机制是**转移核**（或转移概率），表示为 $T(\theta' | \theta)$。此函数给出当链当前处于状态 $\theta$ 时，移动到状态 $\theta'$ 的概率密度。MCMC算法通过仔细设计这个转移核 $T$ 来运作。

> 马尔可夫链的一个简单图示，包含三个状态 ($\theta_1, \theta_2, \theta_3$) 及其相关的转移概率。下一个状态仅取决于当前状态。

### 平稳分布

一个重要的思想是**平稳分布**，通常表示为 $\pi(\theta)$。对于一个具有转移核 $T(\theta' | \theta)$ 的马尔可夫链，如果一旦链的状态分布达到 $\pi(\theta)$，它就保持不变，那么分布 $\pi(\theta)$ 就是平稳的。换句话说，如果链在时间 $t$ 处于状态 $\theta$ 的概率由 $\pi(\theta)$ 给出，那么在时间 $t+1$ 处于状态 $\theta'$ 的概率也由 $\pi(\theta')$ 给出。

形式上，如果 $\pi$ 满足以下条件，它就是一个平稳分布：


$$
\pi(\theta') = \int T(\theta' | \theta) \pi(\theta) d\theta
$$


对于离散状态空间，这变为：


$$
\pi_j = \sum_i T_{ji} \pi_i
$$


其中 $\pi_i$ 是处于状态 $i$ 的概率，而 $T_{ji}$ 是从状态 $i$ 转移到状态 $j$ 的概率。

MCMC的目的是设计一个转移核 $T$，使其唯一的平稳分布 $\pi(\theta)$ 正是我们的目标后验分布 $p(\theta | D)$。如果我们能做到这一点，运行该链就能确保状态分布最终收敛到 $p(\theta | D)$。

### 收敛条件：遍历性

为了使MCMC算法有实际作用，我们需要保证链将确实收敛到所需的平稳分布 $\pi(\theta) = p(\theta | D)$，无论起始状态 $\theta^{(0)}$ 是什么。这个性质被称为**遍历性**。一个遍历的马尔可夫链拥有唯一的平稳分布，且 $\theta^{(t)}$ 的分布在 $t \to \infty$ 时收敛到这个平稳分布。

两个条件通常足以保证遍历性：

1. **不可约性：** 链必须能够最终从状态空间的任何部分（其中 $\pi(\theta) > 0$）到达其他任何部分。这确保链不会只停留在目标分布的一部分中。形式上，对于任何初始状态 $\theta$ 和任何区域 $A$，如果 $\int_A \pi(\theta') d\theta' > 0$，则必须存在某个 $t > 0$ 使得 $P(\theta^{(t)} \in A | \theta^{(0)} = \theta) > 0$。
2. **非周期性：** 链不应被限制在固定的循环中访问状态。例如，不应该只在偶数步后才能返回状态 $\theta$。非周期性有助于确保收敛不会因系统性的周期行为而受阻。许多标准MCMC算法（如具有合理提议的Metropolis-Hastings或Gibbs采样）自动满足此条件，因为通常存在留在同一状态的非零概率 ($T(\theta | \theta) > 0$)。

如果一个链是不可约和非周期的，并且具有平稳分布 $\pi(\theta)$，那么它是遍历的。这意味着：


$$
\lim_{t \to \infty} P(\theta^{(t)} \in A | \theta^{(0)}) = \int_A \pi(\theta') d\theta'
$$


对于状态空间中的任何起始状态 $\theta^{(0)}$ 和任何区域 $A$。这种极限行为正是我们所需要的：在足够的“预热期”之后，状态 $\theta^{(t)}$ 的行为就像从 $\pi(\theta) = p(\theta | D)$ 中抽取的样本一样。

### 细致平衡：一个便捷条件

直接设计转移核 $T$ 并使用积分方程 $\pi(\theta') = \int T(\theta' | \theta) \pi(\theta) d\theta$ 证明 $p(\theta | D)$ 是其平稳分布可能很困难。幸运的是，存在一个更简单、充分的条件：**细致平衡**（也称为可逆性）。

如果从状态 $\theta$ 到状态 $\theta'$ 的概率流等于从 $\theta'$ 返回到 $\theta$ 的概率流，那么马尔可夫链对于分布 $\pi(\theta)$ 满足细致平衡：


$$
\pi(\theta) T(\theta' | \theta) = \pi(\theta') T(\theta | \theta')
$$


如果转移核 $T$ 对于 $\pi$ 满足细致平衡，那么 $\pi$ 是该核的一个平稳分布。我们可以通过对两边关于 $\theta$ 进行积分来证明这一点：


$$
\int \pi(\theta) T(\theta' | \theta) d\theta = \int \pi(\theta') T(\theta | \theta') d\theta
$$


$$
\int T(\theta' | \theta) \pi(\theta) d\theta = \pi(\theta') \int T(\theta | \theta') d\theta
$$


因为 $\int T(\theta | \theta') d\theta = 1$（从 $\theta'$ *转移出去*的总概率为1），我们得到：


$$
\int T(\theta' | \theta) \pi(\theta) d\theta = \pi(\theta')
$$


这正是 $\pi$ 是平稳分布的条件。

大多数常见的MCMC算法，包括Metropolis-Hastings和Gibbs采样，都是专门构建的，以使其对目标后验分布 $p(\theta | D)$ 满足细致平衡。这提供了一种设计算法的有效途径，这些算法在不可约和非周期性的条件下，保证收敛到正确的分布。

### 总结：马尔可夫链理论对MCMC的重要性

理解这些主要思想，即马尔可夫性质、平稳分布、遍历性（通过不可约性和非周期性）以及细致平衡，对MCMC来说是不可或缺的。这一理论为模拟一个精心构建的马尔可夫链为何能使我们生成近似目标后验分布 $p(\theta | D)$ 的样本提供了数学依据。通过确保我们选择的MCMC算法对应于一个对 $p(\theta | D)$ 满足细致平衡的遍历链，我们获得了信心，即在预热期后生成的样本可以用于贝叶斯推断，从而使我们能够近似后验均值、方差、可信区间以及其他感兴趣的量。以下章节将基于这一理论知识，介绍Metropolis-Hastings、Gibbs采样和哈密顿蒙特卡洛等具体的MCMC算法。

## 参考资料

- [Bayesian Data Analysis](https://www.routledge.com/Bayesian-Data-Analysis/Gelman-Carlin-Stern-Dunson-Vehtari-Rubin/p/book/9781439840955) — Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, and Donald B. Rubin (2013)
  Publisher: Chapman & Hall
  贝叶斯推断的标准教材，广泛涵盖MCMC算法及其理论基础。
- [Monte Carlo Statistical Methods](https://link.springer.com/book/10.1007/978-1-4757-4145-2) — Christian P. Robert, George Casella (2004)
  Publisher: Springer; DOI: [10.1007/978-1-4757-4145-2](https://doi.org/10.1007/978-1-4757-4145-2)
  一本关于蒙特卡洛和马尔可夫链蒙特卡洛方法的指南，包含详细的数学基础。
- [Monte Carlo sampling methods using Markov chains and their applications](http://www.jstor.org/stable/2334940) — W. K. Hastings (1970)
  Journal: Biometrika; Publisher: Biometrika Trust; Volume: 57; Pages: 97-109; DOI: [10.1093/biomet/57.1.97](https://doi.org/10.1093/biomet/57.1.97)
  一篇基础论文，介绍了广义的Metropolis-Hastings算法并讨论了马尔可夫链在采样中的应用。
- [An introduction to MCMC for machine learning](https://doi.org/10.1023/A:1020281327116) — Andrieu, C., de Freitas, N., Doucet, A., Jordan, M. I. (2003)
  Journal: Machine Learning; Volume: 50; Pages: 5-43; DOI: [10.1023/A:1020281327116](https://doi.org/10.1023/A%3A1020281327116)
  一篇综述文章，专为机器学习从业者总结了MCMC理论和算法。

---

[上一节](01-%E8%92%99%E7%89%B9%E5%8D%A1%E7%BD%97%E7%A7%AF%E5%88%86%E7%9A%84%E5%8E%9F%E7%90%86.md) · [下一节](03-Metropolis-Hastings%20%E7%AE%97%E6%B3%95%E5%8F%98%E4%BD%93.md)

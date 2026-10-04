---
course: "bayesian-machine-learning"
chapter: "markov-chain-monte-carlo-methods"
lesson: "nuts-sampler"
sourceId: 3545
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-2-markov-chain-monte-carlo-methods/nuts-sampler"
title: "无U形转弯采样器 (NUTS)"
description: "了解NUTS算法，它是HMC的一种自适应扩展，用于自动调优。"
order: 6
plots: ["plots/3545-0.json"]
sourceHash: "8cc33deccb9f71778c71379364a1a253899afd1a840ca7596062ee8bc9ce5318"
sourceCorrections: []
---

虽然哈密顿蒙特卡洛 (HMC) 相比 Metropolis-Hastings 或 Gibbs 采样等更简单的 MCMC 方法具有显著优势，尤其是在高维或复杂后验几何中，但它也引入了自己的调优参数 (parameter)。正如我们在上一节中看到的，HMC 依赖于使用特定步长 $\epsilon$ 模拟固定数量的蛙跳步数 $L$ 的哈密顿动力学。恰当地选择 $L$ 很有挑战性：如果太小，采样器会表现得像随机游走，收敛缓慢；如果太大，轨迹可能会形成一个“U形转弯”，返回到其起始点附近，这会浪费计算资源并降低采样效率。最优地设置 $L$ 通常需要仔细的、针对模型的调优。

无U形转弯采样器 (NUTS) 的开发正是为了解决这个问题。它是 HMC 的一种自适应扩展，能自动为每次迭代确定合适的蛙跳步数，从而无需手动指定 $L$。NUTS 保留了 HMC 使用梯度信息高效采样的优点，同时增加了其稳定性并减轻了调优负担。

### HMC 中的 U形转弯问题

想象一下模拟哈密顿动力学。我们从一个位置 $\theta$ 开始，并随机采样一个动量 $p$。蛙跳积分器生成一系列状态 $(\theta_1, p_1), (\theta_2, p_2), \dots, (\theta_L, p_L)$。如果我们将此积分继续进行太多步（即 $L$ 很大），轨迹最终可能会因为由后验概率定义的势能而弯曲回来。在某个点，轨迹可能会开始返回到初始位置 $\theta_0$。在此点之后继续模拟是低效的，因为新提议的点与轨迹中较早的点相关，并且从起始点采样的总距离减小。这种反转就是 NUTS 试图避免的“U形转弯”。



![具有U形转弯趋势的HMC轨迹](plots/3545-0.json)



> 模拟的哈密顿轨迹，从红色星形点开始。NUTS 的目标是在路径显著反向（在红色“x”附近标示）之前停止采样，从而避免计算浪费。

### NUTS 如何调整轨迹长度

NUTS 不固定 $L$，而是通过从当前点 $\theta^{(t)}$ 开始，向前和向后模拟哈密顿轨迹，动态地构建一组候选点。它使用一种巧妙的递归策略来完成此操作，该策略在每一步中有效地使所采样轨迹段的长度加倍。

以下是核心思想：

1. **初始化：** 从当前样本 $\theta^{(t)}$ 开始，带有随机动量 $p$。这构成了初始的候选状态“树”，只包含 $\theta^{(t)}$。
2. **递归加倍：** 迭代地扩展轨迹。在每个扩展步骤中，算法随机选择向前或向后模拟，使其在该方向上采样的蛙跳步数加倍。这构建了一个二叉树结构，其中的叶节点代表沿模拟轨迹访问的状态。
3. **U形转弯检查：** 关键是，在每个加倍步骤之后，NUTS 会检查刚添加的轨迹段是否产生了 U形转弯。该检查评估整体轨迹的初始点和新到达的终点是否正在彼此靠近。数学上，这通常涉及检查连接端点的向量 (vector)与这些端点处的动量向量之间的点积：$(\theta_{max} - \theta_{min}) \cdot p_{min} < 0$ 或 $(\theta_{max} - \theta_{min}) \cdot p_{max} < 0$。如果正在采样的子树中的*任何地方*检测到 U形转弯，算法就会停止在该方向上的扩展。这可以防止轨迹折返。
4. **采样：** 加倍过程持续进行，直到新提议的子树满足 U形转弯条件或达到最大树深度（作为保护措施）。这定义了一组有效的候选状态（所构建树的叶节点，不包括那些触发 U形转弯的子树中的叶节点）。NUTS 然后从此组有效状态中采样*下一个状态* $\theta^{(t+1)}$。为了保持细致平衡（MCMC 收敛的一个要求），采样不只来自最终端点，而是考虑沿有效路径生成的所有状态，通常从由接受概率定义的“切片”中均匀采样。

### NUTS 的优点

- **自动轨迹长度：** 最重要的实际优势是 NUTS 自动确定每次迭代的蛙跳步数 ($L$)，适应后验分布的局部曲率。这消除了手动调优 $L$ 的需要。
- **高效率：** NUTS 通常继承 HMC 卓越的采样效率，使其非常适合随机游走方法难以处理的高维和复杂后验。
- **稳定性：** 尽管步长 $\epsilon$ 仍需设置（通常在预热阶段使用双重平均等方法自动调优），但与固定 $L$ 的标准 HMC 相比，NUTS 通常对 $\epsilon$ 的具体选择不那么敏感。

### 注意事项

- **每次迭代的计算成本：** 单次 NUTS 迭代可能比单次 HMC 迭代的计算成本更高，因为它涉及构建树和执行 U形转弯检查，平均模拟的蛙跳步数可能比固定 $L$ 的 HMC 更多。然而，NUTS 通常需要更少的总迭代次数才能收敛到目标分布，可能导致更快的整体推断。
- **实现复杂度：** NUTS 算法比基本 HMC 更难正确实现。幸运的是，主流概率编程库中都有其实现。
- **步长调优：** 尽管 $L$ 是自动处理的，但步长 $\epsilon$ 仍然是影响性能的一个重要参数 (parameter)。现代实现通常在预热阶段使用复杂的自适应方案来找到合适的 $\epsilon$。

### NUTS 的实践应用

由于其稳定性及自动调优能力，NUTS 已成为 Stan 和 PyMC 等广泛使用的概率编程框架中的默认 MCMC 算法。当您使用这些工具时，您通常可以从 NUTS 高效的采样中获益，而无需深入了解轨迹长度调优的具体细节，从而让您可以更专注于模型规范和解释。然而，理解 NUTS 背后的原理有助于诊断潜在问题，并理解为什么它通常比早期的 MCMC 方法表现得如此出色。

## 参考资料

- [The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo](https://www.jmlr.org/papers/v15/hoffman14a.html) — Matthew D. Hoffman, Andrew Gelman (2014)
  Journal: JMLR; Publisher: JMLR; Volume: 15; Pages: 1593-1623; DOI: [10.5555/2627435.2627515](https://doi.org/10.5555/2627435.2627515)
  这是介绍无 U 型转弯采样器 (NUTS) 算法的原始论文。
- [MCMC Using Hamiltonian Dynamics](http://www.cs.toronto.edu/~radford/ham-mcmc.abstract.html) — Radford M. Neal (2011)
  Journal: Handbook of Markov Chain Monte Carlo; Publisher: Chapman & Hall/CRC; Pages: 113-162; DOI: [10.1201/b10905-6](https://doi.org/10.1201/b10905-6)
  一篇关于哈密顿蒙特卡洛的综述文章，为 NUTS 提供了理论背景。
- [Bayesian Data Analysis](https://doi.org/10.1201/b16018) — Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, Donald B. Rubin (2013)
  Publisher: CRC Press; Pages: 675; DOI: [10.1201/b16018](https://doi.org/10.1201/b16018)
  一本关于贝叶斯推断的标准教科书，涵盖了包括 HMC 和 NUTS 在内的 MCMC 方法。
- [Stan User's Guide](https://mc-stan.org/docs/stan-users-guide/index.html) — Stan Development Team (2011)
  官方文档，提供 Stan 中 NUTS 实现的实用细节和信息。

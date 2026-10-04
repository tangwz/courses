# 广义优势估计 (GAE)

来源：[原文](https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-3-advanced-policy-gradients-actor-critic/generalized-advantage-estimation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

REINFORCE 等基本策略梯度方法存在高方差问题。优势演员-评论家 (A2C) 框架通过使用状态价值函数 $V(s)$ 作为基线来缓解此问题。在 A2C 中，策略梯度更新依赖于优势函数，$A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$。主要的问题是：如何利用智能体收集的数据，最有效地估计这个优势函数？

存在多种 $A(s_t, a_t)$ 的估计方法，每种方法在偏差和方差之间都有各自的权衡。请记住，我们的目标是获得一个能够产生稳定且高效策略更新的估计值。

考虑 $t$ 时刻的 TD 误差：


$$
\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
$$


这个 $\delta_t$ 是优势函数 $A(s_t, a_t)$ 的一个估计。为什么？因为 $r_{t+1} + \gamma V(s_{t+1})$ 是 $Q(s_t, a_t)$ 的一个估计。直接使用 $\delta_t$ 作为优势估计值，能够提供低方差，因为它只依赖于下一个奖励和下一个状态的价值函数估计。然而，它可能存在偏差，特别是当价值函数估计 $V(s)$ 不准确时。这基本就是基本一步式演员-评论家算法中使用的优势估计。

另一种方法是使用蒙特卡洛估计。我们将优势函数估计为总折扣回报 $G_t$ 减去基线 $V(s_t)$：


$$
\hat{A}_t^{MC} = G_t - V(s_t) = \left( \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \right) - V(s_t)
$$


在实际应用中，我们使用累积到剧集结束（或固定时间范围）的回报。如果 $V(s_t)$ 是真实的价值函数，这个估计是无偏的，但它会遇到高方差问题，因为它在轨迹上累加了许多随机奖励 $r_{t+k+1}$。

我们还可以定义 N 步优势估计，它结合了这两种极端情况：


$$
\hat{A}_t^{(N)} = \left( \sum_{k=0}^{N-1} \gamma^k r_{t+k+1} + \gamma^N V(s_{t+N}) \right) - V(s_t)
$$


这使用了 $N$ 步的实际奖励，然后利用价值函数估计 $V(s_{t+N})$ 进行自举。增加 $N$ 通常会减少偏差，但会增加方差。这表明存在一系列可能的估计器。

### 广义优势估计 (GAE)

广义优势估计 (GAE) 提供了一种更精巧的方法来处理这种偏差-方差权衡。它引入了一个参数 (parameter) $\lambda$（其中 $0 \le \lambda \le 1$），以明确控制偏差和方差之间的权重 (weight)。GAE 公式将优势估计 $\hat{A}_t$ 计算为多个时间步的 TD 误差 $\delta_{t+l}$ 的指数加权平均值：


$$
\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$


这里，$\gamma$ 是标准折扣因子，$\lambda$ 是用于调整权衡的 GAE 参数。让我们检查 $\lambda$ 的边界情况：

- **情况 $\lambda = 0$：**
  如果我们将 $\lambda = 0$，求和项将只剩下第一项（$l=0$）：

  
  $$
  \hat{A}_t^{GAE(\gamma, 0)} = (\gamma \cdot 0)^0 \delta_t = \delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
  $$
  

  这正是单步 TD 误差估计。它通常具有低方差，但如果价值函数 $V$ 不准确，它可能会有偏差。
- **情况 $\lambda = 1$：**
  如果我们将 $\lambda = 1$，公式变为：

  
  $$
  \hat{A}_t^{GAE(\gamma, 1)} = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} = \sum_{l=0}^{\infty} \gamma^l (r_{t+l+1} + \gamma V(s_{t+l+1}) - V(s_{t+l}))
  $$
  

  这个和可以展开。如果我们忽略函数逼近误差并假设 $V$ 是真实的价值函数，这个和会收敛到：

  
  $$
  \hat{A}_t^{GAE(\gamma, 1)} \approx \left( \sum_{l=0}^{\infty} \gamma^l r_{t+l+1} \right) - V(s_t) = G_t - V(s_t)
  $$
  

  这基本就是无偏蒙特卡洛优势估计，它倾向于具有高方差。
- **中间 $\lambda$ ($0 < \lambda < 1$)：**
  $\lambda$ 在 0 和 1 之间的值会产生介于 TD(0) 优势和蒙特卡洛优势之间的估计。较高的 $\lambda$（接近 1）会给予长期奖励信息更大的权重，以增加方差为代价减少偏差。较低的 $\lambda$（接近 0）更依赖于当前的价值函数估计，从而减少方差但可能增加偏差。这允许实践者根据具体问题和价值函数估计的质量来调整估计器。



[交互图表：GAE 偏差-方差权衡与 Lambda](https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-3-advanced-policy-gradients-actor-critic/generalized-advantage-estimation#plot-mkec6h)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        0,
        0.25,
        0.5,
        0.75,
        1
      ],
      "y": [
        1,
        0.7,
        0.4,
        0.2,
        0.1
      ],
      "mode": "lines",
      "name": "偏差",
      "line": {
        "color": "#f03e3e",
        "width": 3
      }
    },
    {
      "x": [
        0,
        0.25,
        0.5,
        0.75,
        1
      ],
      "y": [
        0.1,
        0.3,
        0.6,
        0.8,
        1
      ],
      "mode": "lines",
      "name": "方差",
      "line": {
        "color": "#1c7ed6",
        "width": 3
      }
    }
  ],
  "layout": {
    "title": "GAE 偏差-方差权衡与 Lambda",
    "xaxis": {
      "title": "Lambda (λ)",
      "range": [
        -0.05,
        1.05
      ]
    },
    "yaxis": {
      "title": "相对大小",
      "range": [
        0,
        1.1
      ]
    },
    "legend": {
      "yanchor": "top",
      "y": 0.99,
      "xanchor": "left",
      "x": 0.01
    }
  }
}
```

</details>



> 图示 GAE 参数 $\lambda$ 从 0（TD 优势）到 1（蒙特卡洛优势）时，偏差如何减小而方差如何增大。

### GAE 的实际计算

GAE 定义中的无穷和在计算上不实用。幸运的是，我们可以使用递归公式高效地计算它，通常从轨迹末尾或长度为 $T$ 的经验批次的末尾开始反向计算。

令 $s_T$ 为序列中的最后一个状态。

1. 计算所有步长 $t = 0, 1, ..., T-1$ 的 TD 误差：

   
   $$
   \delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
   $$
   

   如果 $s_{t+1}$ 是一个终止状态，则 $V(s_{t+1})$ 定义为 0。
2. 从 $t = T-1$ 开始反向计算 GAE 估计，直到 $t = 0$：
   如果 $s_T$ 是终止状态，设置 $\hat{A}_T^{GAE} = 0$。如果在 $T$ 步之后使用自举（即，剧集未终止），如果 $T+1$ 的数据可用，你可能会计算 $\delta_T = r_{T+1} + \gamma V(s_{T+1}) - V(s_T)$，或者简单地设置 $\hat{A}_T^{GAE}=0$ 并依赖于到 $T-1$ 的 TD 误差。对于批次结束时非终止状态的常见做法是，最初将 $\delta_{T-1}$ 用作 $\hat{A}_{T-1}^{GAE}$，或者直接计算 $\hat{A}_{T-1}^{GAE} = \delta_{T-1}$。一个稍微更精确的方法是从假设 $\hat{A}_T^{GAE} = 0$ 开始递归。

   然后，反向迭代：

   
   $$
   \hat{A}_t^{GAE} = \delta_t + \gamma \lambda \hat{A}_{t+1}^{GAE}
   $$
   

   注意，$V(s_t)$ 和 $V(s_{t+1})$ 由评论家网络当前的估计提供。

### 为何使用 GAE？

GAE 已成为许多现代演员-评论家算法中的一个标准组成部分，特别是 TRPO 和 PPO。它的主要优点是与蒙特卡洛估计相比，方差显著降低，这通常会带来更稳定、更快速的学习，同时能控制与简单 TD 误差估计相比所引入的偏差。

参数 (parameter) $\lambda$ 就像一个控制旋钮。虽然 $\lambda=1$ 对应于无偏蒙特卡洛估计，$\lambda=0$ 对应于可能有偏差的 TD(0) 估计，但中间的值（例如 $\lambda=0.95$ 或 $\lambda=0.97$）通常通过找到良好的平衡来获得最佳的经验性能。选择最优的 $\lambda$ 取决于具体问题，并在智能体训练期间成为另一个需要调整的超参数 (hyperparameter)。

总之，广义优势估计提供了一种有原则且有效的方法，用于在演员-评论家算法中估计优势函数。通过引入 $\lambda$ 参数，GAE 提供了一种灵活的机制来管理重要的偏差-方差权衡，对高级策略梯度方法的稳定性和性能做出了重要贡献。

## 参考资料

- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel (2015)
  Journal: arXiv preprint arXiv:1506.02438; DOI: [10.48550/arXiv.1506.02438](https://doi.org/10.48550/arXiv.1506.02438)
  介绍了广义优势估计 (GAE) 方法，为策略梯度方法中处理偏差-方差权衡提供了一个原则性方案。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本全面的教材，涵盖了强化学习的基本概念，包括策略梯度、Actor-Critic 方法、TD 误差、蒙特卡洛方法和价值函数估计，为理解 GAE 提供了理论背景。
- [Proximal Policy Optimization Algorithms](https://doi.org/10.48550/arXiv.1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  提出了近端策略优化 (PPO) 算法，这是一种广泛使用的 Actor-Critic 方法，它有效地利用广义优势估计 (GAE) 来实现稳定高效的策略更新。

---

[上一节](04-%E4%BC%98%E5%8A%BF%20Actor-Critic%20%28A2C%29%20%E5%92%8C%20A3C.md) · [下一节](06-%E6%B7%B1%E5%BA%A6%E7%A1%AE%E5%AE%9A%E6%80%A7%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%20%28DDPG%29.md)

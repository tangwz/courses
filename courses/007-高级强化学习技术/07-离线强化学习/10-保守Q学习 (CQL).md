# 保守Q学习 (CQL)

来源：[原文](https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-7-offline-reinforcement-learning/cql-offline)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然像BCQ这样的策略限制方法直接将学习到的策略限制为与离线数据集中相似的动作，但**保守Q学习 (CQL)** 采取了不同的办法。CQL没有修改策略搜索空间，而是修改Q函数学习目标本身，以对抗分布偏移导致的出界（OOD）动作值过高估计问题，这是分布偏移的主要结果。

当标准Q学习更新（如DQN或SAC中的更新）在贝尔曼目标中使用动作上的最大Q值时，就会出现根本问题：$y = r + \gamma \max_{a'} Q_{\bar{\theta}}(s', a')$。如果Q函数 $Q_{\bar{\theta}}$ 错误地为OOD动作 $a'$（在生成数据集 $\mathcal{D}$ 的行为策略 $\pi_b$ 下不太可能或未见的动作）分配高值，这些错误就会在学习过程中传播，导致次优的最终策略。

### CQL原理：惩罚OOD动作值

CQL在标准Q学习损失函数 (loss function)中引入了一个正则项。这个正则项促使Q函数 $Q_\theta(s, a)$ 对OOD动作分配*低*值，同时确保数据集中*实际存在*的动作的Q值保持准确或通过贝尔曼更新被推高。这使得Q值得到一个“保守”的估计，从而防止智能体利用未见动作中可能被高估的值。

### CQL目标函数

CQL目标函数通常将一个标准的基于TD的损失 $L_{TD}(\theta)$ 与一个状态-动作值正则器 $R_{CQL}(\theta)$ 结合。组合后的目标函数是：

$L_{CQL}(\theta) = \alpha R_{CQL}(\theta) + L_{TD}(\theta)$

这里，$\alpha \ge 0$ 是一个超参数 (parameter) (hyperparameter)，用于衡量保守惩罚的权重 (weight)。$L_{TD}(\theta)$ 可以是DQN中的均方贝尔曼误差，或者是SAC中的软贝尔曼残差。

核心部分是正则器 $R_{CQL}(\theta)$。一种常见形式的目标是最小化在某种提议分布 $\mu(a|s)$ 下的Q值，同时最大化从数据集 $\mathcal{D}$ 中抽样得到的动作的Q值：

$R_{CQL}(\theta) = \mathbb{E}_{s \sim \mathcal{D}} \left[ \mathbb{E}_{a \sim \mu(a|s)} [Q_\theta(s, a)] - \mathbb{E}_{a \sim \pi_b(a|s)} [Q_\theta(s, a)] \right]$

或者，使用对数-和-指数（log-sum-exp）形式（它隐含地覆盖所有动作），正则器可以写为：

$R_{CQL}(\theta) = \mathbb{E}_{s \sim \mathcal{D}} \left[ \log \sum_a \exp(Q_\theta(s,a)) - \mathbb{E}_{a \sim \pi_b(a|s)}[Q_\theta(s,a)] \right]$

在实际操作中，$\mathbb{E}_{a \sim \pi_b(a|s)}[Q_\theta(s,a)]$ 通过从数据集 $\mathcal{D}$ 中抽样得到的转移 $(s, a, r, s')$ 中的特定动作 $a$ 来近似。期望 $\mathbb{E}_{a \sim \mu(a|s)}$ 或对数-和-指数项需要对状态 $s$ 的多个动作进行抽样或评估Q值。这些动作可以均匀抽样，也可以从智能体当前策略中抽样。

### 正则器背后的原理

我们来分析一下正则器的作用：

1. **最小化OOD动作的Q值：** 第一项（例如，$\mathbb{E}_{a \sim \mu(a|s)} [Q_\theta(s, a)]$ 或 $\log \sum_a \exp(Q_\theta(s,a))$) 有效地*降低*了状态 $s$ 在数据集中*未*被充分代表的动作的估计Q值。对数-和-指数项的作用类似于一个“软最大值”，它惩罚那些除非被第二项抵消，否则任何动作都具有高Q值的状态。
2. **支持分布内动作的Q值：** 第二项（例如，$- \mathbb{E}_{a \sim \pi_b(a|s)} [Q_\theta(s, a)]$) 专门*提高*了在数据集 $\mathcal{D}$ 中与状态 $s$ 配对观察到的动作 $a$ 的Q值。

最小化整体的 $R_{CQL}(\theta)$ 项会迫使学习到的Q函数 $Q_\theta(s, a)$ 对数据中已见的动作赋予的值高于（或低于有下限的）数据中未见动作的值。TD损失 $L_{TD}(\theta)$ 同时确保这些分布内Q值与贝尔曼方程保持一致。



[交互图表：CQL对Q值的影响](https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-7-offline-reinforcement-learning/cql-offline#plot-10ly9vx)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "CQL对Q值的影响",
    "xaxis": {
      "title": "动作空间"
    },
    "yaxis": {
      "title": "Q值",
      "range": [
        -1,
        6
      ]
    },
    "legend": {
      "traceorder": "reversed"
    }
  },
  "data": [
    {
      "x": [
        "动作1 (数据内)",
        "动作2 (数据内)",
        "动作3 (出界)",
        "动作4 (出界)"
      ],
      "y": [
        4.5,
        3.5,
        5.0,
        4.0
      ],
      "type": "bar",
      "name": "标准Q学习 (可能的高估)",
      "marker": {
        "color": "#adb5bd"
      }
    },
    {
      "x": [
        "动作1 (数据内)",
        "动作2 (数据内)",
        "动作3 (出界)",
        "动作4 (出界)"
      ],
      "y": [
        4.6,
        3.6,
        2.0,
        1.5
      ],
      "type": "bar",
      "name": "CQL Q值 (保守)",
      "marker": {
        "color": "#4263eb"
      }
    }
  ]
}
```

</details>



> 标准Q学习可能会高估出界（OOD）动作的值。CQL增加了一个惩罚项，它会降低OOD动作的Q值，同时确保数据集中已见动作的值通过TD更新保持其依据。

### 保守Q学习的优点

- **直接减轻高估：** CQL直接解决OOD动作的Q函数高估问题，这是离线强化学习 (reinforcement learning)不稳定性的一个根本原因。
- **理论支撑：** 在某些条件下，CQL保证学习到的Q函数是真实Q函数的下界，从而促成保守的决策。
- **灵活性：** CQL可以很容易地与各种Q学习框架结合，包括DQN和SAC等演员-评论家方法。当与演员-评论家方法结合使用时，演员被训练以最大化学习到的保守Q函数。
- **出色的实证表现：** CQL在众多离线强化学习基准任务中展示了当前最佳或具有竞争力的结果。

### 实施考量

- **正则器的选择：** CQL正则器存在不同的形式，针对离散或连续动作空间量身定制。具体的实现细节对性能影响很大。
- **超参数 (parameter) (hyperparameter) $\alpha$：** 保守性权重 (weight) $\alpha$ 很重要。较小的 $\alpha$ 可能不足以惩罚OOD动作，而非常大的 $\alpha$ 可能过度抑制Q值，这可能会阻碍在数据分布内学习最优策略。通常需要对 $\alpha$ 进行调优。
- **动作抽样：** 对于OOD动作惩罚项（例如，$\mathbb{E}_{a \sim \mu(a|s)} [Q_\theta(s, a)]$），动作需要从合适的分布 $\mu$ 中抽样。常见的选择包括均匀随机抽样或从当前学习到的策略中抽样。

与策略限制方法相比，CQL提供了一种不同的理念。它允许策略潜在地代表任何动作，但依靠学习到的保守Q值来指导策略转向数据支持的动作。这种基于值的正则化 (regularization)提供了一种有效的替代方案，用于处理从固定数据集中学习所固有的挑战。

## 参考资料

- [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643) — Sergey Levine, Aviral Kumar, George Tucker, Justin Fu (2020)
  DOI: [10.48550/arXiv.2005.01643](https://doi.org/10.48550/arXiv.2005.01643)
  一篇关于离线强化学习领域的全面综述和教程，为包括CQL在内的各种方法提供了背景，并讨论了正在进行的研究挑战。

---

[上一节](09-%E5%80%BC%E5%87%BD%E6%95%B0%E6%AD%A3%E5%88%99%E5%8C%96%E6%96%B9%E6%B3%95.md) · [下一节](11-%E7%A6%BB%E7%BA%BF%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E5%AE%9E%E7%8E%B0%E6%96%B9%E9%9D%A2%E7%9A%84%E8%80%83%E9%87%8F.md)

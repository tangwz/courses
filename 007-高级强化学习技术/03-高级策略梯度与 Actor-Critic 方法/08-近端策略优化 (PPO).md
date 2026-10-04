# 近端策略优化 (PPO)

来源：[原文](https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-3-advanced-policy-gradients-actor-critic/ppo-algorithm)

[返回章节目录](README.md) · [返回课程目录](../README.md)

近端策略优化 (PPO) 是一种旨在解决强化学习 (reinforcement learning)中常见实际难题的算法，例如信任区域策略优化 (TRPO) 中存在的那些。尽管 TRPO 通过限制 KL 散度为单调策略提升提供了有力的理论保证，但其实际应用可能具有挑战性。TRPO 通常需要计算二阶导数，例如费舍尔信息矩阵，并解决带约束的优化问题，这通常使用共轭梯度法。PPO 简化了这一过程，旨在仅使用一阶优化实现 TRPO 的数据效率和可靠性能，从而使其在实施和调整上简单得多。PPO 已成为深度强化学习应用中的主要算法，因其在性能、简洁性和稳定性之间取得了平衡而备受推崇。

PPO 属于策略梯度方法，并在 Actor-Critic 框架下运行。与 TRPO 类似，它旨在限制策略在每个更新步骤中的变化幅度，以避免性能崩溃。PPO 不施加严格的 KL 散度约束，而是通常使用*截断代理目标*或*自适应 KL 惩罚*来避免过大的策略更新。

### 代理目标函数

回顾标准的策略梯度目标函数，它旨在最大化预期折扣回报。我们通常使用一个代理目标函数，该函数包含用于收集数据的新策略 $\pi_\theta$ 和旧策略 $\pi_{\theta_{old}}$ 之间的概率比：


$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$


标准代理目标函数，有时称为 CPI（保守策略迭代）目标函数，为：


$$
L^{CPI}(\theta) = \hat{\mathbb{E}}_t [ r_t(\theta) \hat{A}_t ]
$$


这里，$\hat{\mathbb{E}}_t$ 表示对一批收集到的时间步的经验平均值，而 $\hat{A}_t$ 是时间步 $t$ 处优势函数的一个估计（通常使用广义优势估计 GAE 计算，如前所述）。直接最大化 $L^{CPI}(\theta)$ 可能导致策略更新过大，特别是当 $r_t(\theta)$ 严重偏离 1 时。

### PPO 变体 1：截断代理目标

这是 PPO 最常见的变体。它通过截断概率比 $r_t(\theta)$ 来修改代理目标函数，使其保持在 1 附近的一个小区间内。截断操作取决于优势函数 $\hat{A}_t$ 的符号。

目标函数定义为：


$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) ]
$$


我们来分析一下 $\min$ 项：

1. **第一项：** $r_t(\theta) \hat{A}_t$ 是标准的 CPI 目标函数。
2. **第二项：** $\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t$。`clip` 函数将比率 $r_t(\theta)$ 限制在 $[1-\epsilon, 1+\epsilon]$ 的范围内。超参数 (parameter) (hyperparameter) $\epsilon$ (epsilon) 定义了截断范围（例如，通常为 0.1 或 0.2）。

$\min$ 操作确保最终的目标函数是未截断目标函数的一个下界（一个悲观估计）。

- **如果 $\hat{A}_t > 0$（优势为正）：** 目标函数变为 $\min(r_t(\theta) \hat{A}_t, (1+\epsilon) \hat{A}_t)$。这意味着如果策略更新会导致概率比 $r_t(\theta)$ 超过 $1+\epsilon$，则梯度会被截断，防止策略变化过大。我们仅能从将 $r_t(\theta)$ 增加到 $1+\epsilon$ 中受益。
- **如果 $\hat{A}_t < 0$（优势为负）：** 目标函数变为 $\min(r_t(\theta) \hat{A}_t, (1-\epsilon) \hat{A}_t)$。由于 $\hat{A}_t$ 是负值，这相当于取 $\max(r_t(\theta), 1-\epsilon) \hat{A}_t$。这会阻止策略更新导致概率比 $r_t(\theta)$ 降至 $1-\epsilon$ 以下。我们仅承担将 $r_t(\theta)$ 减小到 $1-\epsilon$ 所带来的惩罚。

本质上，截断消除了策略在单次更新中变化过大的动力，间接限制了新旧策略之间的 KL 散度。



[交互图表：PPO 截断目标直观理解 (eps=0.1)](https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-3-advanced-policy-gradients-actor-critic/ppo-algorithm#plot-1vf53n3)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4
      ],
      "y": [
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4
      ],
      "mode": "lines",
      "name": "未截断 (A>0)",
      "line": {
        "color": "#74c0fc"
      }
    },
    {
      "x": [
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.1,
        1.1,
        1.1
      ],
      "y": [
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.1,
        1.1,
        1.1
      ],
      "mode": "lines",
      "name": "已截断 (A>0, eps=0.1)",
      "line": {
        "color": "#f03e3e",
        "dash": "dash"
      }
    },
    {
      "x": [
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4
      ],
      "y": [
        -0.6,
        -0.7,
        -0.8,
        -0.9,
        -1.0,
        -1.1,
        -1.2,
        -1.3,
        -1.4
      ],
      "mode": "lines",
      "name": "未截断 (A<0)",
      "line": {
        "color": "#74c0fc",
        "dash": "dot"
      }
    },
    {
      "x": [
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4
      ],
      "y": [
        -0.9,
        -0.9,
        -0.9,
        -0.9,
        -1.0,
        -1.1,
        -1.2,
        -1.3,
        -1.4
      ],
      "mode": "lines",
      "name": "已截断 (A<0, eps=0.1)",
      "line": {
        "color": "#f03e3e",
        "dash": "longdash"
      }
    }
  ],
  "layout": {
    "title": {
      "text": "PPO 截断目标直观理解 (eps=0.1)"
    },
    "xaxis": {
      "title": {
        "text": "概率比 r(theta)"
      },
      "range": [
        0.5,
        1.5
      ]
    },
    "yaxis": {
      "title": {
        "text": "目标贡献（按 |A| 缩放）"
      },
      "range": [
        -1.5,
        1.5
      ]
    },
    "legend": {
      "y": 0.1,
      "x": 0.1
    }
  }
}
```

</details>



> PPO 中截断机制对正负优势（$\hat{A}_t$）的影响。虚线显示了当概率比 $r_t(\theta)$ 偏离 1.0 过远时，对目标函数的贡献是如何被限制的。

### PPO 变体 2：自适应 KL 惩罚

PPO 的另一种表述方式采用对 KL 散度的惩罚而非硬性截断。目标函数为：


$$
L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t [ r_t(\theta) \hat{A}_t - \beta \text{KL}[\pi_{\theta_{old}}(\cdot|s_t) || \pi_{\theta}(\cdot|s_t)] ]
$$


这里，$\beta$ 是一个系数，它根据新旧策略之间的 KL 散度来惩罚策略的较大偏差。$\beta$ 通常不是固定值，而是在训练过程中动态调整：

1. 计算策略更新后的实际 KL 散度 $d_{KL} = \hat{\mathbb{E}}_t[\text{KL}[\pi_{\theta_{old}}(\cdot|s_t) || \pi_{\theta}(\cdot|s_t)]]$。
2. 如果 $d_{KL}$ 大于目标值 $d_{targ}$，则增加 $\beta$（以便未来更严重地惩罚 KL 散度）。
3. 如果 $d_{KL}$ 小于 $d_{targ}$，则减小 $\beta$（以允许更大的更新）。

虽然这种方法在思路上更接近原始 TRPO 的约束，但 $\beta$ 的自适应方案增加了程序逻辑，并且从经验上看，截断代理目标通常表现同样出色，并且更容易实施和调整。

### 完整的 PPO 算法

PPO 通常使用 Actor-Critic 架构实现。最终优化的目标函数通常将截断策略代理损失 ($L^{CLIP}$) 与值函数损失 ($L^{VF}$) 和熵奖励 ($S$) 结合起来，以鼓励策略通过尝试新动作来收集信息：


$$
L^{PPO}(\theta) = \hat{\mathbb{E}}_t [ L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t) ]
$$


- $L^{VF}(\theta) = (V_\theta(s_t) - V_t^{targ})^2$ 是值函数 (critic) 的平方误差损失。$V_\theta(s_t)$ 是评论者的估计值，而 $V_t^{targ}$ 是目标值，通常是经验回报或 GAE 估计本身。
- $S[\pi_\theta](s_t)$ 是策略 $\pi_\theta$ 在状态 $s_t$ 下的熵。最大化熵有助于保持策略的多样性。
- $c_1$ 和 $c_2$ 分别是用于加权值损失和熵奖励的超参数 (parameter) (hyperparameter)。

整个算法流程如下：

1. **初始化：** 初始化 Actor 策略网络 $\pi_\theta$ 和 Critic 值网络 $V_\phi$（参数为 $\theta, \phi$）。通常，一些层是共享的。
2. **循环：** 重复多次迭代：
   a. **收集数据：** 在环境中运行当前策略 $\pi_{\theta_{old}}$（其中 $\theta_{old}$ 是更新前的参数），持续 $T$ 个时间步（或 $N$ 条轨迹），收集 $(s_t, a_t, r_{t+1}, s_{t+1})$。
   b. **计算优势：** 计算所有时间步 $t$ 的优势估计 $\hat{A}_t$，通常采用基于收集到的数据和当前值函数 $V_\phi$ 的 GAE。同时计算值目标 $V_t^{targ}$。
   c. **优化：** 重复 $K$ 个时期：
   i. 从收集到的数据中采样小批量数据。
   ii. 使用当前参数 $\theta, \phi$ 计算 PPO 目标函数 $L^{PPO}$。
   iii. 使用梯度上升优化器（如 Adam）对 $L^{PPO}$ 更新网络参数 $\theta, \phi$。
   d. 设置 $\theta_{old} \leftarrow \theta$。

在同一批数据上进行多轮（$K > 1$）小批量更新，与仅进行单次梯度更新后就丢弃数据的方法相比，提高了样本效率。

### PPO 的优点

**优点：**

- **简洁性：** 比 TRPO 更容易实现，主要依赖于标准的一阶优化器。
- **性能：** 在许多连续和离散控制基准上取得了顶尖或接近顶尖的成果。
- **稳定性：** 通常比简单的策略梯度方法更稳定，对超参数 (parameter) (hyperparameter)的敏感度也更低。
- **样本效率：** 每批数据可用于多次梯度更新，提高了样本效率。

**要点：**

- **超参数敏感性：** 虽然 PPO 常被认为是可靠的，但仍有核心的超参数需要调整，包括截断因子 $\epsilon$、学习率、GAE 参数 ($\lambda, \gamma$)、系数 $c_1, c_2$、小批量数据大小以及优化时期数 $K$。
- **缺乏严格保证：** 截断机制是一种启发式方法；与 TRPO 不同，它不提供单调改进的正式保证，尽管它在实践中表现非常出色。
- **实现细节：** 细小的实现选择（例如，值函数截断、奖励缩放、观测归一化 (normalization)、网络初始化）能显著影响性能，正如许多 PPO 实际实现和库中强调的那样。

总而言之，PPO 代表了策略梯度方法在实践中的一项重大进步。通过采用截断代理目标等机制，它提供了一种有效稳定策略更新的方法，且无需像 TRPO 那样增加实现开销，使其成为解决强化学习 (reinforcement learning)难题的通用且强大的工具。

---

[上一节](07-%E4%BF%A1%E4%BB%BB%E5%8C%BA%E5%9F%9F%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%20%28TRPO%29.md) · [下一节](09-%E8%BD%AF%E6%BC%94%E5%91%98-%E8%AF%84%E8%AE%BA%E5%AE%B6%20%28SAC%29.md)

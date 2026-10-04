# 对比：REINFORCE 与 A2C/A3C

来源：[原文](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-5-actor-critic-methods/reinforce-vs-actor-critic)

[返回章节目录](README.md) · [返回课程目录](../README.md)

直接对REINFORCE算法以及A2C和A3C等行动者-评论者（Actor-Critic）方法的基本架构进行比较很有益处。两者都旨在优化参数 (parameter)化策略 $\pi_\theta(a|s)$，但它们通过不同机制实现此目标，从而产生不同的性能特点。

REINFORCE作为一种纯蒙特卡洛策略梯度方法，根据从状态-动作对 $(s_t, a_t)$ 开始直到回合结束所经历的完整回报 $G_t$ 更新策略参数 $\theta$。其更新规则与 $\nabla_\theta \log \pi_\theta(a_t|s_t) G_t$ 成比例。

行动者-评论者方法，这里由A2C和A3C代表，保持两个不同的组成部分（或单个网络的两个头部）：

- **行动者（Actor）**：根据评论者（Critic）的反馈更新策略参数 $\theta$。
- **评论者（Critic）**：使用与时序差分（TD）学习相关的方法，学习一个值函数（通常是 $V_\phi(s)$，有时是 $Q_\phi(s, a)$）。

这种策略更新的学习信号生成方式的根本区别导致了几个重要的不同之处：

### 方差与稳定性

- **REINFORCE**：依赖于蒙特卡洛回报 ($G_t$)。由于 $G_t$ 是回合剩余部分中许多随机奖励和状态转换的总和，即使从相同的状态-动作对开始，其值也可能因轨迹不同而显著变化。学习信号 ($\\nabla_\theta \log \pi_\theta(a_t|s_t) G_t$) 的这种高方差会使训练过程不稳定且收敛缓慢。引入基线可以缓解这种情况，但其对完整回报的核心依赖依然存在。
- **A2C/A3C**：使用评论者（Critic）的值估计来计算一个低方差信号，通常是*优势函数* $A(s_t, a_t)$。一种常见形式是TD优势：$A(s_t, a_t) \approx r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$。这个信号仅取决于即时奖励和下一状态的估计值，而不是整个轨迹的回报。使用这种基于TD的估计显著降低了策略梯度更新的方差，相较于 $G_t$。策略更新与 $\nabla_\theta \log \pi_\theta(a_t|s_t) A(s_t, a_t)$ 成比例。这种较低的方差通常会带来更稳定、更快的学习。

### 偏差

- **REINFORCE**：蒙特卡洛回报 $G_t$ 是真实期望回报 $Q^{\pi_\theta}(s_t, a_t)$ 的*无偏*估计。虽然单个样本有噪声（高方差），但其期望是正确的。
- **A2C/A3C**：评论者（Critic）的值函数 $V_\phi(s)$ 是一个*估计值*。如果评论者（Critic）的估计不准确（训练早期尤其如此），由此产生的优势 $A(s_t, a_t)$ 是真实优势的*有偏*估计。这会给策略更新引入偏差。然而，这种偏差通常可以接受，因为它换来了方差的显著降低。随着评论者（Critic）的改进，偏差倾向于减小。这反映了机器学习 (machine learning)中经典的偏差-方差权衡。

### 样本效率

- **REINFORCE**：更新只在回合结束时发生，一旦该回合中所有步骤 $t$ 的完整回报 $G_t$ 已知。这可能效率不高，因为学习只在收集到可能很长的轨迹之后才发生。
- **A2C/A3C**：可以在每个步骤（或一小批步骤）之后更新行动者（Actor）和评论者（Critic），使用评论者（Critic）的TD误差和行动者（Actor）的优势估计。这使得从不完整的回合中学习成为可能，并且通常比蒙特卡洛REINFORCE具有更高的样本效率。信息通过值估计传播得更快。

### 实现复杂性

- **REINFORCE**：需要实现一个策略网络并计算蒙特卡洛回报。添加基线（如状态值函数）会增加复杂性，使其更接近行动者-评论者结构，但基线本身通常仍使用蒙特卡洛更新。
- **A2C/A3C**：需要管理行动者（Actor）和评论者（Critic）网络（它们可能共享某些层）。更新过程涉及协调两个组成部分的更新，可能使用不同的损失函数 (loss function)（行动者的策略梯度损失，评论者通常使用MSE损失）。A3C通过异步执行和并行工作器增加了复杂性。

### 总结比较

| 特点 | REINFORCE | A2C / A3C |
| --- | --- | --- |
| **算法类型** | 策略梯度（蒙特卡洛） | 行动者-评论者（策略梯度 + 值估计） |
| **策略更新信号** | 完整回报 $G_t$（通常带有基线） | 优势估计 $A(s, a)$（例如，TD优势） |
| **方差** | 高（由于 $G_t$） | 低（由于基于TD的值估计） |
| **偏差** | 无偏（对 $Q^{\pi_\theta}$ 的估计） | 有偏（由于 $V_\phi$ 的近似） |
| **样本效率** | 较低（每回合更新） | 较高（每步或每批次更新） |
| **架构** | 策略网络（可选基线网络） | 行动者网络 + 评论者网络（可共享层） |
| **稳定性** | 可能不稳定 | 通常更稳定 |

### 学习稳定性可视化

方差差异通常体现在学习曲线的平滑度上。虽然高度依赖于实现细节和超参数 (parameter) (hyperparameter)，但我们可能预期A2C/A3C会展现出比REINFORCE潜在的更不稳定的进展更稳定的改进。



[交互图表：学习进展：REINFORCE 对比 A2C](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-5-actor-critic-methods/reinforce-vs-actor-critic#plot-su517t)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        0,
        50,
        100,
        150,
        200,
        250,
        300,
        350,
        400,
        450,
        500
      ],
      "y": [
        -500,
        -400,
        -350,
        -200,
        -300,
        -150,
        -100,
        -180,
        -50,
        -120,
        -20
      ],
      "mode": "lines",
      "name": "REINFORCE",
      "line": {
        "color": "#ff8787"
      }
    },
    {
      "x": [
        0,
        50,
        100,
        150,
        200,
        250,
        300,
        350,
        400,
        450,
        500
      ],
      "y": [
        -500,
        -450,
        -380,
        -320,
        -250,
        -200,
        -150,
        -100,
        -80,
        -50,
        -30
      ],
      "mode": "lines",
      "name": "A2C",
      "line": {
        "color": "#74c0fc"
      }
    }
  ],
  "layout": {
    "title": "学习进展：REINFORCE 对比 A2C",
    "xaxis": {
      "title": "训练回合数"
    },
    "yaxis": {
      "title": "平均回合奖励"
    },
    "template": "plotly_white"
  }
}
```

</details>



> 训练期间奖励积累的比较。与REINFORCE相比，A2C由于梯度方差较低，通常表现出更平滑且可能更快的收敛。

### REINFORCE与A2C/A3C的选择

- 选择 **REINFORCE**（尤其是在带有基线的情况下），如果：
  - 简单性是首要考虑。
  - 回合相对较短，非终止状态不重要。
  - 你需要一个无偏梯度估计（尽管高方差在实际应用中可能会抵消这一益处）。
- 选择 **A2C/A3C**，如果：
  - 你需要更稳定和更快的收敛。
  - 样本效率很重要。
  - 你正在处理从不完整回合中学习有益的环境（例如，长期或连续任务）。
  - 你有足够的计算资源来处理稍微复杂的架构和更新规则（特别是A3C的并行性）。

在实践中，行动者-评论者方法（如A2C和A3C及其后续版本）已在许多应用中在很大程度上取代了基本的REINFORCE，因为评论者（Critic）所带来的偏差-方差权衡，它们在稳定性上有所改进，效率也更高。它们代表着将基于值和基于策略方法优点结合起来的重要一步。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: The MIT Press
  涵盖强化学习基础算法的综合性教材，包括蒙特卡洛方法、TD学习、策略梯度和Actor-Critic方法。
- [Asynchronous Methods for Deep Reinforcement Learning](https://proceedings.mlr.press/v48/mniha16.html) — Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Timothy Harley, David Silver, Koray Kavukcuoglu (2016)
  Journal: Proceedings of the 33rd International Conference on Machine Learning (ICML); Publisher: PMLR; Volume: 48; Pages: 1928-1937; DOI: [10.48550/arXiv.1602.01783](https://doi.org/10.48550/arXiv.1602.01783)
  介绍了异步优势Actor-Critic (A3C) 及其同步变体 (A2C)，展示了它们在深度强化学习任务中的有效性。

---

[上一节](05-Actor-Critic%20%E5%AE%9E%E7%8E%B0%E7%9A%84%E8%80%83%E9%87%8F.md) · [下一节](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%BC%80%E5%8F%91%20A2C%20%E5%AE%9E%E7%8E%B0.md)

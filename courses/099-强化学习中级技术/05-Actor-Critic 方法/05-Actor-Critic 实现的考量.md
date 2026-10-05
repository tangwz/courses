# Actor-Critic 实现的考量

来源：[原文](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-5-actor-critic-methods/actor-critic-implementation-considerations)

[返回章节目录](README.md) · [返回课程目录](../README.md)

实现 Actor-Critic 方法涉及多种设计选择，这些选择会显著影响性能和稳定性。Actor-Critic 方法以其分离的 Actor 和 Critic 组件为特点。构建这些智能体时会遇到的实际问题将在此讨论。

### 共享与独立网络参数 (parameter)

一个基本的架构决定是 Actor 和 Critic 应该使用完全独立的神经网络 (neural network)，还是共享某些层。

- **独立网络：** Actor ($\pi_\theta(a|s)$) 和 Critic ($V_\phi(s)$ 或 $Q_\phi(s, a)$) 被实现为不同的神经网络，每个都有自己的一组参数（分别为 $\theta$ 和 $\phi$）。它们使用各自的损失函数 (loss function)独立优化。
- **共享网络：** 一种常见做法，特别适用于图像等高维输入，是让网络共享初始层（例如，用于特征提取的卷积层），然后分支到单独的头部，分别用于策略输出（Actor）和价值输出（Critic）。

> Actor-Critic 方法中共享与独立网络架构的比较。共享网络可以提升效率，而独立网络提供更大的独立性。

**共享网络的优势：**

- **参数效率：** 减少参数总量，可能加快训练并降低内存使用。
- **特征共享：** 共享层学习到的表示对预测状态值和决定该状态下的最佳行动都有用。这种共享学习有时能带来更好的性能。

**共享网络的缺点：**

- **优化干扰：** Actor 损失和 Critic 损失的梯度会影响共享参数。如果目标没有很好地对齐 (alignment)，或损失量级差异很大，这可能产生冲突的更新。可能需要仔细调整学习率或损失权重 (weight)。
- **表示的妥协：** 共享层可能需要做出妥协以同时服务于 Actor 和 Critic 头部，这可能导致其中一个或两者产生次优的表示。

在实践中，共享参数很常见且通常有效，特别是在使用 A2C 或 A3C 等标准实现时。

### Actor 损失函数 (loss function)

Actor 的目标是调整其策略参数 (parameter) $\theta$ 以最大化预期回报。这通常通过对目标函数 $J(\theta)$ 执行梯度上升来实现。等效地，我们最小化该目标的负值，得到 Actor 损失 $L_{actor}(\theta)$。

使用策略梯度定理和 Critic 的估计来降低方差，梯度更新通常基于以下形式：


$$
\nabla_\theta J(\theta) \approx \mathbb{E}_t [ \nabla_\theta \log \pi_\theta(a_t|s_t) A_t ]
$$


这里，$A_t$ 表示在状态 $s_t$ 下执行行动 $a_t$ 的**优势**。这个优势通常使用 Critic 来估计。一个常用估计是基于 TD 误差 $\delta_t$：


$$
A_t \approx \delta_t = R_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$


其中，$V_\phi(s)$ 是由参数为 $\phi$ 的 Critic 网络提供的状态价值估计，$R_{t+1}$ 是执行行动 $a_t$ 后获得的奖励，$\gamma$ 是折扣因子。请注意，在计算梯度 $\nabla_\theta$ 时，优势项 $A_t$（以及 Critic 的输出 $V_\phi$）被视为一个常数基线，不依赖于 $\theta$。

需要最小化的 Actor 损失函数是：


$$
L_{actor}(\theta) = - \mathbb{E}_t [ \log \pi_\theta(a_t|s_t) A_t ]
$$


这个损失鼓励那些带来高于预期优势（正 $A_t$）的行动，并抑制那些带来低于预期优势（负 $A_t$）的行动。

### Critic 损失函数 (loss function)

Critic 的目标是学习状态价值函数 $V(s)$（或有时是行动价值函数 $Q(s, a)$）的准确估计。它的训练方式与 DQN 或其他基于价值的方法中的价值函数类似，通常使用时序差分 (TD) 学习。

Critic 最小化其当前估计 $V_\phi(s_t)$ 与目标值之间的差异。一个常用目标是 TD 目标，$Y_t = R_{t+1} + \gamma V_\phi(s_{t+1})$，假设回合尚未结束。Critic 损失 $L_{critic}(\phi)$ 通常是估计值与目标值之间的均方误差 (MSE)：


$$
L_{critic}(\phi) = \mathbb{E}_t [ (Y_t - V_\phi(s_t))^2 ] = \mathbb{E}_t [ (R_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t))^2 ]
$$


在计算梯度 $\nabla_\phi$ 时，目标 $Y_t$（如果进行自举，包括 $V_\phi(s_{t+1})$ 项）通常被视为一个固定目标，类似于 DQN 中目标网络的工作方式。这有助于稳定 Critic 的训练。在实践中，梯度仅针对 $V_\phi(s_t)$ 计算。

### 熵正则化 (regularization)

策略梯度方法，包括 Actor-Critic，有时可能会过早地收敛到次优的确定性策略。为了鼓励试探并防止策略过快地变得过于确定，通常会添加熵正则化。

熵 $H(\pi_\theta(\cdot|s_t))$ 衡量策略在状态 $s_t$ 下行动分布的随机性或不确定性。通过在目标函数中添加熵奖励（或从损失中减去熵惩罚），我们鼓励策略保持一定程度的随机性。

修改后的 Actor 损失变为：


$$
L_{actor+entropy}(\theta) = - \mathbb{E}_t [ \log \pi_\theta(a_t|s_t) A_t + \beta H(\pi_\theta(\cdot|s_t)) ]
$$


这里，$\beta$ 是一个超参数 (parameter) (hyperparameter)，控制熵正则化的强度。较高的 $\beta$ 值会鼓励更多的试探。熵的计算取决于策略分布：

- 对于**分类**分布（离散行动）：$H = - \sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)$。
- 对于**高斯**分布（连续行动）：它取决于标准差。

添加熵正则化是 A2C 和 A3C 等现代 Actor-Critic 实现中的一种标准技术。

### 同步与异步更新

最初的 A3C（异步优势 Actor-Critic）使用多个工作线程，每个线程与其自身的副本环境进行交互。这些工作线程独立计算梯度，并将其异步应用于共享的全局网络。这种异步性有助于消除更新所用数据的相关性，从而提升稳定性。

然而，A2C（优势 Actor-Critic）作为一种流行的替代方案出现。A2C 使用多个并行工作线程收集经验，但它执行同步更新。一个中央控制器从所有工作线程收集经验批次，基于聚合的批次计算梯度，并一步更新网络参数 (parameter)。所有工作线程随后接收更新后的参数。

A2C 通常能达到与 A3C 相似或更好的性能，由于批处理倾向于更有效地使用 GPU 等硬件，并且实现起来更简单。许多现代实现偏好同步 A2C 方法，通常使用向量 (vector)化环境来高效管理并行模拟。

### 处理状态和行动空间

网络架构必须适应特定的环境：

- **状态输入：** 如果状态由原始像素表示（例如，来自游戏屏幕），通常使用卷积神经网络 (neural network) (CNN) 作为初始层（通常在 Actor 和 Critic 之间共享）。如果状态是特征向量 (vector)，则多层感知机 (MLP) 适用。
- **行动输出（Actor）：**
  - **离散行动：** Actor 网络为每个可能的行动输出 logits（原始得分）。这些 logits 参数 (parameter)化一个分类分布，从中采样一个行动。
  - **连续行动：** Actor 网络输出定义连续概率分布的参数，最常见的是高斯分布。例如，它可能为每个行动维度输出均值 $\mu$ 和标准差 $\sigma$。然后从 $N(\mu, \sigma^2)$ 中采样一个行动。
- **价值输出（Critic）：** Critic 网络通常输出一个单一的标量值，表示估计的状态价值 $V(s)$。

这些实现选择、网络结构、损失函数 (loss function)、正则化 (regularization)和更新策略是相互依赖的，需要仔细考量和调整，才能在 Actor-Critic 智能体中实现有效的学习。通常需要对学习率（Actor 和 Critic 可能分开设置）、熵系数 $\beta$ 和折扣因子 $\gamma$ 等超参数 (hyperparameter)进行实验。

## 参考资料

- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  关于强化学习的综合性教材，详细介绍了Actor-Critic方法、策略梯度理论和价值函数近似。
- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu (2016)
  Journal: International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.1602.01783](https://doi.org/10.48550/arXiv.1602.01783)
  介绍了A3C算法，其特点是异步更新、共享网络架构和熵正则化，适用于深度Actor-Critic智能体。
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel (2016)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1506.02438](https://doi.org/10.48550/arXiv.1506.02438)
  介绍了广义优势估计（GAE），一种在策略梯度方法中减少方差的优势项（$A_t$）估计方法，常用于Actor-Critic。
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  描述了PPO，一种稳定且广泛采用的策略梯度算法，它使用带有裁剪目标的Actor-Critic架构。

---

[上一节](04-%E5%BC%82%E6%AD%A5%E4%BC%98%E5%8A%BF%E5%8F%82%E4%B8%8E%E8%80%85-%E8%AF%84%E4%BB%B7%E8%80%85%E7%AE%97%E6%B3%95%20%28A3C%29.md) · [下一节](06-%E5%AF%B9%E6%AF%94%EF%BC%9AREINFORCE%20%E4%B8%8E%20A2C-A3C.md)

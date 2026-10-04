---
course: "intro-to-reinforcement-learning"
chapter: "introduction-policy-gradient-methods"
lesson: "reinforce-algorithm"
sourceId: 1667
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-8-introduction-policy-gradient-methods/reinforce-algorithm"
title: "REINFORCE 算法"
description: "描述 REINFORCE 算法，这是一种基本的蒙特卡洛策略梯度方法。"
order: 3
plots: []
sourceHash: "9d3e11686da4c044cfd364edcc9a335844eedc23a7b8e0009f886b4ac239588e"
sourceCorrections: []
---

REINFORCE 是一种用于直接优化参数 (parameter)化策略 $\pi_\theta(a|s)$ 的特定算法。REINFORCE 也称为蒙特卡洛策略梯度，它提供了一种直接的方式，根据与环境交互的结果来调整策略参数 $\theta$。

REINFORCE 的核心理念很直观：增加导致良好结果的动作的概率，并减少导致不良结果的动作的概率。由于我们处理的是参数化策略，“增加概率”意味着根据策略性能梯度的指示方向调整参数 $\theta$。

REINFORCE 是一种*蒙特卡洛*方法，因为它只在观察到整个回合的完整回报后才学习。它不像时序差分方法（SARSA、Q-learning）那样在回合中间更新参数。相反，它会等到回合结束，计算从每个状态 $S_t$ 开始获得的回报 ($G_t$)，然后根据这些回报更新策略参数。

### REINFORCE 更新规则

令 $J(\theta)$ 表示我们要最大化的目标函数，通常是初始状态的期望回报。策略梯度定理提供了一种计算该目标函数相对于策略参数 (parameter) $\theta$ 的梯度的方法。REINFORCE 使用基于回合经验的梯度的采样版本。

对于单个回合，在回合结束后应用于参数 $\theta$ 的更新，考虑每个时间步 $t$，为：


$$
\theta \leftarrow \theta + \alpha \sum_{t=0}^{T-1} G_t \nabla_\theta \log \pi_\theta(A_t|S_t)
$$


或者，更新可以在回合的每个步骤 $t$ 增量地进行，使用在回合结束*后*计算的回报 $G_t$：


$$
\theta \leftarrow \theta + \alpha G_t \nabla_\theta \log \pi_\theta(A_t|S_t)
$$


让我们分析一下这个更新规则的组成部分：

- $\alpha$: 学习率，一个小的正标量，决定步长。
- $G_t$: 从时间步 $t$ 开始直到回合结束的总折扣回报。它的计算公式为 $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... + \gamma^{T-t-1} R_T$，其中 $T$ 是终止时间步，$\gamma$ 是折扣因子。这个 $G_t$ 值表示在状态 $S_t$ 中采取动作 $A_t$ 后（并随后遵循策略 $\pi_\theta$）实际经历的结果。
- $\nabla_\theta \log \pi_\theta(A_t|S_t)$: 这是在状态 $S_t$ 中采取特定动作 $A_t$ 的概率的对数，根据当前策略参数化 $\theta$ 计算出的梯度。这个项常被称为“资格向量 (vector)”。它指向参数空间中能最快增加在状态 $S_t$ 中选择动作 $A_t$ 的对数概率的方向。

该更新规则本质上是将参数 $\theta$ 推向由 $\nabla_\theta \log \pi_\theta(A_t|S_t)$ 指定的方向，并由回报 $G_t$ 进行缩放。

- 如果 $G_t$ 为正（粗略地说，好于平均结果），则更新会增加将来在状态 $S_t$ 中采取动作 $A_t$ 的概率。
- 如果 $G_t$ 为负（差于平均结果），则更新会减少将来在状态 $S_t$ 中采取动作 $A_t$ 的概率。

更新的幅度与回报 $G_t$ 成正比，因此导致明显更好或更差结果的动作会导致更大的参数调整。

### REINFORCE 算法步骤

以下是 REINFORCE 算法的典型概述：

1. **初始化**：选择一个策略参数 (parameter)化 $\pi_\theta(a|s)$（例如，一个带有参数 $\theta$ 的神经网络 (neural network)），并随机或使用预定权重 (weight)初始化 $\theta$。设置学习率 $\alpha$。
2. **循环（针对每个回合）**：
   a. **生成回合**：通过遵循当前策略 $\pi_\theta$ 运行一个完整的回合。这意味着从 $S_0$ 开始，选择 $A_0 \sim \pi_\theta( \cdot | S_0)$，观察 $R_1, S_1$，选择 $A_1 \sim \pi_\theta( \cdot | S_1)$，观察 $R_2, S_2$，依此类推，直到达到终止状态 $S_T$。存储状态、动作和奖励的序列：$(S_0, A_0, R_1, S_1, A_1, R_2, ..., S_{T-1}, A_{T-1}, R_T)$。
   b. **计算回报**：对于回合中的每个时间步 $t$（从 $t = 0$ 到 $T-1$）：
   计算回报 $G_t = \sum_{k=t+1}^{T} \gamma^{k-t-1} R_k$。
   c. **更新参数**：对于回合中的每个时间步 $t$（从 $t = 0$ 到 $T-1$）：
   计算梯度 $\nabla_\theta \log \pi_\theta(A_t|S_t)$。
   更新策略参数：$\theta \leftarrow \theta + \alpha G_t \nabla_\theta \log \pi_\theta(A_t|S_t)$。
3. **重复**：返回步骤 2 以生成更多回合并从中学习。

更新可以在一个回合内的所有步骤中累积并在结束时一次性应用，也可以在计算每个 $G_t$ 后增量应用。现代深度学习 (deep learning)框架通常通过自动微分来方便地计算 $\nabla_\theta \log \pi_\theta(A_t|S_t)$。

### 策略的表示

策略 $\pi_\theta(a|s)$ 需要是一个函数，它以状态 $s$ 和参数 (parameter) $\theta$ 作为输入，并输出每个可能动作 $a$ 的概率。对于离散动作空间，一种常见的方法是使用神经网络 (neural network)输出每个动作的分数（logits），然后通过 softmax 函数将这些分数转换为概率：


$$
\pi_\theta(a|s) = \text{softmax}(\text{NN}_\theta(s))_a = \frac{e^{\text{logit}_a}}{\sum_{b} e^{\text{logit}_b}}
$$


其中 $\text{NN}_\theta(s)$ 是神经网络对于状态 $s$ 的输出向量 (vector)，$\text{logit}_a$ 是对应于动作 $a$ 的元素。然后可以计算梯度 $\nabla_\theta \log \pi_\theta(a|s)$ 相对于网络参数 $\theta$ 的值。

### 优点与缺点

REINFORCE 简单，并为更高级的策略梯度方法奠定了基础。

**优点：**

- 能够自然地学习随机策略。
- 通过适当参数 (parameter)化策略（例如，输出高斯分布的参数）来处理连续动作空间。
- 与其他一些方法相比，相对容易理解和实现。

**缺点：**

- **高方差：** 最大的缺点。更新取决于蒙特卡洛回报 $G_t$，即使使用相同的策略，仅仅由于环境或策略本身的随机性，它在不同回合之间也可能显著变化。这种高方差使得学习缓慢且不稳定。
- **样本效率低下：** 学习只在回合结束时发生，可能需要许多回合才能取得显著进展，特别是对于长回合。
- **收敛性：** 有时会收敛到局部最优而不是全局最优。

高方差问题尤为重要。想象一个回合，其中大多数动作都很好，但临近结束时一个随机的不幸事件导致所有前面步骤的总回报 $G_t$ 非常低。REINFORCE 会错误地惩罚该回合中较早采取的所有动作。这促使了基线（baselines）和 Actor-Critic 方法等技术的发展，我们将在接下来讨论这些技术，以减轻这种方差并提高学习的稳定性和效率。

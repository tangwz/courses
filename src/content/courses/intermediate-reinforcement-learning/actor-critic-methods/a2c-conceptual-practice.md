---
course: "intermediate-reinforcement-learning"
chapter: "actor-critic-methods"
lesson: "a2c-conceptual-practice"
sourceId: 5170
sourceUrl: "https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-5-actor-critic-methods/a2c-conceptual-practice"
title: "实践：开发 A2C 实现"
description: "概述实现 A2C 智能体所需的步骤和组成部分。"
order: 7
plots: []
sourceHash: "d964def462db6d9fa6530e647f682b92bbc31c64029a6cc040a04bc599b5cb9c"
sourceCorrections: []
---

构建 A2C 智能体需要概述具体步骤，重点关注 Actor 和 Critic 组成部分如何交互和学习。此应用实践有助于理解 Actor-Critic 方法，特别是优势 Actor-Critic (A2C) 的结构和基本原理。

设想我们需要为 CartPole 或简单网格之类的环境实现 A2C。我们将侧重于结构和数据流，而非具体的代码，假设你熟悉 TensorFlow 或 PyTorch 等深度学习 (deep learning)框架。

### 核心组成部分

一个 A2C 智能体主要由两个神经网络 (neural network)组成，它们可能共享一些层：

1. **Actor 网络**：

   - **输入**：环境状态 $s$。
   - **输出**：可能动作的概率分布（对于离散动作空间）或某个分布的参数 (parameter)，例如均值和标准差（对于连续动作空间）。这代表了策略 $\pi_\theta(a|s)$。
   - **架构**：通常是几层隐藏层（例如，带 ReLU 激活的全连接层）。最后一层使用 softmax 激活来生成离散动作的概率。
   - **参数**：由 $\theta$ 表示。
2. **Critic 网络**：

   - **输入**：环境状态 $s$。
   - **输出**：一个单个标量值，代表估计的状态值 $V_\phi(s)$。
   - **架构**：与 Actor 类似，常共享初始层以有效处理状态表示。最后一层通常是单个线性神经元（无激活或线性激活）来输出值。
   - **参数**：由 $\phi$ 表示。

### 共享层架构

Actor 和 Critic 共享初始层是常见做法。这使得两个组成部分都能从共享的输入状态表示中获益，可能会提升学习效率并减少参数 (parameter)总数。

> 图示了常见的 A2C 架构，其中处理状态的初始层在 Actor 和 Critic 头之间共享。

### 训练循环

典型的 A2C 训练过程涉及重复执行以下步骤：

1. **交互并收集数据**：让智能体（使用当前的 Actor 策略 $\pi_\theta$）与环境交互固定数量的步数或回合。存储每个步骤 $t$ 收集到的转移 $(s_t, a_t, r_{t+1}, s_{t+1})$。
2. **计算优势估计**：对于收集批次中的每个转移 $(s_t, a_t, r_{t+1}, s_{t+1})$：

   - 从 Critic 获取当前状态值估计 $V_\phi(s_t)$。
   - 从 Critic 获取下一个状态值估计 $V_\phi(s_{t+1})$。如果 $s_{t+1}$ 是终止状态，则使用 $V_\phi(s_{t+1}) = 0$。
   - 计算 TD 误差（在基本 A2C 中常被用作优势估计）：
     
     $$
     \delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
     $$
     
   - 优势估计 $A(s_t, a_t)$ 在这种基本形式下就是 $\delta_t$。注意：更高阶的方法，例如广义优势估计 (GAE)，可以提供更优的估计，但使用相同的原理。
3. **计算价值目标**：训练 Critic 在步骤 $t$ 的目标是奖励和下一个状态值的折现和：

   
   $$
   y_t = r_{t+1} + \gamma V_\phi(s_{t+1})
   $$
   
4. **计算损失**：

   - **Actor 损失 (策略损失)**：旨在提高带来高于预期回报（正优势）的动作的概率。它使用所采取动作的负对数概率，并根据优势加权计算，通常包含一个熵奖励以鼓励更多尝试：
     
     $$
     L_{actor}(\theta) = - \sum_t \left( \log \pi_\theta(a_t|s_t) \cdot A(s_t, a_t) + \beta H(\pi_\theta(\cdot|s_t)) \right)
     $$
     
     其中 $A(s_t, a_t)$ 为优势（在此梯度计算中被视为常数），$H$ 为策略熵，$\beta$ 为熵系数。
   - **Critic 损失 (价值损失)**：旨在使 Critic 的价值估计 $V_\phi(s_t)$ 更接近计算出的目标 $y_t$。通常使用均方误差 (MSE)：
     
     $$
     L_{critic}(\phi) = \sum_t (y_t - V_\phi(s_t))^2
     $$
     
   - **总损失**：通常是 Actor 和 Critic 损失的加权和：
     
     $$
     L_{total} = L_{actor}(\theta) + c \cdot L_{critic}(\phi)
     $$
     
     其中 $c$ 是价值损失的加权因子（例如，0.5）。
5. **执行梯度更新**：

   - 计算 $L_{total}$ 相对于网络参数 (parameter)（Actor 部分为 $\theta$，Critic 部分为 $\phi$）的梯度。如果层是共享的，来自两种损失的梯度都会回传到共享参数。
   - 使用 Adam 等优化器更新参数：
     
     $$
     \theta \leftarrow \theta - \alpha_\theta \nabla_\theta L_{total}
     $$
     
     
     $$
     \phi \leftarrow \phi - \alpha_\phi \nabla_\phi L_{total}
     $$
     
     其中 $\alpha_\theta$ 和 $\alpha_\phi$ 分别是 Actor 和 Critic 的学习率（它们可以相同）。
6. **重复**：返回步骤 1，直到智能体达到期望的性能水平。

### 实现考量

- **批处理**：A2C 通常会在执行更新之前收集一批经验（例如，16、32 或更多步），比逐步更新更稳定。
- **归一化 (normalization)**：归一化状态或优势有时可以提升训练稳定性，尽管这会增加复杂性。
- **超参数 (parameter) (hyperparameter)调整**：找到学习率、折扣因子 $\gamma$、熵系数 $\beta$ 和价值损失权重 (weight) $c$ 的适合值对于良好的性能通常是必要的。
- **环境处理**：确保正确处理回合终止（重置环境，将终止状态的值置零）。

通过梳理这些组成部分和信息流，你为实现和调试各种强化学习 (reinforcement learning)任务的 A2C 智能体奠定了坚实的基础。下一步是将这种结构转化为代码，使用你选择的深度学习 (deep learning)库。

## 参考资料

- [Asynchronous Methods for Deep Reinforcement Learning](http://proceedings.mlr.press/v48/mniha16.pdf) — Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu (2016)
  Journal: Proceedings of The 33rd International Conference on Machine Learning; Publisher: PMLR; Volume: 48; Pages: 1928-1937; DOI: [10.5591/978-1-57735-989-2-icml2016-0428](https://doi.org/10.5591/978-1-57735-989-2-icml2016-0428)
  介绍了异步优势演员-评论家（A3C）算法，它是同步A2C的概念前身和基础算法，详细阐述了演员-评论家架构、带熵正则化的策略梯度以及价值函数学习。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本全面的教材，涵盖了强化学习的理论基础，包括演员-评论家方法、策略梯度和价值函数近似的详细解释。
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438.pdf) — John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel (2016)
  Journal: International Conference on Learning Representations (ICLR 2016); DOI: [10.48550/arXiv.1506.02438](https://doi.org/10.48550/arXiv.1506.02438)
  提出了广义优势估计（GAE），一种改进的优势函数估计方法，常用于现代演员-评论家算法（如A2C）中，以提升性能和稳定性。
- [Stable Baselines3: Reliable Reinforcement Learning Implementations](https://www.jmlr.org/papers/volume22/20-1364/20-1364.pdf) — Antonin Raffin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Lee (2021)
  Journal: Journal of Machine Learning Research; Publisher: Microtome Publishing; Volume: 22; Pages: 1-8; DOI: [10.5555/3524938.3525049](https://doi.org/10.5555/3524938.3525049)
  介绍了Stable Baselines3，一个流行且维护良好的库，提供了各种强化学习算法（包括A2C）的稳健实现，是实际代理开发的实践指南和参考。

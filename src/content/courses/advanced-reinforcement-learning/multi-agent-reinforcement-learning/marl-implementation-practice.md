---
course: "advanced-reinforcement-learning"
chapter: "multi-agent-reinforcement-learning"
lesson: "marl-implementation-practice"
sourceId: 3482
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-6-multi-agent-reinforcement-learning/marl-implementation-practice"
title: "MARL实现实践"
description: "在多智能体环境中动手实践MARL算法（例如IQL、MADDPG）的实现。"
order: 11
plots: []
sourceHash: "61ceb422149028cc2f63a9c017e40dcf4b59bab87853609ca924df87c4002117"
sourceCorrections: []
---

多智能体强化学习 (reinforcement learning) (MARL) 的实际实现，需要仔细考虑环境配置、算法选择以及多智能体系统固有的具体难题。实现 MARL 算法的指导侧重常见模式与注意点。

### 配置MARL环境

在编写智能体代码之前，你需要一个合适的多智能体环境。与单智能体环境（如OpenAI Gym的经典API）不同，MARL环境必须同时管理多个智能体，为每个智能体提供观测，接收它们的动作，并返回各自的奖励。

PettingZoo 等框架已成为 MARL 研究与开发的标准。PettingZoo 提供多种环境（经典游戏、粒子模拟、机器人），并配有一致的API，专为多智能体交互设计。它的API通常涉及遍历智能体、获取各自的观测、通过联合动作推进环境，以及接收各自的奖励和完成标志。

```python
# PettingZoo 用法示例
import pettingzoo.mpe as mpe

env = mpe.simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=100, continuous_actions=False)
observations, infos = env.reset()

while env.agents: # 当智能体活跃时循环
# 智能体决策循环
    actions = {}
    for agent_id in env.agents:
        # 获取特定智能体的观测
        agent_obs = observations[agent_id]
        # 策略根据观测选择动作
        actions[agent_id] = policy(agent_obs, agent_id) # 替换为你自己的智能体策略

    # 使用所有智能体的动作推进环境
    observations, rewards, terminations, truncations, infos = env.step(actions)

    # 处理奖励，存储转移数据等
    # ... 处理终止和截断 ...

env.close()
```

熟悉你所选环境框架的特定API，留意智能体ID的处理方式、观测和动作的组织方式（通常是把智能体ID映射到数据的字典），以及如何为单个智能体而非整个回合提供终止/截断信号。

### 基准方法：独立学习器（例如 IQL）

最简单的方法是使用标准的单智能体算法（如DQN或DDPG），将每个智能体视为一个独立学习器。每个智能体 $i$ 都维护自己的策略 $\pi_i(a_i | o_i)$ 以及可能的价值函数 $Q_i(o_i, a_i)$，它们仅使用其局部观测 $o_i$ 和奖励 $r_i$ 进行训练。在使用Q学习变体时，这通常被称为独立Q学习（IQL）。

**实现概要 (IQL)：**

1. **初始化：** 创建 $N$ 个独立的DQN智能体，每个智能体都有自己的Q网络、目标网络，以及可能有的回放缓冲区。
2. **数据收集：** 在每个环境步骤中：
   - 每个智能体 $i$ 观测 $o_i$。
   - 每个智能体 $i$ 根据其策略 $\pi_i$ 选择动作 $a_i$（例如，在其 $Q_i$ 上使用 epsilon-greedy）。
   - 执行联合动作 $(a_1, ..., a_N)$。
   - 接收各自的下一个观测 $o'_i$ 和奖励 $r_i$。
   - 将转移 $(o_i, a_i, r_i, o'_i)$ 存储在智能体 $i$ 的回放缓冲区中。
3. **训练：** 定期地从每个智能体的回放缓冲区中采样批次，并使用标准的DQN损失更新其Q网络：
   
   $$
   L(\phi_i) = \mathbb{E}_{(o_i, a_i, r_i, o'_i) \sim D_i} \left[ (r_i + \gamma \max_{a'} Q_{\text{目标}}(o'_i, a'; \phi_i^-) - Q(o_i, a_i; \phi_i))^2 \right]
   $$
   
   定期更新目标网络 $\phi_i^-$。

尽管易于实现，独立学习常常表现不佳，因为每个智能体都将环境感知为非平稳的，这源于其他智能体策略的变化。这违反了标准强化学习 (reinforcement learning)算法的马尔可夫假设。然而，它可作为一个有用的基准方法。

### 集中训练与去中心化执行 (CTDE)

CTDE方法旨在通过使用集中式信息（如其他智能体的观测或动作）来减轻训练过程中的非平稳性，同时确保执行仅依赖局部信息。MADDPG是一个典型的例子。

**实现概要 (MADDPG)：**

MADDPG将DDPG扩展到多智能体环境。每个智能体 $i$ 有一个演员网络 $\pi_i(o_i; \theta_i)$ 产生确定性动作 $a_i$，以及一个集中式评论家网络 $Q_i(s, a_1, ..., a_N; \phi_i)$，它估计给定全局状态 $s$ 某种表示（可以是所有观测 $(o_1, ..., o_N)$ 的拼接）下联合动作 $(a_1, ..., a_N)$ 的价值。

1. **初始化：**
   - 创建 $N$ 个演员网络 $\pi_i(o_i; \theta_i)$ 和 $N$ 个目标演员网络 $\pi'_i(o_i; \theta'_i)$。
   - 创建 $N$ 个评论家网络 $Q_i(s, a_1, ..., a_N; \phi_i)$ 和 $N$ 个目标评论家网络 $Q'_i(s, a_1, ..., a_N; \phi'_i)$。
   - 初始化一个共享的回放缓冲区 $D$，用于存储完整的转移数据：$(s, o_1..N, a_1..N, r_1..N, s', o'_1..N)$。
2. **数据收集：**
   - 每个智能体 $i$ 观测 $o_i$。
   - 每个智能体 $i$ 选择动作 $a_i = \pi_i(o_i; \theta_i) + \text{噪声}$。
   - 执行联合动作 $(a_1, ..., a_N)$。
   - 接收各自的奖励 $r_i$ 和下一个观测 $o'_i$，以及下一个全局状态 $s'$。
   - 将完整的转移元组存储在共享缓冲区 $D$ 中。
3. **训练 (从 $D$ 中采样批次):**
   - **更新评论家网络：** 对于每个智能体 $i$，计算目标Q值：
     
     $$
     y_i = r_i + \gamma Q'_i(s', a'_1, ..., a'_N; \phi'_i) \quad \text{其中 } a'_j = \pi'_j(o'_j; \theta'_j)
     $$
     
     最小化评论家 $i$ 的TD误差：
     
     $$
     L(\phi_i) = \mathbb{E} \left[ (y_i - Q_i(s, a_1, ..., a_N; \phi_i))^2 \right]
     $$
     
   - **更新演员网络：** 对于每个智能体 $i$，使用采样的策略梯度更新其策略：
     
     $$
     \nabla_{\theta_i} J(\theta_i) \approx \mathbb{E} \left[ \nabla_{\theta_i} \pi_i(o_i) \nabla_{a_i} Q_i(s, a_1, ..., a_N; \phi_i)|_{a_i=\pi_i(o_i)} \right]
     $$
     
   - **软更新目标网络：** 对所有目标网络进行软更新。

> MADDPG等典型CTDE架构中的数据流。演员网络使用局部观测进行执行。集中式评论家网络在训练期间使用全局状态和所有动作，为演员网络提供稳定的学习信号。

### 实现时的实际考虑

- **参数 (parameter)共享：** 如果智能体是同质的（相同的观测/动作空间和目标），你可以通过让智能体共享其网络（演员和/或评论家网络）的权重 (weight)来显著提升样本效率并减少参数数量。实现这一点的方法是在前向传播时为多个智能体使用相同的网络实例，并在反向传播 (backpropagation)时累积/平均梯度。
- **网络架构：** 根据观测空间使用合适的网络类型（MLP、CNN、RNN）。对于集中式评论家网络，考虑如何最好地将全局状态和所有动作组合作为输入（例如，拼接后接MLP）。
- **库与框架：** 利用MARL库，如RLlib（它支持MARL和各种算法，包括MADDPG、基于PPO的MARL）、EPyMARL或MARLlib。这些库通常处理管理多个智能体、分布式执行和标准算法实现的复杂性。
- **调试：** MARL调试具有挑战性。分别监控每个智能体的奖励、演员和评论家网络的损失，并观察涌现 (emergence)行为。非平稳性可能导致不稳定；检查所有智能体的学习是否都取得进展，或者是否有某些智能体的策略崩溃。在环境中可视化智能体行为通常是必不可少的。

实现MARL算法需要仔细管理智能体间的交互、数据流和训练过程。从IQL开始提供了一个基准，而转向MADDPG等CTDE方法则解决了MARL的基本难题，为更复杂的合作或竞争行为提供了可能。尝试不同的环境、架构和超参数 (hyperparameter)对于MARL的成功应用非常重要。

## 参考资料

- [PettingZoo Documentation](https://pettingzoo.farama.org/) — The PettingZoo Team (2024)
  Publisher: Farama Foundation
  PettingZoo 库的官方文档，提供了多智能体环境 API 的详细信息和示例。
- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf) — Ryan Lowe, YI WU, Aviv Tamar, Jean Harb, OpenAI Pieter Abbeel, Igor Mordatch (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 30; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 6379-6390; DOI: [10.5591/978-1-57753-625-2](https://doi.org/10.5591/978-1-57753-625-2)
  介绍了 MADDPG 算法，这是一种多智能体强化学习中集中训练分散执行 (CTDE) 的基础方法。
- [Multi-Agent Reinforcement Learning: A Comprehensive Survey](https://ieeexplore.ieee.org/document/9044949) — Yaodong Yang, Ronghua Liu, Bin Shi, Xu Han, and Guoliang Fan (2020)
  Journal: IEEE Transactions on Cybernetics; Publisher: IEEE; Volume: 50; Pages: 3626-3642; DOI: [10.1109/TCYB.2020.2979203](https://doi.org/10.1109/TCYB.2020.2979203)
  一份广泛的调查报告，涵盖了各种多智能体强化学习范式，包括独立学习基线、CTDE 方法和常见挑战。
- [RLlib Documentation](https://docs.ray.io/en/latest/rllib/index.html) — The Ray Project Developers (2024)
  Publisher: Ray Project
  RLlib 的官方文档，这是一个可扩展的强化学习库，支持各种多智能体强化学习算法及其实现。

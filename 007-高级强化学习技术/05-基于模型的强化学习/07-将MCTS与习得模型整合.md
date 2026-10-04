# 将MCTS与习得模型整合

来源：[原文](https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-5-model-based-reinforcement-learning/mcts-learned-models)

[返回章节目录](README.md) · [返回课程目录](../README.md)

学习环境动态 ($P_\theta(s'|s, a)$) 和奖励函数 ($R_\phi(s, a, s')$) 的方法可用于决策。虽然简单的轨迹采样能提供一些益处，但将习得模型与蒙特卡洛树搜索（MCTS）这类强效规划算法整合，能为前瞻规划提供更有条理的方式，可能带来明显更强的策略。

MCTS擅长处理大的搜索空间，它通过在模拟轨迹中智能地平衡探索与利用。在其传统应用中（如游戏），MCTS依赖于一个完美的模拟器或环境的生成模型。在模型基强化学习 (reinforcement learning)中，我们用习得的、近似的模型替代这个完美的模拟器。这使得智能体能利用其内在理解来“向前思考”，即使这种理解并非完美。

### 结合习得模型的MCTS循环

MCTS的核心过程保持不变，它由重复多次的四个步骤组成，用于构建一个以当前状态 $s_t$ 为根的搜索树：选择、扩展、模拟和反向传播 (backpropagation)。习得模型的主要整合点在扩展和模拟步骤中。

1. **选择：** 从根节点（代表当前状态 $s_t$）开始，根据树策略递归选择子节点，直到达到叶节点。常用的树策略是UCT（树上置信上限），它平衡了对估计值高的节点 $Q(s, a)$ 的利用和对访问次数少的节点 $N(s, a)$ 的尝试。选择过程本身不直接使用习得的动态模型，但依赖于通过确实使用了模型的模拟来更新的值。
2. **扩展：** 当选择过程到达代表状态 $s_L$ 的叶节点 $L$ 时，如果状态不是终止状态，并且节点尚未完全扩展（即并非所有动作都已尝试过），则选择一个未尝试的动作 $a$。使用习得的转移模型 $P_\theta(s'|s_L, a)$ 来采样一个可能的下一个状态 $s'$。创建一个代表 $s'$ 的新子节点。这是习得模型决定正在构建的搜索树结构的重要环节。如果 $P_\theta$ 是确定性的，则只有一个 $s'$; 如果是概率性的，我们采样一个结果。
3. **模拟（Rollout）：** 从新扩展的节点（状态 $s'$）开始，执行模拟或“Rollout”直到达到终止状态或预设的最大深度。此模拟通过重复以下步骤进行：

   - 根据默认的“Rollout”策略 $\pi_{rollout}$（通常很简单，例如随机动作选择）选择一个动作 $a_{sim}$。
   - 使用习得的转移模型采样下一个状态 $s'_{sim}$：$s'_{sim} \sim P_\theta(\cdot | s_{sim}, a_{sim})$。
   - 使用习得的奖励模型获得奖励 $r_{sim}$：$r_{sim} = R_\phi(s_{sim}, a_{sim}, s'_{sim})$，如果奖励模型是概率性的则采样获得。
     在此“Rollout”期间获得的累积折扣奖励 $G$ 是模拟的结果。这一步非常依赖习得的转移模型和奖励模型来生成未来的轨迹。
4. **反向传播：** 使用模拟结果 $G$ 更新从新扩展节点到根节点路径上所有节点的统计数据（访问次数 $N$ 和值估计 $Q$）。对于路径上因采取动作 $a$ 而到达的状态 $s$ 对应的节点，更新 $N(s,a)$ 和 $Q(s,a)$。通常，$Q(s,a)$ 会根据从状态 $s$ 采取动作 $a$ 后观察到的平均回报进行更新：

   
   $$
   N(s,a) \leftarrow N(s,a) + 1
   $$
   
   
   $$
   Q(s,a) \leftarrow Q(s,a) + \frac{G - Q(s,a)}{N(s,a)}
   $$
   

在重复执行这四个步骤多次后（计算预算通常以模拟次数或时间来衡量），智能体选择在环境中执行一个动作。这通常是与根节点访问次数最多的子节点对应的动作，因为访问次数常作为搜索找到最佳动作的指示。

> 运用习得环境模型的MCTS循环。习得的转移模型 $P_\theta$ 在扩展和模拟期间决定状态转移。习得的奖励模型 $R_\phi$ 在模拟期间提供奖励。

### 改进：学习策略和价值网络（AlphaZero风格）

将MCTS与习得组件整合方面的一项重要进展来自AlphaGo及其后续AlphaZero等系统。这些系统不再仅仅依靠随机“Rollout”进行模拟步骤和简单的UCT进行选择/扩展，而是使用与MCTS规划同步训练的神经网络 (neural network)：

- **策略网络 $\pi_\psi(a|s)$：** 这个网络预测从给定状态 $s$ 开始的有前景的动作。其输出可用于使MCTS选择偏向更有前景的分支，并可能引导扩展步骤，比普通UCT更有效地集中搜索工作。
- **价值网络 $V_\omega(s)$：** 这个网络估计从状态 $s$ 得到的预期未来回报。其输出可用于提前终止模拟步骤，用习得的价值估计 $V_\omega(s')$ 替代嘈杂的蒙特卡洛“Rollout”结果 $G$，可能减少方差和计算成本。

这些网络通常使用MCTS搜索本身生成的数据进行训练。MCTS的访问次数作为策略网络的训练目标（鼓励其预测MCTS更多尝试的动作），而MCTS搜索的最终结果（通常是源自搜索结果的改进价值估计）作为价值网络的目标。这创建了一个强大的反馈循环，更好的规划带来更好的网络，进而带来更优秀的规划。虽然实现这样的系统很复杂，但这展现了习得模型/函数与搜索算法之间整合的潜在程度。

### 优点与考量

将MCTS与习得模型整合有几个潜在的优点：

- **性能提升：** MCTS执行前瞻搜索，能找到比纯反应式策略更好的解决方案，尤其是在需要规划的任务中。
- **有导向的尝试（通过规划）：** 搜索过程本身可以看作是习得模型状态空间内一种有导向的尝试形式。
  "\* **潜在的样本效率提升：** 如果习得模型相当准确，通过MCTS生成模拟经验比与环境交互要快得多也便宜得多，可能减少对样本的需求（尽管模型学习本身需要样本）。"

然而，这种方法并非没有挑战：

- **模型精度至关重要：** MCTS规划的性能对习得模型 $P_\theta$ 和 $R_\phi$ 的精度高度敏感。模型中的误差会在“Rollout”期间累积，并误导搜索，导致不佳的动作选择。这通常被称为*模型偏差*或*累积模型误差*的问题。
- **计算成本：** 在每个时间步运行数千次MCTS模拟可能计算成本很高，可能限制在实时场景中的适用性。复杂性随期望的搜索深度和广度而变化。
- **平衡模型学习与规划：** 在投入资源改进模型与投入资源使用当前模型进行规划之间存在权衡。

将MCTS与习得模型整合代表了模型基RL中一种精妙的方法。它使得智能体能够将从经验中学习的好处与树搜索提供的深思熟虑的规划能力相结合，为解决复杂的序列决策问题提供了一个强大的工具，前提是模型精度和计算成本的挑战能够有效管理。

## 参考资料

- [Bandit based Monte-Carlo Planning](https://link.springer.com/chapter/10.1007/11871842_29) — Levente Kocsis and Csaba Szepesvári (2006)
  Journal: Machine Learning: ECML 2006, Lecture Notes in Computer Science; Publisher: Springer; Volume: 4212; Pages: 282-293; DOI: [10.1007/11871842_29](https://doi.org/10.1007/11871842_29)
  介绍了应用于树的上限置信度（UCT）算法，这是蒙特卡洛树搜索的一个核心组成部分。
- [A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go through Self-Play](https://science.sciencemag.org/content/sci/362/6419/1140.full.pdf) — David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dmitry Kummer, Thore Graepel, Timothy Lillicrap, Karen Simonyan, and Demis Hassabis (2018)
  Journal: Science; Publisher: American Association for the Advancement of Science (AAAS); Volume: 362; Pages: 1140-1144; DOI: [10.1126/science.aar6404](https://doi.org/10.1126/science.aar6404)
  介绍了AlphaZero算法，展示了深度神经网络与MCTS在通用游戏对弈中的结合。该工作与“复杂化”部分高度相关。

---

[上一节](06-%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B%E6%A0%91%E6%90%9C%E7%B4%A2%EF%BC%88MCTS%EF%BC%89%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86.md) · [下一节](08-%E6%A8%A1%E5%9E%8B%E9%A2%84%E6%B5%8B%E6%8E%A7%E5%88%B6%20%28MPC%29%20%E7%9A%84%E5%85%B3%E8%81%94.md)

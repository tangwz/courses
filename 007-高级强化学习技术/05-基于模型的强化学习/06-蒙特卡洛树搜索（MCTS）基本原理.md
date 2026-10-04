# 蒙特卡洛树搜索（MCTS）基本原理

来源：[原文](https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-5-model-based-reinforcement-learning/mcts-fundamentals)

[返回章节目录](README.md) · [返回课程目录](../README.md)

无模型方法直接从经验中学习，而基于模型的方法涉及到学习环境模型。一旦有了这样的模型，即使只是一个近似模型，如何运用它进行决策？蒙特卡洛树搜索（MCTS）提供了一个强大的规划框架，它使用生成模型或模拟器，使其非常适合基于模型的强化学习 (reinforcement learning)。MCTS 是一种启发式搜索算法，在由状态、动作和奖励定义的场景中，尤其是在状态空间庞大或复杂时，它在寻找最佳决策方面表现出色。它在游戏对弈中取得了显著成就，特别是 AlphaGo 击败围棋冠军，凸显了它的效用。

MCTS 的核心思想是模拟从当前状态出发的许多可能的未来轨迹，并逐步构建搜索树。从这些模拟中收集的统计数据用于估计采取不同动作的长期价值，指引搜索朝向状态-动作空间中更有希望的部分。这使得 MCTS 能够在试探（尝试访问较少的动作）和利用（侧重当前看起来最佳的动作）之间取得平衡。

典型的 MCTS 迭代包含四个步骤，这些步骤会重复多次，以构建搜索树并在做出决策前优化动作价值估计：

1. **选择:** 从根节点（代表当前状态）开始，根据*树策略*选择动作，向下遍历现有树。该策略旨在平衡对不确定分支的试探和对已知良好分支的利用。一种非常常见的树策略是基于树上置信上限（UCT）。在树中的状态节点 $s$ 处，UCT 选择使以下表达式最大化的动作 $a$：

   
   $$
   Q(s, a) + C \sqrt{\frac{\ln N(s)}{N(s, a)}}
   $$
   

   其中，$Q(s, a)$ 是在树中从状态 $s$ 采取动作 $a$ 的当前估计值（平均回报），这代表了利用项。$N(s)$ 是状态 $s$ 在搜索期间被访问的总次数，$N(s, a)$ 是从状态 $s$ 选择动作 $a$ 的次数。平方根项代表试探，它对相对于父状态访问次数尝试较少的动作给予奖励。$C$ 是一个控制试探程度的常数。此选择过程持续进行，直到到达叶节点（尚未完全展开或代表终止状态的节点）。
2. **扩展:** 如果选定的叶节点 $s_L$ 不是终止状态且有未尝试的动作，则选择一个未尝试的动作 $a$。使用环境模型（已知或学习到的）来模拟状态转换。这包括从 $P(s' | s_L, a)$ 中采样下一个状态 $s'$，并可能采样奖励 $R(s_L, a, s')$。在树中添加一个代表 $s'$ 的新子节点，通过动作 $a$ 从 $s_L$ 连接。
3. **模拟（“蒙特卡洛模拟”）:** 从新扩展的节点 $s'$ （如果选定的叶节点 $s_L$ 是终止节点或已完全扩展，则从 $s_L$）开始，执行一次模拟或“蒙特卡洛模拟”。这涉及从状态 $s'$ 遵循一个*默认策略*（通常很简单，例如随机动作选择，但也可能更复杂），直到达到终止状态或预定义的深度限制。在此模拟期间累积的总折扣奖励是其结果，我们称之为 $G$。
4. **反向传播 (backpropagation)（回溯）:** 模拟结果 $G$ 用于更新在选择和扩展阶段访问过的所有节点的统计数据（从新节点 $s'$ 回溯到根节点）。对于此路径上的每个状态-动作对 $(s, a)$，更新其访问次数 $N(s, a)$ 和价值估计 $Q(s, a)$。通常，$Q(s, a)$ 会更新为通过 $(s, a)$ 的模拟获得的回报 $G$ 的运行平均值：

   
   $$
   N(s, a) \leftarrow N(s, a) + 1
   $$
   
   
   $$
   Q(s, a) \leftarrow Q(s, a) + \frac{G - Q(s, a)}{N(s, a)}
   $$
   

   父状态的访问次数 $N(s)$ 也相应增加。

> 单次 MCTS 迭代的可视化。选择阶段（蓝色）根据 UCT 策略向下进行到节点 s6。扩展阶段（红色）添加了动作 a1 和状态 s7。模拟阶段（绿色，虚线）从 s7 开始运行，得到回报 G。回溯阶段（橙色，虚线）使用 G 更新选择路径上的统计数据。

这四个步骤（选择、扩展、模拟、回溯）会重复进行，通常在固定的计算预算下（例如，模拟次数或时间限制）。搜索结束后，从根节点选择的动作通常是具有最高估计 $Q(s_{root}, a)$ 值或最高访问次数 $N(s_{root}, a)$。

### MCTS 的重要组成部分

- **搜索树:** 存储状态-动作空间中已访问的部分。节点代表状态，边代表动作。节点或边会维护统计数据（$Q$ 值、访问次数 $N$）。
- **树策略（例如，UCT）:** 在选择阶段，它决定了*现有树内*的动作选择。它的作用对于在搜索过程中平衡试探和利用非常重要。
- **默认策略（蒙特卡洛模拟策略）:** 在模拟阶段，它用于*显式扩展树之外*的部分。它需要快速执行，因为会进行多次蒙特卡洛模拟。更好的默认策略通常可以提高 MCTS 性能，但即使是随机策略也能发挥出人意料的良好效果，因为 UCT 选择会将搜索精力集中。
- **环境模型/模拟器:** 这是 MCTS 用于扩展和模拟步骤的“引擎”。在基于模型的强化学习 (reinforcement learning)中，这正是所学到的状态转移模型 $P(s'|s, a)$ 和奖励函数 $R(s, a, s')$ 发挥作用的地方。MCTS 重复查询这个学习到的模型来模拟可能的未来情况。

### MCTS 的特点

MCTS 有几个吸引人的特性：

- **非对称增长:** 树的增长是不均匀的，它根据 UCT 策略将计算资源集中在被认为更有希望的区域。
- **随时可用性:** 搜索可以在任何时间点停止，并返回当前最佳动作（基于根节点的访问次数或Q值）。更多的计算通常会带来更好的决策。
- **模型要求:** 它依赖于能够访问环境的模拟器或生成模型。这可以是已知的真实环境动力学，或者，对于本章而言，一个*学习到的*模型。

然而，它也有一些需要考虑的方面：

- **计算成本:** 在每个决策点执行成千上万或数百万次模拟可能需要很高的计算量，特别是当模型预测或模拟步骤本身很慢时。
- **模型依赖性:** 规划的质量在很大程度上取决于用于模拟的模型准确性。学习到的模型中的错误可能会误导搜索。

在基于模型的强化学习 (reinforcement learning)背景下，MCTS 提供了一种规范的方式来利用学习到的动力学进行规划。MCTS 不仅仅是单步展望或简单的轨迹采样，它进行的是一种更具方向性、适应性的、面向未来的搜索，运用学习到的模型来评估动作的长期结果。从 MCTS 获得的见解可以用来直接选择动作，或者进一步改进策略或价值函数，这一点我们将在讨论 AlphaZero 等集成方法时看到。

## 参考资料

- [Bandit-based Monte-Carlo Planning](https://link.springer.com/chapter/10.1007/11871842_29) — Levente Kocsis, Csaba Szepesvári (2006)
  Journal: Machine Learning: ECML 2006, 17th European Conference on Machine Learning; Publisher: Springer; Volume: 4212; Pages: 282-293; DOI: [10.1007/11871842_29](https://doi.org/10.1007/11871842_29)
  介绍了蒙特卡洛树搜索（MCTS）选择阶段的核心组成部分--树上置信上限（UCT）算法。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  关于强化学习的综合性教材，在规划和基于模型的强化学习背景下对MCTS进行了详细介绍。
- [Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961) — David Silver, Aja Huang, Marcin Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Vedran Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, and Demis Hassabis (2016)
  Journal: Nature; Volume: 529; Pages: 484-489; DOI: [10.1038/nature16961](https://doi.org/10.1038/nature16961)
  一篇里程碑式的论文，展示了MCTS与深度学习结合在围棋（AlphaGo）中达到超人表现的成功应用。
- [A Survey of Monte Carlo Tree Search Methods](https://doi.org/10.1109/TCIAIG.2012.2186810) — Cameron Browne, Edward Powley, Daniel Whitehouse, Simon Lucas, Peter I. Cowling, Philipp Rohlfshagen, Stephen Tavener, Diego Perez, Spyridon Samothrakis, and Simon Colton (2012)
  Journal: IEEE Transactions on Computational Intelligence and AI in Games; Publisher: IEEE; Volume: 4; Pages: 1-43; DOI: [10.1109/TCIAIG.2012.2186810](https://doi.org/10.1109/TCIAIG.2012.2186810)
  一份全面综述，概述了MCTS算法、其理论基础和多种应用。

---

[上一节](05-%E4%BD%BF%E7%94%A8%E5%B7%B2%E5%AD%A6%E4%B9%A0%E6%A8%A1%E5%9E%8B%E8%BF%9B%E8%A1%8C%E8%A7%84%E5%88%92%EF%BC%9A%E8%BD%A8%E8%BF%B9%E9%87%87%E6%A0%B7.md) · [下一节](07-%E5%B0%86MCTS%E4%B8%8E%E4%B9%A0%E5%BE%97%E6%A8%A1%E5%9E%8B%E6%95%B4%E5%90%88.md)

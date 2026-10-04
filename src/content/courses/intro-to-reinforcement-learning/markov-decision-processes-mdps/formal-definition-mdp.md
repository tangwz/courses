---
course: "intro-to-reinforcement-learning"
chapter: "markov-decision-processes-mdps"
lesson: "formal-definition-mdp"
sourceId: 1581
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-2-markov-decision-processes-mdps/formal-definition-mdp"
title: "MDP的正式定义"
description: "定义MDP的组成部分：状态、动作、转移概率、奖励和折扣因子。"
order: 2
plots: []
sourceHash: "bcd7afad6de2c7cd2abc3b27f714e78acf937ff4cb880cd308533908f4632bd9"
sourceCorrections: []
---

强化学习 (reinforcement learning)包含一个智能体随时间与环境互动，学习做出能使累积奖励最大化的决策。为了准确地分析和解决这些问题，我们需要一个精确的数学架构。不确定性下序列决策最常用的架构是马尔可夫决策过程，简称MDP。

一个MDP提供了一种正式方式来描述强化学习问题中的环境。它假设环境满足马尔可夫性质，这意味着未来状态和奖励只取决于当前状态和动作，而不依赖于之前的全部历史。我们很快会再次讨论这个性质。

正式地，一个MDP由一个包含五个组成部分的元组定义：$(\mathcal{S}, \mathcal{A}, P, R, \gamma)$。让我们分别介绍每个部分：

### 状态空间 ($\mathcal{S}$)

状态空间 $\mathcal{S}$ 是环境可能处于的所有状态的集合。理想情况下，状态 $s \in \mathcal{S}$ 应包含做出最佳决策所需的所有相关环境信息。可以把它看作某个特定时刻的情况快照。

例如，在一个简单的网格游戏中，一个状态可以是智能体的 $(x, y)$ 坐标。在围棋游戏中，状态会描述棋盘上棋子的布局。状态可以是离散的（如网格游戏示例中，位置数量有限）或连续的（如机械臂的精确关节角度和速度）。状态空间的复杂性和规模显著影响哪些强化学习 (reinforcement learning)算法适用。

A一个重要假设是**马尔可夫性质**。它表明，转移到下一个状态 $s'$ 的概率*只*取决于当前状态 $s$ 和所采取的动作 $a$，与所有之前的状态和动作无关。数学上，如果 $S_t$ 是时间 $t$ 的状态，且 $A_t$ 是时间 $t$ 的动作：


$$
P(S_{t+1} = s' | S_t = s, A_t = a, S_{t-1}, A_{t-1}, ..., S_0, A_0) = P(S_{t+1} = s' | S_t = s, A_t = a)
$$


当前状态 $s_t$ 被认为包含历史中的所有必要信息。此性质大大简化了建模，因为我们不需要跟踪过去事件的整个序列，只需关注当前状态。

### 动作空间 ($\mathcal{A}$)

动作空间 $\mathcal{A}$ 是智能体可以从中选择的所有可能动作的集合。有时，可用动作的集合取决于当前状态 $s$，在这种情况下我们将其表示为 $\mathcal{A}(s)$。动作 $a \in \mathcal{A}$（或 $a \in \mathcal{A}(s)$）是智能体在给定时间步做出的决策。

在我们的网格中，动作可能是 $\mathcal{A} = \{\text{上, 下, 左, 右}\}$。对于控制室温的恒温器，动作可以是 $\mathcal{A} = \{\text{加热, 冷却, 关闭}\}$。动作与状态一样，可以是离散的（有限的选项集合）或连续的（例如，施加到踏板上的力的大小，用实数表示）。

### 转移概率函数 ($P$)

转移概率函数 $P$ 描述了环境的动态特性，常被称为模型。它指定了在智能体采取动作 $a$ 的情况下，从当前状态 $s$ 转移到下一个可能状态 $s'$ 的概率。它正式地表示为一个函数 $P: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow [0, 1]$：


$$
P(s' | s, a) = \Pr(S_{t+1} = s' | S_t = s, A_t = a)
$$


这个函数告诉我们在给定当前状态 ($s$) 和所做动作 ($a$) 的情况下，每个潜在结果 $s'$ 的可能性有多大。由于这些是概率，对于任何起始状态 $s$ 和动作 $a$，所有可能后继状态 $s'$ 的和必须等于 1：


$$
\sum_{s' \in \mathcal{S}} P(s' | s, a) = 1 \quad \text{对于所有 } s \in \mathcal{S}, a \in \mathcal{A}(s)
$$


转移函数描绘了环境的基本规则。如果对于某个特定的 $s'$，有 $P(s' | s, a) = 1$，则该 $(s, a)$ 对的转移是确定性的。更常见的情况是，环境是随机的，这意味着在相同状态下采取相同动作可能以不同概率导致不同结果（例如，机器人移动可能受车轮打滑影响）。

### 奖励函数 ($R$)

奖励函数 $R$ 定义了智能体从环境中获得的即时反馈信号。它量化 (quantization)了转移或状态的期望程度。奖励是标量值，$r \in \mathbb{R}$。有几种定义奖励函数的常见方式：

- 基于状态-动作-下一个状态三元组：$R(s, a, s')$。在时间 $t+1$ 收到的奖励 $r_{t+1}$ 取决于状态 $s_t$、动作 $a_t$ 和结果状态 $s_{t+1}$。
- 基于状态-动作对：$R(s, a)$。奖励只取决于采取动作时的状态和动作本身。
- 基于抵达的状态：$R(s')$。

一个常用定义给出在当前状态和动作下预期即时奖励：


$$
R(s, a) = \mathbb{E}[R_{t+1} | S_t = s, A_t = a] = \sum_{s' \in \mathcal{S}} P(s' | s, a) \sum_{r} r \cdot p(r | s, a, s')
$$


其中 $p(r | s, a, s')$ 是通过 $a$ 从 $s$ 转移到 $s'$ 时获得奖励 $r$ 的概率。像 $R(s, a, s')$ 这样的更简单形式也很常见。

奖励函数隐含地规定了强化学习 (reinforcement learning)智能体的目标。智能体学习选择能使时间上的*累积*奖励最大化的动作，而不仅仅是即时奖励。设计一个有效的奖励函数是应用强化学习的一个重要方面。例如，在游戏中，只在结束时给予奖励（赢+1，输-1）会导致稀疏奖励，这会使学习变慢。提供中间奖励（例如，在国际象棋中捕获对手棋子）可以产生更密集的反馈，但必须谨慎进行，以避免鼓励不理想的短期策略。

### 折扣因子 ($\gamma$)

折扣因子 $\gamma$ (gamma) 是一个介于0和1之间的值 ($0 \le \gamma \le 1$)。它控制着未来奖励相对于即时奖励的重要性。未来 $k$ 个时间步收到的奖励，其价值仅相当于立即收到时价值的 $\gamma^k$ 倍。

折扣因子有几个作用：

1. **确保收敛性：** 在可能无限运行的任务（持续任务）中，如果奖励有界，折扣确保未来奖励的无限和收敛到一个有限值。这使得目标在数学上可处理。
2. **模拟偏好：** 它允许我们模拟对更快获得奖励的偏好。如果 $\gamma = 0$，智能体只关心即时奖励（它是“短视的”）。如果 $\gamma$ 接近 1，智能体是“有远见的”，并且几乎将未来奖励与即时奖励同等看待。
3. **处理不确定性：** 折扣可以隐含地考虑未来的不确定性。过程可能会意外终止，使得远期未来奖励的获得变得不那么确定。

智能体的目标通常被表述为最大化*预期折扣回报*，我们稍后会更正式地定义它。它是未来所获得的折扣奖励的总和。

> 正式定义马尔可夫决策过程的五个组成部分。

这五个组成部分，$\mathcal{S}$、$\mathcal{A}$、$P$、$R$ 和 $\gamma$，共同为MDP架构下的序列决策问题提供了完整的描述。它们定义了环境的结构以及强化学习 (reinforcement learning)智能体旨在实现的目标：寻找一种选择动作的策略（一个策略），以最大化预期长期折扣奖励。理解这些组成部分对于掌握我们之后将要讨论的用于解决MDP的算法非常必要。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: A Bradford Book, The MIT Press; Pages: 47-62
  强化学习领域的经典教材，对马尔可夫决策过程及其五个核心组成部分提供了全面而易懂的定义。
- [Lecture Notes: Reinforcement Learning and Control](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGgrqk2TVJK-oJUbrY3CWU03CC-t5YIyHFMDBqIQaIJ8XHzWsVJFY3YrJ13hgMx6bZpD2oROv1x4s4k1MsFsjVfz1xCL4sC70cS8WAY2n5hQwr97qc6maqVf87WWsko8tHii_wKy3QD_KKLLHDszPgxJZv_ow==) — Andrew Ng (2019)
  Publisher: Stanford University; Pages: 1-5
  在机器学习背景下，对马尔可夫决策过程提供了简洁而严谨的数学介绍。
- [Artificial Intelligence: A Modern Approach](https://www.pearson.com/us/higher-education/program/Russell-Artificial-Intelligence-A-Modern-Approach-4th-Edition/PGM248883.html) — Stuart Russell, Peter Norvig (2021)
  Publisher: Pearson; Pages: 641-645
  这本全面的AI教材将马尔可夫决策过程解释为序列决策的正式框架，为其应用提供了广阔的背景。

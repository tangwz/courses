---
course: "advanced-reinforcement-learning"
chapter: "model-based-reinforcement-learning"
lesson: "dyna-architectures"
sourceId: 3445
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-5-model-based-reinforcement-learning/dyna-architectures"
title: "Dyna 架构：学习与规划的结合"
description: "Dyna-Q 算法将真实经验与学习模型生成的模拟经验结合。"
order: 4
plots: []
sourceHash: "26dee44bfee9a78721adbcf113c236bd83a4c483e87bb8e95f763a561703f331"
sourceCorrections: []
---

这是“Dyna 架构：学习与规划的结合”一节的内容：

与纯粹的无模型方法相比，基于模型的强化学习 (reinforcement learning)通过学习环境的动态模型，具备更高的样本效率的潜力。然而，仅仅学习模型是不够的；智能体需要一种机制来有效使用这个模型以改进其策略或价值函数。由 Richard Sutton 引入的 Dyna 架构，提供了一个基本框架，用于结合模型学习、直接强化学习（从真实经验中学习）和规划（从模型生成的模拟经验中学习）。

### Dyna 理念：并行学习与规划

其核心在于，Dyna 架构交替进行在环境中行动、从产生的真实经验中学习、更新环境模型以及使用模型执行规划步骤。这种并行过程使智能体能够从两种信息源中获益：

1. **直接强化学习 (reinforcement learning)：** 智能体与实际环境互动，接收一个转移 $(s, a, r, s')$，并基于此真实经验直接更新其价值函数或策略。这是标准的无模型学习。
2. **模型学习：** 相同的真实转移 $(s, a, r, s')$ 被用于更新智能体的内部环境模型。目标是使由转移函数 $P_{model}(s'|s, a)$ 和奖励函数 $R_{model}(s, a)$ 构成的模型，更好地近似真实环境动态。
   "3. **规划（间接强化学习）：** 智能体使用其学习到的模型生成*模拟*经验。它采样一个先前访问过的状态 $s$ 和在该状态下执行的行动 $a$，查询模型以预测下一个状态 $s'$ 和奖励 $r$，然后使用这个模拟转移 $(s, a, r, s')$ 执行更新。这使智能体能够“演练”或“思考”过去的经验，并在不需要进一步互动的情况下传播价值信息。"

这构成了 Dyna 架构的基础。规划步骤通过学习模型引导的模拟重放，从每次真实互动中获取更多信息，从而基本上增强了学习过程。

### Dyna-Q：一种具体实现

Dyna 架构最知名的实例是 Dyna-Q。它将 Q-学习用于直接强化学习 (reinforcement learning)更新和规划更新，以及一个简单的表格模型用于模型学习（尽管也可以使用函数近似）。

以下是 Dyna-Q 的典型流程：

1. **初始化：** 初始化 Q 函数 $Q(s, a)$ 和模型 $Model(s, a)$（例如，存储每个 $(s, a)$ 对观察到的 $r, s'$）。
2. **循环：**
   a. **行动：** 观察当前状态 $s$。基于 $Q(s, \cdot)$ 使用 epsilon-贪婪策略选择行动 $a$。执行行动 $a$，观察奖励 $r$ 和下一个状态 $s'$。
   b. **直接强化学习更新：** 使用真实经验 $(s, a, r, s')$ 更新 Q 函数：
   $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$
   c. **模型学习：** 将转移 $(s, a) \rightarrow (r, s')$ 存储在模型中。如果模型是概率性的或使用函数近似，则基于此转移更新模型参数 (parameter)。对于一个简单的表格确定性模型，这意味着记录在状态 $s$ 执行行动 $a$ 导致了奖励 $r$ 和下一个状态 $s'$。
   d. **规划：** 重复 $N$ 次（其中 $N$ 是规划步骤的数量）：
   i. 随机采样一个先前观察到的状态 $\hat{s}$ 和在该状态下执行的行动 $\hat{a}$。
   ii. 查询模型以获取与 $(\hat{s}, \hat{a})$ 相关的预测奖励 $\hat{r}$ 和下一个状态 $\hat{s}'$。
   iii. 使用这个*模拟*经验应用 Q-学习更新：
   $Q(\hat{s}, \hat{a}) \leftarrow Q(\hat{s}, \hat{a}) + \alpha [\hat{r} + \gamma \max_{a'} Q(\hat{s}', a') - Q(\hat{s}, \hat{a})]$

> Dyna-Q 过程结合了真实互动、模型学习以及使用从学习模型中获得的模拟互动进行的规划。

规划步骤的数量 $N$ 是一个超参数 (hyperparameter)，它平衡了花在规划上的计算量与真实互动速率。更大的 $N$ 意味着每一步真实操作需要更多计算，但可能在所需真实互动次数方面更快收敛。

### 优点与考量

"Dyna 架构的主要优点是**提高的样本效率**。通过使用学习到的模型生成模拟经验，智能体可以基于单次互动执行多次更新，从而更快地在状态-行动空间中传播价值信息。这在互动成本高、耗时或有风险的场景中特别有益。"

然而，Dyna 架构也面临一些挑战：

1. **计算成本：** 规划阶段增加了每一步的计算开销。如果 $N$ 很大或状态/行动空间很广，规划可能会变得计算密集。
2. **模型准确性：** 规划的有效性很大程度上取决于学习模型的准确性。如果模型不准确（由于数据不足、非平稳性或难以捕获的随机性），模拟经验可能具有误导性。使用不佳的模型进行规划，可能通过强化不正确的价值估计或策略而损害性能。这通常被称为*模型偏差*问题。

尽管有这些考量，Dyna-Q 及其变体在结合无模型和基于模型方法优势方面迈出了重要一步。它们提供了一种实际方式，可以运用学习到的模型进行规划，与纯粹的无模型方法相比，通常能显著提升学习速度，特别是在数据稀缺的学习早期阶段。这种交替进行行动、学习和规划的基本思想在现代基于模型的强化学习 (reinforcement learning)研究中仍然具有影响力。

## 参考资料

- [Integrated Architectures for Learning, Planning, and Reacting Based on "Dyna"](https://dl.acm.org/doi/10.1145/104523.104529) — Richard S. Sutton (1990)
  Journal: SIGART Bulletin; Publisher: Association for Computing Machinery (ACM); Volume: 1; Pages: 160-163; DOI: [10.1145/104523.104529](https://doi.org/10.1145/104523.104529)
  介绍基础的Dyna架构，结合了直接强化学习、模型学习和规划。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本广泛认可的强化学习教材，全面解释了包括Dyna-Q在内的Dyna架构。第8章尤为相关。
- [Model-Based Reinforcement Learning: A Survey](https://dl.acm.org/doi/10.1145/3547271) — Mo Chen, Jingyue Liu, Xiaohong Li, Yuyang Shi, Zhirong Liu, Yong Liu, and Dacheng Tao (2022)
  Journal: ACM Computing Surveys; Publisher: Association for Computing Machinery; Volume: 55; Pages: 1-36; DOI: [10.1145/3547271](https://doi.org/10.1145/3547271)
  对模型强化学习的最新综述，为Dyna原理如何在新研究中发展和解决（包括模型精度等挑战）提供了背景。

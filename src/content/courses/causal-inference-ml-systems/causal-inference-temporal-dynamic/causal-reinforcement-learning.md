---
course: "causal-inference-ml-systems"
chapter: "causal-inference-temporal-dynamic"
lesson: "causal-reinforcement-learning"
sourceId: 4407
sourceUrl: "https://apxml.com/zh/courses/causal-inference-ml-systems/chapter-5-causal-inference-temporal-dynamic/causal-reinforcement-learning"
title: "因果强化学习与离策略评估"
description: "将因果推断思路融入强化学习，主要关注离策略评估和策略学习。"
order: 6
plots: []
sourceHash: "6331e0fee955c601fb03430313ec4f531248f63707f4f9c17ac873ebca617114"
sourceCorrections: []
---

强化学习 (reinforcement learning)（RL）提供了一个应对不确定性的强大顺序决策框架，一个智能体通过与环境的交互来优化其动作。这种顺序决策是动态干预方案和时序系统研究的核心。然而，标准的强化学习方法通常严重依赖从交互数据中学到的关联模式。结合因果推断原则为更可靠的策略评估、理解智能体行为以及培养在不断变化条件下泛化能力更强的智能体提供了途径。在此，考察因果推断与强化学习的交叉点，特别关注离策略评估（OPE）中的难题及因果方法。

## 强化学习 (reinforcement learning)的因果视角

在典型的马尔可夫决策过程（MDP）设置中，智能体在离散时间步 $t$ 与环境进行交互。在每个步骤中，智能体观察状态 $s_t \in \mathcal{S}$，根据其策略 $\pi(a_t | s_t)$ 采取动作 $a_t \in \mathcal{A}$，接收奖励 $r_t = R(s_t, a_t)$，并转换到新状态 $s_{t+1} \sim P(s_{t+1} | s_t, a_t)$。目标通常是找到一个策略 $\pi$，使其最大化预期的累积折扣奖励，即回报 $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$，其中 $\gamma \in [0, 1)$ 是折扣因子。

因果推断引入了若干有价值的观点：

1. **环境动态作为因果机制：** 转换函数 $P(s_{t+1} | s_t, a_t)$ 和奖励函数 $R(s_t, a_t)$ 可以被视为结构性因果机制。使用因果框架（如适应时间的SCM）而非纯粹的预测模型来建模它们，可以增进对环境的理解和外推能力，特别是在某些机制可能发生变化的环境中。
2. **动作作为干预：** 选择动作 $a_t$ 是对系统的干预。智能体的策略 $\pi$ 根据观察到的状态决定了一系列干预。
3. **离策略评估作为因果问题：** 离策略评估旨在利用不同行为策略 $\pi_b$ 生成的数据，估计评估策略 $\pi_e$ 的预期回报。这是一个根本性的因果问题：“如果我们*干预*并部署策略 $\pi_e$，预期回报 $E[G_0^{\pi_e}]$ 会是多少？” 我们寻求的是在干预下的预期结果 $E[Y^{do(\pi=\pi_e)}]$，其中 $Y$ 代表回报。

## 离策略评估中混杂因素的难题

离策略评估中的一个显著难题源于混杂因素。行为策略 $\pi_b$ 选择的动作 $a_t$ 是基于状态 $s_t$ 的。这个状态 $s_t$ 不仅影响动作 $a_t$，还可能通过环境动态影响未来的状态 $s_{t+1}, s_{t+2}, \dots$ 和奖励 $r_t, r_{t+1}, \dots$，这与动作的直接效应无关。如果我们天真地使用来自 $\pi_b$ 的数据来评估 $\pi_e$，我们可能会错误地将由 $\pi_b$ 下遇到的状态引起的结果归因于 $\pi_e$ 在这些状态下会采取的动作。

考虑以下表示MDP中一个步骤的简化因果图：

> MDP 中一个时间步的简化因果图。状态 $S_t$ 影响动作 $A_t$（通过策略 $\pi_b$），也直接影响奖励 $R_t$ 和下一状态 $S_{t+1}$（通过环境动态）。这使得 $S_t$ 成为评估不同策略 $\pi_e$ 时，动作 $A_t$ 对未来结果影响的混杂因素。

标准重要性采样（IS）试图通过对轨迹进行重新加权来校正策略间的分布差异：


$$
\hat{V}_{IS}^{\pi_e} = \frac{1}{N} \sum_{i=1}^{N} \left( \prod_{t=0}^{T-1} \frac{\pi_e(a_{i,t} | s_{i,t})}{\pi_b(a_{i,t} | s_{i,t})} \right) G_{i,0}
$$


这里 $i$ 表示轨迹，而 $G_{i,0}$ 是轨迹 $i$ 的回报。尽管在某些假设下（包括正性假设：当 $\pi_e(a|s) > 0$ 时，$\pi_b(a|s) > 0$）理论上是无偏的，但IS方法常面临极高的方差，特别是对于长轨迹，因为重要性比率的乘积可能急剧增大或趋于零。

## 离策略评估的因果方法

因果推断提供了开发离策略评估（OPE）估计器的技术，通常通过使用环境模型或价值函数以及策略模型来实现。

### 估计

与静态干预效应估计中的方法（与第3章讨论的双重机器学习 (machine learning)相关）类似，离策略评估（OPE）估计器可以通过将重要性加权与预期未来回报模型（例如，在评估策略下估计的Q函数 $Q^{\pi_e}(s, a)$ 或值函数 $V^{\pi_e}(s)$）结合来构建。初始状态 $s_0$ 下预期回报的常见双重鲁棒（DR）估计器形式如下：


$$
\hat{V}_{DR}^{\pi_e} = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T-1} \gamma^t \left( \rho_{i, 0:t-1} r_{i,t} - (\rho_{i, 0:t} \hat{Q}^{\pi_e}(s_{i,t}, a_{i,t}) - \rho_{i, 0:t-1} \hat{V}^{\pi_e}(s_{i,t})) \right)
$$


（存在简化形式）。其中，$\rho_{i, 0:t} = \prod_{k=0}^{t} \frac{\pi_e(a_{i,k} | s_{i,k})}{\pi_b(a_{i,k} | s_{i,k})}$ 是直到时间 $t$ 的累积重要性比率，$\hat{Q}^{\pi_e}, \hat{V}^{\pi_e}$ 是评估策略的估计值函数。

一个重要特性是**双重鲁棒性**：如果策略比率模型（隐含在 $\rho$ 中）*或*值函数模型（$\hat{Q}^{\pi_e}, \hat{V}^{\pi_e}$）中的*任一*被正确设定，则估计器是渐近一致的（收敛到真实值 $E[G_0^{\pi_e}]$），不要求两者都正确。这通常会带来比纯IS方法更低的方差，特别是当值函数模型捕捉到环境动态的重要方面时。

### 与边际结构模型（MSMs）的关联

离策略评估中使用的技术，特别是DR估计器，与流行病学和计量经济学中用于估计存在时变混杂因素的时变干预的因果效应的方法（如MSMs）有相似之处。在强化学习 (reinforcement learning)背景下，动作 $a_t$ 是时变干预，状态 $s_t$ 包含时变混杂因素（受过去动作影响并影响未来动作和结果），而回报 $G_t$ 是结果。运用MSM原则涉及使用逆概率加权（类似于IS）或G-计算（类似于基于模型的价值估计）或其组合（如DR）来调整状态历史的混杂效应。这要求与序贯可忽略性类似的假设：对于所有 $t$，都有 $Y^{a_0, ..., a_T} \perp A_t | H_t$，其中 $H_t = (S_0, A_0, R_0, ..., S_t)$ 是截至时间 $t$ 的历史，而 $Y$ 是在动作序列下的潜在结果（回报）。

### 处理未观测混杂因素

标准离策略评估方法假定为实现条件独立性（序贯可忽略性）所需的所有相关状态信息 $s_t$ 均已被观测。如果关键状态组成部分未被观测（潜在状态变量），这些方法可能产生有偏估计。此情形与第4章讨论的未观测混杂因素问题相对应。

正在研究先进方法：

- **强化学习 (reinforcement learning)的近端因果推断：** 将近端方法（使用与未观测混杂因素相关的代理变量，如第4章所述）应用于强化学习设置。这涉及找到满足特定条件独立性关系的变量，即使存在隐藏状态混杂也能进行识别和估计。
- **敏感性分析：** 量化 (quantization)在存在未观测混杂因素时，OPE估计可能发生的变化。

## 因果推断在策略学习与泛化中的作用

在评估现有策略时，因果推断可以为学习过程本身提供信息：

- **基于因果模型的强化学习 (reinforcement learning)：** 学习环境转换 $P(s_{t+1}|s_t, do(a_t))$ 和奖励 $R(s_t, do(a_t))$ 的显式因果模型，可以提供优于纯粹预测模型的优势。因果模型旨在捕获底层机制，可能实现：
  - 更有效的规划。
  - 更好地泛化到未见状态或部分因果模块保持不变的略微不同的环境。
  - 对环境本身的干预进行推理 (inference)。
- **强化学习中的因果发现：** 将因果发现算法（来自第2章和时间序列因果发现部分）应用于交互过程中生成的 $(s_t, a_t, r_t, s_{t+1})$ 数据流。发现控制环境的因果图可以指导模型构建，并可能识别出处理混杂因素有用的工具变量或其他结构。
- **迁移学习 (transfer learning)：** 因果模型有助于迁移学习。如果我们知道源任务和目标任务之间哪些因果机制发生了变化，我们可以重用学习模型的未变部分，从而可能加速目标环境中的学习。

## 实践考量

将因果推断集成到强化学习 (reinforcement learning)中涉及几个实践考量：

- **假设有效性：** 因果OPE方法依赖于序贯可忽略性（或使用近端推断等方法时的替代方案）和正性等假设。验证这些假设很困难，且常需要领域知识。敏感性分析很重要。
- **模型设定：** 双重鲁棒方法要求估计策略比率或值函数。其性能在很大程度上取决于这些模型的质量，尤其是在高维状态/动作空间中（这与第3章相关联）。
- **计算开销：** 因果方法，特别是那些涉及复杂模型或发现算法的方法，可能计算密集。
- **软件实现：** 尽管标准强化学习库已很成熟，但专注于因果强化学习和鲁棒离策略评估的库则不那么常见，尽管存在研究性质的实现（例如，在RLlib等框架内或独立实现）。

总而言之，将因果视角应用于强化学习，特别是离策略评估，使得研究从关联模式匹配转向理解动态系统中干预（动作）的效果。虽然标准IS提供了基本校正，但双重鲁棒估计等方法通过结合环境或值模型，提供了显著的方差降低。处理未观测状态混杂以及使用因果模型进行策略学习，是构建更可靠、更具适应性智能体的活跃且重要的研究方向。

## 参考资料

- [Doubly Robust Off-Policy Value Estimation for Reinforcement Learning](http://proceedings.mlr.press/v48/jiang16.pdf) — Nan Jiang, Lihong Li (2016)
  Journal: International Conference on Machine Learning (ICML); Pages: 1950-1959; DOI: [10.5555/3045390.3045470](https://doi.org/10.5555/3045390.3045470)
  这是一篇开创性论文，专门针对强化学习中的离策略评估引入并分析了双重稳健估计器，解决了重要性采样的方差过高问题。
- [Causal Inference in Sequential Decision-Making: The Confounding-Bridge Approach](https://www.aaai.org/ojs/index.php/AAAI/article/view/9344) — Elias Bareinboim, Andrew Forney, Judea Pearl (2015)
  Journal: AAAI Conference on Artificial Intelligence; Publisher: Association for the Advancement of Artificial Intelligence (AAAI); Volume: 29; Pages: 2686-2692; DOI: [10.1609/aaai.v29i1.9344](https://doi.org/10.1609/aaai.v29i1.9344)
  这篇论文为动态系统和序贯决策中的因果推断奠定了理论基础，对于理解强化学习和离策略评估中的因果视角至关重要。
- [Proximal Reinforcement Learning: A Causal Framework for Hidden Confounding in RL](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0c32c58908f2445b231ffc2419c968f9-Paper-Conference.pdf) — Yue Shi, Eric Tchetgen Tchetgen, Ioana Bica (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 35; Pages: 1642-1655; DOI: [10.48550/arXiv.2206.01429](https://doi.org/10.48550/arXiv.2206.01429)
  这篇论文引入了近端因果推断方法，以解决强化学习环境中离策略评估和学习中未观测混淆问题。

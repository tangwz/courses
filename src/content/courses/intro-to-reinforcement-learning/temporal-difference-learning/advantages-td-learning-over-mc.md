---
course: "intro-to-reinforcement-learning"
chapter: "temporal-difference-learning"
lesson: "advantages-td-learning-over-mc"
sourceId: 1644
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-5-temporal-difference-learning/advantages-td-learning-over-mc"
title: "TD学习相对于蒙特卡洛方法的优势"
description: "Compare the benefits of TD learning (e.g., learning online, handling continuing tasks) versus MC methods."
order: 3
plots: []
sourceHash: "b0df37809207b5df3a9cbbc4afeb7edc1987e8e55eb99447fe5e0e77c1e46e7f"
sourceCorrections: []
---

蒙特卡洛（MC）方法通过对完整回报求平均，提供了一种从经验中学习的直观方式，而时序差分（TD）学习通常表现出几个显著的优势，使其在许多强化学习 (reinforcement learning)场景中成为一种更广泛使用的方法。

### 从不完整回合和连续任务中学习

主要区别在于学习更新发生的*时机*。MC方法需要等到一个回合结束，确定回报$G_t$后，才能更新该回合中访问过的状态的任何价值估计。

> MC更新（左）使用整个回合的结果，而TD更新（右）仅使用下一个奖励和下一个状态的估计价值。

相比之下，TD方法仅在一个时间步后就更新其价值估计，使用观测到的奖励$R_{t+1}$和下一个状态价值$V(S_{t+1})$的当前估计。例如，TD(0)更新规则如下所示：


$$
V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]
$$


这种从单个转移（$S_t, A_t, R_{t+1}, S_{t+1}$）中学习的能力具有重要的影响：

1. **适用于连续任务：** 许多强化学习 (reinforcement learning)问题没有明确定义的回合。考虑控制化学过程、管理投资组合或机器人持续运行。这些是理论上无限期运行的*连续任务*。MC方法无法直接应用，因为没有“最终结果”可等待。TD方法通过逐步更新，自然地处理这些情况。
2. **处理长回合的效率：** 即使在回合式任务中，如果回合非常长（例如，像围棋这样的复杂游戏，或长时间模拟），等待回合结束会大幅延迟学习。TD提供更频繁的更新，使智能体能够在回合中途学习并潜在地改进其行为。

### 在线学习能力

TD学习天生适合*在线*学习。智能体可以观测一个转移、收到奖励、移到下一个状态，并立即更新刚刚离开的状态的价值。这使得智能体在与环境交互时能够“即时”调整其知识和潜在行为。

MC方法通常*离线*运行。首先收集整个回合，然后计算回报，最后更新该回合中所有访问状态的价值估计。这需要存储回合轨迹（状态、动作、奖励）直到终止。虽然离线更新是可行的，但对于需要实时适应或有内存限制的系统，在线学习通常更可取。

### 较低方差的更新和收敛速度

尽管MC更新使用实际的采样回报$G_t$，这是真实价值$v_\pi(S_t)$的无偏估计，但该回报可能具有高方差。最终结果$G_t$取决于潜在的长串动作、状态转移和奖励，每个都带来随机性。目标更新值的高方差会使学习过程嘈杂且收敛缓慢。

TD方法朝向一个*TD目标*更新：$R_{t+1} + \gamma V(S_{t+1})$（用于价值预测）或$R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$（用于Q学习）。该目标仅依赖于一个随机奖励$R_{t+1}$和一个随机转移到$S_{t+1}$。价值估计$V(S_{t+1})$或$Q(S_{t+1}, A_{t+1})$本身用于更新（这就是自举）。

因为TD目标依赖的随机事件比完整的MC回报少，所以它通常具有低得多的方差。但代价是：TD目标是*有偏的*。它使用当前的估计$V(S_{t+1})$，这可能不完全准确，特别是在学习早期。

这引入了经典的偏差-方差权衡：

- **MC：** 高方差，零偏差（使用实际采样回报）。
- **TD：** 低方差，一些偏差（使用估计价值）。

实践中，TD更新的较低方差通常导致比MC方法更快的收敛，尽管存在偏差。学习过程趋于更平滑，对单个嘈杂回合不那么敏感。

### 优势总结

总而言之，TD学习相对于MC方法的主要优势包括：

- **无需等待回合结束即可学习：** 使其能够用于连续任务，并在长回合任务中提供更快的更新。
- **在线学习：** 更新可在交互过程中逐步发生。
- **较低方差：** 更新基于单个转移，通常导致学习噪声更小，实践中比高方差的完整回报收敛更快。

这些好处使SARSA和Q学习等TD方法成为强化学习 (reinforcement learning)中的根本算法，特别适用于回合很长、不存在回合或在线适应很重要的场景。然而，也值得记住的是，TD中自举引入的偏差有时会引起问题，特别是与函数逼近结合时，这是一个我们稍后会讨论的话题。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  这本教科书阐述了强化学习基础，比较了蒙特卡洛和时序差分方法，它们的异同以及实际优势。
- [Learning to predict by methods of temporal differences](https://link.springer.com/article/10.1007/BF00115009) — Richard S. Sutton (1988)
  Journal: Machine Learning; Volume: 3; Pages: 9-44; DOI: [10.1007/BF00115009](https://doi.org/10.1007/BF00115009)
  这篇原始论文介绍了时序差分（TD）学习，包括TD(0)算法，为许多强化学习算法奠定基础。

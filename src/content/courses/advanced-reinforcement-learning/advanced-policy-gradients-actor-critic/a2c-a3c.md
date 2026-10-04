---
course: "advanced-reinforcement-learning"
chapter: "advanced-policy-gradients-actor-critic"
lesson: "a2c-a3c"
sourceId: 3387
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-3-advanced-policy-gradients-actor-critic/a2c-a3c"
title: "优势 Actor-Critic (A2C) 和 A3C"
description: "优势演员-评论家方法的同步 (A2C) 和异步 (A3C) 变体。"
order: 4
plots: []
sourceHash: "5e79ac43cba8d3df95f10b8f7dbfa21f774e1458742a19bf5b614f77bff68892"
sourceCorrections: []
---

优势演员-评论家 (A2C) 及其异步对应算法异步优势演员-评论家 (A3C) 是演员-评论家架构中的高级算法。这些方法通过引入*优势函数*显著提升了稳定性和性能，从而降低了基本策略梯度估计中固有的方差。

回顾策略梯度定理，它提出以$\nabla_\theta \log \pi_\theta(a|s)$的方向更新策略参数 (parameter)$\theta$，并由动作优劣的某种度量进行加权。虽然使用完整回报$G_t$是无偏的，但它存在高方差的问题。使用一个基线，例如状态价值函数$V(s)$，会得到优势函数：


$$
A(s, a) = Q(s, a) - V(s)
$$


策略梯度更新则变为：


$$
\nabla_\theta J(\theta) \approx \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) A(s, a)]
$$


从直观上看，优势$A(s, a)$衡量了动作$a$相对于从状态$s$采取的平均动作有多好。使用优势能使梯度信号居中，鼓励好的动作（正优势）并抑制不好的动作（负优势），从而带来更稳定的更新。

在实际中，我们不知道真正的$Q(s, a)$和$V(s)$。演员-评论家方法使用一个习得的评论家$V_\phi(s)$（由$\phi$参数化）来近似$V(s)$。在时间$t$时，优势函数的一个常用估计使用时序差分（TD）误差：


$$
A^{\text{估算}}(s_t, a_t) \approx r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) = \delta_t
$$


这里，$r_{t+1} + \gamma V_\phi(s_{t+1})$作为$Q(s_t, a_t)$的一个估计，而$V_\phi(s_t)$是由评论家提供的基线。评论家本身则通过最小化该价值预测中的误差进行训练，通常在TD误差上使用均方误差（MSE）损失：


$$
\mathcal{L}(\phi) = \mathbb{E} \left[ (r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t))^2 \right]
$$


A2C 和 A3C 的主要区别在于它们如何收集经验和应用这些更新。

### 优势演员-评论家 (A2C)

A2C 是优势演员-评论家方法的同步并行版本。其工作方式如下：

1. **并行执行者：** 多个执行者并行运行，每个执行者在自己的环境实例中进行固定步数（例如 $T$ 步）的交互。
2. **经验收集：** 每个执行者收集一个轨迹片段 $(s_t, a_t, r_{t+1}, s_{t+1})$，其中 $t = 1, \dots, T$。
3. **同步：** 所有执行者暂停，并将他们收集的经验汇集到一个批次中。
4. **优势计算：** 使用当前的评论家网络$V_\phi$，为批次中所有转换计算优势$A^{\text{估算}}(s_t, a_t)$。通常，这里会使用更精细的估计方法，如后面讨论的广义优势估计（GAE）。
5. **梯度计算：** 使用批处理数据计算执行者（$\nabla_\theta$）和评论家（$\nabla_\phi$）的梯度。执行者梯度使用估算的优势，而评论家梯度旨在最小化价值预测误差。
6. **同步更新：** 计算出的梯度（通常在并行执行者中取平均）用于更新中央执行者和评论家网络。
7. **重复：** 执行者使用更新后的网络参数 (parameter)继续交互。

A2C 的一个特点是其同步性。所有执行者在中央网络更新发生之前彼此等待。这通常使得GPU的运用更有效，因为GPU擅长同时处理大量数据。尽管A3C最初被提出，但A2C通常被认为更容易实现，并且在GPU资源可用时可以达到相似或更好的性能。

### 异步优势演员-评论家 (A3C)

A3C 引入了异步更新的思路，以稳定学习，而无需像DQN那样依赖经验回放缓冲区。它使用多个工作器，每个工作器都有自己的环境实例和演员-评论家网络的本地副本。这些工作器独立运行并更新一个共享的全局网络。

> A3C 中的异步训练结构。多个工作器与其环境交互，使用从全局网络拉取的参数 (parameter)在本地计算梯度，然后异步地将这些梯度推送到全局网络进行更新。

A3C 中每个工作器的工作流程如下：

1. **重置梯度：** 清除本地梯度缓冲区。
2. **同步参数：** 将参数$(\theta_g, \phi_g)$从全局网络复制到本地网络$(\theta', \phi')$。
3. **交互：** 与环境交互$T_{\max}$步，或直到达到终止状态，收集$(s_t, a_t, r_{t+1}, s_{t+1})$。
4. **计算优势：** 为收集到的轨迹片段计算优势估计$A^{\text{估算}}(s_t, a_t)$，通常使用本地评论家$V_{\phi'}$。
5. **计算梯度：** 根据收集到的经验和优势，计算执行者和评论家的本地梯度$\Delta\theta'$和$\Delta\phi'$。
6. **异步更新：** 将计算出的梯度应用于*全局*网络参数$(\theta_g, \phi_g)$。此更新无需加锁，这意味着多个工作器可以同时更新全局网络。
7. **重复：** 返回步骤1。

A3C 有效性的主要原因在于，来自多个工作器对环境不同部分的并行处理有助于解除训练数据之间的关联性。每个工作器的经验流可能不同，并且异步应用更新可以防止简单在线强化学习 (reinforcement learning)方法中强相关更新可能导致的不稳定性。与单线程算法或甚至A2C（在CPU受限时）相比，这通常会带来在多核CPU上更快的训练时间。

### A2C vs. A3C：快速比较

| 特点 | A2C (优势演员-评论家) | A3C (异步优势演员-评论家) |
| --- | --- | --- |
| **更新方式** | 同步 | 异步 |
| **并行方式** | 在所有执行者完成后才更新 | 工作器独立更新全局网络 |
| **硬件利用** | 在GPU上高效（批处理） | 在多核CPU上高效（并行执行） |
| **数据去关联** | 通过批量处理多样化轨迹实现 | 通过工作器的异步更新实现 |
| **实现难度** | 通常更简单 | 更复杂（线程、参数 (parameter)服务器） |
| **稳定性** | 通常被认为略微更稳定/可复现 | 有时可能受到陈旧梯度的影响 |

虽然A3C是一个具有里程碑意义的算法，展示了异步训练的效用，但A2C（通常与GAE一起实现用于优势估计）已成为一种非常流行且性能良好的基准方法。它通常在GPU加速环境中取得与A3C相似或更好的结果，并且更易于实现和调试。

A2C 和 A3C 都代表了在基础策略梯度方法上的一大进步，它们通过有效使用评论家来估计优势函数，从而降低了梯度方差，并实现了更稳定和高效的学习。它们是TRPO和PPO等后续改进方法所依据的基础。

## 参考资料

- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu (2016)
  Journal: ICML 2016; Pages: 1928-1937; DOI: [10.48550/arXiv.1602.01783](https://doi.org/10.48550/arXiv.1602.01783)
  介绍异步优势演员-评论家（A3C）算法，一种通过异步更新实现稳定高效深度强化学习的方法。文章也包含了对同步变体的描述。
- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本关于强化学习的经典教材，提供了策略梯度、价值函数和演员-评论家方法等理论基础。
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel (2016)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1506.02438](https://doi.org/10.48550/arXiv.1506.02438)
  提出了广义优势估计（GAE），一种稳健的优势函数估计方法，显著提升了A2C和A3C等演员-评论家算法的性能。
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍近端策略优化（PPO）算法，一种广泛使用的策略梯度算法，它在A2C和TRPO的基础上，实现了性能与实现简易性之间的平衡。

---
course: "advanced-reinforcement-learning"
chapter: "advanced-policy-gradients-actor-critic"
lesson: "ddpg-algorithm"
sourceId: 3393
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-3-advanced-policy-gradients-actor-critic/ddpg-algorithm"
title: "深度确定性策略梯度 (DDPG)"
description: "一种专为连续动作空间设计的离策略Actor-Critic算法，结合了DQN的思路。"
order: 6
plots: []
sourceHash: "8f88a4a8045d5d17b7a5cfe3e2e79aa642af56fb617e861354285f5b7f5e36f0"
sourceCorrections: []
---

尽管标准Actor-Critic方法相较于REINFORCE能降低方差，但将它们直接应用于连续动作空间的环境会带来挑战。像A2C这样的算法通常依赖于从随机策略$\pi(a|s)$中采样动作并计算期望，当动作空间无限时，这会变得复杂。

深度确定性策略梯度（DDPG）通过将深度Q网络（DQN）的思路融入到一种专为连续控制设计的离策略Actor-Critic框架中，提供了一个巧妙的解决方案。该算法由Lillicrap等人于2015年提出，在处理机器人操作和自动驾驶模拟器等问题上产生了影响。

### 从随机策略到确定性策略

DDPG中的主要修改是使用*确定性*策略，表示为$\mu(s; \theta^\mu)$。Actor网络不再输出动作的概率分布，而是直接将状态$s$映射到特定的动作$a$:


$$
a = \mu(s; \theta^\mu)
$$


这里，$\theta^\mu$代表Actor网络的参数 (parameter)。这种转变显著简化了策略梯度计算。回顾一下，标准策略梯度定理涉及对从策略中采样的动作的期望。对于确定性策略，期望消失，目标函数$J(\theta^\mu)$（期望回报）的梯度可以通过链式法则推导出来，这利用了Critic对Actor所选动作的评估。

### Actor网络和Critic网络

DDPG维护着两个主要的神经网络 (neural network)，以及它们各自的目标版本：

1. **Actor网络 ($\mu(s; \theta^\mu)$):** 接收状态$s$作为输入，并输出一个特定的连续动作$a$。其目标是通过最大化预期的未来奖励来学习最优策略。
2. **Critic网络 ($Q(s, a; \theta^Q)$):** 同时接收状态$s$和动作$a$作为输入，并输出相应的Q值（从状态$s$开始，执行动作$a$，然后遵循策略$\mu$的预期回报）。其目标是准确评估当前Actor策略的动作值函数。

以下是DDPG架构的可视化表示：

> DDPG智能体与环境交互，将经验存储在经验回放缓冲区中。更新涉及从缓冲区采样、使用目标网络计算目标值，并分别根据TD误差和策略梯度调整Actor和Critic参数 (parameter)。目标网络会缓慢地向主网络更新。

### 训练Critic

Critic网络$Q(s, a; \theta^Q)$的训练方式与DQN中的Q网络类似。它旨在最小化均方贝尔曼误差（MSBE）。我们从经验回放缓冲区$D$中采样一个包含转换 $(s_i, a_i, r_i, s'_i)$ 的小批量数据。对于每个转换，我们计算目标值$y_i$:


$$
y_i = r_i + \gamma Q'(s'_{i}, \mu'(s'_{i}; \theta^{\mu'}); \theta^{Q'})
$$


注意，目标值计算使用*目标*Actor网络$\mu'$来选择下一个动作$a'_{i} = \mu'(s'_{i}; \theta^{\mu'})$，并使用*目标*Critic网络$Q'$来评估下一个状态-动作对的值。这种目标网络（$Q'$和$\mu'$）的使用将目标值与正在积极训练的网络（$Q$）解耦，显著提升了稳定性，就像在DQN中一样。

Critic的损失函数 (loss function)是目标值$y_i$与Critic当前估计$Q(s_i, a_i; \theta^Q)$之间的均方差：


$$
L(\theta^Q) = \frac{1}{N} \sum_i (y_i - Q(s_i, a_i; \theta^Q))^2
$$


此损失通过对Critic参数 (parameter)$\theta^Q$的梯度下降 (gradient descent)进行最小化。

### 训练Actor

Actor网络$\mu(s; \theta^\mu)$使用确定性策略梯度进行更新。目标是调整Actor的参数 (parameter)$\theta^\mu$，以根据当前的Critic产生最大化预期Q值的动作。Actor目标函数$J(\theta^\mu)$的梯度使用经验回放缓冲区中的小批量数据进行近似：


$$
\nabla_{\theta^\mu} J(\theta^\mu) \approx \frac{1}{N} \sum_i \nabla_a Q(s_i, a; \theta^Q) |_{s=s_i, a=\mu(s_i; \theta^\mu)} \nabla_{\theta^\mu} \mu(s_i; \theta^\mu)
$$


这看起来很复杂，但直观地讲它意味着：

1. 对于批次中的每个状态$s_i$，找到当前Actor策略将输出的动作$a = \mu(s_i; \theta^\mu)$。
2. 询问Critic，如果我们稍微改变动作$a$，Q值$Q(s_i, a; \theta^Q)$会如何变化（这就是$\nabla_a Q$）。
3. 计算Actor参数$\theta^\mu$的变化如何影响输出动作$a$（这就是$\nabla_{\theta^\mu} \mu$）。
4. 使用链式法则结合这些梯度，以确定改变$\theta^\mu$将如何影响Q值。
5. 使用梯度*上升*更新$\theta^\mu$（因为我们想最大化$Q$）。

此更新使用Critic $Q$作为评估器，引导Actor做出更好的动作，而无需直接采样动作和估计期望，使其适用于连续空间。

### 离策略学习：经验回放与目标网络

DDPG是一种*离策略*算法。像DQN一样，它使用了从基于值的方法中借鉴的两种机制，以提升稳定性和样本效率：

1. **经验回放：** 转换 $(s_t, a_t, r_t, s_{t+1})$ 存储在一个大型经验回放缓冲区$D$中。在训练期间，会从$D$中随机采样小批量数据。这打破了连续经验之间的时间相关性，使网络能从多样化的过去经验中学习，从而带来更稳定和高效的学习。
2. **目标网络：** DDPG维护着独立的、作为主Actor和Critic网络缓慢更新副本的目标网络$\mu'(s; \theta^{\mu'})$和$Q'(s, a; \theta^{Q'})$。通常，在每次主网络更新后，会使用一种称为Polyak平均的“软”更新规则：
   
   $$
   \theta' \leftarrow \tau \theta + (1-\tau) \theta'
   $$
   
   其中$\tau \ll 1$（例如，$\tau = 0.001$或$0.005$）是一个控制更新速度的超参数 (parameter) (hyperparameter)。这些稳定的目标网络为Critic的TD误差计算提供了稳定的目标，并稳定了整体学习过程。

### 确定性策略中的探索

由于策略$\mu(s; \theta^\mu)$是确定性的，如果在训练期间不加干预，它将始终为给定状态输出相同的动作。这会阻碍探索。为确保智能体充分探索环境，DDPG在Actor的输出动作中添加噪声，*仅限于训练期间*：


$$
a_t = \mu(s_t; \theta^\mu) + \mathcal{N}_t
$$


噪声过程$\mathcal{N}_t$可以是简单的高斯噪声，但最初的DDPG论文使用了奥恩斯坦-乌伦贝克（OU）过程，它能生成时间相关噪声。这种相关噪声有助于在动量是重要因素的物理控制任务中进行探索。然而，更简单的不相关高斯噪声通常也表现良好且更易于实现。噪声的尺度通常会在训练过程中衰减。在评估或部署期间，此噪声会被关闭，智能体仅根据$a_t = \mu(s_t; \theta^\mu)$采取行动。

### DDPG 总结

DDPG将Actor-Critic结构与DQN的见解（经验回放缓冲区、目标网络）结合，创建了一种适用于连续动作空间的有效离策略算法。

**优点：**

- 直接处理连续动作空间。
- 离策略特性使得使用经验回放缓冲区进行样本高效学习成为可能。
- 相较于TRPO等信任区域方法，其理念相对简单。

**缺点：**

- 与DQN类似，对超参数 (parameter) (hyperparameter)（学习率、目标更新率$\tau$、噪声参数、网络架构）敏感。
- Critic仍可能存在Q值高估问题，尽管目标网络有助于缓解此问题。后续算法如TD3（双重延迟DDPG）专门解决了这一局限。
- 性能有时会因环境和调优而变得不稳定。

尽管存在缺点，DDPG代表了深度强化学习 (reinforcement learning)在连续领域的一个重要进展，并为理解TD3和SAC等更新算法奠定了基础。

## 参考资料

- [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971) — Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1509.02971](https://doi.org/10.48550/arXiv.1509.02971)
  介绍深度确定性策略梯度（DDPG）算法的原始论文。
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, Demis Hassabis (2015)
  Journal: Nature; Volume: 518; Pages: 529–533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  关于深度Q网络（DQN）的开创性论文，介绍了DDPG借鉴的经验回放和目标网络思想。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: The MIT Press
  介绍了强化学习的核心思想，包括策略梯度和Actor-Critic方法。
- [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477) — Scott Fujimoto, Herke van Hoof, and David Meger (2018)
  Journal: International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.1802.09477](https://doi.org/10.48550/arXiv.1802.09477)
  介绍了双延迟DDPG（TD3）算法，该算法解决了DDPG的一些弱点，尤其是Q值高估问题。
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/) — Josh Achiam (2018)
  Publisher: OpenAI
  一份权威指南，提供了DDPG及相关DRL算法的解释。

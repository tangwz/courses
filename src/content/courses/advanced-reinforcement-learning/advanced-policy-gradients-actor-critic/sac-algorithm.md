---
course: "advanced-reinforcement-learning"
chapter: "advanced-policy-gradients-actor-critic"
lesson: "sac-algorithm"
sourceId: 3403
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-3-advanced-policy-gradients-actor-critic/sac-algorithm"
title: "软演员-评论家 (SAC)"
description: "一种基于最大熵强化学习框架的离策略演员-评论家方法，促进环境交互。"
order: 9
plots: []
sourceHash: "17bb822f499210a6a2aeeca1abb3d6d3256bf3aebbf47527d8d82e297fb0ac60"
sourceCorrections: []
---

软演员-评论家 (SAC) 是一种采用演员-评论家架构的离策略算法，特别适合连续控制问题。由Google Brain和加州大学伯克利分校的研究团队同期开发，SAC 采纳了**最大熵强化学习 (reinforcement learning)**的原理。该框架修改了标准强化学习目标，旨在最大化预期累积奖励以及策略的熵，以促进代理更好地进行环境交互并增强其稳定性。

### 最大熵目标

在标准强化学习 (reinforcement learning)中，目标通常是找到一个策略 $\pi$，使其最大化折扣奖励的预期总和：
$J(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} [ \gamma^t r(s_t, a_t) ]$
其中 $\rho_\pi$ 是由策略 $\pi$ 产生的状态-动作边际分布。

SAC 在此目标中引入了一个熵项。策略旨在最大化奖励，同时尽可能随机地行动：
$J_{SAC}(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} [ \gamma^t (r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))) ]$
这里，$\mathcal{H}(\pi(\cdot|s_t)) = \mathbb{E}_{a \sim \pi(\cdot|s_t)} [-\log \pi(a|s_t)]$ 是策略 $\pi$ 在状态 $s_t$ 的熵。温度参数 (parameter) $\alpha$ 控制着熵项与奖励的相对重要性。较高的 $\alpha$ 促进更多的环境交互 (更高的熵)，而较低的 $\alpha$ 则优先考虑奖励最大化。这种熵最大化鼓励代理更广泛地进行环境交互，并避免过早收敛到次优的确定性策略。与DDPG等方法相比，这也使得算法对超参数 (hyperparameter)的敏感度降低。

### SAC 架构组成

SAC 使用了多个神经网络 (neural network)，在理念上类似于TD3，但因最大熵目标和随机策略的使用而有所修改：

1. **随机策略网络（演员）$\pi_\phi(a|s)$：** 与DDPG或TD3使用确定性演员进行主要策略评估不同，SAC 使用随机演员。对于连续动作空间，该网络通常输出高斯分布的均值和标准差（或通过Tanh函数压缩的高斯分布来限制动作）。在训练和执行过程中，动作都从该分布中采样。参数 (parameter) $\phi$ 经过优化以最大化SAC目标。
2. **Q函数网络（评论家）$Q_{\theta_1}(s,a)$、$Q_{\theta_2}(s,a)$：** SAC 使用两个独立的Q网络，由 $\theta_1$ 和 $\theta_2$ 参数化，以减少Q值估计中的正偏差。这种“裁剪双Q”技巧借鉴自TD3。两个评论家都经过训练以逼近软Q值，软Q值包含了未来动作的熵奖励。
3. **目标Q网络 $Q_{\theta_{targ, 1}}(s,a)$、$Q_{\theta_{targ, 2}}(s,a)$：** 每个Q网络都维护有对应的目标网络。这些目标使用Polyak平均（在线网络参数和当前目标参数的加权平均）缓慢更新，为Q函数更新提供了稳定的目标。
   $\theta_{targ} \leftarrow \tau \theta + (1 - \tau) \theta_{targ}$
   其中 $\tau$ 是一个小的常数（例如0.005）。

在SAC的早期版本中，使用了一个单独的状态值网络 $V_\psi(s)$，但为了简化，较新的实现通常从Q函数和策略中隐式地推导出值。

### 网络训练

SAC 采用离策略操作，从存储在回放缓冲区 $\mathcal{D}$ 中的转换 $(s, a, r, s', d)$ 中学习。在每个训练步骤中，从 $\mathcal{D}$ 中采样一个迷你批次来更新网络。

#### 评论家（Q函数）更新

Q网络经过训练以最小化软贝尔曼残差。转换 $(s, a, r, s')$ 的目标值 $y$ 使用目标Q网络计算，并包含熵项：
$y(r, s') = r + \gamma (1-d) \left( \min_{i=1,2} Q_{\theta_{targ, i}}(s', a') - \alpha \log \pi_\phi(a'|s') \right), \quad \text{处 } a' \sim \pi_\phi(\cdot|s')$
请注意，$a'$ 是从给定下一状态 $s'$ 的当前策略 $\pi_\phi$ 中采样的一个*新*动作。使用两个目标Q值的最小值（裁剪双Q）。$(1-d)$ 项处理终止状态，如果 $d$（完成）为真，则下一状态的值为零。

每个Q网络的损失函数 (loss function)是均方贝尔曼误差（MSBE）：
$L_Q(\theta_i) = \mathbb{E}_{(s,a,r,s',d) \sim \mathcal{D}} \left[ \left( Q_{\theta_i}(s,a) - y(r, s') \right)^2 \right]$
此损失通过梯度下降 (gradient descent)最小化。

#### 演员（策略）更新

策略网络 $\pi_\phi$ 经过更新以最大化预期的软Q值加上策略熵。一个常见的最小化目标函数是：
$L_\pi(\phi) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\phi(\cdot|s)} \left[ \alpha \log \pi_\phi(a|s) - \min_{i=1,2} Q_{\theta_i}(s, a) \right]$
这里，动作 $a$ 是从回放缓冲区中的状态 $s$ 对应的当前策略 $\pi_\phi$ 中采样的。期望通过迷你批次近似计算。如果处理连续动作（例如，采样 $\epsilon \sim \mathcal{N}(0, I)$ 并计算 $a = \tanh(\mu_\phi(s) + \sigma_\phi(s) \odot \epsilon)$），梯度计算需要重参数 (parameter)化技巧。最小化此损失鼓励策略选择那些能够带来高Q值（由两个评论家中的最小值估计）并且具有高熵（低对数概率）的动作。

#### 温度参数 ($\alpha$) 更新（可选但建议）

温度 $\alpha$ 对性能影响很大。它平衡了奖励和熵目标。手动设置需要仔细调整。SAC 的一个主要优点是它能够**自动调整 $\alpha$**。这是通过提出另一个优化问题来实现的，其中目标是维持目标熵水平 $\mathcal{H}_0$。这个目标通常是启发式设定的，例如，设为动作空间的负维度：$\mathcal{H}_0 = -|\mathcal{A}|$。

$\alpha$ 的损失函数（必须是非负的，所以通常优化 $\log \alpha$）是：
$L(\alpha) = \mathbb{E}_{a_t \sim \pi_t(\cdot|s_t)} [ -\alpha \log \pi_t(a_t|s_t) - \alpha \mathcal{H}_0 ]$
通过梯度下降最小化此损失会调整 $\alpha$，使得如果策略的当前熵低于目标 $\mathcal{H}_0$，则 $\alpha$ 减小（使奖励更重要），如果熵更高，则 $\alpha$ 增大（使熵更重要）。

### 算法概览

1. 初始化策略网络 $\pi_\phi$、Q网络 $Q_{\theta_1}, Q_{\theta_2}$、目标网络 $Q_{\theta_{targ, 1}}, Q_{\theta_{targ, 2}}$ ($\theta_{targ} \leftarrow \theta$) 和回放缓冲区 $\mathcal{D}$。初始化 $\alpha$（如果自动调整）。
2. 对于每个环境步：
   a. 采样动作 $a_t \sim \pi_\phi(\cdot|s_t)$。
   b. 执行动作 $a_t$，观测奖励 $r_t$ 和下一状态 $s_{t+1}$。
   c. 将转换 $(s_t, a_t, r_t, s_{t+1}, d_t)$ 存储到 $\mathcal{D}$ 中。
3. 对于每个梯度步（通常每个环境步有多个梯度步）：
   a. 从 $\mathcal{D}$ 中采样一个迷你批次 $B = \{(s, a, r, s', d)\}$。
   b. 使用目标网络、策略采样 $a' \sim \pi_\phi(\cdot|s')$ 和当前 $\alpha$ 计算Q目标 $y$。
   c. 通过最小化 $L_Q(\theta_i)$ 更新Q函数 $\theta_1, \theta_2$。
   d. 通过最小化 $L_\pi(\phi)$ 更新策略 $\phi$。这需要从策略中采样 $a \sim \pi_\phi(\cdot|s)$。
   e. 如果自动调整 $\alpha$，则通过最小化 $L(\alpha)$ 更新 $\alpha$。
   f. 更新目标网络：$\theta_{targ, i} \leftarrow \tau \theta_i + (1 - \tau) \theta_{targ, i}$。

> 一个图示，表明软演员-评论家各组成部分之间的交互。数据从环境流向回放缓冲区。来自缓冲区的迷你批次用于更新策略（演员）、Q函数（评论家），以及可选的温度参数 (parameter) $\alpha$。目标网络为评论家更新提供稳定值，并通过Polyak平均缓慢更新。

### SAC的优点

- **样本效率：** 作为一种离策略算法，SAC 可以有效地从回放缓冲区复用数据，与PPO或A2C等在线策略方法相比，这通常会带来更好的样本效率，尤其是在复杂、高维度的连续控制任务中。
- **稳定性：** 裁剪双Q学习、目标网络的使用，特别是熵最大化框架，有助于实现比DDPG等算法更稳定的学习。自动温度调整通过调整环境交互/利用平衡进一步提高了稳定性。
- **环境交互：** 熵项明确鼓励代理进行环境交互。这种内置的环境交互机制可能非常有效，尤其是在奖励稀疏或动态复杂的任务中，朴素的环境交互策略可能失效。
- **性能：** SAC 在广泛的连续控制基准测试中展现出先进的性能，例如在MuJoCo或PyBullet中发现的那些。

通过结合Q学习（离策略更新、回放缓冲区）、策略梯度（演员更新）和最大熵框架中的思想，SAC 在演员-评论家系列中提供了一种强大且通常高效的方法，以应对有挑战性的强化学习 (reinforcement learning)问题。它对平衡奖励最大化与策略熵的侧重，使其在开发稳定高效的强化学习代理方面取得了重要的进展。

## 参考资料

- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf) — Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, Sergey Levine (2018)
  Journal: Proceedings of the 35th International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.1801.01290](https://doi.org/10.48550/arXiv.1801.01290)
  介绍了软演员-评论家（SAC）算法，提出了一种基于最大熵正则化的离策略演员-评论家方法，适用于连续控制问题。
- [Reinforcement Learning with Deep Energy-Based Policies](https://arxiv.org/pdf/1706.01518.pdf) — Tuomas Haarnoja, Yan Duan, Aurick Zhou, Marvin Ren, Pieter Abbeel, Sergey Levine (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning (ICML 2017); Publisher: PMLR; Volume: Volume 70; Pages: 1356-1365; DOI: [10.48550/arXiv.1706.01518](https://doi.org/10.48550/arXiv.1706.01518)
  提出了最大熵强化学习的理论基础，SAC算法在此基础上发展。
- [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf) — Scott Fujimoto, Herke van Hoof, David Precup (2018)
  Journal: Proceedings of the 35th International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.1802.09477](https://doi.org/10.48550/arXiv.1802.09477)
  介绍了TD3算法，包括SAC采用的截断双Q学习技巧和延迟策略更新。
- [Spinning Up in Deep RL: Soft Actor-Critic](https://spinningup.openai.com/en/latest/algorithms/sac.html) — Josh Achiam (2018)
  Publisher: OpenAI
  为SAC算法提供了清晰的解释和实践指南，包含伪代码和实现细节。

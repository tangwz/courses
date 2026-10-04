---
course: "advanced-reinforcement-learning"
chapter: "advanced-policy-gradients-actor-critic"
lesson: "actor-critic-implementation-practice"
sourceId: 3407
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-3-advanced-policy-gradients-actor-critic/actor-critic-implementation-practice"
title: "演员-评论家方法实现实践"
description: "动手实践实现PPO或SAC等算法，以用于标准强化学习基准测试。"
order: 10
plots: ["plots/3407-0.json"]
sourceHash: "f43fd107918e4dfdd54ab2277cb8d61c62ae97f0fc8d90c50c8a2a07291eedc4"
sourceCorrections: []
---

演员-评论家方法，例如优势演员-评论家 (A2C)、异步演员-评论家 (A3C)、深度确定性策略梯度 (DDPG)、信任区域策略优化 (TRPO)、近端策略优化 (PPO) 和软演员-评论家 (SAC)，是实际深度强化学习 (reinforcement learning)的根本。一个复杂的演员-评论家算法被实现，重点是近端策略优化 (PPO)。选择 PPO 是因为它性能优异，并且相对于 TRPO 而言实现起来更为简单。PPO 的实际实现揭示了构建高效深度强化学习智能体所涉及的挑战和设计选择。

### 实现准备

我们将重点实现带有裁剪替代目标的 PPO，这是一种常见且有效的变体。目标是在标准强化学习 (reinforcement learning)环境中训练一个智能体，例如 Gymnasium 库（OpenAI Gym 的维护分支）提供的环境。

**1. 环境：**
选择一个合适的*环境*。连续控制环境，如 `Pendulum-v1`、`LunarLanderContinuous-v2` 或 MuJoCo 任务（如果已安装），是测试针对连续动作空间设计的算法的良好选择，尽管 PPO 在离散动作空间中也能很好地工作（`CartPole-v1`、`LunarLander-v2`）。为了便于说明，我们将假设一个连续控制任务。

```python
# 环境设置示例 (使用 Gymnasium)
import gymnasium as gym
# env = gym.make("LunarLanderContinuous-v2", render_mode="human") # 示例
env = gym.make("Pendulum-v1") # 更简单的示例
observation_space = env.observation_space
action_space = env.action_space
print(f"观察空间: {observation_space}")
print(f"动作空间: {action_space}")
```

**2. 网络架构：**
您需要两个神经网络 (neural network)：

- **演员网络：** 接收状态 $s$ 作为输入，并输出策略分布的参数 (parameter)。对于连续动作，这通常是高斯分布的均值 $\mu_\theta(s)$ 和标准差 $\sigma_\theta(s)$。标准差可以是与状态无关的（学习到的参数）或与状态相关的（由网络输出）。
- **评论家网络：** 接收状态 $s$ 作为输入，并输出估计的状态值 $V_\phi(s)$。

这两个网络通常会共享一些用于特征提取的初始层，特别是当输入是高维数据（如图像）时，但也可以完全独立。对于基于向量 (vector)的观察，简单的多层感知机 (MLP) 通常就足够了。

```python
# 网络结构 (PyTorch 示例)
import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        # 共享层 (可选)
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        # 演员头
        self.actor_mean = nn.Linear(64, action_dim)
        # 可学习的对数标准差
        self.log_std = nn.Parameter(torch.ones(action_dim) * action_std_init)
        # 评论家头
        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        x = self.shared_layers(state)
        action_mean = self.actor_mean(x)
        value = self.critic(x)
        # 创建动作分布
        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)
        return dist, value

# 注意：与状态相关的标准差也很常见，
# 此时网络直接输出对数标准差，而不是单独的参数。
```

### 数据收集与优势计算

PPO 是一种同策略算法，这意味着它需要使用当前策略收集的新数据来执行更新。

**1. 交互循环：**
智能体使用当前的演员策略与*环境*交互固定数量的步长（例如 `T` 步，通常称为“rollout 长度”）。存储这些转换：$(s_t, a_t, r_{t+1}, s_{t+1}, \text{done}_t)$。动作 $a_t$ 从策略分布 $\pi_\theta(a_t|s_t)$ 中采样。存储所采取动作的对数概率 $\log \pi_\theta(a_t|s_t)$ 也非常重要，因为 PPO 目标函数需要它。

**2. 优势计算 (GAE)：**
一旦完成了 $T$ 步的 rollout，就需要计算每一步的优势估计。虽然可以使用简单的一步 TD 误差 ($r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$)，但广义优势估计 (GAE) 通常能提供更好的偏差-方差权衡。

回顾 GAE 公式：


$$
\hat{A}^{GAE}_t = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \delta_{t+l}
$$


其中 $\delta_{t+l} = r_{t+l+1} + \gamma V_\phi(s_{t+l+1}) - V_\phi(s_{t+l})$ 是步长 $t+l$ 处的 TD 误差，$\gamma$ 是折扣因子，且 $\lambda$ 是 GAE 平滑参数 (parameter) ($0 \le \lambda \le 1$)。

您使用评论家网络计算 rollout 中所有状态的 $V_\phi(s)$。然后，计算所有步长的 $\delta_t$，最后计算 $\hat{A}^{GAE}_t$。评论家更新的目标将是优势与值函数估计的和：$V_{target, t} = \hat{A}^{GAE}_t + V_\phi(s_t)$。

*提示：* 优势归一化 (normalization)（减去均值并除以批次中优势的标准差）是一种常见技术，可以显著稳定训练。

### PPO 目标函数与优化

有了收集到的 rollout 数据和计算出的优势，您现在可以计算 PPO 损失函数 (loss function)了。PPO 通常会在同一批 rollout 数据上执行多个梯度更新的 epoch。

**1. 裁剪替代目标 (演员损失)：**
PPO 的核心是裁剪目标函数，它旨在阻止策略进行大幅更新。首先，计算概率比率：


$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$


其中 $\theta_{old}$ 代表更新前的策略参数 (parameter)（用于收集数据）。由于我们存储了对数概率，这可以很容易地计算为 $r_t(\theta) = \exp(\log \pi_\theta(a_t|s_t) - \log \pi_{\theta_{old}}(a_t|s_t))$。

PPO 裁剪目标函数为：


$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$


其中 $\hat{A}_t$ 是优势估计（例如，来自 GAE），且 $\epsilon$ 是一个定义裁剪范围的小型超参数 (hyperparameter)（例如 0.1 或 0.2）。`clip(x, min_val, max_val)` 函数将值 `x` 限制在 `min_val` 和 `max_val` 之间。该目标函数取未裁剪项和裁剪项的最小值，当优势估计表明策略更新会带来益处（正优势）或损害（负优势）时，它会有效惩罚比率 $r_t(\theta)$ 超出 $[1-\epsilon, 1+\epsilon]$ 区间的变动。演员的目标是*最大化*此目标函数，因此损失函数通常是 $L^{CLIP}(\theta)$ 的负值。

**2. 值函数损失 (评论家损失)：**
评论家通过最小化其预测值 $V_\phi(s_t)$ 与计算目标值 $V_{target, t}$ 之间的均方误差进行更新：


$$
L^{VF}(\phi) = \mathbb{E}_t \left[ (V_\phi(s_t) - V_{target, t})^2 \right]
$$


一些 PPO 实现也会根据数据收集期间的值预测 $V_{\phi_{old}}(s_t)$ 的变化，对值损失进行类似于策略损失的裁剪。

**3. 熵奖励 (可选但推荐)：**
为了鼓励*尝试*，通常会在演员的目标函数中添加一个熵奖励（或从损失中减去）。熵 $H(\pi_\theta(\cdot|s_t))$ 衡量策略的随机性。


$$
L^S(\theta) = \mathbb{E}_t [H(\pi_\theta(\cdot|s_t))]
$$


最终的总损失通常为：


$$
L_{total}(\theta, \phi) = -L^{CLIP}(\theta) + c_1 L^{VF}(\phi) - c_2 L^S(\theta)
$$


其中 $c_1$（例如 0.5）和 $c_2$（例如 0.01）是加权系数。

**4. 优化：**
使用 Adam 等优化器通过最小化 $L_{total}$ 来更新参数 $\theta$ 和 $\phi$。对收集到的批次数据执行多个梯度步长（epoch）。

```python
# PPO 更新步骤 (PyTorch 风格伪代码)

# 假设：
# actions, states, old_log_probs 是来自 rollout 的张量
# advantages, value_targets 是计算出的 GAE/目标
# policy_net 是演员-评论家网络
# optimizer 更新 policy_net 参数
# K_EPOCHS 是每个 rollout 的更新 epoch 数量
# EPSILON 是裁剪参数

for _ in range(K_EPOCHS):
    # 在 rollout 数据上评估当前策略
    dist, values = policy_net(states)
    new_log_probs = dist.log_prob(actions).sum(axis=-1) # 对于多维动作求和
    entropy = dist.entropy().mean()

    # 计算比率
    ratios = torch.exp(new_log_probs - old_log_probs)

    # 演员损失 (裁剪替代目标)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - EPSILON, 1 + EPSILON) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()

    # 评论家损失 (均方误差)
    critic_loss = nn.functional.mse_loss(values.squeeze(), value_targets)

    # 总损失
    # c1 和 c2 是超参数 (值损失系数, 熵系数)
    loss = actor_loss + c1 * critic_loss - c2 * entropy

    # 执行优化步骤
    optimizer.zero_grad()
    loss.backward()
    # 可选：梯度裁剪
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
    optimizer.step()
```

### 实践考虑与调试

- **超参数 (parameter) (hyperparameter)：** PPO 对学习率、rollout 长度 ($T$)、更新 epoch 数量 ($K$)、批次大小（通常与 $T$ 相关）、$\gamma$、$\lambda$、$\epsilon$、$c_1$ 和 $c_2$ 等超参数很敏感。从文献中常报告的值（例如 $\epsilon=0.2, \lambda=0.95, \gamma=0.99$）开始，并系统地进行调整。
- **归一化 (normalization)：** 对观察值（减去均值，除以通过运行缓冲区或初始样本计算出的标准差）和优势进行归一化对于在不同*环境*中稳定训练非常重要。
- **初始化：** 请注意网络权重 (weight)的初始化，尤其是最终层。用小权重初始化演员的输出层可以防止策略在初始阶段发生大幅变化。适当初始化动作标准差也同样重要。
- **调试：** 在训练期间监控重要指标：
  - 平均回合奖励。
  - 演员和评论家损失值。
  - 策略的熵（应随时间缓慢下降）。
  - 更新后新旧策略之间的近似 KL 散度（PPO 目标是使其保持在一个较小的值）。
  - 值函数估计（应与实际回报相关）。

在*环境*中可视化智能体的行为 (`render_mode="human"`) 可以提供定性*见解*，尤其是在训练初期或调试失败时。



![PPO 训练表现 (Pendulum-v1)](plots/3407-0.json)



> 示例图表呈现了学习 Pendulum-v1 任务的智能体在训练时间步中平均每回合奖励的增加。监控此类曲线对于评估训练进度至关重要。

### 进阶练习

一旦您有了可运行的 PPO 实现，请考虑以下扩展：

- **实现 SAC：** 尝试实现软演员-评论家 (Soft Actor-Critic)。这涉及管理多个 Q 网络、目标网络和熵温度参数 (parameter) $\alpha$。在连续控制任务上比较其性能和样本效率与 PPO 的差异。
- **不同环境：** 在更广泛的*环境*中测试您的实现，包括具有离散动作或不同观察类型（例如，基于图像的 Atari 游戏，需要 CNN）的环境。
- **超参数 (hyperparameter)扫描：** 系统地*研究*不同超参数设置对学习性能和稳定性的影响。

### 运用现有库

尽管从头开始实现算法对于学习而言具有无价的价值，但 Stable Baselines3 (SB3)、RLlib (Ray 的一部分) 或 Tianshou 等成熟库提供了经过充分测试、优化且功能丰富的 PPO、SAC 和其他高级算法实现。在尝试了自己的实现之后*研究*它们的代码库，可以为您提供更深入的*见解*，了解高效的设计模式、高级功能（如循环策略或多智能体扩展）和实际优化。对于将强化学习 (reinforcement learning)应用于复杂问题而言，运用这些库通常比重新实现所有内容更具实用性。

这个实践练习巩固了您对演员-评论家方法的理解，为您应对强化学习中更复杂的挑战做准备。请记住，实现通常涉及细致的调优和调试，这些是在实际应用这些高级技术时非常重要的技能。

## 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍了近端策略优化（PPO）算法，这是一种流行且稳定的策略梯度方法。
- [High-dimensional continuous control using generalized advantage estimation](https://arxiv.org/abs/1506.02438) — John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel (2015)
  Journal: arXiv preprint arXiv:1506.02438; DOI: [10.48550/arXiv.1506.02438](https://doi.org/10.48550/arXiv.1506.02438)
  提出了广义优势估计（GAE），一种结合n步回报和TD(lambda)以提高策略梯度方法中优势估计效果的方法。
- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) — Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine (2018)
  Journal: International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.1801.01290](https://doi.org/10.48550/arXiv.1801.01290)
  介绍了软执行者-评论家（SAC），这是一种优化最大熵目标的离策略执行者-评论家算法。
- [Gymnasium Documentation](https://gymnasium.farama.org/) — The Gymnasium Developers (2024)
  Gymnasium 的官方文档，提供了强化学习环境中环境及其使用方法的详细信息。
- [PPO (Proximal Policy Optimization)](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) — Stable Baselines3 Contributors (2024)
  Stable Baselines3 库中 PPO 算法的实现细节和使用示例，对于实际应用和基准测试很有用。

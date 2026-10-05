# 实践：实现 REINFORCE

来源：[原文](https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-8-introduction-policy-gradient-methods/practice-implementing-reinforce)

[返回章节目录](README.md) · [返回课程目录](../README.md)

REINFORCE 算法（也称为蒙特卡洛策略梯度）通过直接调整策略 $\pi_\theta(a|s)$ 的参数 (parameter) $\theta$ 来最大化预期的未来奖励。其主要思想是采样完整的轨迹，计算每一步的收益 $G_t$，然后根据策略梯度定理所指引的方向更新策略参数。具体来说，如果某个动作在状态 $s_t$ 处导致高收益 $G_t$，则其对数概率应增加；否则，则降低。

更新规则，在最简单的形式下（不带基线），如下所示：

$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) G_t$

其中，$\alpha$ 是学习率，$\nabla_\theta \log \pi_\theta(a_t|s_t)$ 是所采取动作的对数概率的梯度，$G_t$ 是从时间步 $t$ 开始的收益。

我们将使用 Gymnasium 库中的经典 `CartPole-v1` 环境。在此环境中，目标是通过左右移动小车来平衡其上的杆子。状态由四个连续值（小车位置、小车速度、杆子角度、杆子角速度）表示，动作空间是离散的（0 表示向左，1 表示向右）。这非常适合 REINFORCE，因为策略需要将连续状态映射到离散动作。

### 环境与策略网络的设置

首先，请确保已安装 Gymnasium 和像 PyTorch 这样的深度学习 (deep learning)库。

```python
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import matplotlib.pyplot as plt

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 创建环境
env = gym.make('CartPole-v1')

# 获取状态和动作维度
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"State dimensions: {state_dim}")
print(f"Action dimensions: {action_dim}")
```

现在，让我们定义策略网络。一个简单的多层感知器 (MLP) 应该足以处理 CartPole。这个网络将把状态作为输入，并输出每个可能动作（左或右）的概率。

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        # 输出动作的原始得分 (logits)
        action_scores = self.fc2(x)
        # 使用 softmax 将得分转换为概率
        # 使用 dim=-1 将 softmax 应用于动作维度
        return F.softmax(action_scores, dim=-1)

# 实例化策略网络和优化器
policy_net = PolicyNetwork(state_dim, action_dim).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3) # 学习率
```

### REINFORCE 训练循环

REINFORCE 的核心是与环境进行交互以收集轨迹（状态、动作和奖励的序列），然后使用这些轨迹更新策略网络。

1. **收集轨迹：** 使用当前策略运行一个或多个完整回合。对于每一步，存储状态、采取的动作和获得的奖励。
2. **计算收益：** 对于回合中的每一步 $t$，计算折扣收益 $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$，其中 $T$ 是回合长度，$\gamma$ 是折扣因子（对于分段任务通常设置为 0.99 或 1.0）。
3. **计算策略损失：** 计算回合的损失项。这通常是所有时间步 $t$ 上 $\log \pi_\theta(a_t|s_t) G_t$ 的负和。我们使用负值是因为优化器执行梯度*下降*，而我们希望对预期收益执行梯度*上升*。
4. **更新策略网络：** 对损失进行反向传播 (backpropagation)，并使用优化器更新网络参数 (parameter) $\theta$。

让我们构建训练循环：

```python
# 定义一个结构来存储轨迹数据
SavedAction = namedtuple('SavedAction', ['log_prob', 'value']) # 这里的 Value 指的是收益 G_t

def select_action(state):
    """根据策略网络的输出概率选择一个动作。"""
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs = policy_net(state)
    # 根据动作概率列表创建分类分布
    m = torch.distributions.Categorical(probs)
    # 从分布中采样一个动作
    action = m.sample()
    # 存储采样动作的对数概率
    policy_net.saved_action = SavedAction(m.log_prob(action), 0) # G_t 的占位符
    return action.item()

def calculate_returns(rewards, gamma=0.99):
    """计算一个回合的折扣收益。"""
    R = 0
    returns = []
    # 反向遍历奖励
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R) # 前置以保持顺序
    # 归一化收益（可选但通常有帮助）
    returns = torch.tensor(returns, device=device)
    returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())
    return returns

def finish_episode(episode_rewards):
    """在一个回合结束时执行 REINFORCE 更新。"""
    policy_loss = []
    returns = calculate_returns(episode_rewards)

    # 检索保存的对数概率并将其与计算出的收益关联
    for (log_prob, _), G_t in zip(policy_net.all_saved_actions, returns):
         # REINFORCE 目标：最大化 log_prob * G_t
         # 损失是负目标
        policy_loss.append(-log_prob * G_t)

    # 对回合的损失求和
    optimizer.zero_grad() # 重置梯度
    loss = torch.stack(policy_loss).sum() # 合并损失
    loss.backward() # 计算梯度
    optimizer.step() # 更新网络权重

    # 清除为下一回合保存的动作
    del policy_net.all_saved_actions[:]

# 训练参数
num_episodes = 1000
gamma = 0.99
log_interval = 50 # 每 50 回合打印一次状态
max_steps_per_episode = 1000 # 防止回合过长

all_episode_rewards = []
episode_durations = []

# 主训练循环
for i_episode in range(num_episodes):
    state, _ = env.reset()
    episode_rewards = []
    policy_net.all_saved_actions = [] # 存储回合的对数概率

    for t in range(max_steps_per_episode):
        action = select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        policy_net.all_saved_actions.append(policy_net.saved_action)
        episode_rewards.append(reward)
        state = next_state

        if done:
            break

    # 回合结束，更新策略
    finish_episode(episode_rewards)

    # 记录
    total_reward = sum(episode_rewards)
    all_episode_rewards.append(total_reward)
    episode_durations.append(t + 1)

    if i_episode % log_interval == 0:
        avg_reward = np.mean(all_episode_rewards[-log_interval:])
        print(f'Episode {i_episode}\tAverage Reward (last {log_interval}): {avg_reward:.2f}\tLast Duration: {t+1}')

    # 可选：如果问题已解决则停止训练
    # 如果在连续 100 个回合中平均奖励超过 475，则认为 CartPole-v1 已解决
    if len(all_episode_rewards) > 100:
         if np.mean(all_episode_rewards[-100:]) > 475:
              print(f'\nSolved in {i_episode} episodes!')
              break

env.close()
```

### 结果可视化

训练结束后，绘制每回合奖励图有助于查看智能体是否有效学习。

```python
# 绘制结果
plt.figure(figsize=(12, 6))
plt.plot(all_episode_rewards)
plt.title('REINFORCE：随时间变化的每回合奖励')
plt.xlabel('回合')
plt.ylabel('总奖励')

# 计算并绘制滚动平均值
rolling_avg = np.convolve(all_episode_rewards, np.ones(100)/100, mode='valid')
plt.plot(np.arange(99, len(all_episode_rewards)), rolling_avg, label='100 回合滚动平均值', color='orange')
plt.legend()
plt.grid(True)
plt.show()
```



[交互图表：REINFORCE：随时间变化的每回合奖励](https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-8-introduction-policy-gradient-methods/practice-implementing-reinforce#plot-5ssl93)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "scatter",
      "mode": "lines",
      "name": "每回合奖励",
      "x": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9
      ],
      "y": [
        10,
        20,
        15,
        25,
        30,
        22,
        18,
        28,
        35,
        29
      ],
      "line": {
        "color": "#339af0"
      }
    },
    {
      "type": "scatter",
      "mode": "lines",
      "name": "100 回合滚动平均值",
      "x": [
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108
      ],
      "y": [
        18.5,
        19.2,
        19.8,
        20.5,
        21.0,
        20.7,
        20.3,
        20.9,
        21.5,
        21.2
      ],
      "line": {
        "color": "#fd7e14"
      }
    }
  ],
  "layout": {
    "title": "REINFORCE：随时间变化的每回合奖励",
    "xaxis": {
      "title": "回合"
    },
    "yaxis": {
      "title": "总奖励"
    },
    "template": "plotly_white"
  }
}
```

</details>



> 图表显示了训练期间每回合获得的总奖励，以及 100 回合的滚动平均值，以便观察学习趋势。

### 讨论

这个实现呈现了基本的 REINFORCE 算法。你应该会观察到智能体的表现逐渐改善，表现为更长的回合持续时间和更高的总奖励。

- **高方差：** 你可能会注意到（或者多次运行后会体会到）的一个特点是学习过程中的高方差。有时 REINFORCE 学习得很快，而有时它可能会遇到困难或需要更长时间。这是因为梯度估计依赖于蒙特卡洛采样（完整回合），这可能会产生噪声。
- **归一化 (normalization)的重要性：** 对收益进行归一化（`returns = (returns - returns.mean()) / (returns.std() + eps)`）是一种常见的技术，有助于稳定训练。它将收益集中在零附近，这意味着导致高于平均收益的动作概率会增加，而导致低于平均收益的动作概率会降低。
- **基线：** 如本章前面所述，从收益 $G_t$ 中减去一个基线（例如状态价值估计 $V(s_t)$）可以明显减少方差而不会引入偏差。更新将涉及 $(G_t - b(s_t))$ 而不仅仅是 $G_t$。实现这一点通常会导致一种 Actor-Critic 方法，其中一个网络（评论者）估计基线 $V(s_t)$，而另一个网络（执行者）根据优势 $G_t - V(s_t)$ 更新策略。

这个实践示例为理解和实现策略梯度方法提供了基础。你可以自由尝试不同的超参数 (parameter) (hyperparameter)（学习率、网络结构、折扣因子），或者尝试实现一个基线，看看它如何影响性能。

## 参考资料

- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本标准教材，涵盖了策略梯度方法、REINFORCE以及使用基线进行方差缩减的内容。
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Core Team (2023)
  PyTorch深度学习框架的官方文档，用于实现策略网络和训练循环。
- [Gymnasium Documentation](https://gymnasium.farama.org/) — The Gymnasium Team (2023)
  Gymnasium库的官方文档，提供了CartPole-v1等用于强化学习智能体训练的环境详情。

---

[上一节](06-%E5%AF%B9%E6%AF%94%E5%9F%BA%E4%BA%8E%E4%BB%B7%E5%80%BC%E5%92%8C%E5%9F%BA%E4%BA%8E%E7%AD%96%E7%95%A5%E7%9A%84%E6%96%B9%E6%B3%95.md)

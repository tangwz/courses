# 动手实践：实现REINFORCE算法

来源：[原文](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-4-policy-gradient-methods/reinforce-implementation-practical)

[返回章节目录](README.md) · [返回课程目录](../README.md)

逐步指导如何使用Python和PyTorch或TensorFlow等深度学习 (deep learning)库来实现基本的REINFORCE算法，该算法常被称为蒙特卡洛策略梯度。此实现将应用于一个经典控制问题——CartPole，此环境可通过Gymnasium（前身为OpenAI Gym）库获得。

我们的目标是训练一个智能体，使其学习策略$\pi(a|s; \theta)$，以便尽可能长时间地在小车上平衡杆子。REINFORCE通过根据完整回合的结果调整策略参数 (parameter)$\theta$来实现此目标。

### 环境准备

首先，请确保您已安装Gymnasium（`pip install gymnasium[classic_control]`）。我们将使用`CartPole-v1`环境。它提供观测值（小车位置、小车速度、杆子角度、杆子角速度），预期离散动作（0表示向左推，1表示向右推），并且在杆子保持直立的每个时间步都给予+1的奖励。如果杆子角度超出某个阈值，小车离中心太远，或经过500个时间步后，一个回合便会结束。

```python
# Gymnasium使用示例
import gymnasium as gym

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"State dimensions: {state_dim}") # 输出: 4
print(f"Action dimensions: {action_dim}") # 输出: 2
```

### 策略网络

我们需要一个函数逼近器来表示我们的策略$\pi(a|s; \theta)$。一个简单的前馈神经网络 (neural network)适用于CartPole。该网络将状态作为输入，并为每个可能的动作输出概率。

- **输入层：** 大小与状态维度匹配（CartPole为4）。
- **隐藏层：** 一个或多个带有激活函数 (activation function)（如ReLU）的层，以引入非线性。像64或128个神经元的大小通常效果不错。
- **输出层：** 大小与离散动作的数量匹配（CartPole为2）。应用`softmax`激活函数以确保输出表示动作的有效概率分布。

```python
# 策略网络结构示例（使用PyTorch）
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        action_probs = F.softmax(self.fc2(x), dim=-1)
        return action_probs
```

### REINFORCE智能体逻辑

让我们分解REINFORCE智能体的核心组成部分。

1. **初始化：** 创建`PolicyNetwork`的一个实例。选择一个优化器，例如Adam，来更新网络的权重 (weight)（$\theta$）。设置学习率。

   ```python
   # 初始化示例
   policy_net = PolicyNetwork(state_dim, action_dim)
   optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
   gamma = 0.99 # 折扣因子
   ```
2. **动作选择：** 给定状态$s$，将其通过`policy_net`以获取动作概率。从此概率分布中采样一个动作$a$。在PyTorch中，您可以使用`torch.distributions.Categorical`。存储所选动作的对数概率$\log \pi(a|s; \theta)$；更新时会用到它。

   ```python
   # 动作选择示例
   def select_action(state, policy_net):
       state_tensor = torch.FloatTensor(state).unsqueeze(0) # 添加批次维度
       action_probs = policy_net(state_tensor)
       distribution = torch.distributions.Categorical(action_probs)
       action = distribution.sample()
       log_prob = distribution.log_prob(action)
       return action.item(), log_prob
   ```
3. **回合执行：** 在环境中运行一个完整的回合：

   - 重置环境以获取初始状态$s_0$。
   - 循环直到回合结束：
     - 使用当前策略网络和状态$s_t$选择动作$a_t$。记录对数概率$\log \pi(a_t|s_t; \theta)$。
     - 在环境中执行动作$a_t$，获取下一个状态$s_{t+1}$、奖励$r_{t+1}$和终止信号。
     - 存储奖励$r_{t+1}$和对数概率$\log \pi(a_t|s_t; \theta)$。
     - 更新当前状态$s_t \leftarrow s_{t+1}$。
   - 保留回合期间收集的所有对数概率和奖励的列表。
4. **回报计算：** 回合结束后（在步骤$T$），计算回合中每个时间步$t$的折扣回报$G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_{k+1}$。一种有效计算此值的方法是从回合末尾开始向后迭代：

   - 初始化$G_T = 0$。
   - 对于$t = T-1, T-2, \dots, 0$：
     - $G_t = r_{t+1} + \gamma G_{t+1}$。
   - 通常，对回合中的回报进行标准化处理（例如，减去平均值并除以标准差）有助于稳定训练。

   ```python
   # 回报计算示例
   def calculate_returns(rewards, gamma):
       returns = []
       discounted_return = 0
       for r in reversed(rewards):
           discounted_return = r + gamma * discounted_return
           returns.insert(0, discounted_return) # 预先添加以保持顺序

       returns = torch.tensor(returns)
       # 标准化回报（可选但建议）
       returns = (returns - returns.mean()) / (returns.std() + 1e-9)
       return returns
   ```
5. **损失计算和梯度更新：** 计算REINFORCE损失。目标是最大化预期回报，因此我们执行梯度*上升*。大多数深度学习 (deep learning)库实现的是梯度*下降*，因此我们最小化目标函数的*负值*。一个回合的损失是：

   
   $$
   L(\theta) = - \sum_{t=0}^{T-1} \log \pi(a_t|s_t; \theta) G_t
   $$
   

   使用存储的对数概率和回合中计算的回报来计算此和。然后，执行反向传播 (backpropagation)并使用优化器更新网络参数 (parameter)。

   ```python
   # 更新步骤示例
   def update_policy(log_probs, returns, optimizer):
       loss = []
       for log_prob, Gt in zip(log_probs, returns):
           loss.append(-log_prob * Gt) # 负号表示通过最小化实现梯度上升

       optimizer.zero_grad()
       policy_loss = torch.stack(loss).sum() # 对回合中的损失求和
       policy_loss.backward()
       optimizer.step()
   ```

### 训练循环

整个训练过程涉及运行多个回合并在每个回合后更新策略网络。

```python
# 简化训练循环结构
num_episodes = 1000
episode_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_log_probs = []
    episode_rewards_raw = []
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action, log_prob = select_action(state, policy_net)
        next_state, reward, terminated, truncated, _ = env.step(action)

        episode_log_probs.append(log_prob)
        episode_rewards_raw.append(reward)

        state = next_state

    # 回合结束后计算回报并更新策略
    returns = calculate_returns(episode_rewards_raw, gamma)
    update_policy(episode_log_probs, returns, optimizer)

    total_episode_reward = sum(episode_rewards_raw)
    episode_rewards.append(total_episode_reward)

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1}, Average Reward (last 50): {sum(episode_rewards[-50:])/50:.2f}")

env.close()
```

### 训练进度可视化

绘制每回合的总奖励（或移动平均值）对于判断智能体是否在学习很重要。



[交互图表：CartPole-v1上的REINFORCE训练进度](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-4-policy-gradient-methods/reinforce-implementation-practical#plot-14rmgml)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        0,
        50,
        100,
        150,
        200,
        250,
        300,
        350,
        400,
        450,
        500,
        550,
        600,
        650,
        700,
        750,
        800,
        850,
        900,
        950
      ],
      "y": [
        21.5,
        25.8,
        35.2,
        48.9,
        65.1,
        88.3,
        115.6,
        150.2,
        195.7,
        230.4,
        280.1,
        325.9,
        370.5,
        410.8,
        435.2,
        450.6,
        465.3,
        478.9,
        485.1,
        490.7
      ],
      "type": "scatter",
      "mode": "lines+markers",
      "name": "平均奖励（移动平均）"
    }
  ],
  "layout": {
    "title": "CartPole-v1上的REINFORCE训练进度",
    "xaxis": {
      "title": "回合"
    },
    "yaxis": {
      "title": "平均奖励（每50回合）",
      "range": [
        0,
        510
      ]
    },
    "height": 400
  }
}
```

</details>



> CartPole上REINFORCE的典型学习曲线，显示了每50个回合窗口的平均奖励随时间提高。最大可能奖励为500。

### 加入基线

正如前面讨论的，REINFORCE存在高方差问题。从回报$G_t$中减去一个基线$b(s_t)$可以大幅减少这种方差，同时不引入偏差。一个常见基线是状态价值函数$V(s_t)$。我们可以使用另一个神经网络 (neural network)（“评论家”）来估计$V(s_t)$，该网络经过训练以预测从状态$s_t$获得的预期回报。

修改后的更新目标是最小化：


$$
L(\theta) = - \sum_{t=0}^{T-1} \log \pi(a_t|s_t; \theta) (G_t - V(s_t; \phi))
$$


其中$V(s_t; \phi)$是由参数 (parameter)为$\phi$的评论家网络估计的值。评论家网络本身通常通过监督学习 (supervised learning)进行训练，通过最小化其预测$V(s_t; \phi)$与实际计算的回报$G_t$之间的平方误差。这种架构引导我们走向演员-评论家方法，我们将在下一章中介绍这些方法。

即使是简单的基线，例如回合的平均回报，有时也能有所帮助。在此次实践中，我们侧重于REINFORCE算法的核心。尝试基线是下一步的重要工作。

这个动手示例展示了REINFORCE算法的基本机制。尽管它很简单，但它强调了我们如何能够基于采样的轨迹及其回报，使用梯度上升直接优化策略。请记住，调整超参数 (hyperparameter)（学习率、网络架构、折扣因子、标准化）通常是获得良好性能所必需的。

## 参考资料

- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  全面介绍了强化学习，包括策略梯度方法和REINFORCE算法的详细解释。
- [Gymnasium Documentation](https://gymnasium.farama.org/) — The Gymnasium Developers (2023)
  Gymnasium库的官方文档，提供了CartPole等经典控制问题的环境细节和使用指南。
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/) — OpenAI (2018)
  Publisher: OpenAI
  高质量的教育资源，解释了包括REINFORCE在内的多种深度强化学习算法，并提供了实践实现见解。

---

[上一节](06-%E6%96%B9%E5%B7%AE%E5%87%8F%E5%B0%91%E7%9A%84%E5%9F%BA%E7%BA%BF.md) · [下一节](../05-Actor-Critic%20%E6%96%B9%E6%B3%95/01-%E7%BB%93%E5%90%88%E7%AD%96%E7%95%A5%E5%92%8C%E4%BB%B7%E5%80%BC%E8%AF%84%E4%BC%B0.md)

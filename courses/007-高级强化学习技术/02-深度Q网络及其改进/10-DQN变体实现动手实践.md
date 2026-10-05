# DQN变体实现动手实践

来源：[原文](https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-2-deep-q-networks/dqn-implementation-practice)

[返回章节目录](README.md) · [返回课程目录](../README.md)

为深度Q网络（DQN）及双DQN（DDQN）、对决网络架构等几项重要改进提供实用实现指导和代码示例。这些实现的起点是标准的DQN。假定您熟悉Python、深度学习 (deep learning)库（如PyTorch或TensorFlow），并能够与强化学习 (reinforcement learning)环境（例如，使用Gymnasium）进行交互。

我们的目的并非提供一个完整、优化、即插即用的库，而是说明实现DDQN和对决DQN所需的具体代码修改，从DQN代理结构开始。

### 基本DQN结构

在为DDQN或对决DQN进行修改之前，让我们回顾典型DQN代理的必要组成部分：

1. **Q网络：** 一个神经网络 (neural network)（例如，`torch.nn.Module`或`tf.keras.Model`），它接收状态表示作为输入，并为每个可能动作输出Q值。
2. **目标网络：** 一个与Q网络结构相同的网络，但其权重 (weight)根据Q网络的权重定期（或通过Polyak平均缓慢）更新。用于稳定目标值计算。
3. **回放缓冲区：** 一个存储转换`$(s, a, r, s', d)$`（状态、动作、奖励、下一状态、完成标记 (token)）的数据结构（通常是双端队列）。
4. **代理类：** 协调与环境的交互和学习过程。主要方法通常包含：
   - `act(state)`: 根据当前状态选择动作，使用epsilon-greedy策略，参照Q网络。
   - `step(state, action, reward, next_state, done)`: 将转换存储在回放缓冲区中，并可能触发学习更新。
   - `learn()`: 从回放缓冲区采样一批转换，计算目标Q值，计算目标Q值和预测Q值之间的损失（例如，MSE或Huber损失），并在Q网络上执行梯度下降 (gradient descent)步骤。定期更新目标网络。

### 实现双DQN (DDQN)

双DQN的核心思想是减少标准Q学习和DQN中存在的过高估计偏差。回顾标准DQN对转换$(s, a, r, s', d)$的目标计算：


$$
Y_t^{\text{DQN}} = r + \gamma (1 - d) \max_{a'} Q_{\theta^-}(s', a')
$$


这里，$Q_{\theta^-}$表示目标网络。max运算符使用目标网络既*选择*最佳的下一动作，又*评估*该动作的价值。DDQN将此解耦：


$$
Y_t^{\text{DoubleDQN}} = r + \gamma (1 - d) Q_{\theta^-}(s', \arg\max_{a'} Q_{\theta}(s', a'))
$$


*在线*网络 ($Q_{\theta}$) 用于为下一状态$s'$选择最佳动作 $a' = \arg\max_{a'} Q_{\theta}(s', a')$，但*目标*网络 ($Q_{\theta^-}$) 用于评估在状态$s'$中执行该动作$a'$的Q值。

**代码修改：**

修改主要发生在`learn`方法中，具体来说，是在计算采样批次的目标Q值处。

```python
# 假设：
# - states, actions, rewards, next_states, dones 是从回放缓冲区采样的批次
# - q_network 是在线网络 (theta)
# - target_q_network 是目标网络 (theta_minus)
# - gamma 是折扣因子
# - 为便于说明，使用类似PyTorch的语法

# 1. 从在线网络获取下一状态的Q值
with torch.no_grad(): # 此处无需追踪梯度
    q_next_online = q_network(next_states) # 形状: (batch_size, 动作数量)

# 2. 使用在线网络选择下一状态的最佳动作
best_actions_next = torch.argmax(q_next_online, dim=1).unsqueeze(-1) # 形状: (batch_size, 1)

# 3. 从目标网络获取下一状态的Q值
with torch.no_grad():
    q_next_target = target_q_network(next_states) # 形状: (batch_size, 动作数量)

# 4. 从目标网络中选择与在线网络选择的最佳动作相对应的Q值
# 使用 gather() 根据 best_actions_next 索引选择Q值
q_target_next = torch.gather(q_next_target, dim=1, index=best_actions_next) # 形状: (batch_size, 1)

# 5. 计算DDQN目标值
# 处理终止状态（其中 next_state 为 None 或 done=True）
# dones 张量对于非终止状态通常为0，对于终止状态为1
target_q_values = rewards.unsqueeze(-1) + (gamma * q_target_next * (1 - dones.unsqueeze(-1)))

# --- 学习步骤的其余部分如下 ---
# 6. 从在线网络获取已采取动作的当前Q值
current_q_values = torch.gather(q_network(states), dim=1, index=actions.unsqueeze(-1))

# 7. 计算损失 (例如, MSE损失)
loss = F.mse_loss(current_q_values, target_q_values)

# 8. 执行梯度下降
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 9. 更新目标网络（定期或通过Polyak平均）
# ...
```

此片段显示了重要差异：使用`q_network`查找`best_actions_next`，然后使用`target_q_network`获取与这些动作相关的值`q_target_next`。

### 实现对决网络架构

对决网络架构分离了状态值函数$V(s)$和动作优势函数$A(s, a)$的估算。其想法是，有时状态的价值与所采取的动作无关，并且网络应该能够表示这一点。这些被组合以生成最终Q值。

一种常见表述为：


$$
Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a'; \theta, \alpha) \right)
$$


其中$\theta$表示共享层的参数 (parameter)，$\beta$表示值流，$\alpha$表示优势流。减去平均优势确保了可识别性并提升了稳定性。另一种方法是减去最大优势：


$$
Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \max_{a'} A(s, a'; \theta, \alpha) \right)
$$


**代码修改：**

主要修改在于Q网络模型的定义。您需要修改其架构，使其从一个共同的特征表示层分出两个输出流（值流和优势流）。

```python
# 使用 PyTorch nn.Module 的示例
import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_units=[64, 64]):
        super(DuelingQNetwork, self).__init__()
        self.action_size = action_size

        # 共享层
        self.fc1 = nn.Linear(state_size, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])

        # 值流
        self.value_stream = nn.Linear(hidden_units[1], 1)

        # 优势流
        self.advantage_stream = nn.Linear(hidden_units[1], action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # 计算值 V(s)
        value = self.value_stream(x) # 形状: (batch_size, 1)

        # 计算优势 A(s, a)
        advantages = self.advantage_stream(x) # 形状: (batch_size, 动作数量)

        # 使用均值减法方法组合 V(s) 和 A(s, a)
        # 注意: 保持维度一致以进行广播
        # value 形状: (batch_size, 1)
        # advantages 形状: (batch_size, 动作数量)
        # advantages.mean(dim=1, keepdim=True) 形状: (batch_size, 1)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        # 替代方法: 使用最大值减法
        # q_values = value + (advantages - advantages.max(dim=1, keepdim=True)[0])

        return q_values # 形状: (batch_size, 动作数量)

# --- 使用方法 ---
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n

# q_network = DuelingQNetwork(state_dim, action_dim)
# target_q_network = DuelingQNetwork(state_dim, action_dim)
# target_q_network.load_state_dict(q_network.state_dict()) # 初始化目标网络权重

# 代理训练循环的其余部分（经验回放、目标更新、损失计算）
# 与标准DQN或DDQN相同。只需使用此DuelingQNetwork
# 用于在线和目标网络。
```

使用对决架构时，损失计算、目标更新和交互循环没有根本性的改变。您只需用对决版本替换您的标准Q网络模型定义，用于在线和目标网络。通过将对决网络架构用于$Q_{\theta}$和$Q_{\theta^-}$并应用前面显示的DDQN目标计算逻辑，您可以轻松地与DDQN结合。

### 实验与观察

实现这些变体是第一步。接下来是进行实验并观察它们的影响。

- **环境：** 从Gymnasium的经典控制环境开始，如`CartPole-v1`或`LunarLander-v2`，以加快迭代速度。然后，转向更复杂的任务，如雅达利游戏（例如，`PongNoFrameskip-v4`，`BreakoutNoFrameskip-v4`），使用适当的包装器进行帧跳过和堆叠。
- **指标：** 追踪每集累积奖励。绘制滚动窗口（例如，100集）内的平均奖励有助于展现学习进度和稳定性。
- **假设影响：**
  - **DDQN：** 初期学习速度可能略慢于普通DQN，但通常会带来更稳定的训练和可能更高的最终性能，因为减少了Q值高估。
  - **对决DQN：** 可能显示更快的学习速度，特别是在状态价值不那么依赖于所采取的具体动作的环境中，或许多动作具有相似Q值的情况下。它使网络能够更高效地学习状态值$V(s)$。

下面是一个图表，展现了您可能观察到的潜在差异：



[交互图表：学习曲线（平均每集奖励）](https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-2-deep-q-networks/dqn-implementation-practice#plot-t5o5ey)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "scatter",
      "mode": "lines",
      "name": "DQN",
      "x": [
        0,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000
      ],
      "y": [
        10,
        25,
        50,
        80,
        110,
        130,
        145,
        155,
        160,
        162,
        163
      ],
      "line": {
        "color": "#4263eb"
      }
    },
    {
      "type": "scatter",
      "mode": "lines",
      "name": "DDQN",
      "x": [
        0,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000
      ],
      "y": [
        8,
        20,
        45,
        75,
        105,
        135,
        155,
        170,
        180,
        185,
        188
      ],
      "line": {
        "color": "#12b886"
      }
    },
    {
      "type": "scatter",
      "mode": "lines",
      "name": "对决DQN",
      "x": [
        0,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000
      ],
      "y": [
        12,
        30,
        65,
        100,
        135,
        160,
        175,
        185,
        190,
        193,
        195
      ],
      "line": {
        "color": "#f59f00"
      }
    }
  ],
  "layout": {
    "title": "学习曲线（平均每集奖励）",
    "xaxis": {
      "title": "集数"
    },
    "yaxis": {
      "title": "平均奖励"
    },
    "template": "plotly_white",
    "legend": {
      "yanchor": "bottom",
      "y": 0.01,
      "xanchor": "right",
      "x": 0.99
    }
  }
}
```

</details>



> DQN、双DQN和对决DQN的学习进展比较，显示训练集数上的平均每集奖励。实际结果会根据环境、实现细节和超参数 (parameter) (hyperparameter)有很大差异。

请记住，超参数调优（学习率、回放缓冲区大小、目标网络更新频率、epsilon衰减计划、网络架构）对于任何这些算法获得良好性能都非常重要。需要进行实验来找到适合您具体问题的设置。结合这些技术，例如在DDQN更新规则中使用对决架构（对决DDQN），是一种常见做法。

## 参考资料

- [Playing Atari with Deep Reinforcement Learning](https://doi.org/10.48550/arXiv.1312.5602) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller (2013)
  Journal: arXiv preprint arXiv:1312.5602; DOI: [10.48550/arXiv.1312.5602](https://doi.org/10.48550/arXiv.1312.5602)
  介绍了深度Q网络的基础论文，首次成功地将深度学习与强化学习结合应用于复杂控制任务。
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — Hado van Hasselt, Arthur Guez, David Silver (2016)
  Journal: Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence (AAAI); Pages: 2094-2100; DOI: [10.48550/arXiv.1509.06461](https://doi.org/10.48550/arXiv.1509.06461)
  介绍了Double DQN，一种减少Q学习中过高估计偏差的方法，提高了训练的稳定性和性能。
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) — Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas (2016)
  Journal: Proceedings of the 33rd International Conference on Machine Learning (ICML); Volume: 48; Pages: 1995-2003; DOI: [10.48550/arXiv.1511.06581](https://doi.org/10.48550/arXiv.1511.06581)
  提出了Dueling网络架构，它将状态值函数和优势函数分离，提高了Q值估计的效率。
- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本全面的教材，涵盖了强化学习的理论基础，包括Q学习和无模型控制。

---

[上一节](09-Rainbow%20DQN%20%E9%9B%86%E6%88%90.md) · [下一节](../03-%E9%AB%98%E7%BA%A7%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E4%B8%8E%20Actor-Critic%20%E6%96%B9%E6%B3%95/01-%E5%9F%BA%E6%9C%AC%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E9%9D%A2%E4%B8%B4%E7%9A%84%E6%8C%91%E6%88%98.md)

---
course: "intro-to-reinforcement-learning"
chapter: "introduction-deep-q-networks-dqn"
lesson: "hands-on-building-basic-dqn"
sourceId: 1664
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-7-introduction-deep-q-networks-dqn/hands-on-building-basic-dqn"
title: "动手实践：构建一个基础DQN"
description: "基础DQN代理的代码演练或简单实现。"
order: 7
plots: ["plots/1664-0.json"]
sourceHash: "b66d06348a0bf5da36e2e2fbdd8caeac5f1fad2d04d817100696a9cf31d39a58"
sourceCorrections: []
---

构建基础的DQN（Deep Q-Networks）代理，需要掌握其具体的实践步骤。这包括理解Q学习中使用神经网络 (neural network)所遇到的难点，以及经验回放和目标网络等解决办法。DQN代理的结构和逻辑将得到概述，重点关注其主要组成部分如何协同。Gymnasium库中的经典`CartPole-v1`环境可作为试验平台。此环境具备连续状态空间（小车位置、小车速度、杆子角度、杆子角速度）和离散动作空间（向左推或向右推），因此它适合用于展示DQN，且不过于复杂。

通常，你需要Gymnasium等库来提供环境，NumPy用于数值运算，以及PyTorch或TensorFlow等深度学习 (deep learning)框架来构建神经网络。

### 1. 经验回放缓冲区

首先，我们需要经验回放缓冲区。它的作用是存储转换（状态、动作、奖励、下一状态、完成标志），并允许我们从过往经验中采样随机小批量数据。这打破了用于训练的连续样本之间的相关性，提升了稳定性。

一个简单的实现是使用Python的`collections.deque`，并设定最大长度。

```python
import collections
import random
import numpy as np

# 经验回放缓冲区结构
class ReplayBuffer:
    def __init__(self, capacity):
        # 使用deque，因为它自动处理最大大小
        self.buffer = collections.deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        """在缓冲区中存储一个转换元组。"""
        # 确保状态是NumPy数组以保持一致性
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        # 将经验元组添加到deque中
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """采样一个经验小批量。"""
        # 随机选择批次的索引
        batch_indices = random.sample(range(len(self.buffer)), batch_size)
        # 获取与采样索引对应的经验
        experiences = [self.buffer[i] for i in batch_indices]

        # 将批次解压为状态、动作等单独的数组
        states, actions, rewards, next_states, dones = zip(*experiences)

        # 转换为NumPy数组，供网络批量处理
        return (np.concatenate(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.concatenate(next_states),
                np.array(dones, dtype=np.uint8))

    def __len__(self):
        """返回缓冲区当前大小。"""
        return len(self.buffer)

# 示例用法：
# buffer = ReplayBuffer(capacity=10000)
# buffer.store(state, action, reward, next_state, done)
# if len(buffer) > batch_size:
#     states, actions, rewards, next_states, dones = buffer.sample(batch_size)
```

### 2. Q网络和目标网络

我们需要两个具有相同架构的神经网络 (neural network)：主Q网络（其权重 (weight)$\theta$我们频繁更新）和目标网络（其权重$\theta^{-}$定期从Q网络更新）。对于CartPole，一个简单的多层感知机（MLP）就足够了。

- **输入层：** 大小匹配状态维度（CartPole为4）。
- **隐藏层：** 一层或两层全连接层，带ReLU激活函数 (activation function)（例如，64或128个神经元）。
- **输出层：** 大小匹配离散动作的数量（CartPole为2），带线性激活函数。输出值表示给定输入状态下每个动作的估计Q值。

以下是使用PyTorch风格语法的表示（TensorFlow会类似）：

```python
# 网络定义（使用PyTorch风格结构）
# import torch
# import torch.nn as nn
# import torch.optim as optim

# class QNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=128):
#         super(QNetwork, self).__init__()
#         self.layer1 = nn.Linear(state_dim, hidden_dim)
#         self.layer2 = nn.Linear(hidden_dim, hidden_dim)
#         self.output_layer = nn.Linear(hidden_dim, action_dim)
#         self.relu = nn.ReLU()

#     def forward(self, state):
#         x = self.relu(self.layer1(state))
#         x = self.relu(self.layer2(x))
#         q_values = self.output_layer(x) # Q值的线性激活
#         return q_values

# # 初始化网络
# state_dim = 4 # CartPole状态大小
# action_dim = 2 # CartPole动作大小
# q_network = QNetwork(state_dim, action_dim)
# target_network = QNetwork(state_dim, action_dim)

# # 初始化目标网络权重以匹配Q网络
# target_network.load_state_dict(q_network.state_dict())
# target_network.eval() # 设置目标网络为评估模式

# # Q网络的优化器（例如Adam）
# optimizer = optim.Adam(q_network.parameters(), lr=0.001)
```

请记住定期将权重从`q_network`复制到`target_network`。这通常每`C`步或每`C`个回合进行一次。

### 3. 动作选择（$\epsilon$-贪婪）

训练期间，代理需要在环境的试探和对当前知识的利用之间取得平衡。$\epsilon$-贪婪策略实现了这一点：

- 以$\epsilon$的概率，选择一个随机动作（试探）。
- 以$1-\epsilon$的概率，从Q网络中选择估计Q值最高的动作（利用）。

$\epsilon$通常从高值（例如1.0）开始，并随时间（例如线性或指数地）衰减，趋向一个小的最小值（例如0.01或0.1）。

```python
# \epsilon-贪婪动作选择
# epsilon_start = 1.0
# epsilon_end = 0.1
# epsilon_decay_steps = 10000
# current_step = 0

# def select_action(state, q_network, current_step):
#     # 根据衰减计划计算当前\epsilon
#     epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (current_step / epsilon_decay_steps))

#     if random.random() < epsilon:
#         # 试探：选择一个随机动作
#         action = env.action_space.sample()
#     else:
#         # 利用：根据Q网络选择最佳动作
#         with torch.no_grad(): # 这里不需要梯度计算
#             # 将状态转换为适当的张量格式
#             state_tensor = torch.FloatTensor(state).unsqueeze(0)
#             q_values = q_network(state_tensor)
#             # 选择Q值最高的动作
#             action = q_values.argmax().item()
#     return action
```

### 4. 训练循环

这里是所有组成部分协同之处。代理与环境交互，存储经验，从缓冲区采样，并更新Q网络。

**高层结构：**

```python
# --- 超参数 ---
# num_episodes = 1000
# replay_buffer_capacity = 10000
# batch_size = 64
# gamma = 0.99 # 折扣因子
# target_update_frequency = 100 # 每C步更新目标网络
# learning_rate = 0.001
# \epsilon 参数之前已定义...

# --- 初始化 ---
# env = gym.make('CartPole-v1')
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
# replay_buffer = ReplayBuffer(replay_buffer_capacity)
# q_network = QNetwork(state_dim, action_dim)
# target_network = QNetwork(state_dim, action_dim)
# target_network.load_state_dict(q_network.state_dict())
# target_network.eval()
# optimizer = # 初始化Q网络的优化器（例如Adam）
# loss_fn = nn.MSELoss() # 或Huber损失

# total_steps = 0
# episode_rewards = []

# --- 训练 ---
# for episode in range(num_episodes):
#     state, _ = env.reset()
#     episode_reward = 0
#     done = False

#     while not done:
#         # 1. 选择动作
#         action = select_action(state, q_network, total_steps)

#         # 2. 与环境交互
#         next_state, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated
#         episode_reward += reward

#         # 3. 存储转换
#         replay_buffer.store(state, action, reward, next_state, done)

#         # 更新当前状态
#         state = next_state
#         total_steps += 1

#         # 4. 采样并学习（如果缓冲区有足够样本）
#         if len(replay_buffer) > batch_size:
#             # 采样小批量
#             states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = replay_buffer.sample(batch_size)

#             # --- 将批次转换为张量 ---
#             # states_tensor = torch.FloatTensor(states_batch)
#             # actions_tensor = torch.LongTensor(actions_batch).unsqueeze(1) # gather需要形状为 (batch_size, 1)
#             # rewards_tensor = torch.FloatTensor(rewards_batch)
#             # next_states_tensor = torch.FloatTensor(next_states_batch)
#             # dones_tensor = torch.BoolTensor(dones_batch) # 使用BoolTensor进行遮罩

#             # --- 计算目标Q值 ---
#             with torch.no_grad(): # 目标计算不需要梯度
#                 # 从目标网络获取下一状态的Q值
#                 next_q_values_target = target_network(next_states_tensor)
#                 # 选择最佳动作的Q值（动作上的最大值）
#                 max_next_q_values = next_q_values_target.max(1)[0]
#                 # 终结状态的Q值置零
#                 max_next_q_values[dones_tensor] = 0.0
#                 # 计算目标Q值：R + gamma * max_a' Q_target(S', a')
#                 target_q_values = rewards_tensor + gamma * max_next_q_values

#             # --- 计算预测Q值 ---
#             # 从主Q网络获取当前状态的Q值
#             q_values_pred = q_network(states_tensor)
#             # 选择批次中实际采取动作对应的Q值
#             # 使用gather()根据actions_tensor索引选择Q值
#             predicted_q_values = q_values_pred.gather(1, actions_tensor).squeeze(1)

#             # --- 计算损失 ---
#             loss = loss_fn(predicted_q_values, target_q_values)

#             # --- 执行梯度下降 ---
#             optimizer.zero_grad()
#             loss.backward()
#             # 可选：裁剪梯度以防止梯度爆炸
#             # torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)
#             optimizer.step()

#         # 5. 定期更新目标网络
#         if total_steps % target_update_frequency == 0:
#             target_network.load_state_dict(q_network.state_dict())

#         if done:
#             break

#     episode_rewards.append(episode_reward)
#     print(f"Episode {episode + 1}: Total Reward = {episode_reward}, Epsilon = {epsilon:.3f}") # 提供反馈

# env.close()
```

### 5. 监控进度

监控代理在训练期间的性能很重要。绘制每回合累积的总奖励是可视化学习的标准方法。你应该会看到随时间上升的趋势，表明代理正在学习更好的策略。



![DQN训练进度 (CartPole)](plots/1664-0.json)



> DQN代理学习CartPole的奖励曲线。奖励通常会增加并最终趋于平稳，因为代理掌握了任务（通常达到最大回合长度，例如CartPole-v1的500步）。

### 总结

此次动手实践的步骤概述了一个基础DQN的主要组成部分和训练流程：

1. **环境交互：** 代理选择动作（$\epsilon$-贪婪），并接收状态、奖励和完成信号。
2. **经验回放：** 存储转换以打破相关性并复用过往经验。
3. **神经网络 (neural network)：** Q网络估计动作值，而缓慢更新的目标网络为学习提供稳定的目标。
4. **学习：** 从经验回放缓冲区采样小批量数据，以计算损失（Q网络预测的Q值与目标网络得到的目标Q值之间的差异），并通过梯度下降 (gradient descent)更新Q网络。

这构成了许多高级深度强化学习 (reinforcement learning)算法的基础。你可以在此结构上进行扩展，试验不同的网络架构、超参数 (parameter) (hyperparameter)调优，或考虑DQN的扩展，如Double DQN或Dueling DQN。

## 参考资料

- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin A. Riedmiller (2015)
  Journal: Nature; Volume: 518; Pages: 529–533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  介绍深度Q网络的开创性论文，展示了如何将深度学习与Q学习成功结合以解决复杂的控制任务。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: The MIT Press
  一本全面的教材，涵盖强化学习的理论基础，包括Q学习、策略迭代、价值迭代和贝尔曼方程，这些是理解DQN的先决条件。
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Core Team (2024)
  Publisher: PyTorch Foundation
  PyTorch深度学习框架的官方文档，对于实现实践部分中描述的Q网络、目标网络、优化器和损失函数至关重要。
- [Gymnasium Documentation](https://gymnasium.farama.org/) — Gymnasium Developers (2024)
  Gymnasium库的官方文档，提供了CartPole-v1环境以及实践示例中使用的智能体-环境交互接口。

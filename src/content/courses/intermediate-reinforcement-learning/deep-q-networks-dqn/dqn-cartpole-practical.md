---
course: "intermediate-reinforcement-learning"
chapter: "deep-q-networks-dqn"
lesson: "dqn-cartpole-practical"
sourceId: 5114
sourceUrl: "https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-2-deep-q-networks-dqn/dqn-cartpole-practical"
title: "动手实践：在CartPole上实现DQN"
description: "实现CartPole环境基本DQN智能体的分步指南。"
order: 7
plots: ["plots/5114-0.json"]
sourceHash: "c6da2029a0d0aa8c28ed79171ba93b7cf366c0acd61789d08c3a8f6353b70b62"
sourceCorrections: []
---

深度Q网络利用神经网络 (neural network)来估计Q值，从而克服了表格方法在大状态空间下的局限。我们将使用Gymnasium库和PyTorch实现一个DQN智能体，来解决经典的CartPole控制问题。

CartPole环境是强化学习 (reinforcement learning)中的一个标准基准。目标是使移动小车上的杆子保持直立平衡。

- **状态：** 状态由一个4维向量 (vector)表示：小车位置、小车速度、杆子角度、杆子角速度。
- **动作：** 有两种离散动作：向左推动小车(0)或向右推动小车(1)。
- **奖励：** 每当杆子保持直立，每一步都会得到+1的奖励。
- **终止：** 一个回合在以下情况下结束：杆子角度超出±12度，小车从中心移动超出±2.4单位，或回合长度达到预设限制（在较新的Gymnasium版本中通常是500步）。

我们的目标是训练一个DQN智能体，使其学习一种策略以最大化总奖励，从而尽可能长时间地保持杆子平衡。

### 前提条件

在我们开始之前，请确保你已经安装了必要的库：

```bash
pip install gymnasium torch numpy matplotlib
```

我们将使用`gymnasium`来构建环境，`torch`来构建和训练神经网络 (neural network)，`numpy`进行数值运算，以及`matplotlib`（或类似的库）用于后续的结果可视化。

### 构建DQN组件

我们的DQN实现需要我们之前讨论过的几个重要部分：Q网络、目标网络（Q网络的副本）和经验回放缓冲区。

#### 1. Q网络

我们需要一个神经网络 (neural network)，它以状态表示作为输入，并输出每个可能动作的估计Q值。考虑到CartPole的状态空间相对较小（4维）且动作是离散的（2个动作），一个简单的多层感知机（MLP）就足够了。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class QNetwork(nn.Module):
    """智能体（策略）模型。"""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """初始化参数并构建模型。
        参数
        ======
            state_size (int): 每个状态的维度
            action_size (int): 每个动作的维度
            seed (int): 随机种子
            fc1_units (int): 第一个隐藏层中的节点数
            fc2_units (int): 第二个隐藏层中的节点数
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """构建一个将状态映射到动作值的网络。"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

该网络有一个与状态大小匹配的输入层（4），两个隐藏层，每个有64个单元，并使用ReLU激活函数 (activation function)，以及一个输出层，其单元数等于动作大小（2），提供向左和向右推动的Q值。

#### 2. 经验回放缓冲区

为了存储转换并从中采样进行学习，我们实现了一个回放缓冲区。`collections.deque`常用于高效地添加和删除元素。

```python
import random
import torch
import numpy as np
from collections import deque, namedtuple

class ReplayBuffer:
    """固定大小的缓冲区，用于存储经验元组。"""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """初始化一个ReplayBuffer对象。
        参数
        ======
            action_size (int): 每个动作的维度
            buffer_size (int): 缓冲区的最大大小
            batch_size (int): 每个训练批次的大小
            seed (int): 随机种子
            device (string): 'cpu' 或 'cuda'
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """向内存中添加一个新的经验。"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """从内存中随机采样一批经验。"""
        experiences = random.sample(self.memory, k=self.batch_size)

        # 将经验批次转换为指定设备上的张量
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """返回内部内存的当前大小。"""
        return len(self.memory)
```

该缓冲区存储`Experience`元组，并提供`add`新经验和`sample`随机批次进行训练的方法。

### DQN智能体

现在我们将这些组件组合成一个`Agent`类。这个类将管理Q网络、目标网络、回放缓冲区和学习过程。

```python
import numpy as np
import random
from collections import namedtuple, deque

# 定义超参数（示例值）
BUFFER_SIZE = int(1e5)  # 回放缓冲区大小
BATCH_SIZE = 64         # 小批次大小
GAMMA = 0.99            # 折扣因子
TAU = 1e-3              # 用于目标参数的软更新
LR = 5e-4               # 学习率
UPDATE_EVERY = 4        # 更新网络的频率

# 检查GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """与环境交互并从中学习的智能体。"""

    def __init__(self, state_size, action_size, seed):
        """初始化一个Agent对象。

        参数
        ======
            state_size (int): 每个状态的维度
            action_size (int): 每个动作的维度
            seed (int): 随机种子
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q网络
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # 回放内存
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device)
        # 初始化时间步（用于每UPDATE_EVERY步更新）
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # 将经验保存到回放内存中
        self.memory.add(state, action, reward, next_state, done)

        # 每UPDATE_EVERY步学习。
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # 如果内存中有足够的样本，获取随机子集并学习
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """根据当前策略，返回给定状态下的动作。

        参数
        ======
            state (array_like): 当前状态
            eps (float): epsilon，用于epsilon-greedy动作选择
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval() # 将网络设置为评估模式
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train() # 将网络设置回训练模式

        # Epsilon-greedy动作选择
        if random.random() > eps:
            # 选择最佳动作（利用）
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # 选择一个随机动作（试探）
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """使用给定批次的经验元组更新值参数。

        参数
        ======
            experiences (Tuple[torch.Tensor]): (s, a, r, s', done) 元组的集合
            gamma (float): 折扣因子
        """
        states, actions, rewards, next_states, dones = experiences

        # 从目标模型中获取（下一状态的）最大预测Q值
        # 我们将输出Q_目标_next从计算图中分离 -> 不为目标网络计算梯度
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # 计算当前状态的Q目标
        # 目标 = 奖励 + Gamma * Q_目标(下一状态, 最大动作) * (1 - done)
        # 如果done为1，则未来奖励为0
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # 从本地模型中获取期望Q值
        # 我们需要实际执行动作的Q值
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # 计算损失（均方误差或Huber损失）
        loss = F.mse_loss(Q_expected, Q_targets)
        # 最小化损失
        self.optimizer.zero_grad() # 清除之前的梯度
        loss.backward()           # 计算梯度
        self.optimizer.step()      # 更新权重

        # ------------------- 更新目标网络 ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """软更新模型参数。
        θ_target = τ*θ_local + (1 - τ)*θ_target

        参数
        ======
            local_model (PyTorch模型): 权重将从中复制
            target_model (PyTorch模型): 权重将复制到
            tau (float): 插值参数
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
```

`Agent`的重要方面：

- **初始化 (`__init__`)：** 创建本地Q网络和目标Q网络（初始时相同），为本地网络设置Adam优化器，并初始化回放缓冲区。
- **步进 (`step`)：** 在每个时间步被调用。它将经验`(s, a, r, s', done)`存储到缓冲区中，并在缓冲区足够满的情况下，每`UPDATE_EVERY`步触发学习过程。
- **行动 (`act`)：** 实现epsilon-greedy策略。以`epsilon`的概率，它选择一个随机动作（试探）；否则，它查询本地Q网络以选择估计Q值最高的动作（利用）。请注意，在动作选择期间使用`eval()`模式是为了禁用dropout或批量归一化 (normalization)更新。
- **学习 (`learn`)：** 这是核心训练逻辑：
  1. 采样一批经验。
  2. 使用**目标网络**和贝尔曼方程计算目标Q值：$目标 = r + \gamma \max_{a'} Q_{目标}(s', a')$ `(1 - dones)`项确保了终止状态的未来价值为零。使用`detach()`是为了防止在计算过程中梯度流入目标网络的参数 (parameter)。
  3. 使用**本地网络**计算批次中*实际执行*动作的预测Q值：$预测值 = Q_{本地}(s, a)$ `.gather(1, actions)`部分选取了批次中存储的特定动作对应的Q值。
  4. 计算`Q_targets`和`Q_expected`之间的损失（MSE损失）。
  5. 执行反向传播 (backpropagation)并使用优化器更新**本地网络**的权重 (weight)。
  6. 调用`soft_update`，将本地网络的权重缓慢地融入目标网络。
- **软更新 (`soft_update`)：** 使用参数`TAU`，逐步将目标网络的权重向本地网络的权重更新。这比不频繁地直接复制权重提供了更高的稳定性。

### 训练循环

最后，我们需要主脚本来初始化环境和智能体，并运行训练回合。

```python
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt

# 初始化环境和智能体
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = Agent(state_size=state_size, action_size=action_size, seed=0)

def train_dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """深度Q学习训练循环。

    参数
    ======
        n_episodes (int): 最大训练回合数
        max_t (int): 每个回合的最大时间步数
        eps_start (float): epsilon的起始值，用于epsilon-greedy动作选择
        eps_end (float): epsilon的最小值
        eps_decay (float): 减小epsilon的乘法因子（每个回合）
    """
    scores = []                        # 包含每个回合得分的列表
    scores_window = deque(maxlen=100)  # 最近100个得分
    eps = eps_start                    # 初始化epsilon
    solved_threshold = 195.0           # 平均得分阈值，用于判断环境是否解决（针对CartPole-v1）

    for i_episode in range(1, n_episodes+1):
        state, info = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated # 如果终止或截断，回合结束
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # 保存最近的得分
        scores.append(score)              # 保存最近的得分
        eps = max(eps_end, eps_decay*eps) # 减小epsilon

        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
        if np.mean(scores_window)>= solved_threshold:
            print(f'\nEnvironment solved in {i_episode-100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            # 保存训练好的模型权重
            torch.save(agent.qnetwork_local.state_dict(), 'dqn_cartpole_weights.pth')
            break
    return scores

# 开始训练
scores = train_dqn()

# 绘制得分图
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('得分')
plt.xlabel('回合数')
plt.title('DQN在CartPole-v1上的训练表现')
plt.grid(True)
plt.show()
```

这个循环运行预设的回合数（`n_episodes`）。在每个回合中：

1. 环境被重置。
2. 智能体逐步与环境交互（`act`，`env.step`）。
3. 每个转换都由智能体处理（`agent.step`），可能触发学习更新。
4. 得分（总奖励）被记录。
5. 每个回合结束后，epsilon会衰减以随时间减少试探。
6. 打印进度（最近100个回合的平均得分）。
7. 如果平均得分达到目标阈值（例如，CartPole-v1的195），训练停止，并且学习到的权重 (weight)被保存。

训练结束后，一个图表显示了每个回合的得分，理想情况下，这应该显示出随着智能体学习而呈上升趋势。



![CartPole上的DQN训练进度示例](plots/5114-0.json)



> 学习曲线示例，显示回合得分随时间增加并最终超过“已解决”阈值。

### 总结

在这个动手实践部分，我们从头开始实现了一个功能完备的深度Q网络智能体。我们定义了网络架构、经验回放机制，以及包含了epsilon-greedy动作选择、从采样批次中学习和目标网络更新的智能体逻辑。通过将其应用于CartPole环境，我们说明了DQN如何在表格方法不切实际的环境中学习有效策略。这为解决更复杂的问题以及理解下一章中讨论的改进提供了坚实的起点。

## 参考资料

- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin A. Riedmiller (2015)
  Journal: Nature; Volume: 518; Pages: 529-533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  介绍了深度Q网络（DQN）算法，该算法将深度神经网络与Q学习相结合，并引入了经验回放和目标网络。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  提供了强化学习概念的详细介绍，包括动态规划、蒙特卡洛方法和时序差分学习（如Q学习）。
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Foundation (2024)
  Publisher: PyTorch Foundation
  提供使用 PyTorch 构建、训练和部署神经网络的指南和API参考。
- [Gymnasium Documentation](https://gymnasium.farama.org/) — Gymnasium Developers (2023)
  Gymnasium 的官方文档，详细说明了如何设置和与CartPole等强化学习环境进行交互。

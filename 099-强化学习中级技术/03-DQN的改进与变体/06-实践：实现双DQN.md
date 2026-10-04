# 实践：实现双DQN

来源：[原文](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-3-dqn-improvements-variants/double-dqn-practice)

[返回章节目录](README.md) · [返回课程目录](../README.md)

双DQN（DDQN）能够缓解标准Q学习和DQN中固有的过高估计偏差。从标准DQN实现过渡到DDQN，只需要对损失函数 (loss function)中目标Q值的计算方式进行一个虽小但能带来显著改变的修改。

回顾一下标准DQN对状态转移 $(s, a, r, s', d)$ 的目标计算（其中 $d$ 表示 $s'$ 是否为终止状态）：

从经验回放缓冲区中第 $i$ 个样本的目标 $y_i$ 为：


$$
y_i = r_i + \gamma \max_{a'} Q_{target}(s'_i, a') \quad \text{(如果 } s'_i \text{ 不是终止状态)}
$$


$$
y_i = r_i \quad \text{(如果 } s'_i \text{ 是终止状态)}
$$


在这里，目标网络 $Q_{target}$ 既用于选择最佳的下一步动作（$a'$），也用于评估该动作的价值。这种耦合关系可能导致过高估计。

### 双DQN的修改

双DQN将此过程解耦。它使用**在线网络**（$Q_{online}$）在下一个状态 $s'_i$ 中选择*最佳动作*，然后使用**目标网络**（$Q_{target}$）来评估该所选动作的*价值*。

双DQN目标 $y_i$ 的计算变为：


$$
a'_{max} = \arg\max_{a'} Q_{online}(s'_i, a')
$$


$$
y_i = r_i + \gamma Q_{target}(s'_i, a'_{max}) \quad \text{(如果 } s'_i \text{ 不是终止状态)}
$$


$$
y_i = r_i \quad \text{(如果 } s'_i \text{ 是终止状态)}
$$


请注意这个变化：我们首先根据*在线*网络在状态 $s'_i$ 中找到使Q值最大化的动作 $a'_{max}$。然后，我们将这个特定的动作 $a'_{max}$ 代入*目标*网络，以获得目标计算所需的Q值估计。

### 实现修改

假设您有一个标准的DQN实现，可能包含一个处理从回放缓冲区采样的批次经验的 `learn` 或 `compute_loss` 方法。您需要修改计算目标Q值的部分。

以下是目标计算代码片段的对比分析（假设 `online_net` 和 `target_net` 是您的网络模型，并且 `next_states`、`rewards`、`dones` 是从采样批次中获取的张量/数组）：

**标准DQN目标计算（代码片段）：**

```python
# 假设 next_states 是从经验回放缓冲区中获取的下一个状态的批次
# 从目标网络获取下一个状态的Q值
next_q_values_target = target_net(next_states)

# 为每个下一个状态选择最大的Q值
max_next_q_values = next_q_values_target.max(dim=1)[0] # 或在NumPy/TF中使用 axis=1

# 计算目标 y_i（处理 dones=True 的终止状态）
target_q_values = rewards + gamma * max_next_q_values * (1 - dones)
```

**双DQN目标计算（代码片段）：**

```python
# 假设 next_states, rewards, dones 是从经验回放缓冲区中获取的批次数据
# 1. 使用*在线*网络在下一个状态中选择最佳动作
next_q_values_online = online_net(next_states)
best_next_actions = next_q_values_online.argmax(dim=1) # 或 axis=1

# 2. 使用*目标*网络评估这些选定的动作
# 从目标网络获取下一个状态的所有Q值
next_q_values_target = target_net(next_states)
# 选择与 best_next_actions 对应的Q值
# 需要仔细的索引（例如，PyTorch/TF 中的 gather）
q_values_of_best_actions = next_q_values_target.gather(1, best_next_actions.unsqueeze(-1)).squeeze(-1)

# 3. 计算目标 y_i（处理 dones=True 的终止状态）
target_q_values = rewards + gamma * q_values_of_best_actions * (1 - dones)
```

核心变化涉及以下步骤：

1. 使用 `online_net` 对 `next_states` 执行前向传播，以找到 `argmax` 动作（$a'_{max}$）。
2. 使用 `target_net` 对 `next_states` 执行前向传播，以获取所有可能下一步动作的Q值。
3. 从步骤2中选择与步骤1中选定动作对应的Q值。这需要仔细的张量索引（例如，PyTorch中的 `gather` 或 TensorFlow中的 `tf.gather_nd`）。
4. 使用这些选定的Q值来计算最终目标 $y_i$。

您的DQN代码其余部分，包括经验回放机制、从在线网络权重 (weight)定期更新目标网络权重、优化器步骤以及环境交互过程中的epsilon-贪婪动作选择，通常保持不变。

> 双DQN中目标Q值组件（$\gamma Q_{target}(s', \arg\max_{a'} Q_{online}(s', a'))$）的计算示意图。在线网络选择最佳动作，目标网络评估该特定动作的价值。

根据上述代码片段和图表修改您现有的DQN智能体代码。测试您的实现，例如再次在CartPole环境或更复杂的Atari环境（如果您正在使用这些）中进行测试。观察训练是否显得更稳定，或者与您的标准DQN实现相比，智能体是否获得更好的性能，请记住结果可能因超参数 (parameter) (hyperparameter)和环境细节而异。这种亲自动手的修改提供了改进DQN算法的直接经验。

## 参考资料

- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — Hado van Hasselt, Arthur Guez, and David Silver (2016)
  Journal: Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence (AAAI-16); DOI: [10.48550/arXiv.1509.06461](https://doi.org/10.48550/arXiv.1509.06461)
  介绍Double DQN的原创论文，通过解耦动作选择和评估，有效缓解了标准DQN中的过高估计偏差。
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Gimel, Andriy Mahendran, Daan Wierstra, and John J. J. Wotters (2015)
  Journal: Nature; Publisher: Springer Nature; Volume: 518; Pages: 529-533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  深度Q网络（DQN）的开创性论文，Double DQN在此基础上进行改进，以提升性能和稳定性。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本提供强化学习广阔理论基础的教材，内容涵盖Q学习、DQN以及包括Double DQN在内的各种改进。(第2版)

---

[上一节](05-%E4%BC%98%E5%85%88%E7%BB%8F%E9%AA%8C%E5%9B%9E%E6%94%BE%20%28%E7%AE%80%E8%A6%81%E6%A6%82%E8%BF%B0%29.md) · [下一节](../04-%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E6%96%B9%E6%B3%95/01-%E5%9F%BA%E4%BA%8E%E4%BB%B7%E5%80%BC%E6%96%B9%E6%B3%95%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)

---
course: "advanced-reinforcement-learning"
chapter: "advanced-exploration-strategies"
lesson: "parameter-space-noise"
sourceId: 3416
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-4-advanced-exploration-strategies/parameter-space-noise"
title: "参数空间噪声用于行为多样性"
description: "直接向策略参数注入噪声以促进行为多样性，与行动空间噪声形成对比。"
order: 4
plots: []
sourceHash: "c7c4058cc75d2fdde1a1d487c40dd1aebe9d7600e13b3edbc995df8d1723b519"
sourceCorrections: []
---

虽然直接在智能体选择的行动（行动空间噪声）中加入随机扰动是一种常用的行为多样性方法，尤其是在连续控制领域，但有时这可能导致行为不稳定且试探效率不高。这是因为在每个时间步施加的噪声通常是独立的，使得智能体的行为快速波动而缺乏连贯性。考虑另一种方案：如果我们不是对最终行动引入噪声，而是对决策过程本身，具体来说是对策略网络的参数 (parameter)引入噪声，会怎么样？这就是参数空间噪声的中心思想。

参数空间噪声不是从策略输出分布的含噪声版本中采样行动，也不是向确定性行动中添加噪声，而是直接扰动策略网络的权重 (weight)。设策略由函数 $\pi_\theta$ 表示，其参数为权重 $\theta$。在使用参数空间噪声时，我们采样一个噪声向量 (vector) $\epsilon$（通常来自高斯分布 $\mathcal{N}(0, \sigma^2 I)$），并在一定时间内（通常是整个回合）修改策略参数。然后，智能体根据这个受扰动的策略 $\pi_{\theta+\epsilon}$ 行动。


$$
\text{行动噪声:} \quad a_t = \pi_\theta(s_t) + \text{噪声}
$$


$$
\text{参数噪声:} \quad a_t = \pi_{\theta+\epsilon}(s_t), \quad \text{其中 } \epsilon \sim \mathcal{N}(0, \sigma^2 I) \text{ 周期性采样}
$$


主要区别在于行为多样性的*时间一致性*。由于参数 $\theta+\epsilon$ 在一个回合（或多个时间步）的持续时间内保持不变，因此由此产生的行为，尽管是试探性的，但与在每一步添加独立噪声相比，更具一致性和结构性。可以将其理解为在每个回合中，以一种略有不同但暂时固定的“个性”来寻找新方法。

> 行动空间噪声与参数空间噪声的噪声注入点对比。参数噪声直接修改策略网络的权重（$\theta$），导致时间上相关的行为调整。

### 参数 (parameter)空间噪声的优点

1. **结构化的试探：** 通过扰动参数，策略本身会发生偏移，从而在一个回合内使行动在时间上产生更协调的变化。这比独立的行动噪声更能有效地找到需要一系列相关行动的复杂行为。
2. **对确定性策略的有效性：** 像深度确定性策略梯度（DDPG）这样的算法会输出确定性行动。简单地向此行动中添加噪声可能不是最有效的试探方式，特别是如果策略梯度对此类噪声敏感的话。扰动参数可以提供一种更平滑的方式来带来行为变化。
3. **依赖于状态的试探：** 由于参数扰动的*影响*通过函数 $\pi_{\theta+\epsilon}(s_t)$ 取决于当前状态 ($s_t$)，因此由此产生的试探行为是隐式地依赖于状态的，与简单的独立于状态的行动噪声不同。

### 实现注意事项

实现参数 (parameter)空间噪声需要仔细考量：

- **噪声采样：** 通常使用高斯噪声。噪声向量 (vector) $\epsilon$ 与策略参数 $\theta$ 具有相同的维度。
- **应用方式：** 噪声 $\epsilon$ 通常在每个回合开始时采样，并添加到当前策略参数 $\theta$ 中。智能体在整个回合中使用这些受扰动的参数 $\theta' = \theta + \epsilon$。原始参数 $\theta$ 仍用于学习更新（例如，计算梯度）。
- **层目标：** 噪声可能应用于策略网络的所有层或仅特定层（例如，仅最终层）。实验通常决定最佳方式。
- **噪声尺度（$\sigma$）：** 噪声的大小是一个重要的超参数 (hyperparameter)。如果太小，试探不足；如果太大，受扰动的策略可能会表现得不合理。
- **自适应噪声尺度：** 为了自动化调整，噪声尺度 $\sigma$ 可以在训练期间进行调整。一种常见方法（由 Plappert 等人于 2017 年提出）是根据原始策略 $\pi_\theta$ 和受扰动策略 $\pi_{\theta+\epsilon}$ 在一批状态上产生的行动之间的“距离”来调整 $\sigma$。如果距离太大，$\sigma$ 减小；如果太小，则增大。这有助于保持相对于当前策略尺度的稳定行为多样性程度。

```python
# 参数噪声在回合循环中的Python代码片段

def train_agent_with_param_noise(agent, env, num_episodes, noise_scale):
    policy_params = agent.policy.get_weights() # 获取原始权重

    for episode in range(num_episodes):
        # 为当前回合采样参数噪声
        param_noise_vector = np.random.normal(0, noise_scale, size=policy_params.shape)
        
        # 创建用于试探的扰动策略
        perturbed_policy_params = policy_params + param_noise_vector
        agent.exploration_policy.set_weights(perturbed_policy_params) # 使用独立的策略实例进行试探

        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 使用扰动后的策略行动
            action = agent.exploration_policy.predict(state) 
            
            next_state, reward, done, _ = env.step(action)
            
            # 使用实际采取的行动存储经验
            agent.memory.store(state, action, reward, next_state, done) 
            
            # 使用原始（未扰动）的策略参数进行学习
            if agent.memory.is_ready():
                agent.learn() # 使用原始策略参数进行更新

            state = next_state
            episode_reward += reward

        # 根据学习步骤更新原始策略参数
        policy_params = agent.policy.get_weights() 

        # 可选：根据行动距离度量自适应调整 noise_scale
        # noise_scale = adapt_noise_scale(...)

        print(f"Episode: {episode}, Reward: {episode_reward}")
```

### 挑战

虽然有效，但参数 (parameter)空间噪声也带来了自己的一系列挑战：

- **超参数 (hyperparameter)敏感性：** 初始噪声尺度和任何自适应参数都需要仔细调整。
- **计算开销：** 生成噪声向量 (vector)并可能调整尺度会增加一些计算成本，相比简单的行动噪声。
- **与学习的关联：** 用于行为多样性的参数噪声与学习的梯度更新之间的关联需要正确管理。使用扰动后的策略进行行动，但使用原始策略进行更新是标准做法。

参数空间噪声提供了一种精密的促进行为多样性的方法，特别适合于时间上一致的试探行为有帮助的情况，例如通过确定性策略梯度方法解决的连续控制任务。它代表了在构建高效试探智能体的工具集中，对行动空间噪声和内在激励手段的一种有价值的替代或补充。

## 参考资料

- [Parameter Space Noise for Exploration](https://arxiv.org/abs/1706.01905) — Matthias Plappert, Rein Houthooft, Prafulla Dhariwal, Szymon Sidor, Richard Y. Chen, Xi Chen, Tamim Asfour, Pieter Abbeel, Marcin Andrychowicz (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1706.01905](https://doi.org/10.48550/arXiv.1706.01905)
  介绍参数空间噪声作为探索策略的开创性论文，并详细阐述了一种自适应噪声缩放机制。
- [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971) — Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1509.02971](https://doi.org/10.48550/arXiv.1509.02971)
  介绍了深度确定性策略梯度 (DDPG) 算法，这是参数空间噪声在探索方面提供显著优势的常见应用场景。
- [Exploration in Deep Reinforcement Learning: A Survey](https://link.springer.com/article/10.1007/s00521-022-07612-2) — Yijun Li, Yihan Ding, Junge Zhang, Jianzhong Ding, Shuzhen Li, and Mengyuan Lin (2022)
  Journal: Neural Computing and Applications; Publisher: Springer; Volume: 35; Pages: 605; DOI: [10.1007/s00521-022-07612-2](https://doi.org/10.1007/s00521-022-07612-2)
  全面回顾了深度强化学习中的各种探索策略，包括参数空间噪声，提供了更广泛的背景信息。
- [Deep Reinforcement Learning Hands-On: Apply advanced RL to games, robots, and other complex environments](https://www.packtpub.com/product/deep-reinforcement-learning-hands-on-second-edition/9781838826994) — Maxim Lapan (2020)
  Publisher: Packt Publishing
  一本实现深度强化学习算法的实用指南，详细介绍了各种探索技术。第二版。

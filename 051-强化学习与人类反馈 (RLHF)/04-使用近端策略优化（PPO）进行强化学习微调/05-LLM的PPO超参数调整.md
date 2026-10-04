# LLM的PPO超参数调整

来源：[原文](https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-4-rl-ppo-fine-tuning/ppo-hyperparameter-tuning-llms)

[返回章节目录](README.md) · [返回课程目录](../README.md)

使用PPO对大型语言模型进行微调 (fine-tuning)，引入了一系列超参数 (parameter) (hyperparameter)，这些参数对于平衡优化效果和训练稳定性非常重要。与标准强化学习 (reinforcement learning)任务不同，大型语言模型是高度复杂、非平稳的环境，其中策略（模型本身）生成的是复杂、高维度的序列（文本）。超参数的微小改变可能导致模型行为、训练时间和对齐 (alignment)质量的显著差异。因此，细致的调整非常必要。

本节提供关于设定和调整RLHF背景下PPO重要超参数的指导，特别关注大型语言模型的特定因素。

### 学习率

学习率决定了策略（actor）和价值（critic）网络在梯度下降 (gradient descent)期间的步长。考虑到大型语言模型的规模以及更新数十亿参数 (parameter)时可能出现的不稳定性，通常偏好使用相对较小的学习率。

- **策略学习率：** 控制大型语言模型的生成策略根据奖励信号和PPO目标进行调整的速度。过高的学习率可能导致策略与初始SFT模型急剧偏离，从而导致高KL散度、无意义的输出和训练崩溃。过低的学习率会导致学习缓慢。典型值通常介于 $1 \times 10^{-6}$ 到 $5 \times 10^{-5}$ 之间。
- **价值学习率：** 控制价值网络学习预测预期回报（奖励）的速度。它通常可以略高于策略学习率，因为价值预测是一个监督回归问题。然而，价值函数的不稳定性可能对优势估计和策略更新产生负面影响。典型值可能介于 $1 \times 10^{-5}$ 到 $1 \times 10^{-4}$ 之间。

通常的做法是使用Adam或AdamW等优化器，配合学习率调度器（例如，线性衰减、余弦衰减）在训练过程中调整学习率。从保守的（低）学习率开始，如果训练表现稳定且进展缓慢，则逐渐提高。

### 批次大小

在RLHF的PPO中，有两种相关的批次大小：

- **采样批次大小：** 并行处理的提示数量，用于生成响应并收集经验（提示、响应、奖励、对数概率）。这通常受限于可用的GPU内存，因为它需要使用策略模型运行推理 (inference)。更大的采样批次在每次迭代中提供更多样化的数据，但会增加内存需求。典型值很大程度上取决于硬件，范围从64到1024或更多。
- **PPO小批次大小：** 在PPO更新轮次期间用于计算梯度的数据块大小。这从较大的采样批次中抽样得到。小批次大小影响梯度估计的方差。较小的小批次会引入更多噪声，但有时有助于跳出局部最优。较大的小批次提供更稳定的梯度，但每次更新步骤需要更多内存。典型值范围从4到64，受限于用于梯度计算的GPU内存。

每次PPO更新使用的总经验量是 `rollout_batch_size`。这些经验会迭代 `ppo_epochs` 次，在每个梯度步骤中处理 `mini_batch_size` 个样本。

### PPO轮次

此超参数 (parameter) (hyperparameter)定义了PPO算法在收集到的采样数据（存储在缓冲区中的经验）上迭代多少次，以更新策略和价值网络。

- **更多轮次：** 允许模型从每批收集到的经验中学习更多，可能提高样本效率。然而，过多的轮次可能导致对当前批次数据过拟合 (overfitting)，并可能导致策略与生成数据的策略偏离过远，违反PPO的假设并引发不稳定。
- **更少轮次：** 更稳定，因为策略更新在数据收集阶段之间较小。然而，这可能需要更多的采样（更多数据收集）才能达到相同的学习水平。

常见值范围从2到10轮。最佳数量通常取决于学习率和小批次大小等其他参数。

### KL散度系数 ($\beta$)

如前所述，KL散度项 $D_{KL}(\pi_{\theta} || \pi_{\text{SFT}})$ 会惩罚策略 $\pi_{\theta}$ 偏离参考策略 $\pi_{\text{SFT}}$（通常是初始SFT模型）过远的情况。系数 $\beta$ 控制这种惩罚的强度。

- **低 $\beta$：** 允许策略更积极地优化奖励信号，但增加了与SFT模型能力显著偏离的风险，可能导致不连贯的文本或通用语言理解的丧失。
- **高 $\beta$：** 使策略保持接近SFT模型，确保稳定并保留语言流畅性，但可能限制模型对奖励信号的优化程度，可能导致对齐 (alignment)效果不佳的行为。

选择 $\beta$ 是一个权衡。一些实现使用*自适应KL控制器*，它在训练期间动态调整 $\beta$，以使KL散度保持接近预定义的目标值（例如，6 nats）。这可以提供比固定 $\beta$ 更高的稳定性。如果使用固定系数，典型的起始值可能在0.01到0.2之间。在训练期间监控实际的KL散度——如果它持续超过期望的阈值（例如，10-15 nats）或归零，则相应调整 $\beta$。



[交互图表：KL系数 (Beta) 对KL散度的影响](https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-4-rl-ppo-fine-tuning/ppo-hyperparameter-tuning-llms#plot-1x39s30)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "name": "Beta=0.01",
      "type": "scatter",
      "mode": "lines",
      "x": [
        0,
        100,
        200,
        300,
        400,
        500
      ],
      "y": [
        0,
        8,
        14,
        19,
        23,
        26
      ],
      "line": {
        "color": "#ff8787"
      }
    },
    {
      "name": "Beta=0.1",
      "type": "scatter",
      "mode": "lines",
      "x": [
        0,
        100,
        200,
        300,
        400,
        500
      ],
      "y": [
        0,
        5,
        7,
        8,
        9,
        10
      ],
      "line": {
        "color": "#748ffc"
      }
    },
    {
      "name": "Beta=0.5",
      "type": "scatter",
      "mode": "lines",
      "x": [
        0,
        100,
        200,
        300,
        400,
        500
      ],
      "y": [
        0,
        1,
        1.5,
        1.8,
        2,
        2.1
      ],
      "line": {
        "color": "#8ce99a"
      }
    },
    {
      "name": "自适应 (目标=6)",
      "type": "scatter",
      "mode": "lines",
      "x": [
        0,
        100,
        200,
        300,
        400,
        500
      ],
      "y": [
        0,
        4,
        7,
        5.5,
        6.5,
        6
      ],
      "line": {
        "color": "#ffc078",
        "dash": "dash"
      }
    }
  ],
  "layout": {
    "title": "KL系数 (Beta) 对KL散度的影响",
    "xaxis": {
      "title": "训练步数"
    },
    "yaxis": {
      "title": "KL散度 (nats)"
    },
    "legend": {
      "title": "KL系数"
    },
    "template": "plotly_white"
  }
}
```

</details>



> PPO训练中不同 $\beta$ 设置下的KL散度趋势。较低的固定 $\beta$ 值允许KL散度增长更大，而较高的值会限制它。自适应控制器会调整 $\beta$ 以使KL散度保持在目标值附近。

### PPO裁剪参数 (parameter) ($\epsilon$)

PPO使用裁剪的代理目标函数限制策略更新的大小。裁剪参数 $\epsilon$ 定义了概率比 $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ 在不被裁剪的情况下允许操作的范围 $[1-\epsilon, 1+\epsilon]$。

- **较小的 $\epsilon$（例如，0.1）：** 导致更小、更保守的策略更新，有助于稳定但可能减慢收敛速度。
- **较大的 $\epsilon$（例如，0.3）：** 允许更大的策略更新，可能加快收敛速度，但如果优势估计存在噪声或学习率过高，则会增加不稳定的风险。

PPO中 $\epsilon$ 的典型值通常介于0.1和0.3之间。一个常见的起点是 $\epsilon = 0.2$。它的效果与学习率和PPO轮次数量相互关联。

### 广义优势估计 (GAE) Lambda ($\lambda$)

GAE用于估计优势函数 $A(s_t, a_t)$，平衡偏差和方差。参数 (parameter) $\lambda$ 控制这种权衡。

- **$\lambda = 1$：** 对应于回报的高方差蒙特卡洛估计。
- **$\lambda = 0$：** 对应于较低方差但可能较高偏差的时序差分（TD）估计（仅使用即时奖励和下一状态的值）。
- **$0 < \lambda < 1$：** 在这两个极端之间进行插值。

在实践中，PPO实现（包括RLHF）中常使用接近1的值，例如 $\lambda = 0.95$。这种选择通常为策略梯度更新提供了良好的平衡。

### 价值函数系数 ($c_1$)

总体的PPO损失函数 (loss function)结合了裁剪的代理目标、价值函数损失，有时还包括熵奖励。价值函数系数（$c_1$，常记作 `vf_coef`）用于缩放预测值 $V_{\phi}(s_t)$ 与目标值（通常通过GAE计算）之间的均方误差损失。


$$
L^{CLIP+VF+S}(\theta, \phi) = \mathbb{E}_t \left[ L^{CLIP}_t(\theta) - c_1 L^{VF}_t(\phi) + c_2 S[\pi_\theta](s_t) \right]
$$


其中 $L^{VF}_t(\phi) = (V_{\phi}(s_t) - V_t^{\text{targ}})^2$。

$c_1$ 的典型值为0.5或1.0。这有助于价值函数与策略一同得到有效训练，因为准确的价值估计对于良好的优势估计很重要。

### 调整策略

为大型语言模型调整PPO通常是一个经验性过程：

1. **从默认值开始：** 从成功的RLHF研究中报告的超参数 (parameter) (hyperparameter)或TRL等库提供的默认值（例如，`PPOConfig`）开始。
2. **优先控制KL：** 确保KL散度保持在合理范围内（例如，< 15-20 nats）。如果策略偏离过快，首先调整 $\beta$ 或自适应KL目标。
3. **监控指标：** 在训练期间跟踪奖励均值/分布、KL散度、策略损失、价值损失和模型熵。定期使用评估提示来评估生成质量。
4. **调整学习率和批次大小：** 如果训练稳定但缓慢，考虑略微增加学习率或调整批次大小（如果内存允许）。如果不稳定，则降低学习率。
5. **调整PPO轮次和裁剪：** 如果在使用多个轮次时出现不稳定，尝试减少轮次数量或收紧裁剪范围（$\epsilon$）。
6. **迭代：** 进行小的、渐进的更改，并观察它们在几个训练步骤中的效果。超参数之间的相互作用复杂，因此隔离单个更改的效果可能很困难，但通常是必要的。

由于训练大型语言模型的计算成本很高，广泛的网格搜索通常不可行。依赖已确定的范围，密切监控训练动态，并根据观察到的行为和评估结果进行明智的调整。

## 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍PPO算法的基础论文。
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  展示PPO如何用于通过人类反馈（RLHF）使大型语言模型与人类偏好对齐的论文。
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel (2015)
  Journal: arXiv preprint arXiv:1506.02438; DOI: [10.48550/arXiv.1506.02438](https://doi.org/10.48550/arXiv.1506.02438)
  提出广义优势估计（GAE），用于稳定的优势函数估计。
- [Fine-tune a LLaMA model with 🤗PEFT & 🤗TRL](https://huggingface.co/blog/trl-peft) — Edward Beeching, Younes Belkada, Leandro von Werra, Sourab Mangrulkar, Lewis Tunstall, Kashif Rasul (2023)
  Publisher: Hugging Face Blog
  使用TRL进行大型语言模型微调的PPO实用指南，包含超参数细节。

---

[上一节](04-%E4%BC%98%E5%8A%BF%E5%92%8C%E5%9B%9E%E6%8A%A5%E7%9A%84%E8%AE%A1%E7%AE%97.md) · [下一节](06-%E5%B8%B8%E7%94%A8%20PPO%20%E5%AE%9E%E7%8E%B0%E5%BA%93%20%28TRL%29.md)

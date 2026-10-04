# 实践：PPO与DPO方法的比较

来源：[原文](https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-6-advanced-rlhf-techniques/comparing-ppo-dpo-practice)

[返回章节目录](README.md) · [返回课程目录](../README.md)

直接偏好优化（DPO）是近端策略优化（PPO）中显式奖励建模和强化学习 (reinforcement learning)循环的替代方法。对DPO和PPO这两种主要对齐 (alignment)技术进行直接比较，有助于阐明其工作机制和实际影响。两种方法都使用人类偏好数据（$(x, y_w, y_l)$，其中$x$是提示，$y_w$是偏好响应，$y_l$是不偏好响应）来引导语言模型产生期望的反应，但其内在的工作方式和实际影响差异很大。

### 工作方式与流程的对比

最根本的区别在于每种方法如何使用偏好数据。

- **基于PPO的RLHF：** 遵循三阶段流程：

  1. **监督微调 (fine-tuning) (SFT)：** 初始策略 $\pi_{\text{SFT}}$ 在高质量示例上训练。
  2. **奖励建模 (RM)：** 一个独立的奖励模型 $r_\phi(x, y)$ 在偏好数据集 $\mathcal{D}$ 上训练。目标通常是在类似布拉德利-特里模型的框架下，最大化观察到的偏好的可能性，即对于 $\mathcal{D}$ 中的配对，$r_\phi(x, y_w) > r_\phi(x, y_l)$。
  3. **RL微调：** SFT策略 $\pi_{\text{SFT}}$ 使用PPO进一步改进。PPO算法最大化从学习到的奖励模型 $r_\phi$ 获得的预期奖励，同时一个KL散度惩罚项 $\beta \mathbb{KL}(\pi || \pi_{\text{ref}})$ 使学习到的策略 $\pi$ 保持接近参考策略 $\pi_{\text{ref}}$（通常是 $\pi_{\text{SFT}}$）。目标函数大致如下：
     
     $$
     \max_{\pi} \mathbb{E}_{x \sim D, y \sim \pi(y|x)} [r_\phi(x, y)] - \beta \mathbb{KL}(\pi(\cdot|x) || \pi_{\text{ref}}(\cdot|x))
     $$
     
- **直接偏好优化 (DPO)：** 跳过显式奖励建模阶段。它直接使用偏好数据优化语言模型策略 $\pi$。DPO根据布拉德利-特里偏好模型下的最优RLHF策略与偏好对上的简单分类目的之间的理论关系，推导出一个损失函数 (loss function)。DPO损失函数为：

  
  $$
  \mathcal{L}_{\text{DPO}}(\pi; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
  $$
  

  在此，$\pi_{\text{ref}}$ 通常是SFT模型，$\beta$ 是一个参数 (parameter)，控制与参考策略的偏差（类似于隐式奖励模型中的逆温度或PPO中的KL系数），$\sigma$ 是Sigmoid函数。此损失直接促使策略 $\pi$ 对偏好响应 $y_w$ 给予比不偏好响应 $y_l$ 更高的似然比（相对于 $\pi_{\text{ref}}$）。

工作流程可以显示如下：

> 基于PPO的RLHF与DPO的高层次工作流程比较。PPO包含一个中间奖励模型训练步骤，而DPO则使用偏好数据直接优化策略。

### 差异总结

| 特点 | 基于PPO的RLHF | 直接偏好优化 (DPO) |
| --- | --- | --- |
| **奖励模型** | 显式训练的独立模型 ($r_\phi$) | 隐式，直接从偏好似然中得出 |
| **训练阶段** | 三阶段：SFT -> RM训练 -> RL微调 (fine-tuning) | 两阶段：SFT -> DPO微调 |
| **优化** | 强化学习 (reinforcement learning) (PPO) | 监督学习 (supervised learning)（类似二分类的损失） |
| **复杂性** | 较高：需要RM基础设施、RL微调、稳定性管理 | 较低：SFT后单一优化阶段 |
| **稳定性** | 可能不稳定（RL方差，奖励欺骗） | 通常更稳定（损失更简单） |
| **超参数 (parameter) (hyperparameter)** | 较多：PPO参数（裁剪、批次等）、KL系数 $\beta$、RM参数 | 较少：主要是DPO参数 $\beta$ |
| **灵活性** | 较高：可检查/塑造RM，可能多目标 | 较低：直接与偏好数据格式绑定 |
| **实现** | 更复杂：独立的RM/RL循环 | 更简单：适合标准微调流程 |

### 实现与调优考量

- **PPO：** 要求仔细实现PPO算法的各部分，包含策略网络和价值网络、优势估算（如GAE）、策略与参考模型分布之间的KL散度计算，以及处理强化学习 (reinforcement learning)中潜在的噪声梯度。调优涉及平衡奖励最大化与KL惩罚项，管理学习率、批处理大小以及PPO特有超参数 (parameter) (hyperparameter)（例如，裁剪epsilon、每批PPO训练的批次数量）。TRL等库简化了这一点，但了解其内在工作方式对于故障排除仍有帮助。调试通常包含观察奖励曲线、KL散度趋势、价值损失和生成样本质量。
- **DPO：** 实现主要包含计算当前策略 ($\pi$) 和参考策略 ($\pi_{\text{ref}}$) 下选择的 ($y_w$) 和拒绝的 ($y_l$) 响应的对数概率，然后将这些值代入DPO损失函数 (loss function)。这通常更契合现有的监督微调 (fine-tuning)框架。主要的超参数是 $\beta$，它控制策略应根据偏好与参考模型偏离的强度。较高的 $\beta$ 会给予偏好数据更高的权重 (weight)。调优通常比PPO更简单，常类似于标准监督学习 (supervised learning)的超参数查找。

### 何时选择其中一种？

- **若选择DPO：**

  - 简单性和稳定性是优先考虑项。
  - 你想避免训练和管理独立奖励模型的额外负担。
  - 你的主要目的是直接基于成对偏好进行对齐 (alignment)，而无需在训练期间使用可解释的标量奖励信号。
  - 你有一个精心整理的偏好数据集。
- **若选择基于PPO的RLHF：**

  - 你需要或想要一个显式奖励模型，或许用于分析、内容过滤或纳入多重目标（通过组合不同的奖励信号）。
  - 你要求对RL优化过程有比DPO损失函数 (loss function)所能提供的更精细的控制。
  - 你正在研究更复杂的奖励塑形或RL技术，这些技术要求显式奖励函数。
  - 你拥有处理RL训练的复杂性和潜在不稳定性的基础设施和专业知识。

PPO和DPO都是使大型语言模型与人类偏好对齐的有效方法。DPO通过将问题重新表述为类似监督学习 (supervised learning)的目的，带来更直接且通常更稳定的途径。PPO虽然更复杂，但提供显式奖励模型的灵活性和完整的强化学习 (reinforcement learning)体系。最佳选择取决于项目的具体限制和目的，包含可用资源、期望的模型反应以及对实现复杂度的接受度。理解此处阐述的权衡，可使你在制定对齐策略时做出明智的决定。

## 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍了近端策略优化（PPO）算法，这是一种广泛用于稳定高效强化学习的策略梯度方法。
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: arXiv preprint arXiv:2203.02155; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  详细介绍了基于PPO的RLHF流程，用于使语言模型与人类指令对齐，提供了PPO工作流的实际范例。

---

[上一节](06-%E4%B8%8A%E4%B8%8B%E6%96%87%E5%92%8C%E6%9D%A1%E4%BB%B6%E5%BC%8FRLHF.md) · [下一节](../07-%E8%AF%84%E4%BC%B0%E3%80%81%E5%88%86%E6%9E%90%E4%B8%8E%E9%83%A8%E7%BD%B2/01-%E5%AF%B9%E9%BD%90%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87.md)

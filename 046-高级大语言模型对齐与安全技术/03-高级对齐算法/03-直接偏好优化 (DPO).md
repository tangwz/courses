# 直接偏好优化 (DPO)

来源：[原文](https://apxml.com/zh/courses/llm-alignment-safety/chapter-3-advanced-alignment-algorithms/direct-preference-optimization-dpo)

[返回章节目录](README.md) · [返回课程目录](../README.md)

人类反馈强化学习 (reinforcement learning) (RLHF) 包含一个多阶段流程：训练一个监督微调 (fine-tuning) (SFT) 模型，根据人类偏好训练一个奖励模型 (RM)，然后使用强化学习（如PPO）针对该奖励模型对SFT模型进行微调。尽管这种流程有效，但它涉及多个可变部分，每个部分都有其复杂性和潜在的不稳定性。准确训练奖励模型可能存在难度，而使用强化学习优化策略可能对超参数 (parameter) (hyperparameter)敏感，并易出现奖励作弊等问题。

直接偏好优化 (DPO) 通过简化这一流程提供了一个有吸引力的替代方案。它完全跳过了显式奖励建模和强化学习阶段。相反，DPO直接优化语言模型策略，使其与数据集中表达的人类偏好对齐 (alignment)。它通过巧妙地将对齐任务重新定义为对偏好数据进行的简单分类问题来实现这一点。

### DPO的主要思想

DPO背后的认识是，标准RLHF目标——旨在最大化奖励同时对偏离基础策略进行正则化 (regularization)——可以直接使用偏好数据进行优化。回顾RLHF中，目标通常是找到一个策略$\pi$来最大化：


$$
\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)} [r(x, y)] - \beta_{RL} D_{KL}(\pi(\cdot|x) || \pi_{ref}(\cdot|x))
$$


其中$r(x,y)$是奖励，$\pi_{ref}$是一个参考策略（通常是SFT模型），$\beta_{RL}$是一个正则化系数，而$D_{KL}$是Kullback–Leibler散度。

理论表明，这个目标的最优解$\pi^*$具有与奖励函数$r(x, y)$和参考策略$\pi_{ref}$相关的特定形式：


$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp(\frac{1}{\beta_{RL}} r(x, y))
$$


其中$Z(x)$是一个分配函数，确保概率总和为一。

此外，奖励模型$r(x, y)$本身是使用偏好数据训练的。假设使用像Bradley-Terry模型这样的偏好模型，给定提示$x$，人类偏好补全$y_w$而不是$y_l$的概率建模为：


$$
P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))
$$


其中$\sigma$是Sigmoid函数。

DPO结合了这些认识。它利用最优策略和奖励函数之间的关系，仅根据最优策略$\pi^*$和参考策略$\pi_{ref}$重写了偏好概率$P(y_w \succ y_l | x)$：


$$
P(y_w \succ y_l | x) = \sigma \left( \beta_{RL} \log \frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \beta_{RL} \log \frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)} \right)
$$


这个方程将偏好数据直接关联到我们想要学习的策略（$\pi^*$，由我们可训练的策略$\pi_{\theta}$近似）和一个已知的参考策略$\pi_{ref}$。DPO通过定义一个损失函数 (loss function)来使用这一点，该损失函数基于最小化在此推导模型下观察到的偏好的负对数似然。

### DPO损失函数 (loss function)

DPO损失函数训练策略$\pi_{\theta}$以满足数据集$\mathcal{D}$中观察到的人类偏好$(x, y_w, y_l)$，其中$y_w$是提示$x$的偏好（胜出）补全，而$y_l$是不偏好（落败）补全。损失定义为：


$$
L_{DPO}(\pi_{\theta}; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_{\theta}(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
$$


我们来分解一下：

- $\pi_{\theta}$：正在优化的语言模型策略（微调 (fine-tuning)）。
- $\pi_{ref}$：固定的参考策略，通常是DPO训练开始时使用的SFT模型。它起到正则化 (regularization)器的作用。
- $(x, y_w, y_l)$：偏好数据集中的一个三元组。
- $\log \frac{\pi_{\theta}(y|x)}{\pi_{ref}(y|x)}$：策略$\pi_{\theta}$与参考策略$\pi_{ref}$对补全$y$赋予的概率的对数比。这一项隐式地表示奖励。
- $\beta$：一个超参数 (parameter) (hyperparameter)，类似于RLHF目标中的$\beta_{RL}$。它控制对满足偏好的重视程度，与保持接近参考策略的程度相平衡。更高的$\beta$意味着更强的对齐 (alignment)压力。
- $\sigma$：Sigmoid函数。
- $\log \sigma(...)$：应用于胜出和落败补全之间隐式奖励差异的逻辑损失。

直观地看，该损失函数促使策略$\pi_{\theta}$增加偏好补全$y_w$的相对对数概率，并降低不偏好补全$y_l$的相对对数概率，这与参考策略$\pi_{ref}$相比。最小化此损失函数直接最大化了策略$\pi_{\theta}$符合人类偏好数据的可能性。

### 实现与训练

实现DPO包含以下步骤：

1. **从SFT模型开始：** 使用监督微调 (fine-tuning)在指令遵循数据上训练一个基础语言模型。这个模型将用作初始的$\pi_{\theta}$和固定的$\pi_{ref}$。
2. **收集偏好数据：** 收集一个包含三元组$(x, y_w, y_l)$的数据集$\mathcal{D}$，其中人类（或可能是AI标注者，如RLAIF中）已表示对于提示$x$，补全$y_w$优于$y_l$。这与RLHF奖励建模阶段所需的数据相同。
3. **计算对数概率：** 在训练期间，对于每个三元组$(x, y_w, y_l)$，计算当前策略$\pi_{\theta}$和固定参考策略$\pi_{ref}$下胜出和落败补全的对数概率：
   - $\log \pi_{\theta}(y_w|x)$ 和 $\log \pi_{\theta}(y_l|x)$（需要对$\pi_{\theta}$进行前向传播并启用梯度）
   - $\log \pi_{ref}(y_w|x)$ 和 $\log \pi_{ref}(y_l|x)$（需要对$\pi_{ref}$进行前向传播并禁用梯度）
4. **计算DPO损失：** 使用这些对数概率来计算如上所述的$L_{DPO}$损失。
5. **优化：** 使用梯度下降 (gradient descent)更新$\pi_{\theta}$的权重 (weight)以最小化损失。参考策略$\pi_{ref}$在整个过程中保持固定。

整个过程更接近于标准监督微调，而不是RLHF流程，这使其更容易实现且训练可能更稳定。

> 传统RLHF流程与更简单的DPO流程的对比。DPO将奖励建模和策略优化结合到一个微调阶段，使用专门的损失函数 (loss function)。

### DPO的优点

- **简洁性：** 主要优点是消除了奖励模型训练和RL优化阶段。这显著简化了对齐 (alignment)流程，降低了工程复杂性和潜在的故障点。
- **稳定性：** DPO避免了将RL算法（如PPO）应用于大型模型时常出现的潜在不稳定性及超参数 (parameter) (hyperparameter)调优挑战。训练过程类似于标准监督学习 (supervised learning)。
- **直接性：** 它直接针对偏好目标优化策略，而不依赖于一个中间的（且可能不完善的）奖励模型作为替代。
- **高效性：** 训练的计算强度低于完整的RLHF过程，特别是与PPO所需的采样和优化循环相比。

### 缺点与考虑

- **数据依赖性：** 与RLHF类似，DPO的有效性取决于偏好数据集$\mathcal{D}$的质量和数量。有偏或有噪声的偏好数据将导致对齐 (alignment)效果不佳的模型。
- **无显式奖励：** DPO不生成显式奖励模型。虽然这简化了训练，但显式RM有时有助于评估补全或理解模型行为。
- **超参数 (parameter) (hyperparameter)调优：** 温度参数$\beta$非常重要。它平衡了对参考模型的遵循与对偏好数据的拟合。设置过低可能导致对齐效果不佳，而设置过高则可能导致策略过度拟合偏好并过度偏离基础SFT模型，从而可能降低其他任务的性能或增加生成伪影。
- **性能上限：** 尽管DPO在实践中由于其稳定性通常表现与RLHF相当或更好，但理论上，一个完美调优的RLHF流程，如果带有准确的奖励模型，在某些情况下可能会实现略好的对齐效果。然而，实现这种完美的RLHF设置通常有难度。

DPO代表了对齐技术的一个重要进展，与传统RLHF相比，它提供了一种更精简且通常更稳定的方法。它的简洁性使其成为许多对齐任务的有吸引力的选择，前提是可获得高质量的偏好数据。理解DPO为指导LLM行为的工具集增加了一个强有力的方法。

## 参考资料

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn (2023)
  Journal: arXiv; DOI: [10.48550/arXiv.2305.18290](https://doi.org/10.48550/arXiv.2305.18290)
  介绍直接偏好优化（DPO）的基础论文，详细阐述了其理论推导以及相对于传统RLHF的经验优势。
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: arXiv; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  这篇OpenAI的开创性论文描述了使用PPO进行的人类反馈强化学习（RLHF）流程，DPO旨在简化该流程。
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍了近端策略优化（PPO），这是一种广泛用于策略优化的强化学习算法，是传统RLHF流程中的关键组成部分。

---

[上一节](02-%E5%9F%BA%E4%BA%8EAI%E5%8F%8D%E9%A6%88%E7%9A%84%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%20%28RLAIF%29.md) · [下一节](04-%E5%AF%B9%E9%BD%90%E4%B8%AD%E7%9A%84%E5%AF%B9%E6%AF%94%E6%96%B9%E6%B3%95.md)

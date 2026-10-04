---
course: "meta-learning-foundation-models"
chapter: "optimization-perspectives-meta-learning"
lesson: "meta-learning-initialization"
sourceId: 4348
sourceUrl: "https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-4-optimization-perspectives-meta-learning/meta-learning-initialization"
title: "元学习初始化策略"
description: "学习优化的初始化以加快后续适应。"
order: 4
plots: []
sourceHash: "1efbe9d26ce0e69fcc38a43e107c3b9d0194d6565d9ee37f838884cef744abd7"
sourceCorrections: []
---

双层优化框架涉及外层循环引导内层适应过程。在这种元优化中，一个重要的目标是学习最优的初始模型参数 (parameter) $\theta$。主要思想并非像标准预训练 (pre-training)那样，只找到在任务间平均表现良好的参数，而是找到一个明确为“快速适应”新任务而优化的初始化 $\theta$，仅需少量示例即可。这种学习到的 $\theta$ 为内层循环的微调 (fine-tuning)过程提供了一个高效的起点。

设想一下参数空间。一个标准的预训练模型可能会找到一个距离许多任务最优解都比较近的点。然而，元学习，特别是像模型无关元学习 (MAML) 这样的方法，寻求的是另一种点：从这个点出发，针对“新的、未见的”任务（这些任务与元训练任务来自同一分布）进行适应会异常迅速，通常只需几个梯度步。

### 目标：学习一个可适应的初始化

回顾元学习的广义双层优化结构。外层循环优化初始参数 (parameter) $\theta$，而内层循环则模拟在特定任务 $\mathcal{T}_i$ 上的适应过程。

1. **内层循环 (适应模拟)：** 对于给定任务 $\mathcal{T}_i$ 及其支持集 $D_i^{supp}$，我们通过从当前初始化 $\theta$ 开始执行一步或多步梯度，计算出适应后的参数 $\phi_i$。对于学习率为 $\alpha$ 的单步更新：
   
   $$
   \phi_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{supp}(\theta)
   $$
   
2. **外层循环 (元优化)：** 目标是更新 $\theta$，使适应后的参数 $\phi_i$ 在相应的查询集 $D_i^{query}$ 上表现良好。外部目标是在模拟适应“之后”最小化跨任务的损失：
   
   $$
   \min_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}^{query}(\phi_i) = \min_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}^{query}(\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{supp}(\theta))
   $$
   

最小化此外部目标的参数向量 (vector) $\theta$ 就是元学习得到的初始化。它已准备好在从分布 $p(\mathcal{T})$ 中采样的任务上进行快速微调 (fine-tuning)。

### 对比元初始化与预训练 (pre-training)

标准的监督预训练通常最小化一个大型多样化数据集上的平均损失：$\min_\theta \mathbb{E}_{(x,y) \sim D_{pretrain}} [\mathcal{L}(f_\theta(x), y)]$。元学习初始化，通过上述MAML式目标，最小化的是适应“之后”的损失。

考虑这个简单的图示：

> 预训练初始化 ($\theta_{pre}$) 旨在实现普遍接近，而元学习初始化 ($\theta_{meta}$) 则经过优化，以快速（例如，在一步梯度内）达到特定任务的最优解 ($\theta^*_{i}$) 之间的差异。

元学习得到的初始化 $\theta_{meta}$ 在任何特定任务上的初始损失可能略高于 $\theta_{pre}$，但其结构使得在内层循环中计算的梯度能够有效地将其移向特定任务的最优解 $\theta^*_i$。

### 强调初始化的算法

- **MAML 及其变体：** MAML、FOMAML 和 iMAML 中的主要机制正是根据上述双层目标对 $\theta$ 进行优化。它们直接计算元梯度 $\nabla_\theta \mathcal{L}^{query}(\phi_i)$ 来更新初始化。
- **Reptile：** Reptile 提供了一种更简单的一阶替代方案，它隐式地优化了一个好的初始化。它重复采样一个任务，在该任务上执行 $k$ 步标准 SGD（从当前 $\theta$ 开始得到 $\phi_i$），然后将初始化 $\theta$ 推向适应后的参数 (parameter) $\phi_i$：
  
  $$
  \theta \leftarrow \theta + \beta (\phi_i - \theta)
  $$
  
  在多个任务上平均后，此更新规则将 $\theta$ 移至参数空间中的一个点，从该点可以通过 SGD 相对容易地达到多个任务最优解。
- **ANIL (几乎无内循环)：** 认识到在内层循环中适应基础模型中数百万或数十亿参数的计算量巨大，ANIL 提出学习一种初始化，其中只有网络的一小部分（通常是最终的分类或预测层，即“头部”）需要在内层循环中进行适应。绝大多数参数（“主体”或特征提取器）在外层循环中优化，以生成能与快速适应的头部良好配合的表示。初始化 $\theta = (\theta_{body}, \theta_{head})$ 是通过学习得到的，但内层循环更新只修改 $\theta_{head}$：
  
  $$
  \phi_i = (\theta_{body}, \theta_{head} - \alpha \nabla_{\theta_{head}} \mathcal{L}_{\mathcal{T}_i}^{supp}(\theta_{body}, \theta_{head}))
  $$
  
  外层循环仍然根据 $\phi_i$ 的表现来优化 $\theta_{body}$ 和 $\theta_{head}$。这明显减轻了内层循环的计算负担，使其对大型模型更加可行。
- **学习初始学习率：** 扩展这一思想，元学习还可以优化内层循环的超参数 (hyperparameter)，例如学习率 $\alpha$。相较于固定的 $\alpha$，可以元学习得到每参数或每层的学习率，从而最大化从学习到的初始化 $\theta$ 进行适应的速度。

### 对基础模型的意义

元学习初始化策略对于适应大型基础模型尤具吸引力。标准的预训练 (pre-training)权重 (weight)提供了一个良好的通用起点，但它们并非为少样本适应而优化。元学习得到的初始化，可能只专注于适应某些组件（如ANIL中或结合下一章讨论的PEFT方法），可以提供一个明显更高效的起点。这使得模型能够以最少的数据和计算量进行下游任务的专门化，克服了标准预训练模型上完全微调 (fine-tuning)或朴素少样本学习 (few-shot learning)的一些限制。

### 实际考量

主要的挑战仍然是元优化（外层循环）的计算开销，特别是对于需要二阶导数或大量内层循环模拟的方法。有效性还取决于是否能获得多样化的元训练任务集 $\mathcal{T}_i$，这些任务能准确反映元测试期间预期的少样本任务类型。构建此类任务分布，特别是对于语言理解或视觉等复杂方面，是一个重要的实际问题。

总而言之，从优化的角度审视元学习，表明学习一个最优的“初始化”是一种有效策略。我们并非仅仅学习一个静态模型，而是学习一个专门为未来快速学习而设计的起点，这是有效少样本适应的核心要求。

## 参考资料

- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://proceedings.mlr.press/v70/finn17a.html) — Chelsea Finn, Pieter Abbeel, and Sergey Levine (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning; Publisher: PMLR; Volume: 70; Pages: 1126-1135; DOI: [10.5555/3305381.3305494](https://doi.org/10.5555/3305381.3305494)
  介绍了MAML，一种用于元学习最优初始参数以实现快速适应的基础算法。

---
course: "meta-learning-foundation-models"
chapter: "advanced-gradient-based-meta-learning"
lesson: "maml"
sourceId: 4306
sourceUrl: "https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-2-advanced-gradient-based-meta-learning/maml"
title: "模型无关元学习 (MAML)"
description: "MAML 的详细数学表述、二阶导数和计算考量。"
order: 1
plots: []
sourceHash: "88015170a06735c514c41fb13b321f66b8e60c2e14eef13a173e2989ebd8b050"
sourceCorrections: []
---

模型无关元学习 (MAML) 是一种梯度元学习方法中十分重要的算法。其主要观点是找到一组初始模型参数 (parameter) $\theta$，使其非常适合快速调整。MAML 并非学习在所有任务上平均表现良好的参数，而是学习那些在新任务上只需少量数据和梯度更新即可在该任务上获得良好表现的参数。

### 数学表述

我们将其形式化。设任务分布为 $p(\mathcal{T})$。在元训练期间，我们抽样一个批次的任务 $\{\mathcal{T}_i\}_{i=1}^B$。每个任务 $\mathcal{T}_i$ 关联着一个损失函数 (loss function) $\mathcal{L}_{\mathcal{T}_i}$，通常包含一个用于适应的支持集 $D_{\mathcal{T}_i}^{supp}$ 和一个用于评估调整后参数 (parameter)的查询集 $D_{\mathcal{T}_i}^{query}$。

核心思想涉及一个两阶段优化过程：

1. **内循环（任务特定调整）：** 对于每个任务 $\mathcal{T}_i$，从共享的初始参数 $\theta$ 开始，我们使用任务的支持集 $D_{\mathcal{T}_i}^{supp}$ 进行一次或多次梯度下降 (gradient descent)步骤。对于学习率为 $\alpha$ 的单次梯度步骤：

   
   $$
   \theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta, D_{\mathcal{T}_i}^{supp})
   $$
   

   这些调整后的参数 $\theta'_i$ 特定于任务 $\mathcal{T}_i$。
2. **外循环（元优化）：** 目标是更新初始参数 $\theta$，以最小化调整后跨任务的预期损失。元目标函数是使用调整后的参数 $\theta'_i$ 在其各自查询集 $D_{\mathcal{T}_i}^{query}$ 上计算的损失之和（或平均值）：

   
   $$
   \min_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(\theta'_i, D_{\mathcal{T}_i}^{query})
   $$
   

   代入 $\theta'_i$ 的表达式（来自内循环），目标变为：

   
   $$
   \min_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta, D_{\mathcal{T}_i}^{supp}), D_{\mathcal{T}_i}^{query})
   $$
   

元参数 $\theta$ 基于此元目标使用梯度下降进行更新，通常使用元学习率 $\beta$：


$$
\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(\theta'_i, D_{\mathcal{T}_i}^{query})
$$


### 元梯度计算

计算元梯度 $\nabla_\theta \sum_{\mathcal{T}_i} \mathcal{L}_{\mathcal{T}_i}(\theta'_i, D_{\mathcal{T}_i}^{query})$ 是 MAML 最复杂的部分。由于 $\theta'_i$ 通过梯度更新步骤依赖于 $\theta$，应用链式法则需要微分内循环的梯度。

我们考虑一个任务 $\mathcal{T}$ 并简化表示：$L_{supp}(\theta) = \mathcal{L}_{\mathcal{T}}(\theta, D_{\mathcal{T}}^{supp})$ 和 $L_{query}(\theta') = \mathcal{L}_{\mathcal{T}}(\theta', D_{\mathcal{T}}^{query})$。调整后的参数 (parameter)为 $\theta' = \theta - \alpha \nabla_\theta L_{supp}(\theta)$。此任务的元梯度为：


$$
\nabla_\theta L_{query}(\theta') = \nabla_\theta L_{query}(\theta - \alpha \nabla_\theta L_{supp}(\theta))
$$


应用链式法则得出：


$$
\nabla_\theta L_{query}(\theta') = \nabla_{\theta'} L_{query}(\theta') \cdot \nabla_\theta (\theta - \alpha \nabla_\theta L_{supp}(\theta))
$$


$$
\nabla_\theta L_{query}(\theta') = \nabla_{\theta'} L_{query}(\theta') \cdot (I - \alpha \nabla_\theta^2 L_{supp}(\theta))
$$


此处，$\nabla_{\theta'} L_{query}(\theta')$ 是查询损失对 *调整后* 参数 $\theta'$ 的梯度，在 $\theta'$ 处计算。项 $\nabla_\theta^2 L_{supp}(\theta)$ 是支持集损失对 *初始* 参数 $\theta$ 的 Hessian 矩阵。

此计算需要计算梯度 $\nabla_{\theta'} L_{query}(\theta')$、Hessian $\nabla_\theta^2 L_{supp}(\theta)$，将 Hessian 乘以 $\alpha$ 和梯度向量 (vector)，并进行矩阵减法和乘法。这涉及二阶导数，使标准 MAML 在计算上要求较高。

### MAML 算法伪代码

以下是 MAML 算法的简化表示：

**算法：MAML**

**要求：** 任务分布 $p(\mathcal{T})$
**要求：** 步长 $\alpha, \beta$

1. 随机初始化 $\theta$
2. **当** 未收敛 **时**
3. \hspace{0.5cm} 抽样一批任务 $\mathcal{T}_i \sim p(\mathcal{T})$，其中 $i=1, \dots, B$
4. \hspace{0.5cm} **对于所有** $\mathcal{T}_i$ **执行**
5. \hspace{1cm} 使用支持集评估 $\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta, D_{\mathcal{T}_i}^{supp})$
6. \hspace{1cm} 计算调整后的参数 (parameter) $\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta, D_{\mathcal{T}_i}^{supp})$ （内循环更新）
7. \hspace{0.5cm} **结束循环**
8. \hspace{0.5cm} 更新 $\theta \leftarrow \theta - \beta \nabla_\theta \sum_{i=1}^B \mathcal{L}_{\mathcal{T}_i}(\theta'_i, D_{\mathcal{T}_i}^{query})$ （外循环更新，需要使用查询集反向传播 (backpropagation)通过步骤 6）
9. **结束当循环**
10. **返回** $\theta$

### 计算考量

MAML 的主要计算难题在于外循环更新（步骤 8），具体来说是涉及二阶导数的元梯度计算。

1. **Hessian 计算/Hessian-向量 (vector)积：** 对于具有数百万或数十亿参数 (parameter)的深度学习 (deep learning)模型，显式构建 Hessian 矩阵 $\nabla_\theta^2 L_{supp}(\theta)$ 在计算上是不可行的，因为其大小为 $d \times d$，其中 $d$ 是参数数量。值得注意的是，元梯度计算仅需要 Hessian 与向量 ($\nabla_{\theta'} L_{query}(\theta')$) 的 *乘积*。这种 Hessian-向量积 (HVP) 通常可以高效计算，无需构建完整的 Hessian，一般使用有限差分或自动微分技术（例如，Pearlmutter 的技巧，涉及第二次反向传播 (backpropagation)）。然而，即使是计算 HVP 也会比标准一阶梯度计算增加可观的计算开销。
2. **内存占用：** 使用自动微分框架的标准实现需要存储内循环更新的计算图，以执行外循环梯度的反向传播。此图包含中间激活和梯度，显著增加内存需求，特别是在使用多个内循环步骤或处理大型基础模型时。
3. **计算图：** 整体计算涉及内循环（每个任务）的前向传播和反向传播，随后是使用调整后的参数在查询集上的前向传播，最后是元梯度计算的反向传播，其本身涉及与内循环梯度相关的计算。这种嵌套结构增加了整体计算成本。

> 元参数 (θ)、任务调整参数 (θ'\_i)、支持/查询损失以及 MAML 中的梯度流之间的关系。外循环根据调整后在查询集上的表现优化 θ，需要通过内循环的梯度更新步骤（红色箭头）进行反向传播，这涉及二阶导数。

这些计算要求促进了像一阶 MAML (FOMAML) 这样的近似方法和像隐式 MAML (iMAML) 这样的其他方法的出现，我们接下来将考量这些方法。对 MAML 精确机制和成本的理解，为评估这些更具扩展性的变体提供了必要的支撑。

## 参考资料

- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://proceedings.mlr.press/v70/finn17a.html) — Chelsea Finn, Pieter Abbeel, Sergey Levine (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning; Publisher: PMLR; Volume: 70; Pages: 1126-1135; DOI: [10.5555/3305890.3306001](https://doi.org/10.5555/3305890.3306001)
  介绍了MAML算法及其数学公式，并展示了其在多个领域的有效性。
- [Meta-Learning with Implicit Gradients](https://arxiv.org/abs/1909.04630) — Aravind Rajeswaran, Chelsea Finn, Sham M. Kakade, Sergey Levine (2019)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: NeurIPS; Volume: 32; Pages: 4944-4955; DOI: [10.48550/arXiv.1909.04630](https://doi.org/10.48550/arXiv.1909.04630)
  引入了隐式MAML (iMAML)，通过使用隐式微分进行元梯度计算，解决了标准MAML的计算挑战。
- [CS 330: Multi-Task and Meta-Learning (Course Materials)](https://cs330.stanford.edu/) — Chelsea Finn (2023)
  Publisher: Stanford University
  提供了元学习算法的详细讲义和解释，包括对MAML的深度分析。

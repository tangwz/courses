---
course: "meta-learning-foundation-models"
chapter: "optimization-perspectives-meta-learning"
lesson: "meta-learning-bilevel-optimization"
sourceId: 4340
sourceUrl: "https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-4-optimization-perspectives-meta-learning/meta-learning-bilevel-optimization"
title: "元学习作为双层优化"
description: "使用双层优化结构来明确元学习问题。"
order: 1
plots: []
sourceHash: "1c775a1cbdd34992020c864c2695500487896b49963ff246164d02b1e878685f"
sourceCorrections: []
---

正如章节概述中所介绍的，理解和明确元学习的一种有效方法是通过双层优化结构。这种观点将此过程界定为并非单个优化问题，而是两个嵌套的优化问题：一个针对跨任务的泛化能力优化元参数 (parameter)的*外层循环*，以及一个将元参数作为起点或指引，针对单个任务的性能优化任务特定参数的*内层循环*。

### 双层优化结构

让我们更明确地定义这些组成部分。我们假设存在一个任务分布 $p(\mathcal{T})$。对于从这个分布中抽取的每个任务 $\mathcal{T}_i$，我们都有一个支持数据集 $\mathcal{D}_i^{tr}$（用于适应）和一个查询数据集 $\mathcal{D}_i^{val}$（用于评估适应结果）。

元学习的目标是找到一组元参数 (parameter)，记作 $\theta$，使得从 $\theta$ 适应而来的模型在新任务上表现良好。适应过程本身就是内层优化循环。对于特定任务 $\mathcal{T}_i$，我们从元参数 $\theta$ 开始，通过在支持集 $\mathcal{D}_i^{tr}$ 上最小化任务特定损失 $L_{task}$ 来找到任务特定参数 $\phi_i$。这种内层优化可以表示为：


$$
\phi_i^*(\theta) = \arg\min_{\phi_i} L_{task}(\phi_i, \mathcal{D}_i^{tr}, \theta)
$$


需要指出的是，得到的最佳任务参数 $\phi_i^*(\theta)$ 是元参数 $\theta$ 的函数。符号 $L_{task}(\phi_i, \mathcal{D}_i^{tr}, \theta)$ 确认了内层优化可能直接依赖于 $\theta$，例如，通过将 $\theta$ 用作初始化或将其纳入正则化 (regularization)项。通常，$\phi_i$ 从 $\theta$ 开始，优化过程从此处进行。

外层循环，即元优化，旨在通过最小化查询集 $\mathcal{D}_i^{val}$ 上的期望损失，找到最佳元参数 $\theta$，其中使用了从内层循环获得的适应参数 $\phi_i^*(\theta)$。因此，元目标函数 $L_{meta}$ 为：


$$
\min_{\theta} L_{meta}(\theta) = \min_{\theta} \mathbb{E}_{\mathcal{T}_i \sim p(\mathcal{T})} [ L_{task}(\phi_i^*(\theta), \mathcal{D}_i^{val}, \theta) ]
$$


这种形式明确表达了“学习适应”的目标。外层循环评估元参数 $\theta$ 如何有效地促成跨任务分布的适应（内层循环）。

> 元学习中的嵌套优化结构。外层循环根据参数 $\phi_i^*$ 的表现优化元参数 $\theta$，这些参数是在内层循环中针对特定任务 $T_i$ 适应而来的。

### 与标准学习的对比

这种双层结构将元学习与标准监督学习 (supervised learning)区分开。在标准学习中，我们通常在一个大型固定数据集上，使用单个目标函数来优化一组参数 (parameter)：


$$
\min_{\theta} \mathbb{E}_{(x, y) \sim \mathcal{D}} [ L(f_{\theta}(x), y) ]
$$


在这里，$\theta$ 直接参数化了预测函数 $f_{\theta}$，目标是最小化整个数据分布 $\mathcal{D}$ 上的平均损失。在元学习中，外层目标 $L_{meta}$ 评估内层优化过程的*结果*。元参数 $\theta$ 不一定是用于特定任务预测的最终参数；相反，它们代表一种状态（例如良好的初始化或学习过程），使得任务特定参数 $\phi_i$ 可以高效地获得。

### 示例：MAML 作为双层优化

模型无关元学习 (MAML) 自然地符合这种结构。

1. **内层循环：** 对于任务 $\mathcal{T}_i$，从元参数 (parameter) $\theta$ 开始，使用支持集 $\mathcal{D}_i^{tr}$ 对任务损失 $L_{task}$ 执行一个或多个梯度下降 (gradient descent)步骤，以找到适应的参数 $\phi_i$。对于学习率为 $\alpha$ 的单次梯度下降步骤：
   
   $$
   \phi_i(\theta) = \theta - \alpha \nabla_{\theta} L_{task}(\theta, \mathcal{D}_i^{tr})
   $$
   
   （注意：我们写 $\phi_i(\theta)$ 而不是 $\phi_i^*(\theta)$，因为它通常是经过少量步骤后对真实最小值的近似。）
2. **外层循环：** 元目标是这些适应参数 $\phi_i(\theta)$ 在相应查询集 $\mathcal{D}_i^{val}$ 上的平均损失。元参数 $\theta$ 使用此元目标的梯度进行更新：
   
   $$
   \min_{\theta} L_{meta}(\theta) = \min_{\theta} \mathbb{E}_{\mathcal{T}_i \sim p(\mathcal{T})} [ L_{task}(\phi_i(\theta), \mathcal{D}_i^{val}) ]
   $$
   
   
   $$
   \theta \leftarrow \theta - \beta \nabla_{\theta} L_{meta}(\theta)
   $$
   
   其中 $\beta$ 是元学习率。

计算 $\nabla_{\theta} L_{meta}(\theta)$ 需要通过内层梯度下降步骤进行微分，从而产生二阶导数（如果内层步骤梯度 $\nabla_{\theta} L_{task}$ 相对于 $\theta$ 进行微分）。

### 固有复杂性

解决双层优化问题带来独特的挑战：

1. **梯度计算：** 计算外层循环的梯度 $\nabla_{\theta} L_{meta}(\theta)$，需要对内层循环的最佳解 $\phi_i^*(\theta)$ 相对于 $\theta$ 进行微分。这通常需要对一个优化过程（`argmin`）进行微分。对于内层循环中使用的梯度下降 (gradient descent)等迭代方法，这会导致复杂的依赖关系和潜在的高计算成本，通常涉及海森矩阵或隐式微分技术。
2. **计算成本：** 嵌套循环结构意味着，对于外层参数 (parameter) $\theta$ 的每次更新，我们需要对一批任务执行一个或多个完整的内层优化过程（或其近似）。这比标准单层优化本身计算强度更大，特别是在处理大型基础模型和大量内层循环步骤时。
3. **收敛性分析：** 收敛的理论分析比标准优化更复杂，因为两个层级间的关联。确保稳定性和收敛性通常需要仔细的算法设计和超参数 (hyperparameter)调整（例如，内层和外层学习率）。

这种双层观点提供了一个严格的数学依据，用于理解许多元学习算法。它突出了学习*如何适应*的核心目标，并为分析旨在高效解决此嵌套优化问题的算法做好了准备，这些算法我们将在后续章节中审视，包括基于梯度下降和隐式微分的技术。

## 参考资料

- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://proceedings.mlr.press/v70/finn17a.html) — Chelsea Finn, Pieter Abbeel, Sergey Levine (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning; Publisher: PMLR; Volume: 70; Pages: 1126-1135
  介绍了MAML，这是一种基础算法，通过明确的内循环和外循环，阐释了作为双层优化的元学习。
- [Meta-Learning in Neural Networks: A Survey](https://ieeexplore.ieee.org/document/9094002) — Timothy Hospedales, Antreas Antoniou, Paul Storkey (2020)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: 44; Pages: 1618-1636; DOI: [10.1109/TPAMI.2020.2995511](https://doi.org/10.1109/TPAMI.2020.2995511)
  全面概述了元学习，包括对基于优化的方法和双层框架的详细讨论，并比较了不同的方法。
- [On the Global Convergence of Model-Agnostic Meta-Learning](https://proceedings.mlr.press/v108/fallah20a.html) — Alireza Fallah, Aryan Mokhtari, Asuman Ozdaglar (2020)
  Journal: Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS 2020); Publisher: PMLR; Volume: 108; Pages: 1082-1092
  提供了MAML全局收敛特性的理论分析，解决了其双层优化公式的复杂性。

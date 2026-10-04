---
course: "meta-learning-foundation-models"
chapter: "optimization-perspectives-meta-learning"
lesson: "bilevel-optimization-algorithms"
sourceId: 4342
sourceUrl: "https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-4-optimization-perspectives-meta-learning/bilevel-optimization-algorithms"
title: "求解双层问题的算法"
description: "基于梯度下降和隐式微分的双层优化方法。"
order: 2
plots: []
sourceHash: "1701e23501cbb7066196a4a52cbd40a0177c313b78d17bfc30a3353f269d0d15"
sourceCorrections: []
---

元学习可以构建为双层优化问题。外层循环旨在找到最优的元参数 (parameter) $\theta$（例如，模型初始化、学习率），以最小化在多个任务上平均的元目标 $L_{meta}$。内层循环通过最小化任务特定的损失 $L_{task}$ 来寻找任务参数 $\phi^*$，这些参数可能从 $\theta$ 开始或受其引导。具体来说：


$$
\min_{\theta} \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})} [ L_{meta}(\phi^*(\theta, \mathcal{T})) ]
$$


$$
\text{满足 } \phi^*(\theta, \mathcal{T}) = \arg\min_{\phi} L_{task}(\phi; \theta, \mathcal{D}_{\mathcal{T}}^{tr})
$$


这里，$\mathcal{T}$ 表示从分布 $p(\mathcal{T})$ 中采样的一个任务，$\mathcal{D}_{\mathcal{T}}^{tr}$ 是任务 $\mathcal{T}$ 的支持集，且 $L_{meta}$ 通常在查询集 $\mathcal{D}_{\mathcal{T}}^{qry}$ 上评估。主要难点在于计算外层目标相对于元参数 $\theta$ 的梯度，这需要通过确定 $\phi^*$ 的内层优化过程。存在一些算法策略来解决这种依赖。

### 通过内层循环展开的梯度下降 (gradient descent)

最直接的方法，例如 MAML 等算法，涉及将内层循环的优化过程视为外层目标计算图的一部分。如果内层循环使用 $K$ 步梯度下降来寻找近似解 $\phi_K$，从 $\phi_0 = f(\theta)$ 开始（其中 $f$ 可能是恒等函数或将元参数 (parameter)映射到初始任务参数的某个函数）：


$$
\phi_{k+1} = \phi_k - \alpha \nabla_{\phi} L_{task}(\phi_k; \theta, \mathcal{D}_{\mathcal{T}}^{tr}) \quad \text{对于 } k = 0, \dots, K-1
$$


我们可以使用链式法则计算外层梯度 $\nabla_{\theta} L_{meta}(\phi_K)$，通过内层优化的所有 $K$ 步进行反向传播 (backpropagation)。这实质上“展开”了内层循环。

**原理：**
梯度计算涉及诸如 $\frac{\partial \phi_K}{\partial \theta}$ 的项。通过 $K$ 步重复应用链式法则得到：


$$
\nabla_{\theta} L_{meta}(\phi_K) = \nabla_{\phi_K} L_{meta} \cdot \frac{\partial \phi_K}{\partial \theta}
$$


其中 $\frac{\partial \phi_K}{\partial \theta}$ 取决于在每一步 $k=0, \dots, K-1$ 上 $L_{task}$ 相对于 $\phi$ 和 $\theta$ 的梯度。如果 $\phi_0 = \theta$，则依赖是直接的。如果 $\theta$ 影响 $L_{task}$ 本身（例如，超参数 (hyperparameter)适应），则会出现额外的项。

**难点：**

- **计算成本：** 经过 $K$ 步优化进行反向传播可能计算密集，特别是当需要 $L_{task}$ 的二阶导数时（如在精确 MAML 中）。成本大致与 $K$ 呈线性关系。
- **内存占用：** 反向传播需要存储 $K$ 个内层步骤的中间激活和梯度，导致大量的内存消耗，对大型基础模型尤其不利。
- **梯度消失/爆炸：** 对于大的 $K$，通过展开的优化路径传播的梯度可能出现消失或爆炸问题，类似于训练深度循环网络。

像 FOMAML 这样的一阶近似方法通过在反向传播期间忽略二阶导数项来降低成本，大幅减少计算量，但可能影响性能。Reptile 通过在任务上重复进行 SGD 步骤来近似元梯度。

> 通过内层循环展开计算梯度。元梯度 $\nabla_{\theta} L_{meta}$ 需要通过产生任务参数 $\phi_K$ 的一系列内层优化步骤进行反向传播。

### 隐式微分方法

另一种方法避免了显式展开内层循环。隐式微分基于内层循环收敛到满足某个最优条件的点 $\phi^*$ 的假设，通常是任务损失的梯度为零：


$$
\nabla_{\phi} L_{task}(\phi^*(\theta); \theta) = 0
$$


假设此条件成立，我们可以对其进行相对于 $\theta$ 的隐式微分。应用链式法则得到：


$$
\frac{d}{d\theta} [\nabla_{\phi} L_{task}(\phi^*(\theta); \theta)] = 0
$$


$$
\nabla^2_{\phi\phi} L_{task} \cdot \frac{\partial \phi^*}{\partial \theta} + \nabla^2_{\phi\theta} L_{task} = 0
$$


这里，$\nabla^2_{\phi\phi} L_{task}$ 是内层目标相对于 $\phi$ 的 Hessian 矩阵，而 $\nabla^2_{\phi\theta} L_{task}$ 是混合偏导数，两者都在 $(\phi^* (\theta), \theta)$ 处评估。我们可以重新排列以找到雅可比矩阵 $\frac{\partial \phi^*}{\partial \theta}$：


$$
\frac{\partial \phi^*}{\partial \theta} = - (\nabla^2_{\phi\phi} L_{task})^{-1} \nabla^2_{\phi\theta} L_{task}
$$


随后可以使用链式法则计算外层梯度 $\nabla_{\theta} L_{meta}(\phi^*)$：


$$
\nabla_{\theta} L_{meta}(\phi^*) = \nabla_{\phi^*} L_{meta} \cdot \frac{\partial \phi^*}{\partial \theta} = - \nabla_{\phi^*} L_{meta} (\nabla^2_{\phi\phi} L_{task})^{-1} \nabla^2_{\phi\theta} L_{task}
$$


**原理：**
重要的是，这种方法避免了显式形成或求逆可能庞大的 Hessian 矩阵 $\nabla^2_{\phi\phi} L_{task}$。相反，计算涉及求解线性系统或计算 Hessian-向量 (vector)积 (HVPs)。例如，计算最终梯度首先涉及计算向量 $v = \nabla_{\phi^*} L_{meta}$，然后求解线性系统：


$$
(\nabla^2_{\phi\phi} L_{task}) z = \nabla^2_{\phi\theta} L_{task} \quad \text{(求解矩阵 } z = \frac{\partial \phi^*}{\partial \theta} \text{ 按列)}
$$


或者直接计算所需的乘积：


$$
g = v^T (\nabla^2_{\phi\phi} L_{task})^{-1} \nabla^2_{\phi\theta} L_{task} \quad \text{(计算涉及逆的 HVP)}
$$


诸如共轭梯度算法等高效方法可以迭代地求解线性系统或计算逆 Hessian-向量积 $(\nabla^2_{\phi\phi} L_{task})^{-1} v^T$，仅需能够计算任意向量 $u$ 的 Hessian-向量积 $\nabla^2_{\phi\phi} L_{task} \cdot u$。这通常可以通过自动微分高效完成，而无需形成完整的 Hessian 矩阵。诸如隐式 MAML (iMAML) 等算法应用了此技术。

**优势：**

- **内存效率：** 不需要存储内层循环的中间激活，这使其可能更适合大型模型和长内层优化周期。内存成本大致与内层步骤数 $K$ 无关。
- **稳定性：** 对于大的 $K$，它可能比展开更稳定，避免通过展开步骤导致的梯度爆炸/消失。

**难点：**

- **内层循环收敛性：** 基于内层循环近似收敛到驻点的假设。如果内层优化提前停止或收敛不佳，性能可能会下降。
- **Hessian 逆计算：** 求解线性系统或计算逆 HVP 仍然可能计算密集，尽管通常比带有二阶导数的完全展开更快。Hessian 的条件数影响共轭梯度等迭代求解器的收敛速度。
- **实现复杂性：** 需要仔细实现 Hessian-向量积计算和迭代线性求解器。

> 通过隐式微分计算梯度。这种方法不展开内层循环，而是使用内层循环的最优条件（$\nabla_{\phi} L_{task} = 0$）和隐函数定理 (IFT) 来计算 $\theta$ 和 $\phi^*$ 之间的关系，从而通常通过 Hessian-向量积 (HVPs) 和线性求解器计算 $\nabla_{\theta} L_{meta}$。

### 方法比较

展开与隐式微分之间的选择涉及权衡：

- **展开（例如，MAML，FOMAML）：**
  - 使用标准自动微分框架实现更简单。
  - 不需要内层循环收敛，即使在少量步骤后也适用。
  - 可能内存密集且计算成本高（特别是二阶）。
  - 对于许多内层步骤，易受梯度问题影响。
- **隐式微分（例如，iMAML）：**
  - 内存效率更高，随内层步骤数扩展性更好。
  - 对于长适应周期，梯度可能更稳定。
  - 需要内层循环近似一个驻点。
  - 涉及求解线性系统（HVP 计算），这可能自身带有计算成本和稳定性问题（例如，Hessian 的条件数）。

对于内存是主要限制的大型基础模型，且适应可能涉及许多有效步骤（即使是隐式的），隐式微分方法提供了一种有吸引力的替代方案。然而，实际性能在很大程度上取决于内层优化问题的具体情况以及 HVP 计算和线性求解器的效率。混合方法或进一步的近似方法也是活跃的研究方向，旨在结合两种模式的优势。

## 参考资料

- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://proceedings.mlr.press/v70/finn17a.html) — Chelsea Finn, Pieter Abbeel, Sergey Levine (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning; Publisher: PMLR; Volume: 70; Pages: 1126-1135; DOI: [10.55989/t2q5](https://doi.org/10.55989/t2q5)
  这篇基础性论文介绍了 MAML，一种重要的元学习算法，它展示了双层优化中内循环展开方法。
- [Bilevel Optimization for Machine Learning: A Survey](https://doi.org/10.1109/TPAMI.2020.3006214) — Yifan Lu, Zhichao Huang, Luyao Niu, Weishan Zhang, Xiang Li, Shouyang Wang, Xin Li, Xiaodong Yang, Song Guo (2020)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: 43; Pages: 4081-4097; DOI: [10.1109/TPAMI.2020.3006214](https://doi.org/10.1109/TPAMI.2020.3006214)
  这篇综述全面概述了应用于包括元学习在内的各种机器学习问题的双层优化技术，并讨论了不同的算法方法。
- [OptNet: Differentiable Optimization as a Layer in Neural Networks](https://proceedings.mlr.press/v70/amos17a.html) — Brandon Amos, J. Zico Kolter (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning; Publisher: PMLR; Volume: 70; Pages: 136-145; DOI: [10.55989/v70-amos17a](https://doi.org/10.55989/v70-amos17a)
  这篇论文为通过优化问题进行微分提供了基础性见解，这是元学习和其他机器学习任务中隐式微分方法的核心技术。
- [On First-Order Meta-Learning Algorithms](https://arxiv.org/abs/1803.02999) — Alex Nichol, Joshua Achiam, John Schulman (2018)
  Journal: arXiv; Volume: abs/1803.02999; DOI: [10.48550/arXiv.1803.02999](https://doi.org/10.48550/arXiv.1803.02999)
  介绍了 Reptile，一种简单高效的一阶元学习算法，通过重复的 SGD 步骤近似元梯度，为基于梯度的元学习提供了另一种视角。

# 一阶MAML (FOMAML) 与 Reptile

来源：[原文](https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-2-advanced-gradient-based-meta-learning/fomaml-reptile-analysis)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管模型无关元学习 (MAML) 提供了一个适应性初始化学习的有效框架，但其对二阶导数（通过梯度计算梯度）的依赖，特别是对于拥有数十亿参数 (parameter)的基础模型而言，带来了很大的计算难题。为完成MAML更新所需的Hessian向量 (vector)积的计算与存储，在内存和计算时间方面都可能耗费巨大。为了处理此问题，已开发出多种一阶近似方法，它们在明显提高效率的同时，通常仍能保持不俗的性能。其中主要包括一阶MAML (FOMAML) 和 Reptile。

### 一阶MAML (FOMAML)

FOMAML 通过省略计算量大的二阶导数项来简化MAML更新。回顾MAML的元目标，其旨在最小化在支持集上执行一步或多步梯度更新*之后*，在查询集上的损失。MAML的元梯度涉及对更新后的损失关于初始参数 (parameter) $\theta$ 进行求导。应用链式法则，这会涉及内层更新步骤的梯度关于 $\theta$ 的求导，从而引入二阶导数（内层损失的Hessian）。

让我们看看任务 $\mathcal{T}_i$ 的单步内层更新：


$$
\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{S}_i}(\theta)
$$


其中 $\mathcal{L}_{\mathcal{S}_i}$ 是任务 $\mathcal{T}_i$ 的支持集 $\mathcal{S}_i$ 上的损失，$\alpha$ 是内层学习率。

完整的MAML更新涉及计算 $\nabla_\theta \mathcal{L}_{\mathcal{Q}_i}(\theta'_i)$，其中 $\mathcal{L}_{\mathcal{Q}_i}$ 是查询集 $\mathcal{Q}_i$ 上的损失。应用链式法则得到：


$$
\nabla_\theta \mathcal{L}_{\mathcal{Q}_i}(\theta'_i) = \nabla_{\theta'} \mathcal{L}_{\mathcal{Q}_i}(\theta'_i) \cdot \nabla_\theta \theta'_i = \nabla_{\theta'} \mathcal{L}_{\mathcal{Q}_i}(\theta'_i) \cdot (I - \alpha \nabla^2_\theta \mathcal{L}_{\mathcal{S}_i}(\theta))
$$


项 $\nabla^2_\theta \mathcal{L}_{\mathcal{S}_i}(\theta)$ 表示支持集损失的Hessian矩阵，这使得此计算耗费很高。

FOMAML 做了一个简单而有效的近似：它完全忽略二阶导数项。这等同于假设内层更新步骤的梯度关于 $\theta$ 是单位矩阵 ($\nabla_\theta \theta'_i \approx I$)。FOMAML 的元梯度近似变为：


$$
\nabla_\theta \mathcal{L}_{\mathcal{Q}_i}(\theta'_i)_{\text{FOMAML}} \approx \nabla_{\theta'} \mathcal{L}_{\mathcal{Q}_i}(\theta'_i)
$$


这意味着FOMAML执行内层更新以得到 $\theta'_i$，计算查询集损失关于这些*已适应*参数 $\theta'_i$ 的梯度，并*直接*将该梯度用作更新 $\theta$ 的元梯度。

跨任务批次的整体 FOMAML 更新为：


$$
\theta \leftarrow \theta - \beta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \nabla_{\theta'} \mathcal{L}_{\mathcal{Q}_i}(\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{S}_i}(\theta))
$$


其中 $\beta$ 是元学习率。

**计算优势：**
FOMAML 的主要优势是计算效率。通过避免Hessian项的计算和反向传播 (backpropagation)，可以实现：

- **内存需求减少：** 无需存储二阶导数所需的计算图。
- **计算速度加快：** 省去了耗费大的Hessian向量 (vector)积。

**性能权衡：**
尽管计算成本更低，但FOMAML是一种近似方法。省略二阶项意味着其更新没有完全考虑内层梯度步骤如何随初始参数 $\theta$ 的变化而改变。这有时可能导致初始参数的优化程度略低，与完整的MAML相比，可能需要更多适应步骤或达到略低的最佳性能。然而，实际应用中，FOMAML 通常表现不俗，特别是当内层学习率 $\alpha$ 较小或损失相对平滑时。其简单性和效率使其成为一个受欢迎的选择，尤其是在扩展到大型模型时。

### Reptile

Reptile 是另一种一阶元学习算法，它与FOMAML类似，旨在找到适合快速适应的初始参数 (parameter) $\theta$。然而，它从略微不同的角度处理问题，并采用了一种独特的更新机制。

Reptile 不像FOMAML那样在固定数量（通常为一步）的内层步骤后，基于查询集性能计算元梯度，而是对采样的任务 $\mathcal{T}_i$ 执行多步 ($k \ge 1$) 标准SGD，以获得适应后的参数 $\phi_i$。然后，它通过稍微向这些适应后的参数 $\phi_i$ 的方向移动来更新初始参数 $\theta$。

Reptile 算法在每次元迭代中按以下步骤执行：

1. 采样一批任务 $\{\mathcal{T}_i\}$。
2. 对于每个任务 $\mathcal{T}_i$：
   a. 初始化任务特定参数 $\phi \leftarrow \theta$。
   b. 使用支持集 $\mathcal{S}_i$ 执行 $k$ 步SGD：
   $\phi \leftarrow \phi - \alpha \nabla_\phi \mathcal{L}_{\mathcal{S}_i}(\phi)$（重复 $k$ 次）。
   c. 存储最终适应后的参数 $\phi_i$。
3. 更新元参数 $\theta$：
   
   $$
   \theta \leftarrow \theta + \epsilon \frac{1}{N} \sum_{i=1}^N (\phi_i - \theta)
   $$
   
   其中 $N$ 是批次中的任务数量，$\epsilon$ 是元学习率（外部步长）。

**机制和解释：**
Reptile 的更新规则只是在当前的元参数 $\theta$ 和经过 $k$ 步任务特定优化后获得的参数 $\phi_i$ 之间进行插值。直观上，Reptile 寻找一个点 $\theta$，使其在参数空间中接近许多不同任务的最优解。

从数学角度看，通过泰勒展开分析可知，Reptile 的更新近似于一个涉及高阶导数的元梯度。与MAML/FOMAML显式地优化在固定步数后的性能（对适应性的敏感度）不同，Reptile 隐式地优化一个初始化，使其在跨任务适应过程中移动的距离最小化。使用多步内层迭代 ($k > 1$) 在 Reptile 中很常见，它允许任务特定优化在元更新前更接近任务最优解。

**计算效率：**
与FOMAML类似，Reptile 是一种一阶方法。它在内层循环（任务适应）中只需进行标准梯度计算。它避免了二阶导数以及相关的计算开销，使其具有可扩展性。

### 比较：FOMAML 与 Reptile

FOMAML 和 Reptile 都为基于梯度的元学习提供了计算高效的 MAML 替代方案。

| 特性 | FOMAML | Reptile |
| --- | --- | --- |
| **核心思想** | 通过忽略二阶导数来近似MAML。 | 将初始参数 (parameter)向多步适应后的参数移动。 |
| **内层循环** | 通常在支持集 ($\mathcal{S}_i$) 上进行1步或少量SGD。 | 通常在支持集 ($\mathcal{S}_i$) 上进行多步 ($k \ge 1$) SGD。 |
| **元更新** | 使用适应参数 $\theta'_i$ 处查询损失 ($\mathcal{L}_{\mathcal{Q}_i}$) 的梯度。 | 在初始 $\theta$ 和适应参数 $\phi_i$ 之间插值。 |
| **解释** | 优化敏感性（在少数步骤后表现良好）。 | 优化接近性（初始参数接近任务最优解）。 |
| **复杂度** | 一阶，高效。需要对内层步骤进行反向传播 (backpropagation)（一次）。 | 一阶，高效。元更新只需前向传播。 |
| **二阶导数** | 否 | 否 |

**实际考量：**

- **内层步数 ($k$)：** FOMAML 通常使用 $k=1$，而 Reptile 常常从 $k > 1$ 中受益。
- **超参数 (hyperparameter)：** 两者都需要调整内层学习率 $\alpha$ 和外层（元）学习率（FOMAML 为 $\beta$，Reptile 为 $\epsilon$）。内层步数 $k$ 是 Reptile 的一个额外超参数。
- **性能：** 性能可能取决于具体任务。如果使用 $k=1$，FOMAML 每元迭代的收敛速度可能略快，而 Reptile 使用 $k>1$ 则可能由于其独特的优化目标而找到不同的解决方案。

FOMAML 和 Reptile 都是元学习工具集中很有用的工具，尤其是在处理大型基础模型带来的计算限制时。它们的一阶特性与完整的MAML相比大幅降低了成本，使得基于梯度的元学习成为在有限数据下适应这些大型架构的更可行策略。两者之间的选择可能取决于具体的适应任务、计算预算以及实验中观察到的实际性能。

## 参考资料

- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) — Chelsea Finn, Pieter Abbeel, Sergey Levine (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning (ICML); Pages: 1126-1135; DOI: [10.48550/arXiv.1703.03400](https://doi.org/10.48550/arXiv.1703.03400)
  本文介绍了模型无关元学习（MAML）框架，并讨论了其一阶近似 FOMAML。
- [On First-Order Meta-Learning Algorithms](https://arxiv.org/abs/1803.02999) — Alex Nichol, Joshua Achiam, John Schulman (2018)
  Journal: arXiv preprint arXiv:1803.02999; DOI: [10.48550/arXiv.1803.02999](https://doi.org/10.48550/arXiv.1803.02999)
  本文介绍了 Reptile 算法，这是一种一阶元学习方法，作为 MAML 的计算高效替代方案。

---

[上一节](01-%E6%A8%A1%E5%9E%8B%E6%97%A0%E5%85%B3%E5%85%83%E5%AD%A6%E4%B9%A0%20%28MAML%29.md) · [下一节](03-%E9%9A%90%E5%BC%8FMAML%20%28iMAML%29.md)

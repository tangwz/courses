---
course: "bayesian-machine-learning"
chapter: "variational-inference-techniques"
lesson: "svi-large-data"
sourceId: 3568
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-3-variational-inference-techniques/svi-large-data"
title: "随机变分推断 (SVI) 针对大规模数据"
description: "应用 SVI 针对大规模数据集进行可扩展贝叶斯推断，使用随机梯度方法。"
order: 5
plots: []
sourceHash: "540181a6e8389a47499c26e3a343439c2dc64c925ec210237d60712fe8ac6f92"
sourceCorrections: []
---

坐标上升变分推断 (CAVI) 是一种用于优化概率模型中证据下界 (ELBO) 的技术。然而，CAVI 的一个局限性是它需要遍历整个数据集来更新每个模型参数 (parameter)。对于现代数据集，它们可能包含数百万甚至数十亿个数据点，这种全数据集遍历会成为一个重要的计算瓶颈。想象一下，仅仅为了对模型参数进行一次调整，就需要处理数TB的数据；这个过程将极其缓慢。

随机变分推断 (SVI) 解决了这一可伸缩性难题，通过借鉴随机优化中的思想，特别是随机梯度下降 (gradient descent) (SGD)。SVI 不使用完整数据集计算 ELBO 的真实梯度，而是使用噪声较大但计算成本较低的梯度，这些梯度是从小而随机选择的数据子集（通常称为小批量）中估计出来的。

### 从批量梯度到随机梯度

召回 ELBO：
$\mathcal{L}(q) = \mathbb{E}_{q(\mathbf{z})} [\log p(\mathbf{x}, \mathbf{z})] - \mathbb{E}_{q(\mathbf{z})} [\log q(\mathbf{z})]$
假设给定隐变量 $\mathbf{z}$ 时，数据点 $\mathbf{x} = \{\mathbf{x}_1, ..., \mathbf{x}_N\}$ 是条件独立的，联合概率通常可以分解为：$\log p(\mathbf{x}, \mathbf{z}) = \log p(\mathbf{z}) + \sum_{n=1}^N \log p(\mathbf{x}_n | \mathbf{z}_n)$。让我们简化并考虑模型中 $\mathbf{z}$ 包含在所有数据点之间共享的全局参数 (parameter) $\beta$，以及每个数据点 $\mathbf{x}_n$ 特有的局部隐变量 $\mathbf{z}_n$。变分分布 $q$ 也相应地因子化，通常是 $q(\beta, \mathbf{z}_{1:N}) = q(\beta; \lambda) \prod_{n=1}^N q(\mathbf{z}_n; \phi_n)$，其中 $\lambda$ 和 $\phi_n$ 分别是全局变量和局部变量的变分参数。

ELBO 就可以写成包含对数据点求和的形式：
$\mathcal{L}(\lambda, \phi_{1:N}) = \mathbb{E}_q [\log p(\beta)] + \sum_{n=1}^N \mathbb{E}_q [\log p(\mathbf{x}_n | \mathbf{z}_n, \beta)] - \mathbb{E}_q [\log q(\beta; \lambda)] - \sum_{n=1}^N \mathbb{E}_q [\log q(\mathbf{z}_n; \phi_n)]$
CAVI 需要优化这个完整的优化目标。然而，SVI 侧重于全局参数 $\lambda$。为了更新 $\lambda$，我们需要梯度 $\nabla_{\lambda} \mathcal{L}$。注意，求和项 $\sum_{n=1}^N \mathbb{E}_q [\log p(\mathbf{x}_n | \mathbf{z}_n, \beta)]$ 使得在不接触所有数据的情况下难以进行计算。

SVI 的核心思想是仅使用小批量数据来近似完整梯度。如果我们均匀随机抽取一个数据点 $\mathbf{x}_n$，我们就可以构建一个与求和项相关的梯度的无偏随机估计。通过从 $\{1, ..., N\}$ 中抽取一个索引的小批量 $M$，我们可以形成一个有噪声的梯度估计：
$\nabla_{\lambda} \hat{\mathcal{L}}_M \approx \nabla_{\lambda} \mathbb{E}_q [\log p(\beta)] + \frac{N}{|M|} \sum_{n \in M} \nabla_{\lambda} \mathbb{E}_q [\log p(\mathbf{x}_n | \mathbf{z}_n, \beta)] - \nabla_{\lambda} \mathbb{E}_q [\log q(\beta; \lambda)]$
缩放因子 $N/|M|$ 修正了以下事实：我们是对一个较小的集合 $M$ 而非完整数据集 $N$ 进行求和，确保随机梯度仍然是真实梯度的无偏估计（考虑到涉及 $q$ 上的期望的依赖关系）。

### SVI 更新算法

SVI 迭代进行，使用这些随机梯度更新全局和局部变分参数 (parameter)。在每个步骤 $t$：

1. **采样小批量：** 从完整数据集中随机选择一个数据点的小子集（小批量）$M_t$。
2. **更新局部参数：** 对于小批量 $M_t$ 中的每个数据点 $\mathbf{x}_n$，更新其对应的局部变分参数 $\phi_n$ 以优化其对 ELBO 的贡献，同时保持当前全局参数 $\lambda^{(t)}$ 不变。这可能涉及一次或多次针对该数据点的类似 CAVI 的更新。
   $\phi_n^{(t+1)} \leftarrow \arg \max_{\phi_n} \mathbb{E}_q [\log p(\mathbf{x}_n | \mathbf{z}_n, \beta)] - \mathbb{E}_q [\log q(\mathbf{z}_n; \phi_n)]$
   （其中期望是在 $\lambda^{(t)}$ 和其他 $\phi_{m \neq n}$ 保持不变的情况下计算的）。
3. **计算随机梯度：** 使用 $n \in M_t$ 的更新后的局部参数 $\phi_n^{(t+1)}$，计算 ELBO 相对于全局参数 $\lambda$ 的有噪声梯度 $\nabla_{\lambda} \hat{\mathcal{L}}_{M_t}$。
4. **更新全局参数：** 使用学习率 $\rho_t$ 进行梯度上升步骤，更新全局变分参数 $\lambda$：
   $\lambda^{(t+1)} \leftarrow \lambda^{(t)} + \rho_t \nabla_{\lambda} \hat{\mathcal{L}}_{M_t}$

这个过程重复进行，循环遍历数据小批量。学习率 $\rho_t$ 通常会随着迭代次数的增加而减小，以确保收敛。

> SVI 更新步骤的视图。从大型数据集中采样一个小批量。这个小批量，连同当前的全局参数 $\lambda$，用于计算有噪声的梯度，然后该梯度更新 $\lambda$。

### 学习率与收敛

学习率调度 $\rho_t$ 的选择对 SVI 的表现很重要。为了使随机更新能够正确收敛，学习率必须满足 Robbins-Monro 条件：
$\sum_{t=1}^{\infty} \rho_t = \infty \quad \text{且} \quad \sum_{t=1}^{\infty} \rho_t^2 < \infty$
常见的选择是多项式衰减调度：
$\rho_t = (\tau_0 + t)^{-\kappa}$
其中 $\kappa \in (0.5, 1]$ 控制衰减率，且 $\tau_0 \geq 0$ 会降低早期迭代的权重 (weight)。调整 $\kappa$ 和 $\tau_0$ 以及小批量大小 $|M|$ 通常需要实验。过大的学习率可能导致不稳定，而过小的学习率可能导致收敛缓慢。

### 自然梯度以实现更快收敛

标准梯度上升在参数 (parameter)空间的欧几里得几何中，沿着最陡峭的上升方向更新参数 $\lambda$。然而，变分参数通常定义概率分布，欧几里得几何可能不是最合适的。分布空间有其自身的几何结构，由费雪信息矩阵 $F(\lambda)$ 描述。

自然梯度通过将标准梯度乘以费雪信息矩阵的逆矩阵来修改更新方向：
$\tilde{\nabla}_{\lambda} \mathcal{L} = F(\lambda)^{-1} \nabla_{\lambda} \mathcal{L}$
SVI 更新规则变为：
$\lambda^{(t+1)} \leftarrow \lambda^{(t)} + \rho_t \tilde{\nabla}_{\lambda} \hat{\mathcal{L}}_{M_t}$
对于属于指数族（一个常见选择）的变分分布，计算自然梯度有时比标准梯度更简单，并且通常会使收敛速度大幅加快，因为它考虑了变分参数空间的信息几何。

### SVI 的优点与考量

**优点：**

- **可伸缩性：** SVI 的主要优点是其处理不适合内存的庞大数据集的能力，因为它在每一步只处理小型小批量数据。
- **速度：** 与批量 VI (CAVI) 或 MCMC 方法相比，它在实际运行时长方面通常收敛速度快得多，尤其是在优化过程的早期阶段。
- **在线学习：** SVI 可以自然地整合新到达的数据点，而无需在整个数据集上重新训练。

**考量：**

- **调优：** 需要仔细调整学习率调度和小批量大小。
- **噪声梯度：** 梯度的随机性质引入了噪声，这与批量方法相比，在后期阶段可能会减慢收敛速度或导致在最优值附近出现振荡。
- **近似质量：** 像所有 VI 方法一样，SVI 在所选变分族内找到一个近似值，这可能无法完美地捕捉真实的后验分布。质量取决于族 $q$ 的灵活性。

SVI 为将贝叶斯推断应用于大规模问题提供了一个强大的工具，在这些问题中，传统方法在计算上变得不可行。它构成了许多现代概率建模应用的核心，特别是在主题建模（例如，在大规模文本语料库上的潜在狄利克雷分配）等方面，并且是贝叶斯深度学习 (deep learning)中所用技术的前身。尽管 MCMC 方法可能提供渐近精确的样本，并且 CAVI 提供确定性更新，SVI 实现了实用平衡，通过发挥随机优化的能力，实现了大规模的近似贝叶斯推断。

## 参考资料

- [Stochastic Variational Inference](https://www.jmlr.org/papers/v14/hoffman13a.html) — Matthew D. Hoffman, David M. Blei, Chong Wang, and John Paisley (2013)
  Journal: Journal of Machine Learning Research; Volume: 14; Pages: 1303-1347; DOI: [10.5555/2503259.2503307](https://doi.org/10.5555/2503259.2503307)
  介绍随机变分推断（SVI）方法，提供其理论基础、算法细节，并展示其在大规模贝叶斯推断中的可伸缩性。
- [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html) — Kevin Patrick Murphy (2022)
  Publisher: MIT Press; Pages: 843-851
  一本标准教材，广泛涵盖概率模型，其中包含关于随机变分推断的详细章节。
- [Stochastic Variational Inference (SVI) in Pyro](https://docs.pyro.ai/en/stable/inference.html#stochastic-variational-inference-svi) — The Pyro Team, Uber AI Labs (2024)
  Pyro概率编程语言中实现SVI的官方文档，包含实际示例和概念说明。

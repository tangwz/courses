# 坐标上升变分推断 (CAVI)

来源：[原文](https://apxml.com/zh/courses/bayesian-machine-learning/chapter-3-variational-inference-techniques/cavi-algorithm)

[返回章节目录](README.md) · [返回课程目录](../README.md)

坐标上升变分推断（CAVI）是一种用于在选定的分布族 $\mathcal{Q}$ 中，找到真实后验 $p(\mathbf{z}|\mathbf{x})$ 的最佳近似 $q(\mathbf{z})$ 的技术。这通常通过最大化证据下界（ELBO）来实现。变分推断（VI）广泛基于此原理。一种常见的近似选择是均值场变分族，它假设潜在变量（或变量组）在近似中相互独立，形式为：
$q(\mathbf{z}) = \prod_{j=1}^M q_j(z_j)$
其中 $\mathbf{z} = \{z_1, \dots, z_M\}$ 是潜在变量。现在的主要问题是：*如何*找出最大化ELBO $\mathcal{L}(q)$ 的最优因子 $q_j(z_j)$？

CAVI提供了一种迭代方法。它每次针对一个因子 $q_j(z_j)$ 最大化ELBO，同时保持所有其他因子 $q_{i \neq j}(z_i)$ 固定不变。这类似于在分布的函数空间中应用的坐标上升优化。

### 推导CAVI更新规则

我们来关注优化ELBO，使其针对单个因子，例如 $q_j(z_j)$。回顾ELBO的表达式：
$\mathcal{L}(q) = \mathbb{E}_{q} [\log p(\mathbf{x}, \mathbf{z})] - \mathbb{E}_{q} [\log q(\mathbf{z})]$
代入均场假设 $q(\mathbf{z}) = \prod_{i=1}^M q_i(z_i)$：
$\mathcal{L}(q) = \int \dots \int \left( \prod_{i=1}^M q_i(z_i) \right) \left( \log p(\mathbf{x}, \mathbf{z}) - \sum_{i=1}^M \log q_i(z_i) \right) dz_1 \dots dz_M$
我们希望找到最大化此表达式的 $q_j(z_j)$，同时保持 $q_{i \neq j}$ 固定。我们把包含 $q_j(z_j)$ 的项分离出来：
$\mathcal{L}(q) = \int q_j(z_j) \left( \int \dots \int \left( \prod_{i \neq j} q_i(z_i) \right) \log p(\mathbf{x}, \mathbf{z}) \left( \prod_{i \neq j} dz_i \right) \right) dz_j - \int q_j(z_j) \log q_j(z_j) dz_j + \text{常数}$
这里，"常数" 表示不依赖于 $q_j(z_j)$ 的项。第一个积分中大括号内的项是关于所有*除* $q_j$ 之外的因子对数联合分布的期望。我们将其记为 $\mathbb{E}_{i \neq j} [\log p(\mathbf{x}, \mathbf{z})]$。因此我们有：
$\mathcal{L}(q) = \int q_j(z_j) \mathbb{E}_{i \neq j} [\log p(\mathbf{x}, \mathbf{z})] dz_j - \int q_j(z_j) \log q_j(z_j) dz_j + \text{常数}$
令 $\log \tilde{p}_j(z_j) = \mathbb{E}_{i \neq j} [\log p(\mathbf{x}, \mathbf{z})]$。该表达式变为：
$\mathcal{L}(q) = \int q_j(z_j) \log \tilde{p}_j(z_j) dz_j - \int q_j(z_j) \log q_j(z_j) dz_j + \text{常数}$
$\mathcal{L}(q) = - \int q_j(z_j) \log \frac{q_j(z_j)}{\tilde{p}_j(z_j)} dz_j + \text{常数}$
$\mathcal{L}(q) = -KL(q_j(z_j) || \tilde{p}_j(z_j)) + \text{常数}$
最大化 $\mathcal{L}(q)$ 针对 $q_j(z_j)$ 而言，等同于最小化 $q_j(z_j)$ 和 $\tilde{p}_j(z_j)$ 之间的Kullback-Leibler（KL）散度。当 $q_j(z_j) = \tilde{p}_j(z_j)$ 时，KL散度被最小化（变为零）。因此，最优解 $q_j^*(z_j)$ 满足：
$\log q_j^*(z_j) = \mathbb{E}_{i \neq j} [\log p(\mathbf{x}, \mathbf{z})] + \text{常数}$
对两边取指数得到：
$q_j^*(z_j) \propto \exp \left( \mathbb{E}_{i \neq j} [\log p(\mathbf{x}, \mathbf{z})] \right)$
常数项由归一化 (normalization)要求 $\int q_j^*(z_j) dz_j = 1$ 确定。

这是CAVI的核心更新规则。它表明，第 $j$ 个因子 $q_j(z_j)$ 的最优分布与联合概率 $p(\mathbf{x}, \mathbf{z})$ 的期望对数取指数后的结果成正比，其中期望是针对所有其他因子 $q_{i \neq j}(z_i)$ 的当前分布来计算的。

### CAVI算法

CAVI算法按以下步骤进行：

1. **初始化：** 初始化每个变分因子 $q_j(z_j)$（对于 $j=1, \dots, M$）的参数 (parameter)。这可能涉及根据为每个 $q_j$ 选择的分布形式来初始化均值、方差或其他相关参数。
2. **迭代：** 重复直到收敛：
   - 对于每个因子 $j = 1, \dots, M$：
     - 计算期望 $\mathbb{E}_{i \neq j} [\log p(\mathbf{x}, \mathbf{z})]$。这需要使用分布 $q_{i \neq j}(z_i)$ 的当前参数。
     - 使用以下规则更新因子 $q_j(z_j)$：
       $q_j(z_j) \leftarrow \frac{\exp \left( \mathbb{E}_{i \neq j} [\log p(\mathbf{x}, \mathbf{z})] \right)}{\int \exp \left( \mathbb{E}_{i \neq j} [\log p(\mathbf{x}, \mathbf{z})] \right) dz_j}$
       在实践中，这通常意味着基于计算出的期望来更新分布 $q_j(z_j)$ 的*参数*。
3. **收敛检查：** 监控ELBO或变分因子的参数。当迭代间的变化低于预设阈值时停止。

> CAVI算法迭代更新每个因子 $q_j(z_j)$，其依据是对数联合概率的期望，该期望又取决于其他因子的当前状态，直至收敛。

### 共轭性的作用

CAVI的实用性通常取决于模型 $p(\mathbf{x}, \mathbf{z})$ 的结构。如果完全条件分布 $p(z_j | \mathbf{x}, \mathbf{z}_{\neg j})$（其中 $\mathbf{z}_{\neg j}$ 表示除 $z_j$ 之外的所有变量）与 $z_j$ 的先验分布属于同一族，则该模型表现出条件共轭性。在这种情况下，对 $q_j^*(z_j)$ 的CAVI更新通常会产生与 $q_j(z_j)$ 初始选择具有相同参数 (parameter)形式的分布。

例如，如果 $q_j(z_j)$ 被选择为高斯分布，并且模型结构允许，那么从期望 $\mathbb{E}_{i \neq j} [\log p(\mathbf{x}, \mathbf{z})]$ 导出的更新 $q_j^*(z_j)$ 也可能是高斯分布。更新步骤随后简化为基于从其他因子 $q_{i \neq j}$ 导出的期望值（矩）来计算新的均值和方差参数。这使得实现变得容易很多，因为我们只需要跟踪和更新因子的参数（例如，均值、方差）。

### CAVI的优点和缺点

**优点：**

- **保证收敛：** ELBO在CAVI的每一步都保证会增加（或保持不变），从而确保收敛到局部最优解。
- **确定性：** 与MCMC不同，CAVI是一种确定性优化算法，在相同初始化下会产生相同结果。
- **简洁性（在共轭情况下）：** 当条件共轭性适用时，更新通常可以通过解析方式导出，并涉及更新标准分布的参数 (parameter)。

**缺点：**

- **局部最优：** CAVI优化ELBO，而ELBO通常是非凸的。该算法可能收敛到局部最优解，从而可能提供对真实后验的糟糕近似。不同的初始化可能导致不同的结果。
- **均场限制：** 精度受到所选变分族表达能力的根本限制。均场假设（因子之间独立）对于具有强后验依赖性的模型可能过于受限。
- **解析计算：** 推导更新规则需要计算 $\mathbb{E}_{i \neq j} [\log p(\mathbf{x}, \mathbf{z})]$。这对于某些模型可能很复杂或在解析上难以处理。
- **可扩展性：** 传统CAVI需要遍历整个数据集来计算全局变量更新（控制所有数据点的参数）的期望，这在大数据集上扩展性不佳。

CAVI提供对变分优化的一种基本认识。尽管它对于中等规模的问题和合适的模型结构有效，但其局限性，尤其是在可扩展性和需要解析推导方面，促使了随机变分推断（SVI）和黑盒 (black box)变分推断（BBVI）等更先进技术的发展，我们将在接下来研究这些方法。

## 参考资料

- [Pattern Recognition and Machine Learning](https://link.springer.com/book/10.1007/978-0-387-45528-0) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: Chapter 10; DOI: [10.1007/978-0-387-45528-0](https://doi.org/10.1007/978-0-387-45528-0)
  全面且基础地介绍了变分推断，包括均值场框架下的坐标上升变分推断（CAVI）的推导和算法。
- [Variational Inference: A Review for Statisticians](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1328493) — David M. Blei, Alp Kucukelbir, Jon D. McAuliffe (2017)
  Journal: Journal of the American Statistical Association; Publisher: Taylor & Francis; Volume: 112; Pages: 859-877; DOI: [10.1080/01621459.2017.1328493](https://doi.org/10.1080/01621459.2017.1328493)
  一篇权威的综述文章，全面概述了变分推断，详细阐述了CAVI的理论基础和实际应用。
- [Probabilistic Machine Learning: Advanced Topics](https://probml.github.io/pml-book/book2.html) — Kevin Patrick Murphy (2023)
  Publisher: MIT Press; Pages: Part IV
  深入且前沿地论述了变分推断，详细探讨了CAVI的机制和应用。
- [Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — David M. Blei, Andrew Y. Ng, and Michael I. Jordan (2003)
  Journal: Journal of Machine Learning Research; Volume: 3; Pages: 993-1022
  这篇开创性论文将均值场坐标上升方法应用于概率主题模型（潜在狄利克雷分配），展示了CAVI在复杂生成模型中的实际实现和有效性。

---

[上一节](03-%E5%B9%B3%E5%9D%87%E5%9C%BA%E8%BF%91%E4%BC%BC%E5%85%B7%E4%BD%93%E5%86%85%E5%AE%B9.md) · [下一节](05-%E9%9A%8F%E6%9C%BA%E5%8F%98%E5%88%86%E6%8E%A8%E6%96%AD%20%28SVI%29%20%E9%92%88%E5%AF%B9%E5%A4%A7%E8%A7%84%E6%A8%A1%E6%95%B0%E6%8D%AE.md)

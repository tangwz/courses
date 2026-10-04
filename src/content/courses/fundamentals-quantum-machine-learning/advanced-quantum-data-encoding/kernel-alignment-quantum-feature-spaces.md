---
course: "fundamentals-quantum-machine-learning"
chapter: "advanced-quantum-data-encoding"
lesson: "kernel-alignment-quantum-feature-spaces"
sourceId: 1044
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-2-advanced-quantum-data-encoding/kernel-alignment-quantum-feature-spaces"
title: "核对齐与量子特征空间"
description: "了解核对齐技术以及量子特征映射在希尔伯特空间中产生的几何结构。"
order: 5
plots: []
sourceHash: "05e277c940d5cc625e7d9d19ac6b3c2abd7a5a23a233d2581136bd0f73b42c5c"
sourceCorrections: []
---

如我们所见，量子特征映射$\phi(x)$提供了一条连接，将经典数据向量 (vector)$x \in \mathcal{X}$转换成可能很大的希尔伯特空间$\mathcal{H}$中的量子态$|\phi(x)\rangle$。这种映射不只是一种形式步骤；它从根本上规定了数据如何进行量子表示的*几何结构*。这些量子态之间的关系，特别是它们的内积$|\langle \phi(x) | \phi(x') \rangle|^2$，决定了后续量子算法，尤其是核方法可以加以利用的结构。

### 量子核与特征空间几何结构

回顾经典核方法，例如支持向量 (vector)机（SVMs），核函数$k(x, x')$计算高维特征空间中特征向量的内积，$k(x, x') = \langle \phi_{classical}(x), \phi_{classical}(x') \rangle$。这些方法的成功取决于$\phi_{classical}$所产生的几何结构是否使得数据在该特征空间中线性可分（或易于处理）。

类似地，量子特征映射$|\phi(x)\rangle$隐含地定义了一个**量子核**：
$k_q(x, x') = |\langle \phi(x) | \phi(x') \rangle|^2$
这个核衡量了数据点$x$和$x'$对应的量子态之间的相似度。这种重叠的大小与态之间的跃迁振幅相关，通常使用特定的测量电路进行估计（如SWAP测试或反演测试，尽管在开发过程中通过模拟器直接计算也很常见）。这个量子核$k_q$随后可以插入到经典核机器中（如SVMs），从而产生诸如量子支持向量机（QSVM）之类的算法，我们将在第3章中了解这些算法。

但我们如何判断所选的特征映射$|\phi(x)\rangle$是否为特定机器学习 (machine learning)任务带来了*有用的*几何结构？一个特征映射可能会创建复杂的纠缠态，但如果其产生的几何结构与数据中潜在的模式（例如类别标签）不一致，那么量子核就不会带来良好的表现。这就引出了核对齐 (alignment)的说法。

### 核目标对齐 (alignment)

核目标对齐提供了一种量化 (quantization)衡量方式，用于评估我们的量子特征映射（由量子核矩阵$K_q$表示）所产生的几何结构与学习任务结构（由一个*理想的*目标核矩阵$K_t$表示）的匹配程度。

给定数据集$D = \{(x_1, y_1), \dots, (x_m, y_m)\}$，我们可以计算$m \times m$量子核矩阵$K_q$，其中$(K_q)_{ij} = k_q(x_i, x_j) = |\langle \phi(x_i) | \phi(x_j) \rangle|^2$。

我们还根据标签$y_i$定义一个目标核矩阵$K_t$。对于$y_i \in \{-1, +1\}$的二分类任务，一种常见选择是理想分类核：
$(K_t)_{ij} = y_i y_j$
这个目标核将来自同一类别的样本对设为$+1$，将来自不同类别的样本对设为$-1$。它完美地捕捉了所需的区分结构。

**核对齐**$A(K_q, K_t)$随后被定义为这两个矩阵之间的余弦相似度，将其视为向量 (vector)，使用弗罗贝尼乌斯内积：
$A(K_q, K_t) = \frac{\langle K_q, K_t \rangle_F}{\|K_q\|_F \|K_t\|_F} = \frac{\sum_{i,j=1}^m (K_q)_{ij} (K_t)_{ij}}{\sqrt{\sum_{i,j=1}^m (K_q)_{ij}^2} \sqrt{\sum_{i,j=1}^m (K_t)_{ij}^2}}$
此处，$\langle \cdot, \cdot \rangle_F$是弗罗贝尼乌斯内积（元素乘积之和），$\| \cdot \|_F$是弗罗贝尼乌斯范数。

对齐度$A(K_q, K_t)$的范围在-1到1之间（尽管对于标准的$k_q$和$K_t=yy^T$定义，通常是非负的）。

- 值接近1表明量子核$K_q$与目标结构$K_t$非常吻合。这表明量子特征映射$|\phi(x)\rangle$将同类数据点在希尔伯特空间中放置得“接近”（高重叠），而将不同类数据点放置得“远”（低重叠），从而可能使数据更容易区分。
- 值接近0表明量子核的结构与目标结构无关，这意味着特征映射未能有效捕捉到与任务相关的关系。

> 核对齐的图示。特征映射$\phi_1$产生的特征空间几何结构使类别（红色与蓝色）区分良好，从而与理想分类核的对齐度很高。特征映射$\phi_2$使类别混杂，导致对齐度较低。

### 使用对齐 (alignment)度设计特征映射

核对齐不只是一种分析工具；它能主动指导量子特征映射的设计。当面对多种潜在的编码策略（不同的电路结构、基态与振幅编码等编码方法、数据重上传电路中的层数）时，你可以：

1. **选择目标核：** 为你的任务选择合适的$K_t$（例如，分类任务使用$yy^T$）。
2. **计算量子核：** 对于每个候选特征映射$\phi_k(x)$，在代表性的训练数据集上计算相应的量子核矩阵$K_{q,k}$。
3. **计算对齐度：** 计算每个候选的对齐度$A(K_{q,k}, K_t)$。
4. **选择最佳映射：** 选择对齐度最高的特征映射$\phi_k(x)$。

这个过程使得可以通过数据驱动的方式选择甚至优化特征映射，*而无需*在训练完整的QML模型（如QSVM）上投入计算资源。如果特征映射本身具有可调参数 (parameter)$\theta$（如编码中使用的变分电路），则可以优化这些参数$\theta$以*最大化*核对齐度$A(K_q(\theta), K_t)$。

### 泛化能力的实用说明

虽然功能强大，但核对齐 (alignment)的使用有一些实际考量：

- **计算成本：** 计算完整的$m \times m$量子核矩阵$K_q$需要对核函数$k_q(x_i, x_j)$进行$O(m^2)$次评估。每次评估都涉及运行量子电路（通常需要多次运行以进行估计）。这可能需要大量计算资源，特别是对于大型数据集或在实际量子硬件上运行时。
- **目标核的选择：** 尽管$K_t = yy^T$是分类任务的常用标准，但其他目标核可能适用于不同任务（例如，回归）或纳入有关数据相似性的先验知识。
- **有限数据效应：** 对齐度是在有限数据集上计算的。其值是对底层数据分布的真实对齐度的经验估计。使用足够大且具有代表性的数据集是很有必要的。

研究核对齐的一个主要原因，是其与所得QML模型的**泛化性能**之间假设的关联。直观来看，在训练数据上能更好捕捉目标结构的核，在未见过的数据上也应表现更好。研究表明，较高的核对齐度通常与QSVM等模型的较低泛化误差相关联，使其成为衡量模型表现的一种有价值的指标。

总之，核对齐提供了一个有益的视角来分析量子特征映射所产生的几何结构。它量化 (quantization)了量子表示对给定学习任务的适用程度，并作为在QML中比较和设计有效数据编码策略的实用工具。理解特征映射设计、由此产生的特征空间几何结构与核对齐之间的关系，对于构建高性能量子机器学习 (machine learning)模型来说是不可或缺的。

## 参考资料

- [Supervised learning with quantum-enhanced feature spaces](https://www.nature.com/articles/s41586-019-0980-2) — Vojtech Havlicek, Antonio D. Córcoles, Kristan Temme, Aram W. Helwegen, Abhinav Kandala, and Jay M. Gambetta (2019)
  Journal: Nature; Publisher: Nature Portfolio; Volume: 567; Pages: 209-212; DOI: [10.1038/s41586-019-0980-2](https://doi.org/10.1038/s41586-019-0980-2)
  介绍了量子增强特征空间和量子核方法的概念，包括量子核的定义，为QSVM等算法奠定了基础。
- [Power of data in quantum machine learning](https://www.nature.com/articles/s41467-021-22539-9) — Hsin-Yuan Huang, Michael Broughton, Masoud Mohseni, Ryan Babbush, Hartmut Neven, Jarrod R. McClean, and Sergio Boixo (2021)
  Journal: Nature Communications; Volume: 12; Pages: 2631; DOI: [10.1038/s41467-021-22539-9](https://doi.org/10.1038/s41467-021-22539-9)
  讨论了数据编码和特征图设计对量子机器学习模型性能和泛化能力的影响，涉及核对齐相关概念。
- [Quantum Machine Learning in Feature Hilbert Spaces](https://doi.org/10.1103/PhysRevLett.122.040504) — Maria Schuld, Nathan Killoran (2019)
  Journal: Physical Review Letters; Publisher: American Physical Society; Volume: 122; Pages: 040504; DOI: [10.1103/PhysRevLett.122.040504](https://doi.org/10.1103/PhysRevLett.122.040504)
  为基于量子核和特征图的量子机器学习提供了理论框架，强调了数据表示的希尔伯特空间视角。

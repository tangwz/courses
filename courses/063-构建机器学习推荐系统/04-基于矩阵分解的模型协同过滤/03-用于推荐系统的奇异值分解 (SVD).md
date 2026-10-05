# 用于推荐系统的奇异值分解 (SVD)

来源：[原文](https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-4-model-based-collaborative-filtering/svd-for-recommendations)

[返回章节目录](README.md) · [返回课程目录](../README.md)

奇异值分解 (SVD) 是用于矩阵分解的一种基础线性代数技术。它提供了一种规范的方法，将任何矩阵分解为另外三个矩阵的乘积，从而展示其内在结构并帮助发现隐含特征。

### SVD 的数学原理

从形式上讲，SVD 表明任何大小为 $m \times n$ 的矩形矩阵 $R$（代表 $m$ 个用户和 $n$ 个物品）都可以分解为三个矩阵：


$$
R = U \Sigma V^T
$$


让我们分析每一个组成部分：

- **$U$**：一个 $m \times m$ 的正交矩阵。它的列被称为“左奇异向量 (vector)”。在推荐场景中，可以将这些向量视为在隐含特征空间中对用户的表达。
- **$\Sigma$ (Sigma)**：一个 $m \times n$ 的矩形对角矩阵。其对角线上的元素称为“奇异值”，代表每个隐含特征的“强度”或权重 (weight)。这些值始终为非负数，通常按降序排列。
- **$V^T$ (V 的转置)**：一个 $n \times n$ 的正交矩阵。它的行（即 $V$ 的列）被称为“右奇异向量”。这些向量可以理解为在同一个隐含特征空间中对物品的表达。

> 将用户-物品矩阵 $R$ 分解为三个不同的矩阵：$U$、$\Sigma$ 和 $V^T$。

### 针对稀疏数据调整 SVD

将这种“纯粹”形式的 SVD 直接应用于推荐问题时面临一个挑战：它要求矩阵是完整的，没有缺失值。而我们的用户-物品交互矩阵 $R$ 几乎总是稀疏的，这意味着大多数条目是未知的。如果我们用零来填充缺失项，算法会将其误认为实际评分为 0，这会使结果产生严重偏差。

因此，我们并不直接使用经典的 SVD 算法，而是采用受 SVD 启发的方法。这些方法旨在找到能够近似原始矩阵的因子矩阵，但**仅针对已知的评分**。此时的目标不再是完美地重构 $R$，而是找到最能体现已观测到的用户-物品交互的隐含特征矩阵 $P$（代表用户）和 $Q$（代表物品）。

在推荐系统文献中，这种改进方法通常仍被称为 SVD，尽管它在技术上是一种近似。其目的是找到 $P$ 和 $Q$，使得它们的乘积 $P \cdot Q^T$ 能够很好地近似 $R$。

### 利用截断 SVD 进行降维

SVD 在推荐系统中的真正作用来自于降维。$\Sigma$ 矩阵中的奇异值按重要性排序。第一个奇异值对应数据中最显著的模式，第二个对应次显著的模式，依此类推。许多靠后的奇异值通常很小，可以被视为噪声。

我们可以通过仅保留前 $k$ 个隐含特征来利用这一点，其中 $k$ 是一个远小于原始用户或物品数量的数值。这个过程被称为**截断 SVD (Truncated SVD)**。我们减小了矩阵的维度：

- $U$ 变为 $U_k$（$m \times k$ 矩阵）。
- $\Sigma$ 变为 $\Sigma_k$（$k \times k$ 矩阵）。
- $V^T$ 变为 $V_k^T$（$k \times n$ 矩阵）。

我们对评分矩阵的新近似值 $\hat{R}$ 为：


$$
\hat{R} = U_k \Sigma_k V_k^T
$$


这有两个主要好处：

1. **泛化能力**：通过关注 $k$ 个最重要的模式，模型捕捉的是内在的偏好结构，而不是死记硬背训练数据中的噪声细节。这有助于模型对未见过的物品做出更好的预测。
2. **效率**：生成的因子矩阵要小得多，使得存储和计算效率大幅提升。

> 使用截断矩阵对 $R$ 进行近似，其中 $k$ 是隐含特征的数量。这降低了维度并捕捉了最显著的模式。

### 根据隐含特征进行预测

一旦模型学习到了用户特征矩阵 $P$（类似于 $U_k \sqrt{\Sigma_k}$）和物品特征矩阵 $Q$（类似于 $(\sqrt{\Sigma_k} V_k^T)^T$），进行预测就变得非常直接。

每个用户 $u$ 由长度为 $k$ 的向量 (vector) $p_u$ 表示，每个物品 $i$ 由长度为 $k$ 的向量 $q_i$ 表示。预测评分 $\hat{r}_{ui}$ 就是这两个向量的点积：


$$
\hat{r}_{ui} = p_u \cdot q_i = \sum_{j=1}^{k} p_{uj}q_{ij}
$$


这个点积衡量了用户偏好与物品特征在学习到的隐含空间中的匹配程度。如果用户向量在某些特征上的值较高，而物品向量在相同特征上的值也较高，那么预测评分就会很高。

因此，主要任务是找到矩阵 $P$ 和 $Q$ 的值。由于我们无法在稀疏矩阵上使用经典 SVD，我们必须转向其他方法。我们将问题重新定义为一个优化任务：找到能够使已知评分的预测误差最小化的隐含特征。下一节将介绍如何使用名为随机梯度下降 (gradient descent)的迭代优化算法来实现这一目标。

## 参考资料

- [Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model](https://dl.acm.org/doi/10.1145/1401890.1401944) — Yehuda Koren, Robert Bell, and Chris Volinsky (2008)
  Journal: Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 426-434; DOI: [10.1145/1401890.1401944](https://doi.org/10.1145/1401890.1401944)
  描述了用于协同过滤和处理稀疏数据的矩阵分解技术（包括类SVD方法），这是为Netflix竞赛开发的。
- [Recommender Systems Handbook](https://link.springer.com/book/10.1007/978-1-0716-1738-4) — Francesco Ricci, Lior Rokach, and Bracha Shapira (2022)
  Publisher: Springer; DOI: [10.1007/978-1-0716-1738-4](https://doi.org/10.1007/978-1-0716-1738-4)
  提供了推荐系统的全面概述，其中包含关于矩阵分解和SVD变体的专门章节。
- [Mining of Massive Datasets](http://www.mmds.org/) — Jure Leskovec, Anand Rajaraman, Jeff Ullman (2020)
  Publisher: Cambridge University Press
  讨论SVD作为一种降维方法及其在推荐系统中的应用，并提供实例。

---

[上一节](02-%E7%9F%A9%E9%98%B5%E5%88%86%E8%A7%A3%E7%AE%80%E4%BB%8B.md) · [下一节](04-%E4%BD%BF%E7%94%A8%E9%9A%8F%E6%9C%BA%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%20%28SGD%29%20%E8%BF%9B%E8%A1%8C%E4%BC%98%E5%8C%96.md)

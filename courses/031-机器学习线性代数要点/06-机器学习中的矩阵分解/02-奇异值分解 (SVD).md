# 奇异值分解 (SVD)

来源：[原文](https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-6-matrix-decompositions-machine-learning/singular-value-decomposition-svd)

[返回章节目录](README.md) · [返回课程目录](../README.md)

奇异值分解 (SVD) 是最通用且信息丰富的矩阵分解方法之一。与主要用于方阵的特征分解不同，SVD 适用于*任意* $m \times n$ 的矩形矩阵。这种普遍性使其在多种机器学习 (machine learning)任务中表现出色，例如从降维到推荐系统。

SVD 的核心思想是将任意矩阵 $A$ 分解为三个矩阵的乘积：

$A = U \Sigma V^T$

我们来详细了解这些组成部分：

1. **$U$**: 一个 $m \times m$ 正交矩阵。其列向量 (vector) $u_1, u_2, ..., u_m$ 被称为 $A$ 的**左奇异向量**。正交意味着其列是标准正交的（单位长度且互相垂直），并且 $U^T U = U U^T = I_m$，其中 $I_m$ 是 $m \times m$ 单位矩阵。这些向量构成矩阵 $A$ 列空间（及其正交补空间）的标准正交基。
2. **$\Sigma$** (Sigma): 一个 $m \times n$ 矩形对角矩阵。这是包含 $A$ 的**奇异值**的矩阵，记作 $\sigma_i$。对角线元素 $\Sigma_{ii} = \sigma_i$ 是非负实数，通常按降序排列（$\sigma_1 \ge \sigma_2 \ge ... \ge \sigma_r > 0$），其中 $r$ 是矩阵 $A$ 的秩。$\Sigma$ 的所有非对角线元素均为零。如果 $m > n$，则底部 $m-n$ 行全为零。如果 $n > m$，则最右侧 $n-m$ 列全为零。奇异值表示沿奇异向量确定的主方向上的“重要性”或大小。
3. **$V^T$**: 一个 $n \times n$ 正交矩阵（$V$ 的转置）。$V$ 的列（或 $V^T$ 的行），$v_1, v_2, ..., v_n$，是 $A$ 的**右奇异向量**。与 $U$ 类似，$V$ 是正交的，意味着 $V^T V = V V^T = I_n$。这些向量构成矩阵 $A$ 行空间（及其正交补空间，即零空间）的标准正交基。

> SVD 将一个 $m \times n$ 矩阵 $A$ 分解为一个 $m \times m$ 正交矩阵 $U$，一个包含奇异值的 $m \times n$ 对角矩阵 $\Sigma$，以及一个 $n \times n$ 正交矩阵 $V$ 的转置。

SVD 对于任何矩阵的存在性是线性代数中的一项基本成果。它本质上说明，任何由矩阵 $A$ 表示的线性变换都可以看作是三个简单操作的连续组合：旋转或反射 ($V^T$)，接着是沿正交轴的缩放（乘以 $\Sigma$），再接着是另一次旋转或反射 ($U$)。

### 理解组成部分

- **奇异值 ($\sigma_i$)**: 这些值量化 (quantization)了矩阵 $A$ 沿奇异向量 (vector)所定方向对空间的拉伸或收缩程度。较大的奇异值对应着矩阵影响更大的方向。非零奇异值的数量等于矩阵 $A$ 的秩。
- **奇异向量 ($u_i$ 和 $v_i$)**: $U$ 的列（左奇异向量）构成与变换 $A$ 相关的输出空间（列空间）的标准正交基。$V$ 的列（右奇异向量）构成输入空间（行空间）的标准正交基。它们表示变换作用的主要方向。

虽然我们在此不详细探讨其存在的完整证明或具体的计算算法（因为 NumPy 等库可以高效处理这些），但理解这种结构非常必要。在接下来的部分中，我们将进一步阐述这种分解的几何意义，并了解它如何支持降维和数据压缩等强大应用。

## 参考资料

- [Introduction to Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/video-lectures/lecture-29-singular-value-decomposition/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press; Pages: Chapter 7: The Singular Value Decomposition (SVD)
  一本基础教科书，全面且易懂地阐释了SVD的数学原理。
- [Numerical Linear Algebra](https://epubs.siam.org/doi/book/10.1137/1.9781611971212) — Lloyd N. Trefethen and David Bau, III (1997)
  Publisher: SIAM; Pages: Chapter 4: The Singular Value Decomposition; DOI: [10.1137/1.9781611971212](https://doi.org/10.1137/1.9781611971212)
  从计算和数值角度严谨地处理SVD，包括其性质和算法。

---

[上一节](01-%E7%9F%A9%E9%98%B5%E5%88%86%E8%A7%A3%E7%AE%80%E4%BB%8B.md) · [下一节](03-SVD%20%E7%9A%84%E5%87%A0%E4%BD%95%E5%90%AB%E4%B9%89.md)

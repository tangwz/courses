# SVD在降维中的应用

来源：[原文](https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-6-matrix-decompositions-machine-learning/svd-dimensionality-reduction)

[返回章节目录](README.md) · [返回课程目录](../README.md)

奇异值分解提供了一种将任意矩阵 $A$ 分解为乘积 $A = U\Sigma V^T$ 的方法。这种分解不仅仅在数学上很精巧。它在简化数据方面，特别是对于降维，被证明非常有用。

其核心思想依赖于对角矩阵 $\Sigma$。$\Sigma$ 的对角线元素，记作 $\sigma_1, \sigma_2, \dots, \sigma_r$（其中 $r$ 是 $A$ 的秩），是 $A$ 的奇异值。按照惯例，这些奇异值按降序排列：$\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$。这些奇异值衡量了沿 $U$ 对应列（左奇异向量 (vector)）和 $V$ 对应列（右奇异向量）所定义方向上所捕获的重要程度或“能量”。

使用SVD进行降维的原理是只保留该分解中最重要的部分。具体来说，我们选择前 $k$ 个奇异值（其中 $k < r$）及其相关的奇异向量。我们丢弃剩余的 $r-k$ 个奇异值和向量，它们对应于数据变化最小的方向。

在数学上，我们通过截断矩阵 $U$、$\Sigma$ 和 $V$ 来形成 $A$ 的一个近似：

1. 保留 $U$ 的前 $k$ 列，形成 $U_k$（一个 $m \times k$ 矩阵）。
2. 保留 $\Sigma$ 的左上角 $k \times k$ 块，形成 $\Sigma_k$（一个对角线元素为 $\sigma_1, \dots, \sigma_k$ 的对角矩阵）。
3. 保留 $V$ 的前 $k$ 列（对应于 $V^T$ 的前 $k$ 行），形成 $V_k$。然后取其转置 $V_k^T$（一个 $k \times n$ 矩阵）。

这样， $A$ 的低秩近似由下式给出：

$A_k = U_k \Sigma_k V_k^T$

这个矩阵 $A_k$ 的秩为 $k$，并且是原始矩阵 $A$ 的*最佳*秩-$k$ 近似，因为它使差值 $\|A - A_k\|_F$ 的Frobenius范数最小化。这是一个基本性质，通常被称为Eckart-Young定理，尽管了解其结果比记住名称更重要：SVD为您提供了将矩阵 $A$ 压缩为秩-$k$ 表示的最佳方式。

### SVD如何进行降维

可以将原始的 $m \times n$ 矩阵 $A$ 看作表示 $m$ 个数据点，每个点具有 $n$ 个特征（反之亦然）。SVD为 $A$ 的行空间（通过 $V$）和列空间（通过 $U$）找到了新的正交基。转换 $A$ 实质上是将行空间基中的向量 (vector)映射到列空间基中，并按奇异值进行缩放。

- **大奇异值 ($\sigma_i$):** 表示数据展现出显著变异的维度（主要方向）。这些方向捕获了数据的主要结构。
- **小奇异值 ($\sigma_i$):** 表示数据变异很小的维度。这些方向可能代表噪声或不太重要的细节。

通过将较小的奇异值设为零（这实际上就是截断为 $A_k$ 所做的事情），我们将原始数据投影到由与最大奇异值相关联的主要方向所张成的低维子空间上。我们保留了最重要的变异，同时丢弃了信息较少的部分。

### 选择维度数量 $k$

保留的维度数量 $k$ 的选择取决于具体的应用。常见的方法包括：

1. **固定数量：** 选择预定的维度数量（例如，为了可视化目的降至 $k=2$ 或 $k=3$）。
2. **解释方差：** 计算由前 $k$ 个奇异值捕获的方差比例。总方差（平方和）与所有奇异值的平方和 $\sum_{i=1}^r \sigma_i^2$ 成比例。我们选择 $k$ 以使保留的方差达到期望的阈值（例如，90%、95%、99%）：
   $\frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^r \sigma_i^2} \ge \text{阈值}$
   绘制奇异值（或其平方的累积和）有助于直观地看到“拐点”出现的位置，这表示增加更多维度时回报递减。



[交互图表：奇异值衰减与累积方差](https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-6-matrix-decompositions-machine-learning/svd-dimensionality-reduction#plot-1dv2kl)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
      ],
      "y": [
        15.1,
        12.3,
        8.5,
        4.1,
        1.9,
        0.8,
        0.3,
        0.1,
        0.05,
        0.02
      ],
      "type": "scatter",
      "mode": "lines+markers",
      "name": "奇异值",
      "line": {
        "color": "#4263eb"
      },
      "marker": {
        "color": "#4263eb"
      }
    },
    {
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
      ],
      "y": [
        0.55,
        0.92,
        0.98,
        0.99,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
      ],
      "type": "scatter",
      "mode": "lines+markers",
      "name": "累积解释方差",
      "yaxis": "y2",
      "line": {
        "color": "#f76707"
      },
      "marker": {
        "color": "#f76707"
      }
    }
  ],
  "layout": {
    "title": "奇异值衰减与累积方差",
    "xaxis": {
      "title": "奇异值索引 (k)"
    },
    "yaxis": {
      "title": "奇异值大小",
      "titlefont": {
        "color": "#4263eb"
      },
      "tickfont": {
        "color": "#4263eb"
      }
    },
    "yaxis2": {
      "title": "累积解释方差 (%)",
      "overlaying": "y",
      "side": "right",
      "range": [
        0,
        1.05
      ],
      "titlefont": {
        "color": "#f76707"
      },
      "tickfont": {
        "color": "#f76707"
      }
    },
    "legend": {
      "x": 0.1,
      "y": -0.3,
      "orientation": "h"
    },
    "margin": {
      "l": 50,
      "r": 50,
      "t": 50,
      "b": 60
    }
  }
}
```

</details>



> 一张典型的图表，显示排序后的奇异值迅速下降，同时累积解释方差快速接近100%。这有助于选择一个合适的 $k$ 值。

### 与主成分分析 (PCA) 的联系

使用SVD进行降维与主成分分析 (PCA) 紧密关联。实际上，SVD提供了一种数值稳定计算数据集主成分的方法。如果数据矩阵 $A$ 的列是中心化的（减去均值），那么右奇异向量 (vector)（$V$ 的列）对应于主方向，并且奇异值的平方 ($\sigma_i^2$) 与每个主成分捕获的方差成比例。应用截断SVD $A_k = U_k \Sigma_k V_k^T$ 实际上是将数据投影到前 $k$ 个主成分上。

### 优点与不足

使用SVD进行降维提供了一些优点：

- **数据压缩：** 减少数据矩阵的存储需求。
- **降噪：** 较小的奇异值通常对应于噪声，这些噪声会被滤除。
- **计算加速：** 后续算法在降维数据上运行更快。
- **可视化：** 允许将高维数据投影到2或3个维度上。

然而，也存在一些需要考虑的方面：

- **信息丢失：** 降维本质上是会造成信息损失的。如果 $k$ 选择得太小，重要信息可能会被丢弃。
- **可解释性：** 新的维度（原始特征的组合）可能更难直接解释。
- **计算开销：** SVD本身的计算对于非常大的矩阵而言可能计算密集，尽管在NumPy (`numpy.linalg.svd`) 和 SciPy (`scipy.linalg.svd`) 等库中存在高效算法。

尽管分解本身的计算开销较大，SVD仍然是许多机器学习 (machine learning)流程中降低数据集复杂度的基本方法，它使得处理更高效，并且有时通过滤除噪声来提高模型的泛化能力。

## 参考资料

- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  对SVD进行了易于理解而全面的介绍，阐释了其性质和几何意义。
- [Matrix Computations](https://doi.org/10.1137/1.9781421407944) — Gene H. Golub, Charles F. Van Loan (2013)
  Publisher: Johns Hopkins University Press; Pages: XXI + 756; DOI: [10.1137/1.9781421407944](https://doi.org/10.1137/1.9781421407944)
  数值线性代数的权威著作，详细阐述了SVD及其Eckart-Young定理。
- [Mathematics for Machine Learning](https://mml-book.github.io/) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press
  在机器学习背景下解释SVD，侧重于其在降维和与PCA的联系中的应用。

---

[上一节](03-SVD%20%E7%9A%84%E5%87%A0%E4%BD%95%E5%90%AB%E4%B9%89.md) · [下一节](05-SVD%E5%9C%A8%E6%95%B0%E6%8D%AE%E5%8E%8B%E7%BC%A9%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8.md)

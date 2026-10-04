# K-Means 算法

来源：[原文](https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-5-unsupervised-learning-clustering/k-means-algorithm)

[返回章节目录](README.md) · [返回课程目录](../README.md)

聚类的目标是将相似的数据点归为一类。完成此任务最常用的算法之一是K-Means。它之所以受欢迎，是因为它相对易于理解和实现，并且通常能给出不错的结果。

K-Means 的核心思想是将数据集划分为预设数量的簇，用字母 $K$ 表示。该算法旨在使这些簇尽可能地明确和紧凑。它通过迭代地找到簇的中心点，称为**质心**，并将每个数据点分配给最近的质心来做到这一点。

可以把质心想作属于特定簇的所有点的几何中心（或算术平均值）。K-Means 算法通过一个迭代过程进行：

### K-Means 算法步骤

1. **初始化**：首先，你必须决定要找出多少个簇，这就是 $K$ 的值。然后，算法会初始化 $K$ 个质心。一种常见的方法是从数据集中随机选择 $K$ 个数据点，并将它们指定为初始质心。还有其他初始化方法，但随机选择是一个直接的起点。
2. **分配步骤**：对于数据集中的*每个*数据点，计算它到*每个* $K$ 个质心的距离。将数据点分配给其质心最近（最接近）的簇。这里最常用的距离测量方法是标准欧几里得距离（你可以用尺子测量的直线距离），但其思想就是“最近的质心”。这一步之后，每个数据点都属于 $K$ 个簇中的一个。
3. **更新步骤**：重新计算每个 $K$ 个质心的位置。质心的新位置是上一步中分配给其簇的所有数据点的平均位置。如果一个簇最终没有数据点分配给它（这种情况可能发生，特别是在初始化不佳时），质心可能会被移除或随机重新分配，但通常它会保持原位，直到在后续迭代中分配到点。
4. **迭代**：重复分配步骤和更新步骤，直到满足停止条件。常见的停止条件有：

   - 质心在迭代之间不再显著移动。
   - 数据点停止改变簇的分配。
   - 已达到预设的最大迭代次数。

当算法停止时，质心代表最终簇的中心，并且每个数据点都属于与其最近的最终质心相关的簇。

### K-Means 可视化

想象一下你在二维图上有一些分散的数据点，你选择 $K=3$。

1. K-Means 首先随机放置三个质心。
2. **分配：** 它在每个质心周围绘制虚拟边界；所有最接近质心 1 的点归入簇 1，最接近质心 2 的点归入簇 2，依此类推。
3. **更新：** 算法计算现在在簇 1 中的所有点的平均位置，并将质心 1 移动到该位置。它对质心 2 和 3 也做同样的操作。
4. **重复：** 因为质心移动了，边界也随之改变。现在点可能更接近*不同的*质心。所以，算法重新分配点，然后再次更新质心位置。这个过程持续进行，直到质心稳定下来。

下面图表显示了在一些二维样本数据上运行 K-Means（$K=3$）后可能的最终状态。点根据其分配的簇着色，较大的标记 (token)表示质心的最终位置。



[交互图表：K-Means 聚类结果 (K=3)](https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-5-unsupervised-learning-clustering/k-means-algorithm#plot-dzybb9)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        1.1,
        1.5,
        1.8,
        2.1,
        2.5,
        5.8,
        6.2,
        6.5,
        6.9,
        7.2,
        4.0,
        4.3,
        3.8,
        4.6,
        4.9
      ],
      "y": [
        2.0,
        1.8,
        2.3,
        1.6,
        2.1,
        7.0,
        6.5,
        7.3,
        6.8,
        7.5,
        3.8,
        4.5,
        4.1,
        4.8,
        4.3
      ],
      "mode": "markers",
      "type": "scatter",
      "marker": {
        "color": [
          "#4263eb",
          "#4263eb",
          "#4263eb",
          "#4263eb",
          "#4263eb",
          "#f76707",
          "#f76707",
          "#f76707",
          "#f76707",
          "#f76707",
          "#37b24d",
          "#37b24d",
          "#37b24d",
          "#37b24d",
          "#37b24d"
        ],
        "size": 8
      },
      "name": "数据点"
    },
    {
      "x": [
        1.8,
        6.52,
        4.32
      ],
      "y": [
        1.96,
        7.02,
        4.3
      ],
      "mode": "markers",
      "type": "scatter",
      "marker": {
        "color": [
          "#1c7ed6",
          "#fd7e14",
          "#40c057"
        ],
        "size": 18,
        "symbol": "cross",
        "line": {
          "width": 2,
          "color": "#495057"
        }
      },
      "name": "质心"
    }
  ],
  "layout": {
    "title": {
      "text": "K-Means 聚类结果 (K=3)"
    },
    "xaxis": {
      "title": "特征 1"
    },
    "yaxis": {
      "title": "特征 2"
    },
    "showlegend": false,
    "width": 600,
    "height": 400,
    "margin": {
      "l": 50,
      "r": 20,
      "t": 50,
      "b": 40
    },
    "template": "plotly_white"
  }
}
```

</details>



> 使用 K-Means 将数据集划分为三个簇的示例。每种颜色表示一个不同的簇，叉号标记了最终质心的位置。

### K-Means 优化目标

在内部，K-Means 试图最小化一个目标函数，称为**簇内平方和 (WCSS)**。这听起来很专业，但其原理很简单：它衡量的是每个数据点与其所属簇的质心之间的总平方距离。通过最小化 WCSS，K-Means 试图使簇尽可能地紧凑（点靠近其质心）和分离。分配和更新步骤的每次迭代通常会减少 WCSS，从而得到一个好的（尽管不总是绝对最佳的）聚类方案。

K-Means 是一种用于在没有标签时划分数据的基本算法。它的简单性和速度使其成为许多聚类任务的理想首选。然而，正如我们接下来将看到的，选择正确的 $K$ 值以及了解算法的局限性是重要的考量。

## 参考资料

- [Some Methods for Classification and Analysis of Multivariate Observations](https://projecteuclid.org/journals/proceedings-of-the-berkeley-symposium-on-mathematical-statistics-and-probability/volume-5/issue-1/Some-methods-for-classification-and-analysis-of-multivariate-observations/bsmsp/1103842941.full) — J. B. MacQueen (1967)
  Journal: Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, Volume 1: Statistics; Publisher: University of California Press; Volume: 1; Pages: 281-297
  介绍了用于寻找聚类中心的迭代算法，该算法后来被称为K-Means。
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: 738
  对K-Means及相关聚类算法进行了全面且基于概率的论述。
- [CS229 Lecture Notes: Unsupervised Learning](http://cs229.stanford.edu/notes/cs229-notes7a.pdf) — Andrew Ng (2008)
  Publisher: Stanford University
  作为一种基础的无监督学习算法，对K-Means进行了清晰的解释。
- [k-means++: The Advantages of Careful Seeding](https://dl.acm.org/doi/10.5555/1283383.1283494) — David Arthur, Sergei Vassilvitskii (2007)
  Journal: Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms; Publisher: Society for Industrial and Applied Mathematics; Pages: 1027-1035; DOI: [10.5555/1283383.1283494](https://doi.org/10.5555/1283383.1283494)
  介绍了一种改进的K-Means初始化方法，以获得更好的聚类结果并加快收敛速度。
- [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) — scikit-learn developers (2024)
  提供了scikit-learn中K-Means实现的实际用法、参数和属性的官方文档。

---

[上一节](02-%E8%81%9A%E7%B1%BB%E7%AE%80%E4%BB%8B.md) · [下一节](04-%E9%80%89%E6%8B%A9%E8%81%9A%E7%B1%BB%E6%95%B0%E9%87%8F%20%28K%29.md)

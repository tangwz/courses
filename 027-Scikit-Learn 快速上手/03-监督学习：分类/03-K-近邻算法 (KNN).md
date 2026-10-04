# K-近邻算法 (KNN)

来源：[原文](https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-3-supervised-learning-classification/knn-algorithm)

[返回章节目录](README.md) · [返回课程目录](../README.md)

K-近邻 (KNN) 算法是一种分类方法，以其简单性和出人意料的有效性而闻名。它是一种基于实例或非参数 (parameter)学习器，代表着与传统线性模型不同的算法家族。

与通过估计参数（如逻辑回归中的系数）学习明确决策边界的模型不同，KNN 基于相似性原则运行。其核心是，KNN 根据特征空间中其 'K' 个最近邻居的多数类别来分类新的数据点。

### KNN 的工作方式：基于接近度的分类

设想你有一个数据集，绘制在一个多维空间中，其中每个轴表示一个特征。这个空间中的每个点都属于一个已知类别。当一个新的未分类数据点出现时，KNN 遵循以下步骤：

1. **计算距离：** 它计算新点与训练数据集中*每个*现有数据点之间的距离。最常用的距离度量是欧几里得距离，尽管也可以使用曼哈顿距离等其他度量。对于 $n$ 维特征空间中的两个点 $p = (p_1, p_2, ..., p_n)$ 和 $q = (q_1, q_2, ..., q_n)$，欧几里得距离为：
   
   $$
   d(p, q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + ... + (p_n - q_n)^2}
   $$
   
2. **识别邻居：** 它识别训练集中距离新点最近（距离最小）的 'K' 个数据点。这些就是它的 'K' 个最近邻居。
3. **多数表决：** 它查看这些 'K' 个邻居的类别。新的数据点被分配到其邻居中的多数类别。例如，如果 K=5，且 3 个邻居属于类别 A，2 个属于类别 B，则新点被分类为类别 A。如果出现平局（例如 K=4，其中 2 个邻居属于类别 A，2 个属于类别 B），通常会随机打破平局或通过考虑距离来解决。

以下可视化说明了 K=3 时的这一点。



[交互图表：KNN 分类示例 (K=3)](https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-3-supervised-learning-classification/knn-algorithm#plot-7dwph4)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "KNN 分类示例 (K=3)",
    "xaxis": {
      "title": "特征 1",
      "range": [
        0,
        9
      ]
    },
    "yaxis": {
      "title": "特征 2",
      "range": [
        0,
        9
      ]
    },
    "legend": {
      "title": {
        "text": "类别"
      }
    },
    "annotations": [
      {
        "x": 5,
        "y": 5,
        "text": "新点",
        "showarrow": true,
        "arrowhead": 1,
        "ax": 0,
        "ay": -40
      }
    ]
  },
  "data": [
    {
      "x": [
        2,
        3,
        4,
        6,
        7,
        8
      ],
      "y": [
        2,
        4,
        6,
        7,
        3,
        5
      ],
      "mode": "markers",
      "type": "scatter",
      "name": "类别 A",
      "marker": {
        "color": "#4dabf7",
        "size": 10
      }
    },
    {
      "x": [
        1,
        3,
        5,
        6,
        8,
        7
      ],
      "y": [
        5,
        7,
        8,
        2,
        1,
        8
      ],
      "mode": "markers",
      "type": "scatter",
      "name": "类别 B",
      "marker": {
        "color": "#f06595",
        "size": 10
      }
    },
    {
      "x": [
        5
      ],
      "y": [
        5
      ],
      "mode": "markers",
      "type": "scatter",
      "name": "新点",
      "marker": {
        "color": "#fab005",
        "size": 12,
        "symbol": "star"
      }
    },
    {
      "x": [
        4,
        6,
        3
      ],
      "y": [
        6,
        7,
        7
      ],
      "mode": "markers",
      "type": "scatter",
      "name": "最近邻居 (K=3)",
      "marker": {
        "color": "rgba(0,0,0,0)",
        "size": 14,
        "line": {
          "color": "#74b816",
          "width": 2
        }
      }
    }
  ]
}
```

</details>



> 新的黄色星形点需要分类。其 3 个最近邻居（绿色圈出）被识别。其中两个邻居属于类别 A（蓝色），一个邻居属于类别 B（粉色）。通过多数表决（2 对 1），新点被分类为类别 A。

### K 值选择

邻居的数量 'K' 是一个超参数 (parameter) (hyperparameter)，它显著影响模型的行为。

- **小 'K' (例如, K=1)：** 模型对噪声和异常值非常敏感。决策边界可能高度不规则，可能导致高方差和过拟合 (overfitting)。
- **大 'K'：** 模型变得更平滑，对局部变化不那么敏感。决策边界变得更通用。然而，如果 'K' 过大（例如，K 等于数据点总数），模型可能会变得过于偏颇，并预测所有新点为多数类别，导致高偏差和欠拟合 (underfitting)。

选择最佳 'K' 通常需要实验，并使用交叉验证（我们将在第 5 章介绍）等方法来评估不同 'K' 值下的性能。在二分类问题中，通常建议选择奇数 'K' 值以避免平局。

### 特征缩放的重要性

使用 KNN 的一个重要方面是它对特征尺度的敏感性。由于 KNN 依赖于距离计算，数值范围较大的特征可能会不成比例地主导距离度量，从而有效地减弱数值范围较小特征的影响。例如，如果一个特征的范围是 0 到 1000，而另一个特征的范围是 0 到 1，那么第一个特征将几乎完全决定距离计算。

因此，在应用 KNN 之前，**几乎总是需要对你的特征进行缩放**。常见的缩放技术包括使用 `StandardScaler` 进行标准化（缩放到零均值和单位方差），或使用 `MinMaxScaler` 进行归一化 (normalization)（缩放到一个范围，通常是 [0, 1]）。我们将在第 4 章：数据预处理和特征工程中详细介绍这些技术。

### KNN 的优点和缺点

**优点：**

- **简单性：** 该算法易于理解和实现。
- **非参数 (parameter)性：** 它对底层数据分布没有强假设。这使得它能够学习复杂的决策边界。
- **适应性：** 当新的训练数据可用时，模型可以轻松适应，无需从头开始重新训练（尽管查找邻居的计算量会变得更大）。

**缺点：**

- **计算成本：** 计算与所有训练点的距离可能很慢且内存密集，特别是对于大型数据集（样本或特征数量多）。这通常被称为“维度灾难”。
- **对不相关特征的敏感性：** 无助于区分类别的特征会降低性能，因为它们仍然参与距离计算。特征选择或降维技术可能会有所帮助。
- **对特征缩放的敏感性：** 如前所述，性能严重依赖于适当的特征缩放。
- **需要最佳 'K'：** 性能对 'K' 的选择很敏感。

KNN 提供了一种基于实例相似性的根本不同的分类方法。在下一节中，我们将介绍如何使用 Scikit-learn 实现 `KNeighborsClassifier` 并将其应用于实际分类问题。

## 参考资料

- [sklearn.neighbors.KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) — scikit-learn developers (2023)
  Scikit-learn K近邻分类器官方文档，详细说明其参数、方法和使用。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  本书为KNN提供了统计和理论基础，涵盖其特性、性能以及维度灾难。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  本书提供了使用Scikit-learn实现和调整KNN的实用指导，包括超参数选择和特征缩放的讨论。

---

[上一节](02-%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%94%A8%E4%BA%8E%E5%88%86%E7%B1%BB.md) · [下一节](04-%E5%9C%A8%20Scikit-learn%20%E4%B8%AD%E5%AE%9E%E7%8E%B0%20KNN.md)

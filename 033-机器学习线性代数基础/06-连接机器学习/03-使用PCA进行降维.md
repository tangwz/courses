# 使用PCA进行降维

来源：[原文](https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-6-linear-algebra-in-ml/dimensionality-reduction-pca)

[返回章节目录](README.md) · [返回课程目录](../README.md)

设想你的数据集有数百个特征。例如，一个房地产数据集可能包括面积、房间数量、建筑年龄、到最近学校的距离、当地犯罪率以及几十个其他变量。虽然更多数据可能有帮助，但拥有过多特征或维度，会使机器学习 (machine learning)模型难以有效学习。这会增加计算时间，并可能导致一个被称为“维度灾难”的问题，即模型从噪声中学习，而非数据中的潜在信息。

为了处理具有大量特征的数据并提升机器学习模型的效率，降维技术是不可或缺的。其目标是减少特征数量，同时尽可能保留数据集中主要的信息。主成分分析（PCA）是实现这一目的最常用且有效的方法之一。

### PCA 的作用

本质上，PCA是一种将数据转换为一组新特征（称为**主成分**）的方法。这些新成分按照它们捕获原始数据方差的多少进行排序。第一个主成分（PC1）旨在捕获尽可能大的方差。第二个主成分（PC2）捕获次大的方差，其条件是它必须与第一个主成分正交（垂直）。所有成分都依此进行。

这个过程会为你提供一个排列好的成分列表。为了减少维度，你只需保留前几个捕获了大部分信息的成分，并丢弃其余的。

### 特征向量 (vector)和特征值的作用

我们第5章学过的特征向量和特征值在这里变得非常有用。PCA通过分析数据集中特征之间的关系来运作。这种关系由一个称为**协方差矩阵**的矩阵表示。

这个协方差矩阵的特征向量指向数据中方差最大的方向。实际上，这些特征向量*就是*主成分。具有最大对应特征值的特征向量是第一个主成分，因为它指向数据中“散布”最大的方向。具有次大特征值的特征向量是第二个主成分，依此类推。

特征值本身说明了每个主成分捕获的方差量。较大的特征值表示其对应的特征向量（和主成分）很有意义。

### 视觉示例

让我们看一个简单的二维数据集。设想将两个特征相互绘制，数据点形成一个细长的云团。



[交互图表：二维数据集的主成分](https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-6-linear-algebra-in-ml/dimensionality-reduction-pca#plot-zcbpdq)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "width": 600,
    "height": 400,
    "xaxis": {
      "title": "特征 1",
      "range": [
        -4,
        4
      ]
    },
    "yaxis": {
      "title": "特征 2",
      "range": [
        -4,
        4
      ]
    },
    "title": "二维数据集的主成分",
    "annotations": [
      {
        "ax": 0,
        "ay": 0,
        "axref": "x",
        "ayref": "y",
        "x": 3.1,
        "y": 1.5,
        "xref": "x",
        "yref": "y",
        "showarrow": true,
        "arrowhead": 2,
        "arrowcolor": "#f03e3e",
        "arrowwidth": 3
      },
      {
        "ax": 0,
        "ay": 0,
        "axref": "x",
        "ayref": "y",
        "x": -0.7,
        "y": 1.5,
        "xref": "x",
        "yref": "y",
        "showarrow": true,
        "arrowhead": 2,
        "arrowcolor": "#fd7e14",
        "arrowwidth": 3
      }
    ],
    "legend": {
      "x": 1.05,
      "y": 1
    }
  },
  "data": [
    {
      "x": [
        -2.5,
        -2,
        -1.8,
        -1.5,
        -1.4,
        -1,
        -0.8,
        -0.5,
        0,
        0.2,
        0.5,
        0.8,
        1,
        1.4,
        1.5,
        1.8,
        2,
        2.5
      ],
      "y": [
        -1.3,
        -1,
        -0.9,
        -0.8,
        -0.7,
        -0.5,
        -0.4,
        -0.3,
        0,
        0.1,
        0.3,
        0.4,
        0.5,
        0.7,
        0.8,
        0.9,
        1,
        1.3
      ],
      "mode": "markers",
      "type": "scatter",
      "name": "数据点",
      "marker": {
        "color": "#4263eb"
      }
    },
    {
      "x": [
        3.2,
        -0.7
      ],
      "y": [
        1.6,
        1.6
      ],
      "mode": "text",
      "text": [
        "PC1",
        "PC2"
      ],
      "textfont": {
        "color": [
          "#f03e3e",
          "#fd7e14"
        ],
        "size": 14
      },
      "type": "scatter",
      "showlegend": false
    }
  ]
}
```

</details>



> 主成分（红色和橙色箭头）标识了数据中方差最大的轴。PC1捕获了最大的散布，而PC2捕获了次大的散布。

在此图中，你可以看到数据沿红色箭头（PC1）方向变化最大。沿橙色箭头（PC2）方向的变化则小得多。如果我们要将此数据集从二维降至一维，我们可以将所有数据点投影到由PC1定义的线上。我们将丢失与PC2相关的信息，但由于PC1捕获了大部分方差，我们将保留数据最主要的结构。

### PCA 的步骤

以下是执行PCA的一个高层总结：

1. **标准化数据：** 调整数据，使每个特征的均值为0，标准差为1。这确保了具有较大尺度的特征不会在分析中占据主导地位。
2. **计算协方差矩阵：** 计算协方差矩阵，以了解数据集中不同特征之间如何相互变化。
3. **找到特征值和特征向量 (vector)：** 计算协方差矩阵的特征值和特征向量。正如我们所说，这是该过程的核心。
4. **排序并选择成分：** 根据其对应的特征值，将特征向量按降序排列。具有最高特征值的特征向量是第一个主成分。然后你决定保留多少成分。一个常见做法是保留足够的成分来解释总方差的某个百分比，例如95%。
5. **转换数据：** 使用选定的特征向量将原始数据转换为新的、低维空间。结果是一个特征较少但保留了大部分原始信息的新数据集。

通过应用PCA，我们利用线性代数的基本原理，特别是特征值和特征向量，使数据量大的数据集更易于处理。这使得它们更容易可视化、处理速度更快，并且通过去除噪声，通常能在机器学习 (machine learning)模型中带来更好的表现。

## 参考资料

- [Principal Component Analysis](http://dx.doi.org/10.1007/b98835) — I. T. Jolliffe (2002)
  Publisher: Springer New York; Pages: 487 p.; DOI: [10.1007/b98835](https://doi.org/10.1007/b98835)
  这本权威著作全面介绍了主成分分析的理论、方法和应用。
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/10.1007/978-0-387-45528-0) — Christopher M. Bishop (2006)
  Publisher: Springer; DOI: [10.1007/978-0-387-45528-0](https://doi.org/10.1007/978-0-387-45528-0)
  这是一本标准教材，在机器学习背景下，对主成分分析提供了详尽的统计和概率处理。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  这本统计学习经典著作有专门章节，解释了主成分分析作为一种降维技术。第2版，提供免费PDF。
- [Linear Algebra and Learning from Data](https://math.mit.edu/~gs/learningfromdata/) — Gilbert Strang (2019)
  Publisher: Wellesley-Cambridge Press
  该书将核心线性代数概念，包括特征向量和特征值，直接与它们在机器学习中的应用（如主成分分析）联系起来。作者官方页面。

---

[上一节](02-%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E7%9A%84%E7%9F%A9%E9%98%B5%E9%97%AE%E9%A2%98.md) · [下一节](04-%E4%BD%BF%E7%94%A8%E7%82%B9%E7%A7%AF%E8%A1%A1%E9%87%8F%E7%9B%B8%E4%BC%BC%E5%BA%A6.md)

# 支持向量机 (SVM) 基本原理

来源：[原文](https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-3-supervised-learning-classification/svm-basics)

[返回章节目录](README.md) · [返回课程目录](../README.md)

支持向量 (vector)机 (SVM) 为分类问题提供了一种强大且通用的方法。与逻辑回归等主要侧重于寻找决策边界的算法不同，SVM 采用了一种独特策略：它们旨在识别不同类别之间最优的决策线或超平面。这个最优超平面被定义为到任一类别最近数据点的距离（即间隔）最大化的那个。

### 最大间隔分类器

设想你有一些数据点，它们属于二维平面上的两个不同类别。你想要画一条直线来分离这两个类别。你可能会画出许多条可能的线。哪一条是最佳的呢？

SVM 通过寻找一条尽可能远离*两个*类别中最近的点的线来回答这个问题。这条分离线（超平面）到最近点的距离称为**间隔**。产生最大间隔的超平面被认为是最佳的。这通常被称为**最大间隔超平面**。

为什么要最大化间隔？从直观上看，较大的间隔表明类别之间有更可靠的分离。具有大间隔的决策边界对数据中的微小扰动不太敏感，并且可能在新数据上表现更好。

通常，对于一个 $n$ 维特征空间，分隔器是一个 $(n-1)$ 维**超平面**。

- 在二维空间中，超平面是一条线。
- 在三维空间中，超平面是一个平面。
- 在更高维度中，它很难被可视化，但在数学上表示一个平坦的子空间。

### 支持向量 (vector)

最接近最大间隔超平面（正好在间隔边缘上）的数据点被称为**支持向量**。它们是数据集中重要的数据点，因为它们“支撑”或确定了超平面。如果你移动任何一个支持向量，最佳超平面的位置可能会改变。相反，移动*不是*支持向量（并且远离间隔）的数据点不会影响超平面，只要它们不越过间隔边界。这个特性使得 SVM 内存高效，特别是在高维空间 (high-dimensional space)中，因为模型定义只依赖于这些少数支持向量。

我们来将这个想法在二维空间中呈现：



[交互图表：SVM：超平面、间隔和支持向量](https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-3-supervised-learning-classification/svm-basics#plot-5ocu0x)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        1,
        1.5,
        2,
        2.5,
        1.8,
        2.2
      ],
      "y": [
        1.5,
        2.5,
        1,
        1.8,
        0.8,
        2.9
      ],
      "mode": "markers",
      "type": "scatter",
      "name": "类别 A",
      "marker": {
        "color": "#4263eb",
        "size": 10
      }
    },
    {
      "x": [
        4,
        4.5,
        5,
        5.5,
        4.2,
        4.8
      ],
      "y": [
        3.5,
        4.5,
        3,
        4.2,
        3.2,
        4.9
      ],
      "mode": "markers",
      "type": "scatter",
      "name": "类别 B",
      "marker": {
        "color": "#f76707",
        "size": 10
      }
    },
    {
      "x": [
        2.5,
        4
      ],
      "y": [
        1.8,
        3.2
      ],
      "mode": "markers",
      "type": "scatter",
      "name": "支持向量",
      "marker": {
        "color": [
          "#4263eb",
          "#f76707"
        ],
        "size": 12,
        "symbol": "circle-open",
        "line": {
          "width": 2
        }
      }
    }
  ],
  "layout": {
    "title": "SVM：超平面、间隔和支持向量",
    "xaxis": {
      "title": "特征 1",
      "range": [
        0,
        6.5
      ],
      "zeroline": false
    },
    "yaxis": {
      "title": "特征 2",
      "range": [
        0,
        5.5
      ],
      "zeroline": false
    },
    "shapes": [
      {
        "type": "line",
        "x0": 0.5,
        "y0": -0.5,
        "x1": 6,
        "y1": 6.2,
        "line": {
          "color": "#495057",
          "width": 2
        },
        "name": "超平面"
      },
      {
        "type": "line",
        "x0": 0.2,
        "y0": 0.8,
        "x1": 4.8,
        "y1": 6.0,
        "line": {
          "color": "#adb5bd",
          "width": 1,
          "dash": "dash"
        },
        "name": "间隔 A"
      },
      {
        "type": "line",
        "x0": 1.4,
        "y0": -1.1,
        "x1": 6.5,
        "y1": 5.0,
        "line": {
          "color": "#adb5bd",
          "width": 1,
          "dash": "dash"
        },
        "name": "间隔 B"
      }
    ],
    "legend": {
      "yanchor": "top",
      "y": 0.99,
      "xanchor": "left",
      "x": 0.01
    },
    "width": 600,
    "height": 450
  }
}
```

</details>



> 这个图表展示了两类数据点。实线表示由 SVM 找到的最大间隔超平面。虚线指示了间隔边界，而位于这些虚线上的点是支持向量（用空心圆圈高亮显示）。

### 处理非线性边界：核技巧

上述想法在数据**线性可分**时表现良好，这意味着你可以画一条直线（或超平面）来完美分离这些类别。但如果数据不是线性可分的怎么办？

考虑呈同心圆排列的数据。没有一条直线可以分离内圈和外环。SVM 使用一种巧妙的技术来处理这个问题，通常称为**核技巧**。

核心思路是将原始特征转换到一个更高维度的空间，在那里数据*可能*变得线性可分。想象将二维同心圆数据投影到三维空间中，使得内圈点的高度与外环点不同。在这个新的三维空间中，你可能可以用一个简单的平面（三维空间中的超平面）来分离这些类别。

核函数计算数据点在这个可能非常高维的特征空间中的点积，而无需实际计算该空间中点的坐标。这使得计算过程高效。常见核函数包括：

- **线性核函数：** 对应于前面描述的最大间隔分类器（无转换）。
- **多项式核函数：** 将数据映射到包含原始特征多项式组合的空间。
- **径向基函数 (RBF) 核函数（或高斯核函数）：** 将数据映射到无限维空间。这是一个非常常用且通常有效的默认选择。

通过使用核函数，SVM 可以学习复杂的非线性决策边界。

支持向量 (vector)机在高维空间 (high-dimensional space)中特别有效（特征数量很多，甚至多于样本数量），并且由于它们依赖于支持向量而内存高效。下一节将详细说明如何使用 Scikit-learn 的 `SVC`（支持向量分类器）估计器实现 SVM 分类器，使你能够实际运用这些知识。

## 参考资料

- [Support-vector networks](https://link.springer.com/article/10.1007/BF00994018) — Corinna Cortes and Vladimir Vapnik (1995)
  Journal: Machine Learning; Volume: 20; Pages: 273-297; DOI: [10.1007/BF00994018](https://doi.org/10.1007/BF00994018)
  这篇原始论文介绍了支持向量机，阐述了最大间隔超平面和核心理论框架。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; Pages: Chapter 12: Support Vector Machines and Other Kernel Methods
  这是一本权威教科书，其中有一章专门介绍支持向量机，详细解释了最大间隔分类器和核方法。
- [1.4. Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html) — Scikit-learn developers (2023)
  Scikit-learn中支持向量机的官方用户指南，提供了实现细节和不同核函数选择的实用信息。
- [CS229 Lecture Notes: Support Vector Machines](http://cs229.stanford.edu/notes/cs229-notes3.pdf) — Andrew Ng (2018)
  来自知名大学课程的综合讲义，提供了支持向量机概念的清晰推导和解释，包括最大间隔分类器和核函数。

---

[上一节](04-%E5%9C%A8%20Scikit-learn%20%E4%B8%AD%E5%AE%9E%E7%8E%B0%20KNN.md) · [下一节](06-%E4%BD%BF%E7%94%A8Scikit-learn%E5%AE%9E%E7%8E%B0SVM.md)

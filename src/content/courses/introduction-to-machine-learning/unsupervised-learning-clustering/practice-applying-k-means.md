---
course: "introduction-to-machine-learning"
chapter: "unsupervised-learning-clustering"
lesson: "practice-applying-k-means"
sourceId: 1520
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-5-unsupervised-learning-clustering/practice-applying-k-means"
title: "动手操作：K-Means 在简单数据上的应用"
description: "实践环节，使用 K-Means 算法对数据点进行分组。"
order: 7
plots: ["plots/1520-0.json", "plots/1520-1.json"]
sourceHash: "70663fc6bfb9bbd6d86cd99c0fb1ac28645ebe694075f0f977dc3bf7254f8fde"
sourceCorrections: []
---

K-Means 算法是一种用于无监督学习 (supervised learning) (unsupervised learning)和聚类的技术。其目的是在不预先知道数据点“真实”标签的情况下，仅依据数据点自身的位置对其进行分组。我们将逐步演示如何将 K-Means 应用于一个简单数据集，以观察其实际效果。

在入门示例中，我们通常采用一种方法：生成模拟数据。这很有用，因为我们可以创建已知存在明显分组的数据，从而更容易通过视觉检查 K-Means 在发现这些分组方面是否做得合理。我们将使用 Python，并配合 Scikit-learn（用于 K-Means 算法）和 Plotly（用于可视化）等常用库。

### 准备工具

首先，请确保您已安装所需的库。如果您在 Google Colab 或 Anaconda 等环境中工作，这些库可能已预装。否则，您通常可以使用 pip 进行安装：

```bash
pip install scikit-learn numpy plotly
```

本示例中，我们需要 `numpy` 进行数值运算（特别是生成数据），`sklearn.cluster` 用于 `KMeans` 算法，以及 `plotly.graph_objects` 用于创建适合网页的交互式图表。

### 生成和可视化简单数据

我们来创建一些二维数据，这些数据明显分为三个组或“斑点”。Scikit-learn 提供了一个方便的 `make_blobs` 函数，正是为此目的而设。

```python
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_blobs

# 生成包含3个明显聚类的模拟数据
X, _ = make_blobs(n_samples=150,  # 总点数
                  centers=3,      # 要生成的聚类数量
                  cluster_std=0.8,# 聚类的标准差（分散程度）
                  random_state=42)# 用于结果复现
                                  # 我们忽略第二个输出 (y)，即真实标签

# X 现在是一个 NumPy 数组，包含 150 行和 2 列（我们的特征）

# 让我们在聚类之前显示原始数据
fig_raw = go.Figure(data=[go.Scatter(
    x=X[:, 0],
    y=X[:, 1],
    mode='markers',
    marker=dict(color='#495057', size=7, opacity=0.8) # 原始数据使用灰色
)])

fig_raw.update_layout(
    title='模拟数据点（聚类前）',
    xaxis_title='特征 1',
    yaxis_title='特征 2',
    width=600,
    height=450,
    plot_bgcolor='#f8f9fa' # 浅色背景
)

# 显示图表（在笔记本/网页环境中）
# fig_raw.show()
```

在应用 K-Means 之前，查看您的数据总是一个好习惯。这是由上述代码生成的图表：



![模拟数据点（聚类前）](plots/1520-0.json)



> 在二维空间中绘制的模拟数据点。我们可以通过视觉方式识别出三个潜在的分组。

如您所见，这些点形成了三个分离得相当好的分组。对于这种简单的二维数据，我们的眼睛可以很轻松地完成这项聚类任务。我们来看看 K-Means 能否重现这种分组。

### 应用 K-Means

现在我们将使用 Scikit-learn 的 `KMeans` 实现。我们需要告诉算法要寻找多少个聚类 ($K$)。由于我们生成的数据有 3 个中心，我们将 $K$ 设置为 3。

```python
from sklearn.cluster import KMeans

# 初始化 K-Means 算法
# n_clusters 是最重要的参数：聚类数量 (K)
# n_init='auto' 使用智能默认值，用于多次运行算法
# 使用不同的质心种子以提高结果。
# random_state 确保初始化的可复现性。
kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)

# 将算法拟合到数据 X
# K-Means 在此进行迭代：将点分配给聚类并更新质心。
kmeans.fit(X)

# 拟合后，模型包含以下结果：
# 1. 每个数据点的聚类分配：
cluster_labels = kmeans.labels_
# 2. 最终聚类中心（质心）的坐标：
centroids = kmeans.cluster_centers_

# print("分配给每个点的聚类标签:", cluster_labels)
# print("最终质心的坐标:\n", centroids)
```

`fit()` 方法在我们的数据 `X` 上运行 K-Means 算法。该算法迭代地将每个点分配给最近的质心，然后根据分配的点重新计算质心位置，直到质心稳定或达到最大迭代次数。

结果存储在 `kmeans` 对象中：

- `kmeans.labels_`：一个数组，第 $i$ 个元素表示分配给 `X` 中第 $i$ 个数据点的聚类索引（在此示例中为 0、1 或 2）。
- `kmeans.cluster_centers_`：一个二维数组，包含每个聚类质心的最终坐标。

### K-Means 结果可视化

现在，我们来可视化相同的数据点，但这次根据 K-Means 分配的聚类标签进行着色。我们还将绘制算法找到的最终质心。

```python
# 定义聚类的颜色 - 使用建议的调色板
cluster_colors = ['#4263eb', '#12b886', '#fd7e14'] # 靛蓝色、青色、橙色
centroid_color = '#f03e3e' # 质心使用红色

# 创建图表
fig_clustered = go.Figure()

# 添加数据点，按聚类标签着色
for i in range(3): # 遍历聚类 0, 1, 2
    points_in_cluster = X[cluster_labels == i]
    fig_clustered.add_trace(go.Scatter(
        x=points_in_cluster[:, 0],
        y=points_in_cluster[:, 1],
        mode='markers',
        marker=dict(color=cluster_colors[i], size=7, opacity=0.8),
        name=f'聚类 {i}'
    ))

# 添加质心
fig_clustered.add_trace(go.Scatter(
    x=centroids[:, 0],
    y=centroids[:, 1],
    mode='markers',
    marker=dict(color=centroid_color, size=14, symbol='x', line=dict(width=3)),
    name='质心'
))

fig_clustered.update_layout(
    title=f'K-Means 聚类结果 (K=3)',
    xaxis_title='特征 1',
    yaxis_title='特征 2',
    width=600,
    height=450,
    plot_bgcolor='#f8f9fa',
    legend_title_text='图例'
)

# 显示图表
# fig_clustered.show()
```

这是结果图表：



![K-Means 聚类结果 (K=3)](plots/1520-1.json)



> 相同的数据点，现在根据 K-Means 算法（$K=3$）分配的聚类进行了着色。红色“x”标记 (token)表示聚类质心的最终位置。

### 结果分析

将 K-Means 结果图与原始数据的初始图进行比较。您会看到 K-Means 成功识别出了我们模拟数据中存在的三个明显分组。每种颜色代表算法发现的一个聚类，红色“x”标记 (token)显示了属于该聚类的所有点的中心（平均位置）。

在这种简单情况下，当聚类分离良好且大致呈球形时，K-Means 表现非常出色。

### 如果选择不同的 $K$ 值会怎样？

还记得关于 $K$ 值选择的讨论吗？让我们简要思考一下，如果我们指示 K-Means 在此数据中寻找，比如 $K=2$ 个聚类，会发生什么。算法仍会运行，但它将被迫将三个可见的分组划分为仅两个聚类。通常，它可能会合并原始分组中的两个，或者将一个分组分散到两个结果聚类中，具体取决于初始质心的放置。同样，选择 $K=4$ 将迫使算法将一个或多个自然分组拆分成更小、可能意义较小的聚类。这项练习强调，尽管 K-Means 在数据划分方面很有效，但 $K$ 值的选择会显著影响结果及其解读。

### 小结

在本实践部分中，您已将 K-Means 算法应用于一个简单且直观可见的数据集。您了解了如何：

1. 使用 `make_blobs` 生成适合聚类实践的模拟数据。
2. 可视化原始数据以理解其结构。
3. 初始化并拟合 Scikit-learn 中的 `KMeans` 模型，指定所需的聚类数量 ($K$)。
4. 从拟合的模型中提取聚类分配 (`labels_`) 和质心位置 (`cluster_centers_`)。
5. 通过根据分配的聚类对数据点进行着色，并绘制最终质心来可视化结果。

"这个动手示例显示了使用 K-Means 在无标签数据中寻找分组的核心过程。尽管数据通常更为复杂且维度更高，但基本步骤保持不变。您现在具备了理解 K-Means 工作原理以及如何使用常用工具实现它的实践基础。"

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本基础教科书，提供了K-Means和各种其他机器学习算法的全面统计学理解。
- [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) — Scikit-learn developers (2023)
  Scikit-learn K-Means算法的官方文档，详细介绍了其参数、属性和使用方法。
- [sklearn.datasets.make_blobs](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html) — Scikit-learn developers (2023)
  用于生成各向同性高斯斑点进行聚类的官方文档，这是创建合成数据的常用方法。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  一本使用Python核心数据科学库（包括NumPy、Scikit-learn和Matplotlib/Plotly）的实用指南，对实现代码非常有帮助。

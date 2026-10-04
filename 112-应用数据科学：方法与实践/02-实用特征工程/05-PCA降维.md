# PCA降维

来源：[原文](https://apxml.com/zh/courses/applied-data-science/chapter-2-practical-feature-engineering/dimensionality-reduction-pca)

[返回章节目录](README.md) · [返回课程目录](../README.md)

随着您生成更多特征，也许是通过交互项或多项式扩展，您可能会发现输入变量的数量非常大。虽然有时有益，但具有许多维度（特征）的数据集可能带来挑战。训练模型可能变得计算成本高昂，并且由于“维度灾难”而增加了过拟合 (overfitting)的风险，这是一种在高维空间 (high-dimensional space)中数据变得稀疏，使模型难以泛化的现象。此外，拥有数百或数千个特征时，解释模型或可视化数据变得困难。

降维技术旨在减少输入变量的数量，同时尽可能保留有意义的信息。主成分分析（PCA）是为此目的最广泛使用的无监督技术之一。它不仅仅选择原始特征的一个子集；相反，它构建了一个新的、更小的特征集，称为主成分，这些成分是原始特征的线性组合。

### 理解主成分分析 (PCA)

PCA的核心思想是找出数据中方差最大的方向。想象您的数据点绘制在多维空间中。PCA识别出数据点最分散的轴，这是第一个主成分（PC1）。然后它找到与第一个轴正交（垂直）的下一个轴，该轴捕获了剩余的最大方差。这是第二个主成分（PC2），以此类推。

每个主成分是原始特征的线性组合：

$PC_i = w_{i1}X_1 + w_{i2}X_2 + ... + w_{ip}X_p$

其中 $PC_i$ 是第 $i$ 个主成分，$X_j$ 是第 $j$ 个原始特征，而 $w_{ij}$ 是定义每个原始特征对主成分贡献的载荷得分或权重 (weight)。

这些成分有两个重要特性：

1. 它们按解释的方差量排序，PC1解释的最多，PC2次之，以此类推。
2. 它们彼此不相关。这对于某些对多重共线性敏感的建模算法可能有利。

从数学角度看，PCA涉及计算数据协方差矩阵的特征向量 (vector)和特征值（或者对数据矩阵执行奇异值分解，SVD）。特征向量给出主成分的方向，相应的特征值表示沿这些方向的方差大小。

### PCA数据标准化处理

在应用PCA之前，几乎总是需要标准化您的数据，这意味着将每个特征缩放到具有零均值和单位方差。为什么？PCA查找最大方差的方向。如果特征具有非常不同的尺度（例如，一个特征范围从0到1，另一个从10,000到1,000,000），则尺度较大的特征将固有地具有更大的方差，并将主导主成分，无论其在捕获数据结构中的实际重要性如何。标准化确保所有特征对分析的贡献相同。

我们可以使用scikit-learn中的`StandardScaler`：

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# 假设 'X' 是您的 pandas DataFrame 或 NumPy 特征数组
# 示例数据（请替换为您的实际数据）
data = {'feature1': np.random.rand(100) * 10,
        'feature2': np.random.rand(100) * 1000,
        'feature3': np.random.rand(100) - 50}
X = pd.DataFrame(data)

print("原始数据样本：")
print(X.head())

# 1. 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n标准化数据样本（均值接近0，标准差接近1）：")
print(pd.DataFrame(X_scaled, columns=X.columns).head())
```

### 使用Scikit-learn应用PCA

数据缩放后，使用scikit-learn的`PCA`类应用PCA非常直接。您通常需要决定保留多少个主成分。

```python
# 2. 应用PCA
# 让我们首先拟合PCA，不指定组件数量
# 以查看每个组件解释了多少方差。
pca_full = PCA()
pca_full.fit(X_scaled)

# 解释方差比例：每个组件解释的方差百分比
explained_variance = pca_full.explained_variance_ratio_
print("\n每个组件的解释方差比例：")
print(explained_variance)

# 累积解释方差
cumulative_variance = np.cumsum(explained_variance)
print("\n累积解释方差：")
print(cumulative_variance)
```

### 选择组件数量

决定保留多少个组件（$k$）的常用方法是检查累积解释方差。您可以设置一个阈值，例如保留足够的组件来解释总方差的95%或99%。绘制累积解释方差有助于可视化这种权衡。



[交互图表：主成分的累积解释方差](https://apxml.com/zh/courses/applied-data-science/chapter-2-practical-feature-engineering/dimensionality-reduction-pca#plot-in9vxp)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "主成分的累积解释方差",
    "xaxis": {
      "title": "组件数量"
    },
    "yaxis": {
      "title": "累积解释方差比",
      "range": [
        0,
        1.05
      ]
    },
    "template": "plotly_white",
    "legend": {
      "yanchor": "bottom",
      "y": 0.01,
      "xanchor": "right",
      "x": 0.99
    }
  },
  "data": [
    {
      "type": "scatter",
      "mode": "lines+markers",
      "name": "累积方差",
      "x": [
        1,
        2,
        3
      ],
      "y": [
        0.66148157,
        0.99829893,
        1.0
      ],
      "marker": {
        "color": "#228be6"
      }
    },
    {
      "type": "scatter",
      "mode": "lines",
      "name": "95%阈值",
      "x": [
        0,
        3
      ],
      "y": [
        0.95,
        0.95
      ],
      "line": {
        "dash": "dash",
        "color": "#fa5252"
      }
    }
  ]
}
```

</details>



> 随着主成分数量的增加，显示累积解释方差的图。一种常见的方法是选择曲线开始趋于平稳或达到所需阈值（例如95%）时的组件数量。

根据图表或累积方差数组，您可以选择组件数量。例如，如果保留2个组件可以解释95%的方差，您可能会认为这已足够。

```python
# 3. 选择组件数量（例如，目标是 >= 95% 方差）
n_components_chosen = 2 # 根据示例输出或图表

# 使用选定的组件数量再次拟合PCA
pca = PCA(n_components=n_components_chosen)
pca.fit(X_scaled)

# 4. 将数据转换到低维空间
X_pca = pca.transform(X_scaled)

print(f"\n原始数据形状：{X_scaled.shape}")
print(f"转换后数据形状（含 {n_components_chosen} 个组件）：{X_pca.shape}")

print("\n转换后数据（前5行）：")
print(X_pca[:5, :])
```

生成的 `X_pca` 数组现在包含由前两个主成分表示的数据。这些新特征捕获了原始数据集中最重要的方差，但处于较低维空间。您现在可以将 `X_pca` 用作机器学习 (machine learning)模型的输入。

### 可视化示例

PCA通常用于将数据降至2或3维以进行可视化。让我们想象将PCA应用于Iris数据集（有4个特征）并绘制前两个主成分。

```python
# 使用Iris数据集的示例（假设已加载）
# from sklearn.datasets import load_iris
# iris = load_iris()
# X_iris = iris.data
# y_iris = iris.target
# target_names = iris.target_names

# scaler_iris = StandardScaler()
# X_iris_scaled = scaler_iris.fit_transform(X_iris)

# pca_iris = PCA(n_components=2)
# X_iris_pca = pca_iris.fit_transform(X_iris_scaled)

# 现在，想象绘制 X_iris_pca[:, 0] 与 X_iris_pca[:, 1]
# 按 y_iris 颜色区分。

# 用于图表说明的虚拟数据（请替换为实际PCA结果）
import plotly.graph_objects as go
dummy_pca_data = np.random.rand(150, 2) * np.array([[5, 2]]) + np.array([[-2, -1]])
dummy_labels = np.random.randint(0, 3, 150)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # 示例颜色

fig = go.Figure()
for i, color in enumerate(colors):
    idx = dummy_labels == i
    fig.add_trace(go.Scatter(
        x=dummy_pca_data[idx, 0], y=dummy_pca_data[idx, 1],
        mode='markers',
        marker=dict(color=color, size=8, opacity=0.7),
        name=f'类别 {i}' # 如果可用，请替换为实际的 target_names
    ))

fig.update_layout(
    title="投影到前两个主成分的数据（示例）",
    xaxis_title="主成分 1",
    yaxis_title="主成分 2",
    template="plotly_white",
    legend_title_text='类别'
)

# 为了以兼容的格式显示图表：
# print(fig.to_json())
```



[交互图表：投影到前两个主成分的数据（示例）](https://apxml.com/zh/courses/applied-data-science/chapter-2-practical-feature-engineering/dimensionality-reduction-pca#plot-wwvx0t)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "投影到前两个主成分的数据（示例）",
    "xaxis": {
      "title": "主成分 1"
    },
    "yaxis": {
      "title": "主成分 2"
    },
    "template": "plotly_white",
    "legend_title": {
      "text": "类别"
    }
  },
  "data": [
    {
      "type": "scatter",
      "x": [
        -0.04,
        1.41,
        2.32,
        1.45,
        0.97,
        0.78,
        1.59,
        2.35,
        2.28,
        0.87,
        0.12,
        0.16,
        1.69,
        0.96,
        0.41,
        0.94,
        0.69,
        0.14,
        1.63,
        1.81,
        1.01,
        0.43,
        1.65,
        0.52,
        1.25,
        1.27,
        0.22,
        0.06,
        1.68,
        1.85,
        0.96,
        2.05,
        1.62,
        1.82,
        1.51,
        1.19,
        1.86,
        2.13,
        2.01,
        1.09,
        1.72,
        1.41,
        1.44,
        0.41,
        0.17,
        2.59,
        2.08,
        2.33
      ],
      "y": [
        -1.64,
        -0.53,
        -0.05,
        -0.56,
        0.23,
        0.39,
        -0.46,
        -0.18,
        -0.64,
        -0.52,
        -1.44,
        -0.09,
        -0.11,
        0.13,
        -0.57,
        -0.51,
        0.43,
        -1.06,
        0.04,
        -0.42,
        -0.02,
        -0.41,
        0.18,
        -0.75,
        -0.54,
        -0.18,
        -1.56,
        -0.25,
        -0.76,
        -0.06,
        0.11,
        0.21,
        0.06,
        -0.44,
        0.19,
        0.07,
        -0.37,
        -0.69,
        -0.82,
        0.04,
        -0.56,
        -0.41,
        0.35,
        0.13,
        -0.68,
        -0.75,
        -0.34,
        0.16
      ],
      "mode": "markers",
      "marker": {
        "color": "#1f77b4",
        "size": 8,
        "opacity": 0.7
      },
      "name": "类别 0"
    },
    {
      "type": "scatter",
      "x": [
        2.01,
        0.03,
        0.76,
        1.84,
        1.93,
        1.11,
        1.32,
        0.48,
        1.98,
        2.08,
        2.45,
        0.36,
        0.39,
        1.98,
        2.75,
        0.08,
        1.05,
        1.06,
        1.93,
        0.21,
        2.67,
        1.6,
        0.88,
        1.4,
        2.42,
        2.8,
        1.43,
        0.01,
        2.74,
        2.22,
        0.32,
        0.6,
        1.41,
        0.89,
        1.32,
        1.2,
        0.78,
        0.63,
        1.34,
        1.96,
        1.14,
        2.71,
        0.1,
        2.61,
        0.24,
        1.08,
        0.97,
        0.64,
        1.68,
        2.31,
        2.46,
        1.46
      ],
      "y": [
        -0.3,
        -0.16,
        -0.63,
        -0.65,
        -0.89,
        -0.87,
        -0.65,
        -1.52,
        -0.73,
        -0.69,
        -0.08,
        -1.55,
        -0.09,
        -0.12,
        -0.33,
        -0.98,
        0.29,
        -0.91,
        0.16,
        -0.01,
        -0.34,
        -0.6,
        0.23,
        -0.47,
        -0.96,
        -0.92,
        -0.83,
        -1.61,
        -0.36,
        -0.42,
        -1.68,
        -0.13,
        0.3,
        0.09,
        0.12,
        0.01,
        -0.54,
        -1.4,
        0.05,
        0.21,
        -0.62,
        -0.23,
        -0.45,
        -0.08,
        -0.91,
        -0.42,
        -0.92,
        0.31,
        -0.11,
        -0.07,
        0.14,
        -0.81
      ],
      "mode": "markers",
      "marker": {
        "color": "#ff7f0e",
        "size": 8,
        "opacity": 0.7
      },
      "name": "类别 1"
    },
    {
      "type": "scatter",
      "x": [
        0.27,
        1.25,
        0.97,
        1.19,
        0.61,
        0.18,
        2.31,
        2.5,
        2.13,
        2.82,
        0.28,
        1.95,
        1.48,
        0.67,
        2.05,
        1.54,
        1.18,
        1.11,
        0.69,
        0.57,
        2.64,
        1.74,
        0.07,
        1.04,
        1.7,
        2.29,
        0.24,
        1.83,
        2.59,
        1.13,
        0.6,
        0.81,
        0.54,
        1.52,
        0.36,
        1.64,
        0.9,
        0.53,
        2.09,
        1.17,
        0.63,
        2.05,
        1.74,
        1.61,
        2.08,
        2.53,
        1.29,
        0.74,
        2.12,
        0.52,
        2.3
      ],
      "y": [
        -1.2,
        0.14,
        -0.59,
        -0.16,
        0.09,
        -1.68,
        -0.28,
        -0.43,
        -0.34,
        -0.42,
        -1.02,
        0.06,
        -0.51,
        -1.61,
        0.04,
        -0.95,
        -0.78,
        -0.13,
        -0.91,
        -0.48,
        -0.99,
        0.04,
        -0.83,
        -0.44,
        -0.03,
        -0.08,
        -0.45,
        -0.1,
        -0.31,
        -0.47,
        -0.37,
        -0.24,
        -1.5,
        -0.1,
        -0.49,
        -0.96,
        -0.01,
        -1.03,
        -0.77,
        -0.18,
        -1.24,
        -0.84,
        -0.88,
        -0.07,
        -0.23,
        -0.86,
        -0.14,
        -0.01,
        -0.37,
        -0.74,
        -0.04
      ],
      "mode": "markers",
      "marker": {
        "color": "#2ca02c",
        "size": 8,
        "opacity": 0.7
      },
      "name": "类别 2"
    }
  ]
}
```

</details>



> 示例散点图，显示投影到前两个主成分的数据。通常，这些成分可以有效地分离不同的类别或显示数据中的聚类，使它们即使在为建模保留更多成分时，也对可视化很有用。

### 注意事项与局限性

尽管功能强大，但PCA并非万能：

- **可解释性：** 主成分是原始特征的组合。虽然PC1可能代表“整体大小”，PC2可能代表某些特征组之间的对比，但与原始的、特定于领域的特征相比，解释它们的确切含义可能具有挑战性。
- **线性假设：** PCA假定特征之间存在线性相关性。如果底层数据结构高度非线性，它可能表现不佳。在这种情况下，诸如核PCA或流形学习方法（例如t-SNE、UMAP，稍后会针对可视化进行介绍）可能更合适。
- **信息丢失：** 通过丢弃方差较低的组件，您不可避免地会丢失一些信息。目的是保留足够的方差，以保持建模任务的基本结构。
- **尺度敏感性：** 如前所述，PCA对数据尺度高度敏感。务必首先标准化或归一化 (normalization)您的数据。

PCA是数据科学家工具箱中用于降维的一种基本方法。它有助于管理高维数据，可能减少噪声，降低计算成本，并辅助可视化，构成了为高效机器学习 (machine learning)建模准备特征的重要步骤。

## 参考资料

- [An Introduction to Statistical Learning: with Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2021)
  Publisher: Springer; Pages: Chapter 10, Section 10.2
  清晰易懂地解释了PCA并提供了实际示例，适合初学者。
- [1.5. Principal component analysis (PCA)](https://scikit-learn.org/stable/modules/decomposition.html#pca) — scikit-learn developers (2023)
  Publisher: scikit-learn project
  scikit-learn PCA实现的官方文档，详细介绍了参数、属性和用法。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; Pages: Chapter 14, Section 14.5
  对PCA提供了基础而严谨的统计学叙述。
- [Machine Learning](https://www.coursera.org/learn/machine-learning) — Andrew Ng, Aarti Bagul, Geoff Ladwig, Eddy Shyu (2012)
  Publisher: Coursera, DeepLearning.AI, Stanford Online
  提供了PCA易于理解的概述和算法细节。

---

[上一节](04-%E4%BA%A4%E4%BA%92%E9%A1%B9%E4%B8%8E%E5%A4%9A%E9%A1%B9%E5%BC%8F%E7%89%B9%E5%BE%81.md) · [下一节](06-%E4%BD%BF%E7%94%A8%E7%BB%9F%E8%AE%A1%E6%96%B9%E6%B3%95%E9%80%89%E6%8B%A9%E7%89%B9%E5%BE%81.md)

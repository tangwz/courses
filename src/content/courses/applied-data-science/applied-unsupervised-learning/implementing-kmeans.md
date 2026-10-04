---
course: "applied-data-science"
chapter: "applied-unsupervised-learning"
lesson: "implementing-kmeans"
sourceId: 1883
sourceUrl: "https://apxml.com/zh/courses/applied-data-science/chapter-4-applied-unsupervised-learning/implementing-kmeans"
title: "实现K-Means聚类"
description: "应用K-Means算法将数据分区为群组。"
order: 2
plots: ["plots/1883-0.json", "plots/1883-1.json"]
sourceHash: "45c86990ed08ff9814299a4e7dabf3a3e59d42826215cf195920a56d4185f119"
sourceCorrections: []
---

K-Means聚类是一种特定且广泛应用于数据分区的技术，属于无监督学习 (supervised learning) (unsupervised learning)的范畴。K-Means的目标直接明了：根据相似性将数据点分成预设数量（$k$）的不同、不重叠的群组。同一群组内的数据点应彼此相似，而不同群组中的点则应不相似。

### K-Means算法说明

K-Means是一种迭代算法，旨在最小化数据点与其所属群组中心之间的距离。群组的“中心”被称为**质心**，通常计算为该群组内所有数据点的平均值。

该算法通过以下主要步骤进行：

1. **初始化**：首先，您必须指定所需群组的数量，$k$。然后，选择$k$个初始质心。常见的初始化策略包括随机选择$k$个数据点，或者使用更智能的方法，例如'k-means++'（scikit-learn中的默认设置），它倾向于选择距离较远的初始质心，从而得到更优和更稳定的结果。
2. **分配步骤**：数据集中的每个数据点都被分配到最近的质心。“近度”通常由数据点（$x$）与每个质心（$\mu_i$）之间的欧几里得距离来确定。如果点$x$到质心$\mu_i$的距离小于到任何其他质心$\mu_j$（对于$j \neq i$）的距离，则点$x$被分配到群组$C_i$。
3. **更新步骤**：所有点都被分配到群组后，重新计算$k$个质心的位置。每个群组的新质心成为在上一步中分配到该群组的所有数据点的平均（平均位置）。
4. **迭代**：步骤2和3重复进行。在每次迭代中，点可能会改变群组分配，质心也会移动。这个过程持续到满足停止标准，例如：
   - 质心在迭代之间不再明显移动。
   - 数据点停止改变群组分配。
   - 达到最大迭代次数。

该算法的目标是最小化**簇内平方和 (WCSS)**，通常称为惯性。这是每个数据点与其所属群组质心之间平方距离的总和：


$$
WCSS = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2
$$


其中$k$是群组数量，$C_i$是群组$i$中的点集，$\mu_i$是群组$i$的质心，$||x - \mu_i||^2$是欧几里得距离的平方。

### 使用Scikit-learn实现K-Means

Scikit-learn通过`sklearn.cluster.KMeans`类提供了K-Means的便捷实现。我们来看一个实际例子。

首先，我们需要一些数据。我们将使用scikit-learn的`make_blobs`函数生成具有不同群组的合成数据，这对于说明K-Means如何工作很有帮助。

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 生成具有3个中心的人工数据
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.80, random_state=42)

# 对数据进行标准化（对K-Means很重要）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 实例化KMeans
# 我们指定n_clusters (k) = 3，因为我们知道此示例的真实结构
# n_init=10 用不同的质心种子运行算法10次
# 最终结果将是惯性方面最好的输出
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)

# 将模型拟合到标准化数据
kmeans.fit(X_scaled)

# 获取群组分配和质心
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# --- 使用Plotly进行可视化 ---

# 创建数据点的散点图，按预测的群组标签着色
fig = go.Figure()

# 添加数据点
fig.add_trace(go.Scatter(
    x=X_scaled[:, 0], y=X_scaled[:, 1],
    mode='markers',
    marker=dict(
        color=labels, # 根据群组标签分配颜色
        colorscale=['#1f77b4', '#ff7f0e', '#2ca02c'], # 示例颜色
        opacity=0.8,
        line=dict(width=0.5, color='#495057')
    ),
    name='数据点'
))

# 添加质心
fig.add_trace(go.Scatter(
    x=centroids[:, 0], y=centroids[:, 1],
    mode='markers',
    marker=dict(
        color='#d62728', # 质心的红色
        size=12,
        symbol='x',
        line=dict(width=2, color='#FFFFFF') # 白色边框以增加可见性
    ),
    name='质心'
))

# 更新布局以获得更好的外观
fig.update_layout(
    title='K-Means聚类结果 (k=3)',
    xaxis_title='特征1（标准化）',
    yaxis_title='特征2（标准化）',
    showlegend=True,
    plot_bgcolor='#e9ecef', # 浅灰色背景
    width=700,
    height=500
)

# 要显示图表（例如在Jupyter环境中）：
# fig.show()

# 或保存为HTML：
# fig.write_html("kmeans_clusters.html")

# Plotly JSON对象（用于嵌入Web环境）
plotly_json = fig.to_json()
```



![K-Means聚类结果 (k=3)](plots/1883-0.json)



> 散点图显示了根据K-Means分配的群组进行着色的数据点。红色的“x”标记 (token)表示群组质心的最终位置。

在代码中：

- `n_clusters=3`：我们告诉K-Means寻找3个群组。
- `init='k-means++'`：使用k-means++初始化方法来获取更优的质心起始点。
- `n_init=10`：用不同的随机初始化运行K-Means算法10次，并返回基于WCSS（惯性）的最佳结果。这有助于减少因初始化不佳而陷入次优解的问题。
- `random_state=42`：确保结果的可复现性。
- `.fit(X_scaled)`：在标准化数据上训练K-Means模型。
- `.labels_`：一个数组，包含分配给每个数据点的群组索引（在此示例中为0、1或2）。
- `.cluster_centers_`：一个NumPy数组，包含最终质心的坐标。

### 确定最优群组数量 (k)

K-Means的一个重要挑战是，您通常事先不知道最优群组数量（$k$）。**肘部法则**是一种常用的启发式方法，用于选择一个合适的$k$值。

其思路是，对一系列不同的$k$值（例如，从1到10）运行K-Means，并计算每次运行的WCSS（惯性）。然后，我们将WCSS与$k$进行绘图。随着$k$的增加，WCSS通常会下降，因为点会更接近较小群组的质心。然而，在某个点之后，下降的速度通常会明显减缓，在图表中形成一个“肘部”形状。肘部点的$k$值通常被认为是群组数量的良好指引。

以下是实现肘部法则的方法：

```python
import plotly.graph_objects as go

# 计算不同k值下的WCSS（惯性）
wcss = []
k_values = range(1, 11) # 测试k从1到10

for k in k_values:
    kmeans_elbow = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans_elbow.fit(X_scaled)
    wcss.append(kmeans_elbow.inertia_) # inertia_ 属性存储WCSS

# --- 使用Plotly进行可视化 ---
fig_elbow = go.Figure()

fig_elbow.add_trace(go.Scatter(
    x=list(k_values),
    y=wcss,
    mode='lines+markers',
    marker=dict(color='#1c7ed6', size=8),
    line=dict(color='#1c7ed6', width=2)
))

# 突出显示潜在的肘部点（例如，本例中k=3）
elbow_k = 3
fig_elbow.add_vline(x=elbow_k, line_width=2, line_dash="dash", line_color="#fa5252",
                    annotation_text=f"潜在肘部点 (k={elbow_k})", annotation_position="top right")

fig_elbow.update_layout(
    title='确定最优k值的肘部法则',
    xaxis_title='群组数量 (k)',
    yaxis_title='簇内平方和 (WCSS)',
    plot_bgcolor='#e9ecef',
    width=700,
    height=400
)

# fig_elbow.show()
plotly_elbow_json = fig_elbow.to_json()
```



![确定最优k值的肘部法则](plots/1883-1.json)



> 图表显示了WCSS与群组数量（k）的关系。“肘部”通常表示在最小化WCSS和保持群组数量适中之间的一种合理权衡。在这个合成示例中，肘部在k=3处清晰可见，与我们生成的真实群组数量相符。

“找到确切的肘部点有时可能是主观的，特别是在曲线可能更平滑的数据中。它作为一种指导，而非明确的规则。其他技术，如轮廓系数，提供了评估群组质量和选择$k$值的替代方法。”

### 特征标准化的意义

K-Means依赖距离计算（通常是欧几里得距离）将点分配到群组并更新质心。如果特征具有不同的尺度（例如，一个特征范围从0到1，另一个从1,000到100,000），那么范围较大的特征将不成比例地影响距离计算，进而影响聚类结果。

因此，在应用K-Means之前，对数值特征进行**尺度调整或归一化 (normalization)**是标准做法。常用的方法包括`StandardScaler`（将特征标准化为零均值和单位方差，如代码示例所示）或`MinMaxScaler`（将特征缩放到固定范围，通常是[0, 1]）。这确保所有特征对距离计算的贡献更均衡。请记住在训练数据上拟合缩放器，并在聚类之前用它转换训练数据和任何新数据。

### K-Means的假设与局限

尽管强大而高效，K-Means在某些假设下运作，这可能限制其在某些情况下的有效性：

1. **球形群组**：K-Means假设群组大致是球形和凸形的。它难以处理细长、非凸或不规则形状的群组。
2. **方差/大小相等**：该算法隐含假设群组具有相似的大小和方差（离散度）。如果群组的密度或大小差异很大，K-Means可能会错误地分割较大或密度较低的群组。
3. **对初始化的敏感性**：最终的聚类结果可能取决于质心的初始放置。使用不同的初始化多次运行算法（`n_init`参数 (parameter)）有助于找到更稳定的解。
4. **对离群值的敏感性**：由于质心是群组内点的平均值，离群值可能会显著地将质心拉离群组的实际中心。可能需要进行预处理来处理离群值。
5. **预设$k$值**：要求用户预先指定群组数量$k$，而这通常是未知的。

理解这些局限对于解释结果以及判断K-Means是否适合您的特定数据集和问题是很重要的。对于具有非球形群组或密度不一的数据集，其他算法，例如DBSCAN（接下来讨论），可能更合适。

## 参考资料

- [Least squares quantization in PCM](https://doi.org/10.1109/TIT.1982.1056489) — Stuart P. Lloyd (1982)
  Journal: IEEE Transactions on Information Theory; Publisher: IEEE; Volume: 28; Pages: 129-137; DOI: [10.1109/TIT.1982.1056489](https://doi.org/10.1109/TIT.1982.1056489)
  介绍了K-Means算法（也称为Lloyd算法）的基础。
- [k-means++: The advantages of careful seeding](https://doi.org/10.1145/1283383.1283494) — David Arthur and Sergei Vassilvitskii (2007)
  Journal: Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms; Publisher: Society for Industrial and Applied Mathematics (SIAM) and Association for Computing Machinery (ACM); Pages: 1027-1035; DOI: [10.1145/1283383.1283494](https://doi.org/10.1145/1283383.1283494)
  介绍了k-means++初始化方法，提高了K-Means聚类的稳定性和质量。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本使用scikit-learn库实现K-Means及其他机器学习算法的实用指南。
- [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) — scikit-learn developers (2024)
  scikit-learn库中K-Means实现的官方文档，详细说明了参数和属性。

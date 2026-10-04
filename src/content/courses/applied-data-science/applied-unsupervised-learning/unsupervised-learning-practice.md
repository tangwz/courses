---
course: "applied-data-science"
chapter: "applied-unsupervised-learning"
lesson: "unsupervised-learning-practice"
sourceId: 1904
sourceUrl: "https://apxml.com/zh/courses/applied-data-science/chapter-4-applied-unsupervised-learning/unsupervised-learning-practice"
title: "动手实践：聚类与异常检测练习"
description: "实践实现聚类和异常检测算法。"
order: 6
plots: ["plots/1904-0.json", "plots/1904-1.json", "plots/1904-2.json", "plots/1904-3.json"]
sourceHash: "bd7736915c7a9c5a20ffb08b8591ae9bef96378211275866116fdad46edf7599"
sourceCorrections: []
---

实际练习展示了无监督学习 (supervised learning) (unsupervised learning)方法的应用。你将使用 `scikit-learn` 实现 K-Means 和 DBSCAN 进行聚类，并在一个生成的数据集上应用一种常用技术进行异常检测。这种动手操作的经历将有助于巩固你对这些算法如何工作以及如何解释其结果的理解。

我们首先生成一个模拟数据集。我们将使用 `scikit-learn` 中的 `make_blobs` 创建离散的点群，然后加入一些随机散布的点，这些点可以看作是离群值或噪声。

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# 生成样本数据
X, y_true = make_blobs(n_samples=400, centers=4,
                       cluster_std=0.80, random_state=42)

# 添加一些远离聚类的噪声点
rng = np.random.RandomState(42)
n_outliers = 30
outliers = rng.uniform(low=np.min(X) - 5, high=np.max(X) + 5, size=(n_outliers, 2))
X = np.vstack([X, outliers])

# 对对尺度敏感的算法（如 K-Means 和 DBSCAN）进行特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 创建 DataFrame 以便可视化
df = pd.DataFrame(X_scaled, columns=['Feature 1', 'Feature 2'])

# 数据的初步可视化
fig_initial = px.scatter(df, x='Feature 1', y='Feature 2',
                         title='Synthetic Dataset with Potential Outliers',
                         color_discrete_sequence=['#495057']) # 对未聚类的点使用灰色
fig_initial.update_layout(showlegend=False)
# fig_initial.show() # 在 Python 环境中显示图表
```



![带有潜在离群值的模拟数据集](plots/1904-0.json)



> 生成数据集特征经过缩放后的初始散点图。离散的群组清晰可见，同时还有一些散布的点。

### 应用 K-Means 聚类

K-Means 的目标是将数据划分成 $k$ 个离散、不重叠的聚类。每个数据点都属于距离其最近的均值（聚类中心）所在的聚类。我们需要指定聚类的数量，$k$。根据可视化结果（以及我们生成数据的方式），$k=4$ 似乎是一个合理的起始点。

```python
from sklearn.cluster import KMeans

# 实例化并拟合 K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10) # 对于未来版本，n_init='auto' 或 10
kmeans.fit(X_scaled)

# 获取聚类分配和聚类中心
df['KMeans Cluster'] = kmeans.labels_.astype(str) # 转换为字符串以便使用离散颜色
centroids = scaler.inverse_transform(kmeans.cluster_centers_) # 将聚类中心转换回原始比例

# 可视化 K-Means 结果
fig_kmeans = px.scatter(df, x='Feature 1', y='Feature 2',
                        color='KMeans Cluster',
                        title='K-Means Clustering Results (k=4)',
                        color_discrete_sequence=px.colors.qualitative.Pastel) # 使用一种美观的颜色序列

# 将聚类中心添加到图表中（转换回缩放后的坐标以便绘图）
fig_kmeans.add_scatter(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1],
                       mode='markers', marker=dict(color='#d6336c', size=12, symbol='x'),
                       name='聚类中心')

# fig_kmeans.show()
```



![K-Means 聚类结果 (k=4)](plots/1904-1.json)



> K-Means 聚类结果，k=4。点根据其分配的聚类着色，聚类中心用 'x' 标记 (token)。注意离群点是如何被分配到最近的聚类的。

K-Means 成功识别出主要群组，但它将每个点（包括我们添加的明显离群点）强制归入一个聚类。这是因为 K-Means 假定聚类是球形的，并将每个点分配给最近的聚类中心。

### 应用 DBSCAN 聚类

DBSCAN（基于密度的带噪声空间聚类应用）将紧密排列的点归为一类，将低密度区域中孤立的点标记 (token)为离群点。它不需要预先指定聚类数量，而是依赖两个参数 (parameter)：`eps`（两个样本之间，其中一个被视为另一个邻域内的最大距离）和 `min_samples`（一个点被视为核心点所需的邻域内的样本数量）。

选择合适的 `eps` 和 `min_samples` 通常需要一些实验或专业知识。让我们尝试一些值。较小的 `eps` 或较大的 `min_samples` 将导致更多的点被归类为噪声。

```python
from sklearn.cluster import DBSCAN

# 实例化并拟合 DBSCAN
# 这些参数可能需要根据数据集密度进行调整
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan.fit(X_scaled)

# 获取聚类分配（-1 表示噪声/离群点）
df['DBSCAN Cluster'] = dbscan.labels_.astype(str) # 转换为字符串以便使用离散颜色

# 可视化 DBSCAN 结果
fig_dbscan = px.scatter(df, x='Feature 1', y='Feature 2',
                        color='DBSCAN Cluster',
                        title=f'DBSCAN Clustering Results (eps={dbscan.eps}, min_samples={dbscan.min_samples})',
                        color_discrete_map={"-1": "#adb5bd"}, # 噪声点使用灰色
                        category_orders={"DBSCAN Cluster": sorted(df['DBSCAN Cluster'].unique(), key=int)}, # 确保 -1 排在第一位
                        color_discrete_sequence=px.colors.qualitative.Pastel) # 实际聚类使用的颜色

# fig_dbscan.show()
```



![DBSCAN 聚类结果 (eps=0.3, min_samples=5)](plots/1904-2.json)



> DBSCAN 聚类结果。标记为 '-1'（灰色）的点被识别为噪声/离群点，因为它们根据选择的 `eps` 和 `min_samples` 不属于任何密集区域。

将此与 K-Means 图进行比较。DBSCAN 成功识别出四个主要聚类，而且值得一提的是，它将大多数模拟离群点（以及可能在主要聚类边缘的一些点）标记为噪声（聚类标签 -1）。这种发现噪声点的能力是基于密度的聚类在处理含有离群值的数据集时的一个重要优势。

### 基本异常检测

尽管 DBSCAN 本身就能识别可被视为异常的噪声点，但其他算法是专门为异常检测设计的。让我们尝试使用孤立森林算法。它通过随机划分数据来运行，并明确识别出孤立的观测值，这意味着它们需要更少的划分才能与其余数据分离。

```python
from sklearn.ensemble import IsolationForest

# 实例化并拟合孤立森林
# 'contamination' 是预期离群点的比例，可设置为 'auto' 或特定值
# 让我们根据添加的噪声点数量（30 / 430）估算，约为 0.07
iso_forest = IsolationForest(contamination=0.07, random_state=42)
iso_forest.fit(X_scaled)

# 预测异常（-1 为异常，1 为正常）
df['Anomaly'] = iso_forest.predict(X_scaled)
df['Anomaly'] = df['Anomaly'].map({1: '正常', -1: '异常'}) # 映射到可读的标签

# 可视化异常检测结果
fig_anomaly = px.scatter(df, x='Feature 1', y='Feature 2',
                         color='Anomaly',
                         title='使用孤立森林进行异常检测',
                         color_discrete_map={'正常': '#1f77b4', '异常': '#d62728'}, # 标准蓝色，醒目红色
                         category_orders={"Anomaly": ["正常", "异常"]}) # 确保图例顺序一致

# fig_anomaly.show()
```



![使用孤立森林进行异常检测](plots/1904-3.json)



> 孤立森林结果，突出显示被识别为异常的点（红色）。

孤立森林识别出许多与 DBSCAN 噪声点相同的点。然而，确切的集合可能因算法逻辑和参数 (parameter)（如 `contamination` 系数）而异。孤立森林是专门为了发现离群点而设计的，而 DBSCAN 则是在识别密集区域时附带发现它们。根据具体目标，一种算法可能比另一种更受青睐。

本次实践演示了如何应用 K-Means 和 DBSCAN 进行聚类，以及使用孤立森林进行异常检测。你看到 K-Means 如何将所有点分配到聚类中，而 DBSCAN 可以识别噪声。孤立森林提供了一种发现离群点的针对性方法。尝试调整参数（K-Means 的 `k`，DBSCAN 的 `eps` 和 `min_samples`，孤立森林的 `contamination`）通常对于特定数据集和分析目标达成预期结果是必要的。

## 参考资料

- [A density-based algorithm for discovering clusters in large spatial databases with noise](http://www.aaai.org/Library/KDD/1996/kdd96-037.php) — Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu (1996)
  Journal: Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96); Publisher: AAAI Press; Pages: 226-231; DOI: [10.1145/233255.233290](https://doi.org/10.1145/233255.233290)
  介绍DBSCAN聚类算法的原始论文。
- [Isolation Forest](https://ieeexplore.ieee.org/document/4781136) — Fei Tony Liu, Kai Ming Ting, Zhi-Hua Zhou (2008)
  Journal: Proceedings of the 2008 Eighth IEEE International Conference on Data Mining; Publisher: IEEE; Pages: 413-422; DOI: [10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17)
  介绍Isolation Forest异常检测算法的原始论文。
- [Clustering](https://scikit-learn.org/stable/modules/clustering.html) — scikit-learn developers (2024)
  scikit-learn中聚类算法（包括K-Means和DBSCAN）的官方指南。
- [Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html) — scikit-learn developers (2024)
  Publisher: scikit-learn
  scikit-learn中异常检测算法（包括Isolation Forest）的官方指南。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://books.google.com/books/about/Hands_On_Machine_Learning_with_Scikit_Le.html?id=oI3fEAAAQBAJ) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 864
  一本实用指南，讲解无监督学习及K-Means、DBSCAN和Isolation Forest等算法，并附有scikit-learn实现。第三版。

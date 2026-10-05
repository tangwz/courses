# 应用 DBSCAN 进行基于密度的聚类

来源：[原文](https://apxml.com/zh/courses/applied-data-science/chapter-4-applied-unsupervised-learning/applying-dbscan)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然 K-Means 在事先已知聚类数量时，对发现球形或凸状聚类很有效，但它难以处理包含任意形状分组或明显噪声的数据集。基于密度的噪声应用空间聚类 (DBSCAN) 提供了一种不同方法，将聚类定义为点密度高的连续区域，这些区域由密度低的区域隔开。这使其擅长识别复杂形状的聚类，并自动将离群点作为噪声处理。

### 密度背后的想法

想象一下点散布在地图上。聚类可以被看作是拥挤区域（高密度），而它们之间的空间则相对空旷（低密度）。DBSCAN 使用两个主要参数 (parameter)将这个想法正式化：

1. **邻域半径 (`eps`)**: 这定义了每个数据点周围的一个半径。一个点的 `eps` 邻域包含在此距离内的所有其他点。
2. **最小样本数 (`min_samples`)**: 这指定了一个点在其 `eps` 邻域内（包括点本身）所需的最少点数，以便将其视为“核心点”——即密集区域中的一个点。

根据这些参数，DBSCAN 将每个点分类为三种类型之一：

- **核心点**: 一个在其 `eps` 邻域内至少有 `min_samples` 个点的点。这些点是聚类的中心。
- **边界点**: 一个在核心点的 `eps` 邻域内，但自身邻域内没有 `min_samples` 个点的点。边界点位于聚类的边缘。
- **噪声点**: 一个既不是核心点也不是边界点的点。这些是离群点或稀疏区域中的点。

聚类是通过连接相邻（彼此之间在 `eps` 距离内）的核心点形成的。然后，边界点被分配给附近核心点所属的聚类。任何通过这些连接从核心点无法到达的点都被标记 (token)为噪声。这里一个重要优点是 DBSCAN 不会强制将每个点归入一个聚类；它会明确识别噪声。

### 使用 Scikit-learn 实现 DBSCAN

Scikit-learn 提供了一个直接的 DBSCAN 实现。让我们看看它如何在 K-Means 可能表现不佳的数据集上工作，比如“双月”数据集。

首先，我们生成数据并将其可视化：

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import seaborn as sns

# 生成样本数据
X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

# 对数据进行缩放（对于像 DBSCAN 这样的基于距离的算法很重要）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 可视化原始缩放后的数据
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], s=50, alpha=0.7)
plt.title('缩放后的双月数据集')
plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```

现在，我们应用 DBSCAN。选择 `eps` 和 `min_samples` 很重要，我们稍后会讨论相关策略。目前，我们先尝试一些合理的值：

```python
# 应用 DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
clusters = dbscan.fit_predict(X_scaled) # 为方便起见，使用 fit_predict

# 获取核心样本索引和标签
core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
labels = dbscan.labels_

# 标签中的聚类数量，如果存在噪声则忽略。
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print(f'估计的聚类数量：{n_clusters_}')
print(f'估计的噪声点数量：{n_noise_}')

# 可视化聚类结果
plt.figure(figsize=(8, 5))

# 绘制非噪声点
unique_labels = set(labels)
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # 噪声点使用黑色
        col = [0, 0, 0, 1]
        marker = 'x'
        markersize = 5
        label = '噪声'
    else:
        marker = 'o'
        markersize = 7
        label = f'聚类 {k}'

    class_member_mask = (labels == k)

    # 绘制核心样本
    xy = X_scaled[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], marker, markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=markersize, label=label if k != -1 else "")

    # 绘制边界样本（非核心）
    xy = X_scaled[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], marker, markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=markersize * 0.6) # 边界点尺寸较小

# 如果存在噪声，单独添加噪声图例项
if n_noise_ > 0:
     plt.plot([], [], 'kx', markersize=5, label='噪声')


plt.title(f'DBSCAN 聚类 (eps=0.3, min_samples=5)')
plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```

> 在缩放后的“双月”数据集上进行 DBSCAN 聚类的可视化。核心点以黑色边缘显示，边界点较小，噪声点用 'x' 标记 (token)。DBSCAN 成功识别了非凸形状。

`labels_` 属性包含每个点的聚类分配。Scikit-learn 使用 `-1` 来表示噪声点。`core_sample_indices_` 属性给出被识别为核心点的点的索引。

### 参数 (parameter)选择：`eps` 和 `min_samples`

DBSCAN 的表现取决于选择适合 `eps` 和 `min_samples` 的值。

- **`min_samples`**: 此参数影响形成聚类所需的最小密度。

  - 一个常见经验法则是根据数据的维度 (D) 设置 `min_samples`，例如 `min_samples >= D + 1` 或 `min_samples >= 2 * D`。对于 2D 数据，`min_samples` 在 3 到 5 之间通常是一个不错的起始值。
  - 较高的值使算法对噪声不那么敏感，但可能合并附近的聚类。较低的值允许密度较小的聚类，但可能对噪声更敏感。考虑数据中预期的密度和噪声水平。领域知识在这里会很有帮助。
- **`eps`**: 此参数决定邻域大小。选择合适的值通常更具挑战性。

  - **K 距离图法**: 一个有用的技术是绘制每个点到其第 k 个最近邻居的距离，其中 k = `min_samples - 1`。
    1. 计算每个点到其第 (min\_samples - 1) 个最近邻居的距离。
    2. 将这些距离按升序排序。
    3. 绘制排序后的距离（y 轴）与点索引（x 轴）的图形。
    4. 寻找图中的“肘部”或“膝盖”——即距离开始急剧上升的区域。此肘部处的距离值通常是 `eps` 的一个良好候选值。它代表一个阈值，在该阈值处，点开始显著远离其邻居，可能表示密集区域与稀疏区域或噪声之间的界限。

让我们实现 K 距离图方法：

```python
from sklearn.neighbors import NearestNeighbors

# 设置 min_samples（例如，根据 2D 数据的 2*D 规则）
min_samples_knn = 4 # k = min_samples - 1 = 5 - 1 = 4

# 计算到第 k 个最近邻居的距离
nn = NearestNeighbors(n_neighbors=min_samples_knn + 1) # 需要 k+1 个邻居来获取 k 个距离
nn.fit(X_scaled)
distances, indices = nn.kneighbors(X_scaled)

# 获取到第 k 个邻居的距离（索引 k）
k_distances = np.sort(distances[:, min_samples_knn], axis=0)

# 绘制 K 距离图
plt.figure(figsize=(8, 5))
plt.plot(k_distances)
plt.title(f'K 距离图 (k={min_samples_knn})')
plt.xlabel('按距离排序的点')
plt.ylabel(f'第 {min_samples_knn} 个最近邻居距离')
plt.grid(True, linestyle='--', alpha=0.6)

# 可选：添加一条线表示潜在的肘部（根据图调整值）
elbow_eps = 0.3
plt.axhline(y=elbow_eps, color='r', linestyle='--', label=f'肘部在 eps={elbow_eps}')
plt.legend()
plt.show()
```

> 缩放后的“双月”数据集的 K 距离图，k=4。y 轴显示每个点到第 4 个最近邻居的距离，按升序排序。距离约为 0.3 处的“肘部”表示 `eps` 的一个合适值。

图中显示了一个明显的肘部。此点（本例中约为 0.3）的距离值表示一个半径，在该半径处密度显著降低，使其成为 `eps` 的良好候选值。请记住，这只是一种经验法则；您可能需要尝试肘部附近的值，并根据目视或使用聚类验证指标（如果适用于无监督环境）来评估结果聚类。

### DBSCAN 的优点和局限

**优点：**

- **无需指定聚类数量：** 与 K-Means 不同，DBSCAN 自动确定聚类数量。
- **发现任意形状：** 擅长识别非凸聚类结构。
- **识别噪声：** 明确将离群点标记 (token)为噪声点，而不是强制它们进入聚类。

**局限：**

- **参数 (parameter)敏感性：** 表现很大程度上取决于 `eps` 和 `min_samples` 的选择。K 距离图有所帮助，但仍可能需要调整。
- **密度可变问题：** 如果聚类的密度显著不同，则存在困难，因为 `eps` 和 `min_samples` 是全局参数。密度较低聚类中的点可能被标记为噪声。OPTICS 或 HDBSCAN 等变体解决了这一局限。
- **依赖距离度量：** 像 K-Means 一样，其有效性依赖于合适的距离度量。在非常高维的空间中（维度灾难），性能会下降，因此特征缩放或降维是重要的前提条件。

### 何时是选择 DBSCAN 的好时机？

DBSCAN 特别适合：

- 预期聚类具有不规则或非球形形状的数据。
- 聚类数量事先未知的情况。
- 包含需要分离的噪声或离群点的数据集。
- 空间数据分析，其中密度是定义聚类的一种自然方式（例如，对地理位置进行分组）。

通过理解密度、核心点和可达性等想法，并使用 K 距离图等技术来指导参数 (parameter)选择，您可以有效地应用 DBSCAN 在复杂、无标签的数据集中找到有意义的模式，而简单的算法可能无法达到此效果。

## 参考资料

- [A density-based algorithm for discovering clusters in large spatial databases with noise](http://www.aaai.org/Library/KDD/1996/kdd96-037.php) — Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu (1996)
  Journal: Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD'96); Publisher: AAAI Press; Pages: 226-231; DOI: [10.1145/233255.233267](https://doi.org/10.1145/233255.233267)
  提出DBSCAN算法的开创性论文，详细阐述了其基于密度的聚类定义和噪声处理方法。
- [sklearn.cluster.DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) — Scikit-learn developers (2024)
  Scikit-learn DBSCAN实现的官方文档，提供API信息、使用示例和参数细节。
- [OPTICS: Ordering Points To Identify the Clustering Structure](https://doi.org/10.1145/304181.304188) — Mihael Ankerst, Markus M. Breunig, Hans-Peter Kriegel, Jörg Sander (1999)
  Journal: ACM SIGMOD Record; Publisher: ACM; Volume: 28; Pages: 49-60; DOI: [10.1145/304181.304188](https://doi.org/10.1145/304181.304188)
  引入了OPTICS，一种在DBSCAN基础上发展而来的方法，通过生成有序表示来处理具有不同密度的集群。

---

[上一节](02-%E5%AE%9E%E7%8E%B0K-Means%E8%81%9A%E7%B1%BB.md) · [下一节](04-%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B%E6%96%B9%E6%B3%95%E7%AE%80%E4%BB%8B.md)

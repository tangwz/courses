---
course: "building-ml-recommendation-system"
chapter: "neighborhood-based-collaborative-filtering"
lesson: "finding-similar-users-items-knn"
sourceId: 7465
sourceUrl: "https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-3-neighborhood-based-collaborative-filtering/finding-similar-users-items-knn"
title: "使用 k-最近邻算法寻找相似用户和物品"
description: "使用 k-最近邻 (k-NN) 算法来识别最相似的用户或物品。"
order: 4
plots: []
sourceHash: "0cd78befacd7801eac416bc43ffcf027deb0f94681a271a7a4d4c2d46298a26e"
sourceCorrections: []
---

识别相似的用户或物品是开发推荐系统的一项基础任务，尤其是在处理用户-物品交互矩阵时。基于邻域的方法，其核心逻辑就在于此。我们需要一种算法，能够接收一个用户或物品，将其与所有其他对象进行比较，并返回其最亲近的“邻居”排名列表。这正是 k-最近邻（k-NN）算法的工作。

### 什么是 k-最近邻？

从本质上讲，k-最近邻是一种直观且易于理解的算法。它的运行原理是：在给定空间中，彼此靠近的事物很可能具有相似性。在用户-物品矩阵的语境下，我们可以将每个用户或每个物品看作多维空间中的一个点。

- 对于**基于用户的协同过滤**，每个用户是一个向量 (vector)，空间的维度是目录中的所有物品。每个维度上的数值是该用户对该物品的评分。
- 对于**基于物品的协同过滤**，每个物品是一个向量，维度是所有的用户。每个维度上的数值是该用户对该物品给出的评分。

k-NN 算法会在这个空间中找到距离目标点最近的 $k$ 个点（即邻居）。这些点之间的“距离”是使用相似度指标计算的，我们将在下一节讨论。参数 (parameter) $k$ 是一个由你选择的数字；例如，如果你设置 $k=5$，算法将找到五个最接近的邻居。

选择合适的 $k$ 值需要权衡。非常小的 $k$（例如 $k=3$）可能会使推荐结果对噪声敏感；一个品味古怪的单一用户就可能对结果产生很大影响。而非常大的 $k$ 可能会使预测过于平滑，从而包含了并非真正相似的邻居，稀释了最相关用户的偏好。最优的 $k$ 通常通过实验和评估来确定。

### 寻找用户的邻域

让我们用一个基于用户的例子来说明。假设我们要为一名叫 Alex 的用户寻找推荐。首先，我们将 Alex 表示为他的评分向量 (vector)。然后，应用 k-NN 算法找到其他评分向量与 Alex 最相似的用户。这些用户构成了 Alex 的邻域。我们的假设是，这个邻域喜欢的、且 Alex 尚未看过的物品，就是非常好的推荐候选项。

> $k=3$ 时 Alex 的邻域。算法根据距离指标识别出 Ben、Chloe 和 David 为最近邻，而 Eva 和 Frank 则因差异太大而未被纳入。

### 寻找物品的邻域

同样的逻辑也适用于基于物品的过滤，只是视角发生了变化。我们不再寻找相似的用户，而是寻找相似的物品。我们将每个物品表示为所有用户对其评分的向量 (vector)。要向 Alex 推荐东西，我们会查看 Alex 已经给出高分的物品，比如“电影 A”。然后，我们使用 k-NN 找到与“电影 A”最相似的 $k$ 部电影（基于全体用户的评分情况）。这些相邻的电影（且 Alex 尚未看过）随后会被推荐给他。

这种基于物品的方法在实际应用中往往更受欢迎。用户的口味可能会随时间变化，但物品之间的关系通常比较稳定。例如，喜欢《星球大战：新希望》的人通常也很有可能喜欢《帝国反击战》。此外，由于物品的数量通常远少于用户的数量，计算物品间的相似度在计算效率上往往更高。

### 使用 Scikit-learn 的实践示例

虽然理解底层机制很重要，但像 Scikit-learn 这样的库提供了高效的 k-NN 实现。让我们看看如何在一个稀疏的用户-物品矩阵中找到第一个用户的邻居。

首先，我们需要一些数据。我们将使用 SciPy 的 `csr_matrix`，这是处理大部分条目为零的稀疏数据的理想格式。

```python
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# 用户-物品矩阵：4 个用户，5 个物品
# 值为 0 表示用户未对该物品评分。
user_item_data = np.array([
    [5, 1, 0, 0, 3],
    [4, 1, 2, 0, 0],
    [0, 0, 5, 4, 1],
    [0, 2, 4, 5, 0]
])

# 创建稀疏矩阵
user_item_sparse = csr_matrix(user_item_data)

print(user_item_sparse)
```

```text
  (0, 0)	5
  (0, 1)	1
  (0, 4)	3
  (1, 0)	4
  (1, 1)	1
  (1, 2)	2
  (2, 2)	5
  (2, 3)	4
  (2, 4)	1
  (3, 1)	2
  (3, 2)	4
  (3, 3)	5
```

现在，让我们配置 `NearestNeighbors` 模型，为每个用户找到 2 个最接近的邻居。我们将使用余弦相似度作为距离指标（注意：Scikit-learn 的 `NearestNeighbors` 使用的是距离，即 $1 - 相似度$）。

```python
# 我们想找到 2 个邻居（不包括用户本人）
k = 3 

# 初始化基于用户过滤的模型
# metric='cosine' 计算余弦距离
nn_model = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')

# 将模型拟合到我们的用户-物品矩阵
nn_model.fit(user_item_sparse)

# 让我们为第一个用户（索引为 0）寻找邻居
target_user_index = 0
target_user_vector = user_item_sparse[target_user_index]

# 寻找最近邻
distances, indices = nn_model.kneighbors(target_user_vector)

print("查询用户索引:", target_user_index)
print("邻居索引:", indices)
print("距离:", distances)
```

输出结果如下所示：

```text
查询用户索引: 0
邻居索引: [[0 1 3]]
距离: [[0.         0.51867123 0.91574913]]
```

以下是如何解读目标用户（索引为 0）的结果：

- **`indices`**：数组 `[[0 1 3]]` 包含了 $k=3$ 个最近邻的索引。第一个结果 `0` 是用户本人，因为其到自身向量 (vector)的距离始终为零。其他邻居是用户 `1` 和用户 `3`。
- **`distances`**：数组 `[[0. 0.518... 0.915...]]` 包含了对应的余弦距离。用户 `1` 较近（距离约为 0.52），而用户 `3` 较远（距离约为 0.92）。

识别出这些邻居后，我们就有了进行预测的基础。接下来的步骤（我们将在后续部分讨论）是使用这些邻居的评分来估算目标用户对未见过物品的评分。我们还将更详尽地分析支持这一过程的相似度指标。

## 参考资料

- [Pattern Recognition and Machine Learning](https://www.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer
  提供机器学习数学基础的综合教材，包括k-最近邻算法原理。
- [Item-based collaborative filtering recommendation algorithms](https://doi.org/10.1145/371920.372071) — Badrul Sarwar, George Karypis, Joseph A Konstan, John Riedl (2001)
  Journal: Proceedings of the 10th International Conference on World Wide Web (WWW '01); Publisher: Association for Computing Machinery (ACM); Pages: 285-295; DOI: [10.1145/371920.372071](https://doi.org/10.1145/371920.372071)
  介绍并评估了基于物品的协同过滤方法的开创性论文，这是推荐系统中广泛采用的方法。
- [sklearn.neighbors.NearestNeighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html) — The scikit-learn team (2024)
  Scikit-learn的k-最近邻算法实现的官方文档，包含参数和查找邻居的用法示例。

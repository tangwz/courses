---
course: "building-ml-recommendation-system"
chapter: "content-based-filtering"
lesson: "generating-user-profiles"
sourceId: 7459
sourceUrl: "https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-2-content-based-filtering/generating-user-profiles"
title: "生成用户画像"
description: "基于用户此前评分或互动过的物品特征来创建用户画像的技术。"
order: 5
plots: []
sourceHash: "a8835a807125a2f415901abb506b96c2fc7e3a9553a8e84c0cd84617ea06fc2d"
sourceCorrections: []
---

为了实现物品推荐，将每个用户的偏好表示为数值向量 (vector)。用户画像通过数值向量汇总了用户喜好的物品类型。通过将用户偏好转换为这种向量格式，就可以识别出与用户画像向量最接近的物品，从而有效地将物品与用户的口味进行匹配。

构建用户画像最直接的方法是聚合用户有过积极互动物品的特征向量。如果一个用户观看并喜欢了几部动作成分很高的科幻电影，那么他们的用户画像向量就应当反映出对这些特定属性的偏好。

### 聚合物品向量 (vector)

让我们考虑一个对多个物品评分过的用户。我们可以通过计算该用户评分过的所有物品向量的平均值来得出他们的画像。

例如，假设一个用户对两部电影《科幻冒险》和《太空歌剧》进行了评分，这两部电影由以下简化后的物品向量表示：

`科幻冒险：[0.8, 0.1, 0.6, 0.0]`（科幻度高，喜剧度低，动作度高）
`太空歌剧：[0.9, 0.0, 0.4, 0.1]`（科幻度高，无喜剧，动作度中等）

一个简单的用户画像可以是这两个向量的对应元素平均值：
`用户画像：[0.85, 0.05, 0.5, 0.05]`

得到的这个向量代表了一个强烈偏好科幻和动作，而对喜剧几乎没有兴趣的用户，这准确地综合了他们的观看历史。

### 使用评分作为权重 (weight)

简单平均法平等地对待每一次互动。但在许多系统中，我们可以获得显式反馈，例如 5 星评分。用户评分为 5 分的电影对画像的影响应该比评分为 3 分的电影更大。我们可以通过计算加权平均值来实现这一点，其中权重就是用户给出的评分。

用户 $u$ 的画像向量 (vector) $\vec{p_u}$ 的公式为：

$\vec{p_u} = \frac{\sum_{i \in I_u} r_{ui} \cdot \vec{v_i}}{\sum_{i \in I_u} |r_{ui}|}$

其中：

- $I_u$ 是用户 $u$ 评分过的物品集合。
- $r_{ui}$ 是用户 $u$ 给物品 $i$ 的评分。
- $\vec{v_i}$ 是物品 $i$ 的特征向量。

这种计算方法确保了评分较高的物品对最终用户画像向量的贡献更大，从而更准确地反映其偏好。

> 用户给出积极评分的物品（电影 A 和电影 C）的特征向量，以其评分为权重进行组合，生成单一的用户画像向量。用户未评分的电影 B 则不产生贡献。

### 实践应用

让我们看看在实践中如何使用 `pandas` 和 `NumPy` 来实现。假设你有一个包含 `['userId', 'movieId', 'rating']` 列的 DataFrame `ratings`，以及另一个索引对应 `movieId` 的 DataFrame 或矩阵 `item_vectors`。

首先，提取特定用户的评分，例如 `userId=1` 的用户：

```python
# 假设 item_vectors 是一个以 movieId 为索引、特征为列的 DataFrame
# ratings 是用户评分 DataFrame。
user_id = 1
user_ratings = ratings[ratings['userId'] == user_id]
```

接着，我们从 `item_vectors` 矩阵中仅选择该用户评分过的物品，并剔除可能缺失特征向量 (vector)的物品。

```python
# 获取用户评分过的电影向量
user_item_vectors = item_vectors.loc[user_ratings['movieId']]

# 对齐评分以用作权重
weights = user_ratings['rating'].values
```

现在，我们可以使用 NumPy 的 `average` 函数来计算物品向量各列的加权平均值，该函数可以方便地接受 `weights` 参数 (parameter)。

```python
import numpy as np

# 计算物品向量的加权平均值
user_profile_vector = np.average(user_item_vectors, axis=0, weights=weights)

print(user_profile_vector)
# 输出: array([0.58, 0.12, 0.81, ...])
```

得到的 `user_profile_vector` 是一个概括了该用户偏好的单一向量。它现在处于与物品向量完全相同的多维空间中。这是迈出的重要一步，因为它允许我们使用之前讨论过的余弦相似度指标，直接将用户画像向量与所有物品向量进行对比。通过找到与该画像最相似的物品，我们就可以生成一份排序后的推荐列表，这正是我们在下一节中要做的内容。

## 参考资料

- [Recommender Systems: An Introduction](http://link.springer.com/chapter/10.1007/978-1-4899-7637-6_1) — Francesco Ricci, Lior Rokach, Bracha Shapira (2015)
  Publisher: Springer; Pages: 1-34; DOI: [10.1007/978-1-4899-7637-6_1](https://doi.org/10.1007/978-1-4899-7637-6_1)
  本书的第二版是推荐系统领域的标准参考，涵盖了基于内容的过滤方法和从物品特征构建用户档案。
- [Learning and Revising User Profiles for Content-Based Filtering](https://dl.acm.org/doi/10.5555/645524.657512) — Michael J. Pazzani, Daniel Billsus (1997)
  Journal: Proceedings of the Fourteenth International Conference on Machine Learning, 1997; Publisher: Association for Computing Machinery; Pages: 322-330; DOI: [10.5555/645524.657512](https://doi.org/10.5555/645524.657512)
  一篇基础性论文，讨论了如何学习和更新用于内容过滤的用户档案，包括聚合技术。
- [Mining of Massive Datasets](http://www.mmds.org/) — Jure Leskovec, Anand Rajaraman, Jeff Ullman (2014)
  Publisher: Cambridge University Press
  本书的第9章概述了推荐系统，包括基于内容的方法和使用物品特征向量构建用户档案。

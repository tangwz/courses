---
course: "building-ml-recommendation-system"
chapter: "neighborhood-based-collaborative-filtering"
lesson: "practice-implementing-item-based-filter"
sourceId: 7469
sourceUrl: "https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-3-neighborhood-based-collaborative-filtering/practice-implementing-item-based-filter"
title: "动手实践：实现基于物品的过滤"
description: "动手构建基于物品的协同过滤推荐系统的实操指南。"
order: 8
plots: []
sourceHash: "d63703d25db40be7683601ad8a984444f69a0f67dca4430a3bf4eb42f35da0f9"
sourceCorrections: []
---

从零开始实现基于物品的协同过滤，可以展示邻域协同过滤的实际应用。在实践中，这种方法通常比基于用户的方法更受欢迎，因为物品之间的相似性往往比用户之间的相似性随时间变化更小。用户的口味可能会变，但例如两部电影之间的关系是相对静态的。

我们将使用著名的 MovieLens 数据集的一个子集。我们的目标是创建一个函数，在给定电影名称的情况下，根据用户的评分模式返回最相似的电影列表。

### 准备数据

首先，我们需要加载数据并将其构建成前面讨论过的用户-物品交互矩阵。假设我们有两个 CSV 文件：包含 `movieId`（电影ID）和 `title`（标题）的 `movies.csv`，以及包含 `userId`（用户ID）、`movieId` 和 `rating`（评分）的 `ratings.csv`。

我们先将这些数据加载到 pandas DataFrame 中并进行合并。

```python
import pandas as pd
import numpy as np

# 加载数据集
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# 合并数据，使评分与电影标题对应
df = pd.merge(ratings_df, movies_df, on='movieId')

print(df.head())
```

这给了我们一个整洁的格式，但还不是我们需要的用户-物品矩阵。我们可以使用 `pivot_table` 函数来创建这个矩阵，以用户为行索引，电影标题为列名，评分为数值。

```python
# 创建用户-物品交互矩阵
user_item_matrix = df.pivot_table(index='userId', columns='title', values='rating')

# 用 0 填充缺失值
user_item_matrix.fillna(0, inplace=True)

print(user_item_matrix.head())
```

用 0 填充缺失值是一种简化处理。它意味着如果用户没有对某个物品评分，他们的偏好是中立的。虽然存在更高级的填补技术，但这是一种常见且有效的起点。生成的矩阵非常稀疏，这意味着其大部分单元格都是零，这在同类数据中很典型。

### 计算物品与物品的相似度

有了矩阵后，我们现在可以计算每对电影之间的相似度。我们将使用余弦相似度，它衡量两个非零向量 (vector)之间夹角的余弦值。在我们的场景中，每部电影都是一个向量，其分量是每个用户的评分。

两个物品 $i$ 和 $j$ 之间的余弦相似度公式为：


$$
\text{相似度}(i, j) = \cos(\theta) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}
$$


这里，$\mathbf{v}_i$ 是物品 $i$ 的评分向量。

虽然我们可以手动计算，但使用 scikit-learn 等库效率更高。`cosine_similarity` 要求样本位于行中，而我们的物品（电影）目前是列。我们必须先转置矩阵。

```python
from sklearn.metrics.pairwise import cosine_similarity

# 转置矩阵，使物品位于行中
item_user_matrix = user_item_matrix.T

# 计算余弦相似度矩阵
item_similarity_matrix = cosine_similarity(item_user_matrix)

# 将结果转换为 DataFrame 以提高可读性
item_similarity_df = pd.DataFrame(item_similarity_matrix, 
                                  index=user_item_matrix.columns, 
                                  columns=user_item_matrix.columns)

print(item_similarity_df.head())
```

`item_similarity_df` 是一个方阵，其中每一行和每一列都代表一部电影。`item_similarity_df.loc['电影 A', '电影 B']` 处的值是电影 A 和电影 B 之间的余弦相似度得分。显然，对角线上的值都为 1，因为每部电影与自身完全相似。

### 构建推荐函数

最后一步：创建一个生成推荐的函数。该函数将接收电影标题，并从相似度矩阵中找到最相似的电影。

> 生成基于物品的推荐的过程，从选择种子物品到产生最终的排序列表。

这是实现该逻辑的 Python 函数。

```python
def get_item_based_recommendations(movie_title, similarity_matrix_df, n_recommendations=5):
    """
    生成基于物品的协同过滤推荐。

    参数:
        movie_title (str): 要获取推荐的电影。
        similarity_matrix_df (pd.DataFrame): 物品相似度矩阵。
        n_recommendations (int): 返回的推荐数量。

    返回:
        推荐电影标题列表。
    """
    # 获取该电影的相似度得分
    similar_scores = similarity_matrix_df[movie_title]

    # 根据相似度得分对电影进行排序（降序）
    similar_scores = similar_scores.sort_values(ascending=False)

    # 排除电影自身（相似度为 1.0）
    similar_scores = similar_scores.drop(movie_title)

    # 返回前 N 个电影标题
    return similar_scores.head(n_recommendations).index.tolist()
```

### 测试推荐系统

让我们用几个例子测试一下函数，看看结果。我们将为一部经典科幻电影和一部著名的犯罪剧情片寻找推荐。

```python
# 获取“星球大战 IV：新希望 (1977)”的推荐
sw_recommendations = get_item_based_recommendations(
    'Star Wars: Episode IV - A New Hope (1977)', 
    item_similarity_df, 
    n_recommendations=5
)
print("'星球大战 IV：新希望 (1977)' 的推荐结果：")
print(sw_recommendations)

print("\n" + "="*50 + "\n")

# 获取“教父 (1972)”的推荐
godfather_recommendations = get_item_based_recommendations(
    'Godfather, The (1972)', 
    item_similarity_df, 
    n_recommendations=5
)
print("'教父 (1972)' 的推荐结果：")
print(godfather_recommendations)
```

你应该会看到非常直观的结果。对于《星球大战》，推荐结果可能会包括该系列的其他影片，如《帝国反击战》和《绝地归来》，以及其他同期的科幻大片。对于《教父》，你可能会看到《教父 2》和其他广受好评的犯罪电影。

这些结果是在完全不了解类型、导演或情节的情况下生成的。它们纯粹源自成千上万用户的集体行为。这展示了协同过滤的效果：它基于共同的口味发现物品之间的关联。

### 总结与预告

在这次实践中，你成功构建了一个完整的基于物品的协同过滤推荐系统。你将原始的用户-物品交互数据转换成了结构化矩阵，使用余弦距离计算了物品间的相似度，并开发了一个生成排序推荐的函数。

然而，这种邻域法也有局限性。随着物品数量的增加，它的扩展性会遇到挑战，因为计算完整的相似度矩阵可能非常消耗资源。此外，它无法推荐没有任何评分的物品。在下一章中，我们将学习基于模型的技术，特别是矩阵分解，它可以帮助应对这些挑战，并通常能产生更准确、更个性化的推荐。

## 参考资料

- [Item-based collaborative filtering recommendation algorithms](https://doi.org/10.1145/371920.372071) — Badrul Sarwar, George Karypis, Joseph A. Konstan, John Riedl (2001)
  Journal: Proceedings of the 10th International Conference on World Wide Web (WWW 2001); Publisher: Association for Computing Machinery (ACM); Pages: 285-295; DOI: [10.1145/371920.372071](https://doi.org/10.1145/371920.372071)
  这篇重要论文介绍了并深入评估了基于项目的协同过滤算法，为本节所示的实际实现提供了理论和实证基础。
- [sklearn.metrics.pairwise.cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) — scikit-learn developers (2024)
  scikit-learn `cosine_similarity` 函数的官方文档，本节使用该函数计算电影向量之间的相似度。
- [pandas.DataFrame.pivot_table](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot_table.html) — pandas development team (2024)
  pandas `DataFrame.pivot_table` 的官方文档，这是本节中数据准备的关键部分，用于构建用户-物品交互矩阵。

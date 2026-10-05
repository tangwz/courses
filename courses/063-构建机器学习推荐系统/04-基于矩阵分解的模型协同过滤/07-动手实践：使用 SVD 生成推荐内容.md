# 动手实践：使用 SVD 生成推荐内容

来源：[原文](https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-4-model-based-collaborative-filtering/practice-recommendations-with-svd)

[返回章节目录](README.md) · [返回课程目录](../README.md)

我们将使用 `surprise` 库构建一个完整的推荐引擎。目标是在 MovieLens 数据集上训练 SVD 模型，并利用该模型为特定用户生成电影推荐排名列表。

### 设置环境与数据

首先，请确保已安装必要的库。我们将使用 `pandas` 进行数据操作，并使用 `surprise` 实现 SVD 模型。我们将使用广泛采用的 MovieLens 100k 数据集。

让我们先将用户评分和电影标题加载到 pandas DataFrame 中。

```python
import pandas as pd
from surprise import Dataset, Reader, SVD

# 加载电影和评分数据集
movies_df = pd.read_csv('ml-latest-small/movies.csv')
ratings_df = pd.read_csv('ml-latest-small/ratings.csv')

# 显示每个 dataframe 的前几行以进行检查
print("Movies DataFrame:")
print(movies_df.head())

print("\nRatings DataFrame:")
print(ratings_df.head())
```

`ratings_df` 包含了格式为 `(userId, movieId, rating)` 的用户-物品交互数据。这是我们协同过滤模型的主要输入。`movies_df` 稍后将用于将推荐结果中的 `movieId` 映射回易于阅读的电影标题。

### 为 Surprise 库准备数据

`surprise` 库要求数据符合特定格式。我们需要定义一个 `Reader` 对象来解析评分数据，并指定评分范围（在本例中，MovieLens 数据集的评分为 0.5 到 5.0）。然后，我们将 DataFrame 加载到 `surprise` 的 `Dataset` 对象中。

```python
# Reader 对象帮助解析文件或 dataframe
reader = Reader(rating_scale=(0.5, 5.0))

# 从 pandas dataframe 加载数据
# 列的顺序必须是：用户、物品、评分
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
```

通过这种方式加载数据，`surprise` 就能理解其结构并将其用于模型训练和预测。

### 训练 SVD 模型

数据准备就绪后，我们现在可以实例化并训练 SVD 模型。在生成最终推荐时，通常做法是在整个数据集上训练模型，以捕捉尽可能多的用户-物品交互模式。`surprise` 库通过 `build_full_trainset()` 方法使这一过程变得非常直接。

```python
# 从整个数据集构建训练集
trainset = data.build_full_trainset()

# 实例化 SVD 算法
svd_model = SVD(n_factors=100, n_epochs=20, random_state=42)

# 在训练集上训练算法
print("正在训练 SVD 模型...")
svd_model.fit(trainset)
print("训练完成。")
```

在这里，我们初始化了具有 100 个隐因子 (`n_factors`) 的 SVD 算法，并将其设置为在优化过程中对数据进行 20 次迭代 (`n_epochs`)。在 `fit` 方法执行完毕后，`svd_model` 就是我们训练好的模型，可以开始进行预测了。

### 为用户生成 Top-N 推荐

推荐系统的主要功能不仅仅是预测单个物品的评分，而是生成用户可能喜欢的物品排名列表。为此，我们需要对用户*尚未*看过的所有物品进行评分预测，然后对这些预测结果进行排序，找出表现最好的物品。

下图说明了生成推荐的工作流程。

> 该过程从目标用户和训练好的模型开始。它会识别出用户尚未互动的物品，为每个物品预测分数，最后返回排序后的前 N 项推荐列表。

让我们在 Python 函数中实现这一逻辑。该函数将用户 ID 和我们训练好的模型作为输入，并返回前 10 部推荐电影。

```python
def get_top_n_recommendations(user_id, model, ratings_df, movies_df, n=10):
    """
    使用训练好的 SVD 模型为指定用户生成 top-N 推荐。
    """
    # 获取所有电影 ID 的列表
    all_movie_ids = ratings_df['movieId'].unique()

    # 获取该用户已评分的电影 ID 列表
    rated_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()

    # 获取该用户未评分的电影 ID 列表
    unrated_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids]

    # 为所有未评分电影预测评分
    predictions = [model.predict(user_id, movie_id) for movie_id in unrated_movie_ids]

    # 按预估评分降序排列预测结果
    predictions.sort(key=lambda x: x.est, reverse=True)

    # 获取前 N 个电影 ID
    top_n_predictions = predictions[:n]
    top_n_movie_ids = [pred.iid for pred in top_n_predictions]

    # 获取前 N 个电影 ID 对应的电影标题
    recommended_movies = movies_df[movies_df['movieId'].isin(top_n_movie_ids)]

    return recommended_movies

# 让我们为 ID 为 50 的用户生成推荐
user_id_to_recommend = 50
recommendations = get_top_n_recommendations(user_id_to_recommend, svd_model, ratings_df, movies_df)

print(f"\n为用户 {user_id_to_recommend} 推荐的前 10 部电影：")
print(recommendations)
```

让我们看看用户 50 的输出结果。

**为用户 50 推荐的前 10 部电影：**

|  | movieId | title | genres |
| --- | --- | --- | --- |
| 14 | 15 | As Good as It Gets (1997) | Comedy, Drama, Romance |
| 27 | 28 | Persuasion (1995) | Drama, Romance |
| 163 | 194 | Stargate (1994) | Action, Adventure, Sci-Fi |
| 176 | 208 | Waterworld (1995) | Action, Adventure, Sci-Fi |
| 227 | 266 | While You Were Sleeping (1995) | Comedy, Romance |
| 315 | 357 | Four Weddings and a Funeral (1994) | Comedy, Romance |
| 451 | 515 | Remains of the Day, The (1993) | Drama, Romance |
| 465 | 529 | My Life as a Dog (Mitt liv som hund) (1985) | Comedy, Drama |
| 470 | 534 | Sense and Sensibility (1995) | Drama, Romance |
| 520 | 608 | Fargo (1996) | Comedy, Crime, Drama, Thriller |

我们的函数成功生成了一份个性化推荐列表。该模型从该用户过去的评分中学习了其偏好，并识别出他们可能喜欢的其他电影。这展示了矩阵分解的效用；即使不了解电影内容的任何信息，模型也能根据在用户-物品交互矩阵中发现的隐因子来推断匹配度。

至此，你已成功构建并使用了一个基于模型的协同过滤系统。下一步是对其性能进行评估，我们将在接下来的章节中介绍。

## 参考资料

- [Matrix Factorization Techniques for Recommender Systems](https://doi.org/10.1109/MC.2009.263) — Yehuda Koren, Robert Bell, and Chris Volinsky (2009)
  Journal: Computer; Publisher: IEEE; Volume: 42; Pages: 30-37; DOI: [10.1109/MC.2009.263](https://doi.org/10.1109/MC.2009.263)
  介绍矩阵分解（包括SVD）在推荐系统协同过滤中应用的奠基性论文。
- [Surprise: A Python scikit for building and analyzing recommender systems](https://surprise.readthedocs.io/en/stable/) — Nicolas Hug (2024)
  `surprise` Python库的官方文档，提供构建和评估推荐系统的指南和API参考。
- [Recommender Systems: An Introduction](https://books.google.com/books?id=f_d_mR74u0gC) — Dietmar Jannach, Markus Zanker, Alexander Felfernig, and Gerhard Friedrich (2011)
  Publisher: Cambridge University Press; DOI: [10.1017/CBO9781316104885](https://doi.org/10.1017/CBO9781316104885)
  一本全面的教科书，涵盖各种推荐系统技术，包括协同过滤和矩阵分解。第二版。

---

[上一节](06-%E4%BD%BF%E7%94%A8%E5%BA%93%E8%BF%9B%E8%A1%8C%E7%9F%A9%E9%98%B5%E5%88%86%E8%A7%A3.md) · [下一节](../05-%E8%AF%84%E4%BC%B0%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F/01-%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E8%AF%84%E4%BC%B0%E7%9A%84%E4%BB%B7%E5%80%BC.md)

---
course: "building-ml-recommendation-system"
chapter: "model-based-collaborative-filtering"
lesson: "using-libraries-for-matrix-factorization"
sourceId: 7475
sourceUrl: "https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-4-model-based-collaborative-filtering/using-libraries-for-matrix-factorization"
title: "使用库进行矩阵分解"
description: "关于如何使用 Surprise 等流行的 Python 库高效实现矩阵分解的指南。"
order: 6
plots: []
sourceHash: "9e0f81d40c706e48066aa23209e49fe96e7c45264d685a527ac9bc271aa4a671"
sourceCorrections: []
---

从基本原理开始实现矩阵分解算法可以帮助我们很好地理解其内部机制。然而，在实际应用中，使用专门的库通常具有明显优势。这些库提供了经过优化且经过充分测试的常用算法实现，让你能够专注于模型设计和评估，而无需关注底层的实现细节。

在 Python 中构建推荐系统时，`surprise` 是一个应用广泛且有效的库。它的名字是 Simple Python RecommendatIon System Engine（简单 Python 推荐系统引擎）的缩写。它提供了一系列即插即用的预测算法和评估工具，其接口对于熟悉 scikit-learn 的人来说非常亲切。

### Surprise 入门

`surprise` 库简化了训练和测试推荐模型的过程。它拥有自己的数据结构，但可以与你熟悉的 `pandas` 库配合使用。典型的工作流包含三个主要组件：`Reader`、`Dataset` 以及算法本身。

- **`Reader`**：该对象用于解析文件或 DataFrame。其主要用途是定义数据集的评分范围。例如，如果用户评分范围是 1 到 5 星，你可以在这里指定。
- **`Dataset`**：这是 `surprise` 中的核心数据结构。它接收原始数据，并在 `Reader` 的帮助下将其转换为库算法可以处理的格式。
- **算法**：这是你想要训练的模型，例如 `SVD`、`KNNBasic` 或 `NMF`。

让我们看看如何将 pandas DataFrame 中的数据加载到 `surprise` 数据集中。假设你有一个名为 `ratings_df` 的 DataFrame，包含 `userID`、`itemID` 和 `rating` 列。

```python
import pandas as pd
from surprise import Dataset, Reader

# 示例评分 DataFrame
data = {'userID': [1, 1, 2, 2, 3, 3],
        'itemID': [101, 102, 101, 103, 102, 104],
        'rating': [5, 3, 4, 2, 5, 4]}
ratings_df = pd.DataFrame(data)

# 1. 使用评分范围初始化 Reader
# 我们的评分范围是 1 到 5
reader = Reader(rating_scale=(1, 5))

# 2. 将 DataFrame 中的数据加载到 Dataset 对象中
# 列的顺序必须是：用户、物品、评分
data = Dataset.load_from_df(ratings_df[['userID', 'itemID', 'rating']], reader)
```

只需几行代码，你的数据就可以用于训练了。

### 训练 SVD 模型

在 `surprise` 中，`SVD` 类实现了我们之前讨论过的矩阵分解技术。它不是线性代数中“纯粹”的奇异值分解（Singular Value Decomposition），而是一个使用随机梯度下降 (gradient descent) (SGD) 优化来查找隐因子矩阵 $P$ 和 $Q$ 的基于模型的算法。

当你实例化 `SVD` 模型时，可以配置其超参数 (parameter) (hyperparameter)，这些参数直接对应于我们之前章节中的概念。

- `n_factors`：隐因子的数量，$K$。
- `n_epochs`：SGD 优化器遍历整个训练数据集的次数。
- `lr_all`：所有参数的学习率，$\gamma$。
- `reg_all`：应用于所有参数的正则化 (regularization)项，$\lambda$。

下图展示了 `surprise` 库中训练模型的典型数据流。

> 该过程从 DataFrame 中的原始数据开始，将其转换为 `surprise` 数据集。然后使用该数据集构建训练集，并在其上拟合 SVD 等算法。训练完成后，模型即可进行预测。

加载数据后，你可以从中构建一个“训练集 (trainset)”。训练集是 `surprise` 算法进行训练的数据结构。然后，实例化你的 `SVD` 模型并调用其 `fit()` 方法。

```python
from surprise import SVD

# 从整个数据集中构建训练集
trainset = data.build_full_trainset()

# 实例化 SVD 算法
# 我们将使用 50 个隐因子，迭代 20 次，并使用默认的学习率和正则化参数
algo = SVD(n_factors=50, n_epochs=20, random_state=42)

# 在训练集上训练模型
algo.fit(trainset)
```

模型现在已经训练完成。`algo` 对象包含所有用户和物品的学习到的隐因子。

### 进行预测

拟合模型后，你可以使用 `predict()` 方法预测任何用户-物品对的评分。该方法使用用户隐因子向量 (vector) $p_u$ 和物品隐因子向量 $q_i$ 的点积来计算估计评分 $\hat{r}_{ui}$。


$$
\hat{r}_{ui} = p_u \cdot q_i
$$


让我们预测用户 `3` 对物品 `101` 的评分。

```python
# 预测特定用户和物品的评分
prediction = algo.predict(uid=3, iid=101)

# 打印预测详情
print(f"用户 ID:      {prediction.uid}")
print(f"物品 ID:      {prediction.iid}")
print(f"估算评分: {prediction.est:.4f}")
```

输出结果如下所示：

```
用户 ID:      3
物品 ID:      101
估算评分: 4.2315
```

`predict()` 方法返回一个 `Prediction` 对象，其中包含一些有用的信息，包括用户 ID、物品 ID、原始评分（如果数据集中存在）以及估计评分 `est`。这个估计评分是模型对用户如何评价该物品的最佳推测。

通过使用像 `surprise` 这样的库，我们只需几行代码就实现了数据加载、训练复杂的矩阵分解模型并生成预测。这省去了实现 SGD 优化循环和正则化 (regularization)的繁琐工作，让我们能够快速构建和测试完整的推荐系统。

在下一节中，我们将在此基础上进一步生成用户的 Top-N 推荐列表，这也是大多数推荐系统的最终目标。

## 参考资料

- [surprise library documentation](https://surprise.readthedocs.io/en/stable/) — Nicolas Hug and surprise contributors (2023)
  `surprise` 库的官方文档，涵盖 API 详情、安装以及使用矩阵分解构建推荐系统的示例。
- [Recommender Systems: The Textbook](https://link.springer.com/book/10.1007/978-3-319-29659-3) — Charu C. Aggarwal (2016)
  Publisher: Springer; DOI: [10.1007/978-3-319-29659-3](https://doi.org/10.1007/978-3-319-29659-3)
  一本全面的教材，涵盖了推荐系统的基本概念和高级主题，其中包含关于协同过滤和各种矩阵分解模型的详细章节。

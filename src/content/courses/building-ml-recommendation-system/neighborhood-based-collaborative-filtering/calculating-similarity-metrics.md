---
course: "building-ml-recommendation-system"
chapter: "neighborhood-based-collaborative-filtering"
lesson: "calculating-similarity-metrics"
sourceId: 7466
sourceUrl: "https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-3-neighborhood-based-collaborative-filtering/calculating-similarity-metrics"
title: "计算相似度指标"
description: "在协同过滤语境下，关于皮尔逊相关系数和余弦相似度等常用相似度指标的指南。"
order: 5
plots: []
sourceHash: "983d713f5e071b711a29b5d04f06dce8df80e64a41fa0c22234b7c660625a74a"
sourceCorrections: []
---

为了构建有效的推荐系统，量化 (quantization)用户或物品之间的关系是必不可少的。相似度指标提供了一种衡量它们相似程度的正式方法。这些函数通过比较代表两个用户或两个物品的向量 (vector)，并生成一个表示它们相似程度的分数。指标的选择是一项设计决策，会显著影响推荐系统的性能。

我们将重点讨论协同过滤中最常用的两种相似度指标：余弦相似度（Cosine Similarity）和皮尔逊相关系数（Pearson Correlation）。

### 余弦相似度：衡量方向

余弦相似度衡量两个非零向量 (vector)之间夹角的余弦值。在推荐系统的语境下，它评估的是两个用户或物品评分向量的方向，而不是它们的大小。这非常有用，因为它捕捉了偏好模式，忽略了评分尺度的差异。例如，一个用户可能是一个严苛的评论者，给所有东西打分都在 1 到 3 之间，而另一个用户则比较大方，打分在 3 到 5 之间。如果他们在不同物品上的评分“形状”相似，余弦相似度仍然可以将他们识别为相似用户。

两个向量 $A$ 和 $B$ 之间的余弦相似度公式为：


$$
\text{相似度}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}
$$


分值范围从 -1（完全相反）到 1（完全相同），0 表示正交或无相关性。在大多数评分是非负数的评分场景中，分值范围在 0 到 1 之间。

让我们看一个简单的基于用户的例子，Alice 和 Bob 两个用户都对三部电影进行了评分。

| 电影 | Alice 的评分 | Bob 的评分 |
| --- | --- | --- |
| 电影 1 | 5 | 4 |
| 电影 2 | 3 | 2 |
| 电影 3 | 4 | 3 |

他们的评分向量分别是 $A = [5, 3, 4]$ 和 $B = [4, 2, 3]$。计算余弦相似度的步骤如下：

1. **计算点积 ($A \cdot B$):**
   $(5 \times 4) + (3 \times 2) + (4 \times 3) = 20 + 6 + 12 = 38$
2. **计算每个向量的模 ($\|A\|$ 和 $\|B\|$):**
   $\|A\| = \sqrt{5^2 + 3^2 + 4^2} = \sqrt{25 + 9 + 16} = \sqrt{50} \approx 7.07$
   $\|B\| = \sqrt{4^2 + 2^2 + 3^2} = \sqrt{16 + 4 + 9} = \sqrt{29} \approx 5.39$
3. **计算相似度:**
   $\text{相似度} = \frac{38}{7.07 \times 5.39} \approx \frac{38}{38.11} \approx 0.997$

0.997 的分数表明 Alice 和 Bob 在电影方面的品味非常相似，因为他们的评分向量指向几乎相同的方向。

### 皮尔逊相关系数：考虑评分偏差

虽然余弦相似度很有效，但它有一个局限性：它没有考虑评分尺度的差异。例如，如果用户 A 对物品的评分为 [1, 2, 3]，用户 B 的评分为 [3, 4, 5]，他们具有完美的线性关系，但余弦相似度无法像预期的那样强烈地捕捉到这一点。

皮尔逊相关系数通过先对数据进行中心化处理来解决这个问题。它确定两个变量之间的线性相关程度。在我们的案例中，它衡量两个用户给出的评分或两个物品收到的评分之间的线性关系。它实际上是余弦相似度的均值中心化版本。

两个用户 $u$ 和 $v$ 之间的皮尔逊相关系数公式为：


$$
\text{相关系数}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{u,i} - \bar{r}_u)(r_{v,i} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{u,i} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{v,i} - \bar{r}_v)^2}}
$$


其中：

- $I_{uv}$ 是用户 $u$ 和用户 $v$ 都评分过的物品集合。
- $r_{u,i}$ 是用户 $u$ 对物品 $i$ 的评分。
- $\bar{r}_u$ 是用户 $u$ 给出的平均评分。

分值范围从 -1 到 1，其中 1 表示完美的正线性关系，-1 表示完美的负线性关系，0 表示没有线性关系。

让我们用一个新用户 Carol 的例子来看看，她是一个比较严苛的评论者。

| 电影 | Alice 的评分 | Carol 的评分 |
| --- | --- | --- |
| 电影 1 | 5 | 3 |
| 电影 2 | 3 | 1 |
| 电影 3 | 4 | 2 |

1. **计算平均评分:**

   - Alice 的平均分：$\bar{r}_{Alice} = (5 + 3 + 4) / 3 = 4$
   - Carol 的平均分：$\bar{r}_{Carol} = (3 + 1 + 2) / 3 = 2$
2. **创建均值中心化评分向量 (vector):**

   - Alice 的中心化向量：$[5-4, 3-4, 4-4] = [1, -1, 0]$
   - Carol 的中心化向量：$[3-2, 1-2, 2-2] = [1, -1, 0]$

如你所见，它们的均值中心化向量是相同的。如果我们计算皮尔逊相关系数（或这些中心化向量的余弦相似度），结果将是 1。这表明存在完美的线性相关，捕捉到了即使原始分数不同，他们的偏好结构也是完全一致的事实。皮尔逊相关系数通过消除个人评分偏差，成功识别了他们的相似性。

> 该图说明了均值中心化如何对齐 (alignment)用户评分。在中心化之前，Alice 和 Carol 的评分处于不同的位置。减去各自的平均评分后，他们的偏好向量完全对齐，显示出潜在的相似性。

### 余弦相似度 vs. 皮尔逊相关系数：该选哪一个？

在这两个指标之间做出选择通常取决于数据的性质。

- **使用皮尔逊相关系数**：当你处理显式评分数据（例如 1-5 星）并且担心用户评分偏差时。对于这种类型的数据，它通常更有效，因为它消除了用户倾向于打高分或打低分的影响。
- **使用余弦相似度**：当你的数据非常稀疏，或者数值的大小本身具有意义时。对于隐式反馈数据（交互是二元的，例如看过/没看过），它也是一个不错的选择，因为此时不存在评分偏差。此外，余弦相似度在计算上通常比皮尔逊相关系数更简单。

### 使用 Scikit-learn 实现

在实践中，你很少需要从头开始实现这些公式。像 Scikit-learn 这样的库提供了计算相似度矩阵的高效函数。例如，你可以使用一行代码为用户-物品矩阵 `R_items` 中的所有物品计算余弦相似度矩阵。

```python
from sklearn.metrics.pairwise import cosine_similarity

# 假设 R_items 是一个行表示物品、列表示用户的矩阵
# 它可以是 pandas DataFrame 或 NumPy 数组
# 例如：
# R_items = [
#   [5, 3, 0],  # 用户 A, B, C 对物品 1 的评分
#   [4, 0, 2],  # 物品 2 的评分
#   [0, 2, 5]   # 物品 3 的评分
# ]

# 计算所有物品对之间的相似度
item_similarity_matrix = cosine_similarity(R_items)

print(item_similarity_matrix)
# 输出可能类似于：
# [[1.         0.78       0.        ]
#  [0.78       1.         0.55      ]
#  [0.         0.55       1.        ]]
```

该矩阵为你提供了每对物品之间的相似度分数，构成了寻找最近邻的基础。

有了这些量化 (quantization)相似度的方法，你现在就可以为任何用户或物品寻找最相关的邻居。下一步是利用来自这些邻居的信息来生成具体的预测。

## 参考资料

- [Recommender Systems: An Introduction](https://doi.org/10.1007/978-1-0716-2197-4) — Francesco Ricci, Lior Rokach, Bracha Shapira (2022)
  Publisher: Springer; DOI: [10.1007/978-1-0716-2197-4](https://doi.org/10.1007/978-1-0716-2197-4)
  一本关于协同过滤、相似性度量及其在推荐系统应用方面的综合性书籍。（第4版）
- [Empirical Analysis of Predictive Algorithms for Collaborative Filtering](https://arxiv.org/pdf/1301.7363) — John S. Breese, David Heckerman, Carl Kadie (1998)
  Journal: Proceedings of the Fourteenth Conference on Uncertainty in Artificial Intelligence (UAI'98); Publisher: Morgan Kaufmann; Pages: 43-52; DOI: [10.5555/2070081.2070087](https://doi.org/10.5555/2070081.2070087)
  一篇基础性论文，对各种预测算法（包括基于相似性度量的算法）在协同过滤中的应用进行了实证比较。
- [sklearn.metrics.pairwise.cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) — scikit-learn developers (2024)
  scikit-learn库中`cosine_similarity`函数的官方文档，对实际应用至关重要。

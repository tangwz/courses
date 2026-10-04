# 平均精度均值 (MAP)

来源：[原文](https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-5-evaluating-recommendation-systems/mean-average-precision)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Precision@k 衡量了前 k 个推荐项中包含多少相关项目，但它有一个明显的局限：它忽略了这些项目的排列顺序。根据 Precision@5 的计算规则，将相关项目排在第 1 位的推荐列表与将其排在第 5 位的列表得分完全相同。在大多数应用场景中，这并不理想。用户更有可能与列表顶部的项目进行交互，因此，一个优秀的推荐系统应当因将相关项目排在更高位置而获得奖励。

这正是平均精度均值 (MAP) 旨在解决的问题。它是一种排序指标，对于将相关项目排在推荐列表较后位置的模型会给予较大的减分。

### 从精确度到平均精度

要理解 MAP，我们首先需要了解**平均精度 (Average Precision, AP)**，它是针对每个用户分别计算的。单个用户的平均精度是在每个包含相关项目的排名位置 `k` 处计算出的 Precision@k 值的平均值。

让我们通过一个例子来说明。假设我们的模型为某位用户生成了包含 6 部电影的推荐排名列表。同时，我们已知测试集中的真实标签（Ground Truth），即该用户实际观看并喜欢的电影集合。

- **推荐列表：** `[电影 C, 电影 A, 电影 F, 电影 B, 电影 H, 电影 D]`
- **相关项目（真实标签）：** `{电影 A, 电影 B, 电影 D}`

现在，我们遍历推荐列表，且*仅在发现相关电影的位置*计算精确度：

1. **位置 1 (电影 C)：** 不相关。不进行操作。
2. **位置 2 (电影 A)：** **相关**。此时，我们已看到 2 个项目，其中 1 个是相关的。
   - Precision@2 = 1/2 = 0.5
3. **位置 3 (电影 F)：** 不相关。不进行操作。
4. **位置 4 (电影 B)：** **相关**。此时，我们已看到 4 个项目，其中 2 个是相关的。
   - Precision@4 = 2/4 = 0.5
5. **位置 5 (电影 H)：** 不相关。不进行操作。
6. **位置 6 (电影 D)：** **相关**。此时，我们已看到 6 个项目，其中 3 个是相关的。
   - Precision@6 = 3/6 = 0.5

为了得到该用户的平均精度，我们对这些精确度得分取平均值。由于总共有 3 个相关项目，我们将计算出的精确度总和除以 3。


$$
ext{AP} = \frac{0.5 + 0.5 + 0.5}{3} = \frac{1.5}{3} = 0.5
$$


现在，考虑一个表现更好的模型，它将同样的相关项目排在了更高的位置。

- **推荐列表 (模型 2)：** `[电影 A, 电影 B, 电影 C, 电影 F, 电影 D, 电影 H]`
- **相关项目（真实标签）：** `{电影 A, 电影 B, 电影 D}`

让我们计算这个新列表的 AP：

1. **位置 1 (电影 A)：** **相关**。
   - Precision@1 = 1/1 = 1.0
2. **位置 2 (电影 B)：** **相关**。
   - Precision@2 = 2/2 = 1.0
3. **位置 3 (电影 C)：** 不相关。
4. **位置 4 (电影 F)：** 不相关。
5. **位置 5 (电影 D)：** **相关**。
   - Precision@5 = 3/5 = 0.6

现在，计算平均值：


$$
ext{AP} = \frac{1.0 + 1.0 + 0.6}{3} = \frac{2.6}{3} \approx 0.867
$$


模型 2 的 AP 得分明显更高，准确地反映了它生成的推荐列表排序更加合理。

下图展示了两个模型的计算过程，说明了相关项目的排名越高，AP 得分就越好。

> 比较同一用户的两个不同推荐模型的平均精度。模型 2 获得了更高的分数，因为它将相关项目 (A, B) 排在了列表的前列。

平均精度的正式定义为：


$$
ext{AP} = \frac{1}{R} \sum_{k=1}^{N} P(k) \times 	ext{rel}(k)
$$


其中：

- $N$ 是推荐列表中的项目数量。
- $R$ 是用户相关的项目总数。
- $P(k)$ 是在切断点 $k$ 处的精确度。
- $ext{rel}(k)$ 是一个指示函数，如果排名为 $k$ 的项目是相关的，则为 1，否则为 0。

### 计算平均精度均值

平均精度给出了单个用户的得分。为了得到一个描述整个模型性能的单一指标，我们为测试集中的每个用户计算 AP，然后取所有这些得分的平均值。这个最终值就是**平均精度均值 (MAP)**。


$$
ext{MAP} = \frac{1}{|U|} \sum_{u \in U} 	ext{AP}_u
$$


其中：

- $U$ 是测试集中所有用户的集合。
- $|U|$ 是用户总数。
- $ext{AP}_u$ 是用户 $u$ 的平均精度。

MAP 得分范围从 0 到 1，数值越高表示模型越好。1.0 的得分意味着模型为每一个用户都完美地将所有相关项目排在了列表的最顶端。

### 在 Python 中实现平均精度

我们将上述逻辑转化为一个简单的 Python 函数。该函数接受一个推荐项目列表和一个相关项目集合，并计算 AP。

```python
import numpy as np

def average_precision(recommended_items, relevant_items):
    """
    计算单个推荐列表的平均精度 (AP)。

    参数:
        recommended_items (list): 推荐项目的排名 ID 列表。
        relevant_items (set): 相关项目的 ID 集合 (真实标签)。

    返回:
        float: 平均精度得分。
    """
    if not relevant_items:
        return 0.0

    # 存储每个相关位置的精确度值
    precision_scores = []
    num_hits = 0

    for i, item_id in enumerate(recommended_items):
        if item_id in relevant_items:
            num_hits += 1
            precision_at_k = num_hits / (i + 1)
            precision_scores.append(precision_at_k)

    if not precision_scores:
        return 0.0

    # AP 是相关位置精确度得分的平均值
    return np.mean(precision_scores)

# 模型 2 的示例
recommended = ['电影 A', '电影 B', '电影 C', '电影 F', '电影 D', '电影 H']
relevant = {'电影 A', '电影 B', '电影 D'}

ap_score = average_precision(recommended, relevant)
print(f"模型 2 的 AP 得分: {ap_score:.4f}")
# 预期输出: 模型 2 的 AP 得分: 0.8667
```

要获取系统的 MAP 得分，您需要为测试集中的每个用户运行此函数，然后计算所有返回的 AP 得分的均值。对于任何关注项目排序的推荐任务，MAP 都是一种标准且有效的指标。

## 参考资料

- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/) — Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  这本基础教材全面阐述了信息检索指标，包括平均精度均值，并提供理论解释和示例。
- [Mining of Massive Datasets](http://www.mmds.org/) — Jure Leskovec, Anand Rajaraman, Jeff Ullman (2020)
  Publisher: Cambridge University Press
  这本书涵盖了大规模数据分析，包括推荐系统算法及其评估指标，如MAP。

---

[上一节](05-%E6%8E%92%E5%90%8D%E6%8C%87%E6%A0%87%EF%BC%9APrecision%40K%20%E4%B8%8E%20Recall%40K.md) · [下一节](07-%E5%BD%92%E4%B8%80%E5%8C%96%E6%8A%98%E6%8D%9F%E7%B4%AF%E8%AE%A1%E5%A2%9E%E7%9B%8A%20%28NDCG%29.md)

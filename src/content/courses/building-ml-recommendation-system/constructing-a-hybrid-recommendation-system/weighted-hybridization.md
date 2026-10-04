---
course: "building-ml-recommendation-system"
chapter: "constructing-a-hybrid-recommendation-system"
lesson: "weighted-hybridization"
sourceId: 7487
sourceUrl: "https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-6-constructing-a-hybrid-recommendation-system/weighted-hybridization"
title: "加权混合推荐"
description: "学习如何构建加权混合系统，使用线性公式组合来自不同推荐器的得分。"
order: 3
plots: ["plots/7487-0.json"]
sourceHash: "29c17c4b304306e2b93974f6358afd72ec18a8882921ff855dc1c3aca9ce0e41"
sourceCorrections: []
---

将不同推荐器组合在一起最直接且有效的方法之一就是加权混合。其主要原理是独立计算多个模型的预测得分，然后使用线性公式将它们组合成一个统一的得分。这种方法允许你平衡每个底层推荐器的影响，从而利用各自的长处。

### 混合得分计算

假设我们构建了两个独立的推荐器：一个基于内容的过滤模型和一个协同过滤模型。对于任何给定的“用户-物品”对，基于内容的模型会产生一个得分，我们称之为 $得分_{内容}$，而协同模型会产生另一个得分，$得分_{协同}$。加权混合通过简单的加权平均来组合这些得分。

最终混合得分的公式为：


$$
得分_{混合} = \alpha \cdot 得分_{内容} + (1-\alpha) \cdot 得分_{协同}
$$


在这个等式中：

- $得分_{混合}$ 是用于为用户进行物品排序的最终预测得分。
- $\alpha$ (alpha) 是由你控制的权重 (weight)参数 (parameter)，其取值在 0 到 1 之间。
- 当 $\alpha = 1$ 时，系统是一个纯粹的基于内容的推荐器。
- 当 $\alpha = 0$ 时，系统是一个纯粹的协同过滤模型。
- 当 $\alpha = 0.5$ 时，两个模型对最终得分的贡献相同。

通过调整 $\alpha$，你可以微调 (fine-tuning)系统以偏向其中一个模型，具体取决于哪种模型在你的特定数据集和目标下表现更好。

### 得分归一化 (normalization)的必要性

这项技术的一个前提是得分归一化。不同模型产生的得分往往处于完全不同的量级。例如，使用余弦相似度的基于内容的模型输出的得分在 -1 到 1 之间（或根据向量 (vector)空间在 0 到 1 之间），而像 SVD 这样的矩阵分解模型预测的评分可能在 1 到 5 之间。

直接组合这些未归一化的得分会使得分范围较大的模型占据不合理的权重 (weight)，从而导致 $\alpha$ 参数 (parameter)失效。为了解决这个问题，必须先将所有模型的得分缩放到一个公共范围，例如 0 到 1。常用的技术是最大最小缩放（Min-Max scaling）：


$$
得分_{归一化} = \frac{得分_{原始} - 最小值(得分集合)}{最大值(得分集合) - 最小值(得分集合)}
$$


在对基于内容和协同过滤模型的得分进行归一化后，你就可以应用加权混合公式来有效地组合它们。

> 该图展示了加权混合系统的流程。来自独立内容模型和协同模型的得分首先被归一化到统一量级，然后结合权重参数 $\alpha$ 生成最终的推荐集。

### 寻找最优权重 (weight)

权重参数 (parameter) $\alpha$ 是一个超参数 (hyperparameter)，其最优值取决于你的数据和组合的具体模型。不能盲目假设等权重（$\alpha = 0.5$）就是最好的，理想的方法是通过实验来调整这个参数。

为此，你可以从训练数据中留出一部分作为验证集。然后，遍历 $\alpha$ 的不同取值（例如，从 0.0 到 1.0，步长为 0.1），用每个值构建混合推荐器，并使用 NDCG 或 Precision@k 等指标在验证集上衡量其表现。表现最好的 $\alpha$ 值就是你应该为最终模型选择的值。



![调优混合权重 (α)](plots/7487-0.json)



> 随着权重参数 $\alpha$ 的变化，验证集上的表现也随之变化。在这个例子中，当 $\alpha$ 约为 0.6 时，达到了最优表现（最高 NDCG@10），这表明该场景下更偏向基于内容的模型。

### 实践中的实现

假设你已经生成并归一化 (normalization)了两个模型的得分，并将其存储在两个 pandas DataFrame 中：`content_scores` 和 `collab_scores`。每个 DataFrame 都有 `user_id`、`item_id` 和归一化后的 `score` 列。

加权组合的一个简单实现如下：

```python
import pandas as pd

# 假设 content_scores 和 collab_scores 已经预先计算并归一化
# DataFrame 示例：
# content_scores = pd.DataFrame({'user_id': [...], 'item_id': [...], 'score': [...]})
# collab_scores = pd.DataFrame({'user_id': [...], 'item_id': [...], 'score': [...]})

# 设置权重
alpha = 0.6

# 合并两个模型的得分
hybrid_scores = pd.merge(content_scores, collab_scores, on=['user_id', 'item_id'], suffixes=('_content', '_collab'))

# 计算加权混合得分
hybrid_scores['hybrid_score'] = alpha * hybrid_scores['score_content'] + (1 - alpha) * hybrid_scores['score_collab']

# 排序以获取针对特定用户的热门推荐
user_id_to_recommend = 101
recommendations = hybrid_scores[hybrid_scores['user_id'] == user_id_to_recommend].sort_values(by='hybrid_score', ascending=False)

print(recommendations.head(10))
```

这段代码展示了合并和组合得分的简便性。`pd.merge` 函数对齐 (alignment)了每个“用户-物品”对的得分，使加权求和计算变得非常直观。

加权混合是构建混合系统的有力起点。它实现简单、计算高效，并且通常比单一模型能带来显著的推荐质量提升。然而，它对所有预测都使用单一的静态权重 (weight)，这可能并不适用于所有情况。在下一节中，我们将研究可以根据上下文 (context)调整混合策略的动态技术。

## 参考资料

- [Recommender Systems Handbook](https://doi.org/10.1007/978-1-0716-2197-4) — Francesco Ricci, Lior Rokach, Bracha Shapira (2022)
  Publisher: Springer; DOI: [10.1007/978-1-0716-2197-4](https://doi.org/10.1007/978-1-0716-2197-4)
  这本全面指南涵盖了各种推荐技术，包括对不同混合推荐系统类型及其实现的详细讨论。
- [Toward the next generation of recommender systems: A survey of the state-of-the-art and possible extensions](https://doi.org/10.1109/TKDE.2005.99) — Gediminas Adomavicius, Alexander Tuzhilin (2005)
  Journal: IEEE Transactions on Knowledge and Data Engineering; Publisher: IEEE; Volume: 17; Pages: 734-749; DOI: [10.1109/TKDE.2005.99](https://doi.org/10.1109/TKDE.2005.99)
  这篇有影响力的综述论文提供了推荐系统基础概述，专门用一个章节介绍各种混合策略，包括加权方法。
- [Hybrid recommender systems: Survey and experiments](http://dx.doi.org/10.1023/A:1021240730564) — Robin Burke (2002)
  Journal: User Modeling and User-Adapted Interaction; Publisher: Springer Netherlands; Volume: 12; Pages: 331-370; DOI: [10.1023/A:1021240730564](https://doi.org/10.1023/A%3A1021240730564)
  这篇论文专门讨论混合推荐系统，对包括加权混合在内的不同方法进行了分类，并探讨了它们的优点和挑战。

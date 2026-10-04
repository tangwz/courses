---
course: "building-ml-recommendation-system"
chapter: "evaluating-recommendation-systems"
lesson: "practice-measuring-model-performance"
sourceId: 7484
sourceUrl: "https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-5-evaluating-recommendation-systems/practice-measuring-model-performance"
title: "动手实践：衡量模型性能"
description: "一个动手实践环节，你将在此计算并解读前几章所建模型的各种评估指标。"
order: 8
plots: ["plots/7484-0.json"]
sourceHash: "413779091768b47a116118f06aedb8e53f27846f351927174209adf5667ad22f"
sourceCorrections: []
---

我们将实现主要的离线评估指标。首先加载数据集，并训练两种不同的模型：基于邻域的 `KNNBasic` 和基于矩阵分解的 `SVD`。随后，系统地衡量它们的表现。这种直接对比能够说明不同模型在不同任务中的长处，以及为什么单一指标往往无法反映全貌。

我们的目标是创建一个可复用的评估工作流，供你在自己的项目中使用。我们将从计算简单的预测准确率开始，逐步过渡到从零实现更复杂但参考价值更高的排序指标。

### 设置环境与数据

首先，我们需要导入工具并准备数据集。我们将使用 `surprise` 库，它简化了数据加载、评估集划分以及标准推荐算法的训练。我们将使用广泛应用的 MovieLens 100k 数据集。

```python
import pandas as pd
from collections import defaultdict

from surprise import Dataset, Reader
from surprise import SVD, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# 加载 movielens-100k 数据集
data = Dataset.load_builtin('ml-100k')

# 将数据划分为训练集和测试集。
# test_size=0.25 表示我们将使用 25% 的数据进行测试。
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
```

完成数据划分后，我们得到了用于模型训练的 `trainset` 和包含模型未见过的用户-物品交互的 `testset`。我们将使用这个 `testset` 作为基准真相来评估模型的预测结果。

### 衡量预测准确率：RMSE 和 MAE

我们从最直接的评估任务开始：衡量模型预测显式评分的准确程度。在需要显示准确星级的系统中，这非常有用。让我们训练 SVD 和基于用户的 k-NN 模型，看看它们的表现如何。

```python
# --- 训练并评估 SVD 模型 ---
svd_model = SVD(random_state=42)
svd_model.fit(trainset)
svd_predictions = svd_model.test(testset)

# 计算 SVD 的 RMSE 和 MAE
print("SVD 模型性能:")
accuracy.rmse(svd_predictions)
accuracy.mae(svd_predictions)

# --- 训练并评估 k-NN 模型 ---
# 使用基于用户的协同过滤
knn_model = KNNBasic(sim_options={'user_based': True})
knn_model.fit(trainset)
knn_predictions = knn_model.test(testset)

# 计算 k-NN 的 RMSE 和 MAE
print("\nk-NN 模型性能:")
accuracy.rmse(knn_predictions)
accuracy.mae(knn_predictions)
```

运行这段代码会产生类似如下的输出：

```
SVD 模型性能:
RMSE: 0.9348
MAE:  0.7371

k-NN 模型性能:
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.9791
MAE:  0.7725
```

从这些结果可以看出，SVD 模型的均方根误差 (RMSE) 和平均绝对误差 (MAE) 均低于 k-NN 模型。这表明 SVD 在预测用户对电影的具体评分方面表现更好。RMSE 为 0.93 意味着其预测在 1 到 5 分的评分标准中，平均偏差不到一星。

### 评估排序质量

虽然预测准确率很有用，但大多数现代推荐系统是根据生成优质**排序列表**的能力来评价的。为此，我们需要排序指标。我们将实现计算 Precision@k 和 Recall@k 的函数，这两个指标衡量前 $k$ 个推荐结果的相关性。

我们的流程如下：

1. 针对每个用户，从模型中获取前 $k$ 个推荐物品。
2. 识别用户在测试集中**真正**喜欢的物品集（即我们的“基准真相”）。我们将“喜欢”定义为评分达到 4.0 或更高。
3. 将推荐集与基准真相集进行对比，查看有多少推荐是相关的。

下面的函数可以自动为 `surprise` 库生成的任何预测结果执行此操作。

```python
def calculate_precision_recall_at_k(predictions, k=10, rating_threshold=4.0):
    """
    返回每个用户的 precision@k 和 recall@k。
    """
    # 首先，将预测结果映射到每个用户。
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # 按估算值降序排列用户评分
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # 前 k 个推荐中相关物品的数量
        n_rel = sum((true_r >= rating_threshold) for (_, true_r) in user_ratings)

        # 前 k 个推荐中评分高于阈值的物品数量
        n_rec_k = sum((est >= rating_threshold) for (est, _) in user_ratings[:k])

        # 前 k 个推荐中既相关又被推荐的物品数量
        n_rel_and_rec_k = sum(
            ((true_r >= rating_threshold) and (est >= rating_threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Precision@k: 推荐物品中相关物品的比例
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@k: 相关物品中被推荐出来的比例
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    # 计算所有用户的平均值
    avg_precision = sum(p for p in precisions.values()) / len(precisions)
    avg_recall = sum(r for r in recalls.values()) / len(recalls)

    return avg_precision, avg_recall
```

现在，我们将此函数应用于 SVD 和 k-NN 模型生成的预测。我们将 $k$ 设置为 10，以评估前 10 个推荐结果。

```python
# 计算 SVD 的 Precision 和 Recall
svd_precision, svd_recall = calculate_precision_recall_at_k(svd_predictions, k=10)
print(f"SVD Precision@10: {svd_precision:.4f}")
print(f"SVD Recall@10: {svd_recall:.4f}")

# 计算 k-NN 的 Precision 和 Recall
knn_precision, knn_recall = calculate_precision_recall_at_k(knn_predictions, k=10)
print(f"\nk-NN Precision@10: {knn_precision:.4f}")
print(f"k-NN Recall@10: {knn_recall:.4f}")
```

这可能会产生以下输出：

```
SVD Precision@10: 0.8384
SVD Recall@10: 0.5891

k-NN Precision@10: 0.8421
k-NN Recall@10: 0.5301
```

这些结果呈现了不同的情况。k-NN 模型的准确率（Precision）略高，这意味着当它推荐前 10 个物品时，用户真正喜欢的概率略大。然而，SVD 模型的召回率（Recall）明显更高，说明它在找回用户喜欢的**全部**物品集方面表现更好。

### 使用 NDCG 衡量排序顺序

准确率和召回率对前 $k$ 列表中的所有物品一视同仁。归一化 (normalization)折现累计增益 (NDCG) 对此进行了改进，它会给予出现在推荐列表靠前位置的相关物品更高的权重 (weight)。

让我们编写一个计算 NDCG 的函数。其逻辑包括计算模型推荐列表的折现累计增益 (DCG)，并用理想折现累计增益 (IDCG) 进行归一化，后者代表了可能的最佳排序。

```python
import numpy as np

def calculate_ndcg_at_k(predictions, k=10, rating_threshold=4.0):
    """
    返回平均 NDCG@k。
    """
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    ndcgs = []
    for uid, user_ratings in user_est_true.items():
        # 按估算评分排序
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # 获取前 k 个推荐物品的相关性得分
        relevance_scores = [(1 if true_r >= rating_threshold else 0) for (_, true_r) in user_ratings[:k]]

        # 计算模型排序的 DCG
        dcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores)])

        # 创建理想排序以计算 IDCG
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance)])

        if idcg == 0:
            continue # 跳过测试集中没有相关物品的用户

        ndcgs.append(dcg / idcg)

    return np.mean(ndcgs)
```

现在，让我们计算两个模型的 NDCG。

```python
# 计算 SVD 的 NDCG
svd_ndcg = calculate_ndcg_at_k(svd_predictions, k=10)
print(f"SVD NDCG@10: {svd_ndcg:.4f}")

# 计算 k-NN 的 NDCG
knn_ndcg = calculate_ndcg_at_k(knn_predictions, k=10)
print(f"k-NN NDCG@10: {knn_ndcg:.4f}")
```

输出将类似于：

```
SVD NDCG@10: 0.8953
k-NN NDCG@10: 0.8879
```

在这里，SVD 的 NDCG 得分略高。这表明 SVD 不仅能找到较多数量的相关物品（高召回率），而且与 k-NN 模型相比，它往往会将这些物品排在 Top-10 列表中更靠前的位置。

### 总结与最终对比

我们已经为这两个模型计算了一系列指标。让我们汇总一下进行最终对比。

| 指标 | SVD | k-NN | 优胜者 |
| --- | --- | --- | --- |
| **RMSE** | **0.9348** | 0.9791 | SVD |
| **MAE** | **0.7371** | 0.7725 | SVD |
| **Precision@10** | 0.8384 | **0.8421** | k-NN |
| **Recall@10** | **0.5891** | 0.5301 | SVD |
| **NDCG@10** | **0.8953** | 0.8879 | SVD |



![模型性能对比](plots/7484-0.json)



> SVD 和 k-NN 模型在主要评估指标上的对比。RMSE 越低越好，而 Precision 和 NDCG 越高越好。

这次实践练习说明了构建推荐系统的一个核心道理：“最佳”模型取决于你的目标。

- 如果你的主要目标是**准确预测用户评分**，SVD 凭借其更优的 RMSE 和 MAE 成为明显的赢家。
- 如果你的目标是提供一个**高度相关的 Top-10 列表**，选择则不那么绝对。k-NN 的准确率略高，但 SVD 的召回率更好，且在 NDCG 上略占优势。

这套评估框架为你提供了跳出简单准确率限制的工具，让你能够衡量对应用真正有意义的指标。通过结合多个指标，你可以更全面地了解模型性能，并在选择和调整推荐算法时做出明智的决定。

## 参考资料

- [Matrix Factorization Techniques for Recommender Systems](https://www.computer.org/csdl/magazine/co/2009/08/mco2009080030/13rRUw976j7) — Yehuda Koren, Robert Bell, Chris Volinsky (2009)
  Journal: Computer; Publisher: IEEE Computer Society; Volume: 42; Pages: 30-37; DOI: [10.1109/MC.2009.263](https://doi.org/10.1109/MC.2009.263)
  这篇有影响力的论文详细介绍了用于准确评级预测的基于SVD的矩阵分解方法，这是本次实践中的核心模型。
- [Surprise: A Python scikit for building and analyzing recommender systems (Official Documentation)](https://surprise.readthedocs.io/en/stable/) — Nicolas Hug and other contributors (2023)
  `surprise`库的官方指南，直接用于实现和评估推荐模型。
- [Metrics for Evaluating Recommender Systems](https://link.springer.com/chapter/10.1007/978-1-4939-0827-9_3) — Joeran Beel, Benno Stein, Marcel G. Klas, Andreas Nürnberger (2015)
  Journal: Recommender Systems Handbook (2nd ed.); Publisher: Springer; Pages: 109-141; DOI: [10.1007/978-1-4939-0827-9_3](https://doi.org/10.1007/978-1-4939-0827-9_3)
  提供了推荐系统各种评估指标的概述，包括预测准确性以及Precision@k、Recall@k和NDCG等排名质量指标。

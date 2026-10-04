---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "hyperparameter-tuning-and-optimization"
lesson: "grid-search-and-randomized-search"
sourceId: 7610
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-6-hyperparameter-tuning-and-optimization/grid-search-and-randomized-search"
title: "网格搜索与随机搜索的使用"
description: "了解怎样使用 Scikit-Learn 的 GridSearchCV 和 RandomizedSearchCV 来自动化确定最佳超参数组合的过程。"
order: 7
plots: []
sourceHash: "3c22052d77758b8ca20653ff26ab62f361ec4714fb52cef5adb660492a2110ee"
sourceCorrections: []
---

优化影响模型性能的主要超参数 (parameter) (hyperparameter)是实现最佳结果的必要环节。找到这些参数最有效的值通常需要系统化的方法，因为逐个调整参数不仅耗时，而且无法捕捉它们之间复杂的相互作用。自动化此搜索过程提供了一种更高效的解决方案。Scikit-Learn 提供了两种出色的工具来完成此任务：`GridSearchCV` 和 `RandomizedSearchCV`。这两种方法都系统地探索参数组合，使用交叉验证来防止过拟合 (overfitting)，并提供对模型性能的可靠评估。

### 全面查找的网格搜索

网格搜索，在 Scikit-Learn 中实现为 `GridSearchCV`，对指定的参数 (parameter)网格执行穷尽搜索。您为要调整的每个超参数 (hyperparameter)定义一组离散值，网格搜索会对这些值的每种可能组合训练并评估模型。它有条不紊且全面，保证能在给定的网格中找到最佳组合。

假设您正在调整两个超参数：`learning_rate`，可能的值为 `[0.01, 0.1, 0.2]`；以及 `n_estimators`，可能的值为 `[100, 200, 300]`。网格搜索将构建一个 3x3 的网格并测试所有九种组合。

> 网格中的每个点代表一种独特的超参数组合，在搜索过程中进行训练和评估。

让我们看看如何使用 Scikit-Learn 和 `XGBClassifier` 来实现这一点。首先，您定义模型和参数网格。

```python
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# 1. 定义模型
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False
)

# 2. 定义参数网格
param_grid = {
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'n_estimators': [100, 200]
}

# 3. 实例化 GridSearchCV
# cv=5 指定 5 折交叉验证
# n_jobs=-1 使用所有可用的 CPU 核心
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# 4. 将搜索拟合到您的数据 (X_train, y_train)
# grid_search.fit(X_train, y_train)

# 拟合后，您可以访问最佳参数和分数
# print(f"找到的最佳参数: {grid_search.best_params_}")
# print(f"最佳准确度分数: {grid_search.best_score_}")
```

在此示例中，网格搜索将训练和评估 $3 \times 3 \times 2 = 18$ 种组合。由于我们使用 5 折交叉验证，这会导致总共 $18 \times 5 = 90$ 次模型拟合。这表明了网格搜索的主要缺点：其计算成本随参数数量和值的增加呈指数增长，这个问题通常被称为“维度灾难”。对于大的搜索空间，网格搜索会慢得令人无法接受。

### 用于高效调优的随机搜索

随机搜索，即 `RandomizedSearchCV`，提供了一种更有效的选择。它不是尝试每种组合，而是从指定分布中抽取固定数量的参数 (parameter)组合 (`n_iter`)。此方法基于这样的观察：对于许多模型，只有少数超参数 (hyperparameter)对性能有较大作用。通过随机抽样，您很有可能找到表现出色的组合，而无需承担穷尽搜索的计算负担。

例如，您无需为 `learning_rate` 提供离散值列表，可以提供一个连续分布（例如，从 0.01 到 0.3），随机搜索将从中抽取值。

> 并非有序网格，随机搜索评估超参数空间内固定数量的随机选定点。

以下是设置 `RandomizedSearchCV` 的方法。此示例使用 `scipy.stats` 库中的分布。

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# 1. 定义模型（与之前相同）
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False
)

# 2. 定义参数分布
param_distributions = {
    'learning_rate': uniform(0.01, 0.2),  # 从均匀分布中抽取样本
    'max_depth': randint(3, 8),          # 抽取 3 到 7 之间的整数
    'n_estimators': randint(100, 400)    # 抽取 100 到 399 之间的整数
}

# 3. 实例化 RandomizedSearchCV
# n_iter 指定采样的参数设置数量
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=20,  # 我们将尝试 20 种不同的组合
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1,
    random_state=42 # 用于结果复现
)

# 4. 将搜索拟合到您的数据
# random_search.fit(X_train, y_train)

# 拟合后，访问最佳结果
# print(f"找到的最佳参数: {random_search.best_params_}")
# print(f"最佳准确度分数: {random_search.best_score_}")
```

在这里，我们设置 `n_iter=20`。这意味着搜索将总共训练 $20 \times 5 = 100$ 个模型，不论分布中有多少参数或值。这使您能直接控制计算预算。尽管不保证能找到绝对最佳组合，但随机搜索通常在寻找非常好的参数集方面出奇地有效，而且远快于网格搜索。

### 选择合适的策略

网格搜索和随机搜索之间的选择，取决于您的计算资源和参数 (parameter)空间的大小。

- **网格搜索** 最适合小的搜索空间，在其中您对有前景的参数值有清晰的了解。其穷尽的特性确保您不会错过定义网格中的最佳点。
- **随机搜索** 在检查大且多样的超参数 (hyperparameter)空间时更出色。它更可能找到对性能作用最大的参数的良好组合，并且允许您通过 `n_iter` 参数直接控制运行时间。

一种常用且有效的工作流程是按顺序使用这两种方法：

1. **广泛查看：** 从 `RandomizedSearchCV` 开始，针对大范围的参数分布，以确定超参数空间中有潜力的区域。
2. **精细调整：** 使用随机搜索的结果来定义一个更小、更集中的网格。然后，在这个更小的网格上运行 `GridSearchCV`，以找到该区域内的最优设置。

尽管 `GridSearchCV` 和 `RandomizedSearchCV` 是有用的工具，但还有其他高级的超参数优化方法可用，例如贝叶斯优化（在诸如 `Hyperopt` 和 `Optuna` 等包中可以找到）。这些方法利用过去的评估结果，更明智地决定接下来要尝试哪些参数组合，通常能更快地得到好的解决方案。然而，对于大多数应用而言，结合随机搜索和网格搜索的结构化方法在性能和简洁性之间提供了很好的平衡。

## 参考资料

- [Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html) — scikit-learn developers (2024)
  Publisher: scikit-learn
  Scikit-Learn `GridSearchCV` 和 `RandomizedSearchCV` 的官方文档，详细说明了其在超参数优化中的实现和用法。
- [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/v13/bergstra12a.html) — James Bergstra and Yoshua Bengio (2012)
  Journal: Journal of Machine Learning Research; Volume: 13(10); Pages: 281-305; DOI: [10.5555/2188385.2188417](https://doi.org/10.5555/2188385.2188417)
  介绍了随机搜索作为网格搜索在超参数优化方面的一种高效替代方法，并为其有效性提供了理论和经验依据。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本涵盖统计学习方法的奠基性教科书，包括对超参数选择、交叉验证和模型评估技术的全面讨论。
- [XGBoost: A Scalable Tree Boosting System](https://dl.acm.org/doi/10.1145/2939672.2939785) — Tianqi Chen and Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 785-794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  介绍 XGBoost 的原始论文，阐述了其并行处理能力、树增强算法和正则化技术。

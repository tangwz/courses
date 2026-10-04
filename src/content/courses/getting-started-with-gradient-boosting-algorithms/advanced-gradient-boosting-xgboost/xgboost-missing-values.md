---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "advanced-gradient-boosting-xgboost"
lesson: "xgboost-missing-values"
sourceId: 7594
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-4-advanced-gradient-boosting-xgboost/xgboost-missing-values"
title: "自动处理缺失值"
description: "了解 XGBoost 在构建树过程中处理缺失值的内置机制，简化数据预处理。"
order: 4
plots: []
sourceHash: "7617f19e83dfcd05d44991f29d52be2f95fdf0de89ce6931b650a1cde99f4921"
sourceCorrections: []
---

为机器学习 (machine learning)模型准备数据时，处理缺失值是一个很大的难题。常用方法包括填充，即用列的平均值、中位数或众数等统计量来替换缺失条目。填充虽然通常必不可少，但它是一种估算。它引入了人造数据，有时会扭曲特征的固有分布，并可能降低模型性能。

XGBoost 通过一种内置的智能机制来解决数据预处理中处理缺失值的挑战，该机制用于处理缺失数据，称为**稀疏感知分裂查找**算法。XGBoost 无需您在训练前进行值填充，而是学习如何为每个特征分裂处理缺失值。

### 稀疏感知分裂查找算法

在构建树时，XGBoost 会遇到特征上的潜在分裂点。如果某些实例的该特征值缺失，算法不会简单地忽略它们。相反，它会评估两种情况：

1. 它暂时将所有缺失值的实例放入**左**子节点，并计算该分裂的信息增益（或损失减少）。
2. 然后它暂时将所有缺失值的实例放入**右**子节点，并再次计算增益。

算法随后选择提供更高增益的方向。此选定路径成为在该特定分裂处具有缺失值的任何实例的**默认方向**。

这个过程不是全局设置。它会在每棵树的每个分裂点重复。对于单个特征，缺失值在一个分裂点可能被发送到左分支，而在树更深处的另一个分裂点则可能被发送到右分支。这使得模型能够根据每个节点的局部情况，为缺失数据学习出最优路径。

> XGBoost 在每个分裂点为缺失值学习一个默认方向。在此示例中，算法确定将特征 X 的缺失值路由到左分支，从而使该特定分裂的信息增益最大化。

### 内置处理的优势

这种方法有两个主要优点：

1. **简化数据预处理**：您通常可以直接将含有缺失值（表示为 `None`、`np.nan` 或空字符串）的数据集传递给模型，而无需手动填充。这节省了时间，并从您的工作流程中移除了一个可能出错的步骤。
2. **从缺失情况中学习**：模型能够在存在缺失数据的情况下识别出预测模式。例如，如果客户没有提供其年收入，这个事实本身可能就是一个与贷款违约风险相关的信号。XGBoost 的算法可以通过学习这些实例的最优路径来捕捉这种关系，有效地将“缺失情况”视为另一条信息。

### 实际应用与考量

使用此功能非常简单。XGBoost 的 Python API 默认处理 NumPy 的 `nan` 值。

```python
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split

# 包含缺失值的示例数据
X = np.array([
    [1, 10], 
    [2, 20], 
    [np.nan, 30], # 第一个特征中的缺失值
    [4, np.nan],  # 第二个特征中的缺失值
    [5, 50]
])
y = np.array([100, 200, 300, 400, 500])

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 直接初始化并训练 XGBoost 模型
# 无需填充步骤
xgbr = xgb.XGBRegressor(objective='reg:squarederror')
xgbr.fit(X_train, y_train)

# 进行预测
predictions = xgbr.predict(X_test)
print(f"测试数据的预测结果: {predictions}")
```

尽管此功能强大，但它不能替代您对数据的理解。您仍应仔细查看值缺失的*原因*。如果数据完全随机缺失，所学的默认方向可能不具特殊意义。然而，如果缺失情况具有系统性（非随机缺失），XGBoost 的方法会非常高效。这种内置处理是众多优化措施之一，使得 XGBoost 成为一个快速、准确且易用的梯度提升算法库。

## 参考资料

- [XGBoost: A Scalable Tree Boosting System](https://dl.acm.org/doi/10.1145/2939672.2939785) — Tianqi Chen, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 785-794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  介绍XGBoost及其处理缺失值的稀疏感知分裂查找算法的原始研究论文。
- [Missing Value Support](https://xgboost.readthedocs.io/en/stable/faq.html#q-how-do-i-handle-missing-values) — XGBoost Contributors (2024)
  XGBoost官方文档，解释其内置的缺失值处理机制和实际应用。
- [Applied Predictive Modeling](https://link.springer.com/book/10.1007/978-1-4614-6849-3) — Max Kuhn and Kjell Johnson (2013)
  Publisher: Springer; DOI: [10.1007/978-1-4614-6849-3](https://doi.org/10.1007/978-1-4614-6849-3)
  一本备受推崇的教科书，提供了数据预处理技术的背景，包括插补策略及其影响。

---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "gradient-boosting-with-scikit-learn"
lesson: "sklearn-gradientboostingregressor"
sourceId: 7585
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-3-gradient-boosting-with-scikit-learn/sklearn-gradientboostingregressor"
title: "Scikit-Learn 的 GradientBoostingRegressor"
description: "关于如何在 Scikit-Learn 中使用 GradientBoostingRegressor 类解决回归问题的指南。学习如何拟合模型和进行预测。"
order: 2
plots: ["plots/7585-0.json"]
sourceHash: "08315404dfe116a23415c35516a9720764484876d134a347d7ddffc36d301141"
sourceCorrections: []
---

对于目标是预测连续值的回归任务，Scikit-Learn 提供了 `GradientBoostingRegressor` 类。该类实现了梯度提升机算法。它通过顺序拟合决策树来构建加性模型，其中每棵新树都经过训练以纠正所有先前树的组合所产生的误差。

`GradientBoostingRegressor` 是一个强大且灵活的工具，适用于从预测房价到预测需求等多种回归问题。其有效性在于它能模拟数据中复杂的非线性关系。

### GradientBoostingRegressor 类

首先，您需要从 `sklearn.ensemble` 导入该类。它的实例化过程很直接，如果您使用过其他 Scikit-Learn 模型，会感到很熟悉。

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# 创建一些合成数据
X = np.random.rand(100, 1) * 10
y = np.sin(X).ravel() + np.random.normal(0, 0.3, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用默认参数实例化模型
gbr = GradientBoostingRegressor(random_state=42)

# 用训练数据拟合模型
gbr.fit(X_train, y_train)

# 对测试集进行预测
y_pred = gbr.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print(f"均方误差: {mse:.4f}")
```

此代码片段演示了标准工作流程：实例化、拟合和预测。尽管默认参数 (parameter)通常能提供一个合理的起点，但了解主要参数对于构建高性能模型非常重要。

### 配置回归器

`GradientBoostingRegressor` 的行为由几个重要参数 (parameter)控制。让我们查看您最常调整的参数。

#### 损失函数 (loss function) (`loss`)

`loss` 参数定义了要优化的损失函数。损失函数的选择取决于您的回归问题的具体情况，特别是它对异常值的敏感性。

- `'ls'`: 默认选项，代表**最小二乘回归**。它最小化 L2 损失，相当于均方误差（$MSE$）。这是一个很好的通用选择，但可能对异常值敏感。
- `'lad'`: **最小绝对偏差**，它最小化 L1 损失，相当于平均绝对误差（$MAE$）。与最小二乘法相比，它对异常值更具鲁棒性。
- `'huber'`: 最小二乘法和最小绝对偏差的组合。它对小误差表现为最小二乘法，对大误差表现为最小绝对偏差，从而平衡了敏感性和鲁棒性。
- `'quantile'`: 允许进行**分位数回归**。该损失函数不预测均值，而是可以用于预测特定分位数（例如，第50百分位数，即中位数）。

#### 模型复杂度和学习

`n_estimators`、`learning_rate` 和 `max_depth` 之间的作用控制着模型拟合训练数据而不过拟合 (overfitting)的能力。

- `n_estimators`: 此参数设置了提升阶段的数量，它对应于集成模型中树的数量。更多的树可以捕捉更复杂的模式，但过多的树可能导致过拟合。
- `learning_rate`: 此参数通常称为*收缩率*，它调整每棵树的贡献。较小的学习率（例如 0.01）需要更大的 `n_estimators` 才能达到相同的训练误差，但通常会产生更好的泛化能力。它有效地减缓了学习过程，防止模型在每棵新树上做出剧烈修正。
- `max_depth`: 这控制了单个决策树的最大深度。浅层树（例如 `max_depth=3`）受到限制，并作为弱学习器，这是提升过程的中心。更深的树可以模拟更复杂的特征交互，但会增加过拟合训练数据的风险。

### 实际示例

让我们构建一个模型来拟合一个更复杂的非线性函数，并可视化其预测。我们将使用一个稍作配置的模型，以观察更改参数 (parameter)的效果。

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# 生成带有噪声的非线性数据集
np.random.seed(0)
X = np.linspace(0, 6, 150)[:, np.newaxis]
y = X * np.sin(X).ravel() + np.random.normal(0, 0.5, 150)

# 实例化并配置模型
gbr_tuned = GradientBoostingRegressor(
    n_estimators=200,      # 更多树
    learning_rate=0.05,    # 更小的学习率
    max_depth=4,           # 稍微更深的树
    loss='ls',             # 标准最小二乘损失
    random_state=42
)

# 拟合模型
gbr_tuned.fit(X, y)

# 创建用于预测可视化的平滑线
X_plot = np.linspace(0, 6, 500)[:, np.newaxis]
y_plot = gbr_tuned.predict(X_plot)
```

通过设置较小的 `learning_rate` 和较大的 `n_estimators`，我们促使模型更渐进地学习潜在模式。`max_depth` 为 4 允许每棵树捕捉适中水平的交互。下面的可视化展示了简单树的集成如何有效地近似了复杂的正弦波函数。



![梯度提升回归器拟合](plots/7585-0.json)



> 模型的预测（红线）紧密遵循带有噪声的训练数据（蓝点）的潜在模式，展示了它学习复杂非线性关系的能力。

构建回归器后，接下来的步骤涉及了解它为何做出这些预测以及如何处理分类问题。在接下来的章节中，我们将了解解释这些模型的方法，并介绍其分类对应的模型 `GradientBoostingClassifier`。

## 参考资料

- [Greedy Function Approximation: A Gradient Boosting Machine](https://doi.org/10.1214/aos/1013203460) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Publisher: Institute of Mathematical Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203460](https://doi.org/10.1214/aos/1013203460)
  介绍了原始梯度提升机算法。
- [sklearn.ensemble.GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) — scikit-learn developers (2023)
  Scikit-Learn `GradientBoostingRegressor` 类的官方文档，包含参数说明。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; Pages: Chapter 10
  涵盖统计学习方法的标准教科书，其中有一章是关于提升方法的。

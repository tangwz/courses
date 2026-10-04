---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "hyperparameter-tuning-and-optimization"
lesson: "structured-approach-to-tuning"
sourceId: 7609
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-6-hyperparameter-tuning-and-optimization/structured-approach-to-tuning"
title: "调优的结构化方法"
description: "遵循一种系统方法来调优超参数，从影响最大的参数开始，并迭代地优化搜索范围。"
order: 6
plots: ["plots/7609-0.json"]
sourceHash: "7ad34a7e7e8e96fa6c8ad01908c7c1d683b0572f9d52c2e1f3701c625bc5544e"
sourceCorrections: []
---

调优梯度提升模型，就像在一个有数十个旋钮和开关的迷宫中穿行。在一次大规模网格搜索中同时调整所有超参数 (parameter) (hyperparameter)，计算成本高昂，且通常效率低下。一种更有效的方法是遵循一个迭代的、有优先级的流程，在该流程中，您按逻辑顺序调整相关参数组。这种方法使您能够系统地优化模型，在每一步的改进基础上进行。

通用策略是首先为固定且相对较高的学习率找到最优的提升轮数。确定了估计器数量后，您就可以调整控制每棵树的结构和复杂度的参数。接着，您可以调整正则化 (regularization)参数以提高泛化能力。最后，您可以降低学习率并重新校准估计器数量，以获得最终的性能提升。

### 迭代调优流程

采用一种结构化方法可以提供可重复的步骤来优化您的模型。尽管具体参数 (parameter)在XGBoost和LightGBM等库之间可能略有不同，但基本原理保持不变。

> 迭代超参数 (hyperparameter)调优过程的图示。

### 步骤1：为固定学习率找到最优估计器数量

`n_estimators`（树的数量）和`learning_rate`参数 (parameter)高度相互依赖。较低的学习率需要更多的树才能收敛。首先，我们将学习率固定在一个合理较高的值，例如0.1，这可以加快迭代速度。然后，我们找到此学习率下最优的树的数量。

大多数梯度提升库都有内置的交叉验证函数可以提供帮助。例如，XGBoost的`xgb.cv`函数会在每个提升轮次评估模型，让您能够识别验证集性能停止提升的点，这种技术称为早期停止。

```python
# 使用XGBoost的cv函数示例
import xgboost as xgb
import pandas as pd

# 假设dtrain是您的DMatrix训练数据
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1,  # 这是学习率
    'max_depth': 5 # 从默认值开始
}

cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    nfold=5,
    early_stopping_rounds=50,
    seed=42,
    verbose_eval=False
)

print(f"Optimal number of estimators: {cv_results.shape[0]}")
# Optimal number of estimators: 150
```

`cv_results.shape[0]`为您提供了在过拟合 (overfitting)发生前理想的提升轮数。在剩余的调优过程中，您将使用此值作为您的`n_estimators`。



![通过早期停止寻找最优估计器](plots/7609-0.json)



> 交叉验证误差趋于平稳并开始上升，这表示在过拟合发生前，提升轮数达到最优。

### 步骤2：调优树特有参数 (parameter)

接下来，着重调整控制每棵树复杂度的参数。这些参数对偏差-方差权衡有很大影响。最常见的包括：

- `max_depth`：树的最大深度。
- `min_child_weight`：子节点中所需的最小实例权重 (weight)和。
- `gamma`：在叶节点上进行进一步划分所需的最小损失减少量。

您可以使用Scikit-Learn的`GridSearchCV`或`RandomizedSearchCV`来寻找这些参数的最佳组合。先从相对较宽的范围开始，必要时再缩小范围。

```python
# 使用GridSearchCV与XGBoost模型
from sklearn.model_selection import GridSearchCV

# 假设X_train, y_train是您的数据
xgb_model = xgb.XGBClassifier(
    learning_rate=0.1,
    n_estimators=150, # 来自步骤1
    objective='binary:logistic',
    eval_metric='logloss'
)

param_grid = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='roc_auc', verbose=1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
# Best parameters: {'max_depth': 5, 'min_child_weight': 3}
```

此步骤之后，用这些新找到的最优值更新您的模型参数。

### 步骤3：调优子采样参数 (parameter)

为了增加随机性并对抗过拟合 (overfitting)，您可以调优子采样参数。这些参数控制用于生长每棵树的数据比例。

- `subsample`：每棵树随机采样的训练实例比例。
- `colsample_bytree`：每棵树随机采样的列比例。

再次，使用网格搜索来找到最佳值，同时保持前几步的参数不变。通常的搜索值范围是0.6到1.0。

### 步骤4：微调 (fine-tuning)正则化 (regularization)参数 (parameter)

L1 (`reg_alpha`) 和 L2 (`reg_lambda`) 正则化参数可以作为最后一步进行调优，以控制模型的复杂度。虽然它们的影响可能不如树结构参数那么显著，但它们提供了一个额外的手段来减少过拟合 (overfitting)。通常在对数尺度上搜索这些参数，例如`[0, 0.01, 0.1, 1, 100]`。

### 步骤5：降低学习率并寻找新的`n_estimators`

在所有其他参数 (parameter)都已调优后，您现在可以降低学习率。较低的学习率通常会导致更好的泛化能力，但需要更多的提升轮数。将学习率设置为较小的值，例如0.05或0.01，并重新运行步骤1中的交叉验证，以寻找`n_estimators`新的、更大的最优值。这个最终调整通常会给模型性能带来虽小但有意义的提升。

通过遵循这种结构化、迭代的方法，您可以高效地在超参数 (hyperparameter)空间中进行调整，并构建一个专门针对您的特定数据集的高度优化的梯度提升模型。

## 参考资料

- [XGBoost: A Scalable Tree Boosting System](https://dl.acm.org/doi/10.1145/2939672.2939785) — Tianqi Chen, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 785-794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  详细介绍了XGBoost算法，这是本节讨论和示例中使用的广泛梯度提升实现。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFmvwzxgvn-zhGTiVbXU23s1WB6ugDksIjbE5dUhP__1mbE1bVuZ-HODoiT3Wn8ZSYWwWXTq0omRHYAzzILZq6IW8_YASQZigZYDKmHP-72sG9RHKfvdKm3D7DBRHz_WZR1HXqs-kjAk6A3pBvgA6rnrgRj_SwaHGs1ZVG77AwgsSrfO3Yn4amyXrpvYotJn0JRF_YCeVKQT7ggK2KN) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 864
  提供了机器学习的实践指导，包括超参数调优的结构化方法和交叉验证技术。
- [3.2. Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html) — Scikit-learn Developers (2024)
  Journal: Scikit-learn User Guide
  解释了Scikit-learn中如`GridSearchCV`和`RandomizedSearchCV`等工具在超参数优化中的使用，如本节所示。

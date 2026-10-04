---
course: "mastering-gradient-boosting-algorithms"
chapter: "regularization-gradient-boosting"
lesson: "practice-applying-regularization"
sourceId: 1975
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-3-regularization-gradient-boosting/practice-applying-regularization"
title: "动手实践：应用正则化"
description: "实现并比较不同正则化方法在一个示例问题上的应用。"
order: 7
plots: ["plots/1975-0.json"]
sourceHash: "4f510ff9b6d92d23943b301bbfa779fa0a8c5da09ce6827ac1b84531e2b970c1"
sourceCorrections: []
---

使用 Scikit-learn 的 `GradientBoostingClassifier` 实现和评估树的约束、收缩、子采样和提前停止的效果，为梯度提升的正则化 (regularization)技术提供了实践应用。目的是观察这些技术如何减少过拟合 (overfitting)并提高在新数据上的泛化能力。

### 准备工作

首先，我们需要必要的工具和一个容易发生过拟合 (overfitting)的数据集。我们将使用常用的 Python 库，并使用 Scikit-learn 生成一个合成的分类数据集。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.datasets import make_classification

# 生成一个合成数据集
X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=10, n_redundant=5,
                           n_clusters_per_class=2, flip_y=0.1,
                           random_state=42)

# 拆分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
```

这样的设置提供了独立的训练集和验证集，这对于评估过拟合和正则化 (regularization)的有效性非常重要。

### 基准模型：一个过拟合 (overfitting)模型

我们首先训练一个 GBM 模型，其参数 (parameter)设置可能导致过拟合。我们将使用相对较多的估计器数量（`n_estimators`），并且没有明确的正则化 (regularization)约束（这可能包含一些隐式正则化，比如默认的 `max_depth`）。我们最初将 `learning_rate` 设置为中等值。

```python
# 基准 GBM - 可能过拟合
gbm_baseline = GradientBoostingClassifier(n_estimators=300,
                                         learning_rate=0.1,
                                         max_depth=5, # 相对较深的树
                                         random_state=42)

gbm_baseline.fit(X_train, y_train)

# 评估性能
y_train_pred_baseline = gbm_baseline.predict(X_train)
y_val_pred_baseline = gbm_baseline.predict(X_val)
y_train_proba_baseline = gbm_baseline.predict_proba(X_train)[:, 1]
y_val_proba_baseline = gbm_baseline.predict_proba(X_val)[:, 1]

print("基准模型性能：")
print(f"  训练集准确率: {accuracy_score(y_train, y_train_pred_baseline):.4f}")
print(f"  验证集准确率: {accuracy_score(y_val, y_val_pred_baseline):.4f}")
print(f"  训练集对数损失: {log_loss(y_train, y_train_proba_baseline):.4f}")
print(f"  验证集对数损失: {log_loss(y_val, y_val_proba_baseline):.4f}")
```

你可能会看到训练集和验证集性能指标（准确率和对数损失）之间存在一个明显的差距。高训练准确率与低验证准确率相结合是过拟合的典型表现。模型对训练数据（包括其中的噪声）学习得过于充分，导致泛化能力不佳。

### 应用正则化 (regularization)方法

现在，让我们系统地应用前面讨论的正则化方法并观察它们的影响。

#### 1. 树的约束（`max_depth`, `min_samples_leaf`）

控制单个树的复杂度是防止它们拟合噪声的直接方法。让我们限制 `max_depth` 并设置每个叶节点所需的最小样本数（`min_samples_leaf`）。

```python
# 应用树约束的 GBM
gbm_tree_reg = GradientBoostingClassifier(n_estimators=300,
                                          learning_rate=0.1,
                                          max_depth=3,         # 较浅的树
                                          min_samples_leaf=10, # 每个叶节点需要更多样本
                                          random_state=42)

gbm_tree_reg.fit(X_train, y_train)

# 评估性能
y_train_pred_tree = gbm_tree_reg.predict(X_train)
y_val_pred_tree = gbm_tree_reg.predict(X_val)
y_train_proba_tree = gbm_tree_reg.predict_proba(X_train)[:, 1]
y_val_proba_tree = gbm_tree_reg.predict_proba(X_val)[:, 1]

print("\n应用树约束的 GBM 性能：")
print(f"  训练集准确率: {accuracy_score(y_train, y_train_pred_tree):.4f}")
print(f"  验证集准确率: {accuracy_score(y_val, y_val_pred_tree):.4f}")
print(f"  训练集对数损失: {log_loss(y_train, y_train_proba_tree):.4f}")
print(f"  验证集对数损失: {log_loss(y_val, y_val_proba_tree):.4f}")
```

将这些结果与基准模型进行比较。你应该会看到训练性能略有下降，但验证性能应有所提高（或者训练和验证性能之间的差距应缩小），这表明泛化能力更好。

#### 2. 收缩（`learning_rate`）

降低 `learning_rate` 迫使模型学习得更慢，需要更多的提升轮次（`n_estimators`）以达到相似的性能，但通常会得到一个泛化能力更好的模型。

```python
# 应用收缩的 GBM
# 降低学习率，可能需要更多估计器才能收敛
gbm_shrinkage = GradientBoostingClassifier(n_estimators=600, # 增加估计器数量
                                           learning_rate=0.05, # 较低的学习率
                                           max_depth=3,       # 保持树约束
                                           min_samples_leaf=10,
                                           random_state=42)

gbm_shrinkage.fit(X_train, y_train)

# 评估性能
y_train_pred_shrink = gbm_shrinkage.predict(X_train)
y_val_pred_shrink = gbm_shrinkage.predict(X_val)
y_train_proba_shrink = gbm_shrinkage.predict_proba(X_train)[:, 1]
y_val_proba_shrink = gbm_shrinkage.predict_proba(X_val)[:, 1]

print("\n应用收缩的 GBM 性能：")
print(f"  训练集准确率: {accuracy_score(y_train, y_train_pred_shrink):.4f}")
print(f"  验证集准确率: {accuracy_score(y_val, y_val_pred_shrink):.4f}")
print(f"  训练集对数损失: {log_loss(y_train, y_train_proba_shrink):.4f}")
print(f"  验证集对数损失: {log_loss(y_val, y_val_proba_shrink):.4f}")
```

再次比较性能。降低学习率通常会带来更平滑的收敛和更好的验证结果，前提是 `n_estimators` 相应地进行了调整。

#### 3. 子采样（`subsample`, `max_features`）

通过在行子集（`subsample`）上训练每棵树引入随机性，或仅考虑特征子集进行分裂（`max_features`），是随机梯度提升的特点。

```python
# 应用子采样的 GBM
gbm_subsample = GradientBoostingClassifier(n_estimators=600,
                                           learning_rate=0.05,
                                           max_depth=3,
                                           min_samples_leaf=10,
                                           subsample=0.7,       # 每棵树使用 70% 的行
                                           max_features=0.8,    # 每次分裂使用 80% 的特征
                                           random_state=42)

gbm_subsample.fit(X_train, y_train)

# 评估性能
y_train_pred_sub = gbm_subsample.predict(X_train)
y_val_pred_sub = gbm_subsample.predict(X_val)
y_train_proba_sub = gbm_subsample.predict_proba(X_train)[:, 1]
y_val_proba_sub = gbm_subsample.predict_proba(X_val)[:, 1]

print("\n应用子采样的 GBM 性能：")
print(f"  训练集准确率: {accuracy_score(y_train, y_train_pred_sub):.4f}")
print(f"  验证集准确率: {accuracy_score(y_val, y_val_pred_sub):.4f}")
print(f"  训练集对数损失: {log_loss(y_train, y_train_proba_sub):.4f}")
print(f"  验证集对数损失: {log_loss(y_val, y_val_proba_sub):.4f}")
```

子采样通常能增强模型稳定性，常常带来更好的验证分数，特别是在具有高方差或相关特征的数据集上。

#### 4. 提前停止

除了固定 `n_estimators`，我们可以在训练期间监控模型在验证集上的性能，并在性能不再提升时停止。Scikit-learn 的 GBM 通过 `n_iter_no_change`、`validation_fraction` 和 `tol` 参数 (parameter)实现此功能。

```python
# 应用提前停止的 GBM
# 使用一部分训练数据作为内部验证集用于提前停止
gbm_early_stop = GradientBoostingClassifier(n_estimators=1000, # 设置一个较高的潜在最大值
                                            learning_rate=0.05,
                                            max_depth=3,
                                            min_samples_leaf=10,
                                            subsample=0.7,
                                            max_features=0.8,
                                            validation_fraction=0.2, # 使用 20% 的训练数据进行验证
                                            n_iter_no_change=10,  # 如果 10 次迭代没有改进则停止
                                            tol=0.0001,
                                            random_state=42)

gbm_early_stop.fit(X_train, y_train)

# 评估性能（使用实际的验证集）
y_train_pred_es = gbm_early_stop.predict(X_train)
y_val_pred_es = gbm_early_stop.predict(X_val)
y_train_proba_es = gbm_early_stop.predict_proba(X_train)[:, 1]
y_val_proba_es = gbm_early_stop.predict_proba(X_val)[:, 1]

print("\n应用提前停止的 GBM 性能：")
print(f"  找到的最佳估计器数量: {gbm_early_stop.n_estimators_}")
print(f"  训练集准确率: {accuracy_score(y_train, y_train_pred_es):.4f}")
print(f"  验证集准确率: {accuracy_score(y_val, y_val_pred_es):.4f}")
print(f"  训练集对数损失: {log_loss(y_train, y_train_proba_es):.4f}")
print(f"  验证集对数损失: {log_loss(y_val, y_val_proba_es):.4f}")

# 备选方案：手动绘制验证误差与迭代次数的关系图
# 训练一个没有自动提前停止的模型
gbm_manual_es = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=42)
gbm_manual_es.fit(X_train, y_train)

# 计算分阶段对数损失（每次迭代后的性能）
staged_val_loss = [log_loss(y_val, proba[:, 1]) for proba in gbm_manual_es.staged_predict_proba(X_val)]
staged_train_loss = [log_loss(y_train, proba[:, 1]) for proba in gbm_manual_es.staged_predict_proba(X_train)]
best_iteration = np.argmin(staged_val_loss) + 1 # +1 因为迭代计数从 1 开始

print(f"\n手动提前停止分析：")
print(f"  最低验证对数损失发生在迭代次数: {best_iteration}")
print(f"  最佳迭代时的验证对数损失: {staged_val_loss[best_iteration-1]:.4f}")

# 训练损失与验证损失的可视化
iterations = np.arange(len(staged_val_loss)) + 1
```



![GBM 训练集与验证集对数损失](plots/1975-0.json)



> 随着提升迭代次数的增加，训练集和验证集上的对数损失。验证损失通常在初期下降，然后随着模型开始过拟合 (overfitting)而上升。提前停止旨在验证损失达到最小值附近时停止训练。

提前停止根据验证性能自动找到 `n_estimators` 的合适值，防止模型在树开始损害泛化能力时继续添加。该图清晰地显示了验证损失开始上升的点，这表明模型出现了过拟合。

### 比较与总结

让我们汇总每个模型的验证准确率和对数损失：

| 正则化 (regularization)方法 | 验证准确率 | 验证对数损失 | 备注 |
| --- | --- | --- | --- |
| 基准模型（过拟合 (overfitting)） | (Value from run) | (Value from run) | `max_depth` 较高，无显式约束 |
| 树约束 | (Value from run) | (Value from run) | `max_depth=3`，`min_samples_leaf=10` |
| + 收缩 | (Value from run) | (Value from run) | 较低的 `learning_rate=0.05`，更多的 `n_estimators` |
| + 子采样 | (Value from run) | (Value from run) | `subsample=0.7`，`max_features=0.8` |
| + 提前停止（自动） | (Value from run) | (Value from run) | 自动找到最优 `n_estimators` |

*（请将“(Value from run)”替换为执行代码时获得的实际指标。）*

你应该会发现，应用正则化方法通常会提升验证性能（更高的准确率，更低的对数损失），与过拟合的基准模型相比。通常，多种方法（如树约束、收缩、子采样和提前停止）的组合会产生最佳结果。

请注意，Scikit-learn 的 `GradientBoostingClassifier` 不像 XGBoost（我们稍后会介绍）那样直接在树叶权重 (weight)上实现 L1/L2 正则化。然而，这里演示的方法——约束树结构、收缩和子采样——是在标准 GBM 框架内控制模型复杂度并防止过拟合的有效手段。

本次实践练习显现了正则化的重要影响。通过审慎地应用这些方法，你可以构建对新数据泛化能力良好的梯度提升模型，而不仅仅是有效拟合训练集。尝试这些方法不同的参数 (parameter)值是模型调优过程的常规组成部分，我们将在后续章节中更详细地讨论。

## 参考资料

- [Greedy Function Approximation: A Unified Approach to Statistical Learning](https://www.jstor.org/stable/2676756) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Publisher: Institute of Mathematical Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1009210746](https://doi.org/10.1214/aos/1009210746)
  这篇基础论文介绍了梯度提升机算法及其理论基础，其中包含了与正则化相关的机制。
- [Stochastic Gradient Boosting](https://doi.org/10.1016/S0167-9473%2801%2900065-2) — Jerome H. Friedman (2002)
  Journal: Computational Statistics & Data Analysis; Publisher: Elsevier; Volume: 38; Pages: 367-378; DOI: [10.1016/S0167-9473(01)00065-2](https://doi.org/10.1016/S0167-9473%2801%2900065-2)
  介绍了随机梯度提升，它利用子采样进行正则化，这是实践部分直接应用的技术。
- [sklearn.ensemble.GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) — scikit-learn developers (2023)
  Scikit-learn梯度提升分类器的官方文档，详细说明了用于应用各种正则化技术的参数。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](http://www-stat.stanford.edu/ElemStatLearn) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  第10章对提升和加性树进行了全面讨论，包括统计特性和正则化策略。

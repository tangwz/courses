---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "gradient-boosting-with-scikit-learn"
lesson: "interpreting-gbm-parameters"
sourceId: 7587
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-3-gradient-boosting-with-scikit-learn/interpreting-gbm-parameters"
title: "解读模型参数"
description: "理解 `n_estimators`、`learning_rate`、`max_depth` 和 `subsample` 等主要参数对模型行为和性能的作用。"
order: 4
plots: []
sourceHash: "9509b69d0bb1bfad3cd8e1452d9e8fc9af309ad903d8a5d5be916adda8a5356c"
sourceCorrections: []
---

一旦实例化 `GradientBoostingClassifier` 或 `GradientBoostingRegressor` 对象，您不仅仅是创建了一个空模型，而是在定义学习过程如何进行的蓝图。这个蓝图由其参数 (parameter)控制，这些参数是管理模型复杂程度、训练速度和泛化表现的主要调节手段。

理解这些参数是您从默认模型转向优化模型的第一步。虽然有许多可用选项，但有几个主要参数对模型行为影响最大。我们将关注那些控制集成模型大小、学习速度及其单个树复杂程度的参数。

### 树的数量：n\_estimators

`n_estimators` 参数 (parameter)指定要构建的顺序树的总数。梯度提升集成中的每棵树都经过训练，以纠正其前一棵树的错误。因此，此参数直接控制提升阶段的数量。

- **作用：** 设置集成模型的大小。
- **对模型的影响：** 更多的树赋予模型更强的能力来拟合训练数据。这可以带来更好的性能，但有一定限度。如果 `n_estimators` 过大，模型可能开始过拟合 (overfitting)，学习训练数据中的噪声而非潜在信号。
- **默认值：** Scikit-Learn 中的默认值为 `100`。

实际中，您可以将 `n_estimators` 视为预算。允许的树越多，模型能够学习的功能就越复杂。

```python
from sklearn.ensemble import GradientBoostingRegressor

# 一个包含200个提升阶段的模型
gbr = GradientBoostingRegressor(n_estimators=200, random_state=42)
```

增加估计器通常会改进模型，但代价是训练时间更长，过拟合风险增加。此风险需与 `learning_rate` 配合管理。

### 控制步长：learning\_rate

`learning_rate`，常被称为“收缩率”，是模型正则化 (regularization)中影响最大的参数 (parameter)之一。它调整每棵树对最终预测的贡献度。较小的 `learning_rate` 意味着每棵树贡献较少，迫使模型在更新时更加保守。

模型在阶段 $m$ 的更新规则可以写成：


$$
F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)
$$


其中，$F_{m-1}(x)$ 是来自前一个集成树的预测，$h_m(x)$ 是新添加的树，$\nu$ 是 `learning_rate`。

- **作用：** 降低每棵独立树预测的权重 (weight)。
- **对模型的影响：** 较低的 `learning_rate` 值（例如 0.01, 0.05）使模型对过拟合 (overfitting)更稳定，但需要更多的 `n_estimators` 才能获得良好拟合。较高的值（例如 0.1, 0.2）使模型学习更快，但增加了过拟合的风险。
- **默认值：** Scikit-Learn 中的默认值为 `0.1`。

```python
# 一个学习率较小的模型
gbr_slow = GradientBoostingRegressor(n_estimators=200, 
                                     learning_rate=0.05, 
                                     random_state=42)
```

`n_estimators` 和 `learning_rate` 之间存在直接权衡。非常小的 `learning_rate` 可能需要数千个估计器才能收敛，而较大的 `learning_rate` 可能只需几百个就能收敛。这种关系是调整梯度提升模型的核心考量。

> 高学习率以大而快的步调趋向最佳拟合，需要较少的树。低学习率以小而谨慎的步调，通常能获得更好的拟合，但需要更多的树。

### 限制树的复杂程度：max\_depth

`n_estimators` 控制树的数量，而 `max_depth` 控制每棵独立树的复杂程度。集成中的每棵树都是一个弱学习器，它们的复杂程度必须受到限制，以防止它们在其残差部分上过拟合 (overfitting)。

- **作用：** 设置每棵决策树的最大深度。
- **对模型的影响：** 更深的树可以模拟更复杂的特征交互。然而，它们也更容易过拟合。梯度提升模型通常在浅层树上表现最佳，因为这迫使模型以加法方式学习，每棵树都捕捉数据关系中一个小的、独特的方面。
- **默认值：** 默认值为 `3`。常见值通常在 3 到 8 之间。

```python
# 一个具有浅层树的模型 (max_depth=2)
gbr_shallow = GradientBoostingRegressor(n_estimators=100, 
                                        learning_rate=0.1,
                                        max_depth=2,
                                        random_state=42)
```

限制树的深度是一种强大的正则化 (regularization)形式。其他相关参数 (parameter)，如 `min_samples_split`（分割节点所需的最小样本数）和 `min_samples_leaf`（叶节点所需的最小样本数），也有助于控制树的复杂程度并防止对小样本组过拟合。

### 引入随机性：subsample

`subsample` 参数 (parameter)为梯度提升过程引入了随机性元素，其灵感来自随机森林中使用的装袋（bagging）技术。它指定用于拟合每棵独立树的训练样本比例。在每个提升迭代中，样本都是无放回抽取的。

- **作用：** 在训练数据的随机子集上训练每棵树。
- **对模型的影响：** 通过在稍微不同的数据集上训练每棵树，`subsample` 降低了整体模型的方差，提升了其对未见数据的泛化能力。这种技术定义了 *随机梯度提升*。
- **默认值：** 默认值为 `1.0`，这意味着所有训练数据都用于每棵树。常见的做法是将其设置为 0.5 到 0.8 之间的值。

```python
# 一个实现随机梯度提升的模型
gbr_stochastic = GradientBoostingRegressor(n_estimators=100,
                                           learning_rate=0.1,
                                           subsample=0.8,
                                           random_state=42)
```

使用小于 1.0 的 `subsample` 值不仅能起到强大的正则化 (regularization)作用，还可以加快训练过程，因为每棵树都是用较少的数据点构建的。这四个参数，`n_estimators`、`learning_rate`、`max_depth` 和 `subsample`，共同构成了构建和优化梯度提升模型的基本要件。掌握它们的作用是充分发挥这些算法全部潜力的主要一步。

## 参考资料

- [Greedy Function Approximation: A Gradient Boosting Machine](https://doi.org/10.1214/aos/1013203451) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Publisher: The Institute of Mathematical Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  介绍梯度提升机算法的原始学术论文，涵盖收缩和集成规模的原则。
- [\`sklearn.ensemble.GradientBoostingRegressor\`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) — scikit-learn developers (2023)
  Scikit-Learn 的 `GradientBoostingRegressor` 官方文档，详细说明其参数及默认值。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; Pages: 337-352; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本标准教科书，涵盖梯度提升及其他集成方法的理论基础和实践方面。
- [Stochastic Gradient Boosting](https://doi.org/10.1016/S0167-9473%2801%2900065-2) — Jerome H. Friedman (2002)
  Journal: Computational Statistics & Data Analysis; Publisher: Elsevier; Volume: 38; Pages: 367-378; DOI: [10.1016/S0167-9473(01)00065-2](https://doi.org/10.1016/S0167-9473%2801%2900065-2)
  介绍了梯度提升中的 `subsample` 参数，这是一种用于减少方差的随机梯度提升方法。

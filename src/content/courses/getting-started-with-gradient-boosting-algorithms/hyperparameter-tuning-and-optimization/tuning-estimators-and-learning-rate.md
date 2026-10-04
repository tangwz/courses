---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "hyperparameter-tuning-and-optimization"
lesson: "tuning-estimators-and-learning-rate"
sourceId: 7606
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-6-hyperparameter-tuning-and-optimization/tuning-estimators-and-learning-rate"
title: "调整估计器数量和学习率"
description: "考察树的数量 (n_estimators) 与学习率之间的关系，以及如何平衡它们以获得最佳结果。"
order: 3
plots: ["plots/7606-0.json"]
sourceHash: "f269f3ac88ab874bdcabf85194d18e18bcbe24b2a77be25ef8b0aa452b5b66d3"
sourceCorrections: []
---

在梯度提升库的众多设置中，估计器数量和学习率可能是影响最大的。这两个超参数 (parameter) (hyperparameter)协同作用，控制模型的收敛性和拟合训练数据的能力。理解它们之间的关系对于构建有效的模型是基本的。

`n_estimators` 参数指定要构建的顺序树的总数。这等同于提升轮次。更多的估计器意味着模型有更多机会纠正其错误，这可以降低训练误差。然而，添加过多的树可能导致模型过拟合 (overfitting)，因为它开始模拟训练数据中的噪声，而不是底层信号。

`learning_rate`，常被称为 `eta` ($\eta$)，它调整每棵树的贡献。在新树被训练以拟合前一阶段的残差后，其预测值会乘以学习率，然后才被添加到整体模型中。较小的学习率会减小每棵树的贡献，从而需要向模型中添加更多的树。

### 逆向关系

这两个参数 (parameter)具有明显的逆向关系。较低的学习率需要较多的估计器才能实现良好的拟合，而较高的学习率则能以较少的估计器更快地拟合训练数据。

设想一个比喻：你正走下山坡，试图找到最低点。

- **高学习率** 就像迈出大步、自信地前进。你迅速下山，但有风险越过底部，走到另一边。
- **低学习率** 就像迈出小步、谨慎地前进。到达底部需要更长时间，但你更有可能找到真正的最小值而不会越过它。

在梯度提升中，这意味着高学习率可能导致模型收敛到次优解或快速过拟合 (overfitting)。较低的学习率会带来更稳定的、通常泛化效果更好的模型，但这会增加计算时间，因为需要更多的树。

经过 $M$ 轮提升后的最终模型预测 $F_M(x)$，是初始预测和所有后续树的贡献的累加，并由学习率 $\eta$ 进行缩放：


$$
F_M(x) = F_0(x) + \eta \sum_{m=1}^{M} h_m(x)
$$


这个方程展示了 $\eta$ 如何调节每棵新树 $h_m(x)$ 对最终预测的影响。

下面的图表展示了这种动态。学习率较高（`learning_rate=0.3`）的模型，其验证误差迅速下降，但随后开始上升，表明过拟合。相比之下，学习率较低（`learning_rate=0.05`）的模型学习速度较慢，但在更多的提升轮次后最终取得了更好的验证分数。



![学习率对模型表现的影响](plots/7606-0.json)



> 较低的学习率需要更多的估计器才能达到最佳表现，但与较高的学习率相比，通常会产生更低的最终验证误差。

### 采用早停的实用调整策略

寻找 `n_estimators` 和 `learning_rate` 的最佳组合可能计算成本很高。一种常见且有效的策略是使用**早停**。这种技术在训练过程中监测模型在单独验证集上的表现，并在验证分数在指定轮次内停止提升时终止进程。这会自动找到给定学习率下的最佳估计器数量。

下面是一个推荐的工作流程：

1. **选择一个相对较高的学习率。** 一个常见的起点是 `learning_rate=0.1`。这使得训练速度可以合理地快。
2. **使用早停确定最佳 `n_estimators`。** 使用大量的潜在估计器（例如，`n_estimators=1000`）来训练模型，但使用早停参数 (parameter)来停止训练。

例如，在XGBoost中，你可以这样实现：

```python
import xgboost as xgb

# 假设 X_train, y_train, X_val, y_val 已定义
model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    eval_metric='logloss',
    early_stopping_rounds=50
)

# 开始训练
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"最佳估计器数量: {model.best_iteration}")
```

`early_stopping_rounds=50` 参数指示模型，如果验证损失（此处为 `logloss`）连续50轮没有改善，就停止训练。最佳树的数量存储在 `model.best_iteration` 属性中。

3. **调整其他超参数 (hyperparameter)。** 在学习率和估计器数量设定到合理值后，你可以继续调整其他控制树复杂度的参数，例如 `max_depth` 和 `subsample`。
4. **降低学习率并增加估计器数量。** 在调整了其他参数后，通常可以通过降低学习率并按比例增加估计器数量来获得最终的表现提升。例如，如果你发现学习率为0.1时，300个估计器是最佳的，你可以尝试学习率为0.05，`n_estimators` 约600；或者学习率为0.01，`n_estimators` 约3000，同样再次使用早停来找到新的最佳树数量。这最后一步会缓慢地将模型微调 (fine-tuning)到更好的最小值。

这种结构化方法比同时对两个参数进行暴力网格搜索要高效得多。它使你能够通过迭代地改进模型的配置来找到一组高性能的设置组合。

## 参考资料

- [Greedy Function Approximation: A Gradient Boosting Machine](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-Function-Approximation-A-Gradient-Boosting-Machine/10.1214/aos/1013203445) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Publisher: Institute of Mathematical Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203445](https://doi.org/10.1214/aos/1013203445)
  这篇基础论文介绍了梯度提升机算法，详细阐述了提升迭代次数（估计器数量）和收缩参数（学习率）在模型构建和正则化中的作用。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://link.springer.com/book/10.1007/978-0-387-84858-7) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  这本权威书籍的第10章全面讲解了提升算法，包括对学习率（收缩）和估计器数量及其对模型性能影响的详细解释。
- [XGBoost Python API Reference](https://xgboost.readthedocs.io/en/stable/python/python_api.html) — XGBoost Contributors (2023)
  XGBoost Python包的官方API参考，详细介绍了`XGBClassifier`类的参数，如`n_estimators`和`learning_rate`（eta），以及`fit`方法的`early_stopping_rounds`参数，用于高效调优。

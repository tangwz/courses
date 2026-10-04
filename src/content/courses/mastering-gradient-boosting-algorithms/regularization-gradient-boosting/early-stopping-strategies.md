---
course: "mastering-gradient-boosting-algorithms"
chapter: "regularization-gradient-boosting"
lesson: "early-stopping-strategies"
sourceId: 1971
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-3-regularization-gradient-boosting/early-stopping-strategies"
title: "早停策略"
description: "使用验证集来确定最佳的提升迭代次数。"
order: 6
plots: ["plots/1971-0.json"]
sourceHash: "5dee3c72b5074907046e697bcf51a3701f117c70a0d7ebf4639a573bad011807"
sourceCorrections: []
---

如我们之前所讲，梯度提升模型按顺序学习，逐次添加弱学习器（通常是树）以纠正前一个集成模型的错误。尽管这种累加过程功能强大，但它可能无限期地持续下去，越来越紧密地拟合训练数据。如果没有干预，模型最终将开始拟合训练数据中固有的噪声，导致过拟合 (overfitting)和在未见数据上表现不佳。早停提供了一种实用且广泛使用的方案来确定最佳的提升迭代次数。

主要思路直接明了：在训练过程中，在一个独立的验证数据集上监测模型的表现，当验证表现不再提升时停止训练。

### 早停如何工作

1. **数据划分：** 在训练之前，您需要将数据划分为至少三个集合：训练集（用于计算梯度和构建树）、验证集（*仅用于*监测性能以进行早停），和测试集（用于训练完成后进行最终、无偏的评估）。验证集应能代表模型在实际部署中将遇到的数据。
2. **监测表现：** 随着提升算法迭代地添加树（也称为轮次或周期），模型的表现将使用选定的度量标准在验证集上进行评估（例如，分类任务的对数损失，回归任务的均方根误差）。
3. **停止条件：** 如果验证分数在预设的连续迭代次数内没有提升，训练就会停止。这个“耐心”参数 (parameter)，在库中常被称为 `early_stopping_rounds`，防止由于验证分数微小的随机波动而过早停止。
4. **模型选择：** 算法通常返回在验证集上产生最佳分数的迭代所对应的模型，而不一定是停止前的最后一次迭代所对应的模型。

### 早停作为一种正则化 (regularization)方法

通过根据验证表现限制提升迭代次数，早停起到了一种正则化的作用。它阻止模型变得过于复杂并拟合训练数据中的噪声。添加更多的树会增加模型的容量；早停找到一个点，在该点之后，容量的进一步增加主要捕捉噪声而不是潜在模式，从而有效控制模型复杂度。

以下是梯度提升模型的典型学习曲线：



![带有早停的典型学习曲线](plots/1971-0.json)



> 训练误差通常持续下降，而验证误差最初会下降，但随着模型过拟合 (overfitting)，随后开始上升。早停的目标是在验证误差曲线的最低点附近停止训练（本例中为第80次迭代）。虚线表示在停止之前性能没有提升的耐心期。

### 实际应用中的实现

大多数现代梯度提升库（XGBoost、LightGBM、CatBoost）都内置支持早停功能。通常，您在调用 `fit` 时通过提供以下内容来启用它：

1. 评估集（或多个评估集）：这是您的验证数据。
2. 评估指标：用于监测验证数据的指标。
3. 早停轮次数量（`early_stopping_rounds`）：即“耐心”值。

例如，在XGBoost中，调用可能如下所示：

```python
# 示例 (XGBoost API)
eval_set = [(X_train, y_train), (X_val, y_val)]

model.fit(X_train, y_train,
          eval_set=eval_set,
          eval_metric='logloss', # 或 'rmse' 等
          early_stopping_rounds=10, # 如果验证集上的对数损失在10轮内没有提升，则停止
          verbose=True) # 查看每轮的表现
```

这种方法显著简化了超参数 (parameter) (hyperparameter)调整。您无需精心调整树的数量（`n_estimators`），可以将其设置为一个合理大的数值，让早停根据验证表现自动找到最佳停止点。

### 注意事项

- **验证集大小和质量：** 早停的有效性很大程度上取决于验证集是否足够大且能代表模型将面对的数据分布。小规模或不具代表性的验证集可能导致性能估算出现噪声，并产生次优的停止点。
- **指标选择：** 确保用于早停的评估指标与您的最终建模目标一致。
- **耐心值：** 将 `early_stopping_rounds` 设置得过低可能由于噪声导致过早停止。设置得过高可能在停止前允许一定程度的过拟合 (overfitting)。这个值本身通常也需要一些调整，但通常不如直接调整 `n_estimators` 那么关键。
- **与学习率的相互作用：** 较小的学习率通常需要更多的提升轮次。因此，当学习率较低时，早停可能会建议更多的轮次。

早停是梯度提升模型正则化 (regularization)的一种基本方法。它提供了一种计算上高效的方式，通过根据未见数据上的表现动态确定提升轮次数量来防止过拟合，使其成为梯度提升模型在实际操作中不可或缺的工具。

## 参考资料

- [Greedy Function Approximation: A Gradient Boosting Machine](https://www.stat.stanford.edu/~jhf/ftp/gbm.pdf) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Publisher: Institute of Mathematical Statistics (IMS); Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  这篇论文介绍了梯度提升机（GBM）算法，为现代提升方法提供了理论基础。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  涵盖统计学习方法，包括提升算法和如提前停止等正则化技术。
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html#training-parameters) — XGBoost Contributors (2024)
  解释XGBoost中提前停止参数及用法的官方文档。
- [Parameters (LightGBM)](https://lightgbm.readthedocs.io/en/latest/Parameters.html#core-parameters) — Microsoft and LightGBM Contributors (2024)
  详细说明LightGBM库中提前停止参数及其用法的官方文档。

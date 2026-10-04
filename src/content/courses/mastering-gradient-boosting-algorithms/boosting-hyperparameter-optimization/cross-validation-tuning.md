---
course: "mastering-gradient-boosting-algorithms"
chapter: "boosting-hyperparameter-optimization"
lesson: "cross-validation-tuning"
sourceId: 2061
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-8-boosting-hyperparameter-optimization/cross-validation-tuning"
title: "调优的交叉验证策略"
description: "将交叉验证有效地整合到调优流程中。"
order: 7
plots: ["plots/2061-0.json"]
sourceHash: "c9863a6d30a95e9e8107388050735f8e15aae67cb2936a26fcddc9475fc75454"
sourceCorrections: []
---

在调优梯度提升模型的超参数 (parameter) (hyperparameter)时，仅依靠单一的训练-验证集划分来评估性能可能会产生误导。选定的超参数可能对该特定的验证集过拟合 (overfitting)，导致在未见过的数据上泛化能力下降。交叉验证 (CV) 通过在数据的多个不同子集上评估模型，为给定超参数集提供了更可靠的性能估计。将 CV 有效地整合到您的调优流程中，对于构建在实际中表现良好的模型是必要的。

### 交叉验证在超参数 (parameter) (hyperparameter)调优中的作用

回顾一下，网格搜索、随机搜索或贝叶斯优化等调优方法通过提出不同的超参数配置并评估其性能来工作。我们不使用单个验证集进行此评估，而是使用交叉验证。

在调优过程中测试的每个超参数组合：

1. *训练*数据被临时划分为 $K$ 个折叠（子集）。
2. 模型被训练 $K$ 次。每次训练都在 $K-1$ 个折叠上进行，并在剩余的一个折叠（保留折叠）上进行验证。
3. 性能指标（例如，AUC、RMSE、LogLoss）在每次 $K$ 运行的保留折叠上计算。
4. $K$ 个性能分数被平均（或有时使用其他统计量如标准差进行汇总），以提供该特定超参数配置的单一、更稳定的性能估计。

这个平均分数随后被调优算法（例如，贝叶斯优化）使用，以决定下一步尝试哪些超参数，或者在网格搜索的情况下，从已测试的组合中找出最佳组合。

### 标准K折交叉验证

最常见的CV策略是K折交叉验证。训练数据被打乱并分成 $K$ 个大小相等的折叠。通常，$K$ 选择为5或10。更高的 $K$ 值在每次迭代中会使用更多数据进行训练，但会增加计算时间。

> 超参数 (parameter) (hyperparameter)调优循环中的基本K折交叉验证过程。

### 用于分类的分层K折交叉验证

在分类问题中，特别是处理不平衡数据集时，标准K折交叉验证可能会随机创建折叠，导致类别分布与整体分布存在显著差异。这可能导致不可靠的性能估计。

**分层K折交叉验证**通过确保每个折叠中每个类别的样本百分比与完整数据集中观察到的百分比一致来解决这个问题。它是分类任务的推荐默认选项。大多数库（如Scikit-learn）都提供特定的实现（`StratifiedKFold`）。

### 用于相关数据的分组K折交叉验证

有时，数据点不是独立的。例如，您可能有来自同一患者的多次测量、同一地点的图像或同一用户会话的日志。如果使用标准K折交叉验证，来自同一组的数据可能会在给定的划分中同时出现在训练集和验证集中。这种数据泄露可能导致过于乐观的性能估计，因为模型学会了识别特定组而不是一般模式。

**分组K折交叉验证**确保属于同一组的所有样本在每次划分中完全分配到训练集或验证集中的一个。您需要一个用于分组的标识符（例如，`patient_id`、`user_id`）。这可以防止模型“偷看”它将被测试的同一组数据，从而更实际地评估泛化性能。

### 时间序列交叉验证

"时间序列数据带来了一个独特的问题：时间顺序很重要。像标准K折交叉验证那样随机打乱数据，会破坏时间依赖性并导致“未来数据偏差”——即利用未来数据来预测过去，这在实际情况中是不可能的。"

时间序列交叉验证的策略保持时间顺序：

1. **滚动预测起点（或TimeSeriesSplit）：** 训练集随时间向前扩展或滑动。

   - 在折叠1上训练，在折叠2上验证
   - 在折叠1-2上训练，在折叠3上验证
   - 在折叠1-2-3上训练，在折叠4上验证
   - ...依此类推。
     " 这模拟了随着新数据可用而定期重新训练模型的场景。Scikit-learn 的 `TimeSeriesSplit` 实现了这一点。"
2. **滑动窗口：** 类似于滚动预测，但训练窗口大小固定，随时间向前滑动。

   - 在折叠1-2上训练，在折叠3上验证
   - 在折叠2-3上训练，在折叠4上验证
   - 在折叠3-4上训练，在折叠5上验证



![时间序列交叉验证 (滚动预测)](plots/2061-0.json)



> 使用滚动预测起点表示时间序列交叉验证。绿色块表示训练数据，红色块表示每个划分迭代的验证数据。

选择合适的时间序列划分取决于您是否预期模式会随时间变化（滑动窗口可能更好），或者旧数据是否仍然具有相关性（滚动预测）。

### 将CV与调优框架结合

现代的超参数 (parameter) (hyperparameter)优化库被设计为与交叉验证配合工作。

- **Scikit-learn：** `GridSearchCV` 和 `RandomizedSearchCV` 等类都有一个 `cv` 参数，您可以在其中指定折叠数量（例如，`cv=5`）或传递一个特定的CV划分器对象（例如，`cv=StratifiedKFold(n_splits=5)` 或 `cv=TimeSeriesSplit(n_splits=5)`）。该框架会自动对它评估的每个超参数集执行CV循环。
- **Optuna/Hyperopt：** 在定义这些库优化的目标函数时，您通常会在其中包含一个交叉验证循环。该函数接受一组超参数（在Optuna中称为 `trial`），执行K折交叉验证（或另一种策略），计算各折叠的平均分数，并返回此分数。然后，优化库使用此返回的分数来指导其搜索。

### 与提前停止的配合

梯度提升模型通常受益于提前停止，以找到最佳的提升轮数（`n_estimators`）并防止过拟合 (overfitting)。将提前停止与交叉验证结合需要谨慎。

在CV循环中评估*单个*超参数 (parameter) (hyperparameter)集的一种常见方法是：

1. 对于 $K$ 个折叠中的每一个：
   - 将 $K-1$ 个训练折叠进一步划分为子训练集和提前停止验证集。
   - 在子训练集上训练模型，使用提前停止集来确定*该折叠*的最佳轮数。
   - 记录在使用为该折叠找到的最佳轮数训练的模型在主要保留验证折叠上的性能分数。同时，记录所使用的轮数。
2. 平均 $K$ 个折叠的性能分数。这就是正在评估的超参数集的分数。
3. 可选地，平均或找出 $K$ 个折叠中找到的最佳轮数的中位数。

当训练*最终*模型时（在使用上述CV过程找到最佳超参数后），您在*整个*训练数据集上进行训练。这个最终模型的提升轮数可以设置为在CV期间找到的平均/中位数轮数，或者如果可用，可以使用单独的最终验证集来确定。一些库（如XGBoost和LightGBM）允许在主要的 `.fit()` 调用期间传递评估集，从而在此最终训练阶段启用提前停止。

### 计算成本和最终模型训练

交叉验证将调优过程的计算成本乘以 $K$ 倍。使用5折CV评估100个超参数 (parameter) (hyperparameter)组合意味着训练500个模型。这是获得可靠性能估计的必要成本。为了管理这一点：

- 如果计算成本过高，使用较少的折叠（$K=3$ 或 $K=5$）。
- 优先选择随机搜索或贝叶斯优化等高效搜索策略，而不是穷举网格搜索。
- 考虑进行初步的宽泛搜索，使用较少的折叠或较少的数据，然后对超参数空间中有希望的区域进行更精细的搜索。

重要的是，请记住调优过程中的交叉验证*仅*用于评估超参数集。一旦找到最佳超参数，您将使用这些最优参数在**整个训练数据集**上最后一次训练您的**最终模型**。从交叉验证过程中获得的性能估计，可作为您对这个最终模型在新、未见过的数据上表现的预期。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; Pages: 241-249
  涵盖交叉验证的理论基础及其用于模型评估和选择的各种形式。
- [Cross-validation iterators](https://scikit-learn.org/stable/modules/cross_validation.html) — scikit-learn developers (2024)
  提供KFold、StratifiedKFold、GroupKFold和TimeSeriesSplit的实际实现和描述，与所讨论的方法直接相关。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 70-74, 84-88, 301-303
  提供使用Scikit-learn将交叉验证与超参数调整技术结合的实践指导。

# XGBoost API: 参数与配置

来源：[原文](https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-4-xgboost-extreme-gradient-boosting/xgboost-api-parameters)

[返回章节目录](README.md) · [返回课程目录](../README.md)

XGBoost的实际应用主要围绕其可配置参数 (parameter)展开。其强大功能和灵活性很大程度上源于这些广泛的设置。理解这些参数对于控制模型行为、优化性能和获得良好泛化能力非常重要。

XGBoost库提供了多种接口，特别是其原生Python API和与Scikit-learn兼容的包装器。虽然参数名称可能略有不同（例如，`eta`与`learning_rate`，`lambda`与`reg_lambda`），但它们的功能保持不变。我们将侧重于适用于这两种接口的核心思想。参数通常分为三类：通用参数、提升器参数和学习任务参数。

### 通用参数 (parameter)

这些参数与提升过程的整体运行有关。

- **`booster`** [default=`gbtree`]: 选择要使用的基础学习器（提升器）类型。选项通常包括 `gbtree`（基于树的模型，最常见）、`gblinear`（线性模型）或 `dart`（为树引入dropout）。我们主要关注 `gbtree`。
- **`verbosity`** [default=1]: 控制训练期间打印的日志信息量。0（静默）、1（警告）、2（信息）、3（调试）。有助于监控进度，但不会影响模型本身。
- **`nthread`** [default=maximum available]: 指定用于训练的并行线程数。设置此项对于资源管理可能很重要，尤其是在共享环境中。Scikit-learn 包装器通常使用 `n_jobs`。

### 提升器参数 (parameter)（针对 `gbtree`）

这些参数直接影响每个提升迭代中构建的单个树。它们是控制模型复杂度并防止过拟合 (overfitting)的主要方式。

#### 学习率

- **`eta`** (或 `learning_rate`) [default=0.3]: 这是应用于每个提升步骤中新添加权重 (weight)（树）的收缩因子。它减少了每棵树的影响，并为未来的树改进模型留下了空间。较低的值（例如0.01-0.1）通常会导致模型数量增多，降低过拟合的风险，但需要更多的提升轮数（`n_estimators` 或 `num_boost_round`）。

#### 树结构控制

- **`max_depth`** [default=6]: 定义每棵树的最大深度。增加此值会使模型更复杂，更容易捕获更细致的相互影响，但也会显著增加过拟合的风险和计算时间。典型值范围为3到10。
- **`min_child_weight`** [default=1]: 指定子节点中所需的实例权重（Hessian）的最小总和。如果树分割步骤导致叶节点的实例权重总和小于 `min_child_weight`，则构建过程将放弃进一步分割。在线性回归中，这仅仅对应于每个节点所需的最小样本数。较大的值会产生更保守的树，防止模型学习特定于少量、可能带有噪声的样本组的关系。此参数作为一种正则化 (regularization)形式。
- **`gamma`** (或 `min_split_loss`) [default=0]: 设置进行分割所需的最小损失减少量。只有当分割结果使损失函数 (loss function)至少改善 `gamma` 时，才会进行节点分割。较大的值会导致更少的分割和更保守的模型。它直接与从XGBoost正则化目标函数派生出的增益计算相关，有效地为分割所需的改进设定了一个阈值。

#### 正则化

- **`lambda`** (或 `reg_lambda`) [default=1]: 权重上的 $L_2$ 正则化项（类似于岭回归）。增加此值会通过惩罚叶节点中的大权重使模型更保守。
- **`alpha`** (或 `reg_alpha`) [default=0]: 权重上的 $L_1$ 正则化项（类似于Lasso回归）。可以导致叶得分的稀疏性。在处理高维特征空间时很有用。

#### 子采样

这些参数引入随机性，使模型更能抵抗噪声并防止过拟合。它们是“随机梯度提升”方法的依据。

- **`subsample`** [default=1]: 为构建每棵树随机采样的训练实例（行）比例。将其设置为0.7或0.8之类的值意味着XGBoost在生长树之前随机选择70%或80%的数据，从而减少方差。
- **`colsample_bytree`** [default=1]: 构建每棵树时随机采样的特征（列）比例。
- **`colsample_bylevel`** [default=1]: 在树内每个分割级别随机采样的特征比例。
- **`colsample_bynode`** [default=1]: 在树内每个节点（分割）随机采样的特征比例。

使用 `colsample_by*` 参数增加了另一层随机性，在处理高维数据集时可能特别有效。在这些参数中，`colsample_bytree` 通常是最常调整的。

#### 其他重要提升器参数

- **`tree_method`** [default='auto']: 树构建算法。'auto' 通常根据数据集大小进行选择。'exact' 使用精确贪婪算法（对于大型数据集计算密集）。'approx' 使用基于分位数的近似贪婪算法进行草图绘制。'hist' 使用基于直方图的贪婪算法（通常更快、更节省内存，类似于 LightGBM）。了解您的数据大小和计算限制有助于选择合适的方法。
- **`scale_pos_weight`** [default=1]: 控制正负权重的平衡，对于不平衡分类问题很有用。一个典型值是 `sum(负实例) / sum(正实例)`。

### 学习任务参数 (parameter)

这些参数定义了训练期间使用的优化目标和评估指标。

- **`objective`** [default=`reg:squarederror`]: 指定学习任务和相应的目标函数。常见选项包括：
  - `reg:squarederror`：使用均方误差损失的回归。
  - `reg:logistic`：逻辑回归（输出概率）。
  - `binary:logistic`：用于二元分类的逻辑回归（输出概率）。
  - `binary:logitraw`：用于二元分类的逻辑回归（输出逻辑转换前的得分）。
  - `multi:softmax`：使用 softmax 目标的多元分类。需要设置 `num_class`。输出预测类别。
  - `multi:softprob`：与 `softmax` 相同，但输出每个类别的概率向量 (vector)。需要 `num_class`。
  - `rank:pairwise`：使用成对损失的学习排序任务。
- **`eval_metric`**: 用于验证数据的评估指标。XGBoost 支持多种指标。例如：
  - `rmse`：均方根误差（回归）。
  - `mae`：平均绝对误差（回归）。
  - `logloss`：负对数似然（分类）。
  - `error`：二元分类错误率。
  - `merror`：多元分类错误率。
  - `auc`：ROC曲线下面积（分类）。
  - `map`：平均精度（排序）。
  - `ndcg`：归一化 (normalization)折现累积增益（排序）。
    您可以提供多个指标，如果启用，最后一个通常用于提前停止。

### 训练控制参数 (parameter)

虽然不严格属于模型本身的参数，但它们控制着训练过程：

- **`n_estimators`** (Scikit-learn 包装器) 或 **`num_boost_round`** (原生 API): 要构建的提升轮数（树数）。这是一个非常重要的参数；轮数过少会导致欠拟合 (underfitting)，而过多则可能导致过拟合 (overfitting)（尽管提前停止有助于缓解这一点）。
- **`early_stopping_rounds`** （在 `fit` 方法中与 `eval_set` 一起使用）: 启用提前停止。如果 `eval_metric` 指定的验证指标在给定数量的连续轮次中没有改善，训练将停止。强烈建议使用此功能，以防止过拟合并自动找到最佳提升轮数。训练期间需要通过 `eval_set` 参数至少提供一个验证集。

### 配置 XGBoost：实际考量

选择合适的参数 (parameter)通常涉及平衡偏差-方差权衡、计算成本和特定数据集特点。

- **从默认值开始，系统地调整：** 从默认设置开始，并使用适当的交叉验证建立基线性能。
- **调整学习率和估计器数量：** 这些参数通常一起调整。较低的 `eta` 通常需要较高的 `n_estimators`。使用提前停止来找到给定 `eta` 的最佳轮数。
- **控制树复杂度：** 调整 `max_depth`、`min_child_weight` 和 `gamma` 以管理过拟合 (overfitting)。`max_depth` 影响很大，其次是 `min_child_weight`。`gamma` 基于损失减少提供更细致的控制。
- **引入随机性：** 将 `subsample` 和 `colsample_bytree`（以及可能的其他参数，例如 `colsample_bylevel`）调整为小于1的值。这通常会改善泛化能力，尤其是在有噪声或高维数据上。
- **应用正则化 (regularization)：** 如果需要进一步正则化，特别是当 `min_child_weight` 或 `gamma` 不足时，使用 `lambda` ($L_2$) 和 `alpha` ($L_1$)。
- **考虑 `tree_method`：** 对于大型数据集，从 `exact` 切换到 `hist` 或 `approx` 可以显著加快训练速度，对准确性影响可能很小。

下图显示了学习率（`eta`）与在验证集上达到最佳性能所需的提升轮数（`n_estimators`）之间的典型关系，假设使用了提前停止。



[交互图表：学习率（eta）对最佳提升轮数的影响](https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-4-xgboost-extreme-gradient-boosting/xgboost-api-parameters#plot-1dh2y9n)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "学习率（eta）对最佳提升轮数的影响",
    "xaxis": {
      "title": "学习率 (eta)"
    },
    "yaxis": {
      "title": "最佳估计器数量（提前停止）"
    },
    "legend": {
      "traceorder": "reversed"
    }
  },
  "data": [
    {
      "x": [
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.3
      ],
      "y": [
        2500,
        1300,
        550,
        280,
        150,
        100
      ],
      "type": "scatter",
      "mode": "lines+markers",
      "name": "最佳估计器数量",
      "marker": {
        "color": "#4263eb"
      }
    }
  ]
}
```

</details>



> 较低的学习率需要更多的提升轮数（树）才能达到最佳性能，但通常会带来更好的泛化能力。提前停止有助于自动找到这个最佳点。

掌握这些参数使您能够充分发挥 XGBoost 的潜力，根据机器学习 (machine learning)任务的特定难点定制其强大的引擎。接下来的步骤涉及查找超参数 (hyperparameter)优化框架（第8章会介绍），以有效地在广泛的参数空间中进行搜索。目前，当您开始实现 XGBoost 模型时，请着重理解此处讨论的每个主要参数的作用和影响。

## 参考资料

- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html) — XGBoost Contributors (2024)
  提供了章节中讨论的所有可配置参数的详细信息，包括其描述和默认值。
- [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785) — Tianqi Chen and Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 785-794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  介绍了原始算法、正则化目标函数和系统设计，为理解XGBoost参数提供了理论基础。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://hastie.su.domains/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本基础教材，涵盖了提升法、决策树和正则化的理论背景，对于掌握XGBoost参数背后的原理至关重要。
- [Scikit-learn API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn) — XGBoost Contributors (2024)
  详细介绍了XGBoost的Scikit-learn兼容API，阐明了`XGBClassifier`和`XGBRegressor`的参数映射和用法。

---

[上一节](06-%E7%B3%BB%E7%BB%9F%E4%BC%98%E5%8C%96%EF%BC%9A%E7%BC%93%E5%AD%98%E6%84%9F%E7%9F%A5%E4%B8%8E%E5%B9%B6%E8%A1%8C%E5%A4%84%E7%90%86.md) · [下一节](08-%E5%AE%9E%E8%B7%B5%E6%93%8D%E4%BD%9C%EF%BC%9A%E5%AE%9E%E7%8E%B0XGBoost.md)

# LightGBM API：参数与配置

来源：[原文](https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-5-lightgbm-light-gradient-boosting/lightgbm-api-parameters)

[返回章节目录](README.md) · [返回课程目录](../README.md)

LightGBM 采用创新技术来提高效率和速度，其中包括基于梯度的单侧采样（GOSS）、专属特征捆绑（EFB）、基于直方图的分裂以及叶子生长式树构建。为了有效利用 LightGBM，需要理解如何通过其功能完善的 API 控制这些机制。此处提供了 LightGBM Python 库中主要参数 (parameter)的指南，使您能够为不同任务和数据集有效配置模型。

LightGBM 库提供与 Scikit-learn 相似的接口（`LGBMClassifier`、`LGBMRegressor`），以及其自身的原生训练 API。虽然参数名称在不同接口间可能略有差异（例如 `num_iterations` 与 `n_estimators`），但其基本功能保持不变。我们将主要使用在 `LGBMClassifier`/`LGBMRegressor` 上下文 (context)中常用的名称，并在相关时提及其他选项。

### 核心提升参数 (parameter)

这些参数控制着整体的提升过程：

- `objective`：定义要优化的损失函数 (loss function)。常见选项包括 `regression` (L2 损失), `regression_l1` (L1 损失), `huber` (Huber 损失), `binary` (二分类的对数损失), `multiclass` (多分类的 softmax 目标), 以及 `lambdarank` (用于排序任务)。也可以提供自定义目标，这将在第 7 章讨论。
- `boosting_type` (或 `boosting`)：指定提升算法。
  - `gbdt`：传统的梯度提升决策树算法。
  - `dart`：采用 Dropout Meets Multiple Additive Regression Trees (DART)，这可以提高模型健壮性但通常需要更多迭代。
  - `goss`：使用基于梯度的单侧采样，如前所述。这在大型数据集上通常更快，但可能需要调整 `top_rate` 和 `other_rate`。
- `num_iterations` (或 `n_estimators`)：要构建的提升轮数（树的数量）。这是最具影响力的参数之一。迭代次数过少会导致欠拟合 (underfitting)，而过多则可能导致过拟合 (overfitting)。通常与 `learning_rate` 和早停机制一起调整。
- `learning_rate` (或 `eta`)：控制每次迭代的步长，减小每棵新树的贡献。较低的值通常需要更多的 `num_iterations`，但可以带来更好的泛化能力。典型值范围从 0.01 到 0.3。

### 树结构参数 (parameter)

这些参数控制着每次提升迭代中生成的单个决策树的复杂度与形态。由于 LightGBM 采用叶子生长策略，这些参数与标准 GBM 或 XGBoost 等层级生长算法的交互方式有所不同。

> 树生长策略比较。层级生长逐层扩展，而 LightGBM 的叶子生长策略扩展增益最大的节点，可能更快地生成更深、更不对称的树。

- `num_leaves`：**LightGBM 中控制树复杂度的最重要参数。** 它定义了单棵树的最大叶子数量。与 `max_depth` 不同，它直接限制了叶子生长树的复杂度。增加 `num_leaves` 允许模型捕获更复杂的模式，但会增加过拟合 (overfitting)的风险。通常将 `num_leaves` 设置为远小于 $2^{max\_depth}$ 的值。
- `max_depth`：尽管叶子生长是默认方式，但您仍可以设置最大深度，以限制任何分支的深度。这作为防止过拟合的另一种保障，尤其当 `num_leaves` 很大时。值 -1 表示没有限制。
- `min_data_in_leaf` (或 `min_child_samples`)：一个重要的正则化 (regularization)参数。它指定叶节点所需的最小数据点数量。更大的值可以防止模型学习仅针对非常小实例组的模式，从而改善泛化能力。
- `min_sum_hessian_in_leaf` (或 `min_child_weight`)：与叶节点相关的另一种正则化参数。它设置了叶节点中所需的 Hessian 值（损失函数 (loss function)的二阶导数）的最小和。对于 L2 损失，这与 `min_data_in_leaf` 对应。对于其他损失函数，它提供了一种更具统计学依据的方式来控制叶子形成。

### 正则化 (regularization)参数 (parameter)

LightGBM 提供显式正则化：

- `lambda_l1` (或 `reg_alpha`)：权重 (weight)（叶子值）上的 L1 正则化项。鼓励叶子输出的稀疏性。
- `lambda_l2` (或 `reg_lambda`)：权重上的 L2 正则化项。主要的正则化项，通过缩小叶子输出来帮助防止过拟合 (overfitting)。
- `min_gain_to_split` (或 `min_split_gain`)：进行分裂所需的最小增益（损失减少量）。正值通过修剪未能充分改善模型的分裂来起到正则化作用。

### 采样参数 (parameter)

这些参数通过对数据或特征进行子采样来引入随机性，这有助于正则化 (regularization)，有时还能加快训练速度：

- `feature_fraction` (或 `colsample_bytree`)：指定为构建每棵树随机选择的特征比例。如果设置为低于 1.0，有助于防止过拟合 (overfitting)，并且在处理大量特征时特别有用。这补充了 EFB，EFB 是捆绑特征而非采样特征。
- `bagging_fraction` (或 `subsample`)：指定每次提升迭代随机采样（不放回）的数据实例比例。需要 `bagging_freq` > 0。这是随机梯度提升的核心。
- `bagging_freq`：执行 bagging 的频率（以迭代次数计）。如果设置为 `k`，则每 `k` 次迭代执行一次 bagging。`0` 表示禁用 bagging。
- `feature_fraction_bynode`：指定在树中每个节点分裂时要考虑的特征比例。这在分裂层面引入了额外的随机性。

### 效率与算法控制

这些参数 (parameter)与 GOSS、EFB 和直方图技术直接相关：

- `boosting_type='goss'`：如前所述，选择 GOSS。参数 `top_rate`（保留大梯度实例的比例）和 `other_rate`（随机采样小梯度实例的比例）控制其行为。调整这些参数是使用 GOSS 的特定要求。
- `enable_bundle` (默认值: True)：控制是否启用专属特征捆绑。禁用它可能在调试时有必要，但通常会降低效率。
- `max_bin`：控制连续特征值离散化（直方图构建）的最大 bin 数量。较小的值会提高训练速度，并可作为正则化 (regularization)手段，但可能导致信息丢失和次优分裂。较大的值会增加精度潜力，但会减慢训练速度并增加内存使用。典型值范围从 63 到 255。

### 类别特征处理

LightGBM 提供对类别特征的原生支持，通常表现优于独热编码：

- `categorical_feature`：用于指定类别列的索引或名称。LightGBM 使用专门的算法（Fisher 方法）来在这些特征上寻找最优分裂。
- `max_cat_threshold`：当使用专门的类别处理算法寻找分裂时，限制考虑的类别数量。
- `cat_smooth`：在分裂计算期间，为类别频率和梯度添加平滑处理，有助于防止在具有多个级别或稀疏数据的类别特征上发生过拟合 (overfitting)。
- `cat_l2`：L2 正则化 (regularization)，专门针对涉及类别特征的分裂。

### 其他操作参数 (parameter)

- `metric`：指定训练期间要计算的评估指标。例如 `l1`、`l2` (MSE)、`rmse`、`auc`、`binary_logloss`、`multi_logloss`。可以提供多个指标。
- `is_unbalance` (布尔值) 或 `scale_pos_weight` (浮点值)：用于处理不平衡数据集的二分类任务。`is_unbalance=True` 自动调整权重 (weight)，而 `scale_pos_weight` 允许手动设置（通常为 `count(negative_class) / count(positive_class)`）。
- `device_type`：设置为 `cpu` 或 `gpu` 以控制处理单元。GPU 支持需要 LightGBM 的特定构建版本。
- `n_jobs`：控制训练时使用的并行线程数量。`-1` 通常表示使用所有可用核心。
- `seed` (或 `random_state`)：设置随机种子，用于算法中采样及其他随机过程的可复现性。

### 配置示例

这里有一个代码片段，演示了如何使用几个常用调整参数 (parameter)实例化 `LGBMClassifier`：

```python
import lightgbm as lgb

# 二分类任务的配置示例
lgbm_clf = lgb.LGBMClassifier(
    objective='binary',          # 二分类对数损失
    metric='auc',                # 使用 AUC 进行评估
    boosting_type='gbdt',        # 标准梯度提升
    num_leaves=31,               # 中等复杂度树
    learning_rate=0.05,          # 相对较小的学习率
    n_estimators=1000,           # 目标树的数量（与早停机制配合使用）
    max_depth=-1,                # 没有显式深度限制（依赖 num_leaves）
    min_child_samples=20,        # 正则化：叶节点中的最小数据量
    subsample=0.8,               # Bagging：每次迭代使用 80% 的数据
    colsample_bytree=0.7,        # 特征比例：每棵树使用 70% 的特征
    reg_alpha=0.1,               # L1 正则化
    reg_lambda=0.1,              # L2 正则化
    n_jobs=-1,                   # 使用所有可用 CPU 核心
    random_state=42,             # 用于可复现性的种子
    # 如果适用，指定类别特征（例如，使用列索引）
    # categorical_feature=[0, 3, 5],
    # 对于大型数据集，考虑使用 'goss' 或调整 'max_bin'
    # max_bin=127
)

# 假设 X_train, y_train 是您的训练数据
# 使用 fit 方法和早停机制来寻找最佳 n_estimators
# lgbm_clf.fit(X_train, y_train,
#              eval_set=[(X_val, y_val)],
#              eval_metric='auc',
#              callbacks=[lgb.early_stopping(100)])
```

理解这些参数及它们彼此的影响对于有效使用 LightGBM 非常重要。尽管默认设置通常能提供一个坚实的基础，但要针对特定的机器学习 (machine learning)问题获得最佳性能，需要根据本章讨论的原理和后续章节（第 8 章：超参数 (hyperparameter)优化）介绍的技术进行细致调整。实验和交叉验证是找到适合您数据集和目标的最佳配置不可或缺的步骤。

## 参考资料

- [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html) — Microsoft and LightGBM Contributors (2024)
  官方文档，详细介绍了所有LightGBM参数及其配置。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125969/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用指南，包含在更广泛的机器学习工作流中使用LightGBM的示例和解释，常展示其与Scikit-learn兼容的API。

---

[上一节](06-%E4%BC%98%E5%8C%96%E7%9A%84%E7%B1%BB%E5%88%AB%E7%89%B9%E5%BE%81%E5%A4%84%E7%90%86.md) · [下一节](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%20LightGBM.md)

---
course: "mastering-gradient-boosting-algorithms"
chapter: "catboost-gradient-boosting"
lesson: "catboost-api-parameters"
sourceId: 2019
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-6-catboost-gradient-boosting/catboost-api-parameters"
title: "CatBoost API: 参数与配置"
description: "CatBoost库中重要参数指南，特别是针对分类数据。"
order: 7
plots: []
sourceHash: "ff798f224b397c72479d6cc9b00d8a4096dc7c27e4ae88fb7f6866e6c86905aa"
sourceCorrections: []
---

CatBoost采用了有序目标统计和无偏树等创新方法。为了有效配置CatBoost模型，需要通过其API控制其行为。了解可用的参数 (parameter)非常重要，尤其是在处理富含分类信息或需要性能优化的数据集时。

主要的接口是`catboost` Python库中的`CatBoostClassifier`和`CatBoostRegressor`类。虽然许多参数与其他梯度提升实现重叠，但CatBoost提供了为其算法专门设计的独特选项。

### 核心训练参数 (parameter)

这些参数控制提升过程的基本方面，类似于Scikit-learn、XGBoost和LightGBM等库中的参数。

- **`iterations`** (或 `n_estimators`): 指定要构建的最大提升轮次或树的数量。这是影响模型复杂度和训练时间最重要的参数之一。其最优值通常使用提前停止来确定。默认值: 1000。
- **`learning_rate`**: 控制每次迭代的步长，减小每棵新树的贡献。较小的值（例如0.01到0.1）通常需要更多的`iterations`，但通过防止过拟合 (overfitting)可以带来更好的泛化能力。默认值: 通常根据数据集大小和迭代次数自动检测，但通常在0.03左右。
- **`depth`**: 定义基本学习器（树）的深度。CatBoost使用*无偏树*，这意味着相同的分裂标准（特征和阈值）应用于所有同级别节点。这种结构有助于加快预测时间并带来一些固有的正则化 (regularization)。典型值范围为4到10。默认值: 6。
- **`l2_leaf_reg`**: 指定叶子值L2正则化项的系数。此惩罚会抑制叶子中过大的权重 (weight)，有助于防止过拟合。对应于XGBoost目标函数中的`lambda`。默认值: 3.0。
- **`loss_function`**: 确定训练期间要优化的目标函数。常见选择包括:
  - 回归: `'RMSE'` (均方根误差 - 默认), `'MAE'` (平均绝对误差), `'Quantile'` (分位数), `'LogLinQuantile'` (对数线性分位数), `'Poisson'` (泊松), `'MAPE'` (平均绝对百分比误差)。
  - 分类: `'Logloss'` (二分类 - 默认), `'MultiClass'` (多分类), `'CrossEntropy'` (交叉熵)。
  - 也可以实现自定义目标函数。
- **`eval_metric`**: 指定训练期间用于评估模型性能的指标，特别是用于提前停止。例子包括`'RMSE'`、`'MAE'`、`'Logloss'`、`'AUC'`、`'Accuracy'`、`'F1'`、`'Precision'`、`'Recall'`、`'MultiClass'`、`'NDCG'`、`'MAP'`。默认值通常与`loss_function`匹配。
- **`random_seed`** (或 `random_state`): 设置随机数生成的种子，确保数据洗牌、特征选择（如果适用）和自助采样等操作的可复现性。

### 分类特征参数 (parameter)

这些是CatBoost的特色参数，使其能够专门处理分类数据。

- **`cat_features`**: 这可以说是发挥CatBoost优势最重要的参数。它接受应被视为分类的列的索引或名称列表。**非常重要的一点是，如果您在此处列出这些特征，则不应自行预处理它们（例如，通过独热编码）。** CatBoost将其内部方法（例如有序目标统计）应用于这些指定的特征。如果未指定（`None`），CatBoost可能会尝试自动检测分类特征，但为了清晰和控制，建议显式提供它们。默认值: `None`。
- **`one_hot_max_size`**: 设置分类特征中唯一值的数量阈值。如果特征的基数小于或等于此值，CatBoost将应用独热编码，而不是其默认的基于目标的统计。对于低基数特征，这有时会更快，但可能无法像有序目标统计那样有效捕获复杂关系。将其设置为较小的值（例如2或10）会强制CatBoost的专用方法处理大多数分类特征。默认值: 2。
- **`max_ctr_complexity`**: 控制在生成特征组合时可以同时组合的分类特征的最大数量。增加此值可以实现更高阶的交互，但会增加计算成本和内存使用。默认值: 4。
- **`has_time`**: 如果输入数据具有时间顺序（例如时间序列），则设置为`True`。这会影响有序目标统计的计算方式，以避免前瞻偏差。默认值: `False`。
- **`simple_ctr`, `combinations_ctr`**: 这些参数允许对分类特征及其组合计算的计数器统计（CTR）的特定类型进行细粒度控制。修改这些参数通常只针对旨在优化特定情况的高级用户。

### 性能与效率参数 (parameter)

这些参数有助于管理计算资源和训练速度。

- **`task_type`**: 指定训练的硬件。设置为`'GPU'`以使用NVIDIA GPU（如果库安装了GPU支持），这可以显著加速大型数据集上的训练。否则，使用`'CPU'`。默认值: `'CPU'`。
- **`devices`**: 如果`task_type='GPU'`，此参数指定要使用的GPU设备（例如，`'0'`表示第一个GPU，`'0:1'`表示前两个）。
- **`thread_count`**: 控制当`task_type='CPU'`时用于计算的CPU线程数。将其设置为-1通常会使用所有可用核心。默认值: -1。
- **`border_count`**: 确定在构建树之前用于离散化数值特征的分裂（bin）数量。较高的值允许更精确的分裂，但会增加直方图构建阶段的内存使用和计算时间。值通常在32到255之间。默认值: 254 (CPU), 128 (GPU)。
- **`leaf_estimation_method`**: 用于计算树叶子中值的方法。`'Newton'`使用二阶导数（收敛速度更快），而`'Gradient'`使用一阶导数（对于某些目标可能更稳定）。默认值: `'Newton'`。

### 正则化 (regularization)与训练控制参数 (parameter)

这些参数提供了额外的机制来防止过拟合 (overfitting)和管理训练过程。

- **`early_stopping_rounds`**: 激活提前停止。如果在指定的验证集（`eval_set`）上的`eval_metric`连续此轮次未改善，训练将停止。这有助于自动找到最佳`iterations`数量并防止过拟合。需要`fit`调用期间提供`eval_set`。默认值: `None` (禁用)。
- **`use_best_model`**: 如果为`True`（且`early_stopping_rounds`处于激活状态），则在训练结束后，将恢复与验证集上最佳分数对应的模型状态。如果为`False`，则保留最终迭代的模型。默认值: 当提供`eval_set`时为`True`，否则为`False`。
- **`subsample`**: 控制为构建每棵树而采样的训练数据比例（行子采样）。小于1.0的值会引入随机性并起到正则化作用，类似于随机梯度提升。默认值: 1.0 (CPU), 0.8 (GPU)。
- **`bootstrap_type`**: 定义观测权重 (weight)采样的方法。
  - `'Bayesian'` (贝叶斯): 默认。使用贝叶斯自助采样，分配从指数分布中抽取的随机权重。这与有序提升机制有内在联系。
  - `'Bernoulli'` (伯努利): 有放回地采样观测值（标准子采样）。当`subsample` < 1.0时使用。
  - `'MVS'` (最小方差采样): 一种更高级的采样技术。
  - `'No'` (无): 不采样。
- **`colsample_bylevel`**: 控制在搜索最佳分裂时，在每个树级别随机选择的特征比例。由于无偏树的特点，在CatBoost中此参数的调整不如在XGBoost/LightGBM中常见。默认值: 1.0。

### 实际配置示例

下面是如何实例化和配置`CatBoostClassifier`，其中包含一些常用参数 (parameter)，包括指定分类特征和设置提前停止：

```python
import pandas as pd
from catboost import CatBoostClassifier, Pool

# 示例数据（请替换为您的实际数据）
train_data = pd.DataFrame({
    'num_feature1': [1.2, 3.4, 0.5, 2.1, 4.5, 1.8],
    'num_feature2': [5, 2, 8, 6, 3, 7],
    'cat_feature1': ['A', 'B', 'A', 'C', 'B', 'A'],
    'cat_feature2': ['X', 'Y', 'Y', 'X', 'X', 'Y'],
    'target': [1, 0, 1, 0, 1, 0]
})
eval_data = pd.DataFrame({
    'num_feature1': [2.5, 0.8, 3.1],
    'num_feature2': [4, 9, 1],
    'cat_feature1': ['B', 'A', 'C'],
    'cat_feature2': ['Y', 'X', 'Y'],
    'target': [0, 1, 0]
})

# 识别分类特征的索引或名称
categorical_features_indices = [2, 3] # 'cat_feature1', 'cat_feature2' 的索引
# 或按名称: categorical_features_names = ['cat_feature1', 'cat_feature2']

# 使用 CatBoost Pool 准备数据，以提高效率并明确声明特征
train_pool = Pool(data=train_data.drop('target', axis=1),
                  label=train_data['target'],
                  cat_features=categorical_features_indices)

eval_pool = Pool(data=eval_data.drop('target', axis=1),
                 label=eval_data['target'],
                 cat_features=categorical_features_indices)

# 配置模型
model = CatBoostClassifier(
    iterations=1000,                 # 最大树数量
    learning_rate=0.05,              # 步长
    depth=6,                         # 树深度（无偏树）
    l2_leaf_reg=3,                   # L2正则化
    loss_function='Logloss',         # 二分类目标函数
    eval_metric='AUC',               # 用于评估和提前停止的指标
    cat_features=categorical_features_indices, # 显式传递分类特征
    early_stopping_rounds=50,        # 如果AUC连续50轮未改善则停止
    random_seed=42,                  # 用于可复现性
    verbose=100,                     # 每100次迭代打印评估指标
    # task_type='GPU',               # 取消注释以在可用时使用GPU
    # devices='0'                    # 如果使用GPU，请指定GPU设备
)

# 训练模型
model.fit(train_pool,
          eval_set=eval_pool,
          # verbose=False, # 如果需要，可以抑制迭代输出
          plot=False       # 设置为 True 以在 Jupyter 中可视化训练过程
         )

# 进行预测
# preds_proba = model.predict_proba(eval_pool)
# preds_class = model.predict(eval_pool)

print(f"获得的最佳分数: {model.get_best_score()['validation']['AUC']:.4f}")
print(f"最佳迭代次数: {model.get_best_iteration()}")
```

> `CatBoostClassifier` 的示例配置，强调了`iterations`、`learning_rate`、`depth`、`l2_leaf_reg`、`loss_function`、`eval_metric`、`cat_features`和`early_stopping_rounds`等重要参数。建议使用`Pool`对象以获得最佳性能和明确的特征指定。

### 配置总结

使用CatBoost API时：

1. **始终指定`cat_features`**: 这是启用CatBoost专门处理的基础。确保这些特征以其原始分类格式传递。
2. **使用`early_stopping_rounds`**: 这是找到合适`iterations`数量并防止过拟合 (overfitting)的标准方法，需要一个`eval_set`。
3. **调整核心参数 (parameter)**: `learning_rate`、`depth`和`l2_leaf_reg`通常是在设置提前停止后首先要调整的参数。
4. **考虑`task_type='GPU'`**: 如果您有兼容的硬件和大型数据集，GPU训练可以提供显著的速度优势。
5. **尝试分类处理**: 尽管默认值通常效果不错，但调整`one_hot_max_size`或检查CTR配置（`max_ctr_complexity`等）在特定情况下可能会带来改进，特别是对于高基数特征或复杂交互。

熟练掌握这些参数可以帮助您微调 (fine-tuning)CatBoost模型，利用它们的强大功能实现出色性能，尤其是在以分类变量为主的数据集上。接下来的章节将在此部分之上进一步展开，包括超参数 (hyperparameter)优化策略和高级应用。

## 参考资料

- [CatBoost Parameters](https://catboost.ai/docs/concepts/algorithm-parameters.html) — CatBoost Development Team (2024)
  官方文档，提供所有CatBoost API参数和配置选项的详细解释及使用示例。
- [CatBoost: Unbiased Boosting with Categorical Features](https://proceedings.neurips.cc/paper_files/paper/2018/file/6f4922f45568161a8cdf4ad2299f6d23-Paper.pdf) — Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, Andrey Gulin (2018)
  Journal: Advances in Neural Information Processing Systems; Publisher: Neural Information Processing Systems Foundation, Inc. (NeurIPS); Volume: 31; Pages: 6639-6649; DOI: [10.55917/cb_catboost](https://doi.org/10.55917/cb_catboost)
  介绍CatBoost算法的原始研究论文，详细阐述了有序目标统计、遗忘树和有序提升，这些是其专用参数的基础。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer
  对包括梯度提升在内的集成方法进行统计和理论论述，有助于理解许多核心训练参数的背景。

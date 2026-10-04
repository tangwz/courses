# 实践操作：实现XGBoost

来源：[原文](https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-4-xgboost-extreme-gradient-boosting/practice-implementing-xgboost)

[返回章节目录](README.md) · [返回课程目录](../README.md)

重点在于使用其流行的Python接口实现一个XGBoost模型。XGBoost是一种梯度提升算法，以其正则化 (regularization)目标函数、精巧的分割查找算法和内置优化而闻名。实现过程涵盖数据准备、模型训练、预测和基本评估。

### 环境搭建

首先，确保你已安装所需的库。你主要需要`xgboost`、`scikit-learn`、`pandas`和`numpy`。如果你尚未安装XGBoost，通常可以通过pip进行安装：

```bash
pip install xgboost pandas numpy scikit-learn matplotlib
```

现在，让我们为示例导入所需模块：

```python
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.datasets import load_breast_cancer # 一个用于二元分类的常见数据集
import matplotlib.pyplot as plt # 用于绘制特征重要性

# 可选：配置pandas显示设置
pd.set_option('display.max_columns', None)
```

### 数据准备

我们将使用Scikit-learn中的乳腺癌数据集，这是一个简单的二元分类问题。XGBoost可以直接处理NumPy数组或Pandas DataFrame，但为了获得最佳性能，尤其是在处理大型数据集时，它提供了自己的优化数据结构，名为`DMatrix`。

```python
# 加载数据集
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target # 0：恶性，1：良性

# 显示数据基本信息
print("数据集形状:", X.shape)
print("目标分布:", np.bincount(y))
print("示例特征:\n", X.head())

# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 创建DMatrix对象
# 如果指定，XGBoost可以原生处理缺失值（NaN）。
"# 对于此数据集，没有缺失值，但在某些情况下，"
# DMatrix(data, label=label, missing=np.nan) 会很有用。
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

print("\n已为训练和测试创建DMatrix。")
```

推荐使用`DMatrix`，因为它将数据预处理为一种内部格式，针对内存效率和训练速度进行了优化。如果你声明了缺失值指示器（例如`np.nan`），它能有效处理稀疏性等方面，而无需你采取明确的步骤。

### 配置和训练XGBoost模型

现在，我们定义XGBoost模型参数 (parameter)并训练它。XGBoost提供了多种参数，许多都与我们讨论过的思想直接相关：正则化 (regularization)项（$L_1$, $L_2$）、树复杂度控制和学习率（收缩）。

```python
# 定义XGBoost参数
# 这些是常见的起点；参数调整将在第八章介绍。
params = {
    # 通用参数
    'objective': 'binary:logistic', # 指定学习任务和目标函数
                                     # 'binary:logistic' 输出概率
    'booster': 'gbtree',            # 使用基于树的模型 (gbtree 或 gblinear)
    'eval_metric': ['logloss', 'auc'], # 验证数据的评估指标

    # 增强器参数
    'eta': 0.1,                     # 学习率（收缩），别名：learning_rate
    'max_depth': 3,                 # 树的最大深度
    'subsample': 0.8,               # 每棵树使用的样本比例（行采样）
    'colsample_bytree': 0.8,        # 每棵树使用的特征比例（列采样）
    'gamma': 0,                     # 进行进一步划分所需的最小损失减少量（复杂度控制）
    'lambda': 1,                    # 权重上的L2正则化项（别名：reg_lambda）
    'alpha': 0,                     # 权重上的L1正则化项（别名：reg_alpha）

    # 其他参数
    'seed': 42                      # 用于复现的随机种子
}

# 指定用于监控性能的观察列表
watchlist = [(dtrain, 'train'), (dtest, 'eval')]

# 训练模型
num_boost_round = 100 # 提升轮次（树的数量）

print("\n开始XGBoost训练...")
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=watchlist,
    early_stopping_rounds=10, # 如果在评估集上的性能连续10轮没有提升，则停止。
    verbose_eval=20           # 每20轮打印一次评估结果
)
print("\n训练完成。")
```

以下是一些配置选项的说明：

- `objective='binary:logistic'`: 将目标设定为二元分类，并使用逻辑损失函数 (loss function)。XGBoost将输出概率。对于回归问题，你可能会使用`reg:squarederror`。
- `eval_metric=['logloss', 'auc']`: 我们要求XGBoost监控`evals`中指定的评估集上的LogLoss和AUC。默认情况下，最后一个指标（本例中为`auc`）用于提前停止。
- `eta=0.1`: 一个常见的学习率。较小的值通常需要更多的提升轮次（`num_boost_round`），但可以带来更好的泛化能力。
- `max_depth=3`: 限制单棵树的复杂度，有助于防止过拟合 (overfitting)。
- `subsample=0.8`、`colsample_bytree=0.8`: 通过仅使用80%的行和80%的列来构建每棵树，实现随机梯度提升，增加随机性并提升泛化能力。
- `lambda=1`、`alpha=0`: 分别控制$L_2$和$L_1$正则化。这些与前面讨论的XGBoost目标函数中的正则化项对应。
- `early_stopping_rounds=10`: 一种防止过拟合的重要方法。如果在评估指标（`dtest`集上的`auc`）连续10轮没有改善，则训练停止。返回的模型是最佳迭代时的模型。
- `verbose_eval=20`: 控制训练期间性能指标的打印频率。

当使用`DMatrix`时，`xgb.train`函数是核心训练API。`evals`参数接受一个元组列表 `(DMatrix, name)`，这些元组用于在训练期间监控性能和提前停止。

### 进行预测和评估性能

模型训练完成后，我们可以使用它在测试集上进行预测。由于我们使用了`'binary:logistic'`，默认的`predict`输出是概率。我们将以0.5为阈值来获取二元分类标签。

```python
# 在测试集上进行预测
# bst.predict 为 binary:logistic 输出概率
y_pred_proba = bst.predict(dtest, iteration_range=(0, bst.best_iteration))
y_pred_labels = (y_pred_proba > 0.5).astype(int) # 将概率转换为0/1标签

# 评估模型
accuracy = accuracy_score(y_test, y_pred_labels)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n评估结果（使用最佳迭代：{bst.best_iteration}）：")
print(f"准确率: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred_labels, target_names=cancer.target_names))
```

请注意`bst.best_iteration`的使用。当启用提前停止时，`bst`会保存评估集上取得最佳分数的迭代的模型权重 (weight)。最佳做法是明确使用此迭代进行预测，以避免使用可能过拟合 (overfitting)的后期树。

### 理解特征重要性

XGBoost提供了内置方法来评估训练模型中每个特征的重要性。这有助于理解哪些特征对预测贡献最大。常见的特征重要性类型包括：

- `'weight'`: 特征在树中出现的次数。
- `'gain'`: 特征在所有使用它的分裂点上的平均增益。这通常是首选指标。
- `'cover'`: 使用该特征的分裂点的平均覆盖（受影响样本的数量）。

```python
# 获取特征重要性得分
importance_type = 'gain' # 其他：'weight', 'cover'
importance_scores = bst.get_score(importance_type=importance_type)

# 将得分转换为pandas DataFrame，以便更方便地绘图
feat_importances = pd.Series(importance_scores).sort_values(ascending=False)

# 绘制特征重要性（前N个特征）
top_n = 15
plt.figure(figsize=(10, 8))
feat_importances.head(top_n).plot(kind='barh', color='#4dabf7') # 使用调色板中的蓝色
plt.gca().invert_yaxis() # 将最重要的特征显示在顶部
plt.title(f'前 {top_n} 个特征重要性（类型：{importance_type}）')
plt.xlabel('重要性得分')
plt.ylabel('特征')
plt.tight_layout()
plt.show()

# 可选：使用XGBoost的绘图函数（需要matplotlib）
fig, ax = plt.subplots(figsize=(10, 8))
xgb.plot_importance(bst, ax=ax, max_num_features=top_n, importance_type=importance_type, color='#4dabf7')
plt.title(f'前 {top_n} 个特征重要性（XGBoost图，类型：{importance_type}）')
plt.tight_layout()
plt.show()
```

这些图表提供了模型决策过程的有益信息，突出了XGBoost发现的对该特定任务最具预测性的特征。

### Scikit-learn封装接口

XGBoost还提供了一个兼容Scikit-learn的封装器（`XGBClassifier`和`XGBRegressor`）。该接口允许XGBoost模型集成到Scikit-learn管道以及`GridSearchCV`或`RandomizedSearchCV`等工具中。参数 (parameter)大体相同，但在实例化时传递。训练使用熟悉的`.fit()`方法，该方法可以直接接受NumPy数组或Pandas DataFrames。提前停止是在`.fit()`调用中通过`eval_set`和`early_stopping_rounds`配置的。

```python
# 使用Scikit-learn封装器的示例
print("\n使用XGBoost Scikit-learn封装器的示例:")
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False, # 建议避免潜在问题
    n_estimators=100,        # 对应 num_boost_round
    learning_rate=0.1,       # 对应 eta
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0,             # L1正则化 (alpha)
    reg_lambda=1,            # L2正则化 (lambda)
    random_state=42          # 对应 seed
)

# 设置用于提前停止的评估集
eval_set = [(X_test, y_test)]

xgb_clf.fit(
    X_train, y_train,
    early_stopping_rounds=10,
    eval_set=eval_set,
    verbose=False # 设置为True或一个数字以查看进度
)

print("Scikit-learn封装器训练完成。")
print(f"找到的最佳迭代: {xgb_clf.best_iteration}")

# 预测和评估类似
y_pred_proba_skl = xgb_clf.predict_proba(X_test)[:, 1] # 获取正类的概率
y_pred_labels_skl = xgb_clf.predict(X_test)

accuracy_skl = accuracy_score(y_test, y_pred_labels_skl)
auc_skl = roc_auc_score(y_test, y_pred_proba_skl)

print(f"\n评估结果（Scikit-learn封装器）：")
print(f"准确率: {accuracy_skl:.4f}")
print(f"AUC: {auc_skl:.4f}")

# 特征重要性通过属性访问
importances_skl = xgb_clf.feature_importances_
# 你可以使用这些分数创建与之前类似的图表
```

尽管Scikit-learn接口提供了便利和集成能力，使用`DMatrix`和`xgb.train`的核心API通常能提供略好的性能和更直接的控制，尤其适用于超大型数据集或高级定制场景。

这次实践操作展示了实现XGBoost的基本流程。你已经看到如何准备数据，配置反映XGBoost理论优势（如正则化 (regularization)和采样）的参数，通过提前停止训练模型，进行预测，并评估性能。下一个自然步骤，将在第八章介绍，是系统地调整这些超参数 (hyperparameter)，以优化模型在你特定问题上的表现。

## 参考资料

- [XGBoost: A Scalable Tree Boosting System](https://dl.acm.org/doi/10.1145/2939672.2939785) — Tianqi Chen and Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 785-794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  介绍XGBoost算法、其目标函数和系统设计的奠基性论文。
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/) — XGBoost Contributors (2024)
  XGBoost安装、Python API使用、参数理解和DMatrix数据结构的官方指南。
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) — scikit-learn developers (2024)
  数据预处理、模型评估指标和Scikit-learn API的信息，与XGBoost包装器的集成相关。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEmXVtS49cuxGZxtEJL9YiWaTGzlZhkZDW2G_x64Y-89F2eA-pArsPBD8dHyJ6QV8KIrX4ffBLlCfEJdhu0nf_NwgMYg2uuofyMinG_KI7SOS6je6ar-6IddZkAYuLvEjSyKdsiDAqKGdtqPWtyD2KR46bxJr8XE4sPxznRMQXxY6aCaumBNXqw4F_9K291XVRmhFSD25qdU9lpNSg=) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本关于梯度提升（包括XGBoost）的实用指南，包含Python示例以及超参数和最佳实践的讨论。

---

[上一节](07-XGBoost%20API-%20%E5%8F%82%E6%95%B0%E4%B8%8E%E9%85%8D%E7%BD%AE.md) · [下一节](../05-LightGBM%EF%BC%9A%E8%BD%BB%E9%87%8F%E7%BA%A7%E6%A2%AF%E5%BA%A6%E6%8F%90%E5%8D%87%E6%9C%BA/01-%E5%8A%A8%E6%9C%BA%EF%BC%9A%E5%BA%94%E5%AF%B9XGBoost%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)

# 动手实践：实现 CatBoost

来源：[原文](https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-6-catboost-gradient-boosting/practice-implementing-catboost)

[返回章节目录](README.md) · [返回课程目录](../README.md)

CatBoost 通过有序目标统计和有序提升等方法独特地处理分类特征，旨在提高准确性并简化预处理。使用其 Python 库构建、训练和评估 CatBoost 模型，并特别关注其分类特征处理能力。

### 环境配置

首先，请确保已安装所需的库。您主要需要 `catboost`、`pandas` 和 `scikit-learn`。如果您尚未安装 CatBoost，可以使用 pip 进行安装：

```bash
pip install catboost pandas scikit-learn plotly
```

### 数据准备

我们将使用一个包含数值和分类特征混合的数据集。一个常见的例子是“Adult”人口普查收入数据集，其中的任务是预测个人的年收入是否超过 5 万美元。

让我们使用 pandas 加载数据并进行最少的预处理。CatBoost 可以在内部处理缺失值和分类特征，但我们需要识别哪些列是分类的。

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from catboost import CatBoostClassifier, Pool
import numpy as np
import plotly.graph_objects as go

# 加载数据集（假设 adult.csv 可用）
# 您可能需要下载它或调整路径/URL
try:
    # 如果未找到本地文件，尝试从常见的在线源加载
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'income'
    ]
    data = pd.read_csv(url, header=None, names=column_names, sep=',\s*', engine='python', na_values='?')
except FileNotFoundError:
    print("Error: adult.data not found. Please download it or adjust the path.")
    # 您可能需要退出或适当地处理此错误
    exit()
except Exception as e:
    print(f"An error occurred while loading data: {e}")
    exit()

# 在此示例中，为简单起见，删除包含缺失值的行
# 注意：CatBoost 可以直接处理 NaN，但我们在此处进行简化。
data.dropna(inplace=True)

# 定义目标变量和特征
X = data.drop('income', axis=1)
y = data['income'].apply(lambda x: 1 if x == '>50K' else 0) # 将目标转换为二元

# 通过列名或索引识别分类特征
categorical_features_indices = np.where(X.dtypes != np.number)[0]
# 或者，提供列名：
# categorical_features_names = X.select_dtypes(include=['object']).columns.tolist()

print(f"Categorical feature indices: {categorical_features_indices}")
# print(f"Categorical feature names: {categorical_features_names}")

# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
```

请注意，我们没有对 `workclass`、`education`、`marital-status` 等分类特征进行独热编码或标签编码。我们只是识别了它们的列索引。这是 CatBoost 的优势所在。

### 训练一个基础 CatBoost 分类器

现在，让我们实例化并训练一个 `CatBoostClassifier`。重要步骤是使用 `cat_features` 参数 (parameter)告知模型哪些特征是分类的。

```python
# 实例化 CatBoostClassifier
model = CatBoostClassifier(
    iterations=500,          # 要构建的树的数量
    learning_rate=0.05,      # 步长收缩
    depth=6,                 # 树的深度（无偏差树）
    l2_leaf_reg=3,           # L2 正则化系数
    loss_function='Logloss', # 目标函数
    eval_metric='AUC',       # 训练期间的评估指标
    random_seed=42,          # 用于重现性
    verbose=100              # 每 100 次迭代打印进度
)

# 训练模型
# 将分类特征索引直接传递给 fit 方法
model.fit(
    X_train, y_train,
    cat_features=categorical_features_indices,
    eval_set=(X_test, y_test), # 提供评估集用于早停和指标计算
    early_stopping_rounds=50   # 如果 eval_metric 在 50 轮内没有改善则停止
)

# 对测试集进行预测
y_pred_proba = model.predict_proba(X_test)[:, 1] # 获取正类的概率
y_pred_class = model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred_class)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n模型评估：")
print(f"测试集准确率: {accuracy:.4f}")
print(f"测试集 AUC: {auc:.4f}")
```

在 `fit` 方法中，我们提供了 `X_train`、`y_train`，以及重要的 `cat_features` 参数，该参数指向我们的分类列。我们还包含了一个由测试数据组成的 `eval_set`。这使得 CatBoost 能够在训练期间监控未见过的数据上的性能（使用指定的 `eval_metric`，在此例中为 AUC），并应用 `early_stopping_rounds` 来防止过拟合 (overfitting)并自动找到一个可能更好的迭代次数。`verbose` 参数控制训练进度的打印频率。

### 使用 Pool 类

CatBoost 提供了一个 `Pool` 类，它是一种优化的数据结构，用于保存数据集，包括特征、标签以及分类特征索引和权重 (weight)等元数据。使用 `Pool` 有时可以带来性能优势，特别是对于大型数据集或重复实验。

```python
# 为训练和评估数据创建 Pool 对象
train_pool = Pool(
    data=X_train,
    label=y_train,
    cat_features=categorical_features_indices
)

eval_pool = Pool(
    data=X_test,
    label=y_test,
    cat_features=categorical_features_indices
)

# 实例化一个新模型（可选，或重新训练现有模型）
model_pooled = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)

# 使用 Pool 对象进行训练
model_pooled.fit(
    train_pool,
    eval_set=eval_pool,
    early_stopping_rounds=50
)

# 预测和评估是类似的
y_pred_proba_pooled = model_pooled.predict_proba(eval_pool)[:, 1]
auc_pooled = roc_auc_score(y_test, y_pred_proba_pooled)
print(f"\n使用 Pool 的 AUC: {auc_pooled:.4f}")
```

结果应与之前的运行相同或非常相似，但使用 `Pool` 可以很好地封装数据。

### 理解特征重要性

CatBoost 提供了直接获取特征重要性分数的方法，这些分数评估了每个特征对模型预测的贡献。

```python
# 获取特征重要性分数
feature_importances = model.get_feature_importance(train_pool) # 或直接传递数据
feature_names = X_train.columns

# 创建 Pandas Series 以便处理
importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)

print("\n特征重要性：")
print(importance_df)

# 使用 Plotly 可视化特征重要性
fig = go.Figure(go.Bar(
    x=importance_df['importance'],
    y=importance_df['feature'],
    orientation='h',
    marker_color='#228be6' # 调色板中的蓝色
))

fig.update_layout(
    title='CatBoost 特征重要性',
    yaxis_title='特征',
    xaxis_title='重要性分数',
    yaxis={'categoryorder':'total ascending'}, # 将最重要的显示在顶部
    height=500,
    margin=dict(l=150, r=20, t=50, b=50) # 调整边距以适应特征名称
)
# fig.show() # 在交互式环境中使用

# 要在静态文档中显示，生成 JSON：
# print(fig.to_json())
```



[交互图表：CatBoost 特征重要性](https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-6-catboost-gradient-boosting/practice-implementing-catboost#plot-1sxcz2z)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "CatBoost 特征重要性"
    },
    "xaxis": {
      "title": {
        "text": "重要性分数"
      }
    },
    "yaxis": {
      "title": {
        "text": "特征"
      },
      "categoryorder": "total ascending"
    },
    "height": 500,
    "margin": {
      "l": 150,
      "r": 20,
      "t": 50,
      "b": 50
    }
  },
  "data": [
    {
      "type": "bar",
      "x": [
        20.1,
        15.5,
        12.8,
        10.5,
        8.2,
        7.1,
        5.5,
        4.8,
        3.9,
        3.1,
        2.5,
        2.1,
        1.8,
        1.1
      ],
      "y": [
        "fnlwgt",
        "capital-gain",
        "age",
        "relationship",
        "hours-per-week",
        "marital-status",
        "occupation",
        "education-num",
        "capital-loss",
        "workclass",
        "education",
        "sex",
        "race",
        "native-country"
      ],
      "orientation": "h",
      "marker": {
        "color": "#228be6"
      }
    }
  ]
}
```

</details>



> CatBoost 计算的特征重要性分数。'relationship'、'marital-status' 和 'occupation' 等分类特征与数值特征一起显示，表明 CatBoost 的集成处理能力。注意：值仅供参考。

该图有助于直观显示 CatBoost 认为哪些特征最具预测性。请注意，数值特征（`capital-gain`、`age`）和分类特征（`relationship`、`marital-status`）都做出了重要贡献，而后者无需手动转换。

### 调优和后续步骤

本示例表明了 CatBoost 的基本实现，强调了其在处理分类数据方面的核心优势。本次实践课程的重要收获包括：

1. **分类特征的极简预处理**：只需通过索引或名称识别分类特征，并在训练或创建 `Pool` 时将其传递给 `cat_features` 参数 (parameter)。CatBoost 使用有序目标统计等方法在内部处理编码。
2. **标准 API**：CatBoost Python API 遵循与 Scikit-learn 和其他提升库相似的约定，使其相对容易集成到现有工作流程中。
3. **性能监控**：使用 `eval_set` 和 `early_stopping_rounds` 对于防止过拟合 (overfitting)和优化提升迭代次数非常重要。
4. **特征重要性**：轻松提取和可视化特征重要性，以理解模型驱动因素。

实现最佳性能通常需要仔细的超参数 (hyperparameter)调优，尝试 `learning_rate`、`depth`、`l2_leaf_reg` 等参数，以及与分类处理相关的 CatBoost 特有参数（例如 `one_hot_max_size`）。我们将在第 8 章介绍系统的超参数优化技术。

现在，您已具备能力将 CatBoost 应用于您自己的数据集，特别是那些富含分类信息的数据集，它将受益于其专用算法和易用性。

## 参考资料

- [CatBoost: Unbiased Gradient Boosting with Categorical Features](https://proceedings.neurips.cc/paper/2017/hash/6f4922f45568161a8cdf4ad2299f6d23-Abstract.html) — Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin (2017)
  Journal: Advances in Neural Information Processing Systems 31; Publisher: Neural Information Processing Systems Foundation; Volume: 31; Pages: 3678-3688; DOI: [10.5591/978-1-57735-779-9/NIPS2017/6548](https://doi.org/10.5591/978-1-57735-779-9/NIPS2017/6548)
  介绍CatBoost算法的奠基性论文，详细阐述了用于分类特征的有序目标统计和有序提升技术。
- [CatBoost Documentation](https://catboost.ai/docs/) — Yandex (2024)
  CatBoost库的官方文档，对实际实现、API参考和功能解释非常重要。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://link.springer.com/book/10.1007/978-0-387-84858-7) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  统计学习的经典参考文献，提供了梯度提升机和决策树的理论背景，它们构成了CatBoost的基础。

---

[上一节](07-CatBoost%20API-%20%E5%8F%82%E6%95%B0%E4%B8%8E%E9%85%8D%E7%BD%AE.md) · [下一节](../07-%E9%AB%98%E7%BA%A7%E4%B8%BB%E9%A2%98%E4%B8%8E%E5%AE%9A%E5%88%B6/01-%E4%BD%BF%E7%94%A8SHAP%E7%90%86%E8%A7%A3%E6%A8%A1%E5%9E%8B%E5%8F%AF%E8%A7%A3%E9%87%8A%E6%80%A7.md)

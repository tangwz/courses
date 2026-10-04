---
course: "advanced-python-programming-ml"
chapter: "custom-ml-estimators-transformers"
lesson: "developing-custom-estimators"
sourceId: 2369
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-6-custom-ml-estimators-transformers/developing-custom-estimators"
title: "开发自定义评估器"
description: "Build custom prediction models (estimators) in Python following the Scikit-learn interface conventions."
order: 3
plots: []
sourceHash: "3ad0776c3cc285880e393e3f19035dbb9f349d3a6766df51f462d4ab5ef804d9"
sourceCorrections: []
---

预测模型本身，即*评估器*，是许多机器学习 (machine learning)任务的核心。尽管专门的数据操作通常由转换器处理，但评估器是主要预测逻辑所在之处。Scikit-learn 提供了一系列丰富的评估器，但有时会出现需要实现自定义建模算法、以独特方式修改现有算法，或封装库中没有的特定预测策略的情况。此处将介绍如何创建自己的评估器，使其与 Scikit-learn 生态系统良好结合。

开发自定义评估器需要遵循与自定义转换器相同的核心 API 原则，确保与 `Pipeline`、`GridSearchCV` 和 `cross_val_score` 等工具的兼容性。

### Scikit-learn 评估器接口

从根本上讲，Scikit-learn 评估器是一个实现特定方法的 Python 类。通过遵循这些约定，您的自定义评估器可以与库的工具实现互操作性。

1. **`__init__(self, **hyperparameters)`**:

   - 构造函数应接受评估器的超参数 (parameter) (hyperparameter)作为显式关键字参数。
   - **非常重要的一点是**，它不应执行任何数据验证或模型拟合。其唯一目的是存储超参数。
   - 每个关键字参数都应有默认值。
   - 将传入的超参数*不变地*存储为公共属性（例如，`self.my_param = my_param`）。这是 `get_params` 和 `set_params` 正确运行所必需的。
   - 可选地，定义 `_estimator_type` 属性（例如，`"classifier"` 或 `"regressor"`）。虽然对于基本功能并非严格要求，但它有助于 Scikit-learn 工具识别评估器类型。
2. **`fit(self, X, y=None, **fit_params)`**:

   - 这是评估器从数据中学习的主要训练方法。
   - 它接受训练数据 `X`（通常是形状为 `[n_samples, n_features]` 的 NumPy 数组或稀疏矩阵）以及可选的目标值 `y`（对于监督学习 (supervised learning)，是形状为 `[n_samples]` 的 NumPy 数组）。
   - `fit` 方法应执行任何必要的输入验证（通常使用 Scikit-learn 的验证工具）。
   - 它包含学习算法的主要逻辑。
   - 学习到的参数（例如，模型系数、聚类中心）**必须**作为名称以尾随下划线 (`_`) 结尾的属性存储，例如 `self.coef_` 或 `self.cluster_centers_`。此约定将学习到的参数与初始化期间设置的超参数区分开来。
   - 该方法**必须**返回 `self`（评估器实例）。
3. **`predict(self, X)`** (适用于监督评估器):

   - 接受新数据 `X`（具有与训练数据相同的特征数量），并返回预测值（例如，类别标签、回归值）。
   - 在进行预测之前，它应该检查 `fit` 是否已被调用（通常通过 `sklearn.utils.validation` 中的 `check_is_fitted`）。
   - 它使用在 `fit` 期间存储的学习到的参数（那些以 `_` 结尾的参数）。
4. **`score(self, X, y)`** (可选，通常由 mixin 提供):

   - 评估模型在给定数据 `X` 和真实标签 `y` 上的性能。
   - 返回一个分数（例如，分类器的准确率，回归器的 R^2）。
5. **`get_params(self, deep=True)`** 和 **`set_params(self, **params)`** (通常继承而来):

   - 这些方法对于超参数调优和模型检查非常重要。
   - `get_params` 获取评估器的超参数。
   - `set_params` 允许修改超参数。
   - 通常通过继承 `sklearn.base.BaseEstimator` 免费获得这些功能。

### 运用基类和混入类

Scikit-learn 在 `sklearn.base` 中提供基类和混入类，以简化自定义评估器开发：

- **`BaseEstimator`**：这是所有评估器的基本类。从 `BaseEstimator` 继承会自动提供符合规范的 `get_params` 和 `set_params` 方法，前提是您的 `__init__` 方法遵循上述约定。
- **`RegressorMixin`**：适用于回归评估器。从该混入类（以及 `BaseEstimator`）继承会提供一个默认的 `score` 方法，用于计算 R^2 分数。您只需要实现 `fit` 和 `predict`。
- **`ClassifierMixin`**：适用于分类评估器。它提供一个默认的 `score` 方法，用于计算平均准确率。您需要实现 `fit` 和 `predict`（通常还包括 `predict_proba`）。
- **`TransformerMixin`**：提供 `fit_transform`。您需要实现 `fit` 和 `transform`。（在自定义转换器的上下文 (context)中会有更多介绍）。
- **`ClusterMixin`**：适用于聚类评估器。提供 `fit_predict`。您需要实现 `fit` 和通常的 `labels_`。

通过继承 `BaseEstimator` 和相关的混入类（例如 `RegressorMixin`），您可以大幅减少样板代码。

### 示例：一个简单均值回归器

让我们构建一个非常基本的回归器，它总是预测训练期间目标变量的平均值。这展现了基本结构。

```python
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

class MeanRegressor(BaseEstimator, RegressorMixin):
    """
    一个简单回归器，预测训练目标变量的均值。
    """
    # 这个简单模型不需要超参数，因此 __init__ 函数非常简洁
    def __init__(self):
        pass # 无超参数需要存储

    def fit(self, X, y):
        """
        学习目标变量 y 的均值。

        参数
        ----------
        X : 形状为 (n_samples, n_features) 的类数组
            训练数据。此评估器会忽略，但 API 要求。
        y : 形状为 (n_samples,) 的类数组
            目标值。

        返回
        -------
        self : 对象
            返回实例本身。
        """
        # 1. 验证输入：检查 X 和 y，如果需要则转换为 NumPy 数组。
        #    确保 y 被视为回归目标。
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)

        # 2. 存储数据属性（可选但推荐）
        #    此评估器忽略 X，因此不需要 n_features_in_，
        #    但这是一种标准做法。
        self.n_features_in_ = X.shape[1]

        # 3. 学习参数：计算 y 的均值
        self.mean_ = np.mean(y)

        # 4. 将评估器标记为已拟合（可选但推荐）
        self.is_fitted_ = True

        # 5. 返回 self
        return self

    def predict(self, X):
        """
        预测 X 中所有样本的学习到的均值。

        参数
        ----------
        X : 形状为 (n_samples, n_features) 的类数组
            待预测的样本。

        返回
        -------
        y_pred : 形状为 (n_samples,) 的 ndarray
            每个样本的预测均值。
        """
        # 1. 检查 fit 是否已被调用
        check_is_fitted(self, 'mean_') # 检查 self.mean_ 是否存在

        # 2. 验证输入 X：确保 X 有效（例如，NumPy 数组）
        #    并且具有 fit 所需的正确特征数量。
        X = check_array(X, accept_sparse=False)
        if X.shape[1] != self.n_features_in_:
             raise ValueError(f"Expected {self.n_features_in_} features but got {X.shape[1]}")

        # 3. 执行预测：返回一个包含存储均值的数组
        #    其长度与 X 中样本的数量相同。
        n_samples = X.shape[0]
        y_pred = np.full(shape=n_samples, fill_value=self.mean_)

        return y_pred

    # 我们从 RegressorMixin 继承 score 方法（计算 R^2）
    # 我们从 BaseEstimator 继承 get_params/set_params

    # 可选，但推荐用于兼容性检查
    def _more_tags(self):
        return {'non_deterministic': False, # 我们的评估器是确定性的
                'requires_y': True}         # Fit 需要 y
```

这个 `MeanRegressor` 显示了主要的组成部分：

- 从 `BaseEstimator` 和 `RegressorMixin` 继承。
- 一个什么也不做的 `__init__`（因为没有超参数 (parameter) (hyperparameter)）。
- 一个 `fit` 方法，它使用 `check_X_y` 验证输入，计算均值，将其存储在 `self.mean_` 中，存储 `n_features_in_`，设置 `is_fitted_`，并返回 `self`。
- 一个 `predict` 方法，它使用 `check_is_fitted` 检查模型是否已拟合，使用 `check_array` 验证输入 `X`，检查特征数量，并根据学习到的 `mean_` 返回预测。

### 输入验证工具函数

Scikit-learn 在 `sklearn.utils.validation` 中提供有用的验证函数：

- `check_array(array, ...)`：检查输入是否为 NumPy 数组或类似类型，必要时进行转换，并执行检查，例如确保非空、有限值、正确的数据类型或特征数量。
- `check_X_y(X, y, ...)`：同时检查 `X` 和 `y`，确保它们具有一致的长度和格式。它在监督评估器的 `fit` 方法中特别有用。您可以为回归指定 `y_numeric=True`，或为分类单独使用 `check_classification_targets(y)`。
- `check_is_fitted(estimator, attributes=None, ...)`：通过验证指定属性（那些以 `_` 结尾的属性）的存在来检查评估器是否已拟合。在 `predict`、`transform` 或 `score` 方法的开始时调用此函数。

使用这些工具函数使您的评估器更符合标准 Scikit-learn 行为。

### 与 Scikit-learn 工具的集成

由于我们的 `MeanRegressor` 遵循 Scikit-learn API（主要通过继承 `BaseEstimator` 和 `RegressorMixin` 并正确实现 `fit`/`predict`），因此它可以与 Scikit-learn 的元评估器和模型评估工具配合使用：

```python
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 生成样本回归数据
X, y = make_regression(n_samples=100, n_features=5, random_state=42)

# 实例化自定义评估器
mean_reg = MeanRegressor()

# 使用交叉验证进行评估
scores = cross_val_score(mean_reg, X, y, cv=5)
print(f"交叉验证 R^2 分数: {scores}")
# 示例输出：交叉验证 R^2 分数: [-0.0015 -0.0032 -0.0079 -0.0006 -0.0064]
# （对于此基线模型，分数接近 0 是预期的）

# 在 Pipeline 中使用它
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('mean_model', MeanRegressor())
])

pipe.fit(X, y)
predictions = pipe.predict(X[:5])
print(f"Pipeline 预测 (前 5 个): {predictions}")
# 示例输出：Pipeline 预测 (前 5 个): [3.61 3.61 3.61 3.61 3.61] (所有预测都是学习到的均值)
```

将自定义逻辑直接集成到标准工作流中而无需修改的能力是遵循 API 约定的一个重要优点。

### 构建更复杂的评估器

“`MeanRegressor` 特意设计得很简单。自定义评估器通常涉及：”

- **超参数 (parameter) (hyperparameter)**：在 `__init__` 中定义并作为属性存储。
- **复杂的 `fit` 逻辑**：实现特定算法（例如，梯度下降 (gradient descent)、自定义树构建、专门的距离度量）。
- **多个学习到的参数**：存储 `fit` 过程的各种结果（例如，`coef_`、`intercept_`、`support_vectors_`）。
- **概率预测**：为分类器实现 `predict_proba(X)` 以返回类别概率。
- **特定验证**：为超参数或输入数据添加领域特定的检查。

在构建更复杂的评估器时，请记住：

1. 从基本框架开始（`BaseEstimator`，适当的混入类）。
2. 明确区分超参数（在 `__init__` 中设置）与学习到的参数（在 `fit` 中设置，以 `_` 结尾）。
3. 恰当地使用验证工具函数（`check_X_y`、`check_array`、`check_is_fitted`）。
4. 确保 `fit` 返回 `self`。
5. 彻底测试，最好使用 `sklearn.utils.estimator_checks.check_estimator`（在测试部分会介绍）。

通过掌握自定义评估器的开发，您可以获得在 Scikit-learn 的可组合框架内封装独特建模方法的能力，从而实现更精细和定制化的机器学习 (machine learning)方案。

## 参考资料

- [Developing your own scikit-learn estimators](https://scikit-learn.org/stable/developers/develop.html) — scikit-learn developers (2024)
  创建与Scikit-learn API兼容的自定义估计器的官方用户指南。它详细说明了所需的方法和约定。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFb1-g3-V6er2GjQXEIudCSGux_zfhJBbqNZ-65VoAbk3nfkcjRhisSC8LGcBNLgDsjl54Gvk3d79pWiip9iUnXi6rd2awELFSVkAprlmQwP6wZHHc_2XH-fhUs3zstexirYK4vkHdTC4PHcg6HrKS3RWTXIrY_hA1JAAGHesNgFG8OjRYQrRk5PVfCVxlwSPoEG4KiWWNz66nJivX2Y9Xux8lTYmFLLQEu_eZQzPllCt0XlYsNuVHEyI8Har_JrUQ==) — Aurélien Géron (2019)
  Publisher: O'Reilly Media
  一本广泛使用的实践书籍，其中包含使用自定义组件（包括估计器）扩展Scikit-learn的实践示例和解释。
- [Scikit-learn: Machine Learning in Python](http://jmlr.org/papers/v12/pedregosa11a.html) — Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, Édouard Duchesnay (2011)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 12; Pages: 2825-2830
  介绍Scikit-learn库的基础论文，提供了对其设计原则和架构的了解，有助于理解自定义估计器的API约定。

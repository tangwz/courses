---
course: "advanced-python-programming-ml"
chapter: "custom-ml-estimators-transformers"
lesson: "scikit-learn-estimator-interface"
sourceId: 2363
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-6-custom-ml-estimators-transformers/scikit-learn-estimator-interface"
title: "Scikit-learn API 和估计器接口"
description: "了解 Scikit-learn API 契约和设计原则，用于创建兼容组件。"
order: 1
plots: []
sourceHash: "3dde1685b14996ba7b9ff47fcd8d27e2685fc783133fe3dee17a9c4cc4143a77"
sourceCorrections: []
---

为有效扩展 Scikit-learn，您的自定义组件必须与其现有基础设施流畅地集成，特别是管道（pipelines）、网格搜索（grid search）和交叉验证工具。这种集成取决于遵循一套一致的设计原则和接口，通常被称为 Scikit-learn API。了解这个 API 是构建强大、可重用自定义估计器和转换器的第一步。

其核心是，Scikit-learn 处理被称为 **估计器** 的对象。估计器是任何能够从数据中学习的对象。这种学习通过其 `fit(X, y=None)` 方法发生。`X` 通常代表输入特征（通常是形状为 `[n_samples, n_features]` 的 NumPy 数组或稀疏矩阵），而 `y` 代表监督学习 (supervised learning)任务中的目标值（形状为 `[n_samples]` 的 NumPy 数组）。对于无监督学习 (unsupervised learning)，`y` 通常被省略或忽略。

### 核心估计器接口

每个 Scikit-learn 估计器都应遵循以下基本约定：

1. **通过 `__init__` 进行初始化**：估计器的所有参数 (parameter)都应直接作为公共属性访问，并在对象的 `__init__` 方法中设置。这些参数控制估计器的行为（例如，模型中的正则化 (regularization)强度，分解中的组件数量）。重要的是，`__init__` *不应* 执行任何实际的学习或数据验证；其唯一目的是存储传递给它的参数。避免将数据（`X` 或 `y`）传递给构造函数。
2. **学习到的属性**：在 `fit` 方法中从数据中学习到的属性必须以一个下划线 (`_`) 结尾。例子包括线性模型中的 `coef_` 或 PCA 中的 `components_`。此约定将初始化期间设置的超参数 (hyperparameter)与拟合期间学习到的参数区分开来。
3. **`fit` 方法**：这是核心学习方法。它将训练数据 `X`（以及监督估计器的 `y`）作为输入。其主要职责是估计并存储学习到的参数（以 `_` 结尾的属性）。一个重点是，`fit` 方法应始终返回估计器实例本身（`self`）。这允许方法链式调用，例如 `estimator.fit(X_train).predict(X_test)`。

```python
# Scikit-learn 兼容估计器的基本结构
from sklearn.base import BaseEstimator, ClassifierMixin # 分类器示例

class MyCustomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hyperparameter1=1.0, hyperparameter2='default'):
        # 存储实例化时传入的超参数
        self.hyperparameter1 = hyperparameter1
        self.hyperparameter2 = hyperparameter2
        # 在此处不进行数据处理或学习！

    def fit(self, X, y):
        # 1. 验证输入数据 X 和 y（可选但建议）
        #    from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
        #    X, y = check_X_y(X, y)

        # 2. 根据 X、y 和 self.hyperparameter1 等执行学习过程。
        #    例如，计算一些内部模型状态。
        #    ... 学习逻辑 ...

        # 3. 存储以一个下划线结尾的学习到的属性
        self.learned_parameter_ = "从数据中导出的某个值"
        self.n_features_in_ = X.shape[1] # 常见的学习到的属性

        # 4. 返回实例
        return self

    def predict(self, X):
        # 1. 检查 fit 是否已被调用
        #    check_is_fitted(self)

        # 2. 验证输入数据 X
        #    X = check_array(X)

        # 3. 使用 self.learned_parameter_ 和 X 进行预测
        #    ... 预测逻辑 ...
        #    predictions = ...
        # return predictions

        # 占位符实现
        # 在实际分类器中，这将使用 learned_parameter_
        # from sklearn.utils.validation import check_is_fitted, check_array
        # check_is_fitted(self)
        # X = check_array(X)
        # 在此处实现预测逻辑
        pass # 替换为实际的预测逻辑

    # 根据估计器类型（ClassifierMixin 添加默认的 score 方法），可能需要其他方法，如 predict_proba, score 等。
```

### 转换器：数据修改

转换器是用于数据预处理和特征工程的特定类型估计器。除了 `fit` 方法外，它们还实现另外两个重要方法：

1. **`transform(X)`**：接收输入数据 `X` 并返回 `X` 的 *转换后* 版本。此方法不应修改估计器的状态；它只使用在 `fit` 期间学习到的参数 (parameter)。
2. **`fit_transform(X, y=None)`**：一个便捷方法，在 *相同* 数据 `X` 上执行拟合和转换。它通常比分别调用 `fit` 和 `transform` 更具计算效率。Scikit-learn 在 `TransformerMixin` 中提供了默认实现，但如果需要提高性能，您可以重写它。默认实现只是调用 `self.fit(X, y).transform(X)`。

```python
# Scikit-learn 兼容转换器的基本结构
from sklearn.base import BaseEstimator, TransformerMixin

class MyCustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, include_feature_indices=None):
        self.include_feature_indices = include_feature_indices
        # 此处不进行学习

    def fit(self, X, y=None):
        # 如有需要，从 X 中学习（例如，用于缩放的最小值/最大值）
        # 存储以一个下划线 '_' 结尾的学习到的属性
        # 例如：self.min_ = X.min(axis=0)
        self.n_features_in_ = X.shape[1]
        # 返回 self
        return self

    def transform(self, X):
        # 检查 fit 是否已被调用
        # from sklearn.utils.validation import check_is_fitted, check_array
        # check_is_fitted(self)
        # X = check_array(X)

        # 使用学习到的属性（如果有）和超参数（如 self.include_feature_indices）应用转换
        # transformed_X = ... 逻辑 ...
        # return transformed_X

        # 占位符实现
        # 在实际转换器中，这将使用 fit 中学习到的参数
        # from sklearn.utils.validation import check_is_fitted, check_array
        # check_is_fitted(self)
        # X = check_array(X)
        # 在此处实现转换逻辑
        pass # 替换为实际的转换逻辑
```

### 预测器：进行预测

预测器是能够根据新输入数据进行预测的估计器。它们实现一个 `predict(X)` 方法，该方法接收未见过的数据 `X` 并根据 `fit` 中学习到的状态返回预测结果。存在不同类型的预测器：

- **分类器**：预测分类标签。通常实现 `predict_proba(X)`（获取每个类别的概率估计）和 `score(X, y)`（评估准确性）。继承自 `ClassifierMixin`。
- **回归器**：预测连续值。通常实现 `score(X, y)`（通常是 R 方系数）。继承自 `RegressorMixin`。
- **聚类器**：将样本分配给聚类（无监督）。可以实现 `predict(X)`（分配给最近的聚类）或 `fit_predict(X)`（拟合并返回训练数据的聚类标签）。继承自 `ClusterMixin`。

### 重要辅助类和函数

Scikit-learn 在 `sklearn.base` 中提供了基类和混合类以简化开发：

- **`BaseEstimator`**：基础的基类。提供 `get_params()` 和 `set_params()` 的默认实现。强烈建议继承它。
  - **`get_params(deep=True)`**：返回一个字典，将参数 (parameter)名称（如在 `__init__` 中定义的）映射到其当前值。对于 `Pipeline` 和 `GridSearchCV` 等元估计器检查和克隆估计器是必不可少的。
  - **`set_params(**params)`**：设置估计器的参数。也被元估计器广泛使用，特别是在超参数 (hyperparameter)调优方面。
- **混合类（`TransformerMixin`, `ClassifierMixin`, `RegressorMixin`, `ClusterMixin`）**：根据您实现的核心方法提供常用方法的默认实现。例如，`TransformerMixin` 根据您的 `fit` 和 `transform` 提供 `fit_transform`。`ClassifierMixin` 基于您的 `predict` 提供一个基于准确性的默认 `score` 方法。

此外，`sklearn.utils.validation` 模块包含对可靠估计器非常重要的函数：

- **`check_X_y(X, y, ...)`**：验证输入特征 `X` 和目标 `y`，将它们转换为标准 NumPy 数组并检查一致性（例如，样本数量）。
- **`check_array(X, ...)`**：验证输入特征 `X`。在 `predict` 或 `transform` 中使用，在这些方法中 `y` 不存在。
- **`check_is_fitted(estimator, attributes=None)`**：通过验证学习到的属性（那些以 `_` 结尾的）是否存在来检查估计器是否已拟合。在 `predict`、`transform` 或其他依赖拟合状态的方法开始时调用此函数。

### 为何遵循 API？

遵循这些约定可确保您的自定义组件在 Scikit-learn 生态系统中正常运行：

- **互操作性**：自定义估计器可以与标准组件一起放入 Scikit-learn `Pipeline` 对象中。
- **超参数 (parameter) (hyperparameter)调优**：`GridSearchCV` 和 `RandomizedSearchCV` 可以检查（`get_params`）和修改（`set_params`）您的估计器的超参数。
- **模型评估**：标准评估工具和交叉验证函数将正常工作。
- **一致性**：熟悉 Scikit-learn 的用户会发现您的自定义组件使用起来很直观。

通过掌握 Scikit-learn API 原则，使用 `__init__` 处理超参数，`fit` 进行学习，使用尾随下划线存储学习到的状态，并运用基类和验证工具，您将获得构建复杂自定义机器学习 (machine learning)组件的能力，这些组件可以完美地集成到既定工作流程中。后续部分将指导您实现这些原则以创建自定义转换器和估计器。

## 参考资料

- [Developing your own scikit-learn estimators](https://scikit-learn.org/stable/developers/develop.html) — scikit-learn developers (2023)
  提供创建与Scikit-learn生态系统集成的自定义组件的指导和示例。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGwyGbJdJeY6OdVoEBkAerpkAlnQS6H4QoBiPh_YlzgAc-6FhRUK3pEyQz-Tx2cFmnjDPBearYVm4LZwogz23B2ASYkos6TTdlQBp2S6PEWPAJXHwJl_iUVnIY-eQOumVg8u_JiRa2ju_pQ5_L1IT1pG-uaW19J9_-Xv1gTK6gv8PrpsoFzhiA=) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 864
  一本广泛使用的教科书，为在Scikit-learn框架内实现自定义估计器提供了实用说明和代码示例。
- [Scikit-learn: Machine Learning in Python](http://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf) — Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, Édouard Duchesnay (2011)
  Journal: Journal of Machine Learning Research; Publisher: JMLR, Inc.; Volume: 12; Pages: 2825-2830; DOI: [10.5555/1953048.2078195](https://doi.org/10.5555/1953048.2078195)
  介绍Scikit-learn库的原始学术论文，阐述了其设计原则和整体架构，包括估计器接口。

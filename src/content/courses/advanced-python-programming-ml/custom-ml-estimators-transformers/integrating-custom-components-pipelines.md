---
course: "advanced-python-programming-ml"
chapter: "custom-ml-estimators-transformers"
lesson: "integrating-custom-components-pipelines"
sourceId: 2377
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-6-custom-ml-estimators-transformers/integrating-custom-components-pipelines"
title: "将自定义组件整合到管道中"
description: "将自定义转换器和估计器与标准Scikit-learn组件一同用于管道。"
order: 6
plots: []
sourceHash: "1036abc3ff3a0360a06bcfd80183fcb6b4a32b2ac9b34fc32176fd4a0a6e19d4"
sourceCorrections: []
---

将按照Scikit-learn API开发的自定义转换器和估计器整合到一致的机器学习 (machine learning)工作流程中，是一项重要的实践。Scikit-learn的`Pipeline`对象是连接多个处理步骤和最终估计器的标准机制。遵循Scikit-learn接口的优点在于，您的自定义组件可以像内置组件一样自然地融入这些管道。

使用`Pipeline`提供了几个重要优点：

- **便利性：** 将多个步骤封装成一个单一对象，其行为类似于标准的Scikit-learn估计器。
- **联合参数 (parameter)选择：** 允许同时对管道内所有步骤的参数进行网格搜索。
- **信息泄露预防：** 确保交叉验证折叠中的缩放或特征选择等步骤仅使用该特定折叠的训练数据信息。

下面我们来看看如何纳入您的自定义成果。

## 整合自定义转换器

假设您已经创建了一个自定义转换器，例如一个根据数据类型选择特定列或执行专门转换的转换器。

```python
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# 假设这是之前定义的自定义转换器
# 选择特定数据类型的列
class DtypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        # 该转换器无需拟合
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("输入必须是pandas DataFrame")
        return X.select_dtypes(include=[self.dtype])

# 示例数据
data = {'numeric_feat1': [1, 2, 3, 4, 5],
        'numeric_feat2': [10.1, 12.3, 9.8, 15.6, 11.2],
        'categorical_feat': ['A', 'B', 'A', 'C', 'B'],
        'target': [0, 1, 0, 1, 0]}
X_train = pd.DataFrame(data).drop('target', axis=1)
y_train = pd.DataFrame(data)['target']

# 创建包含自定义转换器的管道
numeric_pipeline = Pipeline([
    ('select_numeric', DtypeSelector(dtype=np.number)), # 自定义步骤
    ('scale', StandardScaler())                          # 标准步骤
])

# 拟合并转换数据
X_train_processed = numeric_pipeline.fit_transform(X_train)

print("原始形状:", X_train.shape)
print("处理后的形状:", X_train_processed.shape)
print("\n处理后的数据（前2行）:\n", X_train_processed[:2])
```

在此示例中，`DtypeSelector`在`Pipeline`内被视为与任何其他转换器一样。当`numeric_pipeline.fit_transform(X_train)`被调用时：

1. `select_numeric`的`fit`方法被调用，随后是`transform`。
2. `select_numeric.transform`的输出（只包含数值列）被传递给`scale`步骤。
3. `scale`的`fit`方法使用这些中间数据被调用，随后是`transform`。
4. 最终转换后的数据被返回。

您可以组合多个自定义转换器、标准转换器，或者使用像`ColumnTransformer`和`FeatureUnion`这样的结构，它们本身也可以包含自定义组件。

## 整合自定义估计器

类似地，遵循Scikit-learn接口的自定义估计器可以作为`Pipeline`的最后一步。

假设您已经开发了一个`CustomLogisticRegression`估计器（例如带有独特的正则化 (regularization)或优化功能）。

```python
# 假设这是之前定义的自定义估计器
# 它必须实现fit(X, y)和predict(X)方法
class CustomLogisticRegression(BaseEstimator): # 简化示例
    def __init__(self, learning_rate=0.01, iterations=100):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None

    def _sigmoid(self, z):
       return 1 / (1 + np.exp(-np.clip(z, -250, 250))) # 增加了clip以提高稳定性

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1) # 添加截距项
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.iterations):
            z = X @ self.weights
            h = self._sigmoid(z)
            gradient = (X.T @ (h - y)) / n_samples
            self.weights -= self.learning_rate * gradient
        return self

    def predict_proba(self, X):
       X = np.insert(X, 0, 1, axis=1) # 添加截距项
       proba_class_1 = self._sigmoid(X @ self.weights)
       proba_class_0 = 1 - proba_class_1
       return np.vstack((proba_class_0, proba_class_1)).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# 构建完整管道
full_pipeline = Pipeline([
    ('preprocess', numeric_pipeline), # 使用先前的数值管道
    ('classify', CustomLogisticRegression(learning_rate=0.05)) # 自定义估计器
])

# 拟合管道
full_pipeline.fit(X_train, y_train)

# 进行预测
predictions = full_pipeline.predict(X_train)
print("\n预测结果:", predictions)
```

这里，`full_pipeline`结合了在`numeric_pipeline`中定义的预处理步骤和自定义估计器`CustomLogisticRegression`。当`full_pipeline.fit(X_train, y_train)`被执行时：

1. `X_train`如前所述通过`numeric_pipeline.fit_transform`。
2. 完全处理过的数据`X_train_processed`随后与`y_train`一起传递给`classify`步骤的`fit`方法（即`CustomLogisticRegression.fit(X_train_processed, y_train)`）。

当`full_pipeline.predict(X_train)`被调用时，`X_train`会通过`numeric_pipeline.transform`（请注意：只有`transform`，而非`fit_transform`），然后结果被传递给`classify.predict`。

## 参数 (parameter)访问与超参数 (hyperparameter)调优

使用`Pipeline`的一个主要优点是它提供了统一的参数访问和设置接口，这同样适用于您的自定义组件。管道内步骤的参数使用双下划线`__`分隔符进行访问：`step_name__parameter_name`。

```python
# 访问参数
print("\n管道默认参数:")
print(full_pipeline.get_params()['classify__learning_rate'])

# 设置参数
full_pipeline.set_params(classify__learning_rate=0.1, classify__iterations=200)
print("\n管道更新后的参数:")
print(full_pipeline.get_params()['classify__learning_rate'])
print(full_pipeline.get_params()['classify__iterations'])
```

这种语法对于`GridSearchCV`或`RandomizedSearchCV`等超参数优化工具来说非常重要。您可以定义一个搜索空间，包含标准Scikit-learn组件和自定义组件的参数。

```python
from sklearn.model_selection import GridSearchCV

# 定义参数网格，包括自定义估计器参数
param_grid = {
    'preprocess__scale__with_mean': [True, False], # StandardScaler的参数
    'classify__learning_rate': [0.01, 0.05, 0.1],    # CustomLogisticRegression的参数
    'classify__iterations': [100, 200]            # CustomLogisticRegression的参数
}

# 设置GridSearchCV
# 注意：为演示目的，此处使用少量样本数据和少量交叉验证折叠
grid_search = GridSearchCV(full_pipeline, param_grid, cv=2, n_jobs=-1)

# 运行搜索（为简化起见，此处使用已处理数据，
# 但通常会传递原始的X_train）
# 对于此演示，我们直接在已处理数据上拟合网格搜索
# 因为我们的自定义估计器尚未与分类特征交互。
# 在实际场景中，您需要ColumnTransformer来处理不同类型。
# X_train_processed = numeric_pipeline.fit_transform(X_train)
# grid_search.fit(X_train_processed, y_train) # 仅在网格搜索中拟合估计器部分

# 更完整的网格搜索拟合会是这样：
# grid_search.fit(X_train, y_train)

# 在本解释中，为简洁起见，我们将跳过实际的拟合过程。
# print("\n找到的最佳参数:")
# print(grid_search.best_params_)
```

这说明了自定义组件如何整合到标准Scikit-learn模型选择和评估工作流程中，前提是它们正确实现了所需的方法和参数处理（`get_params`、`set_params`）。

## 可视化包含自定义组件的管道

理解复杂管道的结构，特别是涉及自定义步骤的管道，可以通过可视化来辅助。Scikit-learn提供HTML表示，您也可以创建图表。

让我们可视化`full_pipeline`的结构。

> `full_pipeline`的结构，它包含嵌套在管道内的自定义`DtypeSelector`和自定义`CustomLogisticRegression`估计器。

这种可视化阐明了操作顺序以及数据如何流经标准和自定义组件。

通过将您的自定义转换器和估计器整合到Scikit-learn管道中，您获得了一种有效方式来构建复杂的机器学习 (machine learning)工作流程，使它们更具模块化、可复现性，且更容易调优。请记住，这种顺畅结合的要点在于在构建自定义类时严格遵循Scikit-learn API约定。

## 参考资料

- [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html) — scikit-learn developers (2024)
  描述了Scikit-learn的Pipeline对象、其优势以及如何实现处理步骤和估计器的链式连接。
- [Rolling your own estimator](https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator) — scikit-learn developers (2024)
  提供了关于如何通过遵循API约定来创建Scikit-learn兼容的自定义估计器和转换器的指南。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://books.google.com/books?id=y90TEAAAQBAJ) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 850
  一本实践指南，详细介绍了Scikit-learn管道、自定义转换器和超参数调优，并附有示例。第3版。

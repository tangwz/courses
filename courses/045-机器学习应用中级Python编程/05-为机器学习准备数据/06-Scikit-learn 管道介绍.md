# Scikit-learn 管道介绍

来源：[原文](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-5-preparing-data-for-ml/sklearn-pipelines-intro)

[返回章节目录](README.md) · [返回课程目录](../README.md)

如你所见，为机器学习 (machine learning)准备数据通常包含多个连续步骤：处理缺失值、编码分类特征、缩放数值特征，甚至可能生成新特征。逐一应用这些步骤可能会变得繁琐且易出错，特别是当你需要确保将完全相同的转换序列一致地应用于训练数据和任何新数据（例如测试集或生产中遇到的数据）时。

想象一下，在将数据集分割为训练集和测试集之前，你对整个数据集拟合一个 `StandardScaler`。缩放器会从所有数据（包括测试集）中学习均值和标准差。当你训练模型时，它会通过缩放参数 (parameter)隐式地获得关于测试集的信息。这种现象，称为数据泄露，可能在开发过程中导致过于乐观的性能评估，因为你的模型在预处理阶段无意中“看到了”测试数据。将仅从训练数据中学习到的转换应用于测试数据对于可靠的模型评估非常重要。

这就是 Scikit-learn 的 `Pipeline` 对象变得非常有用的地方。`Pipeline` 允许你将多个处理步骤（转换器）以及一个可选的最终估计器（如分类器或回归器）串联成一个单一对象。此对象的行为类似于标准的 Scikit-learn 估计器，具有 `fit`、`transform` 和 `predict` 方法（取决于最后一步）。

### 为何使用管道？

使用 `Pipeline` 具有几个重要优势：

1. **便捷性和封装性：** 它将多个处理步骤捆绑成一个单元，简化了你的代码。你无需多次调用 `fit_transform` 或 `fit` 和 `transform`，只需与单一的管道对象交互。
2. **联合参数 (parameter)选择：** 在进行超参数 (hyperparameter)调整时（例如，使用 `GridSearchCV` 或 `RandomizedSearchCV`），你可以同时调整管道中所有步骤的参数，包括预处理步骤和最终估计器。
3. **防止数据泄露：** 这是一个主要优点。当你在一个管道上调用 `fit` 时，它会正确地仅在提供给 `fit` 方法的训练数据上拟合转换器。中间步骤调用 `fit_transform`，将转换后的数据传递给下一步。当你对新数据（如测试集）调用 `transform` 或 `predict` 时，管道会确保只调用已拟合转换器的 `transform` 方法，一致地应用所学转换，而无需在新数据上重新拟合。

### 管道的结构

构建 `Pipeline` 需要提供一个步骤列表。每个步骤都是一个元组，包含一个独特的名称（你选择的字符串）和一个转换器或估计器的实例。

```python
# 示例结构
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# 定义步骤：（名称，转换器/估计器实例）
steps = [
    ('imputer', SimpleImputer(strategy='mean')), # 步骤 1: 填充缺失值
    ('scaler', StandardScaler()),               # 步骤 2: 缩放特征
    ('classifier', LogisticRegression())        # 步骤 3: 最终估计器
]

# 创建管道
pipe = Pipeline(steps=steps)

# 现在 'pipe' 可以像普通估计器一样使用：
# pipe.fit(X_train, y_train)
# predictions = pipe.predict(X_test)
# score = pipe.score(X_test, y_test)
```

管道中的所有中间步骤*必须*是转换器，这意味着它们必须同时实现 `fit` 和 `transform` 方法。最后一步可以是任何估计器：转换器、分类器或回归器。

当调用 `pipe.fit(X_train, y_train)` 时：

1. `imputer` 在 `X_train` 上拟合，然后 `X_train` 被 `imputer` 转换。
2. 转换后的数据被传递给 `scaler`，`scaler` 进行拟合然后转换数据。
3. 缩放后的数据连同 `y_train` 一起传递给 `classifier`，然后 `classifier` 进行拟合。

当调用 `pipe.predict(X_test)` 时：

1. `X_test` 由*已拟合的* `imputer` 转换。
2. 结果数据由*已拟合的* `scaler` 转换。
3. 缩放后的数据传递给*已拟合的* `classifier` 的 `predict` 方法。

这种顺序应用确保处理逻辑被正确一致地应用，防止在评估阶段发生数据泄露。

这是一个简单的管道结构可视化：

> 显示 Scikit-learn 管道在拟合阶段（上方）和预测阶段（下方）的数据流图。请注意，在拟合阶段，转换器使用 `fit_transform`，而在预测阶段仅使用 `transform`。

当需要对不同列应用不同的转换时（例如，缩放数值列和对分类列进行独热编码），管道通常会变得更复杂。Scikit-learn 的 `ColumnTransformer` 正是为了处理这种情况而设计的，并且在 `Pipeline` 内运行良好，使你能够构建高效的预处理工作流。我们将在后续章节和练习中查看构建这些管道的实际例子。

## 参考资料

- [Chaining and composing estimators](https://scikit-learn.org/stable/modules/compose.html) — Scikit-learn Developers (2023)
  Publisher: Scikit-learn
  官方文档，解释了用于顺序处理的Scikit-learn Pipeline对象，以及用于对不同数据子集应用不同转换的ColumnTransformer。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHWgXamnGNVAyiqGbTAmBm2Nt3du6x05UdR-j7cweO_ZD8rvbEv1daN5i2NaEY8IS01vw52audJ-Dl5Sz5rVup_5_MabidzOEfzsBg8z9--NvYU3t9m2eI9eRq_uXWtv94PJGMXLDqXPtnl-SqSJ9YoCswsQsVQumcvcsuC6XuJQYMRl5Zp6xJj5rCg55g3W63GRQ==) — Aurélien Géron (2022)
  Publisher: O'Reilly Media, Inc.; Pages: 864
  一本包含机器学习系统构建实例的实践指南，涵盖数据预处理、模型训练，以及使用Scikit-learn Pipeline构建高效且一致的工作流。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本涵盖统计学习理论和实践的教科书，包括模型评估、选择以及数据泄露问题。

---

[上一节](05-%E5%B0%86%E6%95%B0%E6%8D%AE%E5%88%92%E5%88%86%E4%B8%BA%E8%AE%AD%E7%BB%83%E9%9B%86%E5%92%8C%E6%B5%8B%E8%AF%95%E9%9B%86.md) · [下一节](07-%E4%BF%9D%E6%8C%81%E6%95%B0%E6%8D%AE%E5%8F%98%E6%8D%A2%E7%9A%84%E4%B8%80%E8%87%B4%E6%80%A7.md)

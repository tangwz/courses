---
course: "getting-started-with-scikit-learn"
chapter: "building-pipelines"
lesson: "accessing-pipeline-steps"
sourceId: 1899
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-6-building-pipelines/accessing-pipeline-steps"
title: "访问流水线步骤"
description: "如何检查和访问已定义的流水线中的各个步骤。"
order: 3
plots: []
sourceHash: "08077b4f0bf1e00c8efd734f3f167cb2dd23023c7f7911f8b85ba7e053746d34"
sourceCorrections: []
---

一个 `Pipeline` 作为一个单独的 Scikit-learn 估计器。然而，它封装了多个步骤，通常是一系列转换器之后接一个最终的预测器。您可能需要检查这些独立的组件，也许是为了理解标量（scaler）学到的参数 (parameter)，或是流水线拟合后线性模型确定的系数。Scikit-learn 提供了简单直接的方法来访问这些内部步骤。

访问 `Pipeline` 组件的主要属性是 `steps`。该属性包含一个列表，其中每个元素都是一个元组，包括您为步骤提供的名称和估计器对象本身。

考虑一个由标量（scaler）和逻辑回归分类器组成的简单流水线：

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# 示例数据（请替换为实际数据）
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([0, 0, 1, 1])

# 定义流水线
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])

# 拟合流水线
pipe.fit(X_train, y_train)
```

### 按索引访问步骤

由于 `steps` 是一个列表，您可以使用标准列表索引来访问各个步骤。每个元素都是一个 `(名称, 估计器)` 元组。

```python
# 获取第一个步骤（标量元组）
first_step_tuple = pipe.steps[0]
print(f"第一个步骤元组: {first_step_tuple}")

# 获取第一个步骤的名称
first_step_name = pipe.steps[0][0]
print(f"第一个步骤名称: {first_step_name}") # 输出: scaler

# 获取第一个步骤的估计器对象
first_step_estimator = pipe.steps[0][1]
print(f"第一个步骤估计器: {first_step_estimator}") # 输出: StandardScaler()

# 获取第二个步骤（分类器元组）
second_step_tuple = pipe.steps[1]
print(f"第二个步骤元组: {second_step_tuple}")

# 获取最终的估计器对象
final_estimator = pipe.steps[-1][1] # 使用 -1 获取最后一个步骤
print(f"最终估计器: {final_estimator}") # 输出: LogisticRegression(...)
```

### 按名称访问步骤

尽管索引访问可行，但可读性可能较差，尤其是在较长的流水线中。Scikit-learn 流水线也支持使用您为步骤分配的名称进行字典式访问。这通常是更便捷和明确的方法。

```python
# 按名称访问标量对象
scaler_object = pipe['scaler']
print(f"按名称访问的标量对象: {scaler_object}") # 输出: StandardScaler()

# 按名称访问分类器对象
classifier_object = pipe['classifier']
print(f"按名称访问的分类器对象: {classifier_object}") # 输出: LogisticRegression(...)
```

这种按名称访问的方式使您的代码对流水线顺序的变化有更强的适应性（尽管改变顺序通常意味着不同的工作流程）。

### 检查流水线中拟合的参数 (parameter)

访问各个步骤的一个主要原因是检查流水线拟合后的状态。当您对整个流水线调用 `fit()` 时，它会依次对每个转换器调用 `fit_transform()`，最后对最后一个估计器调用 `fit()`。拟合后的属性（例如 `StandardScaler` 的 `mean_` 或 `LogisticRegression` 的 `coef_`）存储在流水线内部各自的估计器对象中。

您可以通过命名的步骤来访问这些属性：

```python
# 访问 StandardScaler 学到的均值
scaler_mean = pipe['scaler'].mean_
print(f"标量学到的均值: {scaler_mean}")

# 访问 LogisticRegression 学到的系数
classifier_coef = pipe['classifier'].coef_
print(f"分类器学到的系数: {classifier_coef}")

# 访问 LogisticRegression 学到的截距
classifier_intercept = pipe['classifier'].intercept_
print(f"分类器学到的截距: {classifier_intercept}")
```

这使您能够解释预处理步骤的结果，并理解最终模型的参数，即使它们被整齐地封装在流水线中。

> 示例流水线 `pipe` 中访问步骤及其参数的图示。名称和索引提供了访问内部估计器的途径。

理解如何按名称访问流水线步骤在与 `GridSearchCV` 等工具进行超参数 (hyperparameter)调优时也很重要。您需要使用 `estimator_name__parameter_name` 语法来指定参数，这直接依赖于您分配给流水线步骤的名称。我们将在“流水线中的网格搜索”部分详细介绍这一点。

## 参考资料

- [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html) — Scikit-learn developers (2024)
  Publisher: Scikit-learn Project
  Scikit-learn官方用户指南，解释了管道的目的、结构和基本用法。
- [sklearn.pipeline.Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) — Scikit-learn developers (2024)
  《Pipeline》类的官方API文档，详细介绍了其属性（如`steps`）和方法。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125964/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 864
  一本涵盖Scikit-learn管道的综合实践指南，包括访问内部步骤和检查拟合参数的详细示例。

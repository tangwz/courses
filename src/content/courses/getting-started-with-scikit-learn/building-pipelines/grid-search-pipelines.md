---
course: "getting-started-with-scikit-learn"
chapter: "building-pipelines"
lesson: "grid-search-pipelines"
sourceId: 1901
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-6-building-pipelines/grid-search-pipelines"
title: "使用管道进行网格搜索"
description: "在管道中同时对预处理步骤和估计器进行超参数调整。"
order: 5
plots: []
sourceHash: "7c8ffe5200bf8ef74fa3c943f40415ac12c2cc313e4f3087dae7ce21ecce6693"
sourceCorrections: []
---

管道通过将预处理和模型构建步骤打包，提高你的工作流程效率。每个步骤，包括最终的估计器，都可能拥有需要优化的超参数 (parameter) (hyperparameter)，以获得最佳表现。在管道环境中手动调整这些参数，尤其是在交叉验证期间，会重新引入复杂性和潜在错误。Scikit-learn 的 `GridSearchCV` 与 `Pipeline` 对象很好地结合，能够同时调整*所有*步骤的超参数。

主要的难题是告诉 `GridSearchCV` 管道中 *哪个* 参数属于 *哪个* 步骤。Scikit-learn 为此使用了一个特定的命名约定：`step_name__parameter_name`。你将管道中给步骤分配的名称（例如，'scaler'、'classifier'）与该步骤的参数名称（例如，`LogisticRegression` 的 `C`、`KNeighborsClassifier` 的 `n_neighbors`，甚至是转换器（如 `TfidfTransformer`）的参数，例如 `use_idf`）组合起来，它们之间用双下划线 (`__`) 分隔。

让我们用一个例子来说明这一点。假设我们有一个管道，它首先使用 `StandardScaler` 对数据进行缩放，然后应用一个 `LogisticRegression` 分类器：

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 创建示例数据
X, y = make_classification(n_samples=100, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 定义管道
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42))
])

# 定义参数网格
# 注意 'step_name__parameter_name' 语法
param_grid = {
    'scaler__with_mean': [True, False],  # StandardScaler 的参数
    'classifier__C': [0.1, 1.0, 10.0],   # LogisticRegression 的参数
    'classifier__penalty': ['l1', 'l2'] # LogisticRegression 的参数
}

# 创建 GridSearchCV 对象
grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)

# 拟合 GridSearchCV
# 管道（包括缩放）在每个交叉验证折叠上进行拟合
grid_search.fit(X_train, y_train)

# 打印找到的最佳参数
print(f"Best parameters found: {grid_search.best_params_}")

# best_estimator_ 属性是带有最佳参数的已拟合管道
best_pipeline = grid_search.best_estimator_
print(f"\nBest pipeline found:\n{best_pipeline}")

# 在测试集上评估最佳管道
score = best_pipeline.score(X_test, y_test)
print(f"\nTest set score with best pipeline: {score:.4f}")
```

在此示例中：

1. 我们定义了一个名为 `pipe` 的 `Pipeline`，其中包含名为 'scaler' 和 'classifier' 的步骤。
2. `param_grid` 字典使用诸如 `'scaler__with_mean'` 和 `'classifier__C'` 这样的键。`'scaler__with_mean'` 键告诉 `GridSearchCV` 尝试为名为 'scaler' 的步骤（我们的 `StandardScaler`）使用 `with_mean` 参数值 `[True, False]`。类似地，`'classifier__C'` 和 `'classifier__penalty'` 定向于名为 'classifier' 的步骤（我们的 `LogisticRegression`）的 `C` 和 `penalty` 超参数。
3. `GridSearchCV` 将 `pipe` 对象作为其估计器。
4. 在 `GridSearchCV` 的交叉验证过程中，对于每个参数组合和每个折叠：
   - 折叠的训练部分用于拟合*整个管道*（包括缩放器）。
   - 已拟合的管道用于对折叠的验证部分进行预测。
   - 性能得到评估。
5. 这确保了缩放（或任何其他预处理步骤）*仅*从该特定折叠的训练数据中学习，从而防止验证集中的数据泄漏到预处理步骤中。
6. 拟合后，`grid_search.best_params_` 包含参数名称（使用 `step_name__parameter_name` 格式）及其最优值的字典。
7. 重要的是，`grid_search.best_estimator_` 是一个完全拟合的 `Pipeline` 对象，其中包含了所有已调优步骤的最佳超参数。你可以直接使用这个 `best_estimator_` 管道对新数据进行预测或在最终测试集上进行评估。

这种方法巧妙地结合了交叉验证的强健性、网格搜索的全面性以及管道的组织优势。当你的工作流程涉及预处理步骤时，这是执行超参数调整的标准和推荐方式。同样的 `step_name__parameter_name` 语法也适用于更复杂的管道，包括那些使用 `ColumnTransformer` 构建的管道，在这些情况下，你可能会有嵌套命名，例如 `'preprocessor__num__imputer__strategy'`，如果 'preprocessor' 是你的 `ColumnTransformer` 的名称，'num' 是转换器中应用于数值特征的管道的名称，而 'imputer' 是*该*管道中的一个步骤。

## 参考资料

- [Tuning the hyperparameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html) — Scikit-learn Developers (2023)
  详细说明`GridSearchCV`与`Pipeline`对象的集成，包括`step_name__parameter_name`语法和推荐实践。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  提供使用Scikit-learn管道和网格搜索进行超参数调整的实际示例和解释。
- [Scikit-learn: Machine Learning in Python](https://www.jmlr.org/papers/v12/pedregosa11a.html) — Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournaupeau, Matthieu Brucher, Matthieu Perrot, Édouard Duchesnay (2011)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 12; Pages: 2825-2830; DOI: [10.5555/1953048.2078195](https://doi.org/10.5555/1953048.2078195)
  介绍Scikit-learn库，其模块化架构支持管道的创建和优化。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://statweb.stanford.edu/~tibs/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; Pages: 745; DOI: [10.1007/b94608](https://doi.org/10.1007/b94608)
  提供交叉验证和超参数选择的理论基础，对于理解`GridSearchCV`和管道验证的原理至关重要。

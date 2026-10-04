---
course: "getting-started-with-scikit-learn"
chapter: "model-selection-evaluation"
lesson: "grid-search-hyperparameter-tuning"
sourceId: 1895
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-5-model-selection-evaluation/grid-search-hyperparameter-tuning"
title: "网格搜索用于超参数调整"
description: "使用`GridSearchCV`系统地寻找最佳模型超参数。"
order: 6
plots: []
sourceHash: "cef23a94a48341ce93d1f76cae01638b79afa3e358e59b4adfc97542d3ccd61e"
sourceCorrections: []
---

交叉验证能让我们更可靠地评估模型在未见过数据上的表现，但大多数机器学习 (machine learning)模型也有一些设置，称为**超参数 (parameter) (hyperparameter)**，这些参数并非直接从数据中学习，而是在训练过程开始前设定的。可以把它们看作是学习算法的配置旋钮。例子包括 KNN 中的邻居数量（`n_neighbors`）、支持向量 (vector)机（SVM）中的正则化 (regularization)强度（`C`）或核类型（`kernel`），以及决策树的深度。找到这些超参数的最佳值能大幅影响模型性能。

手动尝试不同的超参数组合、训练模型并使用交叉验证评估模型，会很繁琐且效率不高。Scikit-learn 提供了一种自动执行这种搜索的方法：**网格搜索**。

### 理解超参数 (parameter) (hyperparameter)与参数

区分模型*参数*和*超参数*是很重要的：

- **参数**：这些是模型在*训练过程*中根据数据学习到的值。例子包括线性回归模型中的系数或神经网络 (neural network)中的权重 (weight)。你不需要事先设定它们；算法会找到它们。
- **超参数**：这些是在训练开始*之前*设定的配置。它们配置学习过程本身或定义模型架构的特性。超参数的选择指导模型如何学习参数。例如：KNN 中的 `k`，SVM 中的 `C` 和 `gamma`，以及梯度下降 (gradient descent)中的学习率。

网格搜索侧重于寻找最佳*超参数*。

### 网格搜索方法

网格搜索背后的思路很直接：

1. **定义网格**：为每个你想调整的超参数 (parameter) (hyperparameter)指定一组可能的值。Scikit-learn 期望这是一个字典，里面键是超参数名称（字符串），值是待尝试的值的列表或数组。
2. **训练与评估**：网格搜索系统地尝试网格中超参数值的每一种可能组合。
3. **交叉验证**：对于每种组合，它使用交叉验证（如前一节所述）在训练数据上评估模型的性能。这确保了每种组合的性能评估是可靠的。
4. **选择最佳**：产生最高平均交叉验证分数（根据指定的度量标准）的超参数组合被确定为最佳组合。

### 使用 `GridSearchCV` 实现网格搜索

Scikit-learn 的 `GridSearchCV` 类使得这个过程易于实现。我们来具体看看如何使用它。

#### 1. 导入所需模块

```python
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
```

#### 2. 准备数据

我们将在这个例子中使用 Iris 数据集。我们还需要将数据分成训练集和测试集，因为网格搜索应仅使用训练数据进行，以避免测试集的信息泄露。

```python
# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
```

#### 3. 定义估计器和参数 (parameter)网格

选择你想要调整的模型（估计器）。这里，我们将使用 `SVC` (支持向量 (vector)分类器)。然后，定义 `param_grid` 字典。

```python
# 定义估计器
svm_model = SVC()

# 定义要搜索的超参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],            # 正则化参数
    'gamma': [1, 0.1, 0.01, 0.001],    # 'rbf' 核的核系数
    'kernel': ['rbf', 'linear']        # 核类型
}
```

这个网格为 `C` 指定了 4 个值，为 `gamma` 指定了 4 个值，为 `kernel` 指定了 2 个值。`GridSearchCV` 将评估 $4 \times 4 \times 2 = 32$ 种不同组合的这些超参数 (hyperparameter)。请注意，`gamma` 参数只被 `rbf` 核使用，但 `GridSearchCV` 足够智能，可以处理这种情况。

#### 4. 实例化 `GridSearchCV`

创建一个 `GridSearchCV` 实例，传入估计器、参数网格、交叉验证策略（`cv`）以及可选的评分度量。

```python
# 实例化 GridSearchCV
# cv=5 表示 5 折交叉验证
# scoring='accuracy' 指定要优化的度量
grid_search = GridSearchCV(estimator=svm_model,
                           param_grid=param_grid,
                           cv=5,
                           scoring='accuracy',
                           verbose=1, # 可选：打印进度
                           n_jobs=-1) # 可选：使用所有可用的 CPU 核心
```

- `estimator`：模型实例（`svm_model`）。
- `param_grid`：定义要尝试的超参数的字典（`param_grid`）。
- `cv`：交叉验证划分策略。一个整数（如 5）指定 K 折交叉验证（或分类的层叠 K 折交叉验证）。你也可以传入特定的 CV 划分器对象。
- `scoring`：用于评估每个超参数组合性能的度量标准。常见值包括分类的 `'accuracy'`、`'precision'`、`'recall'`、`'f1'`，以及回归的 `'neg_mean_squared_error'`、`'r2'`。如果为 `None`，则使用估计器的默认评分器。
- `verbose`：控制详细程度。值越高，输出消息越多。
- `n_jobs`：用于并行处理的 CPU 核心数量。`-1` 通常表示使用所有可用核心，这可以大幅加快搜索速度。

#### 5. 拟合 `GridSearchCV`

将 `GridSearchCV` 对象拟合到*训练*数据。这会触发搜索过程。

```python
# 将网格搜索对象拟合到训练数据
grid_search.fit(X_train, y_train)
```

这一步可能需要一些时间，因为它涉及多次训练和评估模型（组合数 × CV 折数）。在我们的例子中，是 $32 \times 5 = 160$ 次模型拟合。

#### 6. 查看结果

拟合完成后，`GridSearchCV` 会将结果存储在几个有用的属性中：

- `best_params_`：一个字典，包含产生最佳平均交叉验证分数的超参数组合。

  ```python
  # 打印找到的最佳参数
  print(f"Best Hyperparameters: {grid_search.best_params_}")
  ```
- `best_score_`：使用 `best_params_` 获得的平均交叉验证分数。

  ```python
  # 打印最佳交叉验证分数
  print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
  ```
- `best_estimator_`：一个估计器实例，它已使用 `best_params_` 在*整个*训练数据集（`X_train`，`y_train`）上自动重新拟合。这通常是你将用于对新数据（如测试集）进行预测的最终模型。

  ```python
  # 获取最佳估计器
  best_svm_model = grid_search.best_estimator_

  # 在测试集上评估最佳模型
  y_pred = best_svm_model.predict(X_test)
  test_accuracy = accuracy_score(y_test, y_pred)
  print(f"Test Set Accuracy with Best Model: {test_accuracy:.4f}")
  ```
- `cv_results_`：一个字典，包含网格搜索过程中评估过的所有组合的详细信息。这对于进行更深入的分析很有用，通常会转换为 Pandas DataFrame 以便查看。

  ```python
  import pandas as pd
  # 显示详细结果（可选）
  cv_results_df = pd.DataFrame(grid_search.cv_results_)
  # print(cv_results_df[['param_C', 'param_gamma', 'param_kernel', 'mean_test_score', 'rank_test_score']].sort_values('rank_test_score').head())
  ```

### 注意事项

- **计算成本**：网格搜索是穷举式的。如果你有很多超参数 (parameter) (hyperparameter)或每个超参数的值范围很广，组合数量会增长得非常快，使得搜索计算成本高昂且耗时。
- **选择网格**：为网格选择合适的范围和值需要一些领域知识或实验。通常，你可能会从一个宽泛、粗糙的网格开始，然后围绕最初找到的最佳值执行一个更窄、更精细的网格搜索。
- **替代方法**：对于非常大的搜索空间，`RandomizedSearchCV`（它随机采样固定数量的组合）或更高级的贝叶斯优化技术等方法可能是更高效的替代方案，尽管 `GridSearchCV` 通常是一个好的起点。

通过使用 `GridSearchCV`，你可以系统地尝试模型的不同超参数设置，利用交叉验证找到在未见过数据上平均表现最佳的配置，从而得到性能更好的机器学习 (machine learning)方案。这个过程是构建有效模型的标准且有价值的一步。

## 参考资料

- [sklearn.model_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) — scikit-learn developers (2023)
  Scikit-learn GridSearchCV 的官方文档，提供了详细的 API 规范、参数和超参数调整的使用示例。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2023)
  Publisher: O'Reilly Media
  一本实用的指南，涵盖了机器学习概念，包括使用 Scikit-learn 的 GridSearchCV 进行超参数调整，并提供代码示例和解释。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本基础教科书，为各种机器学习方法（包括模型选择、交叉验证和超参数考量）提供了严谨的统计和理论背景。
- [Random Search for Hyper-Parameter Optimization](https://jmlr.org/papers/v13/bergstra12a.html) — James Bergstra and Yoshua Bengio (2012)
  Journal: Journal of Machine Learning Research; Volume: 13; Pages: 281-305
  一篇开创性论文，表明随机搜索在超参数优化方面通常比网格搜索更有效，为网格搜索的局限性提供了背景，并介绍了一个重要的替代方案。

# XGBoost API：使用指南

来源：[原文](https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-4-advanced-gradient-boosting-xgboost/xgboost-api-walkthrough)

[返回章节目录](README.md) · [返回课程目录](../README.md)

有效应用XGBoost需要了解其Python接口。该库提供两种主要的模型构建方式：其原生Python API和符合Scikit-Learn API的包装类。首先将介绍原生API，因为它展现了库的主要组成部分，并提供了更大的灵活性。

### 主要数据结构：DMatrix

与直接使用NumPy数组或Pandas DataFrame的Scikit-Learn估计器不同，XGBoost的原生API使用一种名为`DMatrix`的内部数据结构。这是一种内存效率高、性能优化的数据容器，专门为该库的算法设计。将数据转换为`DMatrix`是原生工作流程的第一步。

您可以从多种数据类型创建`DMatrix`，包括NumPy数组、SciPy稀疏矩阵和Pandas DataFrame。为训练创建`DMatrix`时，您需要使用`label`参数 (parameter)提供特征矩阵（您的`X`数据）和目标向量 (vector)（您的`y`数据）。

```python
import xgboost as xgb
import numpy as np
import pandas as pd

# 生成样本训练数据
X_train_data = np.random.rand(100, 5)
y_train_data = np.random.rand(100)

# 从NumPy数组创建DMatrix
dtrain = xgb.DMatrix(X_train_data, label=y_train_data)

print(f"创建对象的类型: {type(dtrain)}")
```

`dtrain`对象现在以适合高速训练的格式保存了我们的数据。对于预测数据，您以相同的方式创建`DMatrix`，但省略`label`参数，因为目标是未知的。

### 指定模型超参数 (parameter) (hyperparameter)

在原生API中，超参数不作为参数传递给模型构造函数。相反，它们在一个Python字典中定义。这个字典包含键值对，其中键是参数名称（例如，`'max_depth'`），值是其设置。

这种方法使得管理、保存和修改参数集变得容易。我们来为一个回归任务定义一个基本的参数字典。

```python
# 在字典中定义模型的超参数
params = {
    'objective': 'reg:squarederror',  # 要最小化的损失函数
    'max_depth': 3,                   # 每棵决策树的最大深度
    'eta': 0.1,                       # 学习率，也称为 'learning_rate'
    'eval_metric': 'rmse'             # 用于在验证集上评估的指标
}
```

`objective`参数是最主要的参数之一，因为它指定了学习任务。常见的`objective`包括用于回归的`reg:squarederror`、用于二元分类的`binary:logistic`和用于多元分类的`multi:softmax`。`eta`参数是学习率的同义词，它控制每个提升迭代的步长。

### 使用 xgb.train 训练模型

将训练数据放入`DMatrix`中，并将参数 (parameter)放入字典后，您可以使用`xgb.train()`函数训练模型。该函数至少需要三个参数：

1. `params`：超参数 (hyperparameter)字典。
2. `dtrain`：包含训练数据的`DMatrix`。
3. `num_boost_round`：要构建的树的总数，相当于Scikit-Learn中的`n_estimators`。

```python
# 设置提升轮数
num_boost_round = 50

# 训练模型
bst = xgb.train(params, dtrain, num_boost_round)
```

该函数返回一个已训练的模型对象，我们将其命名为`bst`。此对象现在可用于对新数据进行预测。

### 生成预测

要进行预测，首先将测试数据集转换为`DMatrix`（不带`label`）。然后，在已训练的模型对象上调用`.predict()`方法。

```python
# 生成样本测试数据
X_test_data = np.random.rand(20, 5)

# 将测试数据转换为DMatrix
dtest = xgb.DMatrix(X_test_data)

# 生成预测
predictions = bst.predict(dtest)

print("样本预测：")
print(predictions[:5])
```

输出是一个NumPy数组，包含模型对测试集中每个样本的预测。下图总结了原生API的工作流程。

> 原生XGBoost API工作流程的总结，从数据准备到预测。

### Scikit-Learn 封装器 API

虽然原生API提供了全面控制，但XGBoost也包含一个与Scikit-Learn兼容的封装器。如果您已经熟悉Scikit-Learn的`.fit()`和`.predict()`语法，或者您想将XGBoost集成到Scikit-Learn的`Pipeline`或超参数 (parameter) (hyperparameter)搜索工具（如`GridSearchCV`）中，这将非常方便。

主要类包括用于回归的`XGBRegressor`和用于分类的`XGBClassifier`。超参数直接传递给模型的构造函数，就像其他Scikit-Learn估计器一样。

让我们使用`XGBRegressor`来复现我们的回归任务。

```python
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 使用之前相同的数据
X, y = X_train_data, y_train_data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用Scikit-Learn语法实例化模型
# 参数作为构造函数的参数传递
xgb_reg = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=50,
    learning_rate=0.1,  # 请注意这里使用的是 'learning_rate' 而非 'eta'
    max_depth=3,
    random_state=42
)

# 使用熟悉的 .fit() 方法拟合模型
xgb_reg.fit(X_train, y_train)

# 使用 .predict() 方法进行预测
predictions_sklearn = xgb_reg.predict(X_test)

# 评估模型
rmse = np.sqrt(mean_squared_error(y_test, predictions_sklearn))
print(f"使用 Scikit-Learn 封装器的 RMSE: {rmse:.4f}")
```

如您所见，这种方法所需的代码更少，并且与标准的Scikit-Learn工作流程完全一致。您无需手动创建`DMatrix`对象；封装器在内部处理数据转换。对于许多应用，特别是涉及交叉验证和自动化调优的应用，Scikit-Learn封装器是更实际的选择。下一节提供了一个实践机会，可以将这些API应用于解决一个完整的建模问题。

## 参考资料

- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/) — XGBoost Contributors (2024)
  XGBoost库的官方指南，包括其原生API、DMatrix数据结构、超参数和Scikit-Learn包装器。是API使用的主要资源。
- [XGBoost: A Scalable Tree Boosting System](https://dl.acm.org/doi/10.1145/2939672.2939785) — Tianqi Chen and Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD); Publisher: ACM; Pages: 785-794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  介绍XGBoost算法及其设计原则的基础论文，这些原则决定了其API的架构。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://books.google.com/books?id=f_8OEAAAQBAJ) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本广泛使用的实用指南，包含梯度提升和XGBoost的章节，在更广泛的机器学习背景下展示了其使用原生和Scikit-Learn API的应用。（第四版）

---

[上一节](05-%E5%AE%89%E8%A3%85%E4%B8%8E%E8%AE%BE%E7%BD%AEXGBoost.md) · [下一节](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%AD%E7%BB%83XGBoost%E6%A8%A1%E5%9E%8B.md)

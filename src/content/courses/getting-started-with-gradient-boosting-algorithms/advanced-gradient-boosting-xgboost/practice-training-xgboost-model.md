---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "advanced-gradient-boosting-xgboost"
lesson: "practice-training-xgboost-model"
sourceId: 7597
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-4-advanced-gradient-boosting-xgboost/practice-training-xgboost-model"
title: "动手实践：训练XGBoost模型"
description: "一个编程实践，您将使用XGBoost库在数据集上训练模型，评估其表现，并查看其特征。"
order: 7
plots: ["plots/7597-0.json"]
sourceHash: "50bd58a62932715c975620ddf0df8f74b9f3024f644a5775ce43913dc64e5264"
sourceCorrections: []
---

您将使用核心Python API训练您的第一个XGBoost模型，从而实现其功能。内容包括加载数据、监督训练、进行预测以及结果解读。此次练习将应用到`DMatrix`数据结构和目标函数使用等主要概念。

我们将使用加州房价数据集，这是一个经典的回归问题，目标是根据多个特征预测加州某区域的房屋中位数价值。

### 环境与数据准备

首先，请确保您已安装所需的库。您需要`xgboost`、用于数据集和评估指标的`scikit-learn`，以及用于数据操作的`pandas`。

我们先加载数据集并将其分为训练集和测试集。这是一个标准步骤，用于将模型训练数据与评估模型在未见过示例上表现的数据分开。

```python
import xgboost as xgb
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# 加载数据集
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

# 创建训练集和测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("训练集形状:", X_train.shape)
print("测试集形状:", X_test.shape)
```

### 将数据转换为DMatrix格式

如前所述，XGBoost有其自己的优化数据结构，名为`DMatrix`。它在内存使用和训练速度方面都非常高效。在进行训练之前，我们需要将pandas DataFrame和Series转换为这种格式。

```python
# 将数据集转换为DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

print("已为训练集和测试集创建DMatrix对象。")
```

### 定义参数 (parameter)和训练模型

数据准备就绪后，下一步是定义模型的超参数 (hyperparameter)。这些参数在一个Python字典中设置。对于这个初始模型，我们将定义一个目标函数、一个评估指标，以及一些控制学习过程和树结构的参数。

- `objective`：`'reg:squarederror'`表明我们正在执行回归任务，并希望使实际值与预测值之间的平方差最小化。
- `eval_metric`：`'rmse'`（均方根误差）将用于在训练期间监督模型在评估集上的表现。
- `eta`：这是学习率，用于调整每棵树的贡献。较低的值会使提升过程更加保守。
- `max_depth`：这控制每棵决策树的最大深度，有助于防止过拟合 (overfitting)。

我们还将创建一个观察列表（watchlist），它是一个`DMatrix`对象列表。模型将在每个提升轮次后报告其在这些数据集上的表现，使我们能实时观察其学习情况。

```python
# 指定模型参数
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'eta': 0.1,
    'max_depth': 4,
    'seed': 42
}

# 指定提升轮次数量
num_boost_round = 100

# 创建观察列表以监督表现
watchlist = [(dtrain, 'train'), (dtest, 'test')]

# 训练模型
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=watchlist,
    verbose_eval=10  # 每10轮打印评估结果
)
```

当您运行此代码时，XGBoost将每10轮打印训练集和测试集的RMSE。您应该会看到两个集合的RMSE都在减小。如果测试集的RMSE开始增加而训练集的RMSE持续减小，这是过拟合的迹象。使用观察列表是确定最佳提升轮次数量的有效方法。

### 进行预测和评估表现

模型训练完成后，我们可以用它对测试数据进行预测。`predict`方法接受`DMatrix`作为输入，并返回一个预测数组。然后，我们可以使用与训练期间监督的相同指标（RMSE）将这些预测与实际值（`y_test`）进行比较。

```python
# 对测试集进行预测
preds = bst.predict(dtest)

# 评估模型
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"测试集上的最终RMSE: {rmse:.4f}")
```

所得的RMSE提供了一个数值，用于衡量模型的预测误差。例如，RMSE为0.5意味着模型的预测平均误差约为$500,000，因为目标变量的单位是$100,000。

### 通过特征重要性解释模型

XGBoost这类基于树的模型的一个主要优点是其可解释性。我们可以轻松查看模型在进行预测时认为哪些特征最重要。XGBoost提供了一种简单的方法来获取这些重要性分数，甚至可以将其绘制出来。

默认的重要性类型`'weight'`衡量一个特征在树中出现的次数。

```python
# 获取特征重要性分数
importance_scores = bst.get_score(importance_type='weight')
print(importance_scores)

# 使用XGBoost内置函数绘制特征重要性
# 需要安装matplotlib
# xgb.plot_importance(bst)
# plt.show()
```

可视化特征重要性使得理解模型行为变得容易得多。下面的图表显示了我们训练模型的特征重要性分数，表明哪些房屋特征对其预测影响最大。



![XGBoost 特征重要性](plots/7597-0.json)



> F-分数表示每个特征在决策树中用于创建分裂的次数。值越高表示越重要。在此例中，中位数收入（MedInc）是使用最频繁的特征。

您现在已经成功训练、评估并解释了一个XGBoost模型。此工作流程构成了几乎所有梯度提升任务的基础。在接下来的章节中，我们将了解其他强大的库，并学习如何系统地调整模型的超参数 (parameter) (hyperparameter)以获得更好的表现。

## 参考资料

- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/) — XGBoost Contributors (Current)
  XGBoost Python API、数据结构和训练过程的全面指南。
- [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785) — Tianqi Chen and Carlos Guestrin (2016)
  Journal: KDD '16: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 785–794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  描述XGBoost算法设计与优化技术的原始学术论文。
- [Applied Predictive Modeling](https://link.springer.com/book/10.1007/978-1-4614-6849-3) — Max Kuhn and Kjell Johnson (2013)
  Publisher: Springer; DOI: [10.1007/978-1-4614-6849-3](https://doi.org/10.1007/978-1-4614-6849-3)
  一本提供预测建模实用背景的书籍，包括梯度提升等集成方法。

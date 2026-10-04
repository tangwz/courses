---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "hyperparameter-tuning-and-optimization"
lesson: "practice-optimizing-gbm-model"
sourceId: 7611
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-6-hyperparameter-tuning-and-optimization/practice-optimizing-gbm-model"
title: "动手实践：优化梯度提升模型"
description: "一次全面的实践课程，你将应用超参数调整技术于XGBoost或LightGBM模型，以提升其预测准确性。"
order: 8
plots: ["plots/7611-0.json"]
sourceHash: "63b81b97e1aacdd2a3bb4d57d9a1b682fcd2a6641c903b640d19f28828ffc78e"
sourceCorrections: []
---

这个实践应用演示了如何使用系统调整方法来优化用于回归任务的XGBoost模型。它首先使用默认模型建立基准性能。然后，使用`RandomizedSearchCV`高效地探索各种超参数 (parameter) (hyperparameter)值。最后，评估调整后的模型以衡量其改进程度。

我们的目标是提高模型在加州住房数据集上的预测准确性。这是一个典型的回归问题，其目的是预测加州各区域的房屋中位数价格。

### 环境搭建

首先，导入所需的库并加载数据集。我们将使用`XGBoost`作为模型，并使用`Scikit-Learn`进行数据处理和调整工具。

```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

# 加载数据集
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

# 创建训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("训练数据形状：", X_train.shape)
print("测试数据形状：", X_test.shape)
```

### 第一步：建立基准模型

在开始调整之前，了解我们当前的性能表现是必不可少的。我们将使用默认参数 (parameter)训练一个`XGBRegressor`模型，并使用均方误差（MSE）评估其在测试集上的性能。较低的MSE表示模型拟合得更好。

```python
# 使用默认参数初始化XGBRegressor
xgb_baseline = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# 训练模型
xgb_baseline.fit(X_train, y_train)

# 在测试集上进行预测
y_pred_baseline = xgb_baseline.predict(X_test)

# 计算并打印基准MSE
mse_baseline = mean_squared_error(y_test, y_pred_baseline)
print(f"基准模型MSE: {mse_baseline:.4f}")
```

这个基准分数是我们希望超越的目标。任何MSE低于此值的模型都表示性能有所提升。

### 第二步：配置并运行随机搜索

当超参数 (parameter) (hyperparameter)空间很大时，网格搜索的计算成本会很高。`RandomizedSearchCV`是一种更有效的方法，它从指定的分布中采样固定数量的参数组合。这种方法使我们能够测试各种值，而无需尝试所有可能的组合。

让我们为前面讨论的超参数定义一个搜索空间：

- `n_estimators`：提升轮次数量。
- `learning_rate`：步长收缩率。
- `max_depth`：树的最大深度。
- `subsample`：每棵树随机采样的观测值比例。
- `colsample_bytree`：每棵树随机采样的特征（列）比例。
- `gamma`：进行进一步划分所需的最小损失减少量。

```python
# 定义随机搜索的超参数网格
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3]
}

# 初始化XGBRegressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# 初始化RandomizedSearchCV
# n_iter 控制要尝试多少种不同的组合。
# cv 是交叉验证折叠的数量。
# n_jobs=-1 使用所有可用的CPU核心来加快处理速度。
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# 将RandomizedSearchCV拟合到训练数据
random_search.fit(X_train, y_train)
```

搜索完成后，`RandomizedSearchCV`会存储它找到的最佳超参数组合。

```python
# 打印找到的最佳参数
print("随机搜索找到的最佳超参数：")
print(random_search.best_params_)
```

### 第三步：训练和评估优化后的模型

现在，我们使用搜索确定的最佳参数 (parameter)来训练一个新的最终模型。通常的做法是在整个训练数据集上训练此模型，以便它能从最多数据中学习。最后，我们在用于基准测试的相同保留测试集上评估其性能。

```python
# 从搜索结果中获取最佳估计器
best_xgb_model = random_search.best_estimator_

# 在测试集上进行预测
y_pred_tuned = best_xgb_model.predict(X_test)

# 计算并打印调整后模型的MSE
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
print(f"调整后模型MSE: {mse_tuned:.4f}")
print(f"相对于基准的改进: {mse_baseline - mse_tuned:.4f}")
```

你应该会看到调整后模型的MSE低于基准模型的MSE，这表明我们的超参数 (hyperparameter)调整过程成功提升了模型的预测性能。

### 性能提升可视化

一个简单的图表可以直观地展示我们优化工作的影响。让我们比较基准模型和调整后模型的均方误差。



![模型性能比较：基准模型 vs. 调整后模型](plots/7611-0.json)



> 均方误差的减少表明了系统性超参数 (parameter) (hyperparameter)调整的价值。

### 总结与展望

在此动手练习中，你遵循了一个结构化过程来优化梯度提升模型。你首先建立了性能基准，然后使用`RandomizedSearchCV`高效地搜寻超参数 (parameter) (hyperparameter)空间，最后通过优化设置训练并评估了最终模型。

这种建立基准、寻找更优参数并评估结果的迭代过程，是应用机器学习 (machine learning)中的一个基本工作流程。虽然我们使用了`RandomizedSearchCV`，但你还可以利用结果来指导更集中的`GridSearchCV`，围绕最有前景的参数值进行进一步的细化搜索。

对于更复杂的优化任务，你可以尝试贝叶斯优化等高级技术，它通常能以更少的迭代次数找到更好的超参数。像Hyperopt和Optuna这样的库提供了实现这些高级策略的强大工具。

## 参考资料

- [XGBoost: A Scalable Tree Boosting System](https://www.semanticscholar.org/paper/XGBoost%3A-A-Scalable-Tree-Boosting-System-Chen-Guestrin/f36611a2f60249c6931758656157f461876527b1) — Tianqi Chen and Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 785-794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  介绍XGBoost的开创性论文，详细阐述了其算法、优化和性能特点。
- [sklearn.model_selection.RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) — scikit-learn developers (2024)
  `RandomizedSearchCV`的官方文档，提供了在scikit-learn框架内进行高效超参数调优的详细用法、参数和示例。
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/) — XGBoost developers (2024)
  XGBoost库的官方综合指南，涵盖了安装、API使用以及模型参数和超参数的详细解释。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://books.google.com/books?id=0iKIEAAAQBAJ) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本高度实用的书籍，其中有一章专门介绍超参数调优，以实践操作的方式演示了使用Scikit-Learn进行`RandomizedSearchCV`等技术。

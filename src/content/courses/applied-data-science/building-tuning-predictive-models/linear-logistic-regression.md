---
course: "applied-data-science"
chapter: "building-tuning-predictive-models"
lesson: "linear-logistic-regression"
sourceId: 1866
sourceUrl: "https://apxml.com/zh/courses/applied-data-science/chapter-3-building-tuning-predictive-models/linear-logistic-regression"
title: "实施线性回归与逻辑回归"
description: "使用scikit-learn实际实现线性回归和逻辑回归模型。"
order: 2
plots: ["plots/1866-0.json", "plots/1866-1.json"]
sourceHash: "cfef0c19419339bfb546a558146386c987afe3cbefd6806791b706a4d9192a75"
sourceCorrections: []
---

线性回归和逻辑回归是监督式机器学习 (machine learning)中的基础算法，分别广泛应用于回归和分类任务。它们的简单、易于理解和计算效率高，使其成为许多建模问题的良好起点。我们将使用Python中流行的scikit-learn库来实施这些模型。

## 实施线性回归

线性回归旨在通过将线性方程拟合到观测数据来模拟因变量（目标）与一个或多个自变量（特征）之间的关系。目标是找到通过数据点的最佳拟合直线（或在更高维度中的超平面）。简单线性回归（一个特征）的方程是 $y = \beta_0 + \beta_1 x$，其中 $y$ 是目标，$x$ 是特征，$\beta_0$ 是截距，$\beta_1$ 是特征的系数。对于多个特征，这可以扩展为：


$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n
$$


Scikit-learn在`sklearn.linear_model`模块中提供了`LinearRegression`类。让我们看一个基本实现。

首先，我们需要一些数据。我们可以使用scikit-learn的`make_regression`函数生成合成回归数据。

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression

# 生成合成数据
X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)

# 转换为DataFrame以便处理（可选）
X_df = pd.DataFrame(X, columns=['Feature'])
y_s = pd.Series(y, name='Target')

# 将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
print(f"测试集形状: X={X_test.shape}, y={y_test.shape}")
```

现在，我们实例化`LinearRegression`模型，并使用`fit`方法在我们的训练数据上对其进行训练。

```python
# 初始化线性回归模型
lr_model = LinearRegression()

# 训练模型
lr_model.fit(X_train, y_train)

# 打印学习到的系数
print(f"截距 (beta_0): {lr_model.intercept_:.2f}")
print(f"系数 (beta_1): {lr_model.coef_[0]:.2f}")
```

模型训练完成后，我们可以使用`predict`方法对新的、未见过的数据（我们的测试集）进行预测。

```python
# 对测试集进行预测
y_pred = lr_model.predict(X_test)
```

最后，我们评估模型的性能。回归的常用指标包括均方误差（MSE）和R平方（$R^2$）分数。MSE衡量实际值和预测值之间的平均平方差（越低越好），而$R^2$表示因变量中可由自变量预测的方差比例（越接近1越好）。

```python
# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"均方误差 (MSE): {mse:.2f}")
print(f"R平方 (R2 分数): {r2:.2f}")
```

我们可以可视化结果，以查看回归线与测试数据的拟合情况。



![线性回归拟合](plots/1866-0.json)



> 散点图显示了测试集中的实际数据点和拟合的线性回归线。

## 实施逻辑回归

尽管其名称，逻辑回归用于分类任务，通常是二元分类（预测两种结果之一）。它使用逻辑函数（也称为S型函数）模拟输入属于某个特定类别的概率。S型函数将任何实数值映射到0到1之间的值。

正类（通常表示为1）的概率公式是：


$$
P(y=1|X) = \frac{1}{1 + e^{-z}}
$$


其中 $z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n$。输出 $P(y=1|X)$ 是估计的概率。然后使用一个阈值（通常是0.5）将此概率转换为类别预测（例如，如果概率 > 0.5，则预测为类别1，否则预测为类别0）。

Scikit-learn提供了`LogisticRegression`类，也在`sklearn.linear_model`中。让我们实现它。

我们将首先使用`make_classification`生成合成分类数据。

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification
import plotly.graph_objects as go

# 生成合成分类数据
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, flip_y=0.1, random_state=42)

# 将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
print(f"测试集形状: X={X_test.shape}, y={y_test.shape}")
```

接下来，实例化并训练`LogisticRegression`模型。

```python
# 初始化逻辑回归模型
log_reg_model = LogisticRegression(random_state=42)

# 训练模型
log_reg_model.fit(X_train, y_train)

# 打印学习到的系数和截距
print(f"截距: {log_reg_model.intercept_[0]:.2f}")
print(f"系数: {log_reg_model.coef_[0][0]:.2f}, {log_reg_model.coef_[0][1]:.2f}")
```

对测试集进行预测。`predict`方法直接输出预测的类别标签，而`predict_proba`给出每个类别的概率估计。

```python
# 进行预测
y_pred_class = log_reg_model.predict(X_test)

# 预测概率
y_pred_proba = log_reg_model.predict_proba(X_test)

# 显示前5个预测及其概率
print("前5个预测类别:", y_pred_class[:5])
print("前5个预测概率（类别0，类别1）:\n", y_pred_proba[:5].round(3))
```

使用分类指标评估模型。准确率是一个常见的起点，但混淆矩阵提供更详细的性能视图，显示真阳性、真阴性、假阳性和假阴性。`classification_report`提供精确率、召回率和F1分数，这些将在稍后更详细地讨论。

```python
# 评估模型
accuracy = accuracy_score(y_test, y_pred_class)
conf_matrix = confusion_matrix(y_test, y_pred_class)
class_report = classification_report(y_test, y_pred_class)

print(f"准确率: {accuracy:.2f}")
print("\n混淆矩阵:\n", conf_matrix)
print("\n分类报告:\n", class_report)
```

为了可视化逻辑回归的核心组成部分S型函数，我们可以绘制它。



![S型（逻辑）函数](plots/1866-1.json)



> S型函数将输入的线性组合（$z$）转换为0到1之间的概率值。

## 线性模型中的正则化 (regularization)

线性模型有时会过拟合 (overfitting)，特别是在特征很多或特征之间高度相关时。正则化是一种通过在模型的损失函数 (loss function)中添加惩罚项来防止过拟合的技术。这种惩罚会抑制系数过大的复杂模型。

两种最常见的正则化类型是：

1. **L1 正则化 (Lasso):** 添加一个等于系数绝对值的惩罚项。这可以导致稀疏模型，其中一些特征系数变为零，从而有效地进行特征选择。在scikit-learn中通过用于回归的`Lasso`类实现。
2. **L2 正则化 (Ridge):** 添加一个等于系数平方的惩罚项。这会将系数缩小到接近零，但很少使它们精确地变为零。通过用于回归的`Ridge`类实现。

Scikit-learn中的逻辑回归默认包含正则化（`penalty='l2'`）。你可以更改惩罚类型（`'l1'`、`'elasticnet'`、`'none'`），并使用`C`参数 (parameter)控制其强度（`C`是正则化强度的倒数；`C`值越小表示正则化越强）。

```python
# 示例：使用L1惩罚的逻辑回归
log_reg_l1 = LogisticRegression(penalty='l1', C=0.5, solver='liblinear', random_state=42)
log_reg_l1.fit(X_train, y_train)

print(f"L1 系数: {log_reg_l1.coef_[0][0]:.2f}, {log_reg_l1.coef_[0][1]:.2f}")

# 示例：岭回归（带L2惩罚的线性回归）
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=1.0) # alpha是正则化强度
ridge_model.fit(X_train_reg, y_train_reg) # 假设X_train_reg, y_train_reg用于回归
print(f"岭回归系数: {ridge_model.coef_}")
```

*(注意：岭回归示例需要回归数据`X_train_reg`、`y_train_reg`)*

## 线性回归与逻辑回归总结

**优点：**

- **可解释性：** 特征与目标（或逻辑回归中目标的对数几率）之间的关系通过系数相对容易理解。
- **计算效率：** 训练和预测通常很快，即使在大型数据集上也是如此。
- **良好基线：** 它们通常提供可靠的基线性能，可与更复杂的模型进行比较。
- **无需超参数 (parameter) (hyperparameter)调整（对于基本版本）：** 标准线性回归没有超参数可调，尽管正则化 (regularization)版本有。

**缺点：**

- **线性假设：** 它们假设特征与目标（或对数几率）之间存在线性关系。如果潜在关系高度非线性，它们可能表现不佳。
- **对异常值的敏感性：** 标准线性回归可能对数据中的异常值敏感。
- **特征工程：** 可能需要仔细的特征工程（例如，创建交互项或多项式特征）来捕获非线性模式。

线性回归和逻辑回归是数据科学家工具包中的基本工具。掌握它们的实现并理解它们的特点，为接下来讨论的更复杂的基于树和集成方法奠定了一个坚实的基础。

## 参考资料

- [sklearn.linear_model: Generalized Linear Models](https://scikit-learn.org/stable/modules/linear_model.html) — scikit-learn developers (2023)
  此官方文档详细介绍了scikit-learn库中线性模型的实现，包括线性回归、逻辑回归、岭回归和Lasso回归，并提供了API参考和示例。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  这是一本涵盖线性回归、逻辑回归和正则化技术统计和理论基础的经典教材。
- [An Introduction to Statistical Learning: with Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2013)
  Publisher: Springer
  这是一本易于理解的教材，介绍了线性回归和逻辑回归，涵盖了基本概念、假设和实际应用。

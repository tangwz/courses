---
course: "getting-started-with-scikit-learn"
chapter: "supervised-learning-regression"
lesson: "interpreting-model-coefficients"
sourceId: 1846
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-2-supervised-learning-regression/interpreting-model-coefficients"
title: "解释模型系数"
description: "理解线性回归模型学得的系数的含义。"
order: 4
plots: ["plots/1846-0.json"]
sourceHash: "d53cc7af9e1a1df9a4b5dcda465aa77f7c159ed7661df6bf3166145389666413"
sourceCorrections: []
---

当使用 Scikit-learn 的 `fit` 方法训练 `LinearRegression` 模型时，模型会学得定义最佳拟合线（或更高维度中的超平面）的最佳参数 (parameter)（系数和截距），通过训练数据。理解这些参数对于理解特征与目标变量之间的关系有很大帮助。

### 线性回归方程

回顾具有 $n$ 个特征的线性回归模型的一般方程：


$$
\hat{y} = b_0 + b_1 x_1 + b_2 x_2 + \dots + b_n x_n
$$


式中：

- $\hat{y}$ 是目标变量的预测值。
- $x_1, x_2, \dots, x_n$ 是输入特征的值。
- $b_0$ 是**截距**（也称为偏置 (bias)项）。它表示当所有特征值（$x_1, \dots, x_n$）为零时 $\hat{y}$ 的预测值。
- $b_1, b_2, \dots, b_n$ 是与每个特征 $x_1, x_2, \dots, x_n$ 相关的**系数**（也称为权重 (weight)或斜率）。每个系数 $b_i$ 表示当其他所有特征保持不变时，对应特征 $x_i$ 每增加一个单位，目标变量 $\hat{y}$ 的预期变化量。

### 在 Scikit-learn 中获取截距和系数

在 Scikit-learn 中拟合 `LinearRegression` 模型后，这些学得的参数 (parameter)存储在模型对象的特定属性中：

- **截距 ($b_0$)**：存储在 `intercept_` 属性中。它将是一个浮点数。
- **系数 ($b_1, \dots, b_n$)**：存储在 `coef_` 属性中。它将是一个 NumPy 数组，其中每个元素对应一个特征的系数。系数的顺序与训练期间使用的特征矩阵 `X` 中的列顺序一致。

假设您有一个已拟合的模型，名为 `lr_model`：

```python
# 假设 X_train, y_train 已定义，且 lr_model 是一个已拟合的 LinearRegression 实例
# 示例：
# from sklearn.linear_model import LinearRegression
# lr_model = LinearRegression()
# lr_model.fit(X_train, y_train) # 首先拟合模型

# 获取截距
intercept = lr_model.intercept_
print(f"截距 (b0): {intercept:.4f}")

# 获取系数
coefficients = lr_model.coef_
print(f"系数 (b1, b2, ...): {coefficients}")

# 如果您有特征名称（例如，来自 Pandas DataFrame）
# feature_names = ['feature1', 'feature2', ...]
# for feature, coef in zip(feature_names, coefficients):
#    print(f"  特征 {feature} 的系数: {coef:.4f}")
```

### 值的解释

- **截距：** `intercept_` 的值表示当所有输入特征都为零时目标变量的预测值。此值的实际意义很大程度上取决于您问题的背景。如果所有特征为零在您的数据集中是一个有意义的情况（例如，特征表示可能为零的计数或数量），那么截距就有直接的解释。如果零超出了您特征的实际范围（例如，根据平方英尺预测房价，而零平方英尺是不可能的），那么截距可能更多是正确放置回归线所需的数学结果，而不是具有独立的实际意义。
- **系数：** `coef_` 数组中的每个值都量化 (quantization)了特定特征与目标变量之间的关系。例如，如果 `feature1` 的系数是 `15.7`，这意味着：

  > “在其他所有特征保持不变的情况下，`feature1` 每增加一个单位，目标变量预计会增加 15.7 个单位。”

  反之，如果一个系数为负，例如 `feature2` 的系数为 `-3.2`，这意味着：

  > “在其他所有特征保持不变的情况下，`feature2` 每增加一个单位，目标变量预计会减少 3.2 个单位。”

短语“在其他所有特征保持不变的情况下”很重要。线性回归假设一个特征对目标的影响独立于其他特征的值，系数反映了这种独立的影响。



![简单线性回归：解释](plots/1846-0.json)



> 简单线性回归线 $y = 10 + 2x$ 的可视化。截距 ($b_0=10$) 是当线穿过 y 轴（$x=0$）时 $y$ 的值。系数或斜率 ($b_1=2$) 表示 $x$ 每增加一个单位， $y$ 增加 2 个单位。

### 关于特征缩放的说明

比较系数的大小以确定哪个特征“最重要”是很诱人的。但是，如果特征处于不同的尺度上，这往往会产生误导。

考虑根据 `years_experience`（范围从 0 到 30）和 `projects_completed`（范围从 0 到 500）预测薪资。`years_experience` 的系数为 2000，`projects_completed` 的系数为 50，这并不自动意味着经验的影响大得多。年限的一个单位变化比项目的一个单位变化所占比例大得多。

为了使系数在重要性上可以直接比较，您通常需要先对特征进行缩放（例如，使用 Scikit-learn 中的 `StandardScaler` 或 `MinMaxScaler`，我们将在第 4 章中介绍）。当特征被缩放到相似的范围（例如标准差为 1）时，较大的系数值通常表示对目标变量的影响更大，这对应于该特征的标准化变化。在不缩放的情况下，应根据其各自特征的单位来解释系数。

## 参考资料

- [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) — Scikit-learn Developers (2023)
  Scikit-learn 的 LinearRegression 类官方文档，详细说明其属性和获取系数与截距的方法。
- [Applied Linear Statistical Models](https://users.stat.ufl.edu/~presnell/Courses/phc6095/Applied_Linear_Statistical_Models_5th_Edition.pdf) — Michael H. Kutner, Christopher J. Nachtsheim, John Neter, William Li (2005)
  Publisher: McGraw-Hill/Irwin; Pages: Chapter 6: Multiple Regression
  一本关于线性统计模型的综合性教材，对回归系数的理论和实际解释提供了深入见解。

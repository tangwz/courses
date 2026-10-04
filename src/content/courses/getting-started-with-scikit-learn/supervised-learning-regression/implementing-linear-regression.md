---
course: "getting-started-with-scikit-learn"
chapter: "supervised-learning-regression"
lesson: "implementing-linear-regression"
sourceId: 1845
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-2-supervised-learning-regression/implementing-linear-regression"
title: "使用Scikit-learn实现线性回归"
description: "在Scikit-learn中使用`LinearRegression`估计器拟合模型。"
order: 3
plots: ["plots/1845-0.json"]
sourceHash: "9cfd8a7ee003d2a8dda68dd894ebaa8f8ffd65904c0e7ca75a4d718d52dbb6ef"
sourceCorrections: []
---

使用Scikit-learn实现线性回归涉及其统一且直接的接口。Scikit-learn提供了各种机器学习 (machine learning)模型，包括线性回归，可以轻松应用。实现此功能的主要工具是`sklearn.linear_model`模块中的`LinearRegression`估计器。

### Scikit-learn估计器API

Scikit-learn的设计围绕着“估计器”的理念。估计器是任何从数据中学习的对象；它可以是分类、回归或聚类算法，也可以是提取有用特征的转换器。所有估计器都遵循统一的模式：

1. **导入：** 导入您需要的特定估计器类。
2. **实例化：** 创建估计器类的实例，并可能配置其超参数 (parameter) (hyperparameter)（尽管`LinearRegression`的超参数不多）。
3. **准备数据：** 将数据组织成特征矩阵`X`和目标向量 (vector)`y`。`X`通常是二维NumPy数组或Pandas DataFrame（形状：`[n_样本数, n_特征数]`），而`y`是一维NumPy数组或Pandas Series（形状：`[n_样本数]`）。
4. **拟合：** 使用`.fit(X, y)`方法在您的数据上训练估计器。这是模型从数据中学习的步骤。对于线性回归，`fit`通过最小化数据集中观测目标与线性近似预测目标之间的残差平方和来计算最优系数（$\beta_1, ..., \beta_p$）和截距（$\beta_0$）。
5. **预测：** 一旦拟合完成，您可以使用`.predict(X_new)`方法对新的、未见过的数据进行预测。`X_new`应与训练数据`X`具有相同数量的特征。

### 实现`LinearRegression`

我们来看一个简单的例子。我们将生成一些大致遵循线性模式的合成数据，然后用`LinearRegression`模型对其进行拟合。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. 生成一些样本数据
# 使结果可重现
np.random.seed(42)
X = 2 * np.random.rand(100, 1) # 特征（Scikit-learn要求为二维）
y = 4 + 3 * X + np.random.randn(100, 1) # 带有噪声的目标变量

# 2. 导入估计器
# 上面已完成：from sklearn.linear_model import LinearRegression

# 3. 实例化估计器
model = LinearRegression()

# 4. 将模型拟合到数据
# Scikit-learn需要X（特征）和y（目标）
model.fit(X, y)

# 5. 检查学习到的参数
# 截距（beta_0）存储在.intercept_中
# 系数（beta_1, ..., beta_p）存储在.coef_中
print(f"截距 (beta_0): {model.intercept_[0]:.4f}")
print(f"系数 (beta_1): {model.coef_[0][0]:.4f}")

# 6. 对新数据进行预测
# 让我们预测X = 0和X = 2的值
X_new = np.array([[0], [2]]) # 新数据点必须是二维数组
y_pred = model.predict(X_new)

print(f"\n对 X_new = [[0], [2]] 的预测：")
print(y_pred)

# 可选：可视化结果
plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.7, label='原始数据')
plt.plot(X_new, y_pred, "r-", linewidth=2, label='拟合回归线')
plt.xlabel("特征 (X)")
plt.ylabel("目标 (y)")
plt.title("线性回归拟合")
plt.legend()
plt.grid(True)
plt.show()
```

运行此代码将：

1. 创建`y`近似为$4 + 3x$的样本数据。
2. 实例化`LinearRegression`模型。
3. 调用`model.fit(X, y)`。在此步骤中，Scikit-learn应用普通最小二乘法，通过最小化实际`y`值与直线预测值（$\hat{y}$）之间的平方差之和，寻找最能拟合数据点的直线。
4. 打印学习到的截距（`model.intercept_`）和系数（`model.coef_`）。您应该会看到接近用于生成数据的原始参数 (parameter)（4和3）的值。轻微的差异是由于我们添加的随机噪声。
5. 使用`model.predict()`计算两个新输入值（0和2）的预测`y`值。
6. 生成一个图表，显示原始数据点和模型学习到的回归线。



![线性回归拟合](plots/1845-0.json)



> 生成数据点的散点图以及Scikit-learn拟合的线性回归线。

### 处理多个特征

上述过程对于多元线性回归（即您有多个输入特征的情况）以完全相同的方式运行。唯一的区别是您的`X`矩阵将有多于一列（每个特征一列）。Scikit-learn的`LinearRegression`会自动处理此情况。

```python
# 带有2个特征的例子
X_multi = 2 * np.random.rand(100, 2) # 现在X有2列
# y = 4 + 3*X_1 + 5*X_2 + 噪声
y_multi = 4 + 3 * X_multi[:, 0] + 5 * X_multi[:, 1] + np.random.randn(100)
y_multi = y_multi.reshape(-1, 1) # 如果需要，将y变为列向量

multi_model = LinearRegression()
multi_model.fit(X_multi, y_multi)

print(f"\n多元回归：")
print(f"截距 (beta_0): {multi_model.intercept_[0]:.4f}")
print(f"系数 (beta_1, beta_2): {multi_model.coef_[0]}")

# 预测需要具有2个特征的输入
X_multi_new = np.array([[0, 0], [2, 3]]) # 预测 [X1=0, X2=0] 和 [X1=2, X2=3]
y_multi_pred = multi_model.predict(X_multi_new)
print(f"\n对 X_multi_new 的预测：")
print(y_multi_pred)
```

`.coef_`属性现在将包含一个具有多个值的数组，`X_multi`中每个特征列对应一个系数。解释方式类似：每个系数表示当对应特征变化一个单位时，目标变量`y`的变化量，前提是所有其他特征保持不变。

这种统一的API使得应用线性回归变得简单，无论您是有一个特征还是多个特征。模型拟合完成后，接下来重要的步骤包括理解学习到的系数的含义，以及评估模型的实际表现，这些内容我们将在后续章节中介绍。

## 参考资料

- [1.1. Linear Models](https://scikit-learn.org/stable/modules/linear_model.html) — scikit-learn developers (2023)
  Scikit-learn官方用户指南中线性模型部分，解释了包括普通最小二乘法（OLS）和通用API在内的线性模型。
- [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) — scikit-learn developers (2023)
  LinearRegression 类的详细API文档，介绍了其参数和属性。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用教材，提供了使用Scikit-learn实现包括线性回归在内的机器学习模型的逐步指导。
- [An Introduction to Statistical Learning with Applications in Python](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor (2023)
  Publisher: Springer
  一本基础教材，涵盖了线性回归和其他机器学习方法的统计理论，并提供了Python示例。

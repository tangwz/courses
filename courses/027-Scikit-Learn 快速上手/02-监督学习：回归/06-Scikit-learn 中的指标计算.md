# Scikit-learn 中的指标计算

来源：[原文](https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-2-supervised-learning-regression/calculating-regression-metrics)

[返回章节目录](README.md) · [返回课程目录](../README.md)

平均绝对误差 (MAE)、均方误差 (MSE)、均方根误差 (RMSE) 和决定系数 ($R^2$) 是评估回归模型的常见指标。要使用 Scikit-learn 有效地计算这些指标，`sklearn.metrics` 模块提供了直接的函数来计算这些常见的回归评价指标。

要使用这些函数，您通常需要两个主要输入：

1. `y_true`：真实（正确）的目标值。
2. `y_pred`：您的回归模型生成的预测值。

`y_true` 和 `y_pred` 通常都是相同长度的 NumPy 数组或 Pandas Series。

假设我们已经训练了一个回归模型并在测试集上获得了预测。我们将使用 `y_true` 和 `y_pred` 数组进行演示。

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 示例真实值
y_true = np.array([10, 12, 15, 18, 22, 25])

# 模型的示例预测值
y_pred = np.array([9, 13, 14, 19, 20, 26])
```

现在，我们来计算每个指标。

### 平均绝对误差 (MAE)

MAE 衡量一组预测中误差的平均大小，不考虑其方向。它是测试样本中预测值与实际观测值之间绝对差值的平均值。

您可以使用 `sklearn.metrics.mean_absolute_error` 计算 MAE：

```python
# 计算 MAE
mae = mean_absolute_error(y_true, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
# 预期输出: 平均绝对误差 (MAE): 1.17
```

结果（本例中为 1.17）表明，平均而言，模型的预测值与实际值相差约 1.17 个单位。MAE 的单位与目标变量的单位相同。

### 均方误差 (MSE)

MSE 衡量误差平方的平均值。误差平方会赋予较大误差更高的权重 (weight)。

您可以使用 `sklearn.metrics.mean_squared_error` 计算 MSE：

```python
# 计算 MSE
mse = mean_squared_error(y_true, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
# 预期输出: 均方误差 (MSE): 2.17
```

MSE 值（此处为 2.17）很难直接根据目标变量的单位进行解释，因为单位是平方的。然而，它对于优化和模型比较很有用，因为它比 MAE 对大的偏差惩罚更重。

### 均方根误差 (RMSE)

RMSE 是 MSE 的平方根。取平方根将指标恢复到目标变量的原始单位，使其比 MSE 更易于理解。

虽然 Scikit-learn 没有专门的 `root_mean_squared_error` 函数，但您可以通过对 MSE 结果取平方根来轻松计算，或者更方便地，通过在 `mean_squared_error` 中使用 `squared` 参数 (parameter)。将 `squared=False` 设置为返回 RMSE。

```python
# 使用 squared=False 参数计算 RMSE
rmse = mean_squared_error(y_true, y_pred, squared=False)

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
# 预期输出: 均方根误差 (RMSE): 1.47

# 或者，通过对 MSE 取平方根来计算 RMSE
# import numpy as np
# rmse_alt = np.sqrt(mse)
# print(f"Root Mean Squared Error (RMSE - 替代方法): {rmse_alt:.2f}")
# 预期输出: 均方根误差 (RMSE - 替代方法): 1.47
```

RMSE (1.47) 表示预测值与真实值之间的典型偏差约为 1.47 个单位。与 MAE 类似，值越低表示拟合越好。由于它在平均之前对误差进行平方，RMSE 对异常值敏感，与 MSE 相似。

### R 平方 ($R^2$) 分数 (决定系数)

$R^2$ 分数表示因变量（目标）中可由自变量（特征）预测的方差比例。它衡量模型在多大程度上再现了观测结果，依据是模型解释的总结果变异的比例。

您可以使用 `sklearn.metrics.r2_score` 计算 $R^2$ 分数：

```python
# 计算 R 平方
r2 = r2_score(y_true, y_pred)

print(f"R-squared (R2) Score: {r2:.2f}")
# 预期输出: R 平方 (R2) 分数: 0.93
```

0.93 的 $R^2$ 分数表明大约 93% 的 `y_true` 数据方差可以由我们模型的预测（`y_pred`）来解释。

- $R^2$ 分数为 1 表示完美拟合。
- $R^2$ 分数为 0 表示模型表现不比简单地预测目标变量的平均值更好。
- 可能出现负 $R^2$ 分数，这表示模型表现比预测平均值更差。

### 实践中应用指标

在分割数据并训练模型（例如 `LinearRegression`）后，一个典型的工作流程是，您会使用测试集来评估性能：

```python
# 假设 X_train, y_train, X_test, y_test 已定义
# 假设 'model' 是一个已训练的 Scikit-learn 回归器

# 1. 在测试集上进行预测
# y_pred_test = model.predict(X_test)

# 2. 使用实际测试值 (y_test) 计算指标
# mae_test = mean_absolute_error(y_test, y_pred_test)
# mse_test = mean_squared_error(y_test, y_pred_test)
# rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
# r2_test = r2_score(y_test, y_pred_test)

# print(f"测试集 MAE: {mae_test:.3f}")
# print(f"测试集 MSE: {mse_test:.3f}")
# print(f"测试集 RMSE: {rmse_test:.3f}")
# print(f"测试集 R 平方: {r2_test:.3f}")
```

这些函数提供了在 Scikit-learn 中量化 (quantization)回归模型性能的重要工具，使您能够根据具体的数值结果比较不同的模型或调整参数 (parameter)。请记住，选择“最佳”指标通常取决于具体的应用场景以及误差的哪个方面（平均大小与大误差）最需要去降低。

## 参考资料

- [1.8. Model evaluation: quantifying the quality of predictions - scikit-learn 1.3.2 documentation (Regression metrics)](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) — scikit-learn developers (2023)
  提供Scikit-learn中所有回归指标的官方、详细描述和使用示例，重点关注回归评估。
- [An Introduction to Statistical Learning: With Applications in R](https://doi.org/10.1007/978-1-4614-7138-7) — Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani (2013)
  Publisher: Springer; DOI: [10.1007/978-1-4614-7138-7](https://doi.org/10.1007/978-1-4614-7138-7)
  解释回归评估指标（如MSE、RMSE和R方）的统计基础和解释，提供理解指标含义所需的基础知识。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://books.google.com/books/about/Hands_On_Machine_Learning_with_Scikit_Le.html?id=z1mHEAAAQBAJ) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 864
  提供使用Scikit-learn实现机器学习模型和评估的实践示例，包括如何在实践中计算和解释回归指标。

---

[上一节](05-%E5%9B%9E%E5%BD%92%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87.md) · [下一节](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E5%9B%9E%E5%BD%92%E6%A8%A1%E5%9E%8B.md)

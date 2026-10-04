---
course: "getting-started-with-scikit-learn"
chapter: "supervised-learning-regression"
lesson: "practice-building-regression-model"
sourceId: 1851
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-2-supervised-learning-regression/practice-building-regression-model"
title: "动手实践：构建回归模型"
description: "在样本数据集上训练、预测和评估线性回归模型。"
order: 7
plots: ["plots/1851-0.json"]
sourceHash: "d64abb33d2b26783ae06495529d8192ea83556eafd6e7cfbeb8771d97e7476de"
sourceCorrections: []
---

使用Scikit-learn构建、训练和评估简单线性回归模型的完整过程将被演示。这将涉及使用真实数据集来预测一个连续目标变量。

### 准备工作：导入和数据

首先，我们需要导入所需的库和模块。我们将需要Pandas进行可能的数据处理（尽管Scikit-learn数据集通常返回NumPy数组或Bunch对象），以及Scikit-learn用于数据集、模型、分割函数和评估指标。

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go # For visualization
```

我们将使用加利福尼亚住房数据集，这是一个在Scikit-learn中直接提供的常用回归任务数据集。目标是根据普查数据中的各种特征，预测加利福尼亚各区域的房屋中位价。

```python
# Load the dataset
california = fetch_california_housing(as_frame=True)
X = california.data
y = california.target

# Display some information about the data
print("特征 (X):")
print(X.head())
print("\n目标 (y) - 房屋中位价:")
print(y.head())
print("\n数据集描述:")
print(california.DESCR[:500] + "...") # 打印描述的前500个字符
```

输出显示了我们特征（如收入中位数、房龄、平均房间数）和目标变量（房屋中位价）的前几行。描述提供了特征和预测任务的背景信息。

### 分割数据以进行可靠评估

在训练之前，标准做法是将数据集分成两部分：训练集和测试集。模型从训练集中学习模式。然后，我们在未见过的测试集上评估其性能，以获得模型在新数据上泛化能力的无偏估计。我们为此使用了Scikit-learn的`train_test_split`函数。

```python
# 将数据分割为训练集（80%）和测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train 的形状: {X_train.shape}")
print(f"X_test 的形状: {X_test.shape}")
print(f"y_train 的形状: {y_train.shape}")
print(f"y_test 的形状: {y_test.shape}")
```

我们设置`test_size=0.2`来分配20%的数据用于测试，并使用`random_state`以确保可重现性，从而保证每次运行代码时都能获得相同的分割结果。

### 训练线性回归模型

现在我们实例化`LinearRegression`模型，并使用我们的训练数据（`X_train`和`y_train`）对其进行拟合。`fit`方法是模型学习特征与目标变量之间关系的地方，它会计算线性方程的最佳系数。

```python
# 创建一个线性回归模型实例
model = LinearRegression()

# 使用训练数据训练模型
model.fit(X_train, y_train)

print("模型训练完成。")
print(f"截距: {model.intercept_}")
print(f"系数: {model.coef_}")
```

拟合后，模型学习了截距和每个特征的系数。这些表示在训练数据中确定的线性关系。

### 进行预测

模型训练完成后，我们现在可以使用它在新数据（未见过的数据）上进行预测。我们对测试集（`X_test`）使用`predict`方法。

```python
# 在测试集上进行预测
y_pred = model.predict(X_test)

# 显示前5个预测值和实际值
print("前5个预测值:", y_pred[:5])
print("前5个实际值:", y_test[:5].values)
```

模型输出一个基于`X_test`中特征的预测房屋中位价数组。我们可以将这些预测值（`y_pred`）与实际已知值（`y_test`）进行比较。

### 评估模型性能

这些预测效果如何？我们需要量化 (quantization)指标来评估模型的性能。我们将使用前面讨论的评估指标：平均绝对误差（MAE）、均方误差（MSE）和R平方（$R^2$）分数。这些指标通过比较预测值（`y_pred`）与实际值（`y_test`）计算得出。

```python
# 计算评估指标
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) # 计算均方根误差
r2 = r2_score(y_test, y_pred)

print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"R平方 (R2 分数): {r2:.4f}")
```

我们来解读这些结果：

- **MAE**：平均而言，模型的预测值偏差约0.53个房屋中位价单位（以10万美元为单位，即53,000美元）。
- **MSE/RMSE**：RMSE约为0.72个单位（72,000美元）。与MAE类似，它衡量预测误差，但MSE（以及RMSE）由于平方运算，对大误差的惩罚更重。
- **R² 分数**：R平方约为0.59，表示测试集中约59%的房屋中位价方差可以由我们的模型根据特征进行解释。R平方为1则表示完美拟合。

### 预测值与实际值可视化

通过散点图比较实际值（`y_test`）与预测值（`y_pred`），可以对模型的性能进行视觉评估。对于一个好的模型，我们期望点紧密聚集在预测值等于实际值的对角线附近。



![实际房屋中位价 vs. 预测房屋中位价](plots/1851-0.json)



> 散点图比较了部分测试数据样本的预测房屋中位价（y轴）与实际值（x轴）。红色虚线表示完美预测（y=x）。点越靠近这条线，表示预测效果越好。

这个可视化证实了评估指标的结果。虽然存在明显的正相关性，但点在理想线附近有些分散，这反映了计算出的R平方分数约为0.59以及非零的误差指标。

您现在已经成功地使用Scikit-learn构建、训练、预测并评估了一个线性回归模型。这个实践工作流程是处理许多回归问题的根本。在后续章节中，我们将了解更复杂的模型和提升性能的技术。

## 参考资料

- [1.1. Linear models](https://scikit-learn.org/stable/modules/linear_model.html) — scikit-learn developers (2024)
  Journal: scikit-learn User Guide
  scikit-learn线性模型模块的官方文档，详细介绍了`LinearRegression`类、其用法及相关概念。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  构建和评估机器学习模型的实用指南，包含使用scikit-learn进行线性回归、数据分割和性能指标的章节。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://www.springer.com/book/9780387848570) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本基础教科书，详细阐述了线性回归、模型选择和性能评估方法的统计处理。

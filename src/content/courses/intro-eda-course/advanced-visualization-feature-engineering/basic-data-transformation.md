---
course: "intro-eda-course"
chapter: "advanced-visualization-feature-engineering"
lesson: "basic-data-transformation"
sourceId: 1197
sourceUrl: "https://apxml.com/zh/courses/intro-eda-course/chapter-5-advanced-visualization-feature-engineering/basic-data-transformation"
title: "基本数据转换：缩放与归一化"
description: "介绍缩放（MinMaxScaler, StandardScaler）和归一化方法及其作用。"
order: 5
plots: ["plots/1197-0.json", "plots/1197-1.json", "plots/1197-2.json"]
sourceHash: "0e876b5971803d5c99a9a70e17c14e7f44febc7f112ad355d7767b6ac14bfede"
sourceCorrections: []
---

不同的数值特征可能具有明显不同的范围、单位和分布。例如，一个特征可能表示年龄（范围从18到80），而另一个表示收入（范围从20,000到200,000）。许多机器学习 (machine learning)算法，特别是那些基于距离计算（如K近邻、支持向量 (vector)机）或梯度下降 (gradient descent)（如线性回归、神经网络 (neural network)）的算法，当数值输入特征处于相似的量级时，性能会更好或收敛更快。标准化和最小-最大缩放是两种常用的特征重缩放技术。

需要注意的是，这些转换改变了数据的*量级*和*位置*（均值），但通常不改变其分布的*形状*。如果一个特征具有偏态分布，缩放不会使其变为正态分布。处理偏态通常需要不同的转换（如对数或Box-Cox转换），这些通常被认为是更高级的特征工程步骤。

### 标准化（Z-分数缩放）

标准化将数据重缩放，使其均值（$\mu$）为0，标准差（$\sigma$）为1。这种转换常被称为Z-分数归一化 (normalization)。标准化的公式是：


$$
X_{	ext{标准化}} = \frac{X - \mu}{\sigma}
$$


其中$X$是原始特征值，$\mu$是该特征列的均值，$\sigma$是该特征列的标准差。

**为什么要使用标准化？**

- 它将数据中心化到零附近。
- 它考虑了特征中的变异性（标准差）。
- 与最小-最大缩放相比，它受异常值的影响较小，因为它使用标准差，标准差对极端值不如最小/最大范围那么敏感。
- 它适用于假设数据以零为中心或依赖距离计算的算法。

让我们看看如何使用Scikit-learn的`StandardScaler`应用标准化。

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# 示例DataFrame
data = {'Age': [25, 30, 35, 40, 45, 50],
        'Salary': [50000, 60000, 75000, 90000, 110000, 150000]}
df = pd.DataFrame(data)

print("原始数据:")
print(df)

# 初始化缩放器
scaler = StandardScaler()

# 对数据进行拟合和转换
# 注意: fit_transform 期望一个二维数组，因此使用 df[['Age', 'Salary']]
scaled_data = scaler.fit_transform(df[['Age', 'Salary']])

# 转换回DataFrame以提高可读性
df_scaled = pd.DataFrame(scaled_data, columns=['Age_scaled', 'Salary_scaled'])

print("\n标准化数据:")
print(df_scaled)

print(f"\n缩放后的均值:\n{df_scaled.mean()}")
print(f"\n缩放后的标准差:\n{df_scaled.std()}")
```

请注意，缩放后均值非常接近0，标准差为1。

### 最小-最大缩放

最小-最大缩放将数据重缩放到一个固定范围，通常是[0, 1]，尽管也可以指定其他范围。最小-最大缩放至[0, 1]范围的公式是：


$$
X_{	ext{缩放}} = \frac{X - X_{	ext{min}}}{X_{	ext{max}} - X_{	ext{min}}}
$$


其中$X$是原始特征值，$X_{min}$是该特征列的最小值，$X_{max}$是该特征列的最大值。

**为什么要使用最小-最大缩放？**

- 它确保所有特征都将具有完全相同的[0, 1]范围。
- 当算法需要特定有界区间内的输入时（例如，某些神经网络 (neural network)激活函数 (activation function)），这很有用。
- 它保留了原始数据值之间的关系。

然而，最小-最大缩放对异常值相当敏感。一个非常大或非常小的值可能会将数据的其余部分显著压缩到一个狭窄的范围。

以下是如何使用Scikit-learn的`MinMaxScaler`应用最小-最大缩放。

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 示例DataFrame（同前）
data = {'Age': [25, 30, 35, 40, 45, 50],
        'Salary': [50000, 60000, 75000, 90000, 110000, 150000]}
df = pd.DataFrame(data)

print("原始数据:")
print(df)

# 初始化缩放器（默认范围为 [0, 1]）
min_max_scaler = MinMaxScaler()

# 拟合和转换
scaled_data_minmax = min_max_scaler.fit_transform(df[['Age', 'Salary']])

# 转换回DataFrame
df_scaled_minmax = pd.DataFrame(scaled_data_minmax, columns=['Age_scaled', 'Salary_scaled'])

print("\n最小-最大缩放数据:")
print(df_scaled_minmax)

print(f"\n缩放后的最小值:\n{df_scaled_minmax.min()}")
print(f"\n缩放后的最大值:\n{df_scaled_minmax.max()}")
```

如预期的那样，两个缩放列的最小值现在为0，最大值为1。

### 标准化与最小-最大缩放的选择

选择通常取决于您计划使用的特定算法和数据的特点：

- **标准化：** 通常优选用于假设数据以零为中心或服从正态分布的算法（即使缩放本身不改变分布形状），以及用于基于距离的算法，如SVM、KNN和PCA。它对异常值更有效。
- **最小-最大缩放：** 当您需要有界区间内的数据，或用于可能对大输入值敏感的算法（如神经网络 (neural network)）时，这种方法很有用。如果在EDA期间识别出数据中存在显著异常值，请谨慎使用。

请考虑您在单变量分析中观察到的分布情况。如果一个特征存在极端异常值，标准化可能是一个更稳妥的默认选择。如果特征的量级差异很大但没有极端异常值，最小-最大缩放会很有效。

让我们使用直方图来可视化缩放对示例数据中“薪资”特征的影响。

### **原始数据**



![Effect of Scaling on Salary Distribution - Original](plots/1197-0.json)



### **标准化数据**



![Effect of Scaling on Salary Distribution - Standardized](plots/1197-1.json)



### **最小-最大缩放数据**



![Effect of Scaling on Salary Distribution - Min-Max Scaled](plots/1197-2.json)



> 薪资分布在标准化和最小-最大缩放前后的比较。注意x轴发生显著变化，反映了新的量级（标准化后约为-1到2，最小-最大缩放后为0到1）。分布的整体形状保持不变，只有量级和位置发生了改变。

### 实现注意事项

在为机器学习 (machine learning)模型准备数据时，避免*数据泄露*是必要的。这意味着测试集的信息不应影响训练过程，包括用于缩放的参数 (parameter)（如均值、标准差、最小值或最大值）。

正确的步骤是：

1. 将数据**划分**为训练集和测试集。
2. 仅在**训练数据**上**拟合**缩放器（`StandardScaler`或`MinMaxScaler`）。这将从训练数据中学习缩放参数（$\mu$，$\sigma$，$min$，$max$）。
3. 使用*已拟合的*缩放器对**训练数据**和**测试数据**进行**转换**。

```python
# 假设 X_train, X_test 是您的特征集（Pandas DataFrame 或 NumPy 数组）
scaler = StandardScaler() # 或 MinMaxScaler()

# 仅在训练数据上拟合
scaler.fit(X_train)

# 转换训练和测试数据
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

应用缩放等转换是EDA期间获得认识的直接结果。通过检查特征范围、分布，并考虑后续建模步骤的要求，您可以就如何最佳地准备数据做出明智的决定。

## 参考资料

- [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html) — Scikit-learn developers (2023)
  官方文档，提供各种数据预处理技术（包括StandardScaler和MinMaxScaler）的解释和使用详情。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2019)
  Publisher: O'Reilly Media
  一本实用指南，涵盖基本数据预处理步骤，包括特征缩放，并提供使用Scikit-learn的清晰示例和代码。
- [Introduction to Machine Learning with Python: A Guide for Data Scientists](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGENd3m8YUKjBg7uGIRlNsIz5LfRzSzWi1PgpBzzHodNWh39vODWo3ymBqk24eBH-Z-I3jK5YBMQ4BeHRMORVpNLZrV9TdbFPLyBOIh1vDX42qI0xz6l-KC3B3EN4yS-5C-QPvO49bGgnDiTfDijxm5_OLuBHbWWCOhLRUdy9ovH6Xu5nM9_rkApULQgDwYfjPEPF_WAsnaLbGyG9rNr_5orw8=) — Andreas C. Müller, Sarah Guido (2016)
  Publisher: O'Reilly Media
  本书在Scikit-learn框架内提供了数据预处理（包括特征缩放）的直观解释和实际演示。
- [Feature Engineering for Machine Learning: Principles and Techniques for Building Powerful Models](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) — Alice Zheng, Amanda Casari (2018)
  Publisher: O'Reilly Media
  介绍了多种特征工程方法，并有章节讨论不同的缩放方法及其对模型训练的影响。

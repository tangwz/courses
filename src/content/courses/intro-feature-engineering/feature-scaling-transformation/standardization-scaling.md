---
course: "intro-feature-engineering"
chapter: "feature-scaling-transformation"
lesson: "standardization-scaling"
sourceId: 1338
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-4-feature-scaling-transformation/standardization-scaling"
title: "标准化 (Z-score 缩放)"
description: "使用StandardScaler将数据转换为均值为零、方差为一的形式。"
order: 2
plots: ["plots/1338-0.json"]
sourceHash: "1f7c85cb5d2e2a360acce74828cfc3177222551f4418dc56a1ab2e1ef918749a"
sourceCorrections: []
---

当特征处于相对相似的尺度时，许多机器学习 (machine learning)算法表现更好或收敛更快。计算数据点间距离的算法（如K近邻）或依赖梯度下降 (gradient descent)优化（如线性回归、逻辑回归、神经网络 (neural network)）的算法，对输入特征的尺度尤其敏感。如果一个特征的范围是0到1，而另一个是0到1,000,000，算法可能会仅仅因为其尺度而非预测价值，错误地赋予范围更大的特征更高的权重 (weight)。

标准化，常被称为Z-score缩放，是一种处理此问题的常见且有效的方法。它将数据转换，使其均值（$\mu$）为0，标准差（$\sigma$）为1。

### 标准化公式

特征中每个值 $x$ 的转换使用以下公式计算：


$$
Z = \frac{x - \mu}{\sigma}
$$


- $x$ 是原始特征值。
- $\mu$ 是特征列的均值。
- $\sigma$ 是特征列的标准差。
- $Z$ 是标准化后的特征值。

每个转换后的值表示原始值偏离均值的标准差数量。大于均值的值将为正，小于均值的值将为负，而等于均值的值将为零。

### 使用Scikit-learn进行标准化

Scikit-learn在其`preprocessing`模块中提供了一个方便的转换器类`StandardScaler`。与其他Scikit-learn转换器一样，它遵循`fit`和`transform`模式。

1. **Fit（拟合）：** `fit`方法计算训练数据中每个特征的均值（$\mu$）和标准差（$\sigma$）。这些计算出的参数 (parameter)存储在缩放器对象中。仅在*训练*数据上拟合缩放器非常重要，以防止测试集的数据泄露。
2. **Transform（转换）：** `transform`方法使用*学到的* $\mu$ 和 $\sigma$ （来自`fit`步骤）将标准化公式应用于数据，从而生成缩放后的特征。在将数据输入模型之前，你将此方法应用于训练数据，以及稍后的任何新数据（如验证集或测试集）。

让我们看看实际操作。假设我们有一个包含“年龄”和“收入”特征的简单数据集：

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 示例数据
data = {'Age': [25, 30, 35, 40, 45, 50, 55, 60],
        'Income': [50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000]}
df = pd.DataFrame(data)

print("原始数据:")
print(df)

# 1. 初始化缩放器
scaler = StandardScaler()

# 2. 在数据上拟合缩放器（计算均值和标准差）
#    在实际场景中，仅在训练数据上拟合
scaler.fit(df)

# 3. 转换数据（应用缩放）
scaled_data = scaler.transform(df)

# 转换回DataFrame以提高可读性
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

print("\n缩放后的数据（标准化）:")
print(scaled_df)

# 你可以查看学到的参数
print(f"\n学到的均值: {scaler.mean_}")
print(f"学到的尺度（标准差）: {scaler.scale_}") # scale_ 是标准差
```

**Output:**

```
Original Data:
   Age  Income
0   25   50000
1   30   55000
2   35   60000
3   40   65000
4   45   70000
5   50   75000
6   55   80000
7   60   85000

Scaled Data (Standardization):
        Age    Income
0 -1.527525 -1.527525
1 -1.091089 -1.091089
2 -0.654654 -0.654654
3 -0.218218 -0.218218
4  0.218218  0.218218
5  0.654654  0.654654
6  1.091089  1.091089
7  1.527525  1.527525

Learned Mean: [   42.5 67500. ]
Learned Scale (Std Dev): [  11.45643924 11456.4392401 ]
```

请注意，缩放后的特征现在以零为中心。具体数值反映了它们相对于均值的原始位置，以标准差为单位衡量。

### 标准化效果的可视化

标准化改变了数据的*尺度*，但保留了其分布的*形状*。如果一个特征在标准化之前是偏斜的，标准化之后它仍然会偏斜，只是尺度不同。



![标准化前后年龄分布](plots/1338-0.json)



> 'Age'特征在标准化之前（左，蓝色）和之后（右，橙色）的分布。请注意，直方图的形状相同，但X轴的尺度已改变以反映以0为中心的Z分数。

### 何时使用标准化

- **假设高斯分布的算法：** 虽然标准化不会*使*数据变为高斯分布，但某些模型在特征具有与标准正态分布相似的属性（均值=0，标准差=1）时表现最佳。
- **基于距离的算法：** KNN、SVM（使用RBF核）和聚类算法（如K-Means）使用距离度量。标准化确保所有特征对距离计算的贡献相等。
- **基于梯度下降 (gradient descent)的算法：** 线性回归、逻辑回归、神经网络 (neural network)等算法在特征标准化后通常收敛更快。它有助于防止由不同特征范围影响梯度更新而导致的振荡或收敛缓慢。
- **主成分分析（PCA）：** PCA对特征的尺度敏感，因为它尝试寻找最大方差的方向。通常建议在应用PCA之前进行标准化。

### 注意事项与潜在缺点

- **对异常值的敏感性：** 标准化中使用的均值（$\mu$）和标准差（$\sigma$）对异常值敏感。少数极端值可以显著改变均值并增大标准差，从而在缩放后压缩“正常”数据点的范围。如果你的数据存在显著异常值，另一种缩放方法可能更好。
- **可解释性：** 标准化后的值（Z-score）是无单位的，表示与均值的标准差，这可能不如原始单位或像[0, 1]这样的最小-最大缩放范围那样直观易懂。

标准化是准备数值特征的核心技术。通过将数据以零为中心并根据其标准差进行缩放，使其更适合多种机器学习 (machine learning)算法，尤其是对特征尺度敏感的算法。请记住，仅在训练数据上拟合`StandardScaler`，然后用它来转换训练集和测试/验证集。

## 参考资料

- [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) — scikit-learn developers (2023)
  Journal: scikit-learn Documentation
  Scikit-learn StandardScaler 的官方文档，详细说明其用法和参数。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFZi9BxH3QPL3PwcWwe9U8DvOfEcxN6PtMuKGV-e8xw1MAjTueJilEO-EXPcN7a-ALz2Dk-nz5ZzR3Sj1zs3sBQtnTkC-30FLYcGKOCC1pRESHXOCJ4UMcoPUT0LyFJGxhVT9_rMLOgNj3Bx4I_hktPl9Y0A74YGq5LOTtzKCjTk5QXoYYwwFYYKwnuG0oMAKmN1ap_uRowh3QFLKg=) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用的机器学习指南，涵盖数据预处理和Python示例的特征缩放。
- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani (2021)
  Publisher: Springer; DOI: [10.1007/978-1-0716-1418-1](https://doi.org/10.1007/978-1-0716-1418-1)
  涵盖基础统计学习方法，包括数据预处理技术的讨论。
- [Feature Engineering and Selection: A Practical Approach for Predictive Models](https://www.taylorfrancis.com/books/9781315108230) — Max Kuhn and Kjell Johnson (2019)
  Publisher: Chapman and Hall/CRC; Pages: 310; DOI: [10.1201/9781315108230](https://doi.org/10.1201/9781315108230)
  一本关于特征工程的专业资源，详细介绍了各种缩放方法。

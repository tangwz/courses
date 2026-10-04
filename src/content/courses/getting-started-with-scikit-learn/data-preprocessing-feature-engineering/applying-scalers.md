---
course: "getting-started-with-scikit-learn"
chapter: "data-preprocessing-feature-engineering"
lesson: "applying-scalers"
sourceId: 1880
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-4-data-preprocessing-feature-engineering/applying-scalers"
title: "在Scikit-learn中应用缩放器"
description: "使用Scikit-learn转换器对数值特征进行缩放。"
order: 3
plots: ["plots/1880-0.json"]
sourceHash: "bcd037c27bf3c86e95bf76424c5f625538535f7437c7a94e6e53bc5cf94d38b3"
sourceCorrections: []
---

当数值输入特征被缩放到标准范围时，许多机器学习 (machine learning)算法会表现得更好。Scikit-learn提供了称为*转换器*的便捷工具来执行这些预处理步骤。这些转换器遵循一致的API，使其易于融入您的工作流程。

与缩放器等转换器配合使用的主要方法有：

1. `fit(X)`: 此方法从输入数据`X`中学习转换所需的参数 (parameter)。例如，`StandardScaler`会计算`X`中每个特征的均值($\mu$)和标准差($\sigma$)。重要的是，您**只应**在训练数据上调用`fit`，以避免测试集数据泄露。
2. `transform(X)`: 此方法将学习到的转换应用于输入数据`X`。它使用在`fit`步骤中计算的参数。您将使用此方法转换训练数据和测试数据（以及任何新数据）。
3. `fit_transform(X)`: 这是一个便利方法，它在单个步骤中对*相同*的数据`X`同时执行拟合和转换。为了提高效率，它通常用于训练集。然而，请记住在测试集上只使用`transform`方法，使用在训练数据上拟合的缩放器。

接下来，我们看看如何使用这个API应用常见的缩放器。

### 应用StandardScaler

`StandardScaler`转换您的数据，使其均值为0，标准差为1。此过程常被称为标准化。应用于每个特征的公式是：

$z = \frac{x - \mu}{\sigma}$

其中 $x$ 是原始特征值，$\mu$ 是特征的均值，$\sigma$ 是其标准差。

以下是使用方法：

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 示例数据（例如，两个特征）
data = pd.DataFrame({
    'FeatureA': [10, 20, 30, 40, 50],
    'FeatureB': [100, 110, 90, 120, 105]
})

# 1. 初始化缩放器
scaler = StandardScaler()

# 2. 将缩放器拟合到数据并进行转换
#    （在“训练”数据上使用fit_transform进行演示）
scaled_data = scaler.fit_transform(data)

# 输出是一个NumPy数组
print("原始数据:\n", data)
print("\n缩放后的数据 (StandardScaler):\n", scaled_data)

# 查看学习到的参数：
print("\nStandardScaler学习到的均值:", scaler.mean_)
print("StandardScaler学习到的比例（标准差）:", scaler.scale_)
```

`scaled_data` NumPy数组现在包含`FeatureA`和`FeatureB`的标准化值。请注意，即使输入是Pandas DataFrame，输出也是一个NumPy数组。如果您需要使用DataFrame，可以将其转换回来：

```python
scaled_df_standard = pd.DataFrame(scaled_data, columns=data.columns)
print("\n缩放后的数据作为DataFrame (StandardScaler):\n", scaled_df_standard)
```

### 应用MinMaxScaler

`MinMaxScaler`将数据缩放到固定范围，通常是$[0, 1]$。它根据每个特征的最小值和最大值计算转换：

$x_{scaled} = \frac{x - min(x)}{max(x) - min(x)}$

当您需要特征限制在特定范围时，这很有用，尽管它可能对异常值敏感，因为最小值和最大值决定了缩放。

```python
from sklearn.preprocessing import MinMaxScaler

# 使用相同的示例数据
data = pd.DataFrame({
    'FeatureA': [10, 20, 30, 40, 50],
    'FeatureB': [100, 110, 90, 120, 105]
})

# 1. 初始化缩放器（默认范围是[0, 1]）
min_max_scaler = MinMaxScaler()
# 您可以指定一个范围，例如MinMaxScaler(feature_range=(-1, 1))

# 2. 拟合并转换
scaled_data_minmax = min_max_scaler.fit_transform(data)

print("原始数据:\n", data)
print("\n缩放后的数据 (MinMaxScaler):\n", scaled_data_minmax)

scaled_df_minmax = pd.DataFrame(scaled_data_minmax, columns=data.columns)
print("\n缩放后的数据作为DataFrame (MinMaxScaler):\n", scaled_df_minmax)
```

请观察`scaled_data_minmax`中的所有值现在都介于0和1之间。

### 应用RobustScaler

`RobustScaler`使用对异常值具有鲁棒性的统计数据。它不使用均值和标准差（如`StandardScaler`）或最小值和最大值（如`MinMaxScaler`），而是使用*中位数*和*四分位距*（IQR）。IQR是第1四分位数（25th百分位数）和第3四分位数（75th百分位数）之间的范围。

转换公式为：

$x_{scaled} = \frac{x - median(x)}{IQR}$

当您的数据集包含可能过度影响`StandardScaler`或`MinMaxScaler`的显著异常值时，`RobustScaler`是一个好的选择。

```python
from sklearn.preprocessing import RobustScaler

# 包含FeatureB异常值的示例数据
data_outlier = pd.DataFrame({
    'FeatureA': [10, 20, 30, 40, 50],
    'FeatureB': [100, 110, 90, 120, 500] # 添加了一个异常值
})

# 1. 初始化缩放器
robust_scaler = RobustScaler()

# 2. 拟合并转换
scaled_data_robust = robust_scaler.fit_transform(data_outlier)

print("包含异常值的原始数据:\n", data_outlier)
print("\n缩放后的数据 (RobustScaler):\n", scaled_data_robust)

scaled_df_robust = pd.DataFrame(scaled_data_robust, columns=data_outlier.columns)
print("\n缩放后的数据作为DataFrame (RobustScaler):\n", scaled_df_robust)

# 与StandardScaler在相同异常值数据上的比较
scaler_std = StandardScaler()
scaled_data_std_outlier = scaler_std.fit_transform(data_outlier)
print("\n包含异常值的缩放数据 (StandardScaler):\n", scaled_data_std_outlier)
```

请注意，当存在异常值时，`RobustScaler`和`StandardScaler`对`FeatureB`的缩放方式有所不同。`RobustScaler`将数据围绕中位数居中并按IQR进行缩放，使得非异常点受极端值（500）的影响较小。而`StandardScaler`受异常值影响，导致其他点在缩放后更紧密地聚集在一起。

以下图表可视化了`data_outlier`示例中'FeatureB'在缩放前、经过`StandardScaler`处理后和经过`RobustScaler`处理后的分布。



![缩放器对含异常值数据（'FeatureB'）的影响](plots/1880-0.json)



> 'FeatureB'分布在应用StandardScaler和RobustScaler前后的对比。请注意，与StandardScaler相比，RobustScaler在大部分数据和异常值之间保持了更好的分离。

选择合适的缩放器取决于您数据的具体情况以及您计划使用的机器学习 (machine learning)算法的要求。`StandardScaler`是一个常用默认选择，但`MinMaxScaler`适用于需要有界输入的算法，而`RobustScaler`在处理异常值时更受推荐。请记住基本规则：**在训练数据上拟合，转换训练数据和测试数据。** Scikit-learn一致的转换器API使这些技术之间的切换变得简单直接。

## 参考资料

- [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html) — scikit-learn developers (2023)
  官方文档，解释了各种预处理技术，包括特征缩放、转换器API（fit/transform）以及StandardScaler、MinMaxScaler和RobustScaler等特定缩放器。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 850
  一本实用指南，涵盖了机器学习基础知识，包括数据预处理、特征缩放方法及其使用Scikit-learn的实现。（第三版）
- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2021)
  Publisher: Springer
  一本广受认可的教科书，提供了机器学习技术的统计基础，包括数据准备和特征缩放的理论基础。（第二版）

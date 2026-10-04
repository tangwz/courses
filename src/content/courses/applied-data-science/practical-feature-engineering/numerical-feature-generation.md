---
course: "applied-data-science"
chapter: "practical-feature-engineering"
lesson: "numerical-feature-generation"
sourceId: 1854
sourceUrl: "https://apxml.com/zh/courses/applied-data-science/chapter-2-practical-feature-engineering/numerical-feature-generation"
title: "从数值数据生成特征"
description: "针对数值变量的分箱、多项式特征和变换等技术。"
order: 1
plots: []
sourceHash: "56f74bbad7a53c8d36b58321b3074bd888fbc0501bf36f3b5efe405157534385"
sourceCorrections: []
---

正如章节引言中提到的，原始数值特征，虽然看起来简单明了，但通常无法直接向机器学习 (machine learning)算法展现其全部能力。像线性回归这样的模型假设线性关系，而其他模型可能难以处理倾斜的分布或大范围数据。从现有数值特征生成新特征可以帮助模型更好地获取潜在模式，处理非线性，并提升整体表现。接下来我们来看一些常用且有效的技术。

### 分箱（离散化）

有时，特征的确切数值不如其所属的范围或箱位重要。分箱，也称离散化，指的是通过将数值归入预设的箱位，将连续数值特征转换为离散分类特征。这有以下几个用处：

1. **捕获非线性影响：** 线性模型只能捕获线性关系。通过对特征进行分箱，你可以让模型对不同范围赋予不同的权重 (weight)或重要性，从而有效地捕获非线性模式。
2. **降低对异常值的敏感度：** 极端值可能对某些模型产生不成比例的影响。分箱将异常值放入特定箱位（通常是第一个或最后一个），从而降低其直接影响。
3. **简化模型：** 对于决策树等模型，分箱可以提供自然的分割点，从而简化模型结构。

分箱主要有两种方法：

- **固定宽度分箱：** 将数据范围划分为等宽的箱位。每个箱位中的数据点数量可能差异很大。
- **基于分位数的分箱：** 划分数据，使每个箱位包含大约相同数量的数据点。箱位宽度将根据数据分布而异。

假设我们有一个包含 `Age` 列的DataFrame `df`。

```python
import pandas as pd
import numpy as np

# 示例数据
data = {'Age': [22, 25, 31, 45, 58, 62, 75, 81, 19, 38]}
df = pd.DataFrame(data)

# 1. 固定宽度分箱（例如，4个箱位）
# 显式定义箱位边界
bins = [18, 30, 45, 60, 100] # 年龄 18-30, 31-45, 46-60, 61-100
labels = ['18-30', '31-45', '46-60', '61+']
df['Age_Bin_Fixed'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True, include_lowest=True)

# 或者让pandas自动确定等宽箱位
# df['Age_Bin_Fixed_Auto'] = pd.cut(df['Age'], bins=4) # 创建4个等宽箱位

# 2. 基于分位数的分箱（例如，4个分位数/四分位数）
df['Age_Bin_Quantile'] = pd.qcut(df['Age'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

print(df)
```

**选择箱位数量：** 箱位数量是一个超参数 (parameter) (hyperparameter)。箱位过少可能会过度简化数据，从而丢失有价值的信息。箱位过多可能会使特征过于细化，接近原始连续变量，并可能导致过拟合 (overfitting)。交叉验证可以帮助确定适当的箱位数量。

### 多项式特征

“线性模型根据定义，对线性关系进行建模（$y = w_1 x_1 + w_2 x_2 + ... + b$）。然而，关系往往是非线性的。多项式特征通过将现有特征提升到某个幂次（例如，$x^2$, $x^3$）以及创建特征之间的交互项（例如，$x_1 \cdot x_2$）来生成新特征。”

考虑一个简单特征 $x$。添加 $x^2$ 允许线性模型拟合二次关系：$y = w_1 x + w_2 x^2 + b$。这仍然是一个*线性模型*，因为它相对于系数（$w_1, w_2, b$）是线性的，即使 $y$ 与原始 $x$ 之间的关系是非线性的。

Scikit-learn 的 `PolynomialFeatures` 转换器是生成这些特征的便捷方法。

```python
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

# 包含两个特征的示例数据
data = {'FeatureA': [1, 2, 3, 4, 5],
        'FeatureB': [2, 3, 5, 5, 7]}
df_poly = pd.DataFrame(data)

# 创建最高2次的多项式特征
# include_bias=False 避免添加全为1的列（截距）
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df_poly)

# 获取特征名称以便清晰查看
feature_names = poly.get_feature_names_out(df_poly.columns)

# 使用这些特征创建新的DataFrame
df_poly_transformed = pd.DataFrame(poly_features, columns=feature_names)

print("原始特征：")
print(df_poly)
print("\n多项式特征（2次）：")
print(df_poly_transformed)

# 3次示例，仅FeatureA
poly_deg3 = PolynomialFeatures(degree=3, include_bias=False)
poly_features_a = poly_deg3.fit_transform(df_poly[['FeatureA']])
feature_names_a = poly_deg3.get_feature_names_out(['FeatureA'])
df_poly_a_transformed = pd.DataFrame(poly_features_a, columns=feature_names_a)

print("\n多项式特征（3次，仅FeatureA）：")
print(df_poly_a_transformed)
```

**考量因素：**

- **维度：** 生成的特征数量随次数和原始特征数量的增加而迅速增长。对于 $n$ 个特征的 $d$ 次多项式，会创建 $\binom{n+d}{d} - 1$ 个特征（不包括偏置 (bias)项）。这可能导致高昂的计算成本和过拟合 (overfitting)（维度灾难）。
- **特征缩放：** 通常，在应用多项式变换*之前*进行特征缩放非常必要，特别是对于高次多项式，以避免数值不稳定性和正则化 (regularization)问题。
- **可解释性：** 尽管多项式特征可以提升模型表现，但它们会使最终模型更难解释，因为系数现在与变换后的特征（$x^2$, $x_1 \cdot x_2$ 等）相关。

### 数学变换

应用对数、平方根或倒数等数学函数有助于稳定方差，使分布更对称（接近正态），并处理倾斜数据。许多机器学习 (machine learning)算法在特征分布更接近高斯（正态）分布时表现更好。

- **对数变换 ($log(x)$)：** 对于高度倾斜数据，特别是右偏数据（右侧长尾），此变换很有用。它会压缩大数值的范围，并扩展小数值的范围。要求数据严格为正 ($x > 0$)。如果存在零或负数，可以使用 $log(x+c)$（例如 $c$ 为1这样的小常数），使所有值变为正数。
- **平方根变换 ($\sqrt{x}$)：** 同样有助于处理右偏数据，但效果弱于对数变换。适用于非负数据 ($x \ge 0$)。
- **Box-Cox 变换：** 一种幂变换族，包括对数和平方根作为特例。它找到一个最优参数 (parameter)（$\lambda$）来转换数据，使其尽可能接近正态分布。要求数据严格为正 ($x > 0$)。
  
  $$
  y^{(\lambda)} = \begin{cases} \frac{x^\lambda - 1}{\lambda} & \text{若 } \lambda \neq 0 \\ \log(x) & \text{若 } \lambda = 0 \end{cases}
  $$
  
- **倒数变换 ($1/x$)：** 在特定情况下可能有用，但较不常见。也要求 $x \neq 0$。

让我们使用NumPy应用对数和平方根变换。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 示例倾斜数据（例如，收入）
np.random.seed(42)
income_data = np.random.gamma(2, scale=2000, size=1000)
income_data = np.round(income_data, 2)
df_trans = pd.DataFrame({'Income': income_data})

# 如果可能存在零值（此处不需要），则为对数变换添加一个小的常数
# df_trans['Income_Log'] = np.log(df_trans['Income'] + 1)

# 应用对数变换
df_trans['Income_Log'] = np.log(df_trans['Income'])

# 应用平方根变换
df_trans['Income_Sqrt'] = np.sqrt(df_trans['Income'])

# 应用Box-Cox变换
# scipy.stats.boxcox 返回变换后的数据和最优 lambda 值
df_trans['Income_BoxCox'], best_lambda = stats.boxcox(df_trans['Income'])

print(f"Box-Cox 的最优 lambda: {best_lambda:.4f}")
print(df_trans[['Income', 'Income_Log', 'Income_Sqrt', 'Income_BoxCox']].head())

# --- 可视化（可选：使用Matplotlib或Plotly进行分布图） ---
# Matplotlib示例（Plotly JSON对于分布图可能很大）
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].hist(df_trans['Income'], bins=30, color='#4dabf7')
axes[0].set_title('原始收入')
axes[1].hist(df_trans['Income_Log'], bins=30, color='#9775fa')
axes[1].set_title('对数变换后')
axes[2].hist(df_trans['Income_Sqrt'], bins=30, color='#69db7c')
axes[2].set_title('平方根变换后')
axes[3].hist(df_trans['Income_BoxCox'], bins=30, color='#ff922b')
axes[3].set_title('Box-Cox变换后')
plt.tight_layout()
# 如果在网络环境中使用，您会保存此图或生成一个Plotly版本。
# plt.show() # 取消注释可在本地显示图表
```

> 直方图说明了变换如何减少偏度。原始收入数据严重右偏。对数和Box-Cox变换产生了更接近对称的分布，而平方根变换效果更温和。

**何时应用变换：**

- 使用直方图或Q-Q图观察特征分布。
- 检查模型残差是否显示与特征大小相关的模式（表明异方差性）。
- 一些模型（如线性回归、逻辑回归、使用特定核函数的SVM、K-Means）通常受益于更接近正态分布或具有相似尺度的特征。基于树的模型通常对特征变换和缩放的敏感度较低。

从数值数据生成特征通常是一个迭代过程。您可以尝试分箱、多项式特征或变换，训练模型，评估其表现，然后根据结果调整您的特征工程策略。这里讨论的技术为机器学习模型创建更具信息量的数值特征提供了坚实支持。

## 参考资料

- [Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) — Alice Zheng, Amanda Casari (2018)
  Publisher: O'Reilly Media
  本书为特征工程的各种技术提供了基础知识和实用方法，包括数值数据的分箱、多项式特征和数学转换。
- [sklearn.preprocessing.PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) — Scikit-learn Developers (2023)
  Journal: Scikit-learn Documentation
  官方文档，解释了用于从数值输入生成多项式和交互特征的PolynomialFeatures转换器的使用方法和参数。
- [scipy.stats.boxcox](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html) — SciPy Developers (2023)
  Journal: SciPy Documentation
  官方文档，详细介绍了Box-Cox变换函数的数学基础和实际应用，用于规范数据分布。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFhg6PFdtDqRH8JL31WRtWzIzcSb42hYtH5c3Y3zihAyisTrU6MScYB6gPFBpfCzi-cKcgLj-yEIz9ft-dEYf5REtF9BikXXFTVS6mdL2gZTBBIU8XAlGwpAtMnA9juyNNNjuOuiHQ_ERhmSbi0p3agyB_vRLdthgVQrVXTIRiVRhKNiCN54Iaql_NP_RpHLMFvXDMfPVOZkeEnDjBnT7657U-Bxv5BVTl4Psnfk20zmmzBEeIx-Tk1DUlSzzuhA7HunWu3AX53xmN5oC_M3MjWyCTMHDc419bRK70Vf2RdoKU7tIZDqDvek7jKgYcNbEa5JIkPqfLDwNT45bP9IJE9vgJolAuL0FRjU3A9a1vN_uTIC5VzJ6IvtNEy_KX1DEs8mRZ7CkHeucyw4glHmH_L5C_TwvEjJz5TksXFcotJrFUFuLKkP1vd8buAc67tHaBkvugACx3FuN24qVfcFYQsN6URUppAWRBocq_RN1eSHiG91veUvsQAFUUgJTrcXY_JBIY=) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用指南，涵盖了预处理技术，包括特征缩放、转换和生成新特征，对于改进各种机器学习模型至关重要。

---
course: "intro-feature-engineering"
chapter: "feature-scaling-transformation"
lesson: "robust-scaling"
sourceId: 1340
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-4-feature-scaling-transformation/robust-scaling"
title: "处理异常值的缩放"
description: "使用`RobustScaler`，它基于百分位数缩放数据，使其对异常值不敏感。"
order: 4
plots: ["plots/1340-0.json"]
sourceHash: "b354e8fe3e66c7fbd105c6fd08dc0c36309578784aa6161cafba32a0709bed3f"
sourceCorrections: []
---

标准化（Standardization）和最小-最大缩放（Min-Max scaling）是将数值特征调整到统一尺度的常用方法。标准化使用均值（μ）和标准差（σ），而最小-最大缩放依赖于最小值和最大值。然而，这两种方法都有一个潜在缺点：对异常值敏感。异常值作为极端数值，会明显影响这些统计量。一个非常大或非常小的数值会大幅改变均值，增大标准差，或改变最大/最小值，导致正常数据被压缩，并且可能使大部分观测值的缩放效果不理想。

当您的数据集包含明显异常值时，使用对这些极端值不那么敏感的缩放方法通常更有用。这就是Scikit-learn中的`RobustScaler`的作用。`RobustScaler`不使用均值、标准差或最小/最大值，而是使用对异常值不易受影响的统计量：中位数和四分位距（IQR）。

IQR是数据中第一四分位数（25%分位数，$Q1$）和第三四分位数（75%分位数，$Q3$）之间的范围。
$IQR = Q3 - Q1$
根据定义，IQR包含数据的中间50%，使其比标准差或总范围（最大值-最小值）更不易受分布两端极端值的影响。

`RobustScaler`通过减去中位数来使数据居中，然后通过除以IQR来缩放数据。其公式为：
$X_{scaled} = \frac{X - 中位数}{IQR}$
主要思想是利用对异常值有抵抗力的分位数。

### 为什么要使用缩放？

考虑一个特征，其值为[1, 2, 3, 4, 5, 100]。

- **均值：** 19.17，**标准差：** 约36.3
- **中位数：** 3.5，**Q1：** 2.25，**Q3：** 4.75，**IQR：** 2.5

如果我们使用`StandardScaler`，异常值（100）会严重影响均值和标准差，导致大多数数据点（[1, 2, 3, 4, 5]）在缩放后被压缩到一个很小的范围内。
如果我们使用`MinMaxScaler`，异常值（100）会决定最大值，也可能再次挤压其他数据点。
使用`RobustScaler`，中位数（3.5）和IQR（2.5）受数值100的影响小得多。基于这些统计量进行缩放，可以更有效地保持非异常值点之间的相对间距。

### Scikit-learn中的实现

使用`RobustScaler`非常简单，并且遵循Scikit-learn转换器的常见API。

````python
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import plotly.graph_objects as go

# 含有异常值的样本数据
data = np.array([1, 2, 3, 4, 5, 100]).reshape(-1, 1)
df = pd.DataFrame(data, columns=['Feature'])

# 初始化缩放器
robust_scaler = RobustScaler()
standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

# 应用缩放器
df['RobustScaled'] = robust_scaler.fit_transform(df[['Feature']])
df['StandardScaled'] = standard_scaler.fit_transform(df[['Feature']])
df['MinMaxScaled'] = minmax_scaler.fit_transform(df[['Feature']])

print(df)

# 可视化（可选 - 展示效果）
fig = go.Figure()

# 原始数据（略微偏移以便查看）
fig.add_trace(go.Scatter(
    x=df.index - 0.1, y=df['Feature'], mode='markers', name='原始数据',
    marker=dict(color='#adb5bd', size=8)
))

# Robust缩放
fig.add_trace(go.Scatter(
    x=df.index - 0.03, y=df['RobustScaled'], mode='markers', name='Robust缩放',
    marker=dict(color='#7950f2', size=8) # violet
))

# 标准缩放
fig.add_trace(go.Scatter(
    x=df.index + 0.03, y=df['StandardScaled'], mode='markers', name='标准缩放',
    marker=dict(color='#1c7ed6', size=8) # blue
))

# 最小-最大缩放
fig.add_trace(go.Scatter(
    x=df.index + 0.1, y=df['MinMaxScaled'], mode='markers', name='最小-最大缩放',
    marker=dict(color='#12b886', size=8) # teal
))

fig.update_layout(
    title='含异常值的缩放方法比较',
    xaxis_title='数据点索引',
    yaxis_title='缩放后的值',
    legend_title='缩放器类型',
    template='plotly_white',
    width=700,
    height=400
)

# 若在交互式环境中运行，显示图表
# fig.show()

# 将图表以Markdown格式表示：
plotly_json = fig.to_json(pretty=False)
print(f'\n```plotly\n{plotly_json}\n```')
````

执行此代码会生成以下DataFrame：

```
   Feature  RobustScaled  StandardScaled  MinMaxScaled
0        1     -1.000000       -0.500674      0.000000
1        2     -0.600000       -0.473060      0.010101
2        3     -0.200000       -0.445446      0.020202
3        4      0.200000       -0.417832      0.030303
4        5      0.600000       -0.390218      0.040404
5      100     38.600000        2.227231      1.000000
```



![含异常值的缩放方法比较](plots/1340-0.json)



> 比较了`RobustScaler`、`StandardScaler`和`MinMaxScaler`如何处理包含明显异常值（数值100）的数据集。请注意，`RobustScaler`相比其他两种方法，使得非异常值点（0-4）保持更分散。

可以看到，与`StandardScaler`（范围从-0.50到-0.39）或`MinMaxScaler`（范围从0.0到0.04）相比，前五个点（“正常值”）使用`RobustScaler`（范围从-1.0到0.6）时保持了更合理的分布。异常值（100）仍然存在，但对其余数据的尺度影响不大。

`RobustScaler`的重要参数 (parameter)：

- `with_centering`: 布尔值（默认为`True`）。如果为`True`，则在缩放前通过减去中位数来使数据居中。
- `with_scaling`: 布尔值（默认为`True`）。如果为`True`，则通过除以IQR来缩放数据。
- `quantile_range`: 元组（浮点数，浮点数），默认为`(25.0, 75.0)`。指定用于计算尺度的分位数范围（默认为IQR）。您可以调整此值，例如，使用第10和第90百分位数`(10.0, 90.0)`。

### 何时选择缩放方法

当满足以下条件时，`RobustScaler`是一个不错的选择：

1. 您的数据包含明显的异常值。
2. 您使用的算法对输入特征的尺度敏感，例如线性模型（线性回归、逻辑回归、SVM）、K近邻（KNN）或使用梯度下降 (gradient descent)优化的神经网络 (neural network)。
3. 在存在异常值的情况下，您希望比`StandardScaler`或`MinMaxScaler`更好地保持非异常值数据点之间的相对距离。

然而，如果您的数据大致呈正态分布且没有明显的异常值，`StandardScaler`可能是一个更传统的选择，通常与某些统计模型的假设相符。与大多数特征工程决定一样，最佳方法通常需要尝试不同的缩放方法，并通过交叉验证评估它们对模型性能的影响。

## 参考资料

- [sklearn.preprocessing.RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) — scikit-learn developers (2023)
  Publisher: scikit-learn project
  RobustScaler 的官方文档，详细介绍了其功能和参数。
- [An Introduction to Statistical Learning with Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2021)
  Publisher: Springer; DOI: [10.1007/978-1-0716-1418-1](https://doi.org/10.1007/978-1-0716-1418-1)
  一本基础性教材，解释了统计学习方法，包括数据缩放和异常值影响背后的统计原理。

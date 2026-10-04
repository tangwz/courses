---
course: "applied-data-science"
chapter: "advanced-data-acquisition-preparation"
lesson: "data-transformation-normalization"
sourceId: 1847
sourceUrl: "https://apxml.com/zh/courses/applied-data-science/chapter-1-advanced-data-acquisition-preparation/data-transformation-normalization"
title: "数据转换与标准化方法"
description: "了解常见的数据转换方法，如缩放、归一化和编码。"
order: 6
plots: ["plots/1847-0.json", "plots/1847-1.json", "plots/1847-2.json", "plots/1847-3.json"]
sourceHash: "23932240f3a5d18718ea3890f7b0e4f71ea86cd09782b51f6ca2d1a3ae25a9d6"
sourceCorrections: []
---

特征转换是数据准备中的一个重要步骤，通常在解决不正确条目和缺失值等初始数据质量问题之后进行。原始特征通常具有不同的尺度和分布，这可能对许多机器学习 (machine learning)算法的性能产生负面影响，特别是那些依赖距离计算（如K-近邻算法或支持向量 (vector)机）或梯度下降 (gradient descent)优化的算法。以下将介绍一些常用方法，用于对数据进行缩放、归一化 (normalization)和转换，使其更适合建模。

## 为何转换数据？

假设有一个数据集，其中包含“年龄”（范围从20到70）和“收入”（范围从30,000到250,000）等特征。如果你在计算距离的算法中直接使用这些数据，“收入”特征仅仅因为其较大的尺度就会主导计算，可能掩盖“年龄”的影响，即使“年龄”同等或更具信息量。转换特征的目标是：

1. **统一尺度：** 将特征置于共同尺度，避免数值较大的特征主导数值较小的特征。
2. **调整分布：** 将偏态分布调整得更对称（通常更接近正态分布），这可以帮助某些模型满足其基本假设并提升性能。
3. **稳定方差：** 某些转换可以帮助稳定特征值范围内的方差。

我们来看看一些广泛使用的方法。

## 数值特征缩放

缩放调整数值特征的范围，而不改变其分布的形状。两种常用方法是最小-最大缩放和标准化。

### 最小-最大缩放（归一化 (normalization)）

最小-最大缩放，通常称为归一化，将特征重新缩放到固定范围，通常是 [0, 1]。特征 $x$ 的公式是：


$$
x_{scaled} = \frac{x - \min(x)}{\max(x) - \min(x)}
$$


此处，$\min(x)$ 和 $\max(x)$ 是该特征在*训练*数据集中的最小值和最大值。

当你需要将数据限制在特定范围内时，此方法很有用。然而，它对异常值相当敏感。单个非常大或非常小的值会将其余数据显著压缩到 [0, 1] 范围的狭窄部分。

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# 偏斜数据样本
np.random.seed(42)
data = pd.DataFrame({
    'FeatureA': np.random.gamma(2, 2, 100) * 10, # 偏斜
    'FeatureB': np.random.normal(50, 10, 100)    # 接近正态
})

# 初始化并拟合缩放器
min_max_scaler = MinMaxScaler()
# 重要：在实际场景中仅对训练数据进行拟合
scaled_data_mm = min_max_scaler.fit_transform(data)
scaled_df_mm = pd.DataFrame(scaled_data_mm, columns=['FeatureA_scaled', 'FeatureB_scaled'])

# 可视化（可选比较）
fig = go.Figure()
fig.add_trace(go.Histogram(x=data['FeatureA'], name='Original Feature A', marker_color='#1c7ed6', nbinsx=15))
fig.add_trace(go.Histogram(x=scaled_df_mm['FeatureA_scaled'], name='MinMax Scaled Feature A', marker_color='#ff922b', nbinsx=15, xaxis='x2', yaxis='y2'))

fig.update_layout(
    title_text='最小-最大缩放效果（形状不变）',
    xaxis_title='原始值', yaxis_title='计数',
    xaxis2=dict(title='缩放值 [0,1]', overlaying='x', side='top'), yaxis2=dict(overlaying='y', side='right'),
    bargap=0.1, height=350, legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99), margin=dict(l=20, r=20, t=50, b=20)
)
# fig.show() # 在交互式环境中显示图表
```



![最小-最大缩放效果（形状不变）](plots/1847-0.json)



> 特征分布在最小-最大缩放前后的比较。范围被压缩到 [0, 1]，但整体形状（偏度）保持不变。

### 标准化（Z-分数缩放）

标准化重新缩放特征，使其均值 ($\mu$) 为 0，标准差 ($\sigma$) 为 1。公式是：


$$
x_{standardized} = \frac{x - \mu}{\sigma}
$$


同样，$\mu$ 和 $\sigma$ 是从*训练*数据计算的。结果分布将具有均值 0 和单位方差。标准化比最小-最大缩放受异常值影响小，并且通常更适用于假定数据呈正态分布并以零为中心算法，或对特征方差敏感的算法。

```python
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# 假设存在前一个例子中的 'data' DataFrame

# 初始化并拟合缩放器
standard_scaler = StandardScaler()
# 重要：在实际场景中仅对训练数据进行拟合
scaled_data_std = standard_scaler.fit_transform(data)
scaled_df_std = pd.DataFrame(scaled_data_std, columns=['FeatureA_scaled', 'FeatureB_scaled'])

# 可视化
fig_std = go.Figure()
fig_std.add_trace(go.Histogram(x=data['FeatureA'], name='Original Feature A', marker_color='#1c7ed6', nbinsx=15))
fig_std.add_trace(go.Histogram(x=scaled_df_std['FeatureA_scaled'], name='Standardized Feature A', marker_color='#7048e8', nbinsx=15, xaxis='x2', yaxis='y2'))

fig_std.update_layout(
    title_text='标准化效果（形状不变）',
    xaxis_title='原始值', yaxis_title='计数',
    xaxis2=dict(title='标准化值 (均值=0, 标准差=1)', overlaying='x', side='top'), yaxis2=dict(overlaying='y', side='right'),
    bargap=0.1, height=350, legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99), margin=dict(l=20, r=20, t=50, b=20)
)
# fig_std.show() # 在交互式环境中显示图表
```



![标准化效果（形状不变）](plots/1847-1.json)



> 特征分布在标准化前后的比较。特征以0为中心，标准差为1，但偏度保持不变。

### 选择缩放方法

- **最小-最大缩放：** 当你需要将特征限制在有界区间 [0, 1] 内（例如，图像像素强度）或用于不假定任何特定分布的算法时使用。小心异常值。
- **标准化：** 通常更适用于假定零均值和单位方差的算法（如线性回归、带正则化 (regularization)的逻辑回归）或当存在异常值时使用距离度量的算法。

## 转换特征分布

有时，仅仅缩放是不够的。如果特征的分布严重偏斜，应用非线性转换可以帮助使其更对称，可能提升模型性能。

### 对数转换

对数函数压缩大值范围并扩展小值范围。这使其能有效减少右偏性（尾部向右延伸的情况）。

- 如果所有值都为正，使用 $log(x)$。
- 如果值包含零，使用 $log(1 + x)$（通过 `numpy.log1p`）。

```python
# 假设存在带有偏斜 'FeatureA' 的 'data' DataFrame
data['FeatureA_log'] = np.log1p(data['FeatureA']) # 使用 log1p 以确保处理可能存在的0值

# 可视化
fig_log = go.Figure()
fig_log.add_trace(go.Histogram(x=data['FeatureA'], name='Original Feature A', marker_color='#1c7ed6', nbinsx=15))
fig_log.add_trace(go.Histogram(x=data['FeatureA_log'], name='Log Transformed Feature A', marker_color='#20c997', nbinsx=15, xaxis='x2', yaxis='y2'))

fig_log.update_layout(
    title_text='对数转换对偏斜数据的影响',
    xaxis_title='原始值', yaxis_title='计数',
    xaxis2=dict(title='Log(1+值)', overlaying='x', side='top'), yaxis2=dict(overlaying='y', side='right'),
    bargap=0.1, height=350, legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99), margin=dict(l=20, r=20, t=50, b=20)
)
# fig_log.show() # 在交互式环境中显示图表
```



![对数转换对偏斜数据的影响](plots/1847-2.json)



> 对数转换使“特征 A”的分布显得更对称（更接近钟形），与原始右偏分布相比。

### Box-Cox 转换

Box-Cox 转换是一种更一般化的幂转换，可以找到接近最优的转换，使你的数据更接近正态分布。它定义为：


$$
x^{(\lambda)} = \begin{cases}
      \frac{x^\lambda - 1}{\lambda} & \text{如果 } \lambda \neq 0 \\
      \ln(x) & \text{如果 } \lambda = 0
      \end{cases}
$$


该转换找到 $\lambda$ (lambda) 的最佳值，以稳定方差并提高正态性。一个重要限制是 **Box-Cox 要求所有数据为正**。

```python
from scipy.stats import boxcox
import plotly.graph_objects as go

# 假设存在带有正值且偏斜的 'FeatureA' 的 'data' DataFrame
# 在应用 Box-Cox 之前确保 FeatureA 为正值
if (data['FeatureA'] > 0).all():
    # 应用 Box-Cox：返回转换后的数据和最优 lambda
    featureA_boxcox, best_lambda = boxcox(data['FeatureA'])
    print(f"Box-Cox 找到的最优 Lambda: {best_lambda:.4f}")

    # 可视化
    fig_boxcox = go.Figure()
    fig_boxcox.add_trace(go.Histogram(x=data['FeatureA'], name='Original Feature A', marker_color='#1c7ed6', nbinsx=15))
    fig_boxcox.add_trace(go.Histogram(x=featureA_boxcox, name='Box-Cox Transformed Feature A', marker_color='#be4bdb', nbinsx=15, xaxis='x2', yaxis='y2'))

    fig_boxcox.update_layout(
        title_text='Box-Cox 转换效果',
        xaxis_title='原始值', yaxis_title='计数',
        xaxis2=dict(title='Box-Cox 转换值', overlaying='x', side='top'), yaxis2=dict(overlaying='y', side='right'),
        bargap=0.1, height=350, legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99), margin=dict(l=20, r=20, t=50, b=20)
    )
    # fig_boxcox.show() # 显示图表
else:
    print("FeatureA 包含非正值。Box-Cox 无法直接应用。")
    # 如果无法应用，则创建一个空的 plotly json
    fig_boxcox_json = '{"layout": {"title": {"text": "Box-Cox Skipped (Non-Positive Data)"},"height": 100}, "data": []}'
```



![Box-Cox 转换效果](plots/1847-3.json)



> Box-Cox 转换也使偏斜数据更对称，在此情况下与对数转换相似，通过自动找到合适的幂转换。

### Yeo-Johnson 转换

Yeo-Johnson 转换在精神上与 Box-Cox 相似，但优点是能够处理**非正数据**。如果你的数据包含零或负数，Yeo-Johnson 是一种合适的替代方案，可实现更接近正态的分布。它在 `scikit-learn` 中以 `PowerTransformer(method='yeo-johnson')` 提供。

## 实践中应用转换：训练/测试集划分

应用任何转换（缩放或分布调整）时的一个要点是**只对训练数据拟合转换器**。你从训练集中学习参数 (parameter)（如最小值/最大值、均值/标准差或 lambda），然后使用这些学习到的参数来转换训练集、验证集和测试集。

为什么？在分割之前对整个数据集进行拟合会导致**数据泄露**。测试集中的信息（例如，其最小值或最大值）会泄露到训练过程中，导致对模型在未见数据上的性能估计过于乐观。

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression # 示例模型
from sklearn.pipeline import Pipeline

# 假设 'X' 是你的特征矩阵，'y' 是你的目标向量
# X = data[['FeatureA', 'FeatureB']] # 示例特征
# y = ... # 你的目标变量

# 先分割数据！
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 仅对训练数据拟合缩放器
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)

# 将相同的已拟合缩放器应用于测试数据
# X_test_scaled = scaler.transform(X_test) # 使用 transform()，而不是 fit_transform()

# -- 使用管道简化此过程 --
# 定义步骤：1. 缩放，2. 模型
# pipe = Pipeline([
#     ('scaler', StandardScaler()),
#     ('classifier', LogisticRegression())
# ])

# 在训练数据上拟合整个管道
# 管道处理缩放器的拟合，然后训练模型
# pipe.fit(X_train, y_train)

# 在测试数据上预测
# 管道自动使用已拟合的缩放器转换测试数据
# predictions = pipe.predict(X_test)
# score = pipe.score(X_test, y_test)
# print(f"模型在测试数据上的得分: {score:.4f}")
```

如上所示，使用 `scikit-learn` 管道是连接预处理步骤和建模的推荐方式。管道确保在交叉验证期间仅在训练折叠上进行拟合，并且正确的转换按顺序应用。

## 总结

数据转换和标准化是机器学习 (machine learning)数据准备中的必要步骤。通过使用最小-最大缩放或标准化等方法将特征调整到可比较的尺度，你可以避免某些特征不适当地影响模型结果。此外，使用对数或 Box-Cox 转换等方法转换偏斜分布可以帮助那些在更对称、类似正态数据上表现更好的模型。请记住基本规则：始终只在训练数据上拟合转换器，然后使用它们来转换训练和测试数据集，以防止数据泄露。这些方法为高效的特征工程和模型构建提供了支持。

## 参考资料

- [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html) — scikit-learn developers (2024)
  提供关于缩放、归一化和转换方法的全面解释和API参考，包括管道的实际使用。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125964/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实践指南，涵盖数据预处理技术，包括缩放和转换，并附有使用scikit-learn的清晰示例。
- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2021)
  Publisher: Springer
  提供统计学基础背景和数据准备实践指导，适用于各种机器学习模型。
- [An Analysis of Transformations](https://doi.org/10.1111/j.2517-6161.1964.tb00553.x) — George E. P. Box and David R. Cox (1964)
  Journal: Journal of the Royal Statistical Society. Series B (Methodological); Publisher: Oxford University Press; Volume: 26; Pages: 211-252; DOI: [10.1111/j.2517-6161.1964.tb00553.x](https://doi.org/10.1111/j.2517-6161.1964.tb00553.x)
  引入Box-Cox变换的原始学术论文，旨在改善正态性和方差同质性。

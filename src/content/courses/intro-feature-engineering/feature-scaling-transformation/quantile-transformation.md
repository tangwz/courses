---
course: "intro-feature-engineering"
chapter: "feature-scaling-transformation"
lesson: "quantile-transformation"
sourceId: 1353
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-4-feature-scaling-transformation/quantile-transformation"
title: "分位数变换"
description: "根据分位数将特征转换为符合均匀或正态分布。"
order: 8
plots: ["plots/1353-0.json", "plots/1353-1.json", "plots/1353-2.json"]
sourceHash: "d2d7d1eaedba8b041e1f748b2e708e4ca519f69b5b3e7cb9af23b14716065415"
sourceCorrections: []
---

分位数变换提供了一种独特的特征分布修改方法。与标准化和归一化 (normalization)等主要调整特征尺度的方法，或对数变换和Box-Cox变换等旨在使分布更接近高斯分布的方法不同，分位数变换是一种非线性过程。它将特征的概率分布映射到特定的均匀分布或正态分布，而与原始分布的形状无关。这是通过利用数据点的秩或分位数来实现的。

### 分位数变换的原理

分位数变换的核心思想是估计特征的经验累积分布函数（CDF），然后使用此CDF将原始值映射到目标输出分布。

1. **估计经验CDF：** 对于特征中的每个数据点$x$，确定其相对于其他点的位置。实质上，我们计算小于或等于$x$的数据点的比例。这为我们提供了CDF值的估计，$F(x) = P(X \le x)$，其范围在0到1之间。
2. **映射到目标分布：** 这些CDF值（如果特征是连续的，这些CDF值将呈均匀分布）然后映射到目标分布的分位数：
   - **均匀分布：** 如果目标是 $[0, 1]$ 上的均匀分布，则映射是直接的。估计的CDF值本身就成为变换后的值。这根据数据点的秩，有效地将它们均匀地分散在 $[0, 1]$ 范围内。
   - **正态分布：** 如果目标是标准正态分布（$N(0, 1)$），则估计的CDF值 $u = F(x)$ 通过标准正态分布的逆CDF（也称作分位数函数或百分点函数）$\Phi^{-1}$ 进行映射。变换后的值变为 $z = \Phi^{-1}(u)$。此过程将数据投影到高斯形状上。

由于此方法依赖于数据点的秩顺序而非其绝对值，它对异常值具有天然的鲁棒性。异常值将被映射到目标分布的极端边缘（例如，对于均匀分布接近0或1，对于正态分布则是较大的负值/正值），但不会像`StandardScaler`或`MinMaxScaler`那样不成比例地影响其他点的变换。

### Scikit-learn中的实现

Scikit-learn提供了`sklearn.preprocessing.QuantileTransformer`类用于此目的。让我们看看如何使用它。

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 生成一些偏斜数据
np.random.seed(42)
data_original = np.random.exponential(scale=2, size=1000).reshape(-1, 1) + 1 # 加1以避免之后使用对数时出现零的问题

# 初始化变换器
qt_uniform = QuantileTransformer(output_distribution='uniform', n_quantiles=1000, random_state=42)
qt_normal = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42)

# 应用变换
data_uniform = qt_uniform.fit_transform(data_original)
data_normal = qt_normal.fit_transform(data_original)

# 创建DataFrame以便绘图
df = pd.DataFrame({
    '原始数据': data_original.flatten(),
    '均匀分位数': data_uniform.flatten(),
    '正态分位数': data_normal.flatten()
})

# --- 可视化 ---
fig = make_subplots(rows=1, cols=3, subplot_titles=('原始指数数据', '均匀分位数变换后', '正态分位数变换后'))

fig.add_trace(go.Histogram(x=df['原始数据'], name='原始', marker_color='#4dabf7'), row=1, col=1)
fig.add_trace(go.Histogram(x=df['均匀分位数'], name='均匀', marker_color='#38d9a9'), row=1, col=2)
fig.add_trace(go.Histogram(x=df['正态分位数'], name='正态', marker_color='#be4bdb'), row=1, col=3)

fig.update_layout(
    title_text='分位数变换对偏斜数据的影响',
    bargap=0.1,
    showlegend=False,
    height=350,
    margin=dict(l=20, r=20, t=60, b=20)
)

# 显示Plotly图表JSON
# print(fig.to_json()) # 你会在你的环境中运行此代码
```

---



![原始指数数据](plots/1353-0.json)



> 原始指数数据的分布。

---



![均匀分位数变换后](plots/1353-1.json)



> 均匀分位数变换后的分布。请注意值是如何均匀分散的。

---



![正态分位数变换后](plots/1353-2.json)



> 正态分位数变换后的分布。数据现在类似于高斯形状。

---

### 优点与考量

分位数变换具有多项优点和重要的考量事项：

- **对异常值的鲁棒性：** 如前所述，它对异常值具有鲁棒性，因为它基于秩而非绝对值进行操作。一个非常大或小的异常值只会占据变换后分布的极端边缘，不会使其他点的整体尺度发生偏斜。
- **处理非高斯数据：** 对于不服从高斯分布的特征特别有用，这通常是许多线性模型（例如，线性回归、逻辑回归）和一些基于距离的算法（例如，K近邻、支持向量 (vector)机）的要求。
- **保持秩顺序：** 这种变换保持数据的秩顺序，这意味着如果原始数据中 $x_1 < x_2$，则变换后的数据中 $T(x_1) < T(x_2)$。这对于许多机器学习 (machine learning)算法很重要，因为值的相对顺序包含信息。
- **使用`fit_transform`时的数据泄露：** 与其他缩放器一样，只在训练数据上应用`fit`，然后对训练和测试数据都应用`transform`，这一点很重要。在分割之前直接将`fit_transform`应用于整个数据集（包括测试数据）可能导致**数据泄露**，即测试集中的信息隐式地影响训练过程，从而导致过分乐观的性能估计。

### 何时使用分位数变换

在以下情况，考虑使用分位数变换：

- 您的特征具有**高度偏斜或非高斯分布**，这可能对分布形状敏感的算法性能产生负面影响。
- 您关注异常值对变换和模型的影响。
- 您希望应用使用**正态分布输入**时表现更好的模型，但传统的对数变换方法不够充分或不适用（例如，由于存在负值或零）。

分位数变换虽然是一个有用的工具，但有时会使模型的可解释性更具挑战性，因为变换后的值不再与原始尺度有直接的线性关系。然而，对于许多预测建模任务，模型性能的提升通常会弥补这一缺点。

## 参考资料

- [sklearn.preprocessing.QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html) — scikit-learn developers (2024)
  Publisher: scikit-learn
  提供Scikit-learn `QuantileTransformer` 类的官方文档，详细说明其参数、用法和工作机制。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGa58jXFcO4A-bmHA25M8_4bBhcyv3_eCjzyZzC7tCYmrgKvXzXzmzjaXWwlX4cfTpixxmZFiYQwE7RVop2IVZYD1XGCcri1Y4BYl46bcR64VCUBaDb9ZSyVj2ZM0Ucdsko-xhceTjeG01Z1-QDS13o5KiuQrWv6buxcjtuQRt-wOHW2oF6jQ==) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用指南，包含Scikit-learn中数据预处理技术的详细章节，展示分位数缩放等变换如何在机器学习流程中应用。

---
course: "intro-feature-engineering"
chapter: "feature-creation"
lesson: "polynomial-features"
sourceId: 1374
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-5-feature-creation/polynomial-features"
title: "多项式特征"
description: "生成多项式特征（例如平方项、立方项、高次项）以对非线性关系进行建模。"
order: 3
plots: ["plots/1374-0.json"]
sourceHash: "70a8285aaa1b7710e9c13a5fd4f95fb9b25d552f8678bd7844c6104e84a519b7"
sourceCorrections: []
---

线性模型，如线性回归或逻辑回归，功能强大且易于理解，但它们天生假定特征与目标变量之间存在线性关系。当这种关系不是直线时会发生什么？一种扩展这些模型以捕捉非线性模式的方法是创建多项式特征。

本质上，多项式特征是通过将现有数值特征提升到一定幂次（如 $x^2$, $x^3$）或将特征相乘（交互项，如 $x_1 x_2$）而得到的新特征。通过将这些非线性项添加到数据集中，线性模型可以学习曲线关系。

考虑一个包含一个特征 $x$ 的简单数据集。如果与目标 $y$ 的真实关系是二次的，例如 $y \approx ax^2 + bx + c$，那么拟合 $y \approx wx + b$ 的标准线性模型将表现不佳。然而，如果我创建一个新特征 $x^2$ 并使用 $x$ 和 $x^2$ 拟合一个线性模型，则模型变为 $y \approx w_1 x + w_2 x^2 + b$。就*系数* ($w_1, w_2, b$) 而言，这仍然是一个线性模型，但它现在可以对原始特征 $x$ 与目标 $y$ 之间的二次关系进行建模。

### 使用 Scikit-learn 生成多项式特征

Scikit-learn 在其 `preprocessing` 模块中提供了一个便捷的转换器 `PolynomialFeatures`，可以自动生成这些特征。

让我们看看它的实际运用。假设我们有一个包含两个特征 `f1` 和 `f2` 的简单数据集：

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# 样本数据：3个样本，2个特征
X = np.array([[2, 3],
              [4, 1],
              [0, 5]])

# 初始化用于2次多项式的PolynomialFeatures转换器
# include_bias=False 移除常数项（由1组成的列）
poly = PolynomialFeatures(degree=2, include_bias=False)

# 拟合并转换数据
X_poly = poly.fit_transform(X)

print("原始特征：\n", X)
print("\n多项式特征（degree=2）：\n", X_poly)
print("\n特征名称：", poly.get_feature_names_out(['f1', 'f2']))
```

输出将是：

```
Original features:
 [[2 3]
  [4 1]
  [0 5]]

Polynomial features (degree=2):
 [[ 2.  3.  4.  6.  9.]  # f1, f2, f1^2, f1*f2, f2^2
  [ 4.  1. 16.  4.  1.]
  [ 0.  5.  0.  0. 25.]]

Feature names: ['f1' 'f2' 'f1^2' 'f1 f2' 'f2^2']
```

正如所见，`PolynomialFeatures(degree=2)` 生成了原始特征 (`f1`, `f2`)、平方项 (`f1^2`, `f2^2`) 和交互项 (`f1*f2`)。

`PolynomialFeatures` 的主要参数 (parameter)有：

- `degree`: 最大多项式特征的次数。次数为2会生成高达 $x^2$, $x_1 x_2$ 的项；次数为3会生成高达 $x^3$, $x_1^2 x_2$, $x_1 x_2^2$ 等的项。
- `interaction_only`: 如果设置为 `True`，则只生成交互特征（不同特征的乘积，如 $x_1 x_2$），而不生成单个特征的高次项（如 $x_1^2$）。默认为 `False`。
- `include_bias`: 如果设置为 `True`（默认值），它会包含一个偏置 (bias)列（只包含1的特征）。这对于线性模型通常很有用，但如果后续的估计器处理截距，有时可能会冗余。我们在示例中将其设置为 `False` 以求清晰。

### 可视化影响

让我们可视化添加多项式特征如何使线性模型拟合非线性数据。我们将创建合成数据，其中 $y$ 大约是 $x$ 的二次函数，并带有一些噪声。

````python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# 假设 plotly 已导入为 px，graph_objects 已导入为 go

# 生成合成非线性数据
np.random.seed(42)
n_samples = 100
X = np.random.rand(n_samples, 1) * 10 - 5 # 特征值在 -5 到 5 之间
y = 0.8 * X**2 + 0.5 * X + 2 + np.random.randn(n_samples, 1) * 4 # 二次关系 + 噪声

# 1. 拟合标准线性回归
linear_reg = LinearRegression()
linear_reg.fit(X, y)
y_pred_linear = linear_reg.predict(X)

# 2. 创建多项式特征（2次）并拟合线性回归
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# 在网格上创建预测以获得平滑的线
X_grid = np.arange(-5, 5, 0.1).reshape(-1, 1)
X_grid_poly = poly_features.transform(X_grid)
y_pred_poly = poly_reg.predict(X_grid_poly)
y_pred_linear_grid = linear_reg.predict(X_grid) # 网格上的线性模型预测

# 创建 Plotly 图表
import plotly.graph_objects as go

fig = go.Figure()

# 添加原始数据的散点图
fig.add_trace(go.Scatter(x=X.flatten(), y=y.flatten(), mode='markers', name='原始数据',
                         marker=dict(color='#228be6', opacity=0.7)))

# 添加标准线性回归拟合的线
fig.add_trace(go.Scatter(x=X_grid.flatten(), y=y_pred_linear_grid.flatten(), mode='lines', name='线性拟合',
                         line=dict(color='#fa5252', width=2)))

# 添加多项式回归拟合的线
fig.add_trace(go.Scatter(x=X_grid.flatten(), y=y_pred_poly.flatten(), mode='lines', name='多项式拟合（2次）',
                         line=dict(color='#51cf66', width=2)))

fig.update_layout(
    title="线性回归与多项式回归拟合对比",
    xaxis_title="特征 (x)",
    yaxis_title="目标 (y)",
    legend_title="模型",
    template="plotly_white",
    width=700,
    height=400,
    margin=dict(l=20, r=20, t=50, b=20) # 减少边距
)

# fig.show() # 在实际环境中，这将显示图表

# 图表 JSON (单行用于嵌入)
chart_json = fig.to_json(pretty=False)
print(f"```plotly\n{chart_json}\n```")
````



![线性回归与多项式回归拟合对比](plots/1374-0.json)



> 标准线性拟合（红线）未能捕捉数据的曲线。多项式拟合（绿线），使用2次特征（$x$ 和 $x^2$），更好地对潜在的二次关系进行建模。

### 考量与最佳实践

虽然功能强大，但多项式特征需要仔细考量：

1. **选择次数**：多项式的次数是一个超参数 (parameter) (hyperparameter)。低次数可能不够灵活，无法捕捉潜在模式（欠拟合 (underfitting)），而非常高次则可能导致模型过于复杂，过度拟合训练数据中的噪声（过拟合 (overfitting)）。最佳次数通常通过交叉验证来确定。
2. **维度爆炸**：生成的特征数量随着次数和原始特征数量的增加而迅速增长。对于 $n$ 个原始特征和次数 $d$，结果特征的数量（包括偏置 (bias)）由二项式系数 $\binom{n+d}{d} = \frac{(n+d)!}{d!n!}$ 给出。这会变得计算成本高昂，并增加过拟合的风险（“维度灾难”）。
3. **特征缩放**：在应用 `PolynomialFeatures` *之前*，通常重要对特征进行缩放（例如，使用 `StandardScaler` 或 `MinMaxScaler`）。这是因为高次多项式项可能会导致非常大或非常小的值，可能引起数值不稳定，或者使模型对具有自然较大范围的特征敏感。缩放可确保所有特征更均匀地贡献。
4. **正则化 (regularization)**：在使用带有线性模型的多项式特征时，几乎总是建议使用正则化（如 Ridge、Lasso 或 ElasticNet）。正则化有助于约束模型系数，防止它们变得过大，从而减少过拟合，尤其是在存在许多多项式特征时。Lasso (L1 正则化) 甚至可以通过将某些系数精确地缩小到零来执行隐式特征选择。

以下是您如何使用 Scikit-learn 管道集成缩放、多项式特征生成和正则化线性模型的方法：

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
# 假设前述示例中的 X 和 y 可用

# 创建一个管道
poly_pipeline = Pipeline([
    ('scaler', StandardScaler()), # 首先缩放特征
    ('poly', PolynomialFeatures(degree=2, include_bias=False)), # 生成多项式特征
    ('ridge_reg', Ridge(alpha=1.0)) # 使用 Ridge 回归进行正则化
])

# 拟合管道
poly_pipeline.fit(X, y)

# 进行预测（管道处理缩放和转换）
# y_pred_pipeline = poly_pipeline.predict(X)
print("管道拟合成功。")
# print("前5个预测：", y_pred_pipeline[:5].flatten())
```

总而言之，多项式特征提供了一种直接的方法，为本质上是线性的模型增加非线性能力。通过生成平方项、立方项和交互项，您赋予这些模型学习更复杂模式的能力。然而，这种能力也伴随着管理增大的特征空间和过拟合的可能性的责任，通常需要仔细选择次数、特征缩放和正则化。

## 参考资料

- [sklearn.preprocessing.PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) — Scikit-learn developers (2023)
  Scikit-learn `PolynomialFeatures` 转换器的官方文档，详细介绍了其参数、用法以及生成多项式和交互项的示例。
- [An Introduction to Statistical Learning: with Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2021)
  Publisher: Springer; Pages: Chapter 7; DOI: [10.1007/978-1-0716-1418-1](https://doi.org/10.1007/978-1-0716-1418-1)
  一本广泛使用的教材，介绍了多项式回归作为将线性模型扩展到非线性关系的方法，涵盖了其基本原理和实际应用。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://hastie.su.domains/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; Pages: Chapter 5; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本高级教材，从严格的统计和数学角度处理基函数展开（包括多项式回归），并讨论了模型复杂度、正则化和维度灾难等相关主题。
- [User Guide: Pipeline and Composite Estimators](https://scikit-learn.org/stable/modules/compose.html) — Scikit-learn developers (2024)
  Publisher: Scikit-learn project
  官方文档，详细介绍了 Scikit-learn Pipeline 的使用，用于链接多个预处理步骤和估计器，对于涉及特征转换和模型拟合的机器学习工作流程至关重要。

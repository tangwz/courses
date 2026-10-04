---
course: "probability-statistics-essentials-ml"
chapter: "introduction-regression-analysis"
lesson: "practice-fitting-evaluating-linear-model"
sourceId: 1447
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-6-introduction-regression-analysis/practice-fitting-evaluating-linear-model"
title: "动手实践：拟合与评估线性模型"
description: "应用回归技术，使用Python将线性模型拟合到数据并评估其性能。"
order: 8
plots: ["plots/1447-0.json", "plots/1447-1.json", "plots/1447-2.json"]
sourceHash: "0470ef224ae4d0c95949fabab1f8fb637a147d33b896484ee132f1ea85b75fde"
sourceCorrections: []
---

简单线性回归模型 $y = \beta_0 + \beta_1 x + \epsilon$ 涉及使用最小二乘法估计其系数 \beta\_0（截距）和 \beta\_1（斜率）。模型评估通常使用 $R^2$ 和均方误差（MSE）等指标。我们将逐步操作，使用Python将此类模型应用于数据并解读结果。

我们将使用常用Python库：Pandas用于数据处理，Matplotlib/Seaborn或Plotly用于可视化，以及Statsmodels和Scikit-learn用于构建回归模型本身。Statsmodels通常提供更详细的统计汇总，对推断很有用，而Scikit-learn在机器学习 (machine learning)流程中广泛用于预测任务。我们将查看这两种方法。

### 设置环境和数据

首先，请确保您已安装所需的库。如果尚未安装，通常可以使用pip进行安装：

```bash
pip install pandas numpy statsmodels scikit-learn plotly
```

现在，让我们导入它们并创建一些示例数据。假设我们有一些数据，记录了广告支出（单位：千美元）和相应的销售额（单位：千件）。

```python
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# 生成合成数据，以便结果可复现
np.random.seed(42) # 用于结果可复现
advertising_spend = np.random.rand(50) * 10 # 支出从0到1万美元
# 销售额 = 大约50 + 3*支出 + 一些噪音
sales = 50 + 3 * advertising_spend + np.random.randn(50) * 5

# 创建一个Pandas数据框
data = pd.DataFrame({'AdvertisingSpend': advertising_spend, 'Sales': sales})

print(data.head())
# 输出：
#    AdvertisingSpend      Sales
# 0          3.745401  61.800130
# 1          9.507143  78.739061
# 2          7.319939  70.486096
# 3          5.986585  68.583330
# 4          1.560186  54.389472
```

### 数据初步可视化

在拟合模型之前，最好先可视化变量之间的关系。散点图很适合这样做。

```python
# 使用Plotly Express创建散点图
fig_scatter = px.scatter(data, x='AdvertisingSpend', y='Sales',
                         title='销售额与广告支出关系图',
                         labels={'AdvertisingSpend': '广告支出 (千美元)', 'Sales': '销售额 (千件)'},
                         template='plotly_white') # 使用一个简洁的模板

# 优化布局以供网页显示
fig_scatter.update_layout(
    title_x=0.5, # 标题居中
    margin=dict(l=40, r=40, t=50, b=40), # 调整边距
    width=600, # 设置宽度
    height=400 # 设置高度
)

# 显示图表（在Jupyter Notebook/脚本中）或转换为JSON以供网页嵌入
# fig_scatter.show()

# 用于网页嵌入：
scatter_json = fig_scatter.to_json(pretty=False)
```



![销售额与广告支出关系图](plots/1447-0.json)



> 显示广告支出与销售额之间关系的散点图。存在明显的正向线性趋势。

散点图表明存在正向线性关系：广告支出增加，销售额也倾向于增加。这种视觉确认支持了使用线性回归模型。

### 使用Statsmodels拟合线性回归模型

Statsmodels提供了一个`OLS`（普通最小二乘法）类，我们可以使用它。它需要我们显式地为预测变量添加一个常数项（即截距 $\beta_0$）。

```python
# 为Statsmodels准备数据
X = data['AdvertisingSpend']
y = data['Sales']
X = sm.add_constant(X) # 为预测变量添加截距项

# 拟合OLS模型
model_sm = sm.OLS(y, X)
results_sm = model_sm.fit()

# 打印模型汇总结果
print(results_sm.summary())
```

此汇总输出包含丰富信息：

```
                            OLS Regression Results
==============================================================================
Dep. Variable:                  Sales   R-squared:                       0.891
Model:                            OLS   Adj. R-squared:                  0.888
Method:                 Least Squares   F-statistic:                     391.6
Date:                Wed, 15 May 2024   Prob (F-statistic):           1.33e-24
Time:                        12:00:00   Log-Likelihood:                -130.91
No. Observations:                  50   AIC:                             265.8
Df Residuals:                      48   BIC:                             269.6
Df Model:                           1
Covariance Type:            nonrobust
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
const               50.9489      0.865     58.928      0.000      49.211      52.687
AdvertisingSpend     2.9094      0.147     19.790      0.000       2.614       3.205
==============================================================================
Omnibus:                        0.513   Durbin-Watson:                   2.205
Prob(Omnibus):                  0.774   Jarque-Bera (JB):                0.621
Skew:                           0.178   Prob(JB):                        0.733
Kurtosis:                       2.595   Cond. No.                         11.7
==============================================================================

Notes:
[1] 标准误差假设误差的协方差矩阵是正确指定的。
```

**解读Statsmodels汇总结果：**

- **`Dep. Variable`**：确认“Sales”是我们的因变量。
- **`Model`**：OLS（普通最小二乘法）。
- **`R-squared`**：0.891。这表示使用此线性模型，大约89.1%的销售额方差可以由“AdvertisingSpend”解释。这对我们的合成数据来说是一个很好的拟合。
- **`Adj. R-squared`**：0.888。与R平方类似，但根据预测变量的数量进行了调整。在比较具有不同数量预测变量的模型时非常有用。
- **`F-statistic`**：391.6。检验模型的整体显著性。
- **`Prob (F-statistic)`**：1.33e-24。这是与F统计量相关的p值。一个非常小的值（通常小于0.05）表明模型作为一个整体具有统计显著性。我们的模型高度显著。
- **`coef`**：
  - `const`：50.95。这是估计的截距（$\hat{\beta}_0$）。这表明如果广告支出为零，预期销售额约为50.95千件。
  - `AdvertisingSpend`：2.91。这是估计的斜率（$\hat{\beta}_1$）。这表明每额外支出一千美元的广告费，销售额预计增加约2.91千件。
- **`std err`**：系数估计值的标准误差，衡量其精确度。
- **`t`**：每个系数的t统计量值，检验系数为零的零假设。
- **`P>|t|`**：与t统计量相关的p值。常数项和“AdvertisingSpend”的p值都非常小（0.000），这表明两者在模型中都是统计上显著的预测变量。
- **`[0.025 0.975]`**：系数的95%置信区间。我们有95%的把握认为真实的截距在49.21到52.69之间，广告支出的真实斜率在2.61到3.21之间。

汇总的底部部分包含了与模型假设相关的诊断测试（如残差的正态性和独立性），我们将在后面提及。

### 使用Scikit-learn拟合线性回归模型

现在让我们使用Scikit-learn执行相同的任务。它更侧重于预测方面。

```python
# 为Scikit-learn准备数据
# X需要是一个2D数组（或DataFrame）
X_sk = data[['AdvertisingSpend']] # 注意双括号
y_sk = data['Sales']

# 初始化并拟合模型
model_sk = LinearRegression()
model_sk.fit(X_sk, y_sk)

# 获取系数
intercept = model_sk.intercept_ # Beta_0
coefficient = model_sk.coef_[0] # Beta_1

print(f"Scikit-learn 截距 (beta_0): {intercept:.4f}")
print(f"Scikit-learn 系数 (beta_1): {coefficient:.4f}")
# Output:
# Scikit-learn 截距 (beta_0): 50.9489
# Scikit-learn 系数 (beta_1): 2.9094

# 对训练数据进行预测
y_pred_sk = model_sk.predict(X_sk)

# 计算评估指标
mse = mean_squared_error(y_sk, y_pred_sk)
r2 = r2_score(y_sk, y_pred_sk) # 与model_sk.score(X_sk, y_sk)相同

print(f"Scikit-learn 均方误差 (MSE): {mse:.4f}")
print(f"Scikit-learn R平方 (R2): {r2:.4f}")
# Output:
# Scikit-learn 均方误差 (MSE): 24.2842
# Scikit-learn R平方 (R2): 0.8906
```

你会注意到系数（$\hat{\beta}_0 \approx 50.95$，$\hat{\beta}_1 \approx 2.91$）和 $R^2$ 值（0.891）与从Statsmodels获得的基本相同。Scikit-learn提供对系数和MSE、$R^2$等常见评估指标的便捷访问，但不会自动生成Statsmodels那样的详细统计汇总。

### 拟合线的可视化

让我们可视化我们拟合的直线如何体现数据。我们可以使用找到的系数，在散点图上绘制回归线。

```python
# 再次创建散点图
fig_line = px.scatter(data, x='AdvertisingSpend', y='Sales',
                      title='销售额与广告支出关系图及拟合线',
                      labels={'AdvertisingSpend': '广告支出 (千美元)', 'Sales': '销售额 (千件)'},
                      template='plotly_white')

# 添加回归线
# 使用任一模型的系数（它们是相同的）
# 直线方程：y = 截距 + 系数 * x
fig_line.add_trace(go.Scatter(x=data['AdvertisingSpend'], y=y_pred_sk, # 使用预测值作为直线的y值
                              mode='lines',
                              name='拟合线',
                              line=dict(color='#fa5252', width=2))) # 使用调色板中的红色

# 优化布局
fig_line.update_layout(
    title_x=0.5,
    margin=dict(l=40, r=40, t=50, b=40),
    width=600,
    height=400,
    showlegend=True
)

# 转换为JSON以供网页嵌入
line_json = fig_line.to_json(pretty=False)
```



![销售额与广告支出关系图及拟合线](plots/1447-1.json)



> 销售额与广告支出的散点图，叠加了OLS回归线。该直线似乎很好地捕捉了数据的核心趋势。

### 模型假设检查（简要）

如前所述，线性回归依赖于一些假设，以确保结果（特别是p值和置信区间）的可靠性。这些包括：

1. **线性性**：X和y之间的关系是线性的。（我们已通过可视化进行检查）。
2. **独立性**：误差（残差）彼此独立。（通常根据数据收集背景或Statsmodels汇总结果中的Durbin-Watson等特定测试进行评估）。
3. **同方差性**：误差在X的所有水平上具有恒定的方差。
4. **正态性**：误差呈正态分布。

检查同方差性（恒定方差）和线性性的常用视觉方法是绘制残差（$\hat{\epsilon} = y - \hat{y}$）与拟合值（$\hat{y}$）的散点图。

```python
# 计算残差（使用Statsmodels结果）
residuals = results_sm.resid
fitted_values = results_sm.fittedvalues

# 使用Plotly创建残差图
fig_resid = go.Figure()
fig_resid.add_trace(go.Scatter(x=fitted_values, y=residuals, mode='markers',
                               marker=dict(color='#1c7ed6', size=6), # 蓝色
                               name='残差'))
# 添加零点水平线
fig_resid.add_hline(y=0, line_width=2, line_dash="dash", line_color="#868e96") # 灰色虚线

fig_resid.update_layout(
    title='残差与拟合值关系图',
    xaxis_title='拟合值（预测销售额）',
    yaxis_title='残差',
    template='plotly_white',
    title_x=0.5,
    margin=dict(l=40, r=40, t=50, b=40),
    width=600,
    height=400,
    showlegend=False
)

# 转换为JSON以供网页嵌入
resid_json = fig_resid.to_json(pretty=False)
```



![残差与拟合值关系图](plots/1447-2.json)



> 残差（实际销售额 - 预测销售额）与拟合（预测）销售额值的图。理想情况下，点应在零点水平线周围随机散布，没有明显模式。

在理想的残差图中，点应在零点水平线周围随机散布，不显示任何明显模式（如曲线或漏斗形状）。我们的图表看起来相当随机，表明线性性和同方差性假设可能成立。漏斗形状（方差随拟合值增加或减少）将表明存在异方差性。曲线模式将表明线性模型可能不是最佳拟合。

通常使用正式的测试和其他图（如用于正态性的Q-Q图）来更严格地评估这些假设。

### 实践步骤总结

在本动手实践部分，我们完成了以下过程：

1. **数据加载与可视化**：使用Pandas和Plotly准备数据并目视检查变量之间的关系。
2. **模型拟合**：采用Statsmodels（`sm.OLS`）和Scikit-learn（`LinearRegression`），使用普通最小二乘法估计回归系数。
3. **结果解读**：分析输出，特别是Statsmodels提供的系数、$R^2$、MSE和统计显著性。
4. **拟合评估**：计算性能指标（$R^2$、MSE），并可视化拟合线与实际数据的对比。
5. **假设检查（简要）**：引入残差分析，检查潜在问题，如非线性或非恒定方差。

这个应用示例展示了如何将简单线性回归的理论转化为可操作的代码和分析。理解这些步骤是转向多重线性回归等更复杂模型前的重要一步，在那些模型中我们使用多个预测变量。

## 参考资料

- [An Introduction to Statistical Learning with Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani (2013)
  Publisher: Springer; DOI: [10.1007/978-1-4614-7138-7](https://doi.org/10.1007/978-1-4614-7138-7)
  一本广受好评的入门教材，涵盖了线性回归、统计推断和机器学习概念。提供了坚实的理论基础和实际示例。
- [statsmodels.regression.linear_model.OLS](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html) — Statsmodels Developers (2024)
  Statsmodels中普通最小二乘（OLS）模型的官方文档，详细说明了其用法以及综合摘要输出的解释。
- [1.1. Linear Models - scikit-learn documentation](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares) — Scikit-learn Developers (2024)
  Scikit-learn中线性模型的官方文档，包括LinearRegression，侧重于预测任务的实现和性能评估。

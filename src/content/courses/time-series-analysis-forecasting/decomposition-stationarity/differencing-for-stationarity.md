---
course: "time-series-analysis-forecasting"
chapter: "decomposition-stationarity"
lesson: "differencing-for-stationarity"
sourceId: 3398
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-2-decomposition-stationarity/differencing-for-stationarity"
title: "实现平稳性：差分"
description: "了解如何应用差分方法使非平稳时间序列变为平稳。"
order: 6
plots: ["plots/3398-0.json"]
sourceHash: "21063eb371b3a216f5a45ce7e0f843f263114b53425b48ff16c9f59c0b93c5f0"
sourceCorrections: []
---

许多时间序列表现出趋势或季节性，使得它们的统计特性（如均值和方差）随时间变化。这种非平稳性对假定数据平稳的ARMA等标准预测模型构成难题。幸运的是，将非平稳数据转换为平稳数据的一种常用且有效的方法是**差分**。

### 差分的原理

差分计算时间序列中连续观测值之间的变化。对于时间序列 $y_t$，一阶差分记作 $\Delta y_t$，计算方式为：

$\Delta y_t = y_t - y_{t-1}$

这种简单的操作通常可以通过消除趋势来稳定时间序列的均值。设想一个具有线性上升趋势的序列。$y_t$ 的值持续增长。然而，连续值之间的*差值*（$y_t - y_{t-1}$）可能大致保持不变，围绕一个稳定的均值波动（代表趋势的斜率）。

### 通过差分消除趋势

让我们考虑一个具有明显趋势的时间序列。应用一阶差分有助于消除此趋势，使序列在均值上平稳。

```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller

# 生成带有趋势的样本数据
np.random.seed(42)
time = pd.date_range(start='2022-01-01', periods=100, freq='D')
# 趋势成分 + 随机噪声
trend = np.linspace(0, 20, 100)
noise = np.random.normal(0, 2, 100)
data = pd.Series(trend + noise, index=time, name='Original Data')

# 计算一阶差分
differenced_data = data.diff().dropna() # dropna 删除第一个 NaN 值

# 检查差分前后的平稳性
adf_original = adfuller(data)
adf_differenced = adfuller(differenced_data)

print(f"原始数据上的ADF检验：p值 = {adf_original[1]:.3f}")
print(f"差分数据上的ADF检验：p值 = {adf_differenced[1]:.3f}")

# 创建图表
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='原始序列 (趋势)', line=dict(color='#4263eb')))
fig.add_trace(go.Scatter(x=differenced_data.index, y=differenced_data, mode='lines', name='一阶差分', line=dict(color='#12b886')))
fig.update_layout(
    title='一阶差分对趋势的影响',
    xaxis_title='时间',
    yaxis_title='值',
    legend_title='序列',
    template='plotly_white',
    width=700,
    height=400
)
# fig.show() # 在实际环境中
```

> 
> 
> ![一阶差分对趋势的影响](plots/3398-0.json)
> 
> 
>
> 原始序列显示出明显的上升趋势，并且ADF检验未能拒绝非平稳性的零假设（p值较高）。应用一阶差分后，结果序列在零附近波动，显得更为平稳。差分序列上的ADF检验现在得到一个非常小的p值，有力地表明序列已平稳。

### 高阶差分

有时，仅一次差分步骤不足以达到目的。例如，具有二次趋势的数据可能需要进行两次差分才能实现平稳性。二阶差分就是一阶差分的差值：

$\Delta^2 y_t = \Delta (\Delta y_t) = \Delta(y_t - y_{t-1}) = (y_t - y_{t-1}) - (y_{t-1} - y_{t-2})$

实际上，通常不需要进行两次以上的差分（即 $d=1$ 或 $d=2$）。你可以通过两次调用 Pandas 中的 `.diff()` 方法来实现二阶差分：

```python
# 二阶差分
differenced_data_2 = data.diff().diff().dropna()
```

在每次差分后，务必（通过视觉检查和使用ADF等检验）检查序列是否平稳。过度差分（差分的次数超过必要）可能会引入不必要的自相关并使建模复杂化。

### 季节性差分

如果你的数据表现出季节性，仅进行一阶差分可能不足以消除重复的季节性模式。**季节性差分**是指计算一个观测值与前一个季节（或周期）中对应观测值之间的差值。

如果 $m$ 是季节性周期（例如，月度数据 $m=12$，季度数据 $m=4$，具有周度模式的日数据 $m=7$），则季节性差分为：

$\Delta_m y_t = y_t - y_{t-m}$

此操作通过将当前值与一个周期前对应的值进行比较来消除季节性成分。在 Pandas 中，你可以使用 `.diff()` 方法中的 `periods` 参数 (parameter)执行季节性差分：

```python
# 例子：生成具有年度季节性（m=12）的数据
time_monthly = pd.date_range(start='2018-01-01', periods=48, freq='MS')
seasonal_component = np.tile(np.sin(np.linspace(0, 2*np.pi, 12)), 4) * 5
trend_monthly = np.linspace(0, 10, 48)
noise_monthly = np.random.normal(0, 1, 48)
data_monthly = pd.Series(trend_monthly + seasonal_component + noise_monthly, index=time_monthly, name='Monthly Data')

# 季节性差分 (m=12)
seasonal_diff_data = data_monthly.diff(periods=12).dropna()

# 有时需要同时进行普通差分和季节性差分
# 首先进行季节性差分，然后进行普通差分
combined_diff_data = data_monthly.diff(periods=12).diff(periods=1).dropna()

# 接下来会进行绘图或ADF检验以确认平稳性...
```

通常，同时具有趋势和季节性的数据可能需要*同时*进行非季节性的一阶差分（$\Delta y_t$）和季节性差分（$\Delta_m y_t$）。常用做法是先应用季节性差分，然后对结果应用非季节性差分。

### 积分与ARIMA模型

为了实现数据平稳性而进行差分的次数是时间序列建模中的一个重要参数 (parameter)。这个差分阶数在非季节性ARIMA(p, d, q)模型中由参数 'd' 表示，在季节性SARIMA(p, d, q)(P, D, Q)m 模型中由参数 'D' 表示。这些模型会根据这些参数在内部处理差分。

当使用依赖差分数据的ARIMA或SARIMA等模型进行预测时，最终预测结果需要转换回原始尺度。这种反向操作称为**积分**（因此ARIMA中的'I'指代此意），它涉及累加差分。当从已拟合的ARIMA/SARIMA模型生成预测结果时，`statsmodels` 等库会自动处理这种积分。

总之，差分是一种使时间序列数据平稳的基本方法，尤其用于消除趋势和季节性。通过应用一阶、季节性或偶尔二阶差分，你为依赖平稳性假设的模型准备好数据。记住在差分后，始终通过视觉检查和统计检验来确认平稳性。

## 参考资料

- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, Greta M. Ljung (2015)
  Publisher: Wiley
  本书是时间序列分析的基础教材，对ARIMA模型和差分法在实现平稳性方面的作用进行了严谨的阐述。
- [Forecasting: Principles and Practice (3rd ed.)](https://otexts.com/fpp3/) — Rob J Hyndman, George Athanasopoulos (2021)
  Publisher: OTexts
  一本实用且易于理解的预测方法指南，包含对差分法、平稳性及其在ARIMA模型中应用的详细说明，可在线免费阅读。
- [pandas.Series.diff](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.diff.html) — pandas developers ()
  pandas `Series.diff()` 方法的官方文档，用于计算时间序列中连续（或滞后）元素之间的差分。
- [statsmodels.tsa.stattools.adfuller](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html) — statsmodels developers (2024)
  statsmodels中增广迪基-富勒（ADF）检验的官方文档，该统计检验用于判断时间序列是否平稳。

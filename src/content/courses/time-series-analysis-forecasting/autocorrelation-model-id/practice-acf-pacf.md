---
course: "time-series-analysis-forecasting"
chapter: "autocorrelation-model-id"
lesson: "practice-acf-pacf"
sourceId: 3418
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-3-autocorrelation-model-id/practice-acf-pacf"
title: "动手实践：ACF/PACF 图的绘制与解读"
description: "生成平稳时间序列数据的ACF和PACF图，并练习解读结果。"
order: 5
plots: ["plots/3418-0.json", "plots/3418-1.json"]
sourceHash: "5411446197fcb66c55695efe1a7046f777122044b6cf17afcb994d6ddd62cac9"
sourceCorrections: []
---

为时间序列生成自相关函数（ACF）和偏自相关函数（PACF）图，并分析它们以推荐可能的模型阶数。在拟合ARIMA模型之前，此过程非常重要，因为这些图提供了关于*平稳*时间序列内在结构（'p'和'q'参数 (parameter)）的线索。

### 前提条件

我们假设您已经准备好一个平稳时间序列。如果您从非平稳数据开始，则应已应用差分等变换（如第2章所述）以达到平稳性。本次练习中，我们将使用人工生成的平稳数据集，以清楚地展示预期的模式。

您需要以下Python库：

- `numpy` 用于数值运算和生成样本数据。
- `pandas` 用于数据处理（如果直接使用NumPy数组，对于此特定的绘图示例而言，其重要性较低）。
- `statsmodels` 用于ACF/PACF计算和绘图函数。
- `plotly.graph_objects` 用于按要求创建图表（我们将根据`statsmodels`的计算手动构建这些图表）。

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
# 我们将根据要求使用plotly进行可视化
# 注意：statsmodels有其基于matplotlib的plot_acf/plot_pacf函数，
# 但我们将提取数据来创建Plotly图表。
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 设置随机种子以保证结果可重现
np.random.seed(42)

# 生成一个样本平稳AR(2)过程：
# y_t = 0.7*y_{t-1} - 0.3*y_{t-2} + noise
ar_params = np.array([0.7, -0.3])
ma_params = np.array([]) # 没有MA（移动平均）分量
ar = np.r_[1, -ar_params] # 添加零滞后系数
ma = np.r_[1, ma_params] # 添加零滞后系数

# 生成500个数据点
n_samples = 500
# 使用ArmaProcess生成数据（statsmodels的一部分）
from statsmodels.tsa.arima_process import ArmaProcess
ar_process = ArmaProcess(ar, ma)
sample_data = ar_process.generate_sample(nsample=n_samples)

# 转换为pandas Series（可选，但常见做法）
ts = pd.Series(sample_data)

print("生成的样本数据（前5个值）：")
print(ts.head())
print(f"\n生成的数据是否平稳（基于生成过程）？是的，AR(2)过程参数在平稳区域内。")
# 通常您会在真实数据上运行ADF检验
# from statsmodels.tsa.stattools import adfuller
# adf_result = adfuller(ts)
# print(f'ADF Statistic: {adf_result[0]}')
# print(f'p-value: {adf_result[1]}') # 低的p-值表示平稳性
```

### 计算和绘制ACF

ACF测量时间序列$y_t$与其滞后值$y_{t-k}$在不同滞后$k$下的相关性。我们使用`statsmodels.tsa.stattools`中的`acf`函数来计算这些值和相应的置信区间。

滞后$k$处的显著峰值表示相隔$k$个周期的观测值之间存在强相关性。对于MA(q)过程，ACF图预计在滞后$q$之前有显著峰值，然后突然截断（落入置信区间内）。对于AR(p)过程，ACF通常衰减更慢（通常呈几何式衰减或正弦波模式）。

```python
# 计算ACF和置信区间
# nlags指定要计算的滞后数量；alpha指定置信水平（0.05表示95%）
acf_values, confint = acf(ts, nlags=20, alpha=0.05)

# 置信区间数组confint的形状为(nlags+1, 2)
# 下限 = confint[:, 0] - acf_values
# 上限 = confint[:, 1] - acf_values
# 注意：acf_values[0]始终为1（与滞后0的相关性）
lags = np.arange(len(acf_values))
conf_lower = confint[:, 0] - acf_values
conf_upper = confint[:, 1] - acf_values

# 为ACF创建Plotly图
fig_acf = go.Figure()

# 添加置信区间带（不包括滞后0）
fig_acf.add_trace(go.Scatter(
    x=np.concatenate([lags[1:], lags[1:][::-1]]), # 多边形形状的x坐标
    y=np.concatenate([conf_upper[1:], conf_lower[1:][::-1]]), # 多边形形状的y坐标
    fill='toself',
    fillcolor='#a5d8ff', # 浅蓝色
    line=dict(color='rgba(255,255,255,0)'), # 无边框线
    hoverinfo="skip",
    showlegend=False,
    name='置信区间'
))

# 添加ACF条/茎（不包括滞后0）
fig_acf.add_trace(go.Scatter(
    x=lags[1:],
    y=acf_values[1:],
    mode='markers',
    marker=dict(color='#1c7ed6', size=8), # 蓝色点
    name='ACF'
))

# 添加从茎到x轴的垂直线
for i in range(1, len(acf_values)):
    fig_acf.add_shape(type='line',
                      x0=lags[i], y0=0,
                      x1=lags[i], y1=acf_values[i],
                      line=dict(color='#495057', width=1.5)) # 灰色线

# 添加滞后0点（始终为1）
fig_acf.add_trace(go.Scatter(
    x=[lags[0]], y=[acf_values[0]], mode='markers', marker=dict(color='#1c7ed6', size=8), showlegend=False
))
fig_acf.add_shape(type='line', x0=lags[0], y0=0, x1=lags[0], y1=acf_values[0], line=dict(color='#495057', width=1.5))

# 更新布局
fig_acf.update_layout(
    title='自相关函数（ACF）',
    xaxis_title='滞后',
    yaxis_title='自相关',
    yaxis_range=[-1, 1.1], # 确保y轴覆盖整个范围加上滞后0
    xaxis=dict(tickmode='linear', dtick=1), # 显示整数滞后
    plot_bgcolor='white',
    height=350,
    margin=dict(l=50, r=20, t=50, b=40)
)

# 显示图表（在笔记本环境中）或保存它
# fig_acf.show() # 取消注释以交互方式显示
```



![自相关函数（ACF）](plots/3418-0.json)



> 生成的AR(2)数据的ACF图。蓝色阴影区域表示95%置信区间。此区域外的相关性具有统计学意义。

### 计算和绘制PACF

PACF测量在移除中间滞后（$y_{t-1}, y_{t-2}, ..., y_{t-k+1}$）的影响后，$y_t$与$y_{t-k}$之间的相关性。我们使用`statsmodels.tsa.stattools`中的`pacf`函数。

对于AR(p)过程，PACF图预计在滞后$p$之前有显著峰值，然后突然截断。这是因为PACF移除了较短滞后的影响，隔离了AR参数 (parameter)所描述的直接关系。对于MA(q)过程，PACF通常衰减更慢。

```python
# 计算PACF和置信区间
# method='ywm'是默认方法，通常推荐使用
pacf_values, confint_pacf = pacf(ts, nlags=20, alpha=0.05, method='ywm')

# 提取置信区间，与ACF类似
lags_pacf = np.arange(len(pacf_values))
conf_lower_pacf = confint_pacf[:, 0] - pacf_values
conf_upper_pacf = confint_pacf[:, 1] - pacf_values

# 为PACF创建Plotly图
fig_pacf = go.Figure()

# 添加置信区间带（不包括滞后0）
fig_pacf.add_trace(go.Scatter(
    x=np.concatenate([lags_pacf[1:], lags_pacf[1:][::-1]]),
    y=np.concatenate([conf_upper_pacf[1:], conf_lower_pacf[1:][::-1]]),
    fill='toself',
    fillcolor='#a5d8ff', # 浅蓝色
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False,
    name='置信区间'
))

# 添加PACF条/茎（不包括滞后0）
fig_pacf.add_trace(go.Scatter(
    x=lags_pacf[1:],
    y=pacf_values[1:],
    mode='markers',
    marker=dict(color='#7048e8', size=8), # 紫罗兰色点
    name='PACF'
))

# 添加从茎到x轴的垂直线
for i in range(1, len(pacf_values)):
    fig_pacf.add_shape(type='line',
                      x0=lags_pacf[i], y0=0,
                      x1=lags_pacf[i], y1=pacf_values[i],
                      line=dict(color='#495057', width=1.5)) # 灰色线

# 添加滞后0点（根据PACF定义始终为1，但有时在图中省略）
# 对于PACF，我们将省略滞后0处的线/标记，因为它不如ACF那样是标准做法
# fig_pacf.add_trace(go.Scatter(
#     x=[lags_pacf[0]], y=[pacf_values[0]], mode='markers', marker=dict(color='#7048e8', size=8), showlegend=False
# ))

# 更新布局
fig_pacf.update_layout(
    title='偏自相关函数（PACF）',
    xaxis_title='滞后',
    yaxis_title='偏自相关',
    yaxis_range=[-1, 1.1], # 确保y轴覆盖整个范围
    xaxis=dict(tickmode='linear', dtick=1), # 显示整数滞后
    plot_bgcolor='white',
    height=350,
    margin=dict(l=50, r=20, t=50, b=40)
)

# fig_pacf.show() # 取消注释以交互方式显示
```



![偏自相关函数（PACF）](plots/3418-1.json)



> 生成的AR(2)数据的PACF图。蓝色阴影区域表示95%置信区间。

### 解读

现在，让我们解读从样本AR(2)数据生成的图：

1. **ACF图分析：**

   - 观察ACF图。相关性在滞后1处较高（约0.65），然后逐渐减小，可能出现振荡（例如，滞后3为负值，滞后4更负）。
   - 相关性在多个滞后处仍具有统计学意义（在蓝色带外），然后似乎缓慢趋向于零。这种衰减模式是AR过程的特点。它在特定滞后$q$之后*没有*显示出急剧截断，这使得纯MA模型的可能性降低。
2. **PACF图分析：**

   - 现在，看看PACF图。我们在滞后1（正值，约0.65）和滞后2（负值，约-0.3）处看到显著峰值。
   - 重要的是，在滞后2之后，PACF值突然下降并落入置信区间*内*。在滞后2之后，它们不具有统计学意义。
   - 滞后$p=2$之后的这种急剧截断是AR(2)过程的显著特征。
3. **结论：**

   - ACF图显示出衰减模式，表明存在AR分量。
   - PACF图在滞后2之后急剧截断，强烈表明$p=2$。
   - 由于ACF衰减且PACF在滞后2处截断，这些图所建议的最可能模型是\*\*AR(2)\*\*模型。这与我们生成数据的方式一致（$y_t = 0.7y_{t-1} - 0.3y_{t-2} + \epsilon_t$）。

以下是基于*平稳*数据ACF/PACF模式的快速参考指南：

| 过程 | ACF模式 | PACF模式 | 建议模型 |
| --- | --- | --- | --- |
| **AR(p)** | 拖尾（几何/正弦波） | 在滞后**p**后截断 | ARIMA(p, 0, 0) |
| **MA(q)** | 在滞后**q**后截断 | 拖尾（几何/正弦波） | ARIMA(0, 0, q) |
| **ARMA(p,q)** | 拖尾 | 拖尾 | ARIMA(p, 0, q) |

*(请记住，ARIMA(p, d, q)中的“d”与达到平稳性所需的差分有关，这在分析ACF/PACF图之前确定。)*

### 您的练习

现在轮到您自己应用这些方法了。

1. 选择一个您能获取的平稳时间序列数据集（可能是第2章练习中使用的数据的差分版本）。如果您没有现成的数据集，请尝试生成一个MA(1)过程（例如，`y_t = 0.6 * noise_{t-1} + noise_t`），类似于我们上面生成AR(2)数据的方式。
2. 使用所示方法生成ACF和PACF图。
3. 仔细检查这些图：
   - ACF是急剧截断，还是拖尾？
   - PACF是急剧截断，还是拖尾？
   - 相关性是几何式衰减还是振荡？
   - 显著相关性在哪个滞后处截断（如果适用）？
4. 根据模式和参考表，确定ARIMA(p, d, q)模型中最可能的$p$和$q$候选值（其中$d$是您可能之前应用的差分阶数）。

“解读ACF和PACF图通常更像一门艺术而非精确科学，尤其是在处理噪声数据时。有时模式并不完全清晰。然而，它们为识别候选模型提供了不可或缺的起点，您将在接下来的章节中学习如何拟合和评估这些模型。”

## 参考资料

- [Time Series Analysis: Forecasting and Control](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118675074) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, and Greta M. Ljung (2015)
  Publisher: John Wiley & Sons; DOI: [10.1002/9781118675074](https://doi.org/10.1002/9781118675074)
  时间序列分析的经典教材，全面详细地介绍了ACF、PACF和ARIMA模型识别。
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts
  一本易于理解的现代教材，提供了ACF/PACF解释和ARIMA模型构建的实用指导，可免费在线获取。

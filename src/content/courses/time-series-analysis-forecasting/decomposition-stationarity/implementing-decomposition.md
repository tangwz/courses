---
course: "time-series-analysis-forecasting"
chapter: "decomposition-stationarity"
lesson: "implementing-decomposition"
sourceId: 3388
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-2-decomposition-stationarity/implementing-decomposition"
title: "在 Python 中实现分解"
description: "使用 statsmodels 等 Python 库进行时间序列分解。"
order: 3
plots: ["plots/3388-0.json"]
sourceHash: "aaf27ec51c4d5d02a8ef3f167c2df1f230f68343eed3606bf4d4c545f6c72b76"
sourceCorrections: []
---

在 Python 中实现时间序列分解通常涉及使用库和函数来分析趋势、季节性和残差等组成部分。

我们已经讨论了时间序列分解的原理以及加法模型和乘法模型之间的区别。现在，让我们在 Python 中实际操作。`statsmodels` 库提供了统计分析工具，包括时间序列分解功能。

我们将使用的主要函数是 `statsmodels.tsa.seasonal.seasonal_decompose`。这个函数实现了经典的分解方法，通常基于移动平均线，将时间序列分为趋势、季节和残差成分。

### 使用 `seasonal_decompose`

`seasonal_decompose` 函数需要时间序列本身，以及关于模型类型和季节周期的信息。

```python
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt # 使用 matplotlib 进行基本绘图示例

# 假设 'series' 是一个带有 DatetimeIndex 的 pandas Series。
# 它应包含你想要分解的时间序列值。
# 示例：
# series = pd.read_csv('your_data.csv',
#                     index_col='Date',
#                     parse_dates=True)['Value']

# --- 执行分解 ---
# model: 指定 'additive' (加法) 或 'multiplicative' (乘法)。
# period: 每个季节周期内的观测数量
#         (例如，月度数据具有年度季节性时为 12，
#          日度数据具有每周季节性时为 7)。

# 针对月度数据（周期=12）的加法模型示例
decomposition_result = seasonal_decompose(series, model='additive', period=12)

# 该函数返回一个 DecomposeResult 对象。你可以通过
# 属性访问各个成分：
trend_component = decomposition_result.trend
seasonal_component = decomposition_result.seasonal
residual_component = decomposition_result.resid
observed_data = decomposition_result.observed # 原始数据

# --- 可视化成分 ---
# 一种常见的可视化方式是堆叠图表：
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

observed_data.plot(ax=axes[0], legend=False, color='#1c7ed6')
axes[0].set_ylabel('观测值')

trend_component.plot(ax=axes[1], legend=False, color='#f76707')
axes[1].set_ylabel('趋势')

seasonal_component.plot(ax=axes[2], legend=False, color='#37b24d')
axes[2].set_ylabel('季节')

residual_component.plot(ax=axes[3], legend=False, color='#adb5bd')
axes[3].set_ylabel('残差')

plt.xlabel('日期')
plt.suptitle('时间序列分解', y=0.92) # 在图表上方稍加标题
plt.tight_layout(rect=[0, 0, 1, 0.9]) # 调整布局以避免标题重叠
plt.show()
```

**选择模型类型：**

- 当季节性波动随时间推移显得相对稳定，无论序列的数值水平如何时，设置 `model='additive'`。季节模式中峰值和谷值的幅度大致保持相同。这对应于 $y_t = T_t + S_t + R_t$ 模型。
- 当季节性波动似乎与序列的水平成比例变化时，设置 `model='multiplicative'`。随着趋势的增加，季节模式的振幅也随之增大。这符合 $y_t = T_t \times S_t \times R_t$ 模型。如果你怀疑存在乘法效应，有时分析序列的对数可以使模式变为加法形式 ($log(y_t) \approx log(T_t) + log(S_t) + log(R_t)$)，这可以简化建模过程。

**指定 `period`：**

这个参数 (parameter)对于正确识别季节模式很重要。它表示一个完整季节周期中的时间步长数量。常见值包括：

- 月度数据具有年度周期时，`period=12`。
- 季度数据具有年度周期时，`period=4`。
- 日度数据具有每周周期时，`period=7`。
- 周度数据具有年度周期时，`period=52` (近似值)。

根据数据已知频率选择正确的 `period` 对于进行有意义的分解是必需的。

### 示例：航空旅客数据分解

让我们将此应用于经典的“AirPassengers”数据集，该数据集以其强劲的上升趋势和递增的年度季节性著称。这种递增的季节性表明乘法模型可能更合适。

我们将进行两种分解以作比较。（假设 `air_passengers` 是一个包含月度旅客数量并按日期索引的 pandas Series。）

```python
# 假设 'air_passengers' Series 已加载并准备好 DatetimeIndex
# 演示用合成数据生成示例：
date_rng = pd.date_range(start='1949-01-01', end='1960-12-01', freq='MS')
passengers = (
    110 +  # 基础水平略高
    (date_rng.year - 1949) * 22 +  # 趋势略陡
    (1 + (date_rng.year - 1949) * 0.12) * ( # 季节性增长略强
        35 * (date_rng.month == 7) + 25 * (date_rng.month == 8) +
        -25 * (date_rng.month == 11) + -20 * (date_rng.month == 2)
    ) +
    pd.Series(np.random.normal(0, 12, size=len(date_rng)), index=date_rng) # 噪声略大
).astype(int)
air_passengers = pd.Series(passengers, index=date_rng)

# 执行乘法分解（月度年度周期为 12）
result_mul = seasonal_decompose(air_passengers, model='multiplicative', period=12)

# 执行加法分解以作比较
result_add = seasonal_decompose(air_passengers, model='additive', period=12)

# -- 使用 Plotly 绘图以获得更好的网页显示效果 --
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np # 用于 np.nan 比较

fig = make_subplots(
    rows=4, cols=2,
    shared_xaxes=True,
    subplot_titles=('乘法模型：观测值', '加法模型：观测值',
                    '乘法模型：趋势', '加法模型：趋势',
                    '乘法模型：季节', '加法模型：季节',
                    '乘法模型：残差', '加法模型：残差'),
    vertical_spacing=0.06 # 间距略大
)

# 添加迹线的函数，处理趋势/残差末端的潜在 NaN 值
def add_trace_safe(fig, data, name, color, row, col, showlegend=True):
    # 只绘制非 NaN 数据点
    valid_index = data.dropna().index
    valid_data = data.dropna().values
    fig.add_trace(go.Scatter(x=valid_index, y=valid_data, mode='lines',
                             name=name, line=dict(color=color), showlegend=showlegend),
                  row=row, col=col)

# 按列添加迹线
add_trace_safe(fig, result_mul.observed, '观测值', '#1c7ed6', 1, 1)
add_trace_safe(fig, result_mul.trend, '趋势', '#f76707', 2, 1)
add_trace_safe(fig, result_mul.seasonal, '季节', '#37b24d', 3, 1)
add_trace_safe(fig, result_mul.resid, '残差', '#adb5bd', 4, 1)

add_trace_safe(fig, result_add.observed, '观测值', '#1c7ed6', 1, 2, showlegend=False)
add_trace_safe(fig, result_add.trend, '趋势', '#f76707', 2, 2, showlegend=False)
add_trace_safe(fig, result_add.seasonal, '季节', '#37b24d', 3, 2, showlegend=False)
add_trace_safe(fig, result_add.resid, '残差', '#adb5bd', 4, 2, showlegend=False)

fig.update_layout(
    height=750, # 略高
    title_text='时间序列分解：乘法模型与加法模型对比 (航空旅客)',
    margin=dict(l=60, r=30, t=90, b=60),
    hovermode='x unified',
    legend_title_text='成分',
    plot_bgcolor='#e9ecef', # 浅灰色背景
    paper_bgcolor='#ffffff' # 白色纸张背景
)
fig.update_xaxes(title_text="日期", row=4, col=1, showgrid=False)
fig.update_xaxes(title_text="日期", row=4, col=2, showgrid=False)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#dee2e6') # 浅色网格线

# 生成用于嵌入的 JSON 输出：
# print(fig.to_json())
```



![时间序列分解：乘法模型与加法模型对比 (航空旅客)](plots/3388-0.json)



> 航空旅客数据集中乘法（左列）和加法（右列）分解的比较图。

### 分解图的解释

- **观测值：** 加载后的原始时间序列数据。
- **趋势：** 这个成分分离出序列的长期走向或水平变化。对于航空旅客数据，它准确反映了多年来的持续增长。请注意趋势线开头和结尾处的 `NaN` 值（缺失部分）。这是因为内部使用的移动平均计算需要当前点前后都有数据点。
- **季节：** 该图显示了每个周期（本例中是年度）重复的模式。
  - 在*乘法*分解中，季节成分围绕 1.0 波动。它表示按比例调整趋势的因子（例如，值 1.2 可能表示比该月的趋势水平高 20%）。请注意，这种模式多年来在形态上显得相对稳定。
  - 在*加法*分解中，季节成分围绕 0 波动。它表示添加到趋势或从趋势中减去的绝对值。对于航空旅客数据，请注意这个加法季节成分的振幅如何随时间增加，与旅客总人数的增长相对应。这表明它不能像乘法模型那样有效捕捉季节性，因为在乘法模型中季节性因子保持更一致。
- **残差：** 这是在移除估计的趋势和季节成分后的余下部分。理想情况下，残差应看起来像随机噪声，没有任何明显的模式或结构。
  - 比较两个残差图，*乘法*分解的残差显得更随机，均匀分布在 1.0 附近，并且随时间具有相对恒定的方差。
  - 然而，*加法*分解的残差随时间表现出一定的发散（方差增大），并可能保留一些潜在模式，尤其是在后期。这进一步印证了乘法模型是这个数据集更好的选择，因为它留下的残差更接近随机噪声。

### `seasonal_decompose` 的考量

尽管 `seasonal_decompose` 是一个有用的探索性工具，但请记住以下几点：

1. **方法：** 它通常使用中心化移动平均线。如果趋势变化迅速或数据中存在急剧的峰值或谷值，这种方法效果可能不佳。季节成分是通过对每个季节的去趋势值进行平均，然后对这些因子进行中心化来计算的。
2. **端点问题：** 如前所述，由于移动平均线需要完整的数据窗口，序列起始和结束点附近的趋势和残差估计值通常缺失（`NaN`）或可靠性较低。
3. **静态季节性假设：** 该方法假设季节模式在整个序列中要么是完全加法，要么是完全乘法。而季节性可能在长时间内以更复杂的方式演变。
4. **描述性，非预测性：** 分解主要用于理解数据的历史结构。它不直接生成预测，尽管理解这些成分对于选择和构建预测模型很重要。STL (使用 Loess 的季节和趋势分解) 等更高级的方法可以提供更大的灵活性。

分解为理解时间序列的潜在结构提供了有价值的见解。检查这些成分，尤其是残差，有助于评估原始序列是否表现出清晰的模式，以及移除这些模式后是否会得到更接近随机噪声的结果。这个评估直接引出了平稳性思想，我们将在接下来的章节中使用统计检验对其进行更正式的研究。

## 参考资料

- [seasonal_decompose: Classical time series decomposition](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html) — statsmodels developers (2024)
  `seasonal_decompose`函数的官方文档，介绍了其用法、参数和输出。
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts
  一本广泛使用的教材，清晰地阐述了时间序列分解，包括加法和乘法模型及其解释。
- [Hands-On Time Series Analysis with Python](https://www.packtpub.com/product/hands-on-time-series-analysis-with-python/9781838647039) — Breen, Brandon, Nicholas O'Malley, and John D. Kelleher (2020)
  Publisher: Packt Publishing
  提供了实现各种时间序列技术（包括分解）的实用指南和Python代码示例。
- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, and Greta M. Ljung (2015)
  Publisher: John Wiley & Sons
  时间序列分析的经典教材，涵盖了经典分解方法及其统计基础。

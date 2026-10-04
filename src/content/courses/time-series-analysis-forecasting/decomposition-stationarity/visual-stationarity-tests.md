---
course: "time-series-analysis-forecasting"
chapter: "decomposition-stationarity"
lesson: "visual-stationarity-tests"
sourceId: 3392
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-2-decomposition-stationarity/visual-stationarity-tests"
title: "平稳性检验：可视化检查"
description: "如何通过查看时间序列图和滚动统计量来可视化评估平稳性。"
order: 4
plots: ["plots/3392-0.json", "plots/3392-1.json", "plots/3392-2.json"]
sourceHash: "49cfa761ba86684437fb2a42e358304850de4dd6ce59d1220c4e60b5d0186895"
sourceCorrections: []
---

平稳性是时间序列分析中的一项基本特性。验证此条件非常重要，因为许多统计模型，包括ARIMA，都假定底层时间序列数据表现出平稳性。评估平稳性的一种主要且直观的方法是可视化检查。虽然单独的可视化检查不具决定性，但绘制数据及其滚动统计量图通常能显示出明显的趋势或方差变化，这些都表明数据不平稳。

### 直接绘制时间序列图

最基本的检查只是简单地绘制时间序列值随时间变化的图。请查找以下图示特征：

1. **趋势：** 是否有明显的长期上升或下降斜率？一致的趋势表明均值不恒定，这违反了平稳性要求。
2. **季节性：** 是否存在以固定时间间隔（例如，每日、每月、每年）重复出现的模式？虽然平稳性的一些定义允许存在可预测的季节性，但强烈的季节性模式通常需要特殊处理（例如季节性差分或SARIMA模型，稍后讨论）。目前，请将其识别为可能不符合简单的平稳性假设的模式。
3. **方差变化：** 波动幅度是否随时间显著变化？例如，序列在后期是否变得波动性更大？这表明存在非恒定方差（异方差性），这是又一个不平稳的表现。

我们来看一个使用Python的例子。假设您的时间序列数据已加载到以时间为索引的Pandas Series或DataFrame中。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go # 使用 Plotly 绘制交互式图表

# 示例数据：创建带有趋势的非平稳序列
dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
data_nonstationary = np.linspace(10, 25, 50) + np.random.normal(0, 2, 50)
ts_nonstationary = pd.Series(data_nonstationary, index=dates)

# 示例数据：创建可能平稳的序列
data_stationary = np.random.normal(15, 3, 50)
ts_stationary = pd.Series(data_stationary, index=dates)

# --- 绘制非平稳数据 ---
fig_nonstationary = go.Figure()
fig_nonstationary.add_trace(go.Scatter(x=ts_nonstationary.index, y=ts_nonstationary.values,
                                     mode='lines', name='时间序列', line=dict(color='#339af0'))) # blue
fig_nonstationary.update_layout(
    title='具有上升趋势的时间序列',
    xaxis_title='日期',
    yaxis_title='值',
    template='plotly_white',
    height=350,
    margin=dict(l=20, r=20, t=40, b=20)
)
# fig_nonstationary.show() # 在Jupyter Notebook中使用

# --- 绘制平稳数据 ---
fig_stationary = go.Figure()
fig_stationary.add_trace(go.Scatter(x=ts_stationary.index, y=ts_stationary.values,
                                   mode='lines', name='时间序列', line=dict(color='#20c997'))) # teal
fig_stationary.update_layout(
    title='可能平稳的时间序列',
    xaxis_title='日期',
    yaxis_title='值',
    template='plotly_white',
    height=350,
    margin=dict(l=20, r=20, t=40, b=20)
)
# fig_stationary.show() # 在Jupyter Notebook中使用
```

运行以上代码（最好在能显示Plotly图表的环境中）会生成两张图。



![具有上升趋势的时间序列](plots/3392-0.json)



> 此图中的明显上升斜率是由于趋势引起的不平稳性的一个强烈信号。



![可能平稳的时间序列](plots/3392-1.json)



> 该序列似乎在一个固定水平（约15）附近波动，方差相对一致。它看起来更可能平稳，尽管还需要更多检查。

### 绘制滚动统计量

一种更具量化 (quantization)特性的可视化方法是绘制滚动均值和滚动标准差（或方差）。如果序列是平稳的，那么滚动均值和滚动标准差应随时间大致保持不变。这些滚动统计量出现显著的上升或下降趋势，或大幅波动，都强烈表明不平稳性。

Pandas提供了`rolling()`方法以方便地计算这些统计量。

```python
# 定义滚动统计量的窗口大小（例如，7天）
window_size = 7

# 计算非平稳序列的滚动均值和标准差
rolling_mean_ns = ts_nonstationary.rolling(window=window_size).mean()
rolling_std_ns = ts_nonstationary.rolling(window=window_size).std()

# --- 绘制带有滚动统计量的非平稳数据 ---
fig_roll_ns = go.Figure()
fig_roll_ns.add_trace(go.Scatter(x=ts_nonstationary.index, y=ts_nonstationary.values,
                               mode='lines', name='原始序列', line=dict(color='#adb5bd', width=1))) # gray
fig_roll_ns.add_trace(go.Scatter(x=rolling_mean_ns.index, y=rolling_mean_ns.values,
                               mode='lines', name='滚动均值', line=dict(color='#f03e3e', width=2))) # red
fig_roll_ns.add_trace(go.Scatter(x=rolling_std_ns.index, y=rolling_std_ns.values,
                               mode='lines', name='滚动标准差', line=dict(color='#fd7e14', width=2))) # orange

fig_roll_ns.update_layout(
    title='带有滚动统计量的非平稳序列（窗口=7）',
    xaxis_title='日期',
    yaxis_title='值',
    template='plotly_white',
    height=400,
    legend_title_text='度量',
    margin=dict(l=20, r=20, t=50, b=20)
)
# fig_roll_ns.show()
```

我们将其可视化给非平稳数据。



![带有滚动统计量的非平稳序列（窗口=7）](plots/3392-2.json)



> 滚动均值明显呈现上升趋势，这证实了第一张图的视觉判断。滚动标准差在这里看起来相对恒定，但均值漂移足以将该序列归类为非平稳。对于平稳示例的类似图表，将显示滚动均值和滚动标准差都围绕着水平线波动。

### 可视化检查的局限性

可视化检查是一个有价值的起点，提供对数据行为的快速认识。然而，它也有局限性：

- **主观性：** 对一个人而言看似趋势或恒定方差的，对另一个人可能并非如此。
- **不明显：** 微小的趋势或方差的轻微变化在视觉上很难发现，尤其是在有噪声的数据中。
- **尺度：** 视觉模式可能会因绘图轴的尺度而产生误导。

因此，尽管可视化检查有助于识别明显的非平稳性，但应始终辅以更严格的**统计检验**，例如增广迪基-福勒（ADF）检验，这些检验提供平稳性的量化 (quantization)衡量。我们将在下一节中介绍这些检验。

## 参考资料

- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts
  全面实用的时间序列预测入门教材，详细解释了平稳性以及绘制原始序列和滚动统计量等可视化诊断方法。
- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118674925) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, and Greta M. Ljung (2016)
  Publisher: John Wiley & Sons
  时间序列分析的经典教材，提供了平稳性的理论定义及其对ARIMA模型的重要性，这些模型在章节中有所提及。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104030/) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: 579
  使用Python的Pandas库进行数据操作和分析的基础指南，包括时间序列数据处理和计算滚动统计量，如章节代码示例所示。
- [Windowing Operations](https://pandas.pydata.org/pandas-docs/stable/user_guide/window.html) — The Pandas Development Team (2023)
  Pandas滚动窗口操作的官方文档，用于计算滚动均值和标准差，以进行时间序列平稳性的可视化检查。

# 使用 SARIMA 进行预测

来源：[原文](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-5-seasonal-arima-sarima/forecasting-with-sarima)

[返回章节目录](README.md) · [返回课程目录](../README.md)

为未来的时间段生成预测是 SARIMA 模型的一个主要应用。当这些模型被开发以充分捕捉数据本身的模式（包括季节性），并且残差类似于白噪声时，它们会非常有效。SARIMA 模型通过包含季节性和非季节性分量，可以进行体现预期周期性行为的预测。

使用已拟合的 `statsmodels` SARIMA 模型进行预测操作简单。通常使用 `.get_forecast()` 方法，它更常用于生成样本外预测（即预测未来值）。另外，`.predict()` 方法可以用于样本内预测（预测原始训练数据中时间点的值）和样本外预测。

假设您有一个名为 `sarima_results` 的已拟合 SARIMA 结果对象。要预测未来 12 个周期，您会使用 `.get_forecast()`：

```python
# 假设 sarima_results 是 statsmodels 中拟合好的 SARIMA 模型
# 预测未来 12 步
forecast_object = sarima_results.get_forecast(steps=12)

# 获取预测平均值
mean_forecast = forecast_object.predicted_mean

# 获取置信区间
confidence_intervals = forecast_object.conf_int(alpha=0.05) # 95% 置信区间

# 显示预测值和置信区间
print("预测值:")
print(mean_forecast)
print("\n置信区间 (95%):")
print(confidence_intervals)
```

`steps` 参数 (parameter)指定您希望预测未来多少个周期。`.get_forecast()` 方法返回一个包含几个有用属性的对象：

- `predicted_mean`：每个未来步长的点预测值。
- `conf_int(alpha=...)`：预测置信区间的下限和上限。`alpha` 参数确定置信水平（例如，`alpha=0.05` 表示 95% 置信度）。
- `se_mean`：预测值的标准误差。

置信区间非常重要。它们提供一个范围，真实未来值在给定概率水平下很可能落在这个范围内。请注意，随着您预测的未来时间越远，置信区间通常会变宽。这体现了更长期预测相关的不确定性增加；模型对远离观测数据的预测不那么确定。

将预测与历史数据和置信区间一起可视化对于了解模型的预测表现很重要。

````python
import pandas as pd
import plotly.graph_objects as go

# 假设 'train_data' 是您的原始时间序列（Pandas Series/DataFrame）
# 假设 'mean_forecast' 是预测值的 Pandas Series
# 假设 'confidence_intervals' 是包含 'lower' 和 'upper' 列的 Pandas DataFrame

# 创建图表
fig = go.Figure()

# 添加历史数据
fig.add_trace(go.Scatter(
    x=train_data.index,
    y=train_data,
    mode='lines',
    name='观测数据',
    line=dict(color='#495057') # 灰色
))

# 添加预测线
fig.add_trace(go.Scatter(
    x=mean_forecast.index,
    y=mean_forecast,
    mode='lines',
    name='SARIMA 预测',
    line=dict(color='#f03e3e') # 红色
))

# 添加置信区间带
fig.add_trace(go.Scatter(
    x=confidence_intervals.index,
    y=confidence_intervals.iloc[:, 0], # 下限
    mode='lines',
    line=dict(width=0),
    hoverinfo='skip',
    showlegend=False
))
fig.add_trace(go.Scatter(
    x=confidence_intervals.index,
    y=confidence_intervals.iloc[:, 1], # 上限
    mode='lines',
    line=dict(width=0),
    fillcolor='rgba(255, 107, 107, 0.2)', # 浅红色填充
    fill='tonexty', # 填充此轨迹与前一个轨迹（下限）之间的区域
    hoverinfo='skip',
    showlegend=False,
    name='置信区间'
))

# 更新布局
fig.update_layout(
    title='SARIMA 预测及置信区间',
    xaxis_title='日期',
    yaxis_title='值',
    hovermode='x unified',
    legend=dict(x=0.01, y=0.99)
)

# 用于嵌入的 plotly JSON 对象（请替换为真实图表的实际数据）
# 静态 Plotly JSON 表示示例：
# ```plotly
# {"layout": {"title": "SARIMA 预测及置信区间", "xaxis": {"title": "日期"}, "yaxis": {"title": "值"}, "hovermode": "x unified", "legend": {"x": 0.01, "y": 0.99}}, "data": [{"type": "scatter", "x": ["2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01", "2022-05-01", "2022-06-01"], "y": [100, 110, 105, 115, 120, 118], "mode": "lines", "name": "观测数据", "line": {"color": "#495057"}}, {"type": "scatter", "x": ["2022-07-01", "2022-08-01", "2022-09-01"], "y": [125, 122, 128], "mode": "lines", "name": "SARIMA 预测", "line": {"color": "#f03e3e"}}, {"type": "scatter", "x": ["2022-07-01", "2022-08-01", "2022-09-01"], "y": [120, 116, 122], "mode": "lines", "line": {"width": 0}, "hoverinfo": "skip", "showlegend": false}, {"type": "scatter", "x": ["2022-07-01", "2022-08-01", "2022-09-01"], "y": [130, 128, 134], "mode": "lines", "line": {"width": 0}, "fillcolor": "rgba(255, 107, 107, 0.2)", "fill": "tonexty", "hoverinfo": "skip", "showlegend": false, "name": "置信区间"}]}
# ```

# fig.show() # 在 Jupyter Notebook 环境或脚本中显示图表
````



[交互图表：SARIMA 预测及置信区间](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-5-seasonal-arima-sarima/forecasting-with-sarima#plot-1j8elwf)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "SARIMA 预测及置信区间",
    "xaxis": {
      "title": "日期",
      "range": [
        "2022-01-01",
        "2022-09-01"
      ]
    },
    "yaxis": {
      "title": "值",
      "range": [
        90,
        140
      ]
    },
    "hovermode": "x unified",
    "legend": {
      "x": 0.01,
      "y": 0.99
    }
  },
  "data": [
    {
      "type": "scatter",
      "x": [
        "2022-01-01",
        "2022-02-01",
        "2022-03-01",
        "2022-04-01",
        "2022-05-01",
        "2022-06-01"
      ],
      "y": [
        100,
        110,
        105,
        115,
        120,
        118
      ],
      "mode": "lines",
      "name": "观测数据",
      "line": {
        "color": "#495057"
      }
    },
    {
      "type": "scatter",
      "x": [
        "2022-07-01",
        "2022-08-01",
        "2022-09-01"
      ],
      "y": [
        125,
        122,
        128
      ],
      "mode": "lines",
      "name": "SARIMA 预测",
      "line": {
        "color": "#f03e3e"
      }
    },
    {
      "type": "scatter",
      "x": [
        "2022-07-01",
        "2022-08-01",
        "2022-09-01"
      ],
      "y": [
        120,
        116,
        122
      ],
      "mode": "lines",
      "line": {
        "width": 0
      },
      "hoverinfo": "skip",
      "showlegend": false
    },
    {
      "type": "scatter",
      "x": [
        "2022-07-01",
        "2022-08-01",
        "2022-09-01"
      ],
      "y": [
        130,
        128,
        134
      ],
      "mode": "lines",
      "line": {
        "width": 0
      },
      "fillcolor": "rgba(255, 107, 107, 0.2)",
      "fill": "tonexty",
      "hoverinfo": "skip",
      "showlegend": false,
      "name": "置信区间"
    }
  ]
}
```

</details>



> 历史数据与 SARIMA 点预测和 95% 置信区间一起显示。区间变宽表示更长的预测期不确定性增加。

在结果时，请检查预测是否在视觉上延续了历史数据中观察到的模式，特别是季节性周期。SARIMA 模型的优势在于其能够将这些重复模式向前推断。但是，请记住这些预测完全基于模型学习到的历史模式。它们假定生成数据的基础过程保持稳定，并且过去的季节性影响将以相同方式延续。意外事件、趋势变化或季节性行为转变没有自然包含在内，可能导致预测误差。因此，使用下一章讨论的指标评估预测准确性是很关键的一步。

## 参考资料

- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, Greta M. Ljung (2015)
  Publisher: Wiley; Pages: 720; DOI: [10.1002/9781118673728](https://doi.org/10.1002/9781118673728)
  ARIMA和SARIMA模型理论的经典教材，包括预测原理和预测区间的构建。
- [\`statsmodels.tsa.statespace.sarimax.SARIMAXResults.get_forecast\`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.get_forecast.html) — The `statsmodels` Developers (2024)
  `statsmodels`中用于从拟合的`SARIMAX`模型生成样本外预测的方法的官方文档，详细说明了其参数和返回属性。
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts
  一本全面的在线教科书，涵盖SARIMA模型、预测方法和预测区间的解释。
- [Plotly Python Graphing Library: Time Series](https://plotly.com/python/time-series/) — Plotly Technologies Inc. (2023)
  Publisher: Plotly Technologies Inc.
  Plotly关于在Python中创建时间序列可视化的官方指南，包括绘制预测线和置信区间的技术。

---

[上一节](06-SARIMA%E6%A8%A1%E5%9E%8B%E8%AF%8A%E6%96%AD.md) · [下一节](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BASARIMA%E6%A8%A1%E5%9E%8B.md)

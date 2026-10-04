# 使用ARIMA模型进行预测

来源：[原文](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-4-arima-models-forecasting/forecasting-with-arima)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在ARIMA模型成功拟合并进行诊断检查以确保其有效性之后，该模型可用于预测未来数值。此过程称为预测。

### 生成预测

`statsmodels`库为已拟合的模型结果对象（在示例中常命名为`results`或`arima_results`）提供了便利的方法，用来生成预测。两种主要方法是`predict()`和`forecast()`。尽管`predict()`可用于样本内预测和样本外预测，但`forecast()`方法专为在训练数据结束后生成预测而设计，并且通常更直接。

假设您有一个已拟合的ARIMA结果对象，名为`arima_results`，它是通过将ARIMA模型拟合到您的时间序列数据而获得的。

#### 使用`forecast()`

`forecast()`方法是生成样本外预测的最简单方法。您只需指定您希望预测的未来步数（时间段）即可。

```python
# 假设arima_results是您已拟合的ARIMA模型对象
# 预测未来12个时间步
forecast_steps = 12
forecast_values = arima_results.forecast(steps=forecast_steps)

print(forecast_values)
```

这将返回一个Pandas Series，其中包含指定未来时间步的点预测。返回的Series的索引通常会遵循您原始数据的时间索引。

#### 使用`predict()`

`predict()`方法提供更多灵活性。它允许您为预测指定`start`和`end`点。这些点可以是索引或时间戳。

- **样本内预测**: 如果`start`和`end`落在原始数据索引范围内，`predict()`生成拟合值。
- **样本外预测**: 如果`start`和/或`end`落在原始数据索引范围之外，`predict()`生成预测。

```python
# 假设arima_results是您已拟合的模型
# 假设原始数据结束于索引'n'或时间戳't_end'

# 获取最后一个观测值的索引
last_index = ts_data.index[-1] # 或者，如果适用，使用整数索引

# 定义预测未来12个步长的起始和结束
# 注意：索引必须与您数据（例如，日期时间）的索引类型兼容
forecast_start_index = last_index + pd.Timedelta(days=1) # 每日数据的示例
forecast_end_index = last_index + pd.Timedelta(days=12) # 每日数据的示例

# 或者，如果适用，使用整数索引
# forecast_start_index = len(ts_data)
# forecast_end_index = len(ts_data) + 11

forecast_values_pred = arima_results.predict(start=forecast_start_index, end=forecast_end_index)

print(forecast_values_pred)
```

尽管`predict()`可用，但在纯粹生成未来值时，`forecast()`通常因其简洁性而更受青睐。

### 获取置信区间

点预测为未来值提供单一最佳估计，但它们不表明与预测相关的不确定性。ARIMA模型作为统计模型，允许我们计算这些预测周围的置信区间。置信区间提供了一个范围，真实未来值预计将以特定概率（例如95%）落在该范围内。

要同时获取点预测和置信区间，请使用`get_forecast()`方法。这将返回一个`PredictionResults`对象，其中包含更详细的信息。

```python
# 假设arima_results是您已拟合的模型
forecast_steps = 12

# 获取预测对象
forecast_obj = arima_results.get_forecast(steps=forecast_steps)

# 提取预测均值（点预测）
predicted_mean = forecast_obj.predicted_mean

# 提取置信区间（默认alpha=0.05表示95%置信区间）
confidence_intervals = forecast_obj.conf_int(alpha=0.05)

# confidence_intervals是一个DataFrame，包含类似'lower y'和'upper y'的列
print("点预测:\n", predicted_mean)
print("\n置信区间 (95%):\n", confidence_intervals)
```

`alpha`参数 (parameter)决定置信水平。`alpha=0.05`对应95%置信区间（1 - alpha），表示在模型正确设定的前提下，我们预计真实值有95%的概率落在计算出的上下限之间。

### 预测可视化

强烈推荐将预测与历史数据及置信区间一同显示。它能直观地理解模型预测及相关的不确定性。

下面是如何使用Plotly绘制此图：

```python
import pandas as pd
import plotly.graph_objects as go

# 假设:
# ts_data: 原始历史时间序列 (Pandas Series)
# predicted_mean: 预测值 (Pandas Series)
# confidence_intervals: 包含'lower y'和'upper y'列的DataFrame

# 示例数据（请替换为您的实际数据）
# 创建虚拟历史数据
dates_hist = pd.to_datetime(pd.date_range(start='2023-01-01', periods=50, freq='D'))
ts_data = pd.Series(range(50), index=dates_hist) + 10 * (pd.Series(range(50))/50)**2 + 5 * pd.np.random.randn(50)

# 创建虚拟预测数据
dates_fcst = pd.to_datetime(pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=1), periods=12, freq='D'))
predicted_mean = pd.Series([ts_data.iloc[-1] + i * 0.5 + 2 * pd.np.random.randn(1)[0] for i in range(1, 13)], index=dates_fcst)
ci_lower = predicted_mean - (pd.Series(range(1, 13)) * 0.8)
ci_upper = predicted_mean + (pd.Series(range(1, 13)) * 0.8)
confidence_intervals = pd.DataFrame({'lower y': ci_lower, 'upper y': ci_upper})

# 创建图表
fig = go.Figure()

# 添加历史数据
fig.add_trace(go.Scatter(
    x=ts_data.index,
    y=ts_data,
    mode='lines',
    name='历史数据',
    line=dict(color='#1c7ed6') # 蓝色
))

# 添加预测线
fig.add_trace(go.Scatter(
    x=predicted_mean.index,
    y=predicted_mean,
    mode='lines',
    name='预测',
    line=dict(color='#f76707') # 橙色
))

# 添加置信区间区域
fig.add_trace(go.Scatter(
    x=confidence_intervals.index.tolist() + confidence_intervals.index.tolist()[::-1], # x values for shape
    y=confidence_intervals['upper y'].tolist() + confidence_intervals['lower y'].tolist()[::-1], # y values for shape
    fill='toself',
    fillcolor='rgba(253, 126, 20, 0.2)', # 橙色透明
    line=dict(color='rgba(255,255,255,0)'), # 无边框线
    hoverinfo="skip", # 不显示图形的悬停标签
    name='95% 置信区间'
))

# 更新布局以更好显示
fig.update_layout(
    title='ARIMA模型预测及置信区间',
    xaxis_title='时间',
    yaxis_title='值',
    hovermode='x unified',
    legend=dict(x=0.01, y=0.99)
)

# 显示图表（在Jupyter环境中，否则使用fig.show()）
# fig.show()

# 如果需要，转换为JSON以便嵌入
# print(fig.to_json())
```



[交互图表：ARIMA模型预测及置信区间](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-4-arima-models-forecasting/forecasting-with-arima#plot-5767or)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "ARIMA模型预测及置信区间"
    },
    "xaxis": {
      "title": {
        "text": "时间"
      }
    },
    "yaxis": {
      "title": {
        "text": "值"
      }
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
        "2023-01-01T00:00:00",
        "2023-01-02T00:00:00",
        "2023-01-03T00:00:00",
        "2023-01-04T00:00:00",
        "2023-01-05T00:00:00",
        "2023-01-06T00:00:00",
        "2023-01-07T00:00:00",
        "2023-01-08T00:00:00",
        "2023-01-09T00:00:00",
        "2023-01-10T00:00:00",
        "2023-01-11T00:00:00",
        "2023-01-12T00:00:00",
        "2023-01-13T00:00:00",
        "2023-01-14T00:00:00",
        "2023-01-15T00:00:00",
        "2023-01-16T00:00:00",
        "2023-01-17T00:00:00",
        "2023-01-18T00:00:00",
        "2023-01-19T00:00:00",
        "2023-01-20T00:00:00",
        "2023-01-21T00:00:00",
        "2023-01-22T00:00:00",
        "2023-01-23T00:00:00",
        "2023-01-24T00:00:00",
        "2023-01-25T00:00:00",
        "2023-01-26T00:00:00",
        "2023-01-27T00:00:00",
        "2023-01-28T00:00:00",
        "2023-01-29T00:00:00",
        "2023-01-30T00:00:00",
        "2023-01-31T00:00:00",
        "2023-02-01T00:00:00",
        "2023-02-02T00:00:00",
        "2023-02-03T00:00:00",
        "2023-02-04T00:00:00",
        "2023-02-05T00:00:00",
        "2023-02-06T00:00:00",
        "2023-02-07T00:00:00",
        "2023-02-08T00:00:00",
        "2023-02-09T00:00:00",
        "2023-02-10T00:00:00",
        "2023-02-11T00:00:00",
        "2023-02-12T00:00:00",
        "2023-02-13T00:00:00",
        "2023-02-14T00:00:00",
        "2023-02-15T00:00:00",
        "2023-02-16T00:00:00",
        "2023-02-17T00:00:00",
        "2023-02-18T00:00:00",
        "2023-02-19T00:00:00"
      ],
      "y": [
        5.28,
        3.48,
        2.07,
        -1.97,
        3.82,
        0.16,
        -2.04,
        11.98,
        5.98,
        11.2,
        9.41,
        12.38,
        11.67,
        14.19,
        15.88,
        14.02,
        14.63,
        18.19,
        16.53,
        22.23,
        22.75,
        22.14,
        21.33,
        26.72,
        29.24,
        28.92,
        34.4,
        33.48,
        28.11,
        31.6,
        31.76,
        34.72,
        37.42,
        33.77,
        35.02,
        37.65,
        38.01,
        41.41,
        41.87,
        44.03,
        45.68,
        45.68,
        49.63,
        51.66,
        49.06,
        54.12,
        58.01,
        59.1,
        61.39,
        55.28
      ],
      "mode": "lines",
      "name": "历史数据",
      "line": {
        "color": "#1c7ed6"
      }
    },
    {
      "type": "scatter",
      "x": [
        "2023-02-20T00:00:00",
        "2023-02-21T00:00:00",
        "2023-02-22T00:00:00",
        "2023-02-23T00:00:00",
        "2023-02-24T00:00:00",
        "2023-02-25T00:00:00",
        "2023-02-26T00:00:00",
        "2023-02-27T00:00:00",
        "2023-02-28T00:00:00",
        "2023-03-01T00:00:00",
        "2023-03-02T00:00:00",
        "2023-03-03T00:00:00"
      ],
      "y": [
        53.33,
        56.94,
        57.18,
        57.13,
        55.53,
        55.37,
        59.73,
        58.5,
        61.81,
        62.4,
        63.68,
        61.64
      ],
      "mode": "lines",
      "name": "预测",
      "line": {
        "color": "#f76707"
      }
    },
    {
      "type": "scatter",
      "x": [
        "2023-02-20T00:00:00",
        "2023-02-21T00:00:00",
        "2023-02-22T00:00:00",
        "2023-02-23T00:00:00",
        "2023-02-24T00:00:00",
        "2023-02-25T00:00:00",
        "2023-02-26T00:00:00",
        "2023-02-27T00:00:00",
        "2023-02-28T00:00:00",
        "2023-03-01T00:00:00",
        "2023-03-02T00:00:00",
        "2023-03-03T00:00:00",
        "2023-03-03T00:00:00",
        "2023-03-02T00:00:00",
        "2023-03-01T00:00:00",
        "2023-02-28T00:00:00",
        "2023-02-27T00:00:00",
        "2023-02-26T00:00:00",
        "2023-02-25T00:00:00",
        "2023-02-24T00:00:00",
        "2023-02-23T00:00:00",
        "2023-02-22T00:00:00",
        "2023-02-21T00:00:00",
        "2023-02-20T00:00:00"
      ],
      "y": [
        54.13,
        58.54,
        59.58,
        60.33,
        59.53,
        60.17,
        65.33,
        64.9,
        69.01,
        70.4,
        72.48,
        71.24,
        52.04,
        54.88,
        54.4,
        54.61,
        52.1,
        54.13,
        50.57,
        51.53,
        53.93,
        54.78,
        55.34,
        52.53
      ],
      "fill": "toself",
      "fillcolor": "rgba(253, 126, 20, 0.2)",
      "line": {
        "color": "rgba(255,255,255,0)"
      },
      "hoverinfo": "skip",
      "name": "95% 置信区间"
    }
  ]
}
```

</details>



> 历史数据以蓝色显示，点预测以橙色显示，阴影区域表示95%置信区间。

注意，置信区间通常随预测时间范围的增加而变宽。这表明未来不确定性增加；较远期的预测固有地比近期预测更不确定。

### 重要考量

- **模型稳定性**: ARIMA预测很大程度上依赖于以下假定：时间序列的统计特性（均值、方差、自相关结构），由模型参数 (parameter)和差分阶数表示，未来保持不变。基础过程的显著变化将使预测无效。
- **不确定性**: 务必考虑置信区间。单独的点预测可能产生误导。区间的宽度与中心预测值同样重要。
- **外生变量**: 标准ARIMA模型不包含可能影响时间序列的外部因素（外生变量）。如果这些因素很重要，ARIMAX等模型或其他基于回归的方法可能更合适。

凭借生成预测并量化 (quantization)其不确定性的能力，您现在拥有一个预测未来趋势的有效工具，此工具基于ARIMA模型识别的历史模式。下一章将扩展这些思路，使用SARIMA模型处理季节性。

## 参考资料

- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, Greta M. Ljung (2015)
  Publisher: Wiley
  一本经典教材，提供ARIMA模型构建、估计和预测的理论基础和方法。
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman, George Athanasopoulos (2021)
  Publisher: OTexts
  一本全面实用的在线教材，涵盖多种预测方法，包括关于ARIMA模型、其应用和预测不确定性的专门章节。
- [statsmodels.tsa.arima.model.ARIMAResults](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMAResults.html) — Statsmodels Developers (2024)
  `statsmodels` 中ARIMA结果对象的官方文档，详细介绍了用于生成预测和置信区间的 `forecast()`、`predict()` 和 `get_forecast()` 等方法。

---

[上一节](07-%E6%A8%A1%E5%9E%8B%E8%AF%8A%E6%96%AD%E4%B8%8E%E6%AE%8B%E5%B7%AE%E5%88%86%E6%9E%90.md) · [下一节](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BAARIMA%E6%A8%A1%E5%9E%8B.md)

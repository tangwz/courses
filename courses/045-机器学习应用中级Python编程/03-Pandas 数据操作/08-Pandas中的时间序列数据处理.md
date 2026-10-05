# Pandas中的时间序列数据处理

来源：[原文](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-3-data-manipulation-pandas/pandas-time-series)

[返回章节目录](README.md) · [返回课程目录](../README.md)

时间序列数据是随时间记录的观测序列，它们在许多机器学习 (machine learning)方面都应用广泛，包括预测、异常检测和信号处理。股票价格、传感器读数、天气模式和用户活动日志都是时间序列的示例。Pandas提供了出色的工具，专门用于高效处理日期和时间数据。下面将说明如何在Pandas中处理基于时间的索引、重采样以及其他时间序列特有的操作。

## 了解Pandas中的日期和时间

Pandas时间序列功能的核心是专门的数据类型和索引结构。

- **Timestamp（时间戳）：** 表示时间中的一个点。它是Pandas中与Python的`datetime.datetime`对象等效的类型，但效率更高，并且能很好地与NumPy和Pandas操作结合使用。
- **DatetimeIndex（日期时间索引）：** 由`Timestamp`对象组成的索引。当您将日期时间对象列设置为DataFrame的索引时，它就会变成`DatetimeIndex`。这支持强大的基于时间的索引和切片功能。
- **Period（时间段）：** 表示一个时间跨度，例如一个特定的月份、年份或季度。
- **Timedelta（时间差）：** 表示两个时间点之间的持续时间或差值。

虽然`Period`和`Timedelta`很有用，但对于大多数机器学习 (machine learning)数据准备任务来说，我们的主要侧重将是`Timestamp`和`DatetimeIndex`。

## 创建时间序列数据

您经常会在数据文件中遇到以字符串形式存在的时间信息。Pandas提供了灵活的方法将这些信息转换为正确的日期时间对象，并设置`DatetimeIndex`。

### 将字符串转换为日期时间对象

`pd.to_datetime()`函数是主要工具，用于将包含日期字符串的标量值、列表或Series转换为Pandas `Timestamp`对象。

```python
import pandas as pd

# 转换单个字符串
print(pd.to_datetime('2023-10-26'))
print(pd.to_datetime('27/10/2023', dayfirst=True)) # 如有需要，指定格式

# 转换列表或Series
date_strings = ['2023-11-01', '2023-11-05', '2023-11-10']
datetime_series = pd.to_datetime(date_strings)
print(datetime_series)
```

### 生成日期范围

为了创建日期序列，`pd.date_range()`非常有用。您可以指定开始日期、结束日期和频率（例如，每日、每月、每小时）。

```python
# 每日频率（默认）
daily_index = pd.date_range(start='2023-01-01', end='2023-01-05')
print(daily_index)

# 每月频率
monthly_index = pd.date_range(start='2023-01-01', periods=4, freq='M') # 'M' 是月末频率
print(monthly_index)

# 工作日频率
business_day_index = pd.date_range(start='2023-10-23', periods=5, freq='B')
print(business_day_index)
```

常用的频率字符串包括 'D'（每日）、'B'（工作日）、'W'（每周）、'M'（月末）、'MS'（月初）、'Q'（季度末）、'QS'（季度初）、'A'（年末）、'AS'（年初）、'H'（每小时）、'T' 或 'min'（每分钟）、'S'（每秒）。

### 数据加载时解析日期

通常，您的数据集会包含一个或多个含有日期信息的列。您可以在加载数据时，使用`pd.read_csv()`等函数中的`parse_dates`参数 (parameter)，直接指示Pandas解析这些日期。

```python
# 假设 'data.csv' 有 'date_str' 和 'value' 列
# df = pd.read_csv('data.csv', parse_dates=['date_str'])

# 创建一个示例DataFrame进行演示
data = {'date_str': ['2023-01-15', '2023-01-16', '2023-01-17'],
        'value': [10, 12, 11]}
df = pd.DataFrame(data)
df['date_str'] = pd.to_datetime(df['date_str']) # 如果加载时未解析，则手动转换
print(df.info()) # 注意 date_str 列现在是 datetime64[ns] 类型
```

## 时间序列的索引和选择

一旦有了`DatetimeIndex`，根据时间选择数据就变得直观。首先，您通常会将日期时间列设置为DataFrame的索引。

```python
import numpy as np

# 创建示例时间序列数据
dates = pd.date_range('20230101', periods=100, freq='D')
ts_data = pd.Series(np.random.randn(100), index=dates)
print(ts_data.head())

# 将日期列设置为索引（如果尚未设置）
# 假设 'df' 有一个名为 'date_col' 的日期时间列
# df.set_index('date_col', inplace=True)
# 对于我们的示例 'ts_data'，索引已设置

# 选择特定日期
print("\n2023-01-05 的数据:")
print(ts_data['2023-01-05'])

# 选择特定年份的数据
print("\n2023 年的数据:")
print(ts_data['2023'].head()) # 有效，因为索引已排序

# 选择特定月份的数据
print("\n2023 年 1 月的数据:")
print(ts_data['2023-01'].head())

# 切片日期范围
print("\n2023-02-10 到 2023-02-15 的数据:")
print(ts_data['2023-02-10':'2023-02-15'])
```

这种强大的索引功能依赖于`DatetimeIndex`的排序状态。如果您创建或修改索引导致其未排序，您可能需要调用`df.sort_index()`以使这些切片方法正常工作。

## 时间序列数据重采样

重采样涉及更改时间序列观测值的频率。这是一种常见操作，用于将数据聚合到较低频率（降采样）或提高频率（升采样），通常需要插值。实现此功能的主要工具是`.resample()`方法。

### 降采样

降采样时，您将数据从较高频率聚合到较低频率（例如，将每日数据聚合到每月数据）。您需要指定如何聚合落入每个新的、更大时间段内的数据点（例如，均值、总和、最大值、最小值）。

```python
# 每日数据示例
dates = pd.date_range('2023-01-01', periods=35, freq='D')
daily_values = pd.Series(np.random.rand(35) * 100, index=dates)

# 重采样到每月频率，取均值
monthly_mean = daily_values.resample('M').mean()
print("\n月度均值:")
print(monthly_mean)

# 重采样到每周频率，取总和
weekly_sum = daily_values.resample('W').sum()
print("\n每周总和:")
print(weekly_sum)
```

### 升采样

升采样会增加频率（例如，将每月数据转换为每日数据）。由于您没有新创建时间点的数据，因此需要决定如何填充它们。常见的方法包括前向填充（`ffill`）、后向填充（`bfill`）或插值。

```python
# 每月数据示例
dates_monthly = pd.date_range('2023-01-01', periods=4, freq='MS') # 月初
monthly_values = pd.Series([10, 15, 12, 18], index=dates_monthly)

# 重采样到每日频率，使用前向填充
daily_ffill = monthly_values.resample('D').ffill()
print("\n每日数据（前向填充）:")
print(daily_ffill.head(10)) # 显示前 10 天

# 重采样到每日频率，使用插值
daily_interpolated = monthly_values.resample('D').interpolate(method='linear')
print("\n每日数据（线性插值）:")
print(daily_interpolated.head(10)) # 显示前 10 天
```

## 滚动窗口操作

滚动窗口计算对于分析趋势和平滑时间序列数据非常重要。它们在一个固定大小的滑动窗口上操作，随着窗口在序列中移动，将一个函数（如均值、总和、标准差）应用于窗口内的数据。

`.rolling()`方法用于此目的。您需要指定窗口大小（周期数）。

```python
# 示例数据
dates = pd.date_range('2023-01-01', periods=20, freq='D')
data = pd.Series(np.random.randn(20).cumsum() + 50, index=dates) # 类似随机游走的数据

# 计算 5 天滚动均值
rolling_mean_5d = data.rolling(window=5).mean()

print("\n原始数据:")
print(data.head(7))
print("\n5 天滚动均值:")
print(rolling_mean_5d.head(7)) # 注意初始的 NaN 值
```

滚动计算的前 `window - 1` 个值将是 `NaN`，因为没有足够的先前数据点来填充窗口。



[交互图表：原始数据 vs. 5 天滚动均值](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-3-data-manipulation-pandas/pandas-time-series#plot-11hedfw)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        "2023-01-01",
        "2023-01-02",
        "2023-01-03",
        "2023-01-04",
        "2023-01-05",
        "2023-01-06",
        "2023-01-07",
        "2023-01-08",
        "2023-01-09",
        "2023-01-10"
      ],
      "y": [
        10,
        12,
        11,
        13,
        15,
        14,
        16,
        17,
        15,
        18
      ],
      "mode": "lines+markers",
      "name": "原始数据",
      "line": {
        "color": "#339af0"
      },
      "marker": {
        "size": 4
      }
    },
    {
      "x": [
        "2023-01-05",
        "2023-01-06",
        "2023-01-07",
        "2023-01-08",
        "2023-01-09",
        "2023-01-10"
      ],
      "y": [
        12.2,
        13.0,
        13.8,
        15.0,
        14.6,
        16.0
      ],
      "mode": "lines",
      "name": "5 天滚动均值",
      "line": {
        "color": "#f76707",
        "width": 2
      }
    }
  ],
  "layout": {
    "title": "原始数据 vs. 5 天滚动均值",
    "xaxis": {
      "title": "日期"
    },
    "yaxis": {
      "title": "值"
    },
    "hovermode": "x unified",
    "template": "plotly_white",
    "legend": {
      "yanchor": "top",
      "y": 0.99,
      "xanchor": "left",
      "x": 0.01
    }
  }
}
```

</details>



> 原始时间序列数据与其 5 天滚动均值进行比较，说明了平滑效果。

其他常见的滚动函数包括 `.rolling(...).sum()`、`.rolling(...).std()`、`.rolling(...).max()` 等。滚动窗口常用于时间序列模型的特征工程中，用于捕获近期趋势或波动性。

## 数据移动和滞后

有时您需要将时间序列值与前期的值进行比较。这可以通过使用`.shift()`方法完成，该方法将数据向前或向后移动指定数量的周期。这对于在预测模型中创建滞后特征或计算同期变化非常有用。

```python
# 创建示例数据
dates = pd.date_range('2023-01-01', periods=5, freq='D')
ts = pd.Series([10, 12, 11, 13, 14], index=dates)

# 将数据向前移动 1 个周期（滞后）
ts_lag1 = ts.shift(1)
print("\n原始序列:")
print(ts)
print("\n滞后序列 (shift=1):")
print(ts_lag1)

# 计算与前一个周期的百分比变化
pct_change = (ts - ts.shift(1)) / ts.shift(1) * 100
# 或者: ts.pct_change() * 100
print("\n百分比变化:")
print(pct_change)

# 将数据向后移动 1 个周期（提前）
ts_lead1 = ts.shift(-1)
print("\n提前序列 (shift=-1):")
print(ts_lead1)
```

请注意，移动操作会在序列的开头（正向移动时）或结尾（负向移动时）引入 `NaN` 值。

## 处理时区

时间序列数据通常包含时区信息，或者需要时区信息才能正确解读。Pandas支持时区本地化和转换。

- `tz_localize()`: 为不含时区信息的日期时间对象分配时区。
- `tz_convert()`: 将日期时间对象从一个时区转换为另一个时区。

```python
# 创建一个“幼稚”的日期时间索引（无时区信息）
naive_dates = pd.date_range('2023-10-26 09:00:00', periods=3, freq='H')
ts_naive = pd.Series([1, 2, 3], index=naive_dates)
print("\n“幼稚”时间序列:")
print(ts_naive)

# 本地化到特定时区（例如，US/Eastern）
ts_eastern = ts_naive.tz_localize('US/Eastern')
print("\n本地化到 US/Eastern:")
print(ts_eastern)

# 转换为另一个时区（例如，Europe/Berlin）
ts_berlin = ts_eastern.tz_convert('Europe/Berlin')
print("\n转换为 Europe/Berlin:")
print(ts_berlin)
```

当处理跨越不同地理区域的数据或涉及夏令时变化时，正确处理时区非常重要。

掌握这些Pandas时间序列工具对于为机器学习 (machine learning)准备时间数据非常重要。您现在可以基于时间索引数据，通过重采样改变其频率，使用滚动窗口分析趋势，并使用移动操作创建滞后特征，这些都是构建有效依赖时间模型的基要步骤。

## 参考资料

- [Time series / date functionality](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html) — Pandas Development Team (2023)
  Pandas官方用户指南，详细介绍了其丰富的时间序列功能，涵盖基本数据类型、索引、重采样和时区处理。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  Pandas创建者的基础著作，深入介绍了数据操作，其中包含专门讨论时间序列工具的章节。

---

[上一节](07-%E5%90%88%E5%B9%B6%E3%80%81%E8%BF%9E%E6%8E%A5%E5%92%8C%E6%8B%BC%E6%8E%A5%E6%95%B0%E6%8D%AE%E5%B8%A7.md) · [下一节](09-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%20Pandas%20%E6%95%B4%E7%90%86%E6%95%B0%E6%8D%AE.md)

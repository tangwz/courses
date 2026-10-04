---
course: "time-series-analysis-forecasting"
chapter: "intro-time-series-data"
lesson: "loading-handling-pandas"
sourceId: 3345
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-1-intro-time-series-data/loading-handling-pandas"
title: "在Pandas中加载和处理时间序列数据"
description: "学习如何使用Python中的Pandas库高效地加载、解析日期并处理时间序列数据。"
order: 3
plots: ["plots/3345-0.json"]
sourceHash: "48dcd4e3a71e849bf572f8d1efb4cfa9764dc692ca547548c27873cf660026d5"
sourceCorrections: []
---

高效处理时间序列数据，依赖于使用合适的工具来加载、构建和操作按时间排序的观测数据序列。Python的Pandas库提供了强大且便捷的数据结构，特别是`Series`和`DataFrame`，它们非常适合处理时间索引数据。Pandas提供了专用函数来解析日期和时间、创建基于时间的索引以及执行时间感知操作，这简化了时间序列分析中的许多常见任务。

### 加载时间序列数据

时间序列数据通常存储在CSV等文件中。Pandas使加载这些数据变得简单，同时提供了对时间序列很重要的选项。目的是确保时间信息被正确解析，并最好用作DataFrame的索引。

考虑一个典型场景，您有一个CSV文件（`stock_data.csv`），其中包含日期和相应值（例如，股票价格）的列：

```csv
Date,Open,High,Low,Close,Volume
2023-01-03,130.28,130.90,124.17,125.07,112117500
2023-01-04,126.89,128.66,125.08,126.36,89113600
2023-01-05,127.13,127.77,124.76,125.02,80962700
2023-01-06,126.01,130.29,124.89,129.62,87754700
...
```

您可以使用`pd.read_csv`直接将其加载到Pandas DataFrame中。`parse_dates`和`index_col`参数 (parameter)在此处特别有用：

```python
import pandas as pd

# 指定包含日期的列进行解析
# 指定用作索引的列
file_path = 'stock_data.csv'
try:
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    print("数据加载成功：")
    print(df.head())
    print("\nDataFrame索引信息：")
    print(df.index)
except FileNotFoundError:
    print(f"错误：文件未在 {file_path} 找到")
    # 如果文件缺失，创建模拟DataFrame以供演示
    dates = pd.to_datetime(['2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06'])
    data = {'Open': [130.28, 126.89, 127.13, 126.01],
            'High': [130.90, 128.66, 127.77, 130.29],
            'Low': [124.17, 125.08, 124.76, 124.89],
            'Close': [125.07, 126.36, 125.02, 129.62],
            'Volume': [112117500, 89113600, 80962700, 87754700]}
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    print("\n正在使用模拟数据进行演示：")
    print(df.head())
    print("\nDataFrame索引信息：")
    print(df.index)
```

设置`index_col='Date'`告诉Pandas使用“Date”列作为DataFrame索引。重要的一点是，`parse_dates=True`指示Pandas尝试将索引列的值解析为日期时间对象。如果您的日期列名称不同，请相应地替换`'Date'`。如果多列包含需要解析的日期/时间信息，您可以向`parse_dates`提供列名或索引的列表。

### DateTimeIndex的优点

当Pandas成功解析日期并将其设为索引时，它会创建`DateTimeIndex`。这种专用索引为时间序列分析提供了许多便利：

1. **基于时间的索引和切片：** 您可以使用直观的日期/时间格式选择数据。
2. **日期分量提取：** 轻松获取年、月、日、星期几等分量。
3. **对齐 (alignment)：** 基于日期/时间索引的DataFrame在算术运算期间会自动对齐。
4. **重采样：** 更改时间序列的频率（例如，将每日数据转换为每月数据）。

让我们看一些使用上面加载的`df`的示例：

```python
# 选择特定年份的数据
print("\n2023年的数据：")
# 假设df索引包含2023年。这将选择索引年份为2023年的所有行。
# 如果数据跨越多年，则只返回2023年的数据。
# 如果原始df很小，使用示例df进行说明
dates_multi_year = pd.to_datetime(['2023-01-03', '2023-01-04', '2024-05-10', '2024-05-11'])
df_multi_year = pd.DataFrame({'Close': [125.07, 126.36, 180.00, 181.50]}, index=dates_multi_year)
df_multi_year.index.name = 'Date'
try:
    print(df_multi_year.loc['2023'])
except KeyError:
    print("示例中没有2023年的数据。")


# 选择特定月份的数据（例如，2023年1月）
print("\n2023年1月的数据：")
try:
     # 使用原始df，或者如果未找到文件则使用模拟df
    print(df.loc['2023-01'])
except KeyError:
    print("没有2023年1月的数据。")


# 选择日期范围内的数据
print("\n2023年1月4日至1月5日的数据：")
try:
    print(df.loc['2023-01-04':'2023-01-05'])
except KeyError:
     print("在指定日期范围内没有可用数据。")

# 获取日期分量（例如，前几条的星期几）
# 星期一=0，星期日=6
print("\n星期几（前5条）：")
print(df.index.dayofweek[:5])
```

### 加载后处理日期

有时，日期可能作为普通字符串或对象加载，或者它们可能分散在多列中。Pandas提供了处理这些情况的灵活性。

- **使用 `pd.to_datetime()`：** 如果您的日期列未自动解析或需要特定格式，请使用 `pd.to_datetime()`。

```python
# 假设'DateString'列的日期格式为'01/03/2023'
# df['Date'] = pd.to_datetime(df['DateString'], format='%m/%d/%Y')

# 如果日期部分分散在不同的列中（年、月、日）
# df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
```

- **设置索引：** 在确保您拥有日期时间列之后，可以使用 `set_index()` 将其设为索引。

```python
# 假设'Date'列现在是日期时间类型但不是索引
# df.set_index('Date', inplace=True)
# inplace=True 直接修改DataFrame
```

### 时区处理

时间序列数据可以是时区感知或时区非感知的。非感知对象没有关联的时区信息。如果您的数据表示跨不同地点的事件或需要进行标准化，处理时区很重要。

- **本地化非感知时间戳：** 为非感知时间戳分配时区。

```python
# 假设df索引是非感知的，表示UTC时间
df_utc = df.tz_localize('UTC')
print("\n已本地化为UTC：")
print(df_utc.head())
print(df_utc.index)

# 本地化到特定时区，例如美国/东部
# 这假设原始时间是在美国/东部记录的
# df_eastern = df.tz_localize('America/New_York')
# print("\n已本地化为美国/东部：")
# print(df_eastern.head())
# print(df_eastern.index)
```

- **转换时区：** 更改感知时间戳的时区。

```python
# 将UTC本地化数据转换为美国/东部
df_eastern_converted = df_utc.tz_convert('America/New_York')
print("\n从UTC转换为美国/东部：")
print(df_eastern_converted.head())
print(df_eastern_converted.index)
```

本地化时要小心；如果知道，您应该本地化到数据记录的*原始*时区。如果之后需要将相同时间表示在不同时区，则使用`tz_convert`。

### 频率和重采样

时间序列通常具有内在频率（例如，每日、每小时、每月）。Pandas有时可以推断此频率，或者您可以显式设置它。了解频率有助于对齐 (alignment)和分析。

- **设置/推断频率：** `DateTimeIndex`的`freq`属性存储此信息。

```python
# 尝试推断频率（如果是不规则的，可能返回None）
print(f"\n推断频率: {pd.infer_freq(df.index)}")

# 对于规则数据，您可以显式设置频率
# 示例：如果数据保证是每日的，即使缺少周末数据
# 使用'B'表示工作日频率
# daily_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
# df = df.reindex(daily_index) # 重新索引可能会为缺失日期引入NaN
# print(f"\n设置频率: {df.index.freq}")
```

- **重采样：** 这种强大的技术更改时间序列的频率，通常涉及聚合（降采样）或插值（升采样）。例如，将每日股票价格转换为每月平均值。

```python
# 将每日数据重采样为每月频率，取'Close'的平均值
df_monthly = df['Close'].resample('M').mean() # 'M'代表月末频率
print("\n月度平均收盘价：")
print(df_monthly)

# 您可以使用其他聚合函数，例如'sum'、'last'、'ohlc'等。
# df_monthly_ohlc = df.resample('M').agg({'Open': 'first',
#                                      'High': 'max',
#                                      'Low': 'min',
#                                      'Close': 'last',
#                                      'Volume': 'sum'})
# print("\n月度OHLCV数据：")
# print(df_monthly_ohlc)
```

重采样是一项基本操作，用于比较不同粒度的时间序列或为预期特定频率的模型准备数据。



![月度平均收盘价（示例）](plots/3345-0.json)



> 通过对每日数据重采样获得的月度平均收盘价示例。

掌握这些用于加载和构建时间序列数据的Pandas技术是高效分析的第一步。在数据正确加载并按时间索引后，您现在可以进行时间特定的操作，如移动、滞后和计算滚动统计量，这些将在下一节中介绍。

## 参考资料

- [Time Series / Date functionality](https://pandas.pydata.org/docs/user_guide/timeseries.html) — The Pandas Development Team (2024)
  Pandas时间序列功能的官方最新资源，涵盖`DateTimeIndex`、`read_csv`参数、时区处理和重采样。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  Pandas创建者撰写的基石性著作，全面解释了Pandas数据结构和操作，包含时间序列数据的专门章节。
- [Hands-On Time Series Analysis with Python: Perform exploratory data analysis and implement a range of forecasting models](https://www.packtpub.com/product/hands-on-time-series-analysis-with-python/9781800569733) — Biniam Kebede (2022)
  Publisher: Packt Publishing
  提供使用Python和Pandas进行时间序列数据处理和分析的实践性、以代码为中心的指导，通过用例补充官方文档。

---
course: "time-series-analysis-forecasting"
chapter: "intro-time-series-data"
lesson: "visualizing-time-series"
sourceId: 3351
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-1-intro-time-series-data/visualizing-time-series"
title: "时间序列数据可视化"
description: "使用 Matplotlib 和 Seaborn 等库绘制时间序列数据以找出模式的技术。"
order: 5
plots: ["plots/3351-0.json"]
sourceHash: "9b359dfd94a29a56eb13e87ce6b762e9f30b4ea3eda9dc99c6bbedab2cd60a8d"
sourceCorrections: []
---

加载并完成平移和计算滚动统计量等初步处理后，下一步便是对时间序列数据进行可视化。绘制数据图表通常是快速直观地了解其潜在结构（包括找出可能的趋势、季节性模式、异常值或结构性变化）的方法。仅仅依赖汇总统计数据可能会产生误导；直观查看能提供非常有用的背景信息。

时间序列数据可视化主要使用 Matplotlib 和 Seaborn，通常借助 Pandas DataFrame 和 Series 的内置绘图功能，这些功能的底层都使用 Matplotlib。这里将详细介绍时间序列特有的图表及其解读。

### 基本时间序列图

最基本的图表是一种简单的折线图，X轴为时间，Y轴为观测值。这种图表构成了找出之前讨论过的核心组成部分的依据。

假设您有一个带有 DatetimeIndex 的 Pandas Series `ts_data`。创建一个基本图表很简单：

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 样本数据创建（请替换为您的实际数据）
date_rng = pd.date_range(start='2020-01-01', end='2022-12-31', freq='M')
data = np.sin(np.linspace(0, 2 * np.pi * 3, len(date_rng))) * 10 + \
       np.linspace(5, 15, len(date_rng)) + \
       np.random.randn(len(date_rng)) * 2 + 20
ts_data = pd.Series(data, index=date_rng, name='Value')

# 创建图表
plt.figure(figsize=(12, 6)) # 设置图表大小以便更好地阅读
plt.plot(ts_data.index, ts_data.values)

# 添加标签和标题以提高清晰度
plt.xlabel("日期")
plt.ylabel("值")
plt.title("基本时间序列图")
plt.grid(True, linestyle='--', alpha=0.6) # 添加网格
plt.tight_layout() # 调整布局
plt.show()
```

查看此图时，请留意：

1. **趋势：** 长期来看是否存在总体上升或下降方向？在下面的例子中，有明显的上升趋势。
2. **季节性：** 是否存在以固定周期（例如，每日、每周、每月、每年）重复出现的模式？该示例显示了大致每12个月重复一次的周期性模式。
3. **变动：** 数据分布是否随时间变化？它是相对恒定（同方差）还是波动（异方差）？
4. **不规则性：** 是否存在不符合一般模式的突然峰值、谷值或平稳期？这些可能是异常值或表明特定事件。



![带有趋势和季节性的样本时间序列](plots/3351-0.json)



> 一张典型的时间序列图，显示了三年内的月度值。上升趋势和年度季节性模式清晰可见。

### 滚动统计量可视化

如上一节（`shifting-lagging-rolling`）所介绍，将滚动平均值和标准差等滚动统计量与原始时间序列一起绘制是一种有用的方法，尤其适用于视觉评估平稳性（我们将在第2章正式讨论）。非恒定的滚动平均值表明存在趋势，而非恒定的滚动标准差则表示方差变化（异方差性）。

```python
# 计算滚动平均值和标准差
rolling_mean = ts_data.rolling(window=6).mean() # 示例：6个月滚动平均值
rolling_std = ts_data.rolling(window=6).std()   # 示例：6个月滚动标准差

# 绘制原始数据和滚动统计量
plt.figure(figsize=(12, 6))
plt.plot(ts_data.index, ts_data.values, color='#4dabf7', label='原始数据')
plt.plot(rolling_mean.index, rolling_mean.values, color='#f76707', label='滚动平均值 (6个月)')
plt.plot(rolling_std.index, rolling_std.values, color='#ae3ec9', label='滚动标准差 (6个月)')

plt.xlabel("日期")
plt.ylabel("值")
plt.title("带有滚动平均值和标准差的时间序列")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
```

查看这些滚动统计量如何变化，可以了解序列属性随时间的稳定性。持续增加的滚动平均值清楚表明存在趋势。

### 使用箱线图考察季节性

虽然折线图显示了随时间变化的季节性，但按季节性周期（例如，月份、季度）分组的箱线图可以提供这些周期之间更清晰的分布比较。它们有助于视觉呈现每个季节的典型值和分布范围。

要创建这些图，您通常需要从 DatetimeIndex 中提取相关时间段（如月份数字），并将其用于分组。

```python
import seaborn as sns

# 确保 ts_data 是一个带有 DatetimeIndex 的 Pandas Series
df = pd.DataFrame({'Value': ts_data})
df['Month'] = df.index.strftime('%b') # 获取月份缩写（一月、二月等）
df['Year'] = df.index.year

# 可选：如果需要，按时间顺序排列月份
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

plt.figure(figsize=(12, 6))
sns.boxplot(x='Month', y='Value', data=df, order=month_order, palette='Blues') # 使用 Seaborn 绘制简单的箱线图
plt.xlabel("月份")
plt.ylabel("值")
plt.title("时间序列月度值分布（箱线图）")
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
```

在月度箱线图中：

- 每个月的中位数线（箱体中心）显示该月的典型值。中位数在月份间的模式表明存在季节性。
- 箱体的高度（四分位距，IQR）显示该月中间50%数据的分布范围。不同的箱体高度可能表明方差随季节变化。
- 触须和异常值显示了每个月的范围和潜在极端值。

这些可视化方法是互补的。折线图显示了总体时间流向，而滚动统计量和箱线图则有助于分析趋势稳定性、季节性等特定属性。有效的可视化并非旨在创建单一的“完美”图表，而是从多个角度来理解时间相关数据中常有的丰富结构，然后才进行更正式的建模。

## 参考资料

- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts
  一本关于预测方法的综合性免费在线教材，包含时间序列可视化和探索性数据分析的详细章节。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本关于使用 Python 的 Pandas 库进行数据处理和分析的基础书籍，包含绘图和时间序列数据处理的实用示例。
- [Visualization](https://pandas.pydata.org/docs/user_guide/visualization.html) — The pandas development team (2024)
  Pandas DataFrame 和 Series 内置绘图功能的官方文档，这些功能基于 Matplotlib 且对时间序列可视化至关重要。
- [seaborn.boxplot](https://seaborn.pydata.org/generated/seaborn.boxplot.html) — Michael Waskom and the seaborn development team (2024)
  Seaborn 箱线图函数的官方文档，本节中使用该函数可视化时间序列数据的季节性分布。

---
course: "time-series-analysis-forecasting"
chapter: "intro-time-series-data"
lesson: "practice-loading-plotting"
sourceId: 3355
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-1-intro-time-series-data/practice-loading-plotting"
title: "动手实践：数据加载与绘图"
description: "应用所学知识，加载示例时间序列数据集并在Python中创建多种可视化图表。"
order: 6
plots: ["plots/3355-0.json", "plots/3355-1.json"]
sourceHash: "da0a28c659d7e783455243a2e193179c683fce012e2791cff679340a4f882e64"
sourceCorrections: []
---

我们将使用 Pandas 加载一个示例时间序列数据集，确保它正确地带有时间索引，并创建一些初步的可视化图表，以了解其基本结构。这些操作涉及数据加载、设置索引以及生成图表，用以识别趋势或季节性等模式。

首先，确保您已导入所需的库。我们将主要使用 Pandas 进行数据处理，以及 Matplotlib（或 Pandas 的绘图后端，它使用 Matplotlib）进行可视化。

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 配置图表以提高可读性
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
```

### 生成示例数据

不加载外部文件，我们来生成一个按月计的时间序列数据集。这有助于重点关注数据处理和绘图方面。我们将创建代表多年来每月小部件销售额的数据，其中包含趋势和季节性。

```python
# 生成一个5年期月度数据的日期范围
dates = pd.date_range(start='2019-01-01', periods=60, freq='MS')

# 生成带有趋势和季节性的数据
np.random.seed(42) # 用于重现性
trend = np.linspace(50, 150, 60) # 线性上升趋势
seasonality = 15 * np.sin(np.arange(60) * (2 * np.pi / 12)) # 月度季节性
noise = np.random.normal(0, 10, 60) # 随机噪声

# 组合各组成部分
sales = trend + seasonality + noise
sales = np.maximum(sales, 10) # 确保销售额为正

# 创建数据框
widget_sales = pd.DataFrame({'Sales': sales}, index=dates)

print("小部件销售示例数据:")
print(widget_sales.head())
print("\n数据信息:")
widget_sales.info()
```

执行此代码会创建一个名为 `widget_sales` 的 Pandas 数据框。`pd.date_range` 函数生成月度时间戳（'MS' 代表月份开始），我们将其直接用作索引。`.head()` 方法显示前几行，确认了数据结构：一个“Sales”列和一个 `DatetimeIndex`。`.info()` 方法确认了索引类型 (`DatetimeIndex`) 和“Sales”列的数据类型 (`float64`)。

### 可视化时间序列

数据加载并正确索引后，下一步是绘制它。简单的折线图通常是时间序列数据的最佳起点，因为它直观地表示了观察值随时间的序列。

```python
# 绘制时间序列数据
widget_sales['Sales'].plot()
plt.title('2019-2023年月度小部件销售额')
plt.xlabel('日期')
plt.ylabel('销售单位')
plt.show()
```

这是组合图表的显示效果：



![2019-2023年月度小部件销售额](plots/3355-0.json)



> 2019年1月至2023年12月的模拟月度小部件销售数据。

从这张初始图表中，我们可以看到：

1. **上升趋势**：销售额在五年期间普遍增加。
2. **季节性模式**：每年似乎都有一个重复出现的模式，峰值和谷值出现在相似的时间。
3. **噪声/不规则性**：线条并非完全平滑，表明在潜在的趋势和季节性模式周围存在随机波动。

### 进一步查看滚动统计数据

虽然正式的分解技术将在下一章介绍，但我们可以使用滚动窗口计算（前面已介绍）来直观地平滑数据并突显趋势。计算滚动平均值有助于抑制季节性和噪声。

让我们计算并绘制一个12个月的滚动平均值，并与原始数据一起呈现。

```python
# 计算12个月滚动平均值
widget_sales['Rolling Mean (12M)'] = widget_sales['Sales'].rolling(window=12).mean()

# 绘制原始数据和滚动平均值
widget_sales['Sales'].plot(label='原始销售额', legend=True)
widget_sales['Rolling Mean (12M)'].plot(label='12个月滚动平均值', legend=True, color='orange')

plt.title('带有12个月滚动平均值的小部件销售额')
plt.xlabel('日期')
plt.ylabel('销售单位')
plt.show()
```

这是组合图表的显示效果：



![带有12个月滚动平均值的小部件销售额](plots/3355-1.json)



> 原始小部件销售数据及其12个月滚动平均值的图表。

代表滚动平均值的橙色线条清晰地显示了上升趋势，平滑了蓝色线条（原始数据）中可见的季节性峰值和谷值。请注意，滚动平均值在最初的11个数据点之后才开始显示，因为它需要完整的12个观察值的窗口才能计算出第一个值。

本次实践练习展示了将时间序列数据加载到 Pandas 中、确保索引设置正确以及进行初步可视化。这些步骤是任何时间序列分析项目的基本要求，提供了对数据行为的初步了解，并为分解和建模等后续分析步骤提供了依据。

## 参考资料

- [pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html) — The pandas development team (2024)
  提供有关 Pandas DataFrame、DatetimeIndex、时间序列功能以及数据处理和绘图相关可视化工具的官方文档。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本关于使用 Python 进行数据操作和分析的指南，涵盖 Pandas DataFrame、时间序列特征和可视化方法。
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts
  一本易于理解的教科书，介绍时间序列概念，如趋势、季节性和可视化数据探索，这些对于解释初始时间序列图至关重要。

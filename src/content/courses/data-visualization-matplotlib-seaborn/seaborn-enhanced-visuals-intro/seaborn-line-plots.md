---
course: "data-visualization-matplotlib-seaborn"
chapter: "seaborn-enhanced-visuals-intro"
lesson: "seaborn-line-plots"
sourceId: 1187
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-4-seaborn-enhanced-visuals-intro/seaborn-line-plots"
title: "使用 Seaborn 绘制折线图 (lineplot)"
description: "学习使用 Seaborn 的 lineplot 函数创建折线图，常用于时间序列或趋势。"
order: 5
plots: ["plots/1187-0.json"]
sourceHash: "44ee1e9a79579c65edef2fc3b02f9f808a48d39f68d6e1eee79d7dd4c25807fd"
sourceCorrections: []
---

折线图是显示连续变量（通常是时间或位置）趋势的基本图表。尽管 Matplotlib 提供了 `plot()` 函数来创建基本折线图，但 Seaborn 的 `lineplot()` 函数功能更丰富，尤其适合处理 X 轴上每个点有多个观测值的数据集，或需要根据分类变量区分线条的情形。

`seaborn.lineplot()` 用于处理 Pandas DataFrames，可以自动聚合数据，并显示趋势线周围的置信区间等统计估计值。

### 绘制基本折线图

让我们从一个简单示例开始。假设我们有表示多天温度读数的数据。Seaborn 让绘图变得简单，尤其当您的数据在 Pandas DataFrame 中时。

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 示例数据：10 天的温度
days = np.arange(1, 11)
temperature = np.array([15, 16, 18, 17, 19, 21, 20, 22, 21, 23])

# 创建 DataFrame
temp_df = pd.DataFrame({'Day': days, 'Temperature': temperature})

# 使用 Seaborn 绘制折线图
plt.figure(figsize=(8, 4)) # 设置图表大小，便于阅读
sns.lineplot(x='Day', y='Temperature', data=temp_df)

plt.title('10 天温度趋势')
plt.xlabel('天')
plt.ylabel('温度 (°C)')
plt.grid(True, linestyle='--', alpha=0.6) # 添加浅色网格
plt.show()
```

这段代码生成一个简单的折线图，显示温度趋势。请注意我们如何将 DataFrame `temp_df` 传递给 `data` 参数 (parameter)，并使用字符串指定 X 轴和 Y 轴的列名。Seaborn 内部处理映射。

### 使用 `hue` 显示多条线

`seaborn.lineplot()` 的一个重要优点是，能够使用 `hue` 参数 (parameter)根据分类变量绘制多条线。这有助于比较不同组的趋势。

假设我们有两个不同城市 A 和 B 的温度数据。

```python
# 两个城市的示例数据
days = np.arange(1, 11)
temp_A = np.array([15, 16, 18, 17, 19, 21, 20, 22, 21, 23])
temp_B = np.array([12, 13, 14, 13, 15, 16, 17, 18, 19, 18])

# 创建适合 Seaborn 的“长格式”DataFrame
data_list = []
for i in range(len(days)):
    data_list.append({'Day': days[i], 'Temperature': temp_A[i], 'City': 'City A'})
    data_list.append({'Day': days[i], 'Temperature': temp_B[i], 'City': 'City B'})

multi_city_df = pd.DataFrame(data_list)

# 显示 DataFrame 的前几行
# print(multi_city_df.head())
#    Day  Temperature    City
# 0    1           15  City A
# 1    1           12  City B
# 2    2           16  City A
# 3    2           13  City B
# 4    3           18  City A

# 为每个城市绘制不同线条的折线图
plt.figure(figsize=(8, 4))
sns.lineplot(x='Day', y='Temperature', hue='City', data=multi_city_df, marker='o') # 添加标记

plt.title('按城市区分的温度趋势')
plt.xlabel('天')
plt.ylabel('温度 (°C)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='城市') # 优化图例
plt.show()
```

Seaborn 自动为代表“城市 A”和“城市 B”的线条分配不同颜色（由 `hue` 参数控制），并添加图例。您还可以根据其他分类变量，使用标记 (token)样式 (`style`) 或线宽 (`size`) 来区分线条。

### 聚合与置信区间

如果同一城市同一天有多个温度读数，会发生什么？默认情况下，`seaborn.lineplot()` 会聚合每个 X 值对应的多个测量值（计算平均值），并绘制此中心趋势。它还通过绘制置信区间（通常为 95% 置信区间）作为线条周围的阴影区域，来显示此估计值周围的不确定性。

我们来模拟每天有多个读数的数据：

```python
# 模拟城市 A 每天的多个读数
np.random.seed(42) # 为了可重复性
days_repeated = np.repeat(np.arange(1, 11), 5) # 每天 5 个读数
temp_A_noisy = np.concatenate([np.random.normal(loc=temp, scale=1.5, size=5) for temp in temp_A])

noisy_df = pd.DataFrame({'Day': days_repeated, 'Temperature': temp_A_noisy})

# 绘制折线图 - Seaborn 默认计算平均值和 95% 置信区间
plt.figure(figsize=(8, 4))
sns.lineplot(x='Day', y='Temperature', data=noisy_df, marker='o')

plt.title('城市 A 温度趋势（含置信区间）')
plt.xlabel('天')
plt.ylabel('温度 (°C)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```



![城市 A 温度趋势（平均值和 95% 置信区间）](plots/1187-0.json)



> 城市 A 的平均每日温度，阴影区域代表围绕平均值的 95% 置信区间，数据基于每天的多个模拟读数。

阴影区域显示了每天温度测量值的变化性或不确定性。这种自动统计估计是 Seaborn 的一个强大功能。

如果您不想显示置信区间，可以通过将 `errorbar` 参数 (parameter)设置为 `None` 来禁用它：

```python
# 绘制不含置信区间的图
plt.figure(figsize=(8, 4))
sns.lineplot(x='Day', y='Temperature', data=noisy_df, errorbar=None, marker='o') # 禁用置信区间

plt.title('城市 A 温度趋势（仅平均值）')
plt.xlabel('天')
plt.ylabel('温度 (°C)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```

您还可以使用 `estimator` 参数（例如 `estimator=np.median`）将聚合函数从平均值更改为其他值，例如中位数或标准差。

### 与 Matplotlib 自定义项结合使用

请记住，Seaborn 图表是基于 Matplotlib 的。在使用 Seaborn 创建图表后，您仍然可以使用 `plt.title()`、`plt.xlabel()`、`plt.ylabel()`、`plt.grid()` 和 `plt.legend()` 等 Matplotlib 函数来自定义图表外观，正如我们在上述示例中所做的那样。

总而言之，`seaborn.lineplot()` 提供便捷而强大的方法来创建信息丰富的折线图，尤其适用于：

- 显示随时间或序列变化的趋势。
- 方便地直接从 Pandas DataFrames 绘图。
- 使用 `hue`、`style` 或 `size` 比较不同类别之间的趋势。
- 当每个 X 值有多个观测值时，自动计算并显示统计聚合值（如平均值）和置信区间。

## 参考资料

- [seaborn.lineplot](https://seaborn.pydata.org/generated/seaborn.lineplot.html) — Michael Waskom and the Seaborn development team (2023)
  Seaborn的lineplot函数的官方API文档，详细说明了创建折线图的参数、用法和示例，包括统计估计和分类映射。
- [Plotting statistical relationships with \`relplot()\` and \`lineplot()\`](https://seaborn.pydata.org/tutorial/relational.html) — Michael Waskom and the Seaborn development team (2023)
  Seaborn关系图的教程，包括lineplot，解释了统计估计、置信区间以及将分类变量映射到颜色、样式和大小等图属性的原理。
- [Chapter 4.14. Visualization with Seaborn](https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  一本Python数据可视化指南，为Matplotlib和Seaborn提供了基础，并演示了如何有效使用Seaborn进行统计图形绘制。

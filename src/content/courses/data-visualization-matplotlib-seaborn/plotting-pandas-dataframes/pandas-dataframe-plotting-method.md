---
course: "data-visualization-matplotlib-seaborn"
chapter: "plotting-pandas-dataframes"
lesson: "pandas-dataframe-plotting-method"
sourceId: 1218
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-8-plotting-pandas-dataframes/pandas-dataframe-plotting-method"
title: "直接从DataFrame进行基础绘图"
description: "了解Pandas DataFrame的内置绘图功能，通过.plot()方法进行操作。"
order: 3
plots: ["plots/1218-0.json"]
sourceHash: "888865c8f527ef38b825cbfc5e4227606b0198a2724fe32894ad515a0b306b4c"
sourceCorrections: []
---

直接从Pandas DataFrame创建可视化图表提供了一种便捷的数据表示方法。Pandas为Series和DataFrame对象提供了一个内置的`.plot()`方法。此方法作为Matplotlib的实用接口，让您能够以最少的代码快速生成常见图表。它对于初步的数据检查尤为实用。

### `.plot()`方法

`.plot()`方法可以直接在DataFrame或Series上调用。当在DataFrame上调用时，它默认会尝试绘制所有包含数值数据的列。如果在一个Series上调用，它会绘制该单个Series中的数据。

假设我们已经将一些数据载入名为`df`的DataFrame中。例如：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # 通常用于自定义图表

# 模拟一些测量数据的示例数据
data = {
    'Temperature': [22.1, 23.5, 24.0, 23.1, 21.9],
    'Humidity': [65, 68, 70, 66, 63],
    'Pressure': [1012, 1010, 1011, 1013, 1014]
}
index_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
df = pd.DataFrame(data, index=index_labels)

print(df)
# 输出：
#      Temperature  Humidity  Pressure
# Mon         22.1        65      1012
# Tue         23.5        68      1010
# Wed         24.0        70      1011
# Thu         23.1        66      1013
# Fri         21.9        63      1014
```

要为“Temperature”列（这是一个Pandas Series）创建一个简单的折线图，您可以这样做：

```python
# 绘制单个列（Series）
df['Temperature'].plot()
plt.ylabel("Temperature (°C)") # 仍使用Matplotlib设置标签等
plt.title("Temperature Trend")
plt.show()
```

如果您在整个DataFrame上调用`.plot()`，Pandas将为每个数值列创建一个折线图，并使用DataFrame的索引作为X轴：

```python
# 绘制整个DataFrame（数值列）
df.plot()
plt.title("Sensor Readings Over Weekdays")
plt.ylabel("Measurement")
plt.show()
```

这会自动根据列名生成图例。请注意，我们仍然在用Pandas方法创建图表后，使用`matplotlib.pyplot`（作为`plt`导入）来添加标题和标签。

### 使用`kind`参数 (parameter)指定图表类型

默认的图表类型是折线图，但`.plot()`方法可以使用`kind`参数绘制其他多种类型。`kind`的一些常见选项包括：

- `'line'` : 折线图（默认）
- `'bar'` : 垂直条形图
- `'barh'` : 水平条形图
- `'hist'` : 直方图
- `'box'` : 箱线图
- `'kde'` : 核密度估计图
- `'area'` : 面积图
- `'pie'` : 饼图
- `'scatter'` : 散点图（需要指定`x`和`y`列）

让我们看几个例子。

**条形图：** 为了比较平均温度和湿度（可能来自不同的DataFrame）：

```python
# 平均值的示例数据
avg_data = {'Avg Temp': [22.9], 'Avg Humidity': [66.4]}
avg_df = pd.DataFrame(avg_data, index=['Average'])

# 创建一个垂直条形图
avg_df.plot(kind='bar')
plt.title("Average Sensor Values")
plt.ylabel("Value")
plt.xticks(rotation=0) # 保持X轴标签水平
plt.show()
```

**直方图：** 要查看单个变量（如“Pressure”）的分布：

```python
# 创建“Pressure”列的直方图
df['Pressure'].plot(kind='hist')
plt.title("Distribution of Pressure Readings")
plt.xlabel("Pressure (hPa)")
plt.show()
```

**散点图：** 为了可视化两个变量（如“Temperature”和“Humidity”）之间的关系，您需要指定`x`和`y`列。

```python
# 创建一个散点图
df.plot(kind='scatter', x='Temperature', y='Humidity')
plt.title("Temperature vs. Humidity")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.grid(True) # 添加网格线便于查看
plt.show()
```



![温度与湿度关系图](plots/1218-0.json)



> 散点图显示了工作日温度和湿度读数之间的关系。

### `.plot()`中的基础自定义设置

`.plot()`方法接受额外的参数 (parameter)，这些参数通常会直接传递给底层的Matplotlib函数。这使得无需为所有内容都进行单独的`plt`调用即可实现一些自定义设置。

常见参数包括：

- `title`: 设置图表标题。
- `xlabel`, `ylabel`: 设置坐标轴标签（尽管通常通过`plt`设置）。
- `legend`: 显示或隐藏图例（`True`/`False`）。
- `figsize`: 以英寸为单位设置图表尺寸，格式为元组`(宽度, 高度)`。
- `grid`: 显示网格线（`True`/`False`）。
- `color`: 为图表元素指定颜色。
- `style`: 定义线条样式（例如，`'-'`, `'--'`, `':'`, `'o-'`）。

以下是之前带有直接应用自定义设置的多折线图：

```python
# 带有直接自定义设置的DataFrame绘图
df.plot(figsize=(8, 4), title="Sensor Readings", grid=True, style=['o-', '--', ':'])
plt.ylabel("Measurement") # 某些设置仍可能偏好使用plt
plt.show()
```

### 何时使用DataFrame的`.plot()`方法

DataFrame上的直接绘图方法很适合用于：

1. **快速查看数据：** 无需太多设置即可快速获得数据的视觉概览。
2. **绘制简单图表：** 高效创建折线图、条形图和直方图等标准图表。
3. **便捷性：** 减少与数据操作流程结合的基础可视化所需代码量。

然而，对于高度自定义、复杂或出版质量的图表，您通常会通过直接使用Matplotlib的面向对象接口或Seaborn库获得更多控制和灵活性，我们将在后续章节中讨论这些。可以将DataFrame的`.plot()`方法看作您可视化过程中的一个便捷的起始步骤。

在接下来的章节中，我们将了解如何在运用Matplotlib和Seaborn的全部功能的同时，仍能利用Pandas DataFrame的结构化特性。

## 参考资料

- [Pandas User Guide - Visualization](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html) — The Pandas Development Team (2024)
  Pandas 内置绘图功能的官方文档，直接解释了 DataFrame 和 Series 的 `.plot()` 方法。
- [Matplotlib User's Guide](https://matplotlib.org/stable/users/index.html) — The Matplotlib Development Team (2024)
  提供了 Matplotlib 的基础知识，这是 Pandas 绘图所依赖的底层库，对进一步的自定义至关重要。

---
course: "data-visualization-matplotlib-seaborn"
chapter: "matplotlib-plotting-fundamentals"
lesson: "matplotlib-scatter-plots"
sourceId: 1153
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-2-matplotlib-plotting-fundamentals/matplotlib-scatter-plots"
title: "散点图：表示关联"
description: "学习如何使用 Matplotlib 的 scatter 函数创建散点图，以表示两个变量之间的关联。"
order: 4
plots: ["plots/1153-0.json", "plots/1153-1.json"]
sourceHash: "e92977345de58ed005fc86283ab37f2bbf2b91ce015e28fd35a12602f967f86b"
sourceCorrections: []
---

虽然折线图非常适合显示数据趋势（其中点的顺序很重要，例如值随时间变化），但我们通常希望了解两个不同变量之间的关联。一个变量增加时，另一个变量是否也倾向于增加？是否存在不明显的联系？对于可视化这类关联，散点图是标准工具。

散点图使用单个点或标记 (token)来表示两个不同数值变量获得的值。一个变量决定水平轴（X轴）上的位置，另一个变量决定垂直轴（Y轴）上的位置。其点通常不通过线连接，主要着眼于点分布形成的模式，而非顺序进展。

### 使用 `scatter()` 创建基本散点图

Matplotlib 提供了 `plt.scatter()` 函数，可方便地创建散点图。它接受两个主要参数 (parameter)：X坐标的数组或列表，以及对应的Y坐标数组或列表。

我们来看一个例子。假设我们有表示不同日期的温度（摄氏度）和冰淇淋销量的数据。我们可以使用散点图来查看是否存在关联。

```python
import matplotlib.pyplot as plt
import numpy as np

# 示例数据：温度（摄氏度）和冰淇淋销量
temperatures = np.array([14.2, 16.4, 11.9, 15.2, 18.5, 22.1, 19.4, 25.1, 23.4, 18.1, 22.6, 17.2])
scoops_sold = np.array([215, 325, 185, 332, 406, 522, 412, 614, 544, 421, 445, 408])

# 创建图和坐标轴对象
fig, ax = plt.subplots()

# 创建散点图
ax.scatter(temperatures, scoops_sold)

# 添加标签和标题（如前所述）
ax.set_xlabel("温度 (°C)")
ax.set_ylabel("冰淇淋销量")
ax.set_title("温度与冰淇淋销量")

# 显示图表
plt.show()
```

执行此脚本将生成一个散点图。图上的每个点表示一天，根据其温度和冰淇淋销量定位。



![温度与冰淇淋销量](plots/1153-0.json)



> 每个标记 (token)显示特定记录温度下的销量。

通过检查图表，您可以观察总体模式。在此例中，随着温度升高（沿X轴向右移动），冰淇淋销量也趋于增加（沿Y轴向上移动）。这表明温度和冰淇淋销量之间存在 *正向关联* 或 *正相关*。如果温度越高销量越低，我们将看到下降趋势，这表明 *负向关联*。如果点随机散布且没有可辨别的模式，则表明变量之间几乎没有关联。

### 自定义散点图外观

就像折线图一样，散点图也可以自定义以提高清晰度或视觉吸引力。`ax.scatter()` 函数（或 `plt.scatter()`）接受几个参数 (parameter)来实现此目的：

- `c`：设置标记 (token)的颜色。您可以使用颜色名称（'red'）、十六进制代码（'#FF5733'）或缩写（例如'g'代表绿色）。
- `s`：设置标记的大小。它接受一个表示标记面积的数值。
- `marker`：指定标记的形状。常见选项包括'o'（圆形，默认）、's'（方形）、'^'（向上三角形）、'd'（菱形）、'\*'（星形）和'+'。
- `alpha`：控制标记的透明度，范围从0（完全透明）到1（完全不透明）。当许多点重叠时，此参数非常有用，可让您查看不同区域中点的密度。

我们来修改我们之前的例子，使用更大、半透明、橙色方形标记。

```python
import matplotlib.pyplot as plt
import numpy as np

# 示例数据（同前）
temperatures = np.array([14.2, 16.4, 11.9, 15.2, 18.5, 22.1, 19.4, 25.1, 23.4, 18.1, 22.6, 17.2])
scoops_sold = np.array([215, 325, 185, 332, 406, 522, 412, 614, 544, 421, 445, 408])

# 创建图和坐标轴对象
fig, ax = plt.subplots()

# 创建自定义散点图
ax.scatter(temperatures, scoops_sold,
           c='#fd7e14',      # 橙色
           s=60,             # 更大的标记尺寸
           marker='s',       # 方形标记
           alpha=0.7)        # 略微透明

# 添加标签和标题
ax.set_xlabel("温度 (°C)")
ax.set_ylabel("冰淇淋销量")
ax.set_title("温度与冰淇淋销量（自定义）")

# 显示图表
plt.show()
```

此代码生成一个具有视觉上独特标记的图表。



![温度与冰淇淋销量（自定义）](plots/1153-1.json)



> 带有橙色方形标记、更大尺寸和透明度的自定义散点图。

请记住，您还可以应用之前学到的其他自定义设置，例如使用 `ax.set_xlim()` 和 `ax.set_ylim()` 设置坐标轴限制，以及使用 `ax.grid()` 添加网格线。

### 散点图与折线图对比

为数据选择合适的图表类型很重要。

- 当您想可视化连续区间或序列上的趋势或进展时（例如跟踪多日的股票价格或测量随时间变化的传感器值），请使用**折线图**（`plot()`）。点之间的连接意味着连续性或顺序。
- 当您想检查两个不同数值变量之间的关联时，请使用**散点图**（`scatter()`）。每个点代表一个独立的观测值，用线连接它们通常没有意义。如果使用 `ax.plot()` 绘制我们的温度与销量数据，它将按照数据点在数组中出现的顺序连接它们，形成一条混乱的折线，不代表有意义的趋势。

散点图是用于查看变量之间潜在关联和模式的**核心工具**，可在应用更复杂的统计分析或机器学习 (machine learning)模型之前使用。它们提供关于两个量之间有何关联的直接视觉验证。

## 参考资料

- [\`matplotlib.pyplot.scatter\`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html) — Matplotlib Development Team (2024)
  Matplotlib散点图函数的官方文档，详细说明所有参数和用法。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/04.02-simple-scatter-plots.html) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  使用Matplotlib进行数据可视化的实用指南，其中包含专门的散点图部分。
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media; Pages: 387
  本书章节提供了可视化数值变量之间关系的理论基础，强调了散点图的作用和设计。

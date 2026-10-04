---
course: "data-visualization-matplotlib-seaborn"
chapter: "basic-matplotlib-plot-types"
lesson: "matplotlib-vertical-horizontal-bars"
sourceId: 1167
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-3-basic-matplotlib-plot-types/matplotlib-vertical-horizontal-bars"
title: "创建垂直和水平条形图"
description: "学习如何在 Matplotlib 中创建垂直 (`bar`) 和水平 (`barh`) 条形图。"
order: 2
plots: ["plots/1167-0.json", "plots/1167-1.json"]
sourceHash: "ca6bf054c3d00b468681707dad67fa20bb9f26641a67ac746db548f84ce5d010"
sourceCorrections: []
---

条形图是比较不同项目或类别数量或大小的优秀工具。假设您想比较不同产品的月销售额、多个城市的人口或不同候选人获得的票数。在这种情况下，条形图能够清晰直观地显示这些比较。Matplotlib 提供了直接的函数来生成垂直和水平条形图。

### 使用 `plt.bar()` 生成垂直条形图

最常见的类型是垂直条形图，其中每个条形的高度代表特定类别的数值大小。我们为此使用 `plt.bar()` 函数。

`plt.bar()` 函数主要需要两个参数 (parameter)：

1. X轴上条形的位置或标签（代表类别）。
2. Y轴上条形的高度（代表数值）。

我们来生成一个简单的垂直条形图，显示某供应商一天内售出的不同水果数量。

```python
import matplotlib.pyplot as plt
import numpy as np

# 数据：类别及其对应值
fruits = ['Apples', 'Oranges', 'Bananas', 'Grapes']
quantity_sold = [45, 30, 60, 25]

# 创建图和轴对象
fig, ax = plt.subplots()

# 生成垂直条形图
ax.bar(fruits, quantity_sold, color='#339af0') # 使用蓝色

# 添加标签和标题以便清晰显示
ax.set_xlabel('Fruit Type')
ax.set_ylabel('Quantity Sold')
ax.set_title('Daily Fruit Sales')

# 显示图表
plt.show()
```



![每日水果销量](plots/1167-0.json)



> 不同水果的销售数量，由蓝色条形的高度表示。

在这个例子中，`fruits` 列表提供了X轴的类别标签，`quantity_sold` 提供了Y轴上条形的数值高度。我们使用从 `plt.subplots()` 获得的 `ax` 对象上的方法添加了标签和标题，这与前面讨论的结构相符。`color` 参数使我们能够指定条形的颜色；这里我们选择了蓝色的一种色调。

### 使用 `plt.barh()` 生成水平条形图

有时，特别是当您有许多类别或类别标签很长时，水平条形图会更易读。Matplotlib 为此提供了 `plt.barh()` 函数（bar horizontal 的缩写）。

`plt.barh()` 函数的工作方式与 `plt.bar()` 类似，但轴的角色互换了：

1. Y轴上条形的位置或标签（类别）。
2. X轴上条形的宽度（数值）。

我们来重新生成水果销售图表，这次是水平方向的。

```python
import matplotlib.pyplot as plt
import numpy as np

# 数据：类别及其对应值
fruits = ['Apples', 'Oranges', 'Bananas', 'Grapes']
quantity_sold = [45, 30, 60, 25]

# 创建图和轴对象
fig, ax = plt.subplots()

# 生成水平条形图
ax.barh(fruits, quantity_sold, color='#51cf66') # 使用绿色

# 添加标签和标题
# 注意：与垂直图相比，轴标签已互换
ax.set_xlabel('Quantity Sold')
ax.set_ylabel('Fruit Type')
ax.set_title('Daily Fruit Sales (Horizontal)')

# 显示图表
plt.show()
```



![每日水果销量（水平）](plots/1167-1.json)



> 不同水果的销售数量，由沿水平轴的绿色条形长度表示。类别垂直排列。

请注意，`barh` 的输入是相同的列表（`fruits`、`quantity_sold`），但它们的解释方式有所不同。`fruits` 现在定义了Y轴上的位置，而 `quantity_sold` 定义了沿X轴延伸的条形长度。因此，轴标签（`set_xlabel`、`set_ylabel`）也与垂直版本互换了。水平条形图在类别名称较长时通常很有用，避免了垂直图X轴上可能出现的重叠问题。

`plt.bar()` 和 `plt.barh()` 都是比较不同数量的基础工具。您可以进一步自定义它们的显示效果，例如更改颜色、添加误差线（代表不确定性）或调整条形宽度，这些方面我们稍后会提到。现在，请着重理解何时使用每种类型以及如何提供基本的类别和数值数据来生成它们。

## 参考资料

- [Bar chart](https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_chart_with_labels.html) — Matplotlib Development Team (2012-2025)
  Publisher: Matplotlib
  官方示例，展示如何使用 `plt.bar()` 和 `plt.barh()` 创建垂直和水平条形图，并提供自定义选项。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/04.00-introduction-to-matplotlib.html) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  涵盖 Matplotlib 基础知识，包括条形图，为数据可视化提供扎实基础。
- [The Visual Display of Quantitative Information](https://www.edwardtufte.com/tufte/books_vdqi) — Edward R. Tufte (2001)
  Publisher: Graphics Press
  视觉传播的经典著作，讨论有效数据显示的原则，包括条形图在比较中的适当使用。

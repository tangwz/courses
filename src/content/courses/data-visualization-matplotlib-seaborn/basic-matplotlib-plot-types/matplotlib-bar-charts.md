---
course: "data-visualization-matplotlib-seaborn"
chapter: "basic-matplotlib-plot-types"
lesson: "matplotlib-bar-charts"
sourceId: 1165
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-3-basic-matplotlib-plot-types/matplotlib-bar-charts"
title: "用于比较的条形图"
description: "学习如何使用 Matplotlib 的 bar 函数创建条形图，以比较分类数据。"
order: 1
plots: ["plots/1165-0.json"]
sourceHash: "85c252778a1861424fe47374dd32d59465ac4a0654c0d5ccc400bd57ba42616f"
sourceCorrections: []
---

数据分析中一项常见的任务是比较不同组或类别之间的数量。例如，您可能想比较不同城市的月平均降雨量、各种产品线的销售数据，或者应用程序不同功能的参与用户数量。条形图是完成这类比较的优秀工具。它们使用长度与所代表的值成比例的矩形条，使视觉比较直接明了。

Matplotlib 提供 `plt.bar()` 函数来创建垂直条形图。其基本用法需要两个核心输入：一个轴上（通常是垂直条形图的 x 轴）的条形位置（或标签），以及另一个轴上（通常是 y 轴）表示值的条形高度。

### 创建简单条形图

让我们从一个基本示例开始。假设我们有一篮子不同种类水果的数量数据：

```python
import matplotlib.pyplot as plt
import numpy as np # 通常很有用，但对于这个简单示例并非严格需要

# 数据
fruit_types = ['Apples', 'Oranges', 'Bananas', 'Grapes']
counts = [15, 12, 20, 8]

# 创建图和轴对象
fig, ax = plt.subplots()

# 创建条形图
ax.bar(fruit_types, counts)

# 添加标签和标题以增强清晰度
ax.set_xlabel('Fruit Type')
ax.set_ylabel('Quantity')
ax.set_title('Number of Fruits in the Basket')

# 显示图表
plt.show()
```

这个脚本首先定义了我们的类别 (`fruit_types`) 及其对应的数值 (`counts`)。然后我们使用 `plt.subplots()` 创建一个图和一个轴对象。可视化的主要部分是 `ax.bar(fruit_types, counts)`，它指示 Matplotlib 绘制条形图。第一个参数 (parameter)提供了 x 轴刻度的标签，第二个参数提供了每个对应条形的高度。最后，我们添加标签和标题，类似于我们在前一章中自定义折线图和散点图的方式，并显示图表。

结果图表清楚地表明，香蕉是数量最多的水果，而葡萄是数量最少的。视觉上的高度差异使这种比较一目了然。



![篮子里的水果数量](plots/1165-0.json)



> 一个比较不同水果数量的简单条形图。

### 自定义条形图外观

像其他 Matplotlib 图表一样，条形图也可以自定义。一种常见的自定义是通过 `bar()` 函数中的 `color` 参数 (parameter)来改变条形的颜色。

```python
import matplotlib.pyplot as plt

# 数据
fruit_types = ['Apples', 'Oranges', 'Bananas', 'Grapes']
counts = [15, 12, 20, 8]

# 创建图和轴对象
fig, ax = plt.subplots()

# 创建不同颜色的条形图
ax.bar(fruit_types, counts, color='#74c0fc') # 使用浅蓝色

# 添加标签和标题
ax.set_xlabel('Fruit Type')
ax.set_ylabel('Quantity')
ax.set_title('Number of Fruits in the Basket (Custom Color)')

# 显示图表
plt.show()
```

这种修改只是改变了视觉外观，而没有改变数据表示。您可以使用颜色名称（如“red”、“green”）、十六进制代码（如“#74c0fc”）或 Matplotlib 支持的其他格式来指定颜色。

条形图是进行分类比较的基本工具。通过将值表示为条形长度，它们提供了一种直观的方式来查看组间的差异和相似性。`plt.bar()` 函数提供了一种直接的方式来生成这些重要的可视化图表。在下一节中，我们将介绍如何创建水平条形图，并讨论何时可能更倾向于使用它们。

## 参考资料

- [\`matplotlib.pyplot.bar\`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html) — Matplotlib Development Team (2024)
  官方 `bar` 函数文档，详细说明其参数、用法以及创建条形图的示例。
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media
  一本关于创建有效可视化的指南，讨论了为分类比较选择条形图等适当图表类型的原则。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  一本关于Python科学计算的实用参考书，其中有一章专门介绍Matplotlib，包括条形图的创建和自定义。

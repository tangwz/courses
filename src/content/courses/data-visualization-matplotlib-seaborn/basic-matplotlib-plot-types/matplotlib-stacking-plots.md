---
course: "data-visualization-matplotlib-seaborn"
chapter: "basic-matplotlib-plot-types"
lesson: "matplotlib-stacking-plots"
sourceId: 1176
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-3-basic-matplotlib-plot-types/matplotlib-stacking-plots"
title: "堆叠图表"
description: "简要介绍在Matplotlib中创建堆叠柱状图或面积图的方法。"
order: 6
plots: ["plots/1176-0.json"]
sourceHash: "8fdc8b11f0767802b12be4f803ff33db5fd094a0a4a1371f6d96f2718860f09d"
sourceCorrections: []
---

有时，你不仅想比较不同类别下的总数量；你还想了解这些总数量是由哪些部分组成的。设想比较不同月份的销售数据。传统柱状图显示每个月的总销售额。但如果你想了解不同产品线（例如，电子产品、服装、家居用品）在每个月总销售额中的贡献，该怎么办？此时，堆叠图表就很有用。堆叠图表让你能够在一个条形图或面积图中，将整体的不同组成部分以层叠方式呈现。

### 堆叠柱状图

堆叠柱状图是传统柱状图的一种变体。传统柱状图通常用于比较不同类别之间的总量。在此基础上，堆叠柱状图更进一步，它不是将柱子并排放置，而是将代表不同子组的段落堆叠在一起。每个柱子的总高度仍代表该类别的总计，但这些段落显示了每个子组的相对贡献。

在Matplotlib中创建堆叠柱状图时，你需要使用 `plt.bar()` 绘制多个数据集，但对于后续的数据集，你需要使用 `bottom` 参数 (parameter)。`bottom` 参数指定新柱子底部应开始的Y坐标。对于绘制的第二个数据集，将 `bottom` 设置为第一个数据集的值。对于第三个数据集，将 `bottom` 设置为第一个和第二个数据集的和，以此类推。

让我们看一个例子。假设我们有三个产品类别（A、B、C）在四个季度（Q1、Q2、Q3、Q4）的销售数据。

```python
import matplotlib.pyplot as plt
import numpy as np

# 示例数据：3个产品类别每季度的销售额
categories = ['Q1', 'Q2', 'Q3', 'Q4']
sales_A = np.array([150, 200, 180, 220])
sales_B = np.array([250, 180, 300, 280])
sales_C = np.array([100, 120, 150, 100])

# 计算B和C的底部位置
bottom_B = sales_A
bottom_C = sales_A + sales_B # A和B的和

# 创建图和轴
fig, ax = plt.subplots(figsize=(8, 5)) # 调整图表大小以提高可读性

# 绘制每个类别的柱子，并进行堆叠
ax.bar(categories, sales_A, label='产品 A', color='#4dabf7') # 蓝色
ax.bar(categories, sales_B, label='产品 B', bottom=bottom_B, color='#69db7c') # 绿色
ax.bar(categories, sales_C, label='产品 C', bottom=bottom_C, color='#ff922b') # 橙色

# 添加标签和标题
ax.set_xlabel('季度')
ax.set_ylabel('总销售额')
ax.set_title('按产品类别划分的季度销售额')
ax.legend() # 显示图例以识别类别

# 优化布局并显示图表
plt.tight_layout()
plt.show()
```



![按产品类别划分的季度销售额](plots/1176-0.json)



> 堆叠柱状图显示了三种产品在四个季度中的销售贡献。每个段的高度代表其销售额，而整个柱子的高度代表该季度的总销售额。

在这段代码中，我们首先绘制 `sales_A`。接着，我们绘制 `sales_B` 并指定 `bottom=sales_A`，这样产品 B 的柱子就会从产品 A 的柱子顶部开始。最后，我们绘制 `sales_C`，其 `bottom` 参数为 `sales_A + sales_B`。`label` 参数对于在调用 `ax.legend()` 时识别哪种颜色对应哪个产品类别很重要。

### 堆叠面积图

与堆叠柱状图类似，堆叠面积图显示了部分如何构成整体，但它们通常用于连续轴，例如时间。不同于柱子，连续的面积区域相互堆叠。Matplotlib 为此提供了 `plt.stackplot()` 函数。尽管它们在显示随时间变化的构成趋势方面很有用，但它们与堆叠柱状图有一个共同的局限性：很难准确判断不在底部的段落的大小或趋势。我们在此不会详细讨论，但请注意这种适用于连续数据的替代方案。

### 何时使用堆叠图

堆叠图表在以下情况有效：

1. 你希望比较不同类别下的总数量。
2. 你还希望展示不同子组对总量的相对（或绝对）贡献。
3. 子组（段）的数量相对较少（通常少于5-6个），以避免视觉混乱。

然而，请谨慎使用。比较段落的精确大小可能很困难，特别是对于那些没有从基线开始的段落。如果主要目的是比较不同主类别下的各个子组，那么独立的或分组的柱状图可能会更清晰。

## 参考资料

- [\`matplotlib.pyplot.bar\` API Reference](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html) — Matplotlib Development Team (2024)
  官方文档，用于创建条形图，包括堆叠条形图的详细用法。
- [\`matplotlib.pyplot.stackplot\` API Reference](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.stackplot.html) — Matplotlib Development Team (2024)
  官方文档，用于创建堆叠面积图。
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media
  一本关于数据可视化原则的综合性书籍，讨论了堆叠图表的恰当使用和潜在误解。
- [Show Me the Numbers: Designing Tables and Graphs to Enlighten](https://www.analytics-press.com/books/show-me-the-numbers) — Stephen Few (2012)
  Publisher: Analytics Press
  关于有效数据可视化的基础指南，提供图表选择和堆叠图表相关挑战的见解。

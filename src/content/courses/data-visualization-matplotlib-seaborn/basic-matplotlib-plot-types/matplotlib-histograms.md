---
course: "data-visualization-matplotlib-seaborn"
chapter: "basic-matplotlib-plot-types"
lesson: "matplotlib-histograms"
sourceId: 1170
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-3-basic-matplotlib-plot-types/matplotlib-histograms"
title: "用于表示分布的直方图"
description: "学习使用 Matplotlib 的 hist 函数来显示单个数值变量的分布。"
order: 3
plots: ["plots/1170-0.json"]
sourceHash: "0cd7fb593557aab9063604b7fab82922bbbda622aede7c7851e4e158b4fc085b"
sourceCorrections: []
---

虽然条形图有助于比较离散类别，但我们通常需要了解单个连续数值变量的*分布*。这些数值是如何分散的？它们是聚集在一个中心点附近，还是分布更均匀？这时，直方图就派上用场了。

直方图是一种图形表示方法，它将一组数据点组织成特定的范围或区间，这些区间称为*数据桶*（bins）。它看起来与条形图相似，但有一个显著差异：直方图显示*连续*或*离散数值*数据的频率分布，而条形图则比较*类别*数据。直方图中每个条的高度表示落入该特定数据桶的数据点数量（频率）。

可以这样理解：就像把数值分到不同的桶里。如果你有一份学生身高列表，直方图可以告诉你多少学生的身高在150-160厘米之间，多少在160-170厘米之间，依此类推。这能让你直观地感知到数据的形状、中心和离散程度。

### 使用 `plt.hist()` 创建直方图

Matplotlib 提供 `plt.hist()` 函数，可以轻松创建直方图。其最基本的用法只需要一个参数 (parameter)：你想要显示其分布的数据序列（例如列表或 NumPy 数组）。

让我们生成一些示例数据，例如学生的考试分数，然后绘制直方图。为了便于说明，我们将使用 NumPy 创建一些服从正态分布的随机数据。

```python
import matplotlib.pyplot as plt
import numpy as np

# 生成一些示例数据（例如，考试分数）
# 使用均值为70、标准差为10的正态分布
np.random.seed(42) # 用于结果可复现
scores = np.random.normal(loc=70, scale=10, size=200)

# 创建直方图
plt.figure(figsize=(8, 5)) # 可选：调整图表大小
plt.hist(scores, color='#339af0') # 将数据传递给 plt.hist()

# 添加标签和标题以提高清晰度
plt.xlabel("考试分数")
plt.ylabel("学生人数（频率）")
plt.title("考试分数分布")

# 显示图表
plt.show()
```

这段代码首先生成200个大致服从正态分布的随机分数。然后，`plt.hist(scores, color='#339af0')` 接收这些数据，并自动计算数据桶的范围和频率，以绘制直方图的条形。我们还使用 `plt.xlabel()`、`plt.ylabel()` 和 `plt.title()` 等函数添加了标签和标题，这些是您之前学习过的。结果是一个图表，显示了有多少学生分数落入不同的分数区间。



![考试分数分布](plots/1170-0.json)



> 一个直方图，显示了200个模拟考试分数的分布。x轴显示分数范围（数据桶），y轴显示落入每个范围的学生人数。其形状近似于钟形曲线（正态分布）。

### 为什么要使用直方图？

直方图是探索性数据分析中的基本工具。它们可以帮助您快速理解：

- **数据形状：** 数据是对称的（如钟形曲线）、偏向一侧（左偏或右偏）、双峰的，还是均匀分布的？
- **集中趋势：** 大多数数值聚集在哪里？您可以直观地估算数据的近似中心。
- **离散程度或变异性：** 数值是紧密集中在中心附近，还是广泛分散在整个范围内？
- **异常值：** 是否有条形远离直方图的主体，表明存在异常值？

默认情况下，`plt.hist()` 会根据输入数据自动确定数据桶的数量和宽度。虽然这种自动选择通常足够，但数据桶的选择会明显影响直方图的外观，并可能改变您对数据分布的理解。我们将在下一节查看如何控制这些数据桶的数量和宽度。

## 参考资料

- [matplotlib.pyplot.hist](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html) — Matplotlib Development Team (2024)
  Matplotlib `hist` 函数的官方文档，详细介绍了其参数、用法以及创建直方图的示例。
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media
  一本关于创建准确有效数据可视化的原则和实践指南，其中包含有关直方图及其正确解释的部分。
- [Practical Statistics for Data Scientists: 50+ Essential Concepts Using R and Python](https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/) — Peter Bruce, Andrew Bruce, and Peter Gedeck (2020)
  Publisher: O'Reilly Media
  本书提供了以数据为中心的统计方法介绍，讨论了使用直方图等工具可视化数据分布对探索性数据分析的重要性。

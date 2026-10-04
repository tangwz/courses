---
course: "data-visualization-matplotlib-seaborn"
chapter: "basic-matplotlib-plot-types"
lesson: "matplotlib-histogram-bins"
sourceId: 1173
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-3-basic-matplotlib-plot-types/matplotlib-histogram-bins"
title: "理解直方图的分箱"
description: "了解分箱数量如何影响Matplotlib直方图的外观和对数据的理解。"
order: 4
plots: ["plots/1173-0.json", "plots/1173-1.json", "plots/1173-2.json"]
sourceHash: "23bbb444ba17211f3e51c3220fa702ada1d3e8b1964f5ee19c73e45de193d14d"
sourceCorrections: []
---

直方图是理解数据分布的一种基本可视化工具。它们显示落在特定范围内的数据点的频率（或计数），从而直观地展示数据的分布情况。在 Python 中，`plt.hist()` 通常用于创建这些可视化图表。直方图中的条形代表这些频率，每个条形对应一个特定的值范围。这些范围被称为**分箱**。可以将分箱想象成沿数轴排列的容器；每个数据点都会根据其值放入相应的容器中。每个容器条形的高度表示其容纳的数据点数量。

了解和控制这些分箱对制作信息量大的直方图非常重要。分箱的选择会很大程度上改变直方图的外观，进而影响对数据分布的理解。

### 分箱的作用

每个分箱覆盖数据取值范围内的特定区间。例如，如果您的数据范围是0到100，您可能会有覆盖0-10、10-20、20-30等的分箱。一个值为15的数据点会落入10-20的分箱中，增加该分箱的计数（从而增加条形的高度）。

默认情况下，Matplotlib的 `plt.hist()` 函数会尝试为数据选择一个合理的分箱数量。但这个默认值并非总是最理想的。

### 分箱数量的影响

我们来看看改变分箱数量如何影响生成的直方图。我们将使用从正态分布中抽取的一些样本数据。

```python
import matplotlib.pyplot as plt
import numpy as np

# 生成一些样本数据
np.random.seed(42) # 为了结果可复现
data = np.random.randn(200) * 1.5 + 5 # 200个点，均值=5，标准差=1.5

# --- 绘制不同分箱数量的图 ---

plt.figure(figsize=(12, 4)) # 创建一个图表以容纳子图

# 图1: 分箱过少
plt.subplot(1, 3, 1) # （行数，列数，面板编号）
plt.hist(data, bins=5, color='#228be6', edgecolor='white')
plt.title('分箱过少 (bins=5)')
plt.xlabel('值')
plt.ylabel('频率')

# 图2: 默认分箱数量 (Matplotlib 决定)
plt.subplot(1, 3, 2)
plt.hist(data, color='#15aabf', edgecolor='white') # 让 Matplotlib 选择分箱
plt.title('默认分箱')
plt.xlabel('值')
# plt.ylabel('频率') # （可选）隐藏中间图的Y轴标签

# 图3: 分箱过多
plt.subplot(1, 3, 3)
plt.hist(data, bins=50, color='#40c057', edgecolor='white')
plt.title('分箱过多 (bins=50)')
plt.xlabel('值')
# plt.ylabel('频率') # （可选）隐藏Y轴标签

plt.tight_layout() # 调整布局以防止重叠
plt.show()
```

从上面代码生成的图中可以看出：

1. **分箱过少：** 只使用5个分箱会生成一个非常粗略的表示。我们损失了许多关于分布形状的细节。图表看起来块状，很难判断数据是否真的是钟形。
2. **默认分箱：** Matplotlib的默认选择（对于此数据量通常约为10个分箱）能更好地表示集中趋势和分散程度。我们可以开始看出正态分布的典型形状。
3. **分箱过多：** 使用50个分箱会导致直方图非常锯齿化。许多分箱的计数很低（甚至为零）。这种“尖刺状”外观可能具有误导性，它强调的是样本中的随机波动，而不是数据分布的实际形状。这可能暗示着一些并非真正重要的模式或空隙。

我们用交互式图表来比较一下。



![5个分箱的直方图](plots/1173-0.json)



> 使用5个分箱的直方图。整体形状得以呈现，但细节有所丢失。



![默认分箱（约10个）的直方图](plots/1173-1.json)



> 使用Matplotlib默认分箱数量的直方图。这通常提供了一个合理的起点。



![50个分箱的直方图](plots/1173-2.json)



> 使用50个分箱的直方图。这显示了过多的细节和噪声，使得数据本身的模式更难看出。

### 如何在Matplotlib中控制分箱

您可以使用 `bins` 参数 (parameter)在 `plt.hist()` 中控制分箱：

1. **指定分箱数量：** 将一个整数传递给 `bins` 参数。Matplotlib会创建指定数量的等宽分箱，覆盖数据取值范围。

   ```python
   # 创建一个包含20个分箱的直方图
   plt.hist(data, bins=20, color='#845ef7', edgecolor='black')
   plt.title('20个分箱的直方图')
   plt.xlabel('值')
   plt.ylabel('频率')
   plt.show()
   ```
2. **指定分箱边缘：** 传递一个列表或NumPy数组，用于定义每个分箱的确切边界（边缘）。这使您可以精确控制分箱的起始和结束位置。如果您提供 $N$ 个边缘，您将得到 $N-1$ 个分箱。

   ```python
   # 定义特定的分箱边缘
   bin_edges = [0, 2, 4, 6, 8, 10] # 创建分箱：[0,2), [2,4), [4,6), [6,8), [8,10]

   plt.hist(data, bins=bin_edges, color='#f76707', edgecolor='black')
   plt.title('自定义分箱边缘的直方图')
   plt.xlabel('值')
   plt.ylabel('频率')
   plt.xticks(bin_edges) # 将x轴刻度设置为与分箱边缘匹配，以便清晰显示
   plt.show()
   ```

   *注意：* 符号 `[0, 2)` 表示分箱包含0但不包含2（最后一个分箱除外，它包含两个边缘）。

### 选择“正确”的分箱数量

那么，应该使用多少个分箱呢？遗憾的是，没有一个完美的答案。这通常需要一定的判断，并取决于：

- **数据量：** 数据越多，通常可以支持更多的分箱，而不会使直方图过于嘈杂。
- **分布的形状：** 某些分布可能需要更多的分箱才能显示出其特定特征。
- **可视化的目的：** 您是在研究数据还是在呈现一个具体发现？

**一般建议：**

- **从默认值开始：** Matplotlib的默认值通常是一个合理的起点。
- **尝试不同值：** 试用几种不同的分箱数量（例如，默认值的一半或两倍），看看外观如何变化。
- **考虑既定规则（可选知识）：** 数据科学家有时会使用诸如Sturges法则或Freedman-Diaconis法则等公式来估计最佳分箱宽度或数量。Matplotlib和Seaborn通常会在其默认计算中纳入这些规则。您通常无需自己计算这些，但了解默认值并非随意选择是很有益的。
- **优先考虑清晰性：** 选择一个分箱数量，使其能清楚地传达分布的主要特征（集中趋势、分散程度、偏度和多峰性），同时避免过于平滑或过度嘈杂。

选择合适的分箱是一项通过经验获得的实用技能。不要害怕尝试不同的值，直到直方图能够有效呈现数据本身的模式。

## 参考资料

- [matplotlib.pyplot.hist](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html) — Matplotlib Development Team (2024)
  Matplotlib 中创建直方图的官方文档，详细说明了 `bins` 参数及其各种分箱选择方法。
- [Practical Statistics for Data Scientists](https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/) — Peter Bruce, Andrew Bruce, and Peter Gedeck (2020)
  Publisher: O'Reilly Media
  一本涵盖与数据科学相关的统计概念的实用指南，包括直方图的用途和解释以及分箱选择的影响。
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media
  提供了创建有效数据可视化的指导原则，深入了解直方图中分箱选择如何影响数据分布的准确呈现。

---
course: "data-visualization-matplotlib-seaborn"
chapter: "seaborn-visualizing-distributions"
lesson: "seaborn-violinplot"
sourceId: 1193
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-5-seaborn-visualizing-distributions/seaborn-violinplot"
title: "小提琴图：结合箱线图和核密度估计图 (violinplot)"
description: "理解并创建使用 Seaborn 的小提琴图，结合了箱线图和核密度估计图的特点。"
order: 4
plots: ["plots/1193-0.json"]
sourceHash: "182038fef1b3a537248874cf2adfeabf1ca8b46aa43aeff142832ae1a07ef6ab"
sourceCorrections: []
---

箱线图通过四分位数对分布进行简洁概括，但未能显示其完整形态。另一方面，核密度估计 (KDE) 图擅长呈现形态，但没有明确标示诸如中位数或四分位距 (IQR) 等概括性统计量。小提琴图巧妙地结合了这两种方法。

小提琴图在两侧镜像显示核密度估计图（形成“小提琴”形状），并且通常在内部包含概括性统计量的表示，类似于箱线图。这让您可以同时看到整体形态、密度、模式（峰值数量）以及重要的概括点，与单独使用箱线图或核密度估计图相比，能对数据分布提供更全面的观察。在比较不同类别间的分布时，它们特别有效。

## 使用 `seaborn.violinplot` 创建小提琴图

Seaborn 通过 `seaborn.violinplot()` 函数让创建小提琴图变得简单。我们从可视化单个数值变量的分布开始。我们将使用常见的“tips”数据集。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 加载示例数据集
tips = sns.load_dataset("tips")

# 为“total_bill”列创建小提琴图
plt.figure(figsize=(6, 4)) # 调整图表大小以提高可读性
sns.violinplot(y=tips["total_bill"])
plt.title("总账单金额分布")
plt.ylabel("总账单 ($)")
plt.show()
```

> 一个展示总账单金额分布的小提琴图。最宽的部分表示大多数账单所在的区间，其形状表明略微右偏。

在此图中：

- **外部形状**是核密度估计图，呈镜像显示。小提琴在任意给定 $y$ 值处的宽度表示该值附近数据点的密度或频率。
- **内部元素**（默认情况下）显示一个微型箱线图：白点是中位数，粗黑条代表四分位距（IQR，从Q1到Q3），细黑线延伸到相邻值（通常是从四分位数起1.5倍IQR的范围，类似于箱线图的触须）。

## 比较不同类别间的分布

小提琴图真正的优势在于比较由分类变量定义的、不同组之间数值变量的分布时显现。我们可以将分类变量分配给 `x` 轴，将数值变量分配给 `y` 轴。

让我们比较每周中每一天的 `total_bill` 分布。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 加载示例数据集
tips = sns.load_dataset("tips")

# 比较不同日期的总账单分布
plt.figure(figsize=(8, 5))
sns.violinplot(x="day", y="total_bill", data=tips, palette="coolwarm") # 使用调色板
plt.title("总账单按天分布")
plt.xlabel("星期")
plt.ylabel("总账单 ($)")
plt.show()
```



![按天划分的总账单分布（Plotly 示例）](plots/1193-0.json)



> 比较按天划分的总账单分布的示例小提琴图。注意：此 Plotly JSON 表示用于演示的简化结构；完整的转换将包含所有数据点或计算出的统计数据，以获得准确的形状。Python 代码生成的 Seaborn 图表提供了标准的可视化效果。

此图方便视觉比较：

- **形态**：与周四和周五相比，周六和周日的账单在高端似乎更分散。
- **中位数**：白点表示每天的典型账单金额。
- **离散度**：主体（IQR）的厚度和整体长度表示变异性。

## 自定义小提琴图

Seaborn 的 `violinplot` 提供多个参数 (parameter)用于自定义：

- **`inner`**：控制小提琴内部的显示。选项包括：
  - `'box'`：（默认）显示迷你箱线图。
  - `'quartiles'`：显示三个四分位数（Q1、中位数、Q3）的线。
  - `'point'` 或 `'stick'`：显示单个观测值或小竖线。
  - `None`：仅显示小提琴形状。
- **`palette`**：根据分类变量为小提琴应用不同颜色（如上所示）。
- **`hue`**：在 x 轴上的每个主要类别内，使用颜色添加另一层分类。
- **`split`**：如果 `hue` 恰好有两个级别，设置 `split=True` 会为 `hue` 变量的每个级别绘制一半小提琴，允许在同一小提琴空间内进行并排直接比较。
- **`scale`**：决定小提琴的宽度如何缩放。
  - `'area'`：（默认）将小提琴缩放至相同面积。
  - `'count'`：按该类别中的观测数量缩放小提琴宽度。
  - `'width'`：将小提琴缩放至相同的最大宽度。

让我们使用 `hue` 和 `split` 参数，比较每天吸烟者和非吸烟者之间的账单：

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 加载示例数据集
tips = sns.load_dataset("tips")

# 按天比较总账单，并按吸烟者状态划分
plt.figure(figsize=(10, 6))
sns.violinplot(x="day", y="total_bill", hue="smoker", data=tips,
               palette="muted", split=True, inner="quartile", scale="count")
plt.title("总账单按天和吸烟者状态分布")
plt.xlabel("星期")
plt.ylabel("总账单 ($)")
plt.legend(title="吸烟者")
plt.show()
```

> 比较按天和吸烟者状态划分的总账单分布的小提琴图。参数 `split=True` 将小提琴合并以便直接比较，`inner='quartile'` 显示四分位数线，而 `scale='count'` 则根据观测数量调整宽度。

此“拆分小提琴”图清楚地表明，例如，在周末（周六、周日），吸烟者和非吸烟者的分布形态和范围不同。`inner='quartile'` 选项用表示第25、50（中位数）和75百分位的线替换了迷你箱线图。`scale='count'` 在观测值较多的地方使小提琴更宽（例如，周四有更多非吸烟者）。

## 解读和使用小提琴图

观察小提琴图时，请注意：

1. **宽度**：较宽的部分表示密度更高（数据点更多）。
2. **形态**：它是对称的、偏斜的（左/右）、单峰的（一个峰值）还是多峰的（多个峰值）？
3. **内部标记 (token)**：找到中位数，并理解由 IQR 或四分位数线表示的离散度。
4. **比较（如果有多个小提琴）**：各类别之间的形态、中位数和离散度有何不同？

当您需要比较组间分布并且分布的形态具有信息量时，小提琴图是一个很好的选择。它们比箱线图提供更多细节，但如果类别或 `hue` 级别过多，则可能会变得视觉复杂。它们提供了一种令人信服的方式来清晰呈现分布差异。

## 参考资料

- [seaborn.violinplot](https://seaborn.pydata.org/generated/seaborn.violinplot.html) — Michael Waskom (2023)
  Seaborn `violinplot`函数的官方文档，详细说明其用法、参数和示例。
- [Violin Plots: A Box Plot-Density Trace Synergism](https://doi.org/10.2307/2685478) — Jerry L. Hintze, Ray D. Nelson (1998)
  Journal: The American Statistician; Publisher: American Statistical Association; Volume: 52; Pages: 181-184; DOI: [10.2307/2685478](https://doi.org/10.2307/2685478)
  介绍小提琴图，将其作为结合箱线图和密度追踪特点的统计图形，用于可视化数据分布。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://www.oreilly.com/library/view/python-data-science/9781491912126/) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  一本关于Python数据科学的指南，其中包含使用Matplotlib和Seaborn创建统计可视化图表（包括分布图）的章节。

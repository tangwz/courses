---
course: "data-visualization-matplotlib-seaborn"
chapter: "seaborn-plotting-categorical-data"
lesson: "seaborn-stripplot-swarmplot"
sourceId: 1203
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-6-seaborn-plotting-categorical-data/seaborn-stripplot-swarmplot"
title: "散点分布图与分群散点图：独立数据点的呈现 (stripplot, swarmplot)"
description: "使用 `stripplot` 和 `swarmplot` 可视化分类数据的独立数据点，避免重叠。"
order: 5
plots: ["plots/1203-0.json", "plots/1203-1.json", "plots/1203-2.json"]
sourceHash: "13ca5c2d0f312c73f96cb12ed1e06e284dd1b4aa7408c6562a984af0dd3c18a8"
sourceCorrections: []
---

尽管柱状图和箱线图为我们提供了每个类别有用的统计概括（如平均值或四分位数），但它们没有显示实际数据点。有时，查看每一个观测值对于了解每个类别内的分布、密集程度和可能的异常值很重要。Seaborn 为此目的提供了两个很棒的函数：`stripplot` 和 `swarmplot`。两者都创建散点图，其中一个轴代表一个分类变量。

### 散点分布图：针对分类的简单散点图

`stripplot` 函数是可视化按类别分组的独立数据点最直接的绘图方式。它本质上是绘制一个散点图，其中一个轴（通常是 x 轴）的位置对应类别，而另一个轴（通常是 y 轴）的位置对应数值。

假设我们有餐厅小费数据，并想要查看每周每天总账单金额的分布情况。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 加载示例小费数据集
tips = sns.load_dataset("tips")

# 绘制一个基本散点分布图
plt.figure(figsize=(8, 5)) # 设置图表大小以提高可读性
sns.stripplot(x="day", y="total_bill", data=tips)
plt.title("每日总账单金额 (散点分布图)")
plt.xlabel("星期")
plt.ylabel("总账单 ($)")
plt.show()
```



![散点分布图示例：按日划分的账单金额](plots/1203-0.json)



> 一个简化示例，使用散点分布图方法绘制了按不同日期划分的独立账单金额。

您会注意到代码示例的输出中（如果您使用完整数据集运行它），许多点直接重叠，特别是在某一天账单金额普遍相同的情况下。这种重叠会使我们难以分辨究竟有多少点位于某个特定值，或难以判断数据的密集程度。

#### 添加抖动

为了解决重叠问题，`stripplot` 有一个有用的参数 (parameter)：`jitter`。将 `jitter=True`（或指定抖动量）设置为 `True` 会在分类轴位置上添加少量随机扰动。这使得点在各自的分类列中水平散开，使独立标记 (token)点更易查看。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 加载示例小费数据集
tips = sns.load_dataset("tips")

# 绘制带抖动的散点分布图
plt.figure(figsize=(8, 5))
sns.stripplot(x="day", y="total_bill", data=tips, jitter=True, palette="Blues") # 添加了抖动和调色板
plt.title("每日总账单金额 (带抖动的散点分布图)")
plt.xlabel("星期")
plt.ylabel("总账单 ($)")
plt.show()
```



![抖动散点分布图示例：按日划分的账单金额](plots/1203-1.json)



> 应用抖动使点在每个类别中水平散开，减少了基本散点分布图中的重叠。

抖动显著提高中等大小数据集的可见性。您还可以将其他变量映射到视觉属性上，如 `hue`（颜色），以比较每个类别内的子组。

### 分群散点图：不重叠的点

虽然抖动有帮助，但随机噪声意味着精确的水平位置没有实际意义。Seaborn 提供 `swarmplot` 作为另一种选择，它智能地沿着分类轴定位点，以防止任何重叠。它将点排列成蜂群状，更清晰地显示分布密集程度。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 加载示例小费数据集
tips = sns.load_dataset("tips")

# 绘制分群散点图
plt.figure(figsize=(8, 5))
sns.swarmplot(x="day", y="total_bill", data=tips, palette="viridis", size=4) # 使用调色板并调整大小
plt.title("每日总账单金额 (分群散点图)")
plt.xlabel("星期")
plt.ylabel("总账单 ($)")
plt.show()
```



![分群散点图示例：按日划分的账单金额](plots/1203-2.json)



> 分群散点图排列点以避免重叠，使得每日数据点分布的密集程度视图更清晰，相比带抖动的散点分布图。注意：为更清晰地显示，此示例使用的点较少。

`swarmplot` 的主要优势是清晰呈现不同数值处的分布形态和密集程度。您可以轻松查看数据点集中的位置。

然而，用于放置不重叠点的算法可能计算量大。对于非常大的数据集（数千个点），生成分群散点图可能会变慢，图表本身可能变得过于密集而难以有效解读。

### 选择 `stripplot` 和 `swarmplot`

- **在以下情况使用 `stripplot`（带 `jitter=True`）：**
  - 您有极大量的数据点，并且 `swarmplot` 过慢或生成的图表过于密集。
  - 对分布和密集程度有个大致的了解，使用随机抖动即可满足要求。
- **在以下情况使用 `swarmplot`：**
  - 您有中小型数据集。
  - 查看精确的分布结构和不重叠的点密集程度很重要。
  - 计算时间不是主要限制。

`stripplot` 和 `swarmplot` 都接受类似的参数 (parameter)进行自定义，包括 `hue`（使用颜色添加另一个分类维度）、`palette`（控制颜色）和 `size`（调整标记 (token)大小）。它们是强大的工具，可以超越概括性统计，检查类别内的原始数据点。通常，它们可以与 `boxplot` 或 `violinplot` 等图表有效结合（通过在同一坐标轴上绘图），以同时提供概括性统计和独立数据点分布。

## 参考资料

- [Categorical data visualization](https://seaborn.pydata.org/tutorial/categorical.html) — Michael Waskom (2023)
  官方教程，提供 Seaborn 中分类数据可视化的全面指南，包括 stripplot 和 swarmplot。
- [seaborn.stripplot](https://seaborn.pydata.org/generated/seaborn.stripplot.html) — Michael Waskom (2024)
  官方 API 参考，记录 stripplot 函数、其参数和使用示例。
- [seaborn.swarmplot](https://seaborn.pydata.org/generated/seaborn.swarmplot.html) — Michael Waskom (2023)
  官方 API 参考，记录 swarmplot 函数、其参数和使用示例。
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media
  提供清晰有效数据可视化的原则，并讨论了如何表示单个数据点和管理密度。

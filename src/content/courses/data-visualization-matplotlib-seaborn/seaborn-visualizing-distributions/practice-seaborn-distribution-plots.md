---
course: "data-visualization-matplotlib-seaborn"
chapter: "seaborn-visualizing-distributions"
lesson: "practice-seaborn-distribution-plots"
sourceId: 1195
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-5-seaborn-visualizing-distributions/practice-seaborn-distribution-plots"
title: "动手实践：绘制数据分布图"
description: "动手实践，使用 Seaborn 创建直方图、KDE 图、箱线图和小提琴图。"
order: 6
plots: ["plots/1195-0.json", "plots/1195-1.json", "plots/1195-2.json", "plots/1195-3.json", "plots/1195-4.json"]
sourceHash: "4fa5675317e2d5dc15cdbc3884639c25038203ac7488b25d142a916d1e7399df"
sourceCorrections: []
---

几种 Seaborn 函数用于可视化数据分布。这些函数包括 `histplot`、`kdeplot`、`boxplot`、`violinplot` 和 `jointplot`，它们被应用到一个真实数据集上，以便更好地了解数据。

### 设置

首先，请确保导入了所需库。我们将使用 Seaborn 进行绘图，Matplotlib 用于可能进行的调整（比如控制图表大小），以及 Pandas 用于处理数据。我们将使用 Seaborn 内置的“tips”数据集。该数据集包含餐馆小费的信息。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 加载示例数据集
tips = sns.load_dataset("tips")

# 显示前几行以了解数据结构
print(tips.head())
```

输出结果会显示诸如 `total_bill`、`tip`、`sex`、`smoker`、`day`、`time` 和 `size` 等列。这些列包含数值和类别数据，非常适合查看数据分布情况。

### 可视化单变量分布：直方图和核密度估计图

我们先看看 `total_bill` 金额的分布。直方图是查看不同账单金额范围出现频率的好方法。

#### 直方图 (`histplot`)

我们可以使用 `sns.histplot()` 绘制 `total_bill` 列的直方图。

```python
plt.figure(figsize=(8, 5)) # 控制图表大小以便阅读
sns.histplot(data=tips, x="total_bill")
plt.title("总账单金额分布")
plt.xlabel("总账单金额 ($)")
plt.ylabel("频数")
plt.show()
```



![总账单金额分布](plots/1195-0.json)



> 总账单金额分布图。多数账单金额在10美元到20美元之间。

#### 带核密度估计（KDE）的直方图

有时，表示估计概率密度的平滑曲线很有用。我们可以通过设置 `kde=True` 添加这条曲线，或者使用 `sns.kdeplot()` 单独绘制。

```python
plt.figure(figsize=(8, 5))
sns.histplot(data=tips, x="total_bill", kde=True, color="#1098ad") # 使用 kde=True
plt.title("带KDE的总账单金额分布")
plt.xlabel("总账单金额 ($)")
plt.ylabel("频数 / 密度")
plt.show()

# 或者单独使用 kdeplot
plt.figure(figsize=(8, 5))
sns.kdeplot(data=tips, x="total_bill", color="#f76707", fill=True) # fill 参数会填充曲线下方的颜色
plt.title("总账单金额的KDE")
plt.xlabel("总账单金额 ($)")
plt.ylabel("密度")
plt.show()
```



![总账单金额的KDE](plots/1195-1.json)



> 总账单金额的平滑核密度估计（KDE），图上可见15-20美元左右的峰值以及右偏态。

与直方图的离散区间相比，KDE 能更平滑地显示分布的形态。

### 比较分布：箱线图和小提琴图

通常，我们希望比较一个数值变量在不同类别中的分布情况。让我们比较按周几划分的 `total_bill` 分布。

#### 箱线图 (`boxplot`)

箱线图使用四分位数来概括分布。箱体表示四分位距（IQR），内部的线是中位数，触须通常延伸到 IQR 的1.5倍。触须之外的点是可能的异常值。

```python
plt.figure(figsize=(8, 6))
sns.boxplot(data=tips, x="day", y="total_bill", palette="viridis") # 使用不同的调色板
plt.title("按日期划分的总账单金额分布")
plt.xlabel("星期几")
plt.ylabel("总账单金额 ($)")
plt.show()
```



![按日期划分的总账单金额分布](plots/1195-2.json)



> 按日期划分的总账单金额分布比较。周末（周六、周日）的账单中位数通常更高，范围也比工作日更广。

#### 小提琴图 (`violinplot`)

小提琴图结合了箱线图和核密度估计图，显示了分布的形状以及摘要统计数据。

```python
plt.figure(figsize=(8, 6))
sns.violinplot(data=tips, x="day", y="total_bill", palette="plasma", inner="quartile") # inner='quartile' 会在内部显示四分位数
plt.title("按日期划分的总账单金额分布（小提琴图）")
plt.xlabel("星期几")
plt.ylabel("总账单金额 ($)")
plt.show()
```



![按日期划分的总账单金额分布（小提琴图）](plots/1195-3.json)



> 按日期划分的总账单金额分布小提琴图。小提琴的形状显示了密度，确认了周末账单金额的分布更广，且高金额账单更集中。

小提琴图相比标准箱线图，能更好地显示分布的形状，如果存在多峰情况，它也能显示出来。

### 可视化联合分布

为了同时理解两个数值变量之间的关系及其各自的分布，我们使用 `sns.jointplot()`。让我们看看 `total_bill` 和 `tip` 的关系。

#### 联合图 (`jointplot`)

该函数会创建一个散点图，显示中心两个变量的关系，并在坐标轴旁显示每个变量的直方图（或核密度估计图）。

```python
sns.jointplot(data=tips, x="total_bill", y="tip", kind="scatter", color="#37b24d") # kind='scatter' 是默认设置
plt.suptitle("总账单金额与小费的联合分布", y=1.02) # 调整标题位置
plt.show()

# 我们也可以使用 kind='kde' 来显示密度等高线
sns.jointplot(data=tips, x="total_bill", y="tip", kind="kde", color="#7048e8")
plt.suptitle("总账单金额与小费的联合KDE", y=1.02)
plt.show()
```



![总账单金额与小费的联合分布](plots/1195-4.json)



> 显示总账单金额和小费金额关系的联合图。散点图显示了正相关关系，而边缘的直方图则显示了每个变量的单独分布。

本实践练习说明了 Seaborn 的分布图如何提供有效的手段来查看和比较数据分布，无论是单变量分布还是变量之间的关系。建议将这些图表应用到 `tips` 数据集中的其他数值和类别列，或您自己的数据上，进行更多尝试。

## 参考资料

- [User Guide and Tutorial: Visualizing distributions of observations](https://seaborn.pydata.org/tutorial/distributions.html) — Seaborn Development Team (2024)
  涵盖Seaborn用于可视化数据分布的函数官方文档，包括直方图、KDE、箱线图和提琴图。
- [Pyplot tutorial](https://matplotlib.org/stable/tutorials/pyplot.html) — Matplotlib Development Team (2024)
  Matplotlib官方教程，演示如何使用`pyplot`进行通用绘图自定义，包括设置图表大小、标题和标签。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本使用Python进行数据操作和分析的指南，包含使用Pandas、Matplotlib和Seaborn进行数据可视化的实践示例。
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media
  提供设计清晰、信息丰富的可视化图表的原则和实践，为直方图和箱线图等图表类型提供概念指导。

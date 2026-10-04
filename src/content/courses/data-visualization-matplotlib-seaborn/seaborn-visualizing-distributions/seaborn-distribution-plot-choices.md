---
course: "data-visualization-matplotlib-seaborn"
chapter: "seaborn-visualizing-distributions"
lesson: "seaborn-distribution-plot-choices"
sourceId: 1194
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-5-seaborn-visualizing-distributions/seaborn-distribution-plot-choices"
title: "理解分布图选择"
description: "关于如何选择合适的Seaborn图表类型来可视化数据分布的建议。"
order: 5
plots: []
sourceHash: "c284831a5c05475e532a82d3760b114584c712effbf1173ba578240a62b7f677"
sourceCorrections: []
---

有几种强大的Seaborn函数可用于可视化数据分布：`histplot`、`kdeplot`、`boxplot`、`violinplot`和`jointplot`。每种函数都对数据集中的潜在规律提供了不同的视角。但是，您如何决定在特定任务中使用哪一种呢？选择合适的图表对于有效传递信息十分重要。接下来，我们来分析每种图表的优点和常见用途。

### 图表选择的指导要点

思考您最希望了解或表达分布的哪一方面：

1. **原始频率与大致形状：** 如果您需要查看落在特定范围（箱）内的数据点数量，并对分布的大致形状有一个了解，\*\*直方图（`histplot`）\*\*是您的初步选择。它直接且易于理解。不过请记住，图表外观会因您选择的箱数量和宽度而变化。
2. **平滑形状与概率密度：** 为了获得更平滑的分布表示，通常被视为对潜在概率密度函数（PDF）的估计，请使用**核密度估计（`kdeplot`）**。KDE图有助于可视化整体形状，避免直方图的锯齿状，并且特别适用于在同一坐标轴上比较多个分布的形状。请注意，它们有时可能会平滑掉一些具体细节，或在数据稀疏的区域显示出密度。
3. **跨类别分布比较（汇总统计量）：** 当您的目标是比较数值变量在不同组或类别之间的集中趋势（中位数）和离散程度（四分位数范围）时，\*\*箱线图（`boxplot`）\*\*非常有效。它清楚地显示中位数、四分位数和潜在的异常值，使比较变得直接。然而，箱线图会隐藏箱体内部分布的具体形状。两个形状不同的分布可能具有非常相似的箱线图。
4. **跨类别分布比较（形状与汇总）：** 如果您希望获得箱线图的比较能力，同时*也*想了解每个类别分布的形状，\*\*小提琴图（`violinplot`）\*\*是一个很好的选择。它实质上结合了箱线图（通常显示在小提琴内部）和两侧镜像的KDE图。这提供了比单独箱线图更丰富的比较，但如果类别太多，可能会变得视觉混乱。
5. **两个变量及其分布之间的关联：** 当您想了解两个数值变量之间如何关联，并且*也*想单独查看每个变量的分布时，请使用\*\*`jointplot`\*\*。此函数会创建一个中心图（通常是散点图或六边形箱体图）来显示关联，同时在边缘显示每个变量的直方图或KDE图。它专门用于结合单变量（one-variable）分布视图的双变量（two-variable）分析。

### 您需要考虑的问题

在创建图表之前，请思考以下问题：

- **我正在可视化多少个变量？**
  - 一个数值变量：`histplot`、`kdeplot`。
  - 一个按类别划分的数值变量：`boxplot`、`violinplot`、多个`kdeplot`。
  - 两个数值变量：`jointplot`（用于关联 + 单独分布）、`scatterplot`或专门的双变量图表。
- **主要目的是什么？**
  - 查看箱体中的精确计数：`histplot`。
  - 查看平滑的形状估计：`kdeplot`。
  - 比较组间的汇总统计量（中位数、四分位数）：`boxplot`。
  - 比较组间的形状和汇总：`violinplot`。
  - 查看两个变量之间的关联及其分布：`jointplot`。
- **受众是谁？**
  - 箱线图是广为人知的汇总图。
  - 小提琴图提供更多细节，但可能需要向某些受众进行更多解释。
  - 直方图易于理解，但依赖于箱体选择。
  - KDE图有效地显示形状，但表示的是密度估计，而非原始计数。

### 汇总表

| 图表类型 | 主要用途 | 优点 | 缺点 |
| --- | --- | --- | --- |
| `histplot` | 显示箱体内的频率计数，大致形状 | 简单，易懂，显示原始计数 | 对箱体大小敏感，可能出现锯齿状 |
| `kdeplot` | 显示分布形状的平滑估计（PDF） | 平滑，适合比较形状，对箱体大小不那么敏感 | 可能隐藏细节，在数据稀疏处可能暗示密度 |
| `boxplot` | 比较跨类别的汇总统计量 | 中位数/四分位数比较清晰，识别异常值 | 隐藏四分位数内的分布形状 |
| `violinplot` | 比较跨类别的分布（形状与汇总） | 显示形状（KDE）和汇总（箱线图） | 类别过多时可能视觉复杂 |
| `jointplot` | 显示关联与单独分布（2个变量） | 结合双变量关联与单变量分布 | 仅限于两个数值变量 |

选择合适的分布图需要了解每种图表强调的内容。通常，您可能会从直方图或KDE图开始进行初步查看，然后根据汇总统计量或完整形状对您的分析是否更具意义，再转向箱线图或小提琴图进行组间比较。不妨尝试不止一种类型，看看哪种图表最有效地讲述您数据中的情况。

## 参考资料

- [Visualizing distributions](https://seaborn.pydata.org/tutorial/distributions.html) — Michael Waskom and the Seaborn development team (2024)
  官方教程，通过详细的使用说明和示例解释了Seaborn的分布绘图函数。
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media
  一本内容全面的书籍，提供了有效数据可视化的原则，包括选择适合不同数据分布的图表类型的指导。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本关于Python中数据处理和可视化的实用指南，其中包含介绍Matplotlib和Seaborn用于绘制分布的章节。

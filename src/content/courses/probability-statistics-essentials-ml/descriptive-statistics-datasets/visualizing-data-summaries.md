---
course: "probability-statistics-essentials-ml"
chapter: "descriptive-statistics-datasets"
lesson: "visualizing-data-summaries"
sourceId: 1405
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-3-descriptive-statistics-datasets/visualizing-data-summaries"
title: "数据总结的可视化"
description: "使用直方图、箱线图和散点图来显示描述性统计数据。"
order: 7
plots: ["plots/1405-0.json", "plots/1405-1.json", "plots/1405-2.json"]
sourceHash: "9f91a198098a942f6751b05afa1d72da0c664d0413e1d61909da32416401936a"
sourceCorrections: []
---

虽然计算平均值、中位数、方差和相关性等数值概括可以得到准确的数值，但数据可视化通常能让人更直接、更直观地认识其特点。图表可以有效传达数据的分布、离散程度、集中趋势和关系，作为我们已介绍过的描述性统计数据的补充。下面我们来看看一些常见且有效的可视化方法。

### 直方图：显示频率分布

直方图是理解单个数值变量分布的根本。它们将数据分组到箱（区间）中，并以条形显示落入每个箱的观测值的频率或计数。

- **集中趋势：** 峰值的位置可以提示众数，而整体的平衡点暗示着平均值和中位数。
- **离散程度：** 直方图的宽度表明离散程度或变异性。窄的直方图表明方差小，而宽的则表明方差大。
- **形状：** 直方图清楚显示分布的形状。您可以视觉评估它是否对称（如正态分布）、偏斜（左偏或右偏）或多峰（有多个峰值）。这直接与偏度和峰度的想法相关联。

考虑一个客户年龄数据集。直方图可以快速显示大多数客户是否属于特定年龄段，年龄分布是偏向年轻客户还是年长客户，以及年龄的离散程度。



![客户年龄分布](plots/1405-0.json)



> 直方图显示按箱分组的客户年龄频率分布。

### 箱线图：概括百分位数

箱线图（或箱须图）根据其五数概括提供分布的紧凑视觉概括：最小值、第一四分位数（Q1，第25百分位数）、中位数（Q2，第50百分位数）、第三四分位数（Q3，第75百分位数）和最大值。

- **箱体：** 表示四分位距（IQR），从Q1延伸到Q3。箱体的长度表明数据中间50%的离散程度。
- **中位线：** 箱体内的线标记 (token)中位数（Q2），显示数据的中心点。
- **须线：** 线从箱体延伸，通常延伸到距离Q1和Q3分别为1.5倍IQR范围内的最小值和最大值。它们展现数据的整体范围，不包括异常值。
- **异常值：** 落在须线之外的点通常单独绘制，有助于识别不寻常的观测值。

箱线图非常适合比较不同组间的分布，并快速掌握集中趋势（中位数）、离散程度（IQR、须线范围）和对称性（比较须线的长度和中位数在箱体内的位置）。



![按学习小组划分的考试分数](plots/1405-1.json)



> 箱线图比较两个不同学习小组的考试分数分布。

### 散点图：分析关系

当您想理解两个数值变量之间的关系时，散点图是标准选择。图表上的每个点代表一对数值，每个变量一个。

- **相关性：** 散点图视觉显示先前计算的相关性。
  - 从左到右向上趋势的点表明正相关。
  - 向下趋势的点表明负相关。
  - 随机分散的点表明很少或没有线性相关性。
- **强度：** 点围绕潜在趋势线的紧密程度表明线性关系的强度。紧密聚集的点意味着比广泛分散的点更强的相关性。
- **模式：** 散点图也可以显示非线性关系、簇或异常值，这些仅从相关系数中可能不明显。

请记住相关性和因果关系之间的区别。散点图可能显示网站访问量和销售额之间存在强关联，但这并不能证明访问量*导致*销售额。



![学习小时数与考试分数](plots/1405-2.json)



> 散点图显示每个学生的学习小时数与考试分数之间的关系。

### 选择合适的图表

- 使用**直方图**或**密度图**（直方图的平滑版本）来理解*单个*数值变量的形状、中心和离散程度。
- 使用**箱线图**简洁地概括分布，比较组间的分布，并识别*单个*数值变量（通常按分类变量分组）的潜在异常值。
- 使用**散点图**检查*两个*数值变量之间的关系和潜在相关性。

Matplotlib、Seaborn 等库以及 Pandas 内置的绘图函数提供了在 Python 中高效创建这些图表的工具。这些视觉辅助是数值描述性统计数据不可或缺的补充，为数据的结构和模式提供更丰富的理解。

## 参考资料

- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media
  涵盖有效数据可视化的原理和技术，包括对本节中讨论的常见图表类型的详细说明。
- [An Introduction to Statistical Learning with Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2021)
  Publisher: Springer
  为理解数据提供了统计学基础，包括可视化变量分布和关系的方法。
- [Seaborn: Statistical Data Visualization](https://seaborn.pydata.org/index.html) — Michael Waskom and the Seaborn development team (2024)
  Python 库的官方文档，提供了创建直方图、箱线图和散点图等统计图表的指南和示例。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本关于 Python 数据处理和分析的实用指南，包括使用 Matplotlib 和 Seaborn 创建各种数据可视化图表的部分。

---
course: "intro-eda-course"
chapter: "bivariate-analysis"
lesson: "numerical-vs-categorical-plots"
sourceId: 1171
sourceUrl: "https://apxml.com/zh/courses/intro-eda-course/chapter-4-bivariate-analysis/numerical-vs-categorical-plots"
title: "数值型与分类型：比较图表"
description: "利用分组箱线图或提琴图等图表，比较不同类别中数值变量的分布。"
order: 4
plots: ["plots/1171-0.json", "plots/1171-1.json"]
sourceHash: "288dac1b06a2050e021baf3f04fa4fab63d9f98f62d1b092831a125d4044419f"
sourceCorrections: []
---

在分析了单个变量以及两个数值型或两个分类型变量之间的关系后，一项常见的分析任务是了解数值型测量值如何在不同组或类别之间变化。例如，不同职位（分类型）的平均工资（数值型）有何不同？或者客户消费（数值型）如何根据会员等级（分类型）而变化？这种分析有助于找出各组在分布、集中趋势和变异性方面的显著差异。

可视化对于这种比较特别有效。尽管计算每组的均值或中位数等汇总统计量能提供量化 (quantization)概览，但图表能更丰富、直观地呈现潜在的分布情况。我们将重点介绍两种比较数值变量在不同类别间分布的有力可视化方法：分组箱线图和提琴图。

### 分组箱线图

箱线图常用于概括单个数值变量的分布。它们显示中位数、四分位数（四分位距，IQR）和潜在异常值。通过将多个箱线图并排显示，每个图对应一个分类变量的一个类别，可以直接比较各组的这些汇总统计量。

使用Seaborn创建分组箱线图很简单。你可以将分类变量指定给x轴（或y轴），将数值变量指定给另一个轴。

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 示例数据生成（请替换为您的实际数据）
np.random.seed(42)
data = {
    'Category': np.random.choice(['Alpha', 'Beta', 'Gamma'], size=150),
    'Value': np.concatenate([
        np.random.normal(50, 15, 50), # Alpha类别
        np.random.normal(65, 10, 50), # Beta类别
        np.random.normal(55, 20, 50)  # Gamma类别
    ])
}
df = pd.DataFrame(data)

# 创建分组箱线图
plt.figure(figsize=(8, 5)) # 设置图表大小以提高可读性
sns.boxplot(x='Category', y='Value', data=df, palette=['#74c0fc', '#ffa94d', '#69db7c']) # 使用蓝色、橙色、绿色

# 添加图表增强
plt.title('数值在不同类别上的分布')
plt.xlabel('类别类型')
plt.ylabel('测量值')
plt.grid(axis='y', linestyle='--', alpha=0.7) # 添加水平网格线

plt.show()
```



![数值在不同类别上的分布](plots/1171-0.json)



> 一个分组箱线图，比较了数值型“值”在三种“类别”类型（Alpha、Beta 和 Gamma）上的分布。Beta 似乎比 Alpha 和 Gamma 具有更高的中位数和更小的变异性（较小的 IQR）。Gamma 显示出最大的分散程度，并且中位数略高于 Alpha。

解读分组箱线图时，请注意：

- **中位数差异：** 比较箱体内的中心线。各组的典型值是否有显著差异？
- **分散程度差异（IQR）：** 比较箱体的高度（IQR）。数值变量在不同类别间的变异性是否有显著不同？
- **箱体重叠：** 显著重叠可能表示组间差异不那么明显，而重叠较少则表示分布不同。
- **异常值：** 观察每个类别中异常值（触须外的点）的存在和位置。某些组是否有更多极端值？

### 提琴图

提琴图是一种替代方法，通常能提供更多信息，用于比较不同类别间的分布。它们将箱线图的汇总统计量与两侧绘制的核密度估计（KDE）结合起来。这提供了对分布“形状”的洞察，展现了箱线图可能掩盖的多峰性（多个峰值）等特征。

提琴在任何给定值处的宽度表示数据点在该值附近的密度。内部可选地显示一个迷你箱线图、四分位数或仅显示中位数点。

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 复用箱线图示例中的样本数据
np.random.seed(42)
data = {
    'Category': np.random.choice(['Alpha', 'Beta', 'Gamma'], size=150),
    'Value': np.concatenate([
        np.random.normal(50, 15, 50), # Alpha类别
        np.random.normal(65, 10, 50), # Beta类别
        np.random.normal(55, 20, 50)  # Gamma类别
    ])
}
df = pd.DataFrame(data)

# 创建分组提琴图
plt.figure(figsize=(8, 5))
sns.violinplot(x='Category', y='Value', data=df, palette=['#74c0fc', '#ffa94d', '#69db7c'], inner='quartile') # 内部显示四分位数

# 添加图表增强
plt.title('数值在不同类别上的分布形状')
plt.xlabel('类别类型')
plt.ylabel('测量值')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
```



![数值在不同类别上的分布形状](plots/1171-1.json)



> 一个分组提琴图，显示了 Alpha、Beta 和 Gamma 类别中“值”的分布形状。该图确认了 Beta 具有更高的集中趋势和更紧密的分布。形状也表明 Gamma 可能比 Alpha 分布更分散一些，即使它们的中位数相对接近。所有分布看起来大致是单峰的。

解读提琴图时，请考虑：

- **形状：** 观察每个提琴的整体形状。它是对称的、偏斜的、平坦的（均匀的）还是多峰的？比较不同类别间的形状。
- **宽度：** 最宽的部分表示数据点最集中的地方。
- **结合信息：** 利用密度形状以及内部箱线图（如果显示）来同时获得对集中趋势、分散程度和分布形状的全面视图。

### 箱线图与提琴图的选择

- **箱线图** 非常适合快速比较多个组的汇总统计量（中位数、四分位数、异常值）。它们通常比提琴图更简洁，尤其是在类别众多时。
- **提琴图** 提供了关于分布形状的更多细节，包括密度和潜在的多峰性。当了解每个类别内分布的“方式”与汇总统计量同等重要时，它们通常更受青睐。然而，当类别过多或分布复杂时，它们可能会变得视觉上复杂。

这两种图表都是有价值的工具，用于观察数值变量和分类变量之间的关系。通过可视化这些比较，你可以更透彻地理解不同组在定量测量方面的表现，识别出仅查看汇总数据可能遗漏的模式。这种理解通常是进行更正式的统计检验或特征工程的起点。

## 参考资料

- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media
  数据可视化的全面指南，涵盖原则、常见图表类型以及通过图形进行有效沟通的方法，与比较图表直接相关。
- [seaborn.boxplot](https://seaborn.pydata.org/generated/seaborn.boxplot.html) — Michael Waskom and the Seaborn development team (2024)
  Seaborn箱线图函数的官方文档，包含参数、示例和可视化分布的详细说明。
- [seaborn.violinplot](https://seaborn.pydata.org/generated/seaborn.violinplot.html) — Michael Waskom and the Seaborn development team (2024)
  Seaborn小提琴图函数的官方文档，提供了其在显示分布形状和汇总统计信息方面的使用见解。
- [Box Plots in Python](https://plotly.com/python/box-plots/) — Plotly Technologies Inc. (2024)
  Plotly官方文档，展示如何使用Python中的Plotly创建箱线图，并提供跨类别比较数值数据的示例。
- [Violin Plots: A Box Plot-Density Trace Synergism](http://dx.doi.org/10.2307/2685478) — Jerry L. Hintze and Ray D. Nelson (1998)
  Journal: The American Statistician; Publisher: American Statistical Association; Volume: 52; Pages: 181-184; DOI: [10.2307/2685478](https://doi.org/10.2307/2685478)
  介绍小提琴图的原始学术论文，强调其结合箱线图摘要和核密度估计视觉细节的能力。

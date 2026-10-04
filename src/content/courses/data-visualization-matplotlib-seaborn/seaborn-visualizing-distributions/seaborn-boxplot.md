---
course: "data-visualization-matplotlib-seaborn"
chapter: "seaborn-visualizing-distributions"
lesson: "seaborn-boxplot"
sourceId: 1192
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-5-seaborn-visualizing-distributions/seaborn-boxplot"
title: "用于汇总统计的箱线图 (boxplot)"
description: "学习使用 Seaborn 的箱线图通过四分位数可视化数据分布。"
order: 3
plots: ["plots/1192-0.json"]
sourceHash: "7bbb89e133843d19b358b4d1eb7a605a97b72dedf1380871e8d81d5efe702bb5"
sourceCorrections: []
---

箱线图，也称为箱须图，提供了一种简洁的数据分布可视化概括方式。当需要比较多个分布或仅需快速概览时，这些图表尤为有用。虽然直方图和核密度估计（KDEs）等更详细的可视化方法能展示分布的完整形态，但箱线图提供了一种标准化地展现基于五数概括的数据的方式。

### 了解箱线图的构成

箱线图直观地表示几个基本的描述性统计量：

1. **中位数 (Q2):** 箱体内部的线表示中位数（即第50百分位数），它代表数据的中点。一半的数据点小于中位数，另一半则大于中位数。
2. **箱体:** 这个中心矩形从第一四分位数 (Q1) 延伸到第三四分位数 (Q3)。
   - **第一四分位数 (Q1):** 第25百分位数。25% 的数据点低于这个值。
   - **第三四分位数 (Q3):** 第75百分位数。75% 的数据点低于这个值（这意味着25% 的数据点高于它）。
3. **四分位距 (IQR):** Q1与Q3之间的距离 ($IQR = Q3 - Q1$)。箱体本身代表数据的中间50%。更长的箱体表示数据中心部分的变异性更大。
4. **触须:** 从箱体延伸出的线表示主体数据的范围。默认情况下，在Seaborn（以及统计学中常见）里，触须延伸到距离箱体边缘（Q1 - 1.5 \* IQR 和 Q3 + 1.5 \* IQR）1.5倍IQR范围内的最远数据点。超出此范围的数据点被视为潜在的异常值。
5. **异常值:** 绘制在触须之外的单个点。这些是根据1.5 \* IQR规则，相对于其余数据而言异常高或异常低的数据点。

箱线图提供了一种简洁的方式来把握数据的集中趋势（中位数）、离散程度（IQR），并识别潜在的异常值。

### 使用 `seaborn.boxplot` 创建箱线图

Seaborn 使用 `seaborn.boxplot` 函数创建箱线图非常简单直接。它与 Pandas DataFrames 配合得特别好。

我们假设已经载入了常用的 'tips' 数据集：

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 载入示例数据集
tips = sns.load_dataset("tips")

# 显示前几行数据
print(tips.head())
```

**绘制单个分布**

要绘制单个数值变量（例如 `total_bill`）的分布，您可以直接传入 DataFrame 列：

```python
# 为 'total_bill' 列创建箱线图
plt.figure(figsize=(6, 4)) # 可选：调整图表大小
sns.boxplot(y=tips["total_bill"]) # 使用 y 轴绘制垂直箱线图
plt.title("总账单金额分布")
plt.ylabel("总账单金额 ($)")
plt.show()
```

这段代码生成一个垂直箱线图，展示数据集中所有总账单金额的中位数、四分位数、触须和异常值。

**比较不同类别之间的分布**

箱线图的一个重要优点是能够比较不同组之间的分布。您通常通过为一个轴（`x` 或 `y`）指定类别变量，为另一个轴指定数值变量来实现这一点。

我们来比较一周中每天的 `total_bill` 分布：

```python
# 创建箱线图，比较不同 'day' 值下的 'total_bill'
plt.figure(figsize=(8, 5)) # 可选：调整图表大小
sns.boxplot(x="day", y="total_bill", data=tips, palette="blue") # x 轴用于类别，y 轴用于数值
plt.title("按日期划分的总账单金额分布")
plt.xlabel("星期")
plt.ylabel("总账单金额 ($)")
plt.show()
```

这里，`x="day"` 指示 Seaborn 为 'day' 列中的每个独特值创建单独的箱线图，使用 `y="total_bill"` 指定的相应 `total_bill` 值。`data=tips` 参数 (parameter)提供 DataFrame。我们还使用了 `palette` 参数来应用预设的颜色方案。

您可以使用 `hue` 参数进一步细分数据，在每个主要类别内进行嵌套比较（例如，比较每天的吸烟者和非吸烟者）。

```python
# 创建嵌套箱线图，比较按 'day' 和 'smoker' 状态划分的 'total_bill'
plt.figure(figsize=(10, 6))
sns.boxplot(x="day", y="total_bill", hue="smoker", data=tips, palette="pastel")
plt.title("按日期和吸烟者状态划分的总账单金额分布")
plt.xlabel("星期")
plt.ylabel("总账单金额 ($)")
plt.legend(title="吸烟者") # 添加图例标题
plt.show()
```

### 解读箱线图结果

在查看单个箱线图或比较多个箱线图时：

- **集中趋势:** 比较中位数线（箱体内部的线）。更高的中位数表示该组数据的中心值更高。
- **离散程度/变异性:** 比较箱体的长度（即IQR）。更长的箱体意味着数据的中间50%更分散。同时，也要比较触须的整体长度。
- **偏度:** 如果中位数更接近Q1（箱体底部）且上触须更长，则分布可能呈右偏（正偏）。如果中位数更接近Q3（箱体顶部）且下触须更长，则可能呈左偏（负偏）。
- **异常值:** 留意触须之外单个点的存在和数量。这些点可能需要进一步查看。

### 视觉示例：箱线图组成部分



![箱线图组成部分示意](plots/1192-0.json)



> 一个箱线图示例，展示了中位数（中心线）、箱体（Q1到Q3）、触须（延伸至1.5\*IQR）和一个异常值点。

### 何时使用箱线图

当您希望时，箱线图特别有效：

- 快速概括一个或多个分布的主要统计属性。
- 比较数值变量在不同类别间的分布。
- 识别数据中的潜在异常值。

与直方图或KDE图相比，它们在分布的具体形态方面提供的细节较少（例如，您无法从标准箱线图轻易看出分布是否呈双峰）。接下来讨论的小提琴图尝试结合箱线图的概括性特征和KDE图的形态信息。

总之，`seaborn.boxplot` 提供了一种强大且简洁的方法，用于可视化汇总统计量和比较分布，使其成为您数据检查工具集中的一个重要工具。

## 参考资料

- [\`seaborn.boxplot\` - seaborn 0.13.2 documentation](https://seaborn.pydata.org/generated/seaborn.boxplot.html) — Michael Waskom and Seaborn Developers (2024)
  提供Seaborn创建箱线图的官方API参考和示例，包含详细参数说明。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  一本全面介绍Python数据科学核心库（如Matplotlib、NumPy、Pandas和Seaborn）的指南，包含统计绘图部分。
- [Matplotlib Documentation](https://matplotlib.org/stable/index.html) — John Hunter, Michael Droettboom, and The Matplotlib Development Team (2024)
  Publisher: Matplotlib Development Team
  提供Matplotlib的通用文档，它是Seaborn使用的底层绘图库，对高级定制有用。

# 汇总统计条形图 (barplot)

来源：[原文](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-6-seaborn-plotting-categorical-data/seaborn-barplot)

[返回章节目录](README.md) · [返回课程目录](../README.md)

分析分类数据时，一个常见任务是比较不同类别间的数值度量。例如，您可能需要比较不同产品类型的平均销售额，或各种教学方法的平均考试分数。Seaborn 的 `barplot` 函数专为此目的而设计。它为各类别内的数值变量计算汇总统计量（如均值），并使用矩形条形进行显示。

### 使用 `seaborn.barplot` 显示集中趋势

`seaborn.barplot()` 函数绘制条形图以表示估计值。其主要作用是计算并绘制定量变量的集中趋势度量（默认是均值），并按一个或多个分类变量的层级进行分组。重要的一点是，它还会使用误差条显示该估计值的不确定性。

让我们看看如何使用它。基本语法涉及为一个轴（通常是 `x` 轴）指定分类变量，为另一个轴（通常是 `y` 轴）指定数值变量，并使用 `data` 参数 (parameter)指定包含数据的 DataFrame。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 加载示例数据集（Seaborn 自带）
tips = sns.load_dataset("tips")

# 创建一个条形图，显示每天的平均总账单
plt.figure(figsize=(8, 5)) # 调整图形大小以提高可读性
sns.barplot(x="day", y="total_bill", data=tips)
plt.title("每天的平均总账单")
plt.xlabel("星期几")
plt.ylabel("平均总账单 ($)")
plt.show()
```

> 生成的图显示了每天的平均“总账单”。每个条形的高度代表计算出的均值。

### 理解估计器

默认情况下，`barplot` 计算各类别数值变量的均值。但是，您可能对不同的汇总统计量感兴趣，例如中位数，它对异常值不那么敏感。您可以使用 `estimator` 参数 (parameter)来控制这一点。此参数接受一个计算所需统计量的函数。常见的选择包括 `numpy.mean`（默认）、`numpy.median`、`numpy.std`（标准差）等。

让我们改为绘制每天的*中位数*总账单：

```python
# 创建一个条形图，显示每天的中位数总账单
plt.figure(figsize=(8, 5))
sns.barplot(x="day", y="total_bill", data=tips, estimator=np.median)
plt.title("每天的中位数总账单")
plt.xlabel("星期几")
plt.ylabel("中位数总账单 ($)")
plt.show()
```

将此图与前一个图比较可能会显示差异，特别是在给定某一天内，如果 `total_bill` 的分布是偏斜的。

### 理解误差条

您会注意到上面图中条形顶部的垂直线。这些是误差条，它们提供了有关计算出的汇总统计量的不确定性或变异性的重要信息。默认情况下，`barplot` 显示均值的 95% 置信区间。

**置信区间**给出了一个估计值的范围，该范围可能包含真实的总体均值。较小的误差条表明变异性较小，并且对点估计（条形高度）的置信度更高，而较大的误差条则表示变异性更大或数据更少，使得估计值不那么精确。比较不同类别间的误差条很重要；如果两个类别的误差条显著重叠，这表明它们均值之间的差异可能不具有统计显著性。

误差条的计算方法和大小可以使用 `errorbar` 参数 (parameter)（它取代了旧的 `ci` 参数）进行控制。常见选项包括：

- `('ci', 95)`: 显示 95% 置信区间（默认）。您可以更改百分比（例如 99）。
- `'sd'`: 显示各类别内数据的标准差。
- `None`: 不绘制误差条。

下面展示如何绘制带有表示标准差的误差条的平均总账单：

```python
# 创建一个带有标准差误差条的条形图
plt.figure(figsize=(8, 5))
sns.barplot(x="day", y="total_bill", data=tips, errorbar='sd') # 使用标准差
plt.title("每天的平均总账单（带标准差）")
plt.xlabel("星期几")
plt.ylabel("平均总账单 ($)")
plt.show()
```

> 注意误差条现在代表了每天账单的离散程度（标准差），而不是均值估计的置信度。

### 水平条形图

有时，特别是当您有许多类别或类别名称很长时，水平条形图更易读。您可以通过将分类变量分配给 `y` 轴，将数值变量分配给 `x` 轴来轻松创建此类图。Seaborn 会自动推断方向。

```python
# 创建一个水平条形图
plt.figure(figsize=(7, 5))
sns.barplot(x="total_bill", y="day", data=tips, estimator=np.mean, errorbar=('ci', 95))
plt.title("每天的平均总账单")
plt.xlabel("平均总账单 ($)")
plt.ylabel("星期几")
plt.show()
```

### 按第二个分类变量分组

`barplot` 还可以显示按*两个*分类变量细分的汇总数据。您可以使用 `hue` 参数 (parameter)来实现这一点。这会创建分组条形图，其中 `hue` 变量不同层级的条形在主轴的每个类别内并排显示。

让我们看看每天的平均总账单，并根据顾客是否吸烟进一步细分：

```python
# 创建一个分组条形图
plt.figure(figsize=(10, 6))
sns.barplot(x="day", y="total_bill", hue="smoker", data=tips)
plt.title("按吸烟状态划分的每天平均总账单")
plt.xlabel("星期几")
plt.ylabel("平均总账单 ($)")
plt.show()
```



[交互图表：每天的平均总账单](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-6-seaborn-plotting-categorical-data/seaborn-barplot#plot-1hgzzg8)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "bar",
      "x": [
        "Thur",
        "Fri",
        "Sat",
        "Sun"
      ],
      "y": [
        17.68,
        17.15,
        20.44,
        21.41
      ],
      "name": "均值",
      "marker": {
        "color": "#228be6"
      },
      "error_y": {
        "type": "data",
        "array": [
          1.35,
          1.45,
          1.04,
          1.24
        ],
        "visible": true
      }
    }
  ],
  "layout": {
    "title": "每天的平均总账单",
    "xaxis": {
      "title": "星期几"
    },
    "yaxis": {
      "title": "平均总账单 ($)"
    },
    "height": 400,
    "width": 600,
    "showlegend": false
  }
}
```

</details>



> 示例显示了每天的平均总账单，误差条表示均值的 95% 置信区间。

总之，`seaborn.barplot` 是一个功能强大的工具，用于比较不同类别中数值变量的汇总度量（如均值或中位数）。误差条的加入提供了有关这些估计值的不确定性或变异性的有益信息，从而实现更明智的比较。请记住，`barplot` 显示的是汇总统计量，这与 `countplot` 有所不同，`countplot` 仅显示每个类别中的观察次数（接下来会介绍）。

## 参考资料

- [seaborn.barplot](https://seaborn.pydata.org/generated/seaborn.barplot.html) — Michael Waskom (2023)
  `seaborn.barplot` 函数的官方文档，详细介绍了其参数、用法以及用于聚合统计和误差线的各种示例。
- [Plotting categorical data](https://seaborn.pydata.org/tutorial/categorical.html) — Michael Waskom (2023)
  Publisher: Seaborn Project
  Seaborn 官方教程，展示了如何可视化分类变量和定量变量之间的关系，包括 `barplot` 及其变体的使用。
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media
  一份关于数据可视化原则的全面指南，从统计学角度讲解如何有效地表示数据以及如何解释比较图中误差线等元素。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  一本关于 Python 数据科学的基础教材，其中有专门的章节介绍 Matplotlib 和 Seaborn 用于创建信息丰富的统计图表。

---

[上一节](01-%E7%90%86%E8%A7%A3%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE.md) · [下一节](03-%E9%A2%91%E7%8E%87%E8%AE%A1%E6%95%B0%E5%9B%BE%20%28countplot%29.md)

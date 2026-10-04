# 直方图和核密度估计 (histplot, kdeplot)

来源：[原文](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-5-seaborn-visualizing-distributions/seaborn-histplot-kdeplot)

[返回章节目录](README.md) · [返回课程目录](../README.md)

理解数据形态是分析工作的基础步骤。直方图和核密度估计（KDE）是Seaborn提供的两种有效工具，用于展示单个数值变量的分布情况。它们有助于解答以下问题：数据集中在哪里？分布是对称的还是偏斜的？是否存在多个峰值？

### 直方图与 `histplot`

直方图可能是可视化分布最常见的方式。它将数据范围划分为一系列区间（称为“bin”），然后统计落入每个bin的数据点数量。直方图中条形的高度表示每个bin内数据点的频率（或计数）。

Seaborn 的 `histplot` 函数使得创建直方图非常直接。它以 Matplotlib 的 `hist` 函数为基础，但与 Pandas DataFrame 的结合更好，并提供更多高级选项，包括自动确定合适的bin数量。

让我们生成一些样本数据并创建一个基本直方图：

```python
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 生成一些服从正态分布的数据
np.random.seed(0) # 用于结果的可复现性
data = np.random.randn(200)

# 使用 Seaborn 的 histplot 创建直方图
sns.histplot(data=data)
plt.title('样本数据的直方图')
plt.xlabel('值')
plt.ylabel('频率')
plt.show()
```

这段代码生成200个服从标准正态分布的随机数，然后使用 `histplot` 将它们绘制出来。Seaborn 会自动选择合适的bin数量来表示数据结构。



[交互图表：使用默认 bin 的直方图](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-5-seaborn-visualizing-distributions/seaborn-histplot-kdeplot#plot-1r5z8bz)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "使用默认 bin 的直方图",
    "xaxis": {
      "title": "值"
    },
    "yaxis": {
      "title": "频率"
    },
    "bargap": 0.1,
    "template": "plotly_white"
  },
  "data": [
    {
      "type": "histogram",
      "x": [
        -2.5,
        -1.5,
        -0.5,
        0.5,
        1.5,
        2.5
      ],
      "y": [
        8,
        45,
        77,
        55,
        14,
        1
      ],
      "marker": {
        "color": "#228be6"
      }
    }
  ]
}
```

</details>



> 一个基本的直方图，显示了样本数据的频率分布。

#### 控制 Bin

bin 的数量和宽度会显著影响直方图的外观以及从中获取的信息。过少的bin可能过于简化分布，从而隐藏重要特征。过多的bin可能使图表显得嘈杂，从而强调随机波动。

`histplot` 函数提供了 `bins` 参数 (parameter)来控制这一点。您可以指定一个整数来表示bin的数量，或者提供一个列表或数组来定义bin的边缘。默认情况下，`histplot` 使用自动算法（通常是 Freedman-Diaconis 或 Sturges）来估算一个合适的bin数量。

让我们看看改变 bin 数量如何影响图表：

```python
# 使用较少 bin 的直方图（例如 5 个）
sns.histplot(data=data, bins=5)
plt.title('包含 5 个 bin 的直方图')
plt.xlabel('值')
plt.ylabel('频率')
plt.show()

# 使用较多 bin 的直方图（例如 30 个）
sns.histplot(data=data, bins=30)
plt.title('包含 30 个 bin 的直方图')
plt.xlabel('值')
plt.ylabel('频率')
plt.show()
```



[交互图表：包含 5 个 bin 的直方图](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-5-seaborn-visualizing-distributions/seaborn-histplot-kdeplot#plot-i4zmzx)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "包含 5 个 bin 的直方图",
    "xaxis": {
      "title": "值"
    },
    "yaxis": {
      "title": "频率"
    },
    "bargap": 0.1,
    "template": "plotly_white"
  },
  "data": [
    {
      "type": "histogram",
      "x": [
        -2.5,
        -1.5,
        -0.5,
        0.5,
        1.5
      ],
      "y": [
        8,
        45,
        77,
        55,
        15
      ],
      "xbins": {
        "size": 1.0
      },
      "marker": {
        "color": "#228be6"
      }
    }
  ]
}
```

</details>



> 仅包含 5 个 bin 的直方图。整体形态可见，但细节有所丢失。



[交互图表：包含 30 个 bin 的直方图](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-5-seaborn-visualizing-distributions/seaborn-histplot-kdeplot#plot-1usjeju)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "包含 30 个 bin 的直方图",
    "xaxis": {
      "title": "值"
    },
    "yaxis": {
      "title": "频率"
    },
    "bargap": 0.1,
    "template": "plotly_white"
  },
  "data": [
    {
      "type": "histogram",
      "x": [
        -2.5,
        -1.5,
        -0.5,
        0.5,
        1.5,
        2.5
      ],
      "y": [
        8,
        45,
        77,
        55,
        14,
        1
      ],
      "nbinsx": 30,
      "marker": {
        "color": "#228be6"
      }
    }
  ]
}
```

</details>



> 包含 30 个 bin 的直方图。这显示了更多细节，但可能显得略微锯齿化。

通常需要对 `bins` 参数进行尝试，以找到最适合您特定数据集的表示形式。

### 核密度估计与 `kdeplot`

直方图将数据分组到离散的bin中，而核密度估计（KDE）则提供了一种平滑、连续的潜在分布表示。可以想象在每个数据点上方放置一个小“核”（一个平滑的隆起，通常是高斯形状），然后将这些核累加起来以获得一条平滑曲线。KDE 曲线在给定点的高度表示该点的概率密度估计值。

Seaborn 的 `kdeplot` 函数创建这些平滑估计：

```python
# 使用 Seaborn 的 kdeplot 创建 KDE 图
sns.kdeplot(data=data)
plt.title('样本数据的 KDE 图')
plt.xlabel('值')
plt.ylabel('密度')
plt.show()
```



[交互图表：KDE 图](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-5-seaborn-visualizing-distributions/seaborn-histplot-kdeplot#plot-2at4mw)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "KDE 图",
    "xaxis": {
      "title": "值"
    },
    "yaxis": {
      "title": "密度"
    },
    "template": "plotly_white"
  },
  "data": [
    {
      "type": "scatter",
      "mode": "lines",
      "x": [
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3
      ],
      "y": [
        0.01,
        0.08,
        0.28,
        0.39,
        0.22,
        0.05,
        0.01
      ],
      "line": {
        "color": "#1c7ed6",
        "width": 2
      }
    }
  ]
}
```

</details>



> 一个平滑的核密度估计图，表示样本数据的分布。

#### 理解带宽

KDE 图的平滑度由一个名为**带宽**的参数 (parameter)控制。较大的带宽会生成更平滑的曲线，可能遮盖住更细致的细节。较小的带宽会导致曲线不那么平滑，更“摆动”，可能过于贴合样本数据。Seaborn 的 `kdeplot` 使用 `bw_adjust` 参数来调整自动计算的带宽（值为 1 表示使用默认值，< 1 使其不那么平滑，> 1 使其更平滑）。

```python
# 带宽较小（平滑度较低）的 KDE 图
sns.kdeplot(data=data, bw_adjust=0.5)
plt.title('KDE 图 (bw_adjust = 0.5)')
plt.xlabel('值')
plt.ylabel('密度')
plt.show()

# 带宽较大（平滑度较高）的 KDE 图
sns.kdeplot(data=data, bw_adjust=2)
plt.title('KDE 图 (bw_adjust = 2)')
plt.xlabel('值')
plt.ylabel('密度')
plt.show()
```

选择合适的带宽类似于在直方图中选择 bin 的数量。Seaborn 的默认设置通常效果良好，但可能需要根据数据和分析目的进行调整。

### 结合直方图和 KDE

有时，在同一张图上同时查看分 bin 表示和平滑估计是有用的。`histplot` 函数通过 `kde` 参数 (parameter)使这变得容易：

```python
# 创建一个叠加了 KDE 的直方图
sns.histplot(data=data, kde=True)
plt.title('叠加 KDE 的直方图')
plt.xlabel('值')
plt.ylabel('频率 / 密度') # 注意：Y轴现在同时表示两者
plt.show()
```



[交互图表：叠加 KDE 的直方图](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-5-seaborn-visualizing-distributions/seaborn-histplot-kdeplot#plot-iwqfgm)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "叠加 KDE 的直方图",
    "xaxis": {
      "title": "值"
    },
    "yaxis": {
      "title": "密度"
    },
    "bargap": 0.1,
    "template": "plotly_white"
  },
  "data": [
    {
      "type": "histogram",
      "x": [
        -2.5,
        -1.5,
        -0.5,
        0.5,
        1.5,
        2.5
      ],
      "y": [
        8,
        45,
        77,
        55,
        14,
        1
      ],
      "marker": {
        "color": "#a5d8ff"
      },
      "histnorm": "density",
      "name": "直方图"
    },
    {
      "type": "scatter",
      "mode": "lines",
      "x": [
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3
      ],
      "y": [
        0.01,
        0.08,
        0.28,
        0.39,
        0.22,
        0.05,
        0.01
      ],
      "line": {
        "color": "#1c7ed6",
        "width": 2
      },
      "name": "KDE"
    }
  ]
}
```

</details>



> 叠加了核密度估计的直方图。直方图的条形已缩放以表示密度，从而与 KDE 曲线匹配。

当 `kde=True` 时，`histplot` 会自动缩放直方图的条形，使其面积总和为 1（表示密度而非原始频率），从而能与同样表示密度的 KDE 曲线进行有意义的比较。

直方图和 KDE 图是查看数据分布的重要工具。`histplot` 提供了一个对 bin 宽度选择敏感的分类视图，而 `kdeplot` 则提供了一个对带宽选择敏感的平滑视图。通常，将它们结合使用可以获得最全面的认识。

## 参考资料

- [seaborn.histplot](https://seaborn.pydata.org/generated/seaborn.histplot.html) — Waskom, Michael, et al. (2023)
  Seaborn `histplot` 函数的官方文档，详细介绍了创建直方图的参数、用法和示例。
- [Chapter 4.14. Visualization With Seaborn](https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html) — VanderPlas, Jake (2016)
  Publisher: O'Reilly Media
  使用 Seaborn 进行统计数据可视化的实用指南，包含直方图和 KDE 的清晰说明和代码示例。
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Wilke, Claus O. (2019)
  Publisher: O'Reilly Media
  一本关于有效数据可视化原则的书籍，其中包含讨论直方图和密度图设计与解释的部分。

---

[上一节](../04-Seaborn%20%E7%AE%80%E4%BB%8B%EF%BC%9A%E6%8F%90%E5%8D%87%E5%9B%BE%E8%A1%A8%E6%95%88%E6%9E%9C/07-%E7%BB%83%E4%B9%A0%EF%BC%9A%E4%BD%BF%E7%94%A8Seaborn%E9%87%8D%E5%88%B6%E5%9B%BE%E8%A1%A8.md) · [下一节](02-%E5%8F%AF%E8%A7%86%E5%8C%96%E8%81%94%E5%90%88%E5%88%86%E5%B8%83%20%28jointplot%29.md)

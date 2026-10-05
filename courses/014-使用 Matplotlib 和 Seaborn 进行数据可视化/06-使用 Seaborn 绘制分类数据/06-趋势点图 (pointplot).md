# 趋势点图 (pointplot)

来源：[原文](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-6-seaborn-plotting-categorical-data/seaborn-pointplot)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然条形图能很好地表示每个类别的中心趋势（如平均值），但有时您更关注这些中心趋势在不同类别或组之间如何*变化*。Seaborn 的 `pointplot` 函数就是为此设计的。它侧重于比较分类轴上不同点之间的点估计及其置信区间。

与使用条形高度表示值的条形图不同，点图用一个点表示估计值（默认为平均值），并用一条垂直线表示不确定性（默认为 95% 置信区间）。点图对于识别趋势特别有用之处在于，它们通常用线条连接属于同一组的点（通过 `hue` 参数 (parameter)指定）。这种视觉连接使判断分类变量之间如何互相影响变得更容易。

让我们看看如何创建一个点图。我们将使用 Seaborn，并假设数据已载入 Pandas DataFrame。设想一个包含餐厅小费信息的数据集，包括总账单、星期几以及顾客是否吸烟。我们可能想了解平均小费金额如何随日期变化，或许可以区分吸烟者和非吸烟者。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 示例 DataFrame（请替换为您的实际数据加载）
# 示例：tips = sns.load_dataset('tips')
data = {
    'day': ['Thur', 'Thur', 'Fri', 'Fri', 'Sat', 'Sat', 'Sun', 'Sun', 'Thur', 'Fri', 'Sat', 'Sun'],
    'tip': [2.5, 3.0, 4.0, 3.5, 5.0, 4.5, 6.0, 5.5, 2.0, 3.8, 4.8, 5.8],
    'smoker': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes']
}
tips = pd.DataFrame(data)

# 创建点图
plt.figure(figsize=(8, 5)) # 调整图表大小以提高清晰度
sns.pointplot(x='day', y='tip', hue='smoker', data=tips, 
              order=['Thur', 'Fri', 'Sat', 'Sun'], # 定义日期的顺序
              palette={'Yes': '#f03e3e', 'No': '#1c7ed6'}, # 使用特定颜色
              markers=['o', 'x'], # 为不同 hue 级别使用不同标记
              linestyles=['-', '--']) # 为不同 hue 级别使用不同线条样式

# 添加标题和标签（可选自定义）
plt.title('Average Tip Amount by Day and Smoker Status')
plt.xlabel('Day of the Week')
plt.ylabel('Average Tip ($)')

# 显示图表
plt.show()
```

### 理解图表

在生成的图表中：

1. **X轴**：显示您指定的类别（例如，“day”）。
2. **Y轴**：显示数值变量（例如，“tip”）。
3. **点（标记 (token)）**：表示 `x` 轴上每个类别 `y` 变量的计算估计值（默认为平均值）。如果使用 `hue`，则在每个 `x` 类别内，每个 `hue` 级别都会有单独的点。不同 `hue` 级别可能会使用不同的标记形状。
4. **垂直线**：表示估计值周围的置信区间（通常为 95%）。这能让您了解估计值的不确定性或变异性。线越宽表示不确定性越大。
5. **连接线**：如果使用了 `hue`，则线条连接不同 `x` 类别中*相同* `hue` 级别的估计值。这是帮助可视化趋势或互相影响的主要特点。不同 `hue` 级别可能会使用不同的线条样式。

### `pointplot` 的参数 (parameter)

- `x`、`y`：水平和垂直轴的变量。一个应为分类变量，另一个为数值变量。
- `data`：包含数据的 Pandas DataFrame。
- `hue`：第二个分类变量，用于进一步对数据进行分组。属于相同 `hue` 级别的点会由线条连接。
- `estimator`：用于计算每个类别点估计的统计函数。默认为 `numpy.mean`。您可以使用 `numpy.median`、`numpy.std` 或任何聚合数据的函数。例如，`estimator=np.median`。
- `ci`：围绕估计值绘制的置信区间的大小。默认为 95（表示 95% 置信区间）。设置为 `None` 可禁用置信区间。它也可以接受 'sd' 来表示标准差。
- `order`、`hue_order`：字符串列表，用于指定 `x` 轴或 `hue` 变量上分类级别的出现顺序。
- `markers`：一个字符串或字符串列表，指定用于点的标记 (token)样式。
- `linestyles`：一个字符串或字符串列表，指定用于连接线的线条样式。
- `palette`：指定用于 `hue` 变量不同级别的颜色。



[交互图表：每日平均小费金额（点图示例）](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-6-seaborn-plotting-categorical-data/seaborn-pointplot#plot-1xq7yrp)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "xaxis": {
      "title": "星期几",
      "categoryorder": "array",
      "categoryarray": [
        "Thur",
        "Fri",
        "Sat",
        "Sun"
      ]
    },
    "yaxis": {
      "title": "平均小费 ($)"
    },
    "title": "每日平均小费金额（点图示例）",
    "autosize": true,
    "height": 350
  },
  "data": [
    {
      "type": "scatter",
      "x": [
        "Thur",
        "Fri",
        "Sat",
        "Sun"
      ],
      "y": [
        2.75,
        3.77,
        4.75,
        5.75
      ],
      "mode": "lines+markers",
      "name": "示例趋势",
      "marker": {
        "color": "#228be6",
        "size": 8
      },
      "line": {
        "color": "#228be6"
      },
      "error_y": {
        "type": "data",
        "array": [
          0.3,
          0.2,
          0.2,
          0.15
        ],
        "arrayminus": [
          0.3,
          0.2,
          0.2,
          0.15
        ],
        "visible": true,
        "color": "#228be6",
        "thickness": 1.5,
        "width": 3
      }
    }
  ]
}
```

</details>



> 一个简化示例，显示每日平均小费的点估计（点），通过线条连接以强调趋势，带有垂直误差条表示置信区间。

### 何时使用点图

点图在以下情况中特别有效：

1. **比较多个类别之间的估计值**：轻松查看一个类别的平均值（或中位数等）是否高于或低于其他类别。
2. **识别互相影响**：当使用 `hue` 时，连接线的斜率可以显示变量之间如何互相影响。如果不同 `hue` 组的线条平行，则表明变量之间没有互相影响。如果它们交叉或斜率差异很大，则表示 `x` 变量的影响取决于 `hue` 变量。
3. **强调变化或趋势**：连接线比 `barplot` 中的单独条形更能明确地显示相邻类别之间的差异或趋势。

尽管 `stripplot` 和 `swarmplot` 显示每个数据点，但 `pointplot`（像 `barplot` 和 `boxplot` 一样）通过估计值和置信区间总结每个类别内的分布。`pointplot` 的独特优势在于它使用连接线来直接比较跨类别的这些汇总统计数据，尤其是在涉及 `hue` 变量时。

## 参考资料

- [seaborn.pointplot](https://seaborn.pydata.org/generated/seaborn.pointplot.html) — Michael Waskom and the Seaborn development team (2024)
  Seaborn pointplot函数的官方文档，详细介绍了其参数、用法和示例。
- [Visualizing statistical relationships - categorical data](https://seaborn.pydata.org/tutorial/categorical.html) — Michael Waskom and the Seaborn development team (2024)
  Seaborn教程，概述了各种分类图，包括点图，并指导其恰当应用。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  一本关于Python数据科学工具的参考书，包含Matplotlib和Seaborn数据可视化工具的详细章节。

---

[上一节](05-%E6%95%A3%E7%82%B9%E5%88%86%E5%B8%83%E5%9B%BE%E4%B8%8E%E5%88%86%E7%BE%A4%E6%95%A3%E7%82%B9%E5%9B%BE%EF%BC%9A%E7%8B%AC%E7%AB%8B%E6%95%B0%E6%8D%AE%E7%82%B9%E7%9A%84%E5%91%88%E7%8E%B0%20%28stripplot%2C%20swarmplot%29.md) · [下一节](07-%E5%AE%9E%E8%B7%B5%E6%93%8D%E4%BD%9C%EF%BC%9A%E5%88%86%E7%B1%BB%E7%89%B9%E5%BE%81%E7%9A%84%E5%8F%AF%E8%A7%86%E5%8C%96.md)

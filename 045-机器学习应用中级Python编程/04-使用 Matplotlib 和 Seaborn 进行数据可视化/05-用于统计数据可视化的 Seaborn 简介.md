# 用于统计数据可视化的 Seaborn 简介

来源：[原文](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-4-data-visualization-matplotlib-seaborn/seaborn-introduction)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然 Matplotlib 为在 Python 中制作可视化图表提供了强大而灵活的支撑，但生成复杂的统计图表通常需要大量定制。在这种情况下，Seaborn 就派上用场了。Seaborn 是一个基于 Matplotlib 的 Python 数据可视化库，它提供了一个高级接口，用于绘制美观且信息丰富的统计图形。

可以将 Seaborn 视为 Matplotlib 的一个补充工具，而不一定是替代品。它建立在 Matplotlib 的能力之上，专门面向统计数据可视化。它的主要优点包括：

1. **简化语法：** Seaborn 为复杂的图表类型提供函数，这些图表类型如果用 Matplotlib 可能需要很多行代码。它的目标是让可视化成为理解数据的重要组成部分。
2. **与 Pandas 集成：** Seaborn 与 Pandas DataFrame 配合得非常好。许多绘图函数可以直接接受 DataFrame 列作为参数 (parameter)，从而简化数据处理。
3. **统计估算：** 多个绘图函数会自动执行必要的统计聚合或估算，以生成信息丰富的视图。
4. **吸引人的默认设置：** Seaborn 自带多种内置主题和调色板，旨在美观且具有统计信息价值，通常只需少量调整即可生成出版质量的图表。

因为 Seaborn 建立在 Matplotlib 之上，您可以在需要时保留使用 Matplotlib 命令的能力，以进一步定制 Seaborn 图表。

### Seaborn 的方法

Seaborn 的设计理念围绕着以数据集为中心的绘图函数。您无需考虑绘制单个数据数组（就像您在使用 Matplotlib 时经常做的那样），您可以直接使用数据集（通常是 Pandas DataFrame），并指定您想要可视化的变量（列）以及如何将它们映射到图表的视觉属性（例如 x 轴、y 轴、颜色、大小等）。

### 准备工作：基本样式设置

在创建图表之前，Seaborn 允许您设置全局美学样式。`seaborn.set_theme()` 函数（或旧版 `seaborn.set()`）会为所有后续的 Matplotlib 和 Seaborn 图表应用吸引人的默认样式。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 应用 Seaborn 默认主题
sns.set_theme()

# 创建一些示例数据
data = pd.DataFrame({
    'x_values': np.random.randn(100),
    'y_values': np.random.randn(100) * 2 + 0.5,
    'category': np.random.choice(['A', 'B'], 100)
})

# 使用 Seaborn 制作一个简单的散点图
sns.scatterplot(data=data, x='x_values', y='y_values', hue='category')
plt.title('简单的 Seaborn 散点图')
plt.show()
```



[交互图表：简单的 Seaborn 散点图](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-4-data-visualization-matplotlib-seaborn/seaborn-introduction#plot-gqb6x1)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "scatter",
      "mode": "markers",
      "x": [
        -0.4,
        -0.11,
        1.48,
        0.49,
        -0.34,
        -0.4,
        -0.89,
        0.13,
        1.03,
        0.48,
        0.3,
        0.38,
        -0.82,
        0.55,
        0.38,
        -0.19,
        -1.0,
        0.26,
        -0.28,
        0.24,
        -0.19,
        -0.56,
        0.54,
        0.27,
        1.48,
        0.22,
        0.73,
        -0.54,
        -0.78,
        -1.1,
        0.24,
        -0.28,
        -1.18,
        -0.49,
        0.18,
        0.3,
        0.8,
        -1.11,
        0.05,
        -0.84,
        0.28,
        -0.33,
        -0.31,
        0.72,
        -0.46,
        -0.78,
        0.32,
        0.14,
        0.56,
        -0.59,
        0.18,
        0.31,
        0.1,
        1.19,
        -1.22,
        0.7,
        0.45,
        0.37,
        1.1,
        0.51,
        -0.98,
        0.09,
        0.05,
        -1.03,
        0.31,
        0.22,
        0.87,
        -1.35,
        0.17,
        1.08,
        -0.38,
        -0.92,
        -1.73,
        -1.36,
        -1.04,
        1.17,
        0.16,
        -0.01,
        1.34,
        -0.84,
        0.32,
        -1.64,
        -1.21,
        -1.34,
        0.59,
        0.18,
        0.38,
        0.62,
        -1.0,
        -0.86,
        -1.29,
        0.2,
        -0.08,
        0.34,
        -0.37,
        0.56,
        -0.12,
        -0.86
      ],
      "y": [
        -1.64,
        2.48,
        -1.24,
        -0.29,
        0.99,
        -1.66,
        -0.29,
        2.08,
        1.83,
        0.81,
        0.41,
        1.65,
        -1.19,
        -1.46,
        -0.17,
        0.31,
        -1.15,
        2.66,
        1.05,
        0.74,
        -1.4,
        -0.1,
        -0.48,
        -1.56,
        -1.4,
        2.18,
        2.19,
        -0.11,
        0.74,
        -1.09,
        -0.6,
        -2.49,
        1.15,
        0.94,
        1.63,
        2.54,
        2.09,
        1.52,
        0.21,
        -2.26,
        -0.41,
        -0.8,
        2.03,
        1.33,
        -0.14,
        1.12,
        -0.19,
        -1.3,
        0.67,
        0.39,
        1.1,
        0.96,
        2.52,
        2.24,
        1.19,
        1.14,
        0.76,
        0.0,
        1.18,
        -1.04,
        2.27,
        -1.05,
        0.89,
        -0.37,
        1.19,
        1.14,
        0.22,
        -1.26,
        0.93,
        2.5,
        -1.26,
        0.22,
        0.92,
        1.51,
        0.25,
        0.31,
        0.12,
        -0.43,
        -0.3,
        -1.06,
        -0.44,
        -0.29,
        -1.84,
        2.7,
        -0.59,
        -1.15,
        1.4,
        -0.56,
        1.37,
        -0.73,
        0.53,
        0.24,
        0.49,
        0.29,
        1.12,
        -1.45,
        -0.14
      ],
      "marker": {
        "color": [
          "#4263eb",
          "#f03e3e",
          "#4263eb",
          "#4263eb",
          "#4263eb",
          "#f03e3e",
          "#4263eb",
          "#f03e3e",
          "#4263eb",
          "#f03e3e",
          "#f03e3e",
          "#f03e3e",
          "#4263eb",
          "#4263eb",
          "#f03e3e",
          "#f03e3e",
          "#f03e3e",
          "#4263eb",
          "#f03e3e",
          "#4263eb",
          "#4263eb",
          "#f03e3e",
          "#4263eb",
          "#4263eb",
          "#f03e3e",
          "#4263eb",
          "#f03e3e",
          "#4263eb",
          "#f03e3e",
          "#f03e3e",
          "#f03e3e",
          "#4263eb",
          "#f03e3e",
          "#f03e3e",
          "#f03e3e",
          "#f03e3e",
          "#4263eb",
          "#4263eb",
          "#f03e3e",
          "#4263eb",
          "#f03e3e",
          "#f03e3e",
          "#f03e3e",
          "#f03e3e",
          "#f03e3e",
          "#4263eb",
          "#f03e3e",
          "#f03e3e",
          "#f03e3e",
          "#4263eb",
          "#4263eb",
          "#f03e3e",
          "#f03e3e",
          "#f03e3e",
          "#4263eb",
          "#4263eb",
          "#f03e3e",
          "#4263eb",
          "#f03e3e",
          "#f03e3e",
          "#f03e3e",
          "#f03e3e",
          "#f03e3e",
          "#f03e3e",
          "#4263eb",
          "#f03e3e",
          "#f03e3e",
          "#f03e3e",
          "#4263eb",
          "#4263eb",
          "#f03e3e",
          "#f03e3e",
          "#4263eb",
          "#f03e3e",
          "#4263eb",
          "#f03e3e",
          "#f03e3e",
          "#4263eb",
          "#4263eb",
          "#f03e3e",
          "#f03e3e",
          "#f03e3e",
          "#4263eb",
          "#f03e3e",
          "#f03e3e",
          "#4263eb",
          "#f03e3e",
          "#4263eb",
          "#4263eb",
          "#f03e3e",
          "#f03e3e",
          "#f03e3e",
          "#4263eb",
          "#f03e3e",
          "#4263eb",
          "#f03e3e",
          "#4263eb",
          "#4263eb"
        ],
        "size": 6
      },
      "name": "A"
    },
    {
      "type": "scatter",
      "mode": "markers",
      "x": [],
      "y": [],
      "marker": {
        "color": "#f03e3e",
        "size": 6
      },
      "name": "B"
    }
  ],
  "layout": {
    "xaxis": {
      "title": {
        "text": "x 值"
      }
    },
    "yaxis": {
      "title": {
        "text": "y 值"
      }
    },
    "title": {
      "text": "简单的 Seaborn 散点图"
    },
    "legend": {
      "traceorder": "reversed"
    },
    "margin": {
      "t": 60
    }
  }
}
```

</details>



> 从 Pandas DataFrame 生成的 Seaborn 散点图，自动根据“category”列分配颜色。

请注意 `sns.scatterplot` 如何直接接收 DataFrame (`data=data`) 以及 x 轴和 y 轴的列名 (`x='x_values'`, `y='y_values'`)。`hue` 参数 (parameter)根据指定的分类列自动为点着色。这种简洁的语法是 Seaborn 的典型特点，与单独使用 Matplotlib 相比，它大大简化了常见统计图表的制作。

在本章的后续内容中，您将学习如何运用 Seaborn 的专用函数，以快速生成富有洞察力的可视化图表，这些图表展示了数据的分布、关系和分类信息，它们是任何机器学习 (machine learning)项目数据审视阶段的重要步骤。

## 参考资料

- [Seaborn: statistical data visualization](https://seaborn.pydata.org/) — Michael Waskom and the Seaborn development team (2024)
  Seaborn库的官方用户指南和API参考，展示其功能和设计。
- [Python for Data Analysis](https://learning.oreilly.com/library/view/python-for-data/9781098104030/) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: 579
  一本全面涵盖Matplotlib、Pandas和Seaborn的书籍，对Python中的数据处理和可视化至关重要。
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media
  本书系统性地指导如何创建有效且信息丰富的统计图表，为Seaborn等工具提供理论基础。
- [Data Visualization in Python: For Data Scientists, Developers, and Analysts](https://www.oreilly.com/library/view/data-visualization-in/9781492067711/) — Kyran Dale (2020)
  Publisher: O'Reilly Media
  一本使用Python库构建数据可视化的实践参考，其中详细探讨了Seaborn。

---

[上一节](04-%E4%BD%BF%E7%94%A8%E5%AD%90%E5%9B%BE.md) · [下一节](06-%E4%BD%BF%E7%94%A8%20Seaborn%20%E5%88%9B%E5%BB%BA%E9%AB%98%E7%BA%A7%E5%9B%BE%E8%A1%A8.md)

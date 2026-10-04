# 使用 Seaborn 创建高级图表

来源：[原文](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-4-data-visualization-matplotlib-seaborn/seaborn-advanced-plots)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管 Matplotlib 提供了 Python 中绘图的基础工具，但 Seaborn 提供了一个更高级的接口，专门用于制作信息丰富且美观的统计图形。Seaborn 是在 Matplotlib 之上构建的，它简化了生成复杂可视化的过程，这在数据分析和机器学习 (machine learning)中很常见，对于在 Matplotlib 中需要大量定制的图表，Seaborn 通常只需一次函数调用即可完成。

本节介绍几种 Seaborn 中的高级图表类型，它们对于分析数据集中变量间的关系和结构特别有用。这些图表有助于发现仅凭简单图表可能不明显的规律。

### 热图

热图非常适合将矩阵式数据可视化，其中单个值由颜色表示。它们常用于显示相关矩阵，以紧凑的视觉格式展示多个变量间的相关系数。暖色通常表示正相关，冷色表示负相关，颜色的深浅代表强度。

要创建热图，通常从二维数组或 Pandas DataFrame 开始。例如，计算 DataFrame 的相关矩阵会得到一个非常适合热图的结构。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 生成示例数据
np.random.seed(42)
data = pd.DataFrame(np.random.rand(10, 5), columns=[f'Var{i}' for i in range(1, 6)])
data['Var3'] = data['Var1'] * 2 + np.random.normal(0, 0.1, 10)
data['Var5'] = -data['Var2'] * 1.5 + np.random.normal(0, 0.2, 10)

# 计算相关矩阵
correlation_matrix = data.corr()

# 创建热图
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f")
plt.title('相关矩阵热图')
plt.show()
```



[交互图表：相关矩阵热图](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-4-data-visualization-matplotlib-seaborn/seaborn-advanced-plots#plot-1i1uss2)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "相关矩阵热图",
    "xaxis": {
      "tickvals": [
        0,
        1,
        2,
        3,
        4
      ],
      "ticktext": [
        "Var1",
        "Var2",
        "Var3",
        "Var4",
        "Var5"
      ]
    },
    "yaxis": {
      "tickvals": [
        0,
        1,
        2,
        3,
        4
      ],
      "ticktext": [
        "Var1",
        "Var2",
        "Var3",
        "Var4",
        "Var5"
      ],
      "autorange": "reversed"
    },
    "coloraxis": {
      "colorscale": "Viridis",
      "colorbar": {
        "title": "相关性"
      }
    }
  },
  "data": [
    {
      "type": "heatmap",
      "z": [
        [
          1.0,
          0.34,
          0.98,
          0.14,
          -0.58
        ],
        [
          0.34,
          1.0,
          0.29,
          -0.18,
          -0.95
        ],
        [
          0.98,
          0.29,
          1.0,
          0.14,
          -0.53
        ],
        [
          0.14,
          -0.18,
          0.14,
          1.0,
          0.18
        ],
        [
          -0.58,
          -0.95,
          -0.53,
          0.18,
          1.0
        ]
      ],
      "x": [
        "Var1",
        "Var2",
        "Var3",
        "Var4",
        "Var5"
      ],
      "y": [
        "Var1",
        "Var2",
        "Var3",
        "Var4",
        "Var5"
      ],
      "colorbar": {
        "title": "相关性"
      },
      "zmin": -1,
      "zmax": 1,
      "hoverongaps": false,
      "showscale": true,
      "colorscale": "Viridis",
      "xgap": 1,
      "ygap": 1
    }
  ]
}
```

</details>



> 相关矩阵可视化为热图。颜色深浅表示相关性的强度，注释显示具体的系数值。

在 `sns.heatmap()` 函数中：

- 第一个参数 (parameter)是数据矩阵（例如 `correlation_matrix`）。
- `annot=True` 在单元格上显示数据值。
- `cmap` 设置颜色映射（例如 'viridis', 'coolwarm', 'YlGnBu'）。
- `fmt=".2f"` 将注释文本格式化为两位小数。

### 配对图

在进行探索性数据分析（EDA）时，同时理解多个数值变量之间的关系通常是必要的。配对图（也称为散点图矩阵）提供一个轴网格，其中数据集中的每个变量都与所有其他变量进行绘图。对角线轴通常显示每个变量的单变量分布（直方图或核密度估计）。

该图对于快速查看双变量关系以及识别不同变量组合中潜在的相关性或规律非常有用。Seaborn 的 `sns.pairplot()` 函数使得生成这些网格变得简单直接。

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 从 Seaborn 加载示例数据集
iris = sns.load_dataset('iris')

# 创建配对图
# 'hue' 根据 'species' 类别为点上色
sns.pairplot(iris, hue='species', palette='viridis')
plt.suptitle('Iris 数据集中的两两关系', y=1.02) # 调整标题位置
plt.show()
```

> 代码为 Iris 数据集生成了一个配对图。非对角线图是散点图，显示特征对之间的关系，并按物种着色。对角线图显示每个物种各特征的分布（KDE）。

`sns.pairplot()` 函数将 DataFrame 作为输入。

- `hue` 参数 (parameter)功能强大；它允许您根据分类变量（例如 `iris` 数据集中的 'species'）为点上色，这样可以很容易地看出不同组在变量对之间是否呈现不同的聚类。
- `palette` 控制用于 `hue` 变量的配色方案。

虽然信息量非常大，但请注意，为具有大量变量的数据集生成配对图可能会变得计算密集且视觉混乱。

### 小提琴图

小提琴图是一种可视化不同类别数值数据分布的方法。它们类似于箱线图，但通过在每侧加入核密度估计（KDE）来提供更多信息。这使您可以查看分布的形状，包括潜在的多峰性（多个峰值），这在标准箱线图中是隐藏的。

`sns.violinplot()` 函数用于创建这些图。它通常将一个分类变量用于 x 轴，一个数值变量用于 y 轴。

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 加载 tips 数据集
tips = sns.load_dataset("tips")

# 创建小提琴图
plt.figure(figsize=(10, 6))
sns.violinplot(x="day", y="total_bill", data=tips, palette="coolwarm")
plt.title('每日总账单金额的分布')
plt.xlabel('星期几')
plt.ylabel('总账单金额 ($)')
plt.show()
```



[交互图表：每日总账单金额的分布](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-4-data-visualization-matplotlib-seaborn/seaborn-advanced-plots#plot-ufopnj)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "每日总账单金额的分布",
    "xaxis": {
      "title": "星期几"
    },
    "yaxis": {
      "title": "总账单金额 ($)"
    },
    "violinmode": "group"
  },
  "data": [
    {
      "type": "violin",
      "x": [
        "Thur"
      ],
      "y": [
        7.51,
        10.34,
        16.97,
        10.59,
        10.65,
        12.43,
        8.35,
        11.17,
        12.03,
        13.42,
        14.26,
        15.95,
        16.99,
        18.78,
        14.73,
        10.51,
        17.92,
        20.69,
        16.47,
        13.16,
        11.59,
        7.74,
        16.58,
        11.35,
        16.27,
        13.81,
        11.02,
        18.29,
        22.76,
        19.49,
        27.2,
        22.82,
        19.08,
        15.06,
        12.48,
        19.81,
        29.03,
        27.18,
        34.3,
        41.19,
        27.05,
        16.82,
        32.68,
        15.98,
        34.83,
        13.03,
        18.28,
        24.71,
        30.4,
        8.52,
        14.52,
        11.38,
        22.49,
        5.75,
        16.32,
        22.12,
        24.08,
        15.69,
        11.61,
        13.13,
        43.11,
        9.68
      ],
      "name": "周四",
      "box": {
        "visible": true
      },
      "meanline": {
        "visible": true
      },
      "marker": {
        "color": "#1c7ed6"
      }
    },
    {
      "type": "violin",
      "x": [
        "Fri"
      ],
      "y": [
        28.97,
        22.48,
        5.98,
        7.25,
        11.35,
        13.42,
        15.38,
        21.01,
        12.06,
        21.16,
        12.6,
        8.58,
        18.64,
        15.98,
        13.42,
        27.28,
        12.16,
        12.46,
        40.17
      ],
      "name": "周五",
      "box": {
        "visible": true
      },
      "meanline": {
        "visible": true
      },
      "marker": {
        "color": "#fa5252"
      }
    },
    {
      "type": "violin",
      "x": [
        "Sat"
      ],
      "y": [
        10.29,
        3.07,
        20.29,
        15.77,
        39.42,
        19.65,
        9.55,
        18.35,
        15.04,
        14.78,
        10.27,
        35.26,
        15.42,
        18.43,
        14.83,
        21.58,
        10.33,
        16.29,
        16.97,
        20.65,
        17.92,
        20.29,
        15.77,
        19.82,
        17.81,
        13.39,
        12.16,
        13.0,
        26.86,
        25.28,
        14.0,
        17.07,
        20.49,
        16.58,
        7.74,
        18.29,
        17.59,
        20.08,
        16.45,
        11.24,
        48.17,
        25.0,
        13.81,
        11.02,
        23.33,
        32.4,
        28.17,
        12.9,
        28.15,
        11.59,
        7.74,
        30.14,
        12.16,
        13.16,
        9.68,
        30.46,
        18.29,
        22.23,
        32.4,
        29.85,
        48.33,
        26.41,
        11.87,
        16.04,
        17.46,
        13.94,
        9.68,
        38.01,
        23.95,
        25.89,
        48.27,
        10.59,
        10.63,
        50.81,
        15.81,
        7.25,
        31.85,
        16.82,
        32.9,
        34.63,
        34.65,
        23.17,
        45.35,
        20.45,
        13.28,
        22.12,
        24.01,
        15.69,
        19.44,
        18.07,
        15.48,
        29.8,
        10.77
      ],
      "name": "周六",
      "box": {
        "visible": true
      },
      "meanline": {
        "visible": true
      },
      "marker": {
        "color": "#ae3ec9"
      }
    },
    {
      "type": "violin",
      "x": [
        "Sun"
      ],
      "y": [
        16.29,
        16.97,
        10.34,
        21.01,
        24.59,
        25.29,
        8.77,
        26.88,
        15.04,
        14.78,
        9.55,
        19.63,
        17.29,
        16.04,
        17.46,
        13.94,
        9.68,
        10.33,
        16.29,
        16.97,
        20.65,
        17.92,
        39.08,
        24.52,
        19.77,
        24.06,
        10.65,
        20.9,
        30.4,
        18.15,
        23.1,
        15.69,
        19.81,
        29.85,
        10.63,
        23.95,
        25.71,
        17.31,
        29.93,
        10.59,
        10.63,
        23.1,
        15.69,
        19.81,
        29.85,
        17.81,
        16.49,
        21.5,
        12.64,
        16.45,
        11.24,
        20.53,
        25.0,
        13.81,
        11.02,
        23.33,
        16.0,
        10.77,
        15.53,
        7.56,
        17.51,
        24.55,
        18.69,
        35.83,
        13.39,
        14.26,
        16.31,
        8.51,
        14.32,
        16.27,
        10.88,
        22.75,
        7.25,
        31.71,
        38.73,
        25.56,
        19.49,
        18.49
      ],
      "name": "周日",
      "box": {
        "visible": true
      },
      "meanline": {
        "visible": true
      },
      "marker": {
        "color": "#f76707"
      }
    }
  ]
}
```

</details>



> 小提琴图显示了每周各天的总账单金额分布。小提琴的宽度表示不同账单金额处数据点的密度。内部元素显示中位数和四分位距，类似于箱线图。

`sns.violinplot()` 的重要参数 (parameter)：

- `x`, `y`: 定义坐标轴的变量。
- `data`: 包含数据的 DataFrame。
- `palette`: 设置 x 轴上不同类别的配色方案。
- 您还可以添加 `hue` 参数，根据另一个分类变量进一步划分小提琴。

这些高级 Seaborn 图表（热图、配对图和小提琴图）提供了强大的方法，通过可视化从数据中获得更细致的理解。它们通常比基本图表更能有效地呈现复杂关系、分布和潜在问题（如异常值或偏斜数据），使它们成为机器学习 (machine learning)项目数据分析阶段的重要工具。

## 参考资料

- [Seaborn: statistical data visualization](https://seaborn.pydata.org/) — Michael Waskom (Ongoing)
  学习和使用Seaborn的主要资源，包含其API和本节介绍的各种图表类型。
- [Matplotlib Documentation](https://matplotlib.org/) — John Hunter, The Matplotlib development team (Ongoing)
  理解Seaborn所依赖的基础绘图库。
- [Python for Data Analysis](https://wesmckinney.com/book) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: 579
  一本实用指南，广泛涵盖使用Pandas进行数据处理以及使用Matplotlib和Seaborn进行数据可视化。
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media
  提供创建有效且信息丰富的可视化图表的一般原则，补充了技术操作指南。

---

[上一节](05-%E7%94%A8%E4%BA%8E%E7%BB%9F%E8%AE%A1%E6%95%B0%E6%8D%AE%E5%8F%AF%E8%A7%86%E5%8C%96%E7%9A%84%20Seaborn%20%E7%AE%80%E4%BB%8B.md) · [下一节](07-%E5%88%86%E5%B8%83%E4%B8%8E%E5%85%B3%E7%B3%BB%E7%9A%84%E5%8F%AF%E8%A7%86%E5%8C%96.md)

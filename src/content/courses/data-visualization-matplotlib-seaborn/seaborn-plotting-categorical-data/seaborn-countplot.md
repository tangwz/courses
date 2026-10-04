---
course: "data-visualization-matplotlib-seaborn"
chapter: "seaborn-plotting-categorical-data"
lesson: "seaborn-countplot"
sourceId: 1199
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-6-seaborn-plotting-categorical-data/seaborn-countplot"
title: "频率计数图 (countplot)"
description: "使用 Seaborn 的 countplot 轻松可视化每个类别内的出现次数。"
order: 3
plots: ["plots/1199-0.json"]
sourceHash: "482db127468a9efa95f3b4cc20b206027e9f285d70ef76c49c6c843a9464e591"
sourceCorrections: []
---

在探索分类数据时，一项常见的任务是统计数据集中每个唯一类别出现的次数。这项任务对于初步数据探索至关重要。例如，您可能想计算来自不同地区的客户数量、不同传感器读数的频率，或者，在常用的练习数据集中，统计一周中不同日期的就餐团体数量。

Seaborn 为此提供了一个方便的函数：`countplot`。它直接作用于您的数据，计算每个类别内的出现次数，然后将这些计数显示为条形，外观类似于条形图，但代表的是频率而非聚合值。

### 使用 `seaborn.countplot`

让我们看看如何使用 `countplot`。我们假设您有一个包含分类数据的 Pandas DataFrame。常见的做法是将 Seaborn 导入为 `sns`，将 Matplotlib 的 pyplot 模块导入为 `plt`。我们也经常直接从 Seaborn 加载示例数据集。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 从 Seaborn 加载一个示例数据集
tips = sns.load_dataset("tips")

# 显示前几行以了解数据
print(tips.head())
```

`tips` 数据集包含关于餐厅小费的信息，包括“day”、“sex”、“smoker”和“time”等分类列。

要创建一个显示一周中每天条目数量的计数图，您将 DataFrame 传递给 `data` 参数 (parameter)，并将列名（在此例中为“day”）指定给 `x` 参数：

```python
# 为“day”列创建一个计数图
plt.figure(figsize=(8, 5)) # 可选：调整图形大小
sns.countplot(x='day', data=tips, palette=['#74c0fc', '#4dabf7', '#339af0', '#228be6'])
plt.title('每天记录的小费数量')
plt.xlabel('星期几')
plt.ylabel('计数')
plt.show()
```

这段代码生成了一个图表，每个条形的高度对应于 `tips` DataFrame 中特定日期的行数（记录数）。



![每天记录的小费数量](plots/1199-0.json)



> `tips` 数据集中每天的记录频率。周六的记录最多，而周五的最少。

### 水平绘制计数图

就像 `barplot` 一样，您可以通过将分类列名分配给 `y` 参数 (parameter)而非 `x` 来创建水平计数图：

```python
# 为“smoker”列创建一个水平计数图
plt.figure(figsize=(7, 4))
sns.countplot(y='smoker', data=tips, palette=['#ffc9c9', '#ff8787'])
plt.title('吸烟者与非吸烟者计数')
plt.xlabel('计数')
plt.ylabel('吸烟者')
plt.show()
```

### 使用 `hue` 添加另一个分类维度

`countplot` 也支持 `hue` 参数 (parameter)，允许您比较跨越第二个分类变量的计数。例如，我们可以查看每天吸烟者和非吸烟者的分布情况：

```python
# 为“day”创建一个计数图，以“smoker”作为色相
plt.figure(figsize=(9, 6))
sns.countplot(x='day', hue='smoker', data=tips, palette=['#a5d8ff', '#ffc9c9']) # 蓝色表示否，红色表示是
plt.title('按吸烟状态区分的每日计数')
plt.xlabel('星期几')
plt.ylabel('计数')
plt.show()
```

这个图表现在显示了每天成对的条形。一个条形代表非吸烟者（“否”）的计数，另一个代表该吸烟者（“是”）的计数。Seaborn 自动添加图例，以澄清 `hue` 变量中哪种颜色对应哪个类别。

### `countplot` 的参数 (parameter)

- `x`, `y`: 指定 DataFrame 中用于沿水平 (`x`) 或垂直 (`y`) 轴绘制的列名。通常只使用一个来定义主要分类变量。
- `data`: 包含数据的 Pandas DataFrame。
- `hue`: DataFrame 中第二个分类列的名称。这会在 `x` 或 `y` 定义的每个主要类别中添加子组比较。
- `order`, `hue_order`: 字符串列表，用于指定类别在轴 (`order`) 或图例 (`hue_order`) 中出现的准确顺序。
- `palette`: 允许您控制条形使用的颜色。您可以使用 Seaborn 调色板名称（如 'viridis'、'magma'）或提供颜色代码列表。

`countplot` 是一个简单但有效的工具，可帮助您快速了解分类变量内的频率分布。它通常是分析包含类别的新数据集时生成的第一批图表之一。

## 参考资料

- [seaborn.countplot](https://seaborn.pydata.org/generated/seaborn.countplot.html) — Michael Waskom and the Seaborn development team (2023)
  `countplot`函数的官方文档，详细说明其参数和用法。
- [Plotting categorical data](https://seaborn.pydata.org/tutorial/categorical.html) — Michael Waskom and the Seaborn development team (2023)
  提供Seaborn分类绘图功能的全面教程，包含`countplot`，并提供实用示例和最佳实践。
- [matplotlib.pyplot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html) — The Matplotlib Development Team (2025)
  Matplotlib `pyplot`模块的参考资料，该模块提供了Matplotlib的状态机接口，用于自定义Seaborn创建的图表。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本关于Python数据分析的基础教材，涵盖Pandas、Matplotlib和Seaborn，为统计绘图提供一般背景。第三版。

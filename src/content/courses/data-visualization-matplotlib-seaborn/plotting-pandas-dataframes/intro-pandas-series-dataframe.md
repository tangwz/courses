---
course: "data-visualization-matplotlib-seaborn"
chapter: "plotting-pandas-dataframes"
lesson: "intro-pandas-series-dataframe"
sourceId: 1216
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-8-plotting-pandas-dataframes/intro-pandas-series-dataframe"
title: "Pandas Series 和 DataFrame 简要介绍"
description: "Pandas Series 和 DataFrame 结构的快速概览，它们在 Python 中进行数据处理时必不可少。"
order: 1
plots: []
sourceHash: "f3134c09354c642de84a28f0b87fcfd7830fd9116bb374742dcb36e14e45812e"
sourceCorrections: []
---

要在 Python 环境中有效管理和组织数据，特别是处理来自 CSV 文件或数据库等外部来源的数据时，Pandas 库提供了必要的工具。Pandas 提供了高性能、易用的数据结构和数据分析工具，构成了 Python 中许多数据科学工作流程的核心支持。

Pandas 的核心是两种主要数据结构：`Series` 和 `DataFrame`。理解它们对于准备数据以使用 Matplotlib 和 Seaborn 进行可视化非常重要。

### Pandas Series

可以将 Pandas `Series` 视为一个一维数组，能够存储任何单一类型的数据（整数、字符串、浮点数、Python 对象等）。它类似于 NumPy 数组，但有一个重要补充：一个相关联的数据标签数组，称为其 **索引**。如果不指定索引，Pandas 会自动创建一个从 0 开始的默认整数索引。

可以将其设想为电子表格或表格中的单列。

下面是一个从 Python 列表创建 `Series` 的简单例子：

```python
import pandas as pd

# 创建一个存储每日气温的 Series
temperatures = pd.Series([22.1, 25.0, 24.3, 26.7, 23.9], name='Temperature (C)')

print(temperatures)
```

运行此代码将得到以下输出：

```
0    22.1
1    25.0
2    24.3
3    26.7
4    23.9
Name: Temperature (C), dtype: float64
```

请注意这两列：左列是索引（本例中为 0 到 4），右列包含实际数据值。`Name` 属性为 `Series` 提供了一个标签，这可能很有用，`dtype` 则显示了这些值的数据类型（此处为 float64）。

### Pandas DataFrame

`DataFrame` 是最常用的 Pandas 对象。它表示一个矩形数据表，包含按顺序排列的列集合，每列可以是不同的值类型（数值、字符串、布尔值等）。可以将 `DataFrame` 设想为：

- 您可能在 Excel 或 Google 表格中使用的电子表格。
- 一个 SQL 表。
- 一个字典，它的键是列名，值是表示这些列的 `Series` 对象。

"特别值得指出的是，`DataFrame` 既有行索引，也有列索引。这种二维结构使得它在处理数据集方面功能非常强大，数据集通常包含每个观测（行）的多个变量（列）。"

我们来创建一个简单的 `DataFrame`：

```python
import pandas as pd

# 多个城市的数据
data = {
    'City': ['London', 'Paris', 'Tokyo', 'New York'],
    'Temperature (C)': [15.2, 18.5, 21.0, 19.8],
    'Humidity (%)': [70, 65, 75, 60]
}

# 从字典创建 DataFrame
weather_df = pd.DataFrame(data)

print(weather_df)
```

输出将是一个结构化表格：

```
       City  Temperature (C)  Humidity (%)
0    London             15.2            70
1     Paris             18.5            65
2     Tokyo             21.0            75
3  New York             19.8            60
```

这里，'City'、'Temperature (C)' 和 'Humidity (%)' 是列标签。数字 0、1、2、3 构成了行索引。此 `DataFrame` 中的每一列实际上是一个 Pandas `Series`。

> 将 DataFrame 视为共享公共索引的 Series 对象集合的视图。

### 为何使用 Pandas 进行可视化？

虽然 Matplotlib 和 Seaborn 可以从简单的列表或 NumPy 数组绘制数据，但使用 Pandas `DataFrames` 提供了显著的优势，尤其当数据集变得复杂时：

1. **带标签的数据：** DataFrames 存储列名和行索引，使代码更具可读性，并且图表更容易理解（例如，轴标签通常可以自动推断）。
   "2. **处理混合数据类型：** 数据集包含多种数据类型（数值、文本、日期）。DataFrames 可以自然地处理这些。"
2. **集成性：** Matplotlib 和 Seaborn（特别是 Seaborn）都设计为与 DataFrames 配合使用。您通常可以将整个 DataFrame 传递给绘图函数，并指定哪些列用于不同的绘图美学（x轴、y轴、颜色、大小等），使用它们的字符串名称。
3. **数据准备：** Pandas 提供了强大的工具用于在绘图*之前*清洗、筛选、分组和转换数据，这是常见的要求。

在接下来的章节中，您将看到如何将数据加载到这些结构中，并直接与 Matplotlib 和 Seaborn 结合使用，以创建富有洞察力的可视化图表。

## 参考资料

- [Pandas User Guide: Intro to Data Structures](https://pandas.pydata.org/docs/user_guide/dsintro.html) — The Pandas Development Team (2023)
  官方文档是理解Pandas Series和DataFrame基本数据结构的首要参考资料。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  本书由Pandas库的创建者撰写，详细解释了Series和DataFrame等数据结构，以及实用的数据处理技术。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  这本手册为核心Python数据科学库提供了坚实基础，其中包括专门介绍用于数据存储和操作的Pandas部分，这对可视化至关重要。

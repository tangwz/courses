---
course: "intermediate-python-programming-ml"
chapter: "data-manipulation-pandas"
lesson: "intro-pandas-structures"
sourceId: 2139
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-3-data-manipulation-pandas/intro-pandas-structures"
title: "Pandas数据结构简介"
description: "理解Pandas的核心数据结构：Series（一维）和DataFrame（二维）。"
order: 1
plots: []
sourceHash: "bcbe404a21edb42ea25ab16a47c40b2a1b04023a152cc7a277f2bf36b8fc572a"
sourceCorrections: []
---

组织和处理机器学习 (machine learning)中常见的数据集，需要有效的数据结构。NumPy为同构数值数组提供了高效数值运算的强大工具，但数据常常以表格形式呈现，混合多种数据类型（数字、文本、日期），并需要行和列的标签。对于这些复杂的数据需求，Pandas库提供了卓越的能力。

Pandas引入了两种对Python数据分析和处理不可或缺的基本数据结构：`Series`和`DataFrame`。这些结构构建在NumPy数组之上，提供增强的灵活性和功能，专门用于处理带标签和关系型数据。可以将它们视为精巧的容器，使数据处理变得直观高效。

### The Pandas Series

**Series**本质上是一维数组，能够保存任何类型的数据（整数、字符串、浮点数、Python对象等）。与NumPy数组相比，Series的决定性特征是其**索引**。索引是关联的标签数组，允许您使用这些标签而不是仅通过整数位置来访问数据。

可以将Series想象成电子表格或表中的单列。Series中的每个值在索引中都有一个对应的标签。如果您在创建Series时不指定索引，Pandas会自动创建一个从0到N-1的默认整数索引，其中N是数据的长度。

我们来创建一个简单的Series：

```python
import pandas as pd
import numpy as np

# 从Python列表创建Series
data = [10, 20, 30, 40, 50]
s = pd.Series(data)
print(s)
```

Output:

```
0    10
1    20
2    30
3    40
4    50
dtype: int64
```

请注意，输出显示了两列：左侧是索引（0到4），右侧是对应的值。`dtype: int64`表示Series中存储值的数据类型。

我们也可以创建一个带有自定义索引的Series：

```python
# 使用自定义标签创建Series
population = {'California': 39538223, 'Texas': 29145505,
              'Florida': 21538187, 'New York': 20201249}
pop_series = pd.Series(population, name='State Population') # 给Series命名
print(pop_series)
```

Output:

```
California    39538223
Texas         29145505
Florida       21538187
New York      20201249
Name: State Population, dtype: int64
```

这里，州名用作索引标签。您可以使用`.values`属性访问底层的NumPy数组，并使用`.index`属性访问索引对象。

```python
print("值:", pop_series.values)
print("索引:", pop_series.index)
```

Output:

```
Values: [39538223 29145505 21538187 20201249]
Index: Index(['California', 'Texas', 'Florida', 'New York'], dtype='object')
```

Series提供了一个强大的根基，它既像NumPy数组（支持矢量化 (quantization)操作），又像Python字典（将索引标签映射到值）。

### The Pandas DataFrame

Series表示单列数据，而**DataFrame**表示完整的表格或电子表格。它是一个二维的、大小可变的、可能异构的表格数据结构，带有标签的轴（行和列）。您可以将DataFrame视为：

- 共享相同索引的Series对象的集合。
- 一个字典，其中键是列名，值是表示这些列的Series对象。
- 一个带有行和列标签的结构化NumPy数组。

DataFrame是Pandas的主力，也是您在执行数据分析和机器学习 (machine learning)模型准备时最常打交道的结构。

我们来使用字典创建一个DataFrame，其中键成为列标签，列表成为列数据：

```python
# 从列表字典创建DataFrame
data = {'State': ['California', 'Texas', 'Florida', 'New York'],
        'Population': [39538223, 29145505, 21538187, 20201249],
        'Area (sq mi)': [163696, 268597, 65758, 54556]}
states_df = pd.DataFrame(data)
print(states_df)
```

Output:

```
        State  Population  Area (sq mi)
0  California    39538223        163696
1       Texas    29145505        268597
2     Florida    21538187         65758
3    New York    20201249         54556
```

与Series一样，如果没有指定索引，Pandas会创建一个默认的整数索引。我们可以在创建时使用`index`参数 (parameter)指定自定义行标签，或者使用`.set_index()`方法将现有列设置为索引（我们将在后面介绍）。

您可以分别使用`.index`和`.columns`属性访问行标签（索引）和列标签。底层数据通常表示为NumPy数组（如果数据类型跨列不同，则为多个数组），可以通过`.values`属性访问，尽管直接操作通常通过Pandas方法进行。

```python
print("索引:", states_df.index)
print("列:", states_df.columns)
# print(states_df.values) # 输出是数据的NumPy数组
```

Output:

```
Index: RangeIndex(start=0, stop=4, step=1)
Columns: Index(['State', 'Population', 'Area (sq mi)'], dtype='object')
```

DataFrame中的每一列都是一个Series：

```python
# 访问列会返回一个Series
population_col = states_df['Population']
print(type(population_col))
print(population_col)
```

Output:

```
<class 'pandas.core.series.Series'>
0    39538223
1    29145505
2    21538187
3    20201249
Name: Population, dtype: int64
```

### Why Use Pandas Structures?

这些结构在数据分析方面相比标准Python列表或字典，甚至原始NumPy数组，提供了若干优势：

1. **基于标签的索引：** 通过有意义的标签（如'Population'或'California'）而不是仅仅通过整数位置来访问数据，使代码更易读，并减少出错的可能性。
2. **处理异构数据：** DataFrames可以轻松容纳同一张表中具有不同数据类型的列。
3. **数据对齐 (alignment)：** Series或DataFrame之间的操作会根据其索引自动对齐数据，简化了涉及相关但可能顺序不同的数据集的计算。
4. **缺失数据处理：** Pandas提供了专门用于查找、筛选和填充缺失数据（通常表示为`NaN`，即非数字）的函数。
5. **集成操作：** 常见的数据处理任务，如分组、连接、合并、重塑和数据透视，都是内置的高性能方法。

理解Series和DataFrame是有效使用Pandas的第一步。它们奠定了基础，后续所有的数据加载、清洗、转换和分析操作都建立在此之上。在接下来的章节中，我们将学习如何创建、检查和处理这些结构，以便为机器学习 (machine learning)任务准备数据。

## 参考资料

- [User Guide: Intro to Data Structures](https://pandas.pydata.org/docs/user_guide/dsintro.html) — The pandas development team (2024)
  官方文档，详细且实时地解释了 Pandas Series 和 DataFrame。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  Pandas 创始人撰写的权威书籍，详细介绍了其核心数据结构，并提供实用示例。
- [Data Structures for Statistical Computing in Python](http://conference.scipy.org/proceedings/scipy2010/mckinney.html) — Wes McKinney (2010)
  Journal: Proceedings of the 9th Python in Science Conference (SciPy 2010); Publisher: SciPy; Pages: 56-61; DOI: [10.25080/majora-92bf1922-00a](https://doi.org/10.25080/majora-92bf1922-00a)
  一篇基础性论文，介绍了构成 Pandas Series 和 DataFrame 的数据结构的设计原则和初始实现。

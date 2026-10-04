---
course: "essential-numpy-pandas"
chapter: "introduction-to-pandas"
lesson: "creating-dataframes"
sourceId: 1755
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-5-introduction-to-pandas/creating-dataframes"
title: "创建 DataFrame"
description: "从字典、列表的列表/字典以及 NumPy 数组创建 DataFrame 的方法。"
order: 5
plots: []
sourceHash: "635ca1392120e83cca634ceebf880dd25b839bef5793957759a425af28c076d9"
sourceCorrections: []
---

Pandas `Series` 代表单列数据，而 `DataFrame` 则是您在多数表格数据分析中会使用的主要结构。可以把它看作是 Python 环境中的电子表格或 SQL 表。它是一个二维、大小可变、可能包含不同类型数据的表格结构，带有标签轴（行和列）。

我们来看看构建 `DataFrame` 的常见方法。通常，您可以从现有数据结构（如 Python 字典或 NumPy 数组）创建它们，或者通过读取文件中的数据来创建（这将在下一章中介绍）。

### 从列表或 NumPy 数组的字典创建

最常用的方法之一是使用 Python 字典，其中键代表所需的列名，值是包含每列数据的列表（或 NumPy 数组）。重要的是，用作值的所有列表或数组必须具有相同的长度，因为每个列表对应一列，并且列表在相同位置的元素构成一行。

```python
import pandas as pd
import numpy as np

# 以列表字典形式的数据
data = {
    'StudentID': ['S001', 'S002', 'S003', 'S004'],
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Score': [85, 92, 78, 88]
}

# 创建 DataFrame
df_from_dict_list = pd.DataFrame(data)

print(df_from_dict_list)
```

此代码会生成以下 `DataFrame`：

```
  StudentID     Name  Score
0      S001    Alice     85
1      S002      Bob     92
2      S003  Charlie     78
3      S004    David     88
```

请注意 Pandas 如何自动为行分配了默认的整数索引（0, 1, 2, 3）。字典的键 (`StudentID`、`Name`、`Score`) 成为了列标签。

您可以使用 NumPy 数组作为字典的值来达到同样的效果：

```python
# 以 NumPy 数组字典形式的数据
data_np = {
    'StudentID': np.array(['S001', 'S002', 'S003', 'S004']),
    'Name': np.array(['Alice', 'Bob', 'Charlie', 'David']),
    'Score': np.array([85, 92, 78, 88])
}

# 创建 DataFrame
df_from_dict_np = pd.DataFrame(data_np)

print(df_from_dict_np)
```

### 从字典列表创建

另一种常见模式是使用一个列表，其中每个元素都是一个字典。在这种情况下，每个字典代表结果 `DataFrame` 中的一*行*。字典中的键成为列标签。Pandas 足够灵活，可以处理某些字典可能缺少键的情况；它会用 `NaN`（非数字）填充这些位置，`NaN` 是 Pandas 表示缺失数据的默认标记 (token)。

```python
# 以字典列表形式的数据
list_of_dicts = [
    {'StudentID': 'S001', 'Name': 'Alice', 'Score': 85},
    {'StudentID': 'S002', 'Name': 'Bob', 'Age': 21}, # 注意：缺少 'Score'，多出 'Age'
    {'StudentID': 'S003', 'Name': 'Charlie', 'Score': 78, 'Age': 22},
    {'StudentID': 'S004', 'Name': 'David', 'Score': 88}
]

# 创建 DataFrame
df_from_list_dict = pd.DataFrame(list_of_dicts)

print(df_from_list_dict)
```

输出显示了 Pandas 如何处理不一致的键：

```
  StudentID     Name  Score   Age
0      S001    Alice   85.0   NaN
1      S002      Bob    NaN  21.0
2      S003  Charlie   78.0  22.0
3      S004    David   88.0   NaN
```

Pandas 从字典中存在的键推断出所有可能的列名 (`StudentID`、`Name`、`Score`、`Age`)。当特定行（字典）缺少某个键时，会插入 `NaN`。此外，请注意 `Score` 和 `Age` 列被自动分配了浮点数据类型 (`float64`)，因为 `NaN` 在技术上是一个浮点值。

### 从 NumPy 数组创建

如果您的数据已存在为二维 NumPy 数组，您可以直接将其转换为 `DataFrame`。默认情况下，Pandas 会为列和行（索引）都分配整数标签。但是，您可以使用 `columns` 和 `index` 参数 (parameter)明确提供标签。

```python
# 一个二维 NumPy 数组
np_array = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# 创建带有默认标签的 DataFrame
df_from_np_default = pd.DataFrame(np_array)
print("带有默认标签的 DataFrame:")
print(df_from_np_default)

# 创建带有自定义标签的 DataFrame
df_from_np_custom = pd.DataFrame(
    np_array,
    columns=['Col_A', 'Col_B', 'Col_C'],
    index=['Row_X', 'Row_Y', 'Row_Z']
)
print("\n带有自定义标签的 DataFrame:")
print(df_from_np_custom)
```

输出:

```
DataFrame with default labels:
   0  1  2
0  1  2  3
1  4  5  6
2  7  8  9

DataFrame with custom labels:
       Col_A  Col_B  Col_C
Row_X      1      2      3
Row_Y      4      5      6
Row_Z      7      8      9
```

> 图示展示了字典键和列表如何映射到 Pandas DataFrame 中的列和数据。如果未指定，行索引通常会自动生成。

### 从 Series 字典创建

您也可以从一个字典构建 `DataFrame`，其中值是 Pandas `Series` 对象。这类似于使用列表字典，数据根据 Series 的索引标签进行对齐 (alignment)。

```python
# 创建两个 Series
s1 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
s2 = pd.Series([15, 25, 35, 45], index=['a', 'b', 'c', 'd']) # 注意索引不同

# 从 Series 字典创建 DataFrame
df_from_series = pd.DataFrame({'Col1': s1, 'Col2': s2})

print(df_from_series)
```

输出:

```
   Col1  Col2
a  10.0    15
b  20.0    25
c  30.0    35
d   NaN    45
```

Pandas 会根据索引标签（'a'、'b'、'c'、'd'）自动对齐数据。由于 `s1` 没有索引 'd'，因此在 `Col1` 中为该行引入了 `NaN`。

这些方法提供了灵活的途径来程序化地创建 DataFrame。一旦您有了 DataFrame，下一步通常是检查其属性和内容，这我们将在后续内容中介绍。请记住，直接从 CSV 或 Excel 等文件加载数据也是将数据导入 DataFrame 的一种非常常见的方式，这将在第 6 章中讨论。

## 参考资料

- [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) — The pandas development team (2024)
  Pandas DataFrame构造函数的官方API参考，详细介绍了从不同数据类型创建DataFrame的各种方法。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  Pandas创建者所著的奠基性书籍，全面涵盖了DataFrame的创建和核心概念，适合深入理解。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/03.01-introducing-pandas-objects.html) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  一本易于理解的指南，提供关于Pandas DataFrame的清晰解释和实用示例，包括各种构造方法，对初学者有益。

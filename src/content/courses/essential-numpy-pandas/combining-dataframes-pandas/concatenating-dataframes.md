---
course: "essential-numpy-pandas"
chapter: "combining-dataframes-pandas"
lesson: "concatenating-dataframes"
sourceId: 1833
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-10-combining-dataframes-pandas/concatenating-dataframes"
title: "连接 DataFrames (pd.concat)"
description: "使用 pd.concat() 将 DataFrames 垂直堆叠（按行）或水平堆叠（按列）。"
order: 2
plots: []
sourceHash: "0e7123c2e1cd2f61378ccf92b350d57fec0c306a9365de6aa6cbe8e28487192c"
sourceCorrections: []
---

合并 DataFrames 最直接的方法之一是进行**连接**。可以将其想象成将纸张（您的 DataFrames）一张叠在另一张上面（追加行）或并排放置（添加列）。Pandas 提供了 `pd.concat()` 函数来实现此功能。

`pd.concat()` 函数将其主要参数 (parameter)设为 Series 或 DataFrame 对象的列表或序列，并巧妙地将它们拼接在一起。

### 沿行连接（垂直堆叠）

默认情况下，`pd.concat()` 垂直堆叠 DataFrames，这意味着它将第二个 DataFrame 的行追加到第一个的末尾，将第三个追加到第二个的末尾，依此类推。这沿 `axis=0` 进行。

让我们创建两个简单的 DataFrame 进行说明：

```python
import pandas as pd

# 创建第一个 DataFrame
df1 = pd.DataFrame({'A': ['A0', 'A1'],
                    'B': ['B0', 'B1']},
                   index=[0, 1])

# 创建第二个 DataFrame
df2 = pd.DataFrame({'A': ['A2', 'A3'],
                    'B': ['B2', 'B3']},
                   index=[2, 3])

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

# 沿行连接（默认行为）
result_rows = pd.concat([df1, df2])

print("\n连接后的 DataFrame (行):")
print(result_rows)
```

**Output:**

```
DataFrame 1:
    A   B
0  A0  B0
1  A1  B1

DataFrame 2:
    A   B
2  A2  B2
3  A3  B3

Concatenated DataFrame (Rows):
    A   B
0  A0  B0
1  A1  B1
2  A2  B2
3  A3  B3
```

可以看到，`df2` 的行直接追加在 `df1` 的行下方。列自动对齐 (alignment)，因为两个 DataFrame 具有相同的列名（'A' 和 'B'）。

### 行连接期间处理索引

请注意 `result_rows` DataFrame 中的索引：`[0, 1, 2, 3]`。在这种情况下，原始索引是唯一的并且很好地组合在一起。然而，如果原始 DataFrame 具有重叠的索引标签怎么办？

让我们修改 `df2` 使其与 `df1` 具有重叠索引：

```python
# 再次创建 df1
df1 = pd.DataFrame({'A': ['A0', 'A1'],
                    'B': ['B0', 'B1']},
                   index=[0, 1])

# 创建带有重叠索引的 df2
df2_overlap = pd.DataFrame({'A': ['A2', 'A3'],
                            'B': ['B2', 'B3']},
                           index=[1, 2]) # 索引 1 与 df1 重叠

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2 (重叠索引):")
print(df2_overlap)

# 使用重叠索引连接
result_overlap = pd.concat([df1, df2_overlap])

print("\n连接后 (重叠索引):")
print(result_overlap)
print("\n结果的索引:")
print(result_overlap.index)
```

**Output:**

```
DataFrame 1:
    A   B
0  A0  B0
1  A1  B1

DataFrame 2 (Overlapping Index):
    A   B
1  A2  B2
2  A3  B3

Concatenated (Overlapping Index):
    A   B
0  A0  B0
1  A1  B1
1  A2  B2  # 重复索引标签 1
2  A3  B3

Index of the result:
Index([0, 1, 1, 2], dtype='int64')
```

默认情况下，`pd.concat()` 会保留原始索引，即使这会导致重复（例如上面例子中的索引 `1`）。尽管 Pandas 允许重复索引，但它们有时会使通过标签选择数据（`.loc`）变得模糊，或在后续操作中导致意外行为。

如果您不需要保留原始索引，并且更喜欢为结果 DataFrame 提供一个清晰、唯一的索引，您可以使用 `ignore_index=True` 参数 (parameter)：

```python
# 连接带有重叠索引，忽略原始索引
result_ignore_index = pd.concat([df1, df2_overlap], ignore_index=True)

print("\n连接后 (忽略索引):")
print(result_ignore_index)
print("\n结果的索引 (已忽略):")
print(result_ignore_index.index)
```

**Output:**

```
Concatenated (Ignoring Index):
    A   B
0  A0  B0
1  A1  B1
2  A2  B2
3  A3  B3

Index of the result (Ignored):
RangeIndex(start=0, stop=4, step=1)
```

将 `ignore_index` 设置为 `True` 会丢弃原始索引，并为连接后的 DataFrame 分配一个新的默认整数索引（`RangeIndex`）。

### 沿列连接（水平堆叠）

您也可以通过指定 `axis=1` 将 DataFrames 并排连接。这会根据它们的索引标签对齐 (alignment) DataFrames，并将它们的列放置在彼此旁边。

让我们创建两个索引相同但列不同的 DataFrame：

```python
# 再次创建 df1
df1 = pd.DataFrame({'A': ['A0', 'A1'],
                    'B': ['B0', 'B1']},
                   index=[0, 1])

# 创建列不同但索引相同的 df3
df3 = pd.DataFrame({'C': ['C0', 'C1'],
                    'D': ['D0', 'D1']},
                   index=[0, 1])

print("DataFrame 1:")
print(df1)
print("\nDataFrame 3:")
print(df3)

# 沿列连接 (axis=1)
result_cols = pd.concat([df1, df3], axis=1)

print("\n连接后的 DataFrame (列):")
print(result_cols)
```

**Output:**

```
DataFrame 1:
    A   B
0  A0  B0
1  A1  B1

DataFrame 3:
    C   D
0  C0  D0
1  C1  D1

Concatenated DataFrame (Columns):
    A   B   C   D
0  A0  B0  C0  D0
1  A1  B1  C1  D1
```

在这里，`pd.concat` 使用索引（`0` 和 `1`）对齐行，并将 `df3` 的列放置在 `df1` 的列旁边。

下面是说明沿 `axis=0`（行）和 `axis=1`（列）连接之间区别的图表。

> 说明垂直（axis=0）和水平（axis=1）连接的图表。

### 处理对齐 (alignment)和缺失数据（连接逻辑）

当连接在*非*连接轴上不能完美对齐的 DataFrames 时会发生什么？例如，当 DataFrames 具有不同行索引时并排连接（`axis=1`），或当它们具有不同列时垂直堆叠（`axis=0`）。

这由 `join` 参数 (parameter)控制，该参数的工作原理与数据库连接类似：

- `join='outer'` (默认): 取索引（或列）的并集。它保留两个 DataFrame 中的所有标签。如果一个标签存在于一个 DataFrame 中但不存在于另一个中，则缺失值将用 `NaN`（非数字）填充。
- `join='inner'`: 取索引（或列）的交集。它只保留在*两个* DataFrame 中都存在的标签。

让我们通过列连接（`axis=1`）来查看这种情况，其中 DataFrames 具有不同的索引：

```python
# df1 的索引为 [0, 1]
print("DataFrame 1:")
print(df1)

# 创建索引不同的 df4 [1, 2]
df4 = pd.DataFrame({'C': ['C1', 'C2'],
                    'D': ['D1', 'D2']},
                   index=[1, 2])
print("\nDataFrame 4:")
print(df4)

# 使用外连接（默认）连接列
result_outer = pd.concat([df1, df4], axis=1, join='outer') # 或者直接 pd.concat([df1, df4], axis=1)
print("\n连接后的列 (外连接):")
print(result_outer)

# 使用内连接连接列
result_inner = pd.concat([df1, df4], axis=1, join='inner')
print("\n连接后的列 (内连接):")
print(result_inner)
```

**Output:**

```
DataFrame 1:
    A   B
0  A0  B0
1  A1  B1

DataFrame 4:
    C   D
1  C1  D1
2  C2  D2

Concatenated Columns (Outer Join):
      A    B    C    D
0   A0   B0  NaN  NaN
1   A1   B1   C1   D1
2  NaN  NaN   C2   D2

Concatenated Columns (Inner Join):
    A   B   C   D
1  A1  B1  C1  D1
```

- **外连接：** 结果 DataFrame 的索引为 `0`、`1` 和 `2`（`[0, 1]` 和 `[1, 2]` 的并集）。由于索引 `0` 只存在于 `df1` 中，因此 'C' 和 'D' 列在该行中为 `NaN`。类似地，由于索引 `2` 只存在于 `df4` 中，因此 'A' 和 'B' 列在该行中为 `NaN`。索引 `1` 在两者中都存在，因此所有列都有值。
- **内连接：** 结果 DataFrame 只包含索引 `1`，因为它是 `df1` 和 `df4` 中都存在的唯一索引标签。

同样的 `join` 逻辑也适用于使用具有不同列名的 DataFrames 进行行连接（`axis=0`）的情况。外连接将包含两个 DataFrame 中的所有列，并用 `NaN` 填充缺失值，而内连接将只保留两者共有的列。

使用 `pd.concat` 进行连接是将较小的数据集组合成较大数据集的基本工具。理解如何控制轴、处理索引以及通过连接管理对齐对于正确组合您的数据非常重要。

## 参考资料

- [pandas.concat](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html) — pandas development team (2024)
  `pd.concat` 函数的官方文档，提供了所有参数及其用法的详细信息。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: Chapter 8
  Pandas 创始人撰写的标准书籍，涵盖了数据组合方法（包括 `pd.concat`），并提供了实际示例和说明。
- [Combining DataFrames With Pandas Concat and Append](https://realpython.com/pandas-concat-append/) — Jonathan Sharpe (2023)
  Publisher: Real Python
  一篇清晰的教程，通过实际代码示例解释 `pd.concat`，有助于理解常见用例和参数。

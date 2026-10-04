---
course: "intermediate-python-programming-ml"
chapter: "data-manipulation-pandas"
lesson: "pandas-merging-joining"
sourceId: 2160
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-3-data-manipulation-pandas/pandas-merging-joining"
title: "合并、连接和拼接数据帧"
description: "根据特定标准，使用 merge、join 和 concat 等不同方法组合多个数据帧。"
order: 7
plots: []
sourceHash: "04d28ecbfa71431d5a4004d446eb31684a955270dae5765081507e9935df71d2"
sourceCorrections: []
---

数据分析常需整合来自不同来源的信息。无论是客户信息在一个文件，订单历史在另一个文件，还是实验结果分布在多个表格中，你需要方法来整合这些不同的数据集。Pandas 提供了功能强大且灵活的函数来合并 Series 和数据帧：`concat`、`merge` 和 `join`。了解何时以及如何使用它们是有效数据准备的基础。

本节重点介绍这些核心的数据组合操作。我们将了解简单的数据堆叠、基于共同列的数据库式连接，以及基于索引的连接。

### 拼接对象使用 `pd.concat`

`pd.concat` 函数沿着某个轴执行拼接操作。它适用于将多个 Series 或数据帧简单地垂直（按行）或水平（按列）堆叠起来。

我们来创建几个简单的数据帧：

```python
import pandas as pd

df1 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2', 'A3'],
    'B': ['B0', 'B1', 'B2', 'B3'],
    'C': ['C0', 'C1', 'C2', 'C3'],
    'D': ['D0', 'D1', 'D2', 'D3']
}, index=[0, 1, 2, 3])

df2 = pd.DataFrame({
    'A': ['A4', 'A5', 'A6', 'A7'],
    'B': ['B4', 'B5', 'B6', 'B7'],
    'C': ['C4', 'C5', 'C6', 'C7'],
    'D': ['D4', 'D5', 'D6', 'D7']
}, index=[4, 5, 6, 7])

df3 = pd.DataFrame({
    'A': ['A8', 'A9', 'A10', 'A11'],
    'B': ['B8', 'B9', 'B10', 'B11'],
    'C': ['C8', 'C9', 'C10', 'C11'],
    'D': ['D8', 'D9', 'D10', 'D11']
}, index=[8, 9, 10, 11])
```

`pd.concat` 最简单的用法是组合一个数据帧列表。默认情况下，它按行拼接（`axis=0`）：

```python
result_rows = pd.concat([df1, df2, df3])
print(result_rows)
# 输出：
#       A    B    C    D
# 0    A0   B0   C0   D0
# 1    A1   B1   C1   D1
# 2    A2   B2   C2   D2
# 3    A3   B3   C3   D3
# 4    A4   B4   C4   D4
# 5    A5   B5   C5   D5
# 6    A6   B6   C6   D6
# 7    A7   B7   C7   D7
# 8    A8   B8   C8   D8
# 9    A9   B9   C9   D9
# 10  A10  B10  C10  D10
# 11  A11  B11  C11  D11
```

请注意，原始数据帧的索引被保留。如果原始数据帧的索引值有重叠，结果数据帧也会有重复的索引值。通常，你不需要原始索引。你可以使用 `ignore_index=True` 参数 (parameter)创建一个新的默认整数索引：

```python
result_ignore_index = pd.concat([df1, df2, df3], ignore_index=True)
print(result_ignore_index)
# 输出：
#       A    B    C    D
# 0    A0   B0   C0   D0
# 1    A1   B1   C1   D1
# 2    A2   B2   C2   D2
# 3    A3   B3   C3   D3
# 4    A4   B4   C4   D4
# 5    A5   B5   C5   D5
# 6    A6   B6   C6   D6
# 7    A7   B7   C7   D7
# 8    A8   B8   C8   D8
# 9    A9   B9   C9   D9
# 10  A10  B10  C10  D10
# 11  A11  B11  C11  D11
```

你也可以通过设置 `axis=1` 来按列拼接。在这种情况下，Pandas 根据索引对齐 (alignment)数据。如果数据帧有不同的索引，结果将包含索引的并集，并在数据缺失的地方引入 `NaN` 值。

我们来创建一个数据帧，以便水平拼接：

```python
df4 = pd.DataFrame({
    'E': ['E0', 'E1', 'E2', 'E3'],
    'F': ['F0', 'F1', 'F2', 'F3']
}, index=[0, 1, 2, 3])

result_cols = pd.concat([df1, df4], axis=1)
print(result_cols)
# 输出：
#     A   B   C   D   E   F
# 0  A0  B0  C0  D0  E0  F0
# 1  A1  B1  C1  D1  E1  F1
# 2  A2  B2  C2  D2  E2  F2
# 3  A3  B3  C3  D3  E3  F3
```

如果索引不能完全对齐：

```python
df5 = pd.DataFrame({
    'G': ['G2', 'G3', 'G4', 'G5']
}, index=[2, 3, 4, 5]) # 不同于 df1 的索引

result_cols_misaligned = pd.concat([df1, df5], axis=1)
print(result_cols_misaligned)
# 输出：
#      A    B    C    D    G
# 0   A0   B0   C0   D0  NaN
# 1   A1   B1   C1   D1  NaN
# 2   A2   B2   C2   D2   G2
# 3   A3   B3   C3   D3   G3
# 4  NaN  NaN  NaN  NaN   G4
# 5  NaN  NaN  NaN  NaN   G5
```

这里，`pd.concat` 取索引的并集（0到5），并用 `NaN` 填充缺失值。

### 使用 `pd.merge` 进行数据库式合并

对于更复杂、基于共同列（键）中值的数据库式连接操作，Pandas 提供了 `pd.merge` 函数。这类似于 SQL 的 `JOIN` 操作。它根据一个或多个共享键对两个数据帧中的行进行对齐 (alignment)。

我们来设置两个数据帧进行合并：

```python
left = pd.DataFrame({
    'key': ['K0', 'K1', 'K2', 'K3'],
    'A': ['A0', 'A1', 'A2', 'A3'],
    'B': ['B0', 'B1', 'B2', 'B3']
})

right = pd.DataFrame({
    'key': ['K0', 'K1', 'K2', 'K4'], # 注意是 K4 而不是 K3
    'C': ['C0', 'C1', 'C2', 'C4'],
    'D': ['D0', 'D1', 'D2', 'D4']
})

print("左侧数据帧：\n", left)
print("\n右侧数据帧：\n", right)
# 输出：
# 左侧数据帧：
#   key   A   B
# 0  K0  A0  B0
# 1  K1  A1  B1
# 2  K2  A2  B2
# 3  K3  A3  B3

# 右侧数据帧：
#   key   C   D
# 0  K0  C0  D0
# 1  K1  C1  D1
# 2  K2  C2  D2
# 3  K4  C4  D4
```

控制合并的主要参数 (parameter)是 `how`，它指定了连接的类型：

- `inner` (默认): 使用两个数据帧中键的交集。只包含在*两个*数据帧中都存在的键。
- `outer`: 使用两个数据帧中键的并集。包含在*任一*数据帧中找到的所有键，用 `NaN` 填充缺失值。
- `left`: 只使用左侧数据帧的键。包含*左侧*数据帧的所有键，并与右侧进行匹配。
- `right`: 只使用右侧数据帧的键。包含*右侧*数据帧的所有键，并与左侧进行匹配。

默认情况下，`pd.merge` 会在两个数据帧中寻找同名列并将其用作连接键。

```python
# 内连接 (默认)
inner_merged = pd.merge(left, right, on='key')
print("\n内连接结果：\n", inner_merged)
# 输出：
# 内连接结果：
#   key   A   B   C   D
# 0  K0  A0  B0  C0  D0
# 1  K1  A1  B1  C1  D1
# 2  K2  A2  B2  C2  D2
```

结果中只显示同时存在于 `left` 和 `right` 中的键 'K0'、'K1' 和 'K2'。`left` 中的 'K3' 和 `right` 中的 'K4' 被排除。

```python
# 外连接
outer_merged = pd.merge(left, right, on='key', how='outer')
print("\n外连接结果：\n", outer_merged)
# 输出：
# 外连接结果：
#   key    A    B    C    D
# 0  K0   A0   B0   C0   D0
# 1  K1   A1   B1   C1   D1
# 2  K2   A2   B2   C2   D2
# 3  K3   A3   B3  NaN  NaN
# 4  K4  NaN  NaN   C4   D4
```

两个数据帧中的所有键都被包含。缺失值用 `NaN` 填充。

```python
# 左连接
left_merged = pd.merge(left, right, on='key', how='left')
print("\n左连接结果：\n", left_merged)
# 输出：
# 左连接结果：
#   key   A   B    C    D
# 0  K0  A0  B0   C0   D0
# 1  K1  A1  B1   C1   D1
# 2  K2  A2  B2   C2   D2
# 3  K3  A3  B3  NaN  NaN
```

`left` 数据帧的所有键（'K0'、'K1'、'K2'、'K3'）都存在。由于 'K3' 不存在于 `right` 中，因此 `right` 的列（'C'、'D'）在该行被 `NaN` 填充。

```python
# 右连接
right_merged = pd.merge(left, right, on='key', how='right')
print("\n右连接结果：\n", right_merged)
# 输出：
# 右连接结果：
#   key    A    B   C   D
# 0  K0   A0   B0  C0  D0
# 1  K1   A1   B1  C1  D1
# 2  K2   A2   B2  C2  D2
# 3  K4  NaN  NaN  C4  D4
```

`right` 数据帧的所有键（'K0'、'K1'、'K2'、'K4'）都存在。由于 'K4' 不存在于 `left` 中，因此 `left` 的列（'A'、'B'）在该行被 `NaN` 填充。

> 各种 `pd.merge` 连接类型（`how` 参数）中包含的键的可视化。除了在左连接/右连接的保留侧缺失外，共同键（K0、K1、K2）始终包含在内。根据连接类型，会包含唯一的键（左侧的 K3，右侧的 K4）。

**基于列名合并：**
如果两个数据帧中的列名不同，你可以使用 `left_on` 和 `right_on` 来指定它们：

```python
left_diff_key = pd.DataFrame({
    'lkey': ['K0', 'K1', 'K2', 'K3'],
    'A': ['A0', 'A1', 'A2', 'A3'],
    'B': ['B0', 'B1', 'B2', 'B3']
})

right_diff_key = pd.DataFrame({
    'rkey': ['K0', 'K1', 'K2', 'K4'],
    'C': ['C0', 'C1', 'C2', 'C4'],
    'D': ['D0', 'D1', 'D2', 'D4']
})

merged_diff_keys = pd.merge(left_diff_key, right_diff_key, left_on='lkey', right_on='rkey', how='inner')
print("\n不同列名合并结果：\n", merged_diff_keys)
# 输出：
# 不同列名合并结果：
#   lkey   A   B rkey   C   D
# 0   K0  A0  B0   K0  C0  D0
# 1   K1  A1  B1   K1  C1  D1
# 2   K2  A2  B2   K2  C2  D2
```

注意，结果中同时存在 'lkey' 和 'rkey' 两列。

**基于索引合并：**
你也可以通过设置 `left_index=True` 或 `right_index=True` 来基于数据帧索引而不是列进行合并。

```python
left_idx = left.set_index('key')
right_idx = right.set_index('key')

print("\n左侧带索引：\n", left_idx)
print("\n右侧带索引：\n", right_idx)

merged_on_index = pd.merge(left_idx, right_idx, left_index=True, right_index=True, how='inner')
print("\n基于索引的合并结果：\n", merged_on_index)
# 输出：
# 左侧带索引：
#       A   B
# key
# K0   A0  B0
# K1   A1  B1
# K2   A2  B2
# K3   A3  B3

# 右侧带索引：
#       C   D
# key
# K0   C0  D0
# K1   C1  D1
# K2   C2  D2
# K4   C4  D4

# 基于索引的合并结果：
#       A   B   C   D
# key
# K0   A0  B0  C0  D0
# K1   A1  B1  C1  D1
# K2   A2  B2  C2  D2
```

你甚至可以混合使用索引和列合并（`left_index=True`，`right_on='col'` 或 `left_on='col'`，`right_index=True`）。

**处理重叠列名：**
如果两个数据帧有同名列（这些列*不是*连接键），`merge` 会自动添加后缀（默认为 `_x`、`_y`）以示区分。你可以使用 `suffixes` 参数自定义这些后缀。

```python
left_overlap = pd.DataFrame({
    'key': ['K0', 'K1'],
    'Value': [1, 2]
})
right_overlap = pd.DataFrame({
    'key': ['K0', 'K1'],
    'Value': [3, 4]
})

merged_overlap = pd.merge(left_overlap, right_overlap, on='key')
print("\n重叠列名合并结果 (默认后缀)：\n", merged_overlap)
# 输出：
# 重叠列名合并结果 (默认后缀)：
#   key  Value_x  Value_y
# 0  K0        1        3
# 1  K1        2        4

merged_custom_suffix = pd.merge(left_overlap, right_overlap, on='key', suffixes=('_left', '_right'))
print("\n自定义后缀合并结果：\n", merged_custom_suffix)
# 输出：
# 自定义后缀合并结果：
#   key  Value_left  Value_right
# 0  K0           1            3
# 1  K1           2            4
```

### 使用 `DataFrame.join` 进行基于索引的连接

Pandas 数据帧也拥有一个实例方法 `join`，它提供了一种便捷的方式，主要基于索引进行合并。它也可以将调用数据帧的列连接到传入数据帧的索引。在底层，它通常使用 `pd.merge`。

使用之前基于索引的数据帧（`left_idx`、`right_idx`）：

```python
# 等同于 pd.merge(left_idx, right_idx, left_index=True, right_index=True, how='inner')
joined_inner = left_idx.join(right_idx, how='inner')
print("\n使用 DataFrame.join 进行内连接结果：\n", joined_inner)
# 输出：
# 使用 DataFrame.join 进行内连接结果：
#       A   B   C   D
# key
# K0   A0  B0  C0  D0
# K1   A1  B1  C1  D1
# K2   A2  B2  C2  D2

# 左连接是 join() 的默认行为
joined_left_default = left_idx.join(right_idx) # how='left' 是默认行为
print("\n使用 DataFrame.join 进行左连接结果 (默认)：\n", joined_left_default)
# 输出：
# 使用 DataFrame.join 进行左连接结果 (默认)：
#       A   B    C    D
# key
# K0   A0  B0   C0   D0
# K1   A1  B1   C1   D1
# K2   A2  B2   C2   D2
# K3   A3  B3  NaN  NaN

# 外连接
joined_outer = left_idx.join(right_idx, how='outer')
print("\n使用 DataFrame.join 进行外连接结果：\n", joined_outer)
# 输出：
# 使用 DataFrame.join 进行外连接结果：
#       A    B    C    D
# key
# K0   A0   B0   C0   D0
# K1   A1   B1   C1   D1
# K2   A2   B2   C2   D2
# K3   A3   B3  NaN  NaN
# K4  NaN  NaN   C4   D4
```

你也可以使用 `join` 中的 `on` 参数 (parameter)，将一个数据帧的列与另一个数据帧的索引进行连接。

```python
# 使用原始 'left' 数据帧（带列）
# 将 'left' 的列与其索引化的 'right_idx' 连接
joined_on_col = left.join(right_idx, on='key', how='inner')
print("\n数据帧列与另一个数据帧索引的连接结果：\n", joined_on_col)
# 输出：
# 数据帧列与另一个数据帧索引的连接结果：
#   key   A   B   C   D
# 0  K0  A0  B0  C0  D0
# 1  K1  A1  B1  C1  D1
# 2  K2  A2  B2  C2  D2
```

`join` 也接受 `lsuffix` 和 `rsuffix` 参数（类似于 `merge` 中的 `suffixes`），以处理非基于索引连接时的重叠列名。

### `concat`、`merge` 和 `join` 的选择

- 当你想垂直或水平堆叠数据帧，而无需基于列值进行复杂对齐 (alignment)时，使用 `pd.concat`。它非常适合组合结构相同的数据集，或根据匹配索引并排添加列。
- 使用 `pd.merge` 可以最灵活地基于共同列值或索引组合数据帧，类似于 SQL 连接。它提供了对连接类型（`inner`、`outer`、`left`、`right`）和键规范（`on`、`left_on`、`right_on`、`left_index`、`right_index`）的明确控制。这通常是组合关系型数据的主要工具。
- 使用 `DataFrame.join` 作为一个方便的简写，主要用于基于索引的连接。它默认为左连接，并且在索引到索引或列到索引的连接中，语法上可能比 `merge` 略微简洁。

掌握这些组合操作对于准备可能最初存在于不同文件或表格中的数据非常重要。通过有效地拼接、合并和连接数据帧，你可以构建进行后续分析和机器学习 (machine learning)模型训练所需的统一数据集。

## 参考资料

- [Merge, join, concatenate and compare](https://pandas.pydata.org/docs/user_guide/merging.html) — pandas development team (2024)
  官方、全面的指南，详细介绍了如何使用 `concat`、`merge` 和 `join` 函数组合 pandas DataFrames 和 Series，并提供示例。
- [Python for Data Analysis: Data Wrangling with pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — McKinney, Wes (2022)
  Publisher: O'Reilly Media
  pandas 创始人撰写的奠基性著作，提供了数据处理的广泛细节和实用示例，包括组合数据集（第三版）。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  一本广受认可的 Python 数据科学指南，清晰地解释了 pandas 中 `concat`、`merge` 和 `join` 操作并提供了示例（参见第 3.6 和 3.7 章）。

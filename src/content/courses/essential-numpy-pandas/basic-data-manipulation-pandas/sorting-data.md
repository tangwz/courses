---
course: "essential-numpy-pandas"
chapter: "basic-data-manipulation-pandas"
lesson: "sorting-data"
sourceId: 1815
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-8-basic-data-manipulation-pandas/sorting-data"
title: "数据排序"
description: "使用 `sort_index()` 和 `sort_values()` 按索引标签或列值对 DataFrame 进行排序。"
order: 8
plots: []
sourceHash: "8fbb6c842b17edd21ce4aa73818d4b1072f01b5a4bc58e4bbce1022c499c8090"
sourceCorrections: []
---

为了进行有效的数据分析，按有意义的顺序排列数据通常是必要的一步。排序允许您按从低到高、字母顺序或根据自定义标准查看数据，使模式更容易发现，特定条目更容易找到。Pandas 提供了灵活的方法，可以根据索引标签或列中的实际值对数据进行排序。

### 按索引排序

有时，您需要根据行或列标签（即索引）排列数据。如果索引表示时间序列、有序类别，或者只是需要按字母或数字顺序排列，这会特别有用。`sort_index()` 方法可处理这种情况。

我们从一个简单的 DataFrame 开始：

```python
import pandas as pd
import numpy as np

data = {'col_b': [4, 7, 1, 8, 5],
        'col_a': ['apple', 'banana', 'orange', 'apple', 'banana'],
        'col_c': [10.0, np.nan, 20.0, 10.0, 15.0]}
df = pd.DataFrame(data, index=['R3', 'R1', 'R5', 'R2', 'R4'])

print("原始 DataFrame:")
print(df)
```

```text
原始 DataFrame:
    col_b   col_a  col_c
R3      4   apple   10.0
R1      7  banana    NaN
R5      1  orange   20.0
R2      8   apple   10.0
R4      5  banana   15.0
```

请注意，行索引（`R3`、`R1`、`R5` 等）未按字母顺序排列。要根据索引标签对 DataFrame 行进行排序：

```python
df_sorted_by_index = df.sort_index()

print("\n按行索引排序的 DataFrame（升序）：")
print(df_sorted_by_index)
```

```text
按行索引排序的 DataFrame（升序）：
    col_b   col_a  col_c
R1      7  banana    NaN
R2      8   apple   10.0
R3      4   apple   10.0
R4      5  banana   15.0
R5      1  orange   20.0
```

默认情况下，`sort_index()` 按升序排序。要按降序排序，请使用 `ascending=False` 参数 (parameter)：

```python
df_sorted_by_index_desc = df.sort_index(ascending=False)

print("\n按行索引排序的 DataFrame（降序）：")
print(df_sorted_by_index_desc)
```

```text
按行索引排序的 DataFrame（降序）：
    col_b   col_a  col_c
R5      1  orange   20.0
R4      5  banana   15.0
R3      4   apple   10.0
R2      8   apple   10.0
R1      7  banana    NaN
```

您也可以通过列索引（列名）进行排序，通过指定 `axis=1`：

```python
df_sorted_by_columns = df.sort_index(axis=1)

print("\n按列索引排序的 DataFrame（升序）：")
print(df_sorted_by_columns)
```

```text
按列索引排序的 DataFrame（升序）：
     col_a  col_b  col_c
R3   apple      4   10.0
R1  banana      7    NaN
R5  orange      1   20.0
R2   apple      8   10.0
R4  banana      5   15.0
```

与许多 Pandas 操作一样，`sort_index()` 默认情况下会返回一个新的已排序 DataFrame。如果您想直接修改原始 DataFrame，请使用 `inplace=True` 参数。使用 `inplace=True` 时请务必小心，因为它会覆盖您的原始数据结构。

```python
df_copy = df.copy() # 在副本上操作以保留原始 df
df_copy.sort_index(inplace=True)

print("\n按索引就地排序后的原始 DataFrame：")
print(df_copy)
```

```text
按索引就地排序后的原始 DataFrame：
    col_b   col_a  col_c
R1      7  banana    NaN
R2      8   apple   10.0
R3      4   apple   10.0
R4      5  banana   15.0
R5      1  orange   20.0
```

### 按值排序

更常见的情况是，您会希望根据一个或多个列中的值对 DataFrame 进行排序。`sort_values()` 方法用于此目的。`sort_values()` 最重要的参数 (parameter)是 `by`，它指定了用于排序的列名（或列名列表）。

我们根据 `col_b` 中的值对原始 DataFrame `df` 进行排序：

```python
df_sorted_by_col_b = df.sort_values(by='col_b')

print("\n按 'col_b' 排序的 DataFrame（升序）：")
print(df_sorted_by_col_b)
```

```text
按 'col_b' 排序的 DataFrame（升序）：
    col_b   col_a  col_c
R5      1  orange   20.0
R3      4   apple   10.0
R4      5  banana   15.0
R1      7  banana    NaN
R2      8   apple   10.0
```

同样，默认排序顺序是升序。使用 `ascending=False` 可进行降序排序：

```python
df_sorted_by_col_b_desc = df.sort_values(by='col_b', ascending=False)

print("\n按 'col_b' 排序的 DataFrame（降序）：")
print(df_sorted_by_col_b_desc)
```

```text
按 'col_b' 排序的 DataFrame（降序）：
    col_b   col_a  col_c
R2      8   apple   10.0
R1      7  banana    NaN
R4      5  banana   15.0
R3      4   apple   10.0
R5      1  orange   20.0
```

您可以通过向 `by` 参数传递一个列名列表来通过多列进行排序。Pandas 将首先按列表中的第一列排序，然后使用第二列来处理并列情况，以此类推。

我们先按 `col_a`（按字母顺序）排序，然后对于 `col_a` 值相同的行，再按 `col_b`（按数字顺序）排序：

```python
df_sorted_by_multi = df.sort_values(by=['col_a', 'col_b'])

print("\n按 'col_a' 然后按 'col_b' 排序的 DataFrame（升序）：")
print(df_sorted_by_multi)
```

```text
按 'col_a' 然后按 'col_b' 排序的 DataFrame（升序）：
    col_b   col_a  col_c
R3      4   apple   10.0
R2      8   apple   10.0
R4      5  banana   15.0
R1      7  banana    NaN
R5      1  orange   20.0
```

请注意，带有“apple”的行在一起，并按 `col_b` 排序（4 然后 8），带有“banana”的行在一起，并按 `col_b` 排序（5 然后 7）。

当通过多列排序时，您还可以为每列指定不同的排序顺序。将布尔值列表传递给 `ascending` 参数，与传递给 `by` 的列表相对应。

我们按 `col_a` 升序和 `col_b` 降序排序：

```python
df_sorted_by_multi_mixed = df.sort_values(by=['col_a', 'col_b'], ascending=[True, False])

print("\n按 'col_a'（升序）然后按 'col_b'（降序）排序的 DataFrame：")
print(df_sorted_by_multi_mixed)
```

```text
按 'col_a'（升序）然后按 'col_b'（降序）排序的 DataFrame：
    col_b   col_a  col_c
R2      8   apple   10.0
R3      4   apple   10.0
R1      7  banana    NaN
R4      5  banana   15.0
R5      1  orange   20.0
```

现在，对于“apple”，`col_b`=8 的行排在 `col_b`=4 的行之前。对于“banana”，`col_b`=7 的行排在 `col_b`=5 的行之前。

#### 排序时处理缺失值

排序时缺失值 (`NaN`) 会怎样？默认情况下，`sort_values()` 会将 `NaN` 值放置在排序输出的*末尾*，无论排序是升序还是降序。您可以使用 `na_position` 参数控制此行为，该参数接受 `'first'` 或 `'last'`。

我们对 `col_c`（包含 `NaN`）进行排序，并明确将 `NaN` 放在首位：

```python
df_sorted_nan_first = df.sort_values(by='col_c', na_position='first')

print("\n按 'col_c' 排序的 DataFrame，NaN 优先：")
print(df_sorted_nan_first)
```

```text
按 'col_c' 排序的 DataFrame，NaN 优先：
    col_b   col_a  col_c
R1      7  banana    NaN
R3      4   apple   10.0
R2      8   apple   10.0
R4      5  banana   15.0
R5      1  orange   20.0
```

与 `sort_index()` 一样，`sort_values()` 也接受 `inplace=True` 参数以直接修改 DataFrame。

排序是一种基本操作，用于组织和理解数据。无论是按索引标签排列行，还是根据列内容排序，`sort_index()` 和 `sort_values()` 方法都提供了必要的工具，为您的 DataFrames 带来结构。

## 参考资料

- [pandas.DataFrame.sort_values](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html) — The pandas development team (2024)
  提供按列值排序DataFrame的完整细节，包括`by`、`ascending`、`inplace`和`na_position`等参数。
- [pandas.DataFrame.sort_index](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_index.html) — The pandas development team (2024)
  展示按索引标签排序DataFrame的完整信息，涵盖`axis`、`ascending`和`inplace`等参数。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  学习Pandas数据分析的参考资料，提供数据操作技巧（如DataFrame排序）的实用指南。第三版。

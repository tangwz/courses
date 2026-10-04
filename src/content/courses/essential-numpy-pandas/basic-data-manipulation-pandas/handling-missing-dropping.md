---
course: "essential-numpy-pandas"
chapter: "basic-data-manipulation-pandas"
lesson: "handling-missing-dropping"
sourceId: 1804
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-8-basic-data-manipulation-pandas/handling-missing-dropping"
title: "处理缺失数据：删除"
description: "使用 dropna() 方法移除含有缺失值的行或列。"
order: 2
plots: []
sourceHash: "5ed59133e5da2d401fd748cdb924d58807f68b49cee4f432d084ca7803367c93"
sourceCorrections: []
---

数据集经常含有空缺，在 Pandas 中表示为 `NaN`（非数字）。处理这些空缺的一个直接方法是简单地移除包含它们的行或列。这通常是一个合理的初步操作，特别是当你的数据中只有很小一部分缺失，或者某个特定行或列有太多缺失值以至于缺乏信息时。

Pandas 为此提供了 `dropna()` 方法。我们来了解它的用法。

### 删除含有缺失值的行

默认情况下，如果行中*任何*值是 `NaN`，`dropna()` 会移除整行。

请看这个示例 DataFrame：

```python
import pandas as pd
import numpy as np

data = {'col1': [1, 2, np.nan, 4, 5],
        'col2': [np.nan, 7, 8, 9, 10],
        'col3': [11, 12, 13, 14, np.nan],
        'col4': ['A', 'B', 'C', 'D', 'E']}
df = pd.DataFrame(data)

print("原始 DataFrame：")
print(df)
```

Output:

```
原始 DataFrame：
   col1  col2  col3 col4
0   1.0   NaN  11.0    A
1   2.0   7.0  12.0    B
2   NaN   8.0  13.0    C
3   4.0   9.0  14.0    D
4   5.0  10.0   NaN    E
```

现在，我们使用 `dropna()` 的默认设置：

```python
df_dropped_rows = df.dropna() # Default is axis=0 (rows) and how='any'

print("\n删除包含任何 NaN 的行后的 DataFrame：")
print(df_dropped_rows)
```

Output:

```
删除包含任何 NaN 的行后的 DataFrame：
   col1  col2  col3 col4
1   2.0   7.0  12.0    B
3   4.0   9.0  14.0    D
```

请注意，行 0、2 和 4 被移除了，因为它们都至少包含一个 `NaN` 值。只有行 1 和 3 被保留，它们在所有列中都含有完整数据。

#### 控制行如何被删除

`dropna()` 方法有参数 (parameter)，可以让你更精细地控制：

1. **`how` 参数**：

   - `how='any'`（默认）：如果行中存在*任何* `NaN` 值，则删除该行。
   - `how='all'`：仅当该行中的*所有*值都是 `NaN` 时，才删除该行。

   我们创建一个 DataFrame，其中有一行完全是 `NaN`：

   ```python
   data_with_all_nan = {'col1': [1, np.nan, np.nan, 4],
                        'col2': [np.nan, 7, np.nan, 9],
                        'col3': [11, 12, np.nan, 14]}
   df_all_nan = pd.DataFrame(data_with_all_nan)

   print("\n可能包含全 NaN 行的原始 DataFrame：")
   print(df_all_nan)

   df_dropped_all = df_all_nan.dropna(how='all')
   print("\n删除所有 NaN 行后的 DataFrame：")
   print(df_dropped_all)
   ```

   Output:

   ```
   可能包含全 NaN 行的原始 DataFrame：
      col1  col2  col3
   0   1.0   NaN  11.0
   1   NaN   7.0  12.0
   2   NaN   NaN   NaN
   3   4.0   9.0  14.0

   删除所有 NaN 行后的 DataFrame：
      col1  col2  col3
   0   1.0   NaN  11.0
   1   NaN   7.0  12.0
   3   4.0   9.0  14.0
   ```

   在这种情况下，当使用 `how='all'` 时，只有所有值都是 `NaN` 的行 2 被删除了。
2. **`thresh` 参数**：这允许你指定一行要保留所需的*非缺失*值的最小数量。例如，`thresh=3` 意味着一行只有在至少有 3 个有效（非 `NaN`）值时才会被保留。

   使用我们原始的 `df`：

   ```python
   # 保留至少有 3 个非 NaN 值的行
   df_thresh3 = df.dropna(thresh=3)

   print("\n保留至少有 3 个非 NaN 值的行后的 DataFrame：")
   print(df_thresh3)
   ```

   Output:

   ```
   保留至少有 3 个非 NaN 值的行后的 DataFrame：
      col1  col2  col3 col4
   0   1.0   NaN  11.0    A  # 已保留 (3 个非 NaN)
   1   2.0   7.0  12.0    B  # 已保留 (4 个非 NaN)
   2   NaN   8.0  13.0    C  # 已保留 (3 个非 NaN)
   3   4.0   9.0  14.0    D  # 已保留 (4 个非 NaN)
   4   5.0  10.0   NaN    E  # 已保留 (3 个非 NaN)
   ```

   在这里，所有行都被保留了，因为每行都至少有 3 个非缺失值。如果我们增加阈值：

   ```python
   # 保留至少有 4 个非 NaN 值的行
   df_thresh4 = df.dropna(thresh=4)

   print("\n保留至少有 4 个非 NaN 值的行后的 DataFrame：")
   print(df_thresh4)
   ```

   Output:

   ```
   保留至少有 4 个非 NaN 值的行后的 DataFrame：
      col1  col2  col3 col4
   1   2.0   7.0  12.0    B
   3   4.0   9.0  14.0    D
   ```

   现在，只保留了行 1 和行 3，因为它们是唯一具有 4 个有效值的行。

### 删除含有缺失值的列

有时，如果列中含有缺失数据，你可能希望移除整个列，特别是当一列有许多 `NaN` 值或对你的分析不重要时。你可以通过将 `axis` 参数 (parameter)设置为 `1`（或 `'columns'`）来实现此操作。

```python
# 删除包含任何 NaN 值的列
df_dropped_cols = df.dropna(axis=1) # axis=1 targets columns

print("\n删除包含任何 NaN 的列后的 DataFrame：")
print(df_dropped_cols)
```

Output:

```
删除包含任何 NaN 的列后的 DataFrame：
  col4
0    A
1    B
2    C
3    D
4    E
```

在我们的示例 `df` 中，列 `col1`、`col2` 和 `col3` 都至少包含一个 `NaN`，因此它们被删除了。只有 `col4` 被保留，因为它没有缺失值。

当应用于列时，`how` 和 `thresh` 参数的用法类似：

- `df.dropna(axis=1, how='all')` 会仅当列中*所有*值都是 `NaN` 时才删除该列。
- `df.dropna(axis=1, thresh=4)` 会仅当列中至少有 4 个非 `NaN` 值时才保留该列。

```python
# 保留至少有 4 个非 NaN 值的列
df_thresh4_cols = df.dropna(axis=1, thresh=4)

print("\n保留至少有 4 个非 NaN 值的列后的 DataFrame：")
print(df_thresh4_cols)
```

Output:

```
保留至少有 4 个非 NaN 值的列后的 DataFrame：
   col1  col2  col3 col4
0   1.0   NaN  11.0    A
1   2.0   7.0  12.0    B
2   NaN   8.0  13.0    C
3   4.0   9.0  14.0    D
4   5.0  10.0   NaN    E
```

在这种情况下，`col1`、`col2` 和 `col3` 各有 4 个非 `NaN` 值（总共 5 行），`col4` 有 5 个。由于所有都达到了 4 的阈值，因此没有列被删除。

### 就地修改 DataFrame

默认情况下，`dropna()` 返回一个*新*的 DataFrame，其中缺失值已被删除，而原始 DataFrame 保持不变。如果你想直接修改原始 DataFrame，可以使用 `inplace=True` 参数 (parameter)。

```python
df_copy = df.copy() # 创建一个副本进行修改

print("\n就地删除前的 DataFrame：")
print(df_copy)

df_copy.dropna(inplace=True) # 直接修改 df_copy

print("\n就地删除后的 DataFrame：")
print(df_copy)
```

Output:

```
就地删除前的 DataFrame：
   col1  col2  col3 col4
0   1.0   NaN  11.0    A
1   2.0   7.0  12.0    B
2   NaN   8.0  13.0    C
3   4.0   9.0  14.0    D
4   5.0  10.0   NaN    E

就地删除后的 DataFrame：
   col1  col2  col3 col4
1   2.0   7.0  12.0    B
3   4.0   9.0  14.0    D
```

请谨慎使用 `inplace=True`。由于它直接修改你的数据，除非你确定不再需要包含 `NaN` 值的原始数据，否则通常更安全的方法是将结果赋给一个新变量。

### 删除数据时的考量

删除缺失数据很简单，但它有代价：你会丢失信息。

- 删除行意味着丢失这些行中所有其他可能有用的信息。
- 删除列意味着对所有数据点完全丢失该特征。

这种策略通常最适用在以下情况：

- 缺失数据的比例非常小。
- 特定行或列绝大部分数据缺失（例如，超过 50-60% 为 NaN），并且不太可能有用。
- 你计划使用的特定分析技术完全无法处理缺失值。

在移除数据之前，务必考量其潜在影响。如果删除看起来过于严厉，下一节将介绍一个替代方案：填充缺失值。

## 参考资料

- [pandas.DataFrame.dropna](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html) — pandas development team (2024)
  `dropna`方法的官方文档，详细说明了其参数、行为以及从DataFrame中删除缺失数据的各种方式。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本使用 Pandas 的实用参考书，其中有专门章节介绍缺失数据处理，包括删除行/列及其原因。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  本书包含数据预处理和清洗的章节，将缺失数据处理置于更广泛的机器学习工作流中。
- [Data Mining: Concepts and Techniques](https://www.elsevier.com/books/data-mining-concepts-and-techniques/han/978-0-12-381479-1) — Jiawei Han, Micheline Kamber, Jian Pei (2011)
  Publisher: Morgan Kaufmann; DOI: [10.1016/C2009-0-61819-5](https://doi.org/10.1016/C2009-0-61819-5)
  一本学术教材，其中有数据预处理章节，解释了缺失数据的理论背景和各种方法，包括消除法。

---
course: "essential-numpy-pandas"
chapter: "combining-dataframes-pandas"
lesson: "joining-dataframes-join"
sourceId: 1843
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-10-combining-dataframes-pandas/joining-dataframes-join"
title: "基于索引的合并 (.join)"
description: "使用便捷的 .join() 方法，主要基于DataFrame索引进行合并。"
order: 6
plots: []
sourceHash: "2b4e1e86ff7c22f139ea872d24a51e7cb65b584984fb8d506774f06006e8f040"
sourceCorrections: []
---

Pandas提供了 `pd.merge` 方法，可用于组合DataFrame，它基于列或索引，提供了强大的、类似SQL的灵活性。此外，Pandas还提供了 `.join()` 方法，专门为根据索引标签组合DataFrame，或将索引与列合并而优化。此方法是执行特定类型合并的高效方式，尤其是索引上的左连接，这十分普遍。

### 基本的基于索引的合并

默认情况下，`DataFrame.join()` 尝试使用其索引与另一个DataFrame（或多个DataFrame）合并。除非另行指定，它会执行左连接，这意味着它保留了调用DataFrame（“左侧”的那个）的所有行，并根据索引包含“右侧”DataFrame的匹配数据。

我们来设置两个简单的DataFrame以作说明：

```python
import pandas as pd

# 左侧DataFrame
left_df = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                        'B': ['B0', 'B1', 'B2']},
                       index=['K0', 'K1', 'K2'])

# 右侧DataFrame
right_df = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                         'D': ['D0', 'D2', 'D3']},
                        index=['K0', 'K2', 'K3'])

print("左侧DataFrame:")
print(left_df)
print("\n右侧DataFrame:")
print(right_df)

# 将 left_df 与 right_df 根据它们的索引合并（默认：左连接）
joined_df = left_df.join(right_df)

print("\n合并后的DataFrame（索引上的左连接）:")
print(joined_df)
```

**Output:**

```
Left DataFrame:
     A   B
K0  A0  B0
K1  A1  B1
K2  A2  B2

Right DataFrame:
     C   D
K0  C0  D0
K2  C2  D2
K3  C3  D3

Joined DataFrame (left join on index):
     A   B    C    D
K0  A0  B0   C0   D0
K1  A1  B1  NaN  NaN
K2  A2  B2   C2   D2
```

请注意，结果 `joined_df` 保留了 `left_df` 的所有索引标签（`K0`、`K1`、`K2`）。对于 `K0` 和 `K2`，它们存在于两个DataFrame的索引中，`right_df` 的相应值（`C0`、`D0`、`C2`、`D2`）被包含进来。对于仅在 `left_df` 中的 `K1`，`right_df` 的列（`C` 和 `D`）被 `NaN`（非数字）填充，表示缺失数据。来自 `right_df` 的索引 `K3` 未被包含，因为它不在 `left_df` 中，并且我们执行了左连接。

### 列上的索引合并

`.join()` 方法不限于索引对索引的合并。您可以使用 `on` 参数 (parameter)将调用DataFrame（左侧）的索引与传入DataFrame（右侧）中的一列或多列进行合并。

```python
# 右侧DataFrame，使用列而非索引
right_df_on_col = pd.DataFrame({'key': ['K0', 'K2', 'K3'],
                                'C': ['C0', 'C2', 'C3'],
                                'D': ['D0', 'D2', 'D3']})

print("左侧DataFrame:")
print(left_df)
print("\n右侧DataFrame:")
print(right_df_on_col)

# 将 left_df 的索引与 right_df_on_col 的列合并
joined_on_col = left_df.join(right_df_on_col.set_index('key'), on=None)

# 或者，更直接地使用 .join 的 'on' 参数（作用于*另一个*DataFrame）：
# 这在 .join() 的使用中不太常见，pd.merge 在这里通常更清晰。
# 但是，如果您在 right_df_on_col 上调用 join：
# joined_on_col = right_df_on_col.join(left_df, on='key') # 将右侧的 'key' 列与左侧的索引合并

# 使用 .join 实现此目的的最常见模式是首先设置索引：
joined_on_col_idiom = left_df.join(right_df_on_col.set_index('key'))

print("\n合并后的DataFrame（左侧索引与右侧列合并）:")
print(joined_on_col_idiom)
```

**Output:**

```
Left DataFrame:
     A   B
K0  A0  B0
K1  A1  B1
K2  A2  B2

Right DataFrame (with column):
  key   C   D
0  K0  C0  D0
1  K2  C2  D2
2  K3  C3  D3

Joined DataFrame (left index on right column):
     A   B    C    D
K0  A0  B0   C0   D0
K1  A1  B1  NaN  NaN
K2  A2  B2   C2   D2
```

在此示例中，我们首先在 `right_df_on_col` 上使用了 `set_index('key')`，将 `key` 列设为其索引，从而允许与 `left_df` 进行标准的索引对索引合并。这是使用 `.join()` 时一个很常见的模式。结果与第一个示例相同，因为逻辑变得一致：基于匹配的索引值（`left_df` 中的 `K0`、`K1`、`K2` 与 `right_df_on_col` 的新索引）进行合并。

虽然 `left_df.join(other, on='col_in_other')` 是*可能*的，但它通常不如使用 `pd.merge(left_df, other, left_index=True, right_on='col_in_other')` 或上面所示的 `set_index` 方法易读。`.join()` 的主要优点在于其在基于索引的操作方面的简洁性。

### 使用 `how` 指定合并类型

正如 `pd.merge` 一样，`.join()` 方法接受 `how` 参数 (parameter)来控制连接类型。选项有：

- `'left'`: (默认) 保留左侧DataFrame的所有键。
- `'right'`: 保留右侧DataFrame的所有键。
- `'outer'`: 保留两个DataFrame的所有键。
- `'inner'`: 只保留在*两个*DataFrame中都找到的键。

我们来看看这些如何影响使用原始 `left_df` 和 `right_df` 的结果：

```python
# 内连接
inner_join = left_df.join(right_df, how='inner')
print("\n内连接（两个DataFrame中都有的键）:")
print(inner_join)

# 外连接
outer_join = left_df.join(right_df, how='outer')
print("\n外连接（任一DataFrame中有的键）:")
print(outer_join)

# 右连接
right_join = left_df.join(right_df, how='right')
print("\n右连接（右侧DataFrame中有的键）:")
print(right_join)
```

**Output:**

```
Inner Join (keys in both):
     A   B   C   D
K0  A0  B0  C0  D0
K2  A2  B2  C2  D2

Outer Join (keys in either):
      A    B    C    D
K0   A0   B0   C0   D0
K1   A1   B1  NaN  NaN
K2   A2   B2   C2   D2
K3  NaN  NaN   C3   D3

Right Join (keys in right):
      A    B   C   D
K0   A0   B0  C0  D0
K2   A2   B2  C2  D2
K3  NaN  NaN  C3  D3
```

请看差异：

- `inner`：只有 `K0` 和 `K2` 出现，因为它们是 `left_df` 和 `right_df` 中都存在的唯一索引标签。
- `outer`：两个DataFrame中的所有唯一索引标签（`K0`、`K1`、`K2`、`K3`）都存在。原始DataFrame中缺少数据的地方用 `NaN` 填充。
- `right`：`right_df` 的所有索引标签（`K0`、`K2`、`K3`）都被保留。`left_df` 的数据在索引匹配时被包含；否则，使用 `NaN`（如 `K3` 所示）。

这是说明左连接的图示：

> 表示一个 `left_df.join(right_df)` 操作。左侧表的所有索引键都被保留。右侧表的匹配键会带入其数据；不匹配的键则导致右侧表的列出现 NaN 值。

### 处理重叠的列名

如果您要合并的DataFrame拥有同名列（除了索引或正在合并的列之外），`.join()` 会引发错误。为解决此问题，您可以使用 `lsuffix` 和 `rsuffix` 参数 (parameter)，分别在左侧和右侧DataFrame的重叠列名后添加区分后缀。

```python
# 另一个右侧DataFrame，但有一个冲突的列名 'B'
right_overlap = pd.DataFrame({'B': ['B_r0', 'B_r2', 'B_r3'], # 重叠列 'B'
                              'D': ['D0', 'D2', 'D3']},
                             index=['K0', 'K2', 'K3'])

print("左侧DataFrame:")
print(left_df)
print("\n带重叠的右侧DataFrame:")
print(right_overlap)

# 尝试不使用后缀进行合并（会引发错误）
# joined_overlap_error = left_df.join(right_overlap) # 引发 ValueError

# 使用后缀解决重叠
joined_overlap_fixed = left_df.join(right_overlap, lsuffix='_left', rsuffix='_right')

print("\n带后缀的合并DataFrame:")
print(joined_overlap_fixed)
```

**Output:**

```
Left DataFrame:
     A   B
K0  A0  B0
K1  A1  B1
K2  A2  B2

Right DataFrame with Overlap:
     B    D
K0  B_r0   D0
K2  B_r2   D2
K3  B_r3   D3

Joined DataFrame with Suffixes:
     A B_left B_right    D
K0  A0     B0    B_r0   D0
K1  A1     B1     NaN  NaN
K2  A2     B2    B_r2   D2
```

正如您所见，`left_df` 中原始的 `B` 列现在变为 `B_left`，而 `right_overlap` 中的 `B` 列在最终结果中现在变为 `B_right`。

### 合并多个DataFrame

您可以通过向 `.join()` 传递一个DataFrame列表，一次性合并两个以上的DataFrame。合并操作根据索引从左到右依次进行。

```python
# 第三个DataFrame
other_df = pd.DataFrame({'E': ['E1', 'E2', 'E4']}, index=['K1', 'K2', 'K4'])

print("左侧DF:")
print(left_df)
print("\n右侧DF:")
print(right_df)
print("\n其他DF:")
print(other_df)

# 将 left_df 与 right_df 和 other_df 合并（默认：左连接）
multi_join = left_df.join([right_df, other_df])
print("\n多DataFrame合并结果（左连接）:")
print(multi_join)

# 外连接示例
multi_join_outer = left_df.join([right_df, other_df], how='outer')
print("\n多DataFrame合并结果（外连接）:")
print(multi_join_outer)
```

**Output:**

```
Left DF:
     A   B
K0  A0  B0
K1  A1  B1
K2  A2  B2

Right DF:
     C   D
K0  C0  D0
K2  C2  D2
K3  C3  D3

Other DF:
     E
K1  E1
K2  E2
K4  E4

Multi-Join Result (left joins):
     A   B    C    D    E
K0  A0  B0   C0   D0  NaN
K1  A1  B1  NaN  NaN   E1
K2  A2  B2   C2   D2   E2

Multi-Join Result (outer joins):
      A    B    C    D    E
K0   A0   B0   C0   D0  NaN
K1   A1   B1  NaN  NaN   E1
K2   A2   B2   C2   D2   E2
K3  NaN  NaN   C3   D3  NaN
K4  NaN  NaN  NaN  NaN   E4
```

默认的左连接只保留初始DataFrame（`left_df`）的索引。外连接组合所有涉及的DataFrame的所有索引。

总而言之，`DataFrame.join()` 提供了一种简洁的语法，主要用于基于索引的合并。虽然 `pd.merge` 提供了更通用的合并能力，但当您的组合逻辑主要依赖于DataFrame索引时，`.join()` 通常更快、更易读，特别是对于执行左连接的常见情况。请记住，`.join()` 内部使用 `pd.merge`，所以底层逻辑是一致的。

## 参考资料

- [DataFrame.join](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html) — pandas development team (2023)
  Pandas `DataFrame.join` 方法的官方文档，说明其参数、行为和示例。
- [pandas.merge](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge.html) — pandas development team (2023)
  Pandas 通用 `pandas.merge` 函数的官方文档，它是 `DataFrame.join` 的基础，并提供更广泛的数据组合功能。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data-analysis/9781098104030/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  第三版。Pandas 创建者撰写的综合指南，提供数据操作的基础知识，包含对 DataFrame 连接和合并操作的详细讨论。

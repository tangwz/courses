---
course: "essential-numpy-pandas"
chapter: "introduction-to-pandas"
lesson: "inspecting-dataframes"
sourceId: 1758
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-5-introduction-to-pandas/inspecting-dataframes"
title: "检查 DataFrames"
description: "使用 head()、tail()、info()、describe()、shape 和 columns 等方法查看 DataFrames。"
order: 6
plots: []
sourceHash: "4a35b225d0c7cafb98ba957950ec6acb7904b9160e93d55cd020da329caf44cf"
sourceCorrections: []
---

在进行任何数据操作或分析之前，了解 Pandas DataFrame 的结构和内容是首要步骤。必须了解 DataFrame 的大小、每列的数据类型以及是否存在缺失值。Pandas 提供了多种便捷的属性和方法，可以快速检查 DataFrame 并回答这些基础问题。

### 获取快速预览: head() 和 tail()

通常，您不需要一次性查看整个数据集，特别是当它有成千上万或数百万行时。`head()` 和 `tail()` 方法非常适合快速查看数据的结构和内容。

- `head()`: 返回 DataFrame 的前 *n* 行。默认情况下，它返回前 5 行。
- `tail()`: 返回 DataFrame 的后 *n* 行。默认情况下，它返回后 5 行。

```python
# 假设 'df' 是您的 DataFrame
# 显示前 5 行
print(df.head())

# 显示前 3 行
print(df.head(n=3)) # or df.head(3)

# 显示后 5 行
print(df.tail())

# 显示后 2 行
print(df.tail(n=2)) # or df.tail(2)
```

使用 `head()` 和 `tail()` 对于验证数据是否正确加载以及初步了解列名及其包含的值的类型很有用。

### 检查维度: shape

要准确查明您的 DataFrame 有多少行和列，请使用 `shape` 属性。它不需要括号 `()`，因为它是一个属性（对象的特征）而不是一个方法（一个动作）。

```python
# 获取维度（行数，列数）
dimensions = df.shape
print(f"DataFrame 维度: {dimensions}")
print(f"行数: {dimensions[0]}")
print(f"列数: {dimensions[1]}")
```

`shape` 属性返回一个元组，其中第一个元素是行数，第二个是列数。了解维度对于理解数据集的规模是基本的。

### 了解数据类型和非空计数: info()

`info()` 方法提供 DataFrame 的简洁概览。这是最有用的检查方法之一。它会告知您：

- 总行数（条目数）。
- 索引的类型。
- 每列的名称。
- 每列中非空（非缺失）值的数量。
- 每列的数据类型 (`dtype`)。
- DataFrame 的大致内存使用情况。

```python
# 显示 DataFrame 的简洁概览
df.info()
```

运行 `df.info()` 可能会产生如下输出：

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype
---  ------        --------------  -----
 0   sepal_length  150 non-null    float64
 1   sepal_width   150 non-null    float64
 2   petal_length  148 non-null    float64
 3   petal_width   150 non-null    float64
 4   species       150 non-null    object
dtypes: float64(4), object(1)
memory usage: 6.0+ KB
```

仔细查看“非空计数”列。将此计数与总条目数（本例中为 150）进行比较，是快速识别包含缺失数据的列（例如 `petal_length` 有 148 个非空值，表示有 2 个缺失值）的方法。它还清楚地显示了 Pandas 为每列推断的数据类型（`float64` 用于带小数的数字，`object` 通常用于字符串）。

### 获取摘要统计信息: describe()

对于包含数值数据的列，`describe()` 方法会计算几种常用摘要统计信息。这为您提供了数据分布的快速量化 (quantization)概览。

```python
# 为数值列生成描述性统计信息
summary_stats = df.describe()
print(summary_stats)
```

输出通常包括：

- `count`: 非空值的数量。
- `mean`: 平均值。
- `std`: 标准差，衡量数据的分散程度。
- `min`: 最小值。
- `25%`: 第一四分位数（百分位数）。
- `50%`: 中位数或第二四分位数。
- `75%`: 第三四分位数。
- `max`: 最大值。

```
       sepal_length  sepal_width  petal_length  petal_width
count    150.000000   150.000000    148.000000   150.000000
mean       5.843333     3.057333      3.758108     1.199333
std        0.828066     0.435866      1.765298     0.762238
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
```

默认情况下，`describe()` 只包含数值列。您可以修改其行为以包含其他类型：

- `df.describe(include='object')`: 显示 `object` 数据类型（通常是字符串）列的摘要统计信息。这包括计数（`count`）、唯一值的数量（`unique`）、最常出现的值（`top`）及其频率（`freq`）。
- `df.describe(include='all')`: 尝试包含所有列的统计信息，在适当的情况下混合数值和对象摘要。

### 列出列名: columns

要仅获取列的名称，请使用 `columns` 属性。

```python
# 获取列标签
column_names = df.columns
print(column_names)
```

这会返回一个包含列名的 `Index` 对象。如果您需要检查列名的确切拼写或以编程方式遍历列，这会很有用。

```
Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'], dtype='object')
```

### 列出索引标签: index

类似地，`index` 属性会为您提供行的标签。

```python
# 获取行标签（索引）
row_labels = df.index
print(row_labels)
```

这会返回另一个 `Index` 对象。默认情况下，DataFrame 会获得一个 `RangeIndex`（从 0 开始的整数），但索引也可以是其他类型，例如日期或特定标签，您将在以后遇到。

```
RangeIndex(start=0, stop=150, step=1)
```

这些检查工具（`head`、`tail`、`shape`、`info`、`describe`、`columns`、`index`）是您处理新数据集时的首选工具。它们帮助您快速定位，理解基本结构和内容，并在进行更复杂的数据清理、转换或分析任务之前，识别潜在问题，例如缺失数据或不正确的数据类型。请养成在创建或加载 DataFrame 时使用它们的习惯。

## 参考资料

- [User Guide - Intro to data structures](https://pandas.pydata.org/docs/user_guide/dsintro.html) — pandas development team (2024)
  介绍DataFrame和Series等核心数据结构，解释其基本属性和方法。
- [User Guide - Essential basic functionality](https://pandas.pydata.org/docs/user_guide/basics.html) — pandas development team (2024)
  直接涵盖了所讨论的大部分检查方法，例如head()、tail()、info()、describe()和shape。
- [Python for Data Analysis](https://wesmckinney.com/book/pandas-basics) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  由Pandas创建者撰写的权威书籍，提供了DataFrame检查和操作的全面说明。(第三版)

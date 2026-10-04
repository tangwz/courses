---
course: "essential-numpy-pandas"
chapter: "basic-data-manipulation-pandas"
lesson: "cleaning-modifying-practice"
sourceId: 1817
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-8-basic-data-manipulation-pandas/cleaning-modifying-practice"
title: "实践练习：清理和修改数据框"
description: "应用处理缺失数据、添加/移除列、重命名和排序的技巧。"
order: 9
plots: []
sourceHash: "73ff6de52f2197acc39a82d03fc8b54ab89eaad3e787f2748bf2abde0198208e"
sourceCorrections: []
---

让我们将本章讨论的技术付诸实践。我们将依照一个常见的流程来操作：找出原始数据集中的问题，清理这些问题，并调整其结构，使其更适合分析。

### 建立示例数据框

我们将建立一个小数据框来表示产品信息，故意包含一些常见问题，例如缺失值和不一致的命名。请确保已导入 Pandas（通常使用 `import pandas as pd`）和 NumPy（通常使用 `import numpy as np`）。

```python
import pandas as pd
import numpy as np

# 建立初始数据框
data = {'ProductID': ['P101', 'P102', 'P103', 'P104', 'P105', 'P106'],
        'Category': ['A', 'B', 'A', 'C', 'B', np.nan],
        'Sales': [150, 200, np.nan, 300, 250, 180],
        'Inventory Count': [25, 0, 10, 5, np.nan, 15],
        'Region': ['North', 'South', 'North', 'West', 'South', 'East']}
df_store = pd.DataFrame(data)

print("原始数据框:")
print(df_store)
```

此数据框包含产品信息，包括它们的类别、销售数字、库存数量和区域。请留意 `np.nan` 值，它们表示缺失数据。

### 找出缺失数据

在处理缺失数据之前，我们需要找出它们。`isnull()` 方法会返回一个布尔型数据框，其中 `True` 表示数据缺失。将它与 `sum()` 结合使用，可以得出每列缺失值的数量。

```python
# 检查缺失值
print("\n每列的缺失值数量:")
print(df_store.isnull().sum())
```

输出:

```
Missing values per column:
ProductID          0
Category           1
Sales              1
Inventory Count    1
Region             0
dtype: int64
```

这说明在 `Category`、`Sales` 和 `Inventory Count` 列中各有一个缺失值。

### 处理缺失数据

我们有几种处理 `NaN` 值的方法。

#### 选项1：删除含有缺失值的行

如果我们认为任何含有缺失值的行都无法使用，我们可以使用 `dropna()` 删除这些行。

```python
# 删除含有任何 NaN 值的行
df_dropped_rows = df_store.dropna()

print("\n删除含 NaN 值的行后的数据框:")
print(df_dropped_rows)
```

请注意，索引为 2、4 和 5 的行（分别对应 `Sales`、`Inventory Count` 和 `Category` 缺失的情况）已被移除。采用这种方法时要小心，因为如果缺失值很普遍，删除行可能导致大量数据丢失。

#### 选项2：填充缺失数据（填充处理）

通常，填充（或估算）缺失值是更好的做法。我们可以用固定值或计算值来填充。

让我们再次使用原始的 `df_store`。我们可能会决定用 'Unknown' 这样的占位符来填充缺失的 `Category`，并用各自列的平均值来填充缺失的 `Sales` 和 `Inventory Count`。

```python
# 建立一个副本，避免在此步骤直接修改原始 df_store
df_filled = df_store.copy()

# 用 'Unknown' 填充缺失的 Category
df_filled['Category'].fillna('Unknown', inplace=True)

# 计算平均销售额（排除 NaN）并填充缺失的 Sales
mean_sales = df_filled['Sales'].mean()
df_filled['Sales'].fillna(mean_sales, inplace=True)

# 计算平均库存（排除 NaN）并填充缺失的 Inventory Count
mean_inventory = df_filled['Inventory Count'].mean()
df_filled['Inventory Count'].fillna(mean_inventory, inplace=True)

print("\n填充 NaN 值后的数据框:")
print(df_filled)
print("\n填充后的缺失值:")
print(df_filled.isnull().sum())
```

`inplace=True` 参数 (parameter)会直接修改数据框。现在，所有缺失值都已替换。用平均值填充是一种常见方法，但最佳策略取决于具体的数据和情境。有时，用中位数或众数填充可能更合适，特别是当数据存在异常值或为分类数据时。

### 删除列或行

有时，整个列或特定的行是不相关或存在问题的。假设在 `df_filled` 数据框中，我们不需要 `Region` 列进行分析。

```python
# 删除 'Region' 列
df_no_region = df_filled.drop(columns=['Region'])

print("\n删除 'Region' 列后的数据框:")
print(df_no_region)
```

我们也可以根据索引标签删除行。假设产品 'P104'（索引 3）已停产，需要移除。

```python
# 删除索引为 3 的行
df_dropped_row_3 = df_no_region.drop(index=3)

print("\n删除索引为 3 的行后的数据框:")
print(df_dropped_row_3)
```

### 添加和修改列

我们通常需要根据现有数据建立新列或修改现有列。

#### 添加新列

让我们添加一个 'SalesPerInventory' 列，通过 'Sales' 除以 'Inventory Count' 来计算。如果库存为 0，我们需要处理潜在的除以零问题。

```python
# 在此示例中，使用删除行 3 之前的数据框
df_to_modify = df_no_region.copy()

# 添加新列 'SalesPerInventory'
# 用一个小数（或 NaN）替换库存 0，以避免除以零错误
df_to_modify['SalesPerInventory'] = df_to_modify['Sales'] / df_to_modify['Inventory Count'].replace(0, np.nan)
# 将所有产生的 NaN（来自除以 NaN 或零）填充为 0，假设这些情况下比率为 0
df_to_modify['SalesPerInventory'].fillna(0, inplace=True)

print("\n添加 'SalesPerInventory' 列后的数据框:")
print(df_to_modify)
```

#### 修改现有列

假设我们想将 'ProductID' 列转换为大写以保持一致性。

```python
# 将 'ProductID' 转换为大写
df_to_modify['ProductID'] = df_to_modify['ProductID'].str.upper()

print("\n'ProductID' 转换为大写后的数据框:")
print(df_to_modify)
```

这里，我们使用了 `.str` 访问器，将字符串方法 `upper()` 应用到 'ProductID' 列。

### 重命名列

列名可能不清楚，或者包含不便于编写代码的字符（例如空格）。让我们将 'Inventory Count' 重命名为更简单的名称，例如 'Stock'。

```python
# 重命名 'Inventory Count' 列
df_renamed = df_to_modify.rename(columns={'Inventory Count': 'Stock'})

print("\n列已重命名的数据框:")
print(df_renamed)
```

### 数据排序

通过排序组织数据可以使其更容易理解。

#### 按值排序

让我们将数据框按 'Sales' 降序排序。

```python
# 按 'Sales' 降序排序
df_sorted_sales = df_renamed.sort_values(by='Sales', ascending=False)

print("\n按 Sales 降序排序的数据框:")
print(df_sorted_sales)
```

我们也可以按多列排序。让我们按 'Category'（升序）排序，然后在每个类别中按 'Stock'（降序）排序。

```python
# 先按 'Category' 升序排序，再按 'Stock' 降序排序
df_sorted_multi = df_renamed.sort_values(by=['Category', 'Stock'], ascending=[True, False])

print("\n按 Category（升序）和 Stock（降序）排序的数据框:")
print(df_sorted_multi)
```

#### 按索引排序

如果需要，您也可以使用 `sort_index()` 按数据框的索引排序。

```python
# 按索引排序
df_sorted_index = df_sorted_multi.sort_index()

print("\n按索引恢复排序的数据框:")
print(df_sorted_index)
```

本次实践课详细介绍了本章中涵盖的核心数据清理和修改技术的使用：找出并处理缺失数据、添加、移除和重命名列，以及数据排序。这些是为进行有意义的分析而准备几乎任何数据集的基本步骤。随着您处理更复杂的数据，您将结合使用这些技术并查看更高级的 Pandas 功能，但这些基本知识是必要的。

## 参考资料

- [Working with missing data](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html) — The pandas development team (2023)
  Publisher: The pandas development team
  关于在Pandas DataFrame中识别、删除和填充缺失值的官方指南。
- [Essential basic functionality](https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html) — The pandas development team (2023)
  涵盖核心DataFrame操作的官方指南，例如列的添加、修改、删除、重命名和排序。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  Pandas创始人撰写的综合指南，涵盖了基本的数据清洗、转换和操作技术。

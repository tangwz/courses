---
course: "essential-numpy-pandas"
chapter: "data-selection-indexing-pandas"
lesson: "setting-dataframe-index"
sourceId: 1793
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-7-data-selection-indexing-pandas/setting-dataframe-index"
title: "设置 DataFrame 索引"
description: "将 DataFrame 的默认整数索引更改为其一列或多列。"
order: 6
plots: []
sourceHash: "3871cb42973ca9a6afbde87fbae474e0db2d90976d4dd1071c15ea38fd8791d4"
sourceCorrections: []
---

DataFrame 通常包含可用作每行唯一标识的列，例如产品 ID、用户名称、时间戳或国家代码。尽管存在通过整数位置或列名选择数据的方法，但在存在这些自然标识符时，使用默认的 `0, 1, 2, ...` 数字索引来访问特定行通常不如直接使用这些标识符直观。

Pandas 允许您将一个或多个现有列指定为 DataFrame 的索引。这能让使用 `.loc` 选择数据时更符合逻辑，并且有时能提升连接或查找等操作的性能。

### 使用 `set_index()` 方法

修改索引的主要工具是 `.set_index()` 方法。它的基本用法是指定您希望用作新索引的列（或多个列）。

让我们创建一个代表产品数据的简单 DataFrame：

```python
import pandas as pd

data = {'ProductID': ['P101', 'P102', 'P103', 'P104'],
        'ProductName': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
        'Category': ['Electronics', 'Accessories', 'Accessories', 'Electronics'],
        'Price': [1200, 25, 75, 300]}
products_df = pd.DataFrame(data)

print("Original DataFrame:")
print(products_df)
print("\nOriginal Index:")
print(products_df.index)
```

输出：

```
Original DataFrame:
  ProductID ProductName     Category  Price
0      P101      Laptop  Electronics   1200
1      P102       Mouse  Accessories     25
2      P103    Keyboard  Accessories     75
3      P104     Monitor  Electronics    300

Original Index:
RangeIndex(start=0, stop=4, step=1)
```

请注意默认的 `RangeIndex`。现在，我们将 `ProductID` 列设为索引：

```python
# 默认情况下，set_index 返回一个新的 DataFrame
products_indexed = products_df.set_index('ProductID')

print("\nDataFrame with ProductID as Index:")
print(products_indexed)
print("\nNew Index:")
print(products_indexed.index)
```

输出：

```
DataFrame with ProductID as Index:
          ProductName     Category  Price
ProductID                                
P101           Laptop  Electronics   1200
P102            Mouse  Accessories     25
P103         Keyboard  Accessories     75
P104          Monitor  Electronics    300

New Index:
Index(['P101', 'P102', 'P103', 'P104'], dtype='object', name='ProductID')
```

请注意以下几个要点：

1. `ProductID` 列不再是普通数据列；它已成为索引。
2. 索引类型现在是一个包含产品 ID 的 `Index` 对象。该索引也保留了原始列的名称（'ProductID'）。
3. 默认情况下，`.set_index()` 返回一个带有修改后索引的*新* DataFrame。原始的 `products_df` 保持不变。

### 使用新索引选择数据

设置有意义索引的主要好处是使用 `.loc` 进行数据选择时有所改进。现在，您可以使用索引标签（产品 ID）来代替整数位置：

```python
# 选择产品 P103 的行
product_info = products_indexed.loc['P103']
print("\nData for P103:")
print(product_info)

# 为产品 P101 和 P104 选择特定列
selected_products = products_indexed.loc[['P101', 'P104'], ['ProductName', 'Price']]
print("\nSelected data for P101 and P104:")
print(selected_products)
```

输出：

```
Data for P103:
ProductName    Keyboard
Category    Accessories
Price                75
Name: P103, dtype: object

Selected data for P101 and P104:
          ProductName  Price
ProductID                   
P101           Laptop   1200
P104          Monitor    300
```

这种使用 `.loc` 基于标签的选择方式，与依赖整数位置相比，通常使代码更易读，且更不易出错，尤其是在 DataFrame 之后可能被排序或过滤的情况下。请记住，无论索引标签如何，`.iloc` 仍使用整数位置 (0, 1, 2, ...)。

> DataFrame 索引从默认的 RangeIndex 变为使用 'ProductID' 列。

### `inplace` 参数 (parameter)

如果您确定要直接修改原始 DataFrame，可以使用 `inplace=True` 参数。

```python
# 创建一个副本进行原地修改
products_df_copy = products_df.copy()

print("\nDataFrame before inplace modification:")
print(products_df_copy.index)

# 直接修改 DataFrame
products_df_copy.set_index('ProductID', inplace=True)

print("\nDataFrame after inplace modification:")
print(products_df_copy.index)
# 原始 products_df 不受影响
```

输出：

```
DataFrame before inplace modification:
RangeIndex(start=0, stop=4, step=1)

DataFrame after inplace modification:
Index(['P101', 'P102', 'P103', 'P104'], dtype='object', name='ProductID')
```

使用 `inplace=True` 可以为大型 DataFrame 节省内存，因为它避免创建副本。然而，通常认为更好的做法是，尤其对于初学者，避免 `inplace` 操作。将结果赋值给新变量（或重新赋值给同一个变量，例如 `products_df = products_df.set_index('ProductID')`）能使数据流更清晰，更便于调试。

### 设置多级索引（层级索引）

您也可以将多个列设为索引，创建一个 `MultiIndex` 或层级索引。当列值的组合能够唯一标识行时，这会很有用。

我们稍微修改一下示例：

```python
data_multi = {'Category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Electronics'],
              'ProductID': ['P101', 'P102', 'P103', 'P104', 'P105'],
              'ProductName': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Webcam'],
              'Price': [1200, 25, 75, 300, 50]}
products_multi_df = pd.DataFrame(data_multi)

# 使用 Category 和 ProductID 设置一个多级索引
products_multi_indexed = products_multi_df.set_index(['Category', 'ProductID'])

print("\nDataFrame with MultiIndex:")
print(products_multi_indexed)
print("\nNew MultiIndex:")
print(products_multi_indexed.index)
```

输出：

```
DataFrame with MultiIndex:
                         ProductName  Price
Category    ProductID                      
Electronics P101              Laptop   1200
Accessories P102               Mouse     25
            P103            Keyboard     75
Electronics P104             Monitor    300
            P105              Webcam     50

New MultiIndex:
MultiIndex([('Electronics', 'P101'),
            ('Accessories', 'P102'),
            ('Accessories', 'P103'),
            ('Electronics', 'P104'),
            ('Electronics', 'P105')],
           names=['Category', 'ProductID'])
```

现在索引有了多个级别。使用 `.loc` 从多级索引 DataFrame 中选择数据通常需要提供一个索引标签元组：

```python
# 选择 ('Accessories', 'P103') 对应的行
keyboard_info = products_multi_indexed.loc[('Accessories', 'P103')]
print("\nData for ('Accessories', 'P103'):")
print(keyboard_info)

# 选择 'Electronics' 类别中的所有产品（使用部分索引）
electronics_products = products_multi_indexed.loc['Electronics']
print("\nData for 'Electronics' Category:")
print(electronics_products)
```

输出：

```
Data for ('Accessories', 'P103'):
ProductName    Keyboard
Price                75
Name: (Accessories, P103), dtype: object

Data for 'Electronics' Category:
          ProductName  Price
ProductID                   
P101           Laptop   1200
P104          Monitor    300
P105           Webcam     50
```

层级索引是一项功能强劲的特点，尽管从多级索引中选择数据可能会有比这些示例更复杂的变体。

设置合适的索引是构建数据，以便在 Pandas 中进行高效分析和选择的必要一步。在下一节中，我们将介绍如何使用 `reset_index()` 来反转此过程。

## 参考资料

- [pandas.DataFrame.set_index](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.set_index.html) — pandas development team (2023)
  set_index 方法的官方 API 文档，详细说明了其参数和行为。
- [MultiIndex / Advanced Indexing](https://pandas.pydata.org/docs/user_guide/advanced.html) — pandas development team (2023)
  Pandas 用户指南中解释层次化索引和相关操作的部分。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data-analysis/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本关于 Pandas 的全面指南，其中讨论了 DataFrame 索引、使用 .loc 进行基于标签的选择以及层次化索引。

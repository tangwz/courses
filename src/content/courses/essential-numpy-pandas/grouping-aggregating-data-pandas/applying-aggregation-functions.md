---
course: "essential-numpy-pandas"
chapter: "grouping-aggregating-data-pandas"
lesson: "applying-aggregation-functions"
sourceId: 1820
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-9-grouping-aggregating-data-pandas/applying-aggregation-functions"
title: "应用聚合函数"
description: "将 sum()、mean()、count()、size()、min()、max() 等函数应用于分组数据。"
order: 3
plots: []
sourceHash: "e5535fefecc48373e54f11f6aa6c2565604f02662a2db35917383bcf2b194aff"
sourceCorrections: []
---

应用聚合函数首先需要通过在 DataFrame 上调用 `groupby()` 方法来创建 `GroupBy` 对象。然后，下一步是对每个组应用一个函数以计算汇总统计量。这一操作是“拆分-应用-合并”模式中的“应用”部分。

Pandas 的 `GroupBy` 对象内置了多个聚合方法，它们与 Series 和 DataFrame 上的对应方法功能非常相似。这些方法会自动独立地对每个组进行操作，然后将结果合并到一个新的 Series 或 DataFrame 中。

让我们看一个简单的数据框，它表示不同产品在各个区域的销售数据：

```python
import pandas as pd
import numpy as np

# 示例数据
data = {'Region': ['North', 'South', 'North', 'South', 'East', 'East', 'North'],
        'Product': ['A', 'A', 'B', 'B', 'A', 'C', 'B'],
        'Sales': [100, 150, 200, 50, 120, 80, 180],
        'Quantity': [10, 15, 20, 5, 12, 8, 15]}
df = pd.DataFrame(data)

print("原始数据框:")
print(df)
```

```text
原始数据框:
  Region Product  Sales  Quantity
0  North       A    100        10
1  South       A    150        15
2  North       B    200        20
3  South       B     50         5
4   East       A    120        12
5   East       C     80         8
6  North       B    180        15
```

现在，让我们按“Region”对这些数据进行分组：

```python
grouped_by_region = df.groupby('Region')
```

`grouped_by_region` 对象现在包含已分离的组，但我们尚未进行任何计算。

### 常用聚合函数

您可以直接将常用聚合函数应用于 `GroupBy` 对象。Pandas 会智能地将函数应用于每个组中合适的列（通常是数值列）。

1. **求和 (`.sum()`)**：计算每个组的值的总和。

   ```python
   # 计算每个区域的总销售额和总数量
   region_totals = grouped_by_region.sum()
   print("\n每个区域的总销售额和总数量:")
   print(region_totals)
   ```

   ```text
   每个区域的总销售额和总数量:
          Sales  Quantity
   Region
   East     200        20
   North    480        45
   South    200        20
   ```

   请注意，输出是一个新的数据框，其中索引是分组键（“Region”），列是原始数据框中的数值列（“Sales”、“Quantity”），包含每个区域的总和值。非数值的“Product”列自动从求和中排除。
2. **平均值 (`.mean()`)**：计算每个组的平均值。

   ```python
   # 计算每个区域的平均销售额和平均数量
   region_means = grouped_by_region.mean(numeric_only=True)
   print("\n每个区域的平均销售额和平均数量:")
   print(region_means)
   ```

   ```text
   每个区域的平均销售额和平均数量:
             Sales  Quantity
   Region
   East   100.000000  10.000000
   North  160.000000  15.000000
   South  100.000000  10.000000
   ```

   同样，输出索引是“Region”，值代表属于每个区域的行的“Sales”和“Quantity”的平均值。
3. **计数 (`.count()`)**：计算每个组中每列的非空条目数。

   ```python
   # 计算每个区域每列的条目数
   region_counts = grouped_by_region.count()
   print("\n每个区域的条目数:")
   print(region_counts)
   ```

   ```text
   每个区域的条目数:
          Product  Sales  Quantity
   Region
   East         2      2         2
   North        3      3         3
   South        2      2         2
   ```

   这里，`count()` 包含“Product”列，因为它计算任何非缺失值，无论数据类型如何。它显示了有多少记录属于每个区域的组。
4. **大小 (`.size()`)**：返回每个组的总行数（包括空值，与 `count()` 不同）。

   ```python
   # 获取每个区域的总行数（大小）
   region_sizes = grouped_by_region.size()
   print("\n每个区域组的大小:")
   print(region_sizes)
   ```

   ```text
   每个区域组的大小:
   Region
   East     2
   North    3
   South    2
   dtype: int64
   ```

   `size()` 的输出是一个 Pandas Series，其中索引是分组键（“Region”），值是属于该组的行数。
5. **最小值 (`.min()`) 和最大值 (`.max()`)**：找出每个组中每个适用列的最小值或最大值。

   ```python
   # 找出每个区域的最小销售额
   region_min_sales = grouped_by_region['Sales'].min() # 应用于特定列
   print("\n每个区域的最小销售额:")
   print(region_min_sales)

   # 找出每个区域的最大数量
   region_max_quantity = grouped_by_region['Quantity'].max()
   print("\n每个区域的最大数量:")
   print(region_max_quantity)
   ```

   ```text
   每个区域的最小销售额:
   Region
   East      80
   North    100
   South     50
   Name: Sales, dtype: int64

   每个区域的最大数量:
   Region
   East     12
   North    20
   South    15
   Name: Quantity, dtype: int64
   ```

   在这些示例中，我们首先从 `GroupBy` 对象中选择一个特定列（`['Sales']` 或 `['Quantity']`），然后再应用聚合。这会得到一个 Series，其中索引是分组键（“Region”），值是该组所选列的最小/最大值。如果您在未首先选择列的情况下直接将 `.min()` 或 `.max()` 应用于 `GroupBy` 对象，它将计算所有适用（通常是数值）列的最小/最大值，类似于 `.sum()` 或 `.mean()`。

### 应用于特定列

正如在 `.min()` 和 `.max()` 中所见的，您可以在分组后将聚合函数应用于特定列。当您只需要某些特征的汇总时，这很有用。

```python
# 计算每个区域的总销售额
total_sales_per_region = df.groupby('Region')['Sales'].sum()
print("\n每个区域的总销售额（特定列）:")
print(total_sales_per_region)

# 计算每个产品的平均数量
avg_quantity_per_product = df.groupby('Product')['Quantity'].mean()
print("\n每个产品的平均数量:")
print(avg_quantity_per_product)
```

```text
每个区域的总销售额（特定列）:
Region
East     200
North    480
South    200
Name: Sales, dtype: int64

每个产品的平均数量:
Product
A    12.333333
B    13.333333
C     8.000000
Name: Quantity, dtype: float64
```

在聚合*之前*选择列 (`df.groupby('Region')['Sales'].sum()`) 通常比计算所有列的聚合然后选择您需要的列 (`df.groupby('Region').sum()['Sales']`) 更有效，尤其是在大型数据集上。

这些基本聚合函数（`sum`、`mean`、`count`、`size`、`min`、`max`、`std`、`var`、`median` 等）涵盖了许多常见的数据汇总任务。它们是理解 Pandas 中按组操作的基本要素。在下一节中，我们将学习如何一次应用多个聚合函数。

## 参考资料

- [Pandas User Guide: Group by: split-apply-combine](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html) — pandas development team (2023)
  这份官方文档详细解释并提供了 Pandas `groupby` 操作和聚合函数的示例。
- [Python for Data Analysis](https://learning.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  这本书由 Pandas 的创建者撰写，是 Python 数据操作的权威指南，其中包含关于 `groupby` 和聚合的专门章节（第三版）。
- [Effective Pandas: Powerful Tools for Data Manipulation](https://www.goodreads.com/book/show/59846272-effective-pandas) — Matt Harrison (2021)
  Publisher: O'Reilly Media; Pages: 497
  这本实用指南为使用 Pandas `groupby` 进行高效数据聚合和操作提供了清晰的解释和示例。

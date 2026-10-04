# 使用 groupby() 方法对数据进行分组

来源：[原文](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-9-grouping-aggregating-data-pandas/grouping-data-groupby)

[返回章节目录](README.md) · [返回课程目录](../README.md)

“分-应用-合并”策略是数据分析中一种强大的模式。此过程的第一步是根据某些条件将数据分成组。Pandas 主要通过 `groupby()` 方法实现此目的。

可以将 `groupby()` 理解为告诉 Pandas：“将此 DataFrame 的行分离到不同的‘桶’中，每个‘桶’包含在特定列（或多列）中具有相同值的行。”此操作不会立即改变 DataFrame 或显示已分离的数据。相反，它会创建一个特殊的中间对象，称为 `GroupBy` 对象。该对象包含所有必要的分组信息，并已准备好进行下一步：对每个组应用函数（例如计算总和或平均值）。

### `groupby()` 方法

基本语法很简单：你在 DataFrame 上调用 `groupby()` 方法，并传入你希望作为分组依据的列名（或列名列表）。

让我们用一个例子来说明。假设我们有一个小数据集，记录了不同产品在各个区域的销售情况。

```python
import pandas as pd

# 示例销售数据
data = {'Region': ['North', 'South', 'North', 'South', 'East', 'East', 'North'],
        'Product': ['A', 'A', 'B', 'B', 'A', 'B', 'A'],
        'Sales': [100, 150, 200, 50, 120, 80, 90]}
df = pd.DataFrame(data)

print("原始 DataFrame:")
print(df)
```

Output:

```
原始 DataFrame:
  Region Product  Sales
0  North       A    100
1  South       A    150
2  North       B    200
3  South       B     50
4   East       A    120
5   East       B     80
6  North       A     90
```

现在，让我们按照‘Region’列对这个 DataFrame 进行分组：

```python
# 按照‘Region’列进行分组
grouped_by_region = df.groupby('Region')

print("\ngroupby('Region') 的结果：")
print(grouped_by_region)
```

Output:

```
groupby('Region') 的结果：
<pandas.core.groupby.generic.DataFrameGroupBy object at 0x...>
```

注意，输出本身并不是分组后的数据。它是一个 `DataFrameGroupBy` 对象。这个对象实际上包含多个小型 DataFrame，‘Region’列中的每个唯一值（‘North’、‘South’、‘East’）对应一个。

> 将 `groupby('Region')` 应用到 DataFrame 的示意图。该方法创建一个 `GroupBy` 对象，其中包含原始 DataFrame 中每个唯一区域的子集引用。

### GroupBy 对象有何用途？

这个 `GroupBy` 对象是分-应用-合并过程的根本。虽然它本身不直接显示太多，但已准备好进行“应用”步骤。你可以对其执行各种操作，例如：

1. **聚合：** 为每个组独立计算汇总统计量（如平均值、总和、计数）。这是最常见的用法，将在下一节（“应用聚合函数”）中说明。
2. **转换：** 执行组特定的计算，并返回与原始 DataFrame 形状一致的结果。
3. **筛选：** 根据某些组级别计算丢弃整个组。
4. **迭代：** 遍历各个组（我们将在后面的“迭代分组”中看到这一点）。

目前，重要的要点是 `df.groupby('column_name')` 能根据指定列高效地分割 DataFrame，并创建一个 `GroupBy` 对象，为后续对每个组的分析打下基础。

## 参考资料

- [Group by: split-apply-combine](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html) — The pandas development team (2023)
  `groupby()`方法的官方文档，详细说明其机制和在数据聚合中的应用。
- [Hands-On Data Analysis with Pandas](https://books.google.com/books?id=i9i_DwAAQBAJ) — Stefanie Molin (2019)
  Publisher: Packt Publishing
  一本实用的资源，包含使用`groupby()`和Pandas中其他数据分析功能的实际案例。

---

[上一节](01-%E6%8B%86%E5%88%86-%E5%BA%94%E7%94%A8-%E7%BB%84%E5%90%88%E6%96%B9%E6%B3%95.md) · [下一节](03-%E5%BA%94%E7%94%A8%E8%81%9A%E5%90%88%E5%87%BD%E6%95%B0.md)

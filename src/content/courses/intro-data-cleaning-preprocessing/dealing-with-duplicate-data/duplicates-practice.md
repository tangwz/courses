---
course: "intro-data-cleaning-preprocessing"
chapter: "dealing-with-duplicate-data"
lesson: "duplicates-practice"
sourceId: 3998
sourceUrl: "https://apxml.com/zh/courses/intro-data-cleaning-preprocessing/chapter-3-dealing-with-duplicate-data/duplicates-practice"
title: "处理重复数据：实践"
description: "通过示例来查找和消除示例数据集中的重复数据。"
order: 6
plots: []
sourceHash: "0e20ec762b18e8a7113d48decea9b29641ede1719383cdcb75524b792759571b"
sourceCorrections: []
---

理解重复数据的存在及其问题至关重要，这对于有效的数据预处理是必不可少的。一个实际操作的例子演示了如何使用Python和广受欢迎的`pandas`库来查找和删除示例数据集中的重复条目。这项实践是准备数据以进行可靠分析或模型训练的一个标准步骤。

首先，假设我们已经将数据加载到pandas DataFrame中。我们来创建一个表示客户订单的小型示例DataFrame，以说明这些原理：

```python
import pandas as pd

# 包含重复数据的示例数据
data = {
    'order_id': [1, 2, 3, 1, 4, 5, 6, 7],
    'customer_id': [10, 20, 10, 10, 30, 20, 10, 40],
    'item': ['Apple', 'Banana', 'Orange', 'Apple', 'Apple', 'Banana', 'Apple', 'Grape'],
    'quantity': [5, 10, 3, 5, 2, 10, 5, 20],
    'status': ['Shipped', 'Pending', 'Shipped', 'Shipped', 'Delivered', 'Cancelled', 'Shipped', 'Pending']
}
orders_df = pd.DataFrame(data)

print("原始DataFrame：")
print(orders_df)
```

运行这段代码将显示我们的初始数据集：

```
Original DataFrame:
   order_id  customer_id    item  quantity     status
0         1           10   Apple         5    Shipped  # 原始数据
1         2           20  Banana        10    Pending
2         3           10  Orange         3    Shipped
3         1           10   Apple         5    Shipped  # 第0行的重复项
4         4           30   Apple         2  Delivered
5         5           20  Banana        10  Cancelled # 与第1行具有相同客户ID/商品
6         6           10   Apple         5    Shipped  # 第0行的重复项
7         7           40   Grape        20    Pending
```

请注意，第0、3和6行在所有列上看起来都相同。第5行与第1行具有相同的`customer_id`和`item`，但`status`不同。

### 识别完全重复的行

为了找到与前面行完全相同的行，`pandas`提供了`duplicated()`方法。它返回一个布尔Series，其中`True`表示该行是前面某行的重复项。

```python
# 识别完全重复项
duplicate_rows = orders_df.duplicated()

print("\n重复行的布尔掩码：")
print(duplicate_rows)
```

Output:

```
Boolean mask for duplicate rows:
0    False
1    False
2    False
3     True  # 这一行与第0行相同
4    False
5    False
6     True  # 这一行与第0行相同
7    False
dtype: bool
```

默认情况下，`duplicated()`会将除了第一次出现之外的所有重复项标记 (token)为重复。我们可以使用这个布尔Series来筛选DataFrame，只查看重复的行：

```python
# 显示实际的重复行
print("\n完全重复的行：")
print(orders_df[duplicate_rows])
```

Output:

```
Complete duplicate rows:
   order_id  customer_id   item  quantity   status
3         1           10  Apple         5  Shipped
6         6           10  Apple         5  Shipped
```

这证实了索引为3和6的行确实是前面某行（本例中为第0行）的完全重复项。

### 基于特定列识别重复项

有时，我们根据列的一个子集来定义重复。例如，如果`customer_id`和`item`相同，无论`order_id`或`status`如何，订单都可能被视为重复。我们可以使用`subset`参数 (parameter)来检查这一点。

```python
# 基于'customer_id'和'item'识别重复项
partial_duplicates = orders_df.duplicated(subset=['customer_id', 'item'])

print("\n部分重复项的布尔掩码（customer_id, item）：")
print(partial_duplicates)

# 显示基于子集被视为重复的行
print("\n基于'customer_id'和'item'被视为重复的行：")
print(orders_df[partial_duplicates])
```

Output:

```
Boolean mask for partial duplicates (customer_id, item):
0    False
1    False
2    False
3     True  # 与第0行具有相同的 customer_id=10, item='Apple'
4    False
5     True  # 与第1行具有相同的 customer_id=20, item='Banana'
6     True  # 与第0行具有相同的 customer_id=10, item='Apple'
7    False
dtype: bool

Rows considered duplicates based on 'customer_id' and 'item':
   order_id  customer_id    item  quantity     status
3         1           10   Apple         5    Shipped
5         5           20  Banana        10  Cancelled
6         6           10   Apple         5    Shipped
```

在这里，第3和第6行被标记 (token)，因为它们重复了第0行中的（10, 'Apple'）组合。第5行被标记，因为它重复了第1行中的（20, 'Banana'）组合。

### 删除重复行

一旦识别出重复项，使用`drop_duplicates()`方法删除它们就非常简单了。

#### 删除完全重复项

要删除所有列都相同的行，我们调用不带任何参数 (parameter)的`drop_duplicates()`。默认情况下，它会保留*第一次*出现，并删除后续的重复项。

```python
# 删除完全重复的行，保留第一次出现
orders_df_no_full_duplicates = orders_df.drop_duplicates()

print("\n删除完全重复项后的DataFrame（保留第一次出现）：")
print(orders_df_no_full_duplicates)
```

Output:

```
DataFrame after removing complete duplicates (keeping first):
   order_id  customer_id    item  quantity     status
0         1           10   Apple         5    Shipped
1         2           20  Banana        10    Pending
2         3           10  Orange         3    Shipped
4         4           30   Apple         2  Delivered
5         5           20  Banana        10  Cancelled
7         7           40   Grape        20    Pending
```

第3和第6行已被删除。

您可以使用`keep`参数来更改保留哪一行：

- `keep='first'`（默认）：保留第一次出现，删除其他。
- `keep='last'`：保留最后一次出现，删除较早的。
- `keep=False`：删除*所有*有重复的行。

使用`keep='last'`的例子：

```python
# 删除完全重复项，保留最后一次出现
orders_df_keep_last = orders_df.drop_duplicates(keep='last')

print("\n删除完全重复项后的DataFrame（保留最后一次出现）：")
print(orders_df_keep_last)
```

Output:

```
DataFrame after removing complete duplicates (keeping last):
   order_id  customer_id    item  quantity     status
1         2           20  Banana        10    Pending
2         3           10  Orange         3    Shipped
4         4           30   Apple         2  Delivered
5         5           20  Banana        10  Cancelled
6         6           10   Apple         5    Shipped  # 保留了（10, 'Apple'）的最后一次重复
7         7           40   Grape        20    Pending
```

现在，第6行被保留，而不是第0行，而第0和第3行被删除。

#### 基于特定列删除重复项

我们也可以将`subset`和`keep`与`drop_duplicates`一起使用。我们来删除`customer_id`和`item`重复的行，只保留每种组合的第一个实例。

```python
# 基于'customer_id'和'item'删除重复项，保留第一次出现
orders_df_no_partial_duplicates = orders_df.drop_duplicates(subset=['customer_id', 'item'], keep='first')

print("\n删除部分重复项后的DataFrame（customer_id, item - 保留第一次出现）：")
print(orders_df_no_partial_duplicates)
```

Output:

```
DataFrame after removing partial duplicates (customer_id, item - keeping first):
   order_id  customer_id    item  quantity    status
0         1           10   Apple         5   Shipped # 第一个（10, Apple）
1         2           20  Banana        10   Pending # 第一个（20, Banana）
2         3           10  Orange         3   Shipped
4         4           30   Apple         2 Delivered
7         7           40   Grape        20   Pending
```

第3、5和6行被删除，因为它们的`(customer_id, item)`对（`(10, 'Apple')`和`(20, 'Banana')`）已经出现过。

### 验证

删除重复项后，验证结果是一个好的习惯。您可以检查新DataFrame的维度，或者重新运行`duplicated().sum()`检查。

```python
# 验证orders_df_no_full_duplicates中没有剩余的完全重复项
remaining_duplicates = orders_df_no_full_duplicates.duplicated().sum()
print(f"\n剩余完全重复项的数量：{remaining_duplicates}")

# 验证orders_df_no_partial_duplicates中没有剩余的部分重复项
remaining_partial = orders_df_no_partial_duplicates.duplicated(subset=['customer_id', 'item']).sum()
print(f"剩余部分重复项的数量（customer_id, item）：{remaining_partial}")
```

Output:

```
Number of remaining complete duplicates: 0
Number of remaining partial duplicates (customer_id, item): 0
```

这证实了相应的`drop_duplicates`操作按预期工作。

本次实际练习说明了如何使用标准工具有效识别和管理重复数据。是检查完全重复项还是使用子集进行检查，完全取决于您的数据背景和分析目标。删除重复项可以确保每条记录都适当地用于您的发现，并防止计数虚高或模型结果出现偏差。

## 参考资料

- [pandas.DataFrame.duplicated](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.duplicated.html) — pandas development team (2023)
  pandas DataFrame中识别重复行的官方文档，包含`subset`和`keep`参数的使用说明。
- [pandas.DataFrame.drop_duplicates](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html) — pandas development team (2023)
  pandas DataFrame中移除重复行的官方文档，详细说明了如何控制保留哪些重复项。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  由pandas创建者撰写的实践指南，提供了使用Python和pandas进行数据清洗、转换和操作的丰富示例和解释。
- [Data Cleaning: A Guide to Data Quality, Data Integration, and Data Governance](https://www.sciencedirect.com/book/9780123973051/data-cleaning) — Laura Sebastian-Coleman (2013)
  Publisher: Elsevier; DOI: [10.1016/B978-0-12-397305-1.00001-3](https://doi.org/10.1016/B978-0-12-397305-1.00001-3)
  一本关于数据清洗原则和方法论的著作，讨论了数据质量的宏观背景及其在数据管理中的重要性。

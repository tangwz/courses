---
course: "applied-data-science"
chapter: "advanced-data-acquisition-preparation"
lesson: "merging-joining-datasets"
sourceId: 1849
sourceUrl: "https://apxml.com/zh/courses/applied-data-science/chapter-1-advanced-data-acquisition-preparation/merging-joining-datasets"
title: "合并与连接不同数据集"
description: "使用连接和合并操作整合来自多个来源数据的方法。"
order: 7
plots: []
sourceHash: "eca2c00317da9c308639cb3822f5b9c2692ef73bce816e87853d6b5fa5bc811a"
sourceCorrections: []
---

数据很少会存在于一张单一、格式完美的表中。通常，所需的信息分散在多个文件、数据库表或通过例如数据库连接或API调用等方法获取的数据流中。为了构建用于分析或建模的完整视图，必须有效地组合这些不同的数据集。将介绍使用合并和连接操作整合数据的方法，主要使用Pandas库提供的功能。

合并和连接操作的核心是**连接键**或**标识符列**的理念。这些是一列或多列，它们存在于您要组合的数据集中，并包含可用于将一个数据集的行与另一个数据集的行进行匹配的公共值。例如，您可能有一个包含客户人口统计信息的数据集，其中包含`customer_id`列，而另一个包含购买历史的数据集也包含`customer_id`。您可以使用此公共列将购买与客户详细信息关联起来。

### 使用`pd.merge`进行数据库风格的连接

Pandas提供了强大的`pd.merge`函数，它实现了数据库风格的连接操作。它在根据匹配键组合数据集方面提供了灵活性。

我们来看两个简单的DataFrame：

```python
import pandas as pd

# 客户数据
customers = pd.DataFrame({
    'customer_id': [101, 102, 103, 104],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'city': ['New York', 'London', 'Paris', 'London']
})

# 订单数据
orders = pd.DataFrame({
    'order_id': [1, 2, 3, 4, 5],
    'customer_id': [102, 104, 101, 102, 105], # 注意：105不在客户数据中
    'product': ['Laptop', 'Keyboard', 'Mouse', 'Monitor', 'Webcam'],
    'amount': [1200, 75, 25, 300, 50]
})

print("客户：")
print(customers)
print("\n订单：")
print(orders)
```

#### 指定连接键

当连接列在两个DataFrame中具有相同名称时，使用`on`参数 (parameter)。

```python
# 使用公共的'customer_id'列进行合并
customer_orders = pd.merge(customers, orders, on='customer_id')

print("\n合并后的数据（默认为内连接）：")
print(customer_orders)
```

如果列名不同，您可以使用`left_on`和`right_on`：

```python
# 假设'orders' DataFrame中的'customer_id'列名为'cust_id'的示例
# orders.rename(columns={'customer_id': 'cust_id'}, inplace=True)
# customer_orders_diff_names = pd.merge(customers, orders, left_on='customer_id', right_on='cust_id')
# print(customer_orders_diff_names) # 如果名称不同，输出会相似
```

#### 连接类型：`how`参数

`pd.merge`中的`how`参数控制结果中包含哪些键，它与标准的SQL连接类型相似。

1. **内连接（`how='inner'`)**：这是默认设置。它只返回连接键在*两个*DataFrame中都存在的行。在我们的示例中，`customer_id` 103（来自`customers`）和105（来自`orders`）在另一个DataFrame中没有匹配项，因此它们被排除在外。

   ```python
   inner_join = pd.merge(customers, orders, on='customer_id', how='inner')
   print("\n内连接：")
   print(inner_join)
   # 输出：包含已下订单的客户101、102、104。
   ```
2. **左连接（`how='left'`)**：返回“左”DataFrame（传入`merge`的第一个）中的所有行，以及来自右DataFrame的匹配行。如果左DataFrame中的某行在右DataFrame中没有匹配项，则右DataFrame中的列将包含`NaN`（非数字）值。

   ```python
   left_join = pd.merge(customers, orders, on='customer_id', how='left')
   print("\n左连接：")
   print(left_join)
   # 输出：包含所有客户（101-104）。客户103的订单详情为NaN。
   ```
3. **右连接（`how='right'`)**：返回“右”DataFrame中的所有行，以及来自“左”DataFrame的匹配行。如果右DataFrame中的某行在左DataFrame中没有匹配项，则左DataFrame中的列将包含`NaN`。

   ```python
   right_join = pd.merge(customers, orders, on='customer_id', how='right')
   print("\n右连接：")
   print(right_join)
   # 输出：包含所有订单。customer_id为105的订单的客户详情为NaN。
   ```
4. **外连接（`how='outer'`)**：返回键存在于左DataFrame或右DataFrame中的所有行。这是键的并集。如果键在一个DataFrame中存在而另一个中不存在，则缺失DataFrame中的列将填充`NaN`。

   ```python
   outer_join = pd.merge(customers, orders, on='customer_id', how='outer')
   print("\n外连接：")
   print(outer_join)
   # 输出：包含所有客户（101-104）和所有订单（包括customer_id 105）。数据缺失处为NaN。
   ```

以下是这些连接类型的可视化展示：

> Pandas合并`how`选项的图示。圆形表示每个表中存在的键，阴影区域表示合并结果中包含的键。

### 使用`df.join`按索引连接

虽然`pd.merge`具有高度的灵活性，但DataFrame也有一个`join`方法（`df.join()`）。它通常方便用于基于DataFrame*索引*而非列进行连接。默认情况下，`df.join`使用调用DataFrame的索引和另一个DataFrame的索引执行左连接。

```python
# 设置索引以进行演示（通常索引是具有意义的键）
customers_indexed = customers.set_index('customer_id')
orders_indexed = orders.set_index('customer_id')

# 基于索引将订单数据连接到客户数据
customer_orders_indexed = customers_indexed.join(orders_indexed, how='inner')

print("\n使用索引连接（内连接）：")
print(customer_orders_indexed)
# 输出类似于内连接，但customer_id现在是索引
```

您还可以使用`join`中的`on`参数 (parameter)，将一个DataFrame的索引与另一个DataFrame中的列连接起来。虽然`pd.merge`可以处理所有这些情况，但当主要目标是基于索引的连接时，`df.join`有时可以提供更简洁的语法。

### 使用`pd.concat`堆叠DataFrame

合并和连接是根据键中的公共值（横向，添加列）来组合DataFrame的。有时，您需要将DataFrame相互堆叠（纵向，添加行）或并排放置，而无需在键上进行匹配。这通过使用`pd.concat`来实现。

设想您收到了相同格式的新订单数据：

```python
new_orders = pd.DataFrame({
    'order_id': [6, 7],
    'customer_id': [103, 101],
    'product': ['Monitor', 'Laptop Case'],
    'amount': [250, 45]
})

# 堆叠原始订单和新订单DataFrame
all_orders = pd.concat([orders, new_orders], ignore_index=True) # ignore_index会重置索引

print("\n拼接后的订单：")
print(all_orders)
```

`pd.concat`接受一个DataFrame或Series列表。`axis`参数 (parameter)控制方向：

- `axis=0`（默认）：纵向堆叠（追加行）。列按名称对齐 (alignment)；不匹配的列将获得`NaN`。
- `axis=1`：横向堆叠（追加列）。行按索引对齐；不匹配的行将获得`NaN`。

当数据集具有相同（或相似）的结构，并且您希望将它们整体组合，而不是通过共享标识符链接特定行时，拼接是合适的。

### 处理常见问题

组合数据集时，预计可能遇到的挑战：

- **重复列名**：如果两个DataFrame具有相同的列名（连接键除外），除非您提供后缀，否则`pd.merge`将引发错误。`suffixes`参数 (parameter)是一个元组，指定要分别附加到左、右DataFrame中重叠列名的字符串。

  ```python
  # 如果客户和订单都包含'date'列：
  # merged = pd.merge(customers, orders, on='customer_id', suffixes=('_cust', '_order'))
  # 结果将包含'date_cust'和'date_order'列。
  ```
- **缺失键和`NaN`值**：如前所示，左连接、右连接和外连接会在未找到匹配项时引入`NaN`值。请准备好使用“处理缺失值的策略”一节中讨论的方法（例如，插补、删除行/列）来处理这些缺失值。连接类型的选择显著影响缺失值的引入方式。
- **数据类型不匹配**：连接键理想情况下在两个DataFrame中应具有相同的数据类型（例如，都是整数或都是字符串）。Pandas有时可能处理混合类型，但这可能导致意外行为或错误。在合并之前确保键具有一致的类型是良好的做法。`df.info()`或`df.dtypes`可以帮助检查类型。
- **多对多关系**：如果键在一个或两个表中不唯一，请注意。在非唯一键上合并可能导致匹配键的笛卡尔积，这可能导致比预期大得多的DataFrame。在合并之前了解您的数据结构。

### 选择合适的方法

- 使用`pd.merge`进行灵活的、基于公共列值的数据库风格连接。这是最通用的选项。
- 使用`df.join`作为便捷方法，尤其是在基于DataFrame索引进行连接时。
- 当您需要根据索引对齐 (alignment)纵向（添加行）或横向（添加列）堆叠DataFrame时，使用`pd.concat`，通常适用于数据集具有兼容结构但不需要通过键进行行级别匹配的情况。

成功整合来自不同来源的数据是准备数据以进行更进一步分析或机器学习 (machine learning)的基础步骤。通过掌握`pd.merge`、`df.join`和`pd.concat`，您获得构建丰富、统一数据集的能力，这些数据集是获取有意义信息所必需的。始终仔细考虑数据集之间的关系，并选择合适的连接策略（`how`参数 (parameter)），以确保结果表准确反映数据中的联系。

## 参考资料

- [Merging, joining, and concatenating](https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html) — pandas development team (2024)
  全面的用户指南，涵盖了使用merge、join和concat操作组合DataFrame的概念和实践方面。
- [pandas.merge](https://pandas.pydata.org/docs/reference/api/pandas.merge.html) — pandas development team (2024)
  pandas.merge函数的官方API参考，详细说明其参数和行为。
- [pandas.concat](https://pandas.pydata.org/docs/reference/api/pandas.concat.html) — pandas development team (2024)
  pandas.concat函数的官方API参考，提供关于堆叠和连接DataFrame或Series的详细信息。

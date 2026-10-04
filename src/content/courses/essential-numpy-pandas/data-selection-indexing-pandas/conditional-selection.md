---
course: "essential-numpy-pandas"
chapter: "data-selection-indexing-pandas"
lesson: "conditional-selection"
sourceId: 1789
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-7-data-selection-indexing-pandas/conditional-selection"
title: "条件选择 (布尔索引)"
description: "根据应用于列值的条件筛选 DataFrame 行。"
order: 5
plots: []
sourceHash: "a20bdf0d0a94e0c6ccf0e3a73a335f5b32d8135c2f265fefdf771d26b644651f"
sourceCorrections: []
---

为了根据列中的特定 *值* 筛选数据，可以使用一种称为 **布尔索引** 的方法。这种方法允许您选择符合特定条件的行，例如查找所有“年龄”列大于 30 的条目，或者“城市”列是“New York”的条目。虽然通过标签 (`.loc`) 或整数位置 (`.iloc`) 选择数据等方法提供了直接访问，但布尔索引提供了根据数据内容进行查询的灵活性。

其核心是，布尔索引涉及根据条件创建一个布尔序列（一个只包含 `True` 或 `False` 值的序列），然后使用该序列从 DataFrame 中选择行。

### 如何运作：创建布尔序列

让我们从一个简单的 DataFrame 开始。假设我们有一些关于个人的数据，包括他们的年龄：

```python
import pandas as pd
import numpy as np

data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
        'Age': [24, 27, 22, 32, 29, 18],
        'City': ['New York', 'Los Angeles', 'New York', 'Chicago', 'Los Angeles', 'Chicago'],
        'Salary': [70000, 80000, 65000, 90000, 85000, 60000]}
df = pd.DataFrame(data)

print(df)
```

输出：

```
      Name  Age         City  Salary
0    Alice   24     New York   70000
1      Bob   27  Los Angeles   80000
2  Charlie   22     New York   65000
3    David   32      Chicago   90000
4      Eve   29  Los Angeles   85000
5    Frank   18      Chicago   60000
```

现在，假设我们想找出哪些人的年龄大于 25 岁。我们可以直接将比较应用于“Age”列：

```python
condition = df['Age'] > 25
print(condition)
```

输出：

```
0    False
1     True
2    False
3     True
4     True
5    False
Name: Age, dtype: bool
```

如您所见，Pandas 对整个“Age”序列逐元素执行比较（`> 25`），生成一个新的布尔值序列。`True` 表示该行满足条件，而 `False` 表示不满足。

### 使用布尔序列进行选择

这个布尔序列就像一个过滤器。您可以将其直接传递到 DataFrame 的方括号 `[]` 中（或 `.loc` 中），以仅选择条件评估为 `True` 的行：

```python
# 选择年龄大于 25 的行
older_than_25 = df[df['Age'] > 25]
print(older_than_25)
```

输出：

```
    Name  Age         City  Salary
1    Bob   27  Los Angeles   80000
3  David   32      Chicago   90000
4    Eve   29  Los Angeles   85000
```

只返回布尔序列中对应 `True` 的行（索引 1、3 和 4）。

这对于其他类型的条件也类似适用，例如字符串比较：

```python
# 选择城市为“New York”的行
in_new_york = df[df['City'] == 'New York']
print(in_new_york)
```

输出：

```
      Name  Age      City  Salary
0    Alice   24  New York   70000
2  Charlie   22  New York   65000
```

### 组合多个条件

“数据分析通常需要同时根据多个条件进行筛选。您可以使用逻辑运算符组合布尔条件：”

- `&` : 逐元素逻辑与
- `|` : 逐元素逻辑或
- `~` : 逐元素逻辑非

**重要提示：** 组合条件时，您**必须**将每个单独的条件用括号 `()` 括起来，因为这涉及到 Python 的运算符优先级规则。使用 Python 标准的 `and`、`or`、`not` 运算符将导致错误或意外行为，因为它们作用于整个对象的真值，而非逐元素操作。

#### 示例：AND (`&`)

让我们找出居住在“Los Angeles”*且*年龄大于 25 岁的人：

```python
# 条件 1：城市是“Los Angeles”
# 条件 2：年龄 > 25
# 使用 & 组合
combined_condition_and = (df['City'] == 'Los Angeles') & (df['Age'] > 25)

print(df[combined_condition_and])
```

输出：

```
  Name  Age         City  Salary
1  Bob   27  Los Angeles   80000
4  Eve   29  Los Angeles   85000
```

只有 Bob 和 Eve 满足这两个条件。

#### 示例：OR (`|`)

现在，让我们找出居住在“Chicago”*或*薪水大于 80000 的人：

```python
# 条件 1：城市是“Chicago”
# 条件 2：薪水 > 80000
# 使用 | 组合
combined_condition_or = (df['City'] == 'Chicago') | (df['Salary'] > 80000)

print(df[combined_condition_or])
```

输出：

```
    Name  Age         City  Salary
3  David   32      Chicago   90000
4    Eve   29  Los Angeles   85000
5  Frank   18      Chicago   60000
```

David 和 Frank 住在 Chicago。Eve 不住在 Chicago，但她的薪水大于 80000。这三个人都被包括在内。

#### 示例：NOT (`~`)

要选择不满足条件的行，请使用波浪号 `~` 运算符。让我们找出所有不居住在“New York”的人：

```python
# 条件：城市是“New York”
# 使用 ~ 取反
not_in_new_york = df[~(df['City'] == 'New York')]
print(not_in_new_york)
```

输出：

```
    Name  Age         City  Salary
1    Bob   27  Los Angeles   80000
3  David   32      Chicago   90000
4    Eve   29  Los Angeles   85000
5  Frank   18      Chicago   60000
```

### 将布尔索引与 `.loc` 结合使用

布尔索引也适用于 `.loc` 访问器。这会特别实用和易读，尤其当您希望根据行条件选择特定列时：

```python
# 选择年龄大于 25 岁的人的姓名和薪水
older_names_salaries = df.loc[df['Age'] > 25, ['Name', 'Salary']]
print(older_names_salaries)
```

输出：

```
    Name  Salary
1    Bob   80000
3  David   90000
4    Eve   85000
```

在这里，`.loc` 的第一个参数 (parameter)是选择行的布尔条件，第二个参数指定了为这些行检索的列。

### 记住以下几点

- **布尔序列：** 将条件应用于 DataFrame 列会创建一个布尔序列（`True`/`False`）。
- **筛选：** 在 `[]` 或 `.loc` 中使用此布尔序列来筛选 DataFrame。
- **逻辑运算符：** 使用 `&`（与）、`|`（或）和 `~`（非）组合条件。
- **括号不可少：** 使用 `&` 或 `|` 时，务必将每个单独的条件用括号 `()` 括起来。示例：`(condition1) & (condition2)`。
- **`.loc` 结合：** 布尔数组可以非常有效地与 `.loc` 结合使用，以便根据条件同时选择行和特定列。

条件选择是数据子集划分的一种基本方法。它允许您隔离满足您条件的特定观察值，为更集中的分析和操作任务奠定基础。

## 参考资料

- [Indexing and selecting data](https://pandas.pydata.org/docs/user_guide/indexing.html#boolean-indexing) — The pandas development team (2024)
  pandas的官方文档，解释了包括布尔索引在内的多种数据选择方法。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data-analysis/9781098104030/) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: 133-145
  pandas创建者所著的关于数据操作技术的书籍，内容涵盖条件选择。
- [Pandas Cookbook: Recipes for Mastering Python's Data Analysis Library](https://www.oreilly.com/library/view/pandas-1x-cookbook-second/9781492067722/) — Matthew Harrison, Theodore Petrou (2020)
  Publisher: O'Reilly Media; Pages: 40-60
  一本基于菜谱的pandas操作指南，提供各种数据选择场景的示例。

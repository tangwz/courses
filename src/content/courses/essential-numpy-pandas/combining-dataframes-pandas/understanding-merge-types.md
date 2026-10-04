---
course: "essential-numpy-pandas"
chapter: "combining-dataframes-pandas"
lesson: "understanding-merge-types"
sourceId: 1839
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-10-combining-dataframes-pandas/understanding-merge-types"
title: "理解合并类型（连接）"
description: "了解不同连接类型：内连接、外连接、左连接、右连接，以及它们如何影响合并结果。"
order: 4
plots: []
sourceHash: "2da4e4a32e3f96a6b1485c58f57e984e43f22a1d8281c1c8b4efe65ca71fcab1"
sourceCorrections: []
---

当您使用 `pd.merge()` 组合 `DataFrame` 对象时，行匹配和最终结果中包含行的方式在很大程度上取决于您执行的合并或连接种类。这种行为由 `pd.merge()` 函数中的 `how` 参数 (parameter)控制。可以将这些合并类型视为决定保留哪些行（根据输入 `DataFrame` 对象中连接键是否匹配）的不同规定。

让我们创建两个简单的 `DataFrame` 对象来准确说明这些不同的连接类型。假设我们有一个包含员工信息的 `DataFrame`，另一个包含他们的项目分配。

```python
import pandas as pd

# 包含员工姓名的 DataFrame
employees = pd.DataFrame({
    'EmpID': [101, 102, 103, 104],
    'Name': ['Alice', 'Bob', 'Charlie', 'David']
})

# 包含项目分配的 DataFrame
projects = pd.DataFrame({
    'EmpID': [103, 104, 105, 106],
    'Project': ['Zeus', 'Apollo', 'Athena', 'Poseidon']
})

print("员工 DataFrame:")
print(employees)
print("\n项目 DataFrame:")
print(projects)
```

运行上述代码，结果如下：

```
Employees DataFrame:
   EmpID     Name
0    101    Alice
1    102      Bob
2    103  Charlie
3    104    David

Projects DataFrame:
   EmpID   Project
0    103      Zeus
1    104    Apollo
2    105    Athena
3    106  Poseidon
```

请注意，`EmpID` 103 和 104 在两个 `DataFrame` 对象中都存在。`EmpID` 101 和 102 只在 `employees` 中，而 `EmpID` 105 和 106 只在 `projects` 中。`EmpID` 列是我们的共同列，常被称为*连接键*，如果未另行指定，`pd.merge()` 将默认使用它。

### 内连接 (`how='inner'`)

内连接是 Pandas 中默认的合并类型。它组合两个 `DataFrame` 对象，只保留那些连接键（我们这里的 `EmpID`）在*左右*两个 `DataFrame` 对象中都存在的行。只在一个 `DataFrame` 对象中存在的键所对应的行将被舍弃。它本质上查找的是键的交集。

```python
# 执行内连接
inner_join_df = pd.merge(employees, projects, on='EmpID', how='inner')

print("内连接结果:")
print(inner_join_df)
```

输出结果只显示了同时存在于两个表中的员工：

```
Inner Join Result:
   EmpID     Name Project
0    103  Charlie    Zeus
1    104    David  Apollo
```

> 内连接只保留在员工和项目两个数据集中都存在的共同 `EmpID` 值（103、104）。

### 外连接 (`how='outer'`)

外连接，有时也称为全外连接，保留*两个* `DataFrame` 对象中的*所有*行。如果一个 `DataFrame` 中的某行在另一个 `DataFrame` 中没有匹配的键，另一个 `DataFrame` 中的列将填充 `NaN`（非数字）值。这种连接类型查找的是键的并集。

```python
# 执行外连接
outer_join_df = pd.merge(employees, projects, on='EmpID', how='outer')

print("外连接结果:")
print(outer_join_df)
```

结果包含了所有员工和所有项目：

```
Outer Join Result:
   EmpID     Name   Project
0  101.0    Alice       NaN
1  102.0      Bob       NaN
2  103.0  Charlie      Zeus
3  104.0    David    Apollo
4  105.0      NaN    Athena
5  106.0      NaN  Poseidon
```

请注意 Alice (101) 和 Bob (102) 的 `Project` 列中显示 `NaN`，因为它们不在 `projects` `DataFrame` 中。同样，分配给 `EmpID` 105 和 106 的项目的 `Name` 列也显示 `NaN`。

> 外连接保留两个数据集中所有 `EmpID` 值，当在另一个数据集中找不到相应匹配项时，使用 `NaN` 填充。

### 左连接 (`how='left'`)

左连接保留*左侧* `DataFrame`（即传递给 `pd.merge()` 的第一个 `DataFrame`，我们示例中的 `employees`）中的*所有*行，并包含*右侧* `DataFrame` (`projects`) 中的匹配行。如果左侧 `DataFrame` 中的键在右侧 `DataFrame` 中不存在，右侧 `DataFrame` 的列将填充 `NaN`。只在右侧 `DataFrame` 中存在的键将被舍弃。

```python
# 执行左连接
left_join_df = pd.merge(employees, projects, on='EmpID', how='left')

print("左连接结果:")
print(left_join_df)
```

这保留了所有员工信息，并在有可用项目信息时将其添加：

```
Left Join Result:
   EmpID     Name Project
0    101    Alice     NaN
1    102      Bob     NaN
2    103  Charlie    Zeus
3    104    David  Apollo
```

来自 `employees` 的所有 `EmpID`（101、102、103、104）都存在。由于 101 和 102 在 `projects` 中没有匹配项，它们的 `Project` 值显示为 `NaN`。来自 `projects` 的 `EmpID` 105 和 106 被排除在外，因为它们不在左侧的 `DataFrame` (`employees`) 中。

> 左连接保留左侧数据集（员工）中的所有 `EmpID` 值，并包含来自右侧数据集（项目）的匹配数据。右侧的非匹配项填充 `NaN`。

### 右连接 (`how='right'`)

右连接是左连接的镜像。它保留*右侧* `DataFrame` (`projects`) 中的*所有*行，并包含来自*左侧* `DataFrame` (`employees`) 的匹配行。如果右侧 `DataFrame` 中的键在左侧 `DataFrame` 中不存在，左侧 `DataFrame` 的列将填充 `NaN`。只在左侧 `DataFrame` 中存在的键将被舍弃。

```python
# 执行右连接
right_join_df = pd.merge(employees, projects, on='EmpID', how='right')

print("右连接结果:")
print(right_join_df)
```

这保留了所有项目信息，并在有可用员工信息时将其添加：

```
Right Join Result:
   EmpID     Name   Project
0  103.0  Charlie      Zeus
1  104.0    David    Apollo
2  105.0      NaN    Athena
3  106.0      NaN  Poseidon
```

来自 `projects` 的所有 `EmpID`（103、104、105、106）都存在。由于 105 和 106 在 `employees` 中没有匹配项，它们的 `Name` 值显示为 `NaN`。来自 `employees` 的 `EmpID` 101 和 102 被排除在外，因为它们不在右侧的 `DataFrame` (`projects`) 中。

> 右连接保留右侧数据集（项目）中的所有 `EmpID` 值，并包含来自左侧数据集（员工）的匹配数据。左侧的非匹配项填充 `NaN`。

选择正确的合并类型是数据准备过程中的一个重要环节。这完全取决于你在最终组合数据集中所需的信息。你是否只关心同时存在于两个来源中的条目（内连接）？你是否需要一个完整的视图，包括只存在于一个来源中的条目（外连接）？或者你主要想用一个特定数据集的信息来补充另一个数据集（左连接或右连接）？掌握这些连接类型能让你精确地控制数据如何组合。

## 参考资料

- [pandas.merge](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge.html) — pandas development team (2024)
  pd.merge()函数的官方文档，提供了其参数和合并类型的完整参考。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  Pandas数据分析的基础书籍，包含对使用合并操作组合DataFrame的全面讨论和示例。
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/03.07-merge-and-join.html) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  Python数据科学的易读指南，解释了Pandas数据操作，包括各种DataFrame合并类型的示例。

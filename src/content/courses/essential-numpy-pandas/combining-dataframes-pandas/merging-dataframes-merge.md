---
course: "essential-numpy-pandas"
chapter: "combining-dataframes-pandas"
lesson: "merging-dataframes-merge"
sourceId: 1836
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-10-combining-dataframes-pandas/merging-dataframes-merge"
title: "数据库风格的合并 (pd.merge)"
description: "根据共同的列或索引组合数据框，类似于 SQL 连接。"
order: 3
plots: []
sourceHash: "228826906c80d89ec913639b450cd4cb25aa68a9b70347ee7e6c9cbb36f8c77c"
sourceCorrections: []
---

根据一个或多个共同列或索引中的*值*来组合 DataFrames 是一个常见需求，类似于在SQL等关系型数据库中连接表。虽然使用 `pd.concat` 进行连接对于堆叠结构相似的数据集很有用，但这些基于值的组合操作需要不同的方法。Pandas 提供了 `pd.merge()` 函数来完成这些操作。

`pd.merge()` 允许你根据指定列（通常称为“键”）中的共享值来组合两个 DataFrame 的行。想象一下，这就像根据在另一个表中找到的标识符在一张表中查找信息。

### 核心思想：基于键进行合并

假设我们有两个简单的数据框：一个包含员工信息，另一个包含部门信息。

```python
import pandas as pd

# 员工数据
employees = pd.DataFrame({
    'employee_id': [101, 102, 103, 104],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'dept_id': [10, 20, 10, 30]
})

# 部门数据
departments = pd.DataFrame({
    'dept_id': [10, 20, 40],
    'dept_name': ['Engineering', 'Marketing', 'Sales']
})

print("员工数据框：")
print(employees)
print("\n部门数据框：")
print(departments)
```

**Output:**

```
Employees DataFrame:
   employee_id     name  dept_id
0          101    Alice       10
1          102      Bob       20
2          103  Charlie       10
3          104    David       30

Departments DataFrame:
   dept_id    dept_name
0       10  Engineering
1       20    Marketing
2       40        Sales
```

两个数据框都共享一个 `dept_id` 列。我们可以使用此列作为“键”来合并这两个数据框，将 `dept_name` 添加到员工信息中。

最简单的合并方法是使用 `on` 参数 (parameter)指定要合并的数据框和要合并的列。

```python
# 根据 'dept_id' 合并员工和部门数据
merged_df = pd.merge(employees, departments, on='dept_id')

print("\n合并后的数据框 (默认 - 内连接)：")
print(merged_df)
```

**Output:**

```
Merged DataFrame (Default - Inner Join):
   employee_id     name  dept_id    dept_name
0          101    Alice       10  Engineering
1          103  Charlie       10  Engineering
2          102      Bob       20    Marketing
```

请注意以下几点：

1. `dept_name` 现在根据 `dept_id` 与相应的员工关联起来。
2. **David** (员工ID 104，部门ID 30) 从结果中消失了。为什么？因为 `dept_id` 30 在 `departments` 数据框中不存在。
3. **销售部** (部门ID 40) 也消失了。为什么？因为 `employees` 数据框中没有员工的 `dept_id` 为 40。

默认情况下，`pd.merge()` 执行**内连接**。这表示它只保留那些（在本例中是 `dept_id`）同时存在于*两个*数据框中的行。我们很快会介绍其他连接类型。

### 指定不同的列名

如果两个数据框中的列名不同怎么办？例如，假设 `employees` 数据框中的部门键名为 `department_code` 而不是 `dept_id`。

```python
# 列名修改后的员工数据
employees_alt = pd.DataFrame({
    'employee_id': [101, 102, 103, 104],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'department_code': [10, 20, 10, 30] # 重命名后的列
})

print("员工数据框 (替代版本)：")
print(employees_alt)
print("\n部门数据框：")
print(departments)
```

**Output:**

```
Employees DataFrame (Alternative):
   employee_id     name  department_code
0          101    Alice               10
1          102      Bob               20
2          103  Charlie               10
3          104    David               30

Departments DataFrame:
   dept_id    dept_name
0       10  Engineering
1       20    Marketing
2       40        Sales
```

在这种情况下，你不能直接使用 `on` 参数 (parameter)。相反，你需要使用 `left_on` 和 `right_on` 分别指定左侧 (`employees_alt`) 和右侧 (`departments`) 数据框的列名。

```python
# 使用 left_on 和 right_on 进行合并
merged_alt_keys = pd.merge(employees_alt, departments,
                           left_on='department_code',
                           right_on='dept_id')

print("\n合并后的数据框 (列名不同)：")
print(merged_alt_keys)
```

**Output:**

```
Merged DataFrame (Different Names):
   employee_id     name  department_code  dept_id    dept_name
0          101    Alice               10       10  Engineering
1          103  Charlie               10       10  Engineering
2          102      Bob               20       20    Marketing
```

结果与之前相同，但请注意，合并后的数据框中包含了 *两个* 列 (`department_code` 和 `dept_id`)。如果其中一个冗余，你可能希望在合并后将其删除。

### 基于多个键进行合并

你还可以基于多个列进行合并。只需向 `on` 参数 (parameter)提供一个列名列表（或者在使用 `left_on` 和 `right_on` 时提供列名列表）。合并操作将只组合那些所有指定列在数据框之间都匹配的行。

```python
# 具有多个潜在键的示例数据框
df_left = pd.DataFrame({
    'key1': ['A', 'B', 'B', 'C'],
    'key2': [1, 2, 1, 2],
    'left_val': [10, 20, 30, 40]
})

df_right = pd.DataFrame({
    'key1': ['B', 'C', 'C', 'D'],
    'key2': [1, 2, 3, 1],
    'right_val': [100, 200, 300, 400]
})

print("左侧数据框：")
print(df_left)
print("\n右侧数据框：")
print(df_right)

# 同时基于 key1 和 key2 进行合并
merged_multi = pd.merge(df_left, df_right, on=['key1', 'key2'])

print("\n基于多个键合并后的数据框 (内连接)：")
print(merged_multi)
```

**Output:**

```
Left DataFrame:
  key1  key2  left_val
0    A     1        10
1    B     2        20
2    B     1        30
3    C     2        40

Right DataFrame:
  key1  key2  right_val
0    B     1        100
1    C     2        200
2    C     3        300
3    D     1        400

Merged on Multiple Keys (Inner Join):
  key1  key2  left_val  right_val
0    B     1        30        100
1    C     2        40        200
```

只有那些 `df_left` 和 `df_right` 中 `key1` 和 `key2` *都*匹配的行被保留。

`pd.merge()` 函数是根据共享信息整合来自不同源的数据的基本工具。了解如何指定键是第一步。接下来，我们将介绍你可以执行的不同*类型*的合并。

## 参考资料

- [pandas.merge](https://pandas.pydata.org/docs/reference/api/pandas.merge.html) — Wes McKinney and the pandas development team (2024)
  pd.merge的官方API参考文档，解释了其参数和行为。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  pandas创建者撰写的书籍，包含数据整合的实用示例。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/03.07-merging-and-joining.html) — Jake VanderPlas (2017)
  Publisher: O'Reilly Media
  该资源解释了`pd.merge`及其与SQL风格连接的关系，并附有示例。

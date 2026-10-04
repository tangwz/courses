---
course: "essential-numpy-pandas"
chapter: "basic-data-manipulation-pandas"
lesson: "renaming-columns"
sourceId: 1814
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-8-basic-data-manipulation-pandas/renaming-columns"
title: "重命名列"
description: "使用 .rename() 方法更改 DataFrame 中列的名称。"
order: 7
plots: []
sourceHash: "1e63e18ad4097271a8fc470d171fc638a76b0a57fa61ba5780bf584ac4816d05"
sourceCorrections: []
---

在清洗和准备数据时，您会发现原始列名通常不理想。它们可能不够清晰，过长，包含空格或特殊字符导致在代码中使用不便，或者 simply 不符合一致的命名约定。重命名列是让您的DataFrame更易于理解和使用的一个常见且主要步骤。

Pandas 提供了灵活的 `.rename()` 方法，专门用于此目的。它允许您更改列名（和索引标签），而不会改变数据本身。

### 使用 `.rename()` 方法

使用 `.rename()` 最常用的方法是向其 `columns` 参数 (parameter)传递一个字典。这个字典应该将旧列名（键）映射到新列名（值）。

我们从一个示例 DataFrame 开始：

```python
import pandas as pd
import numpy as np

# 包含不理想列名的示例 DataFrame
data = {'Student ID': [101, 102, 103, 104],
        'Test Score (Math)': [85, 92, np.nan, 78],
        'Test Score (English)': [76, 88, 95, 80],
        'attendance %': [90, 95, 85, 92]}
df = pd.DataFrame(data)

print("原始 DataFrame:")
print(df)
```

这将输出：

```
Original DataFrame:
   Student ID  Test Score (Math)  Test Score (English)  attendance %
0         101               85.0                    76            90
1         102               92.0                    88            95
2         103                NaN                    95            85
3         104               78.0                    80            92
```

请注意，列名包含空格、括号和 '%' 等符号。我们把它们重命名为更适合编程的形式，使用小写字母和下划线。

```python
# 定义重命名映射
rename_map = {
    'Student ID': 'student_id',
    'Test Score (Math)': 'math_score',
    'Test Score (English)': 'english_score',
    'attendance %': 'attendance_pct'
}

# 使用 .rename() 方法的 columns 参数
df_renamed = df.rename(columns=rename_map)

print("\n重命名后的 DataFrame:")
print(df_renamed)
```

输出显示了新的列名：

```
DataFrame after renaming:
   student_id  math_score  english_score  attendance_pct
0         101        85.0             76              90
1         102        92.0             88              95
2         103         NaN             95              85
3         104        78.0             80              92
```

您不必一次性重命名所有列。如果您只提供部分列的映射，那么只有那些列会被重命名。

### 原地修改 DataFrame

默认情况下，`.rename()` 会返回一个带有更新名称的 *新* DataFrame，同时保持原始 DataFrame 不变。这通常更安全，因为它防止了对原始数据的意外修改。

但是，如果您确定要直接修改 DataFrame，可以使用 `inplace=True` 参数 (parameter)：

```python
# 创建一个副本以演示原地修改
df_copy = df.copy()

print("\n原始 DataFrame (副本):")
print(df_copy)

# 原地重命名列
df_copy.rename(columns=rename_map, inplace=True)

print("\n原地重命名后的 DataFrame:")
print(df_copy)
# 注意：df_copy 现在已被修改，df_renamed 之前是作为新对象创建的
```

Output:

```
Original DataFrame (copy):
   Student ID  Test Score (Math)  Test Score (English)  attendance %
0         101               85.0                    76            90
1         102               92.0                    88            95
2         103                NaN                    95            85
3         104               78.0                    80            92

DataFrame after inplace renaming:
   student_id  math_score  english_score  attendance_pct
0         101        85.0             76              90
1         102        92.0             88              95
2         103         NaN             95              85
3         104        78.0             80              92
```

使用 `inplace=True` 有时能让代码稍微简洁，但请谨慎使用。当对象被直接修改时，追踪更改通常会变得更难，尤其是在较长的分析脚本或 Jupyter notebook 中。

### 重命名索引标签

`.rename()` 方法也可以用于重命名索引标签，使用 `index` 参数 (parameter)。它与 `columns` 参数的工作方式类似，接受一个将旧索引标签映射到新索引标签的字典。

```python
# 索引重命名示例（假设使用之前的 df_renamed）
# 我们先将 student_id 设置为索引
df_indexed = df_renamed.set_index('student_id')
print("\n以 student_id 为索引的 DataFrame:")
print(df_indexed)

# 重命名特定的索引标签
index_rename_map = {101: 'S101', 104: 'S104'}
df_index_renamed = df_indexed.rename(index=index_rename_map)

print("\n重命名索引标签后的 DataFrame:")
print(df_index_renamed)
```

Output:

```
DataFrame with student_id as index:
            math_score  english_score  attendance_pct
student_id                                           
101               85.0             76              90
102               92.0             88              95
103                NaN             95              85
104               78.0             80              92

DataFrame after renaming index labels:
            math_score  english_score  attendance_pct
student_id                                           
S101              85.0             76              90
102               92.0             88              95
103                NaN             95              85
S104              78.0             80              92
```

### 替代方法：赋值给 `df.columns`

如果您需要重命名 *所有* 列，并且知道它们新名称的正确顺序，您可以直接将新名称列表赋值给 DataFrame 的 `.columns` 属性。

```python
# 确保列表长度与列数匹配
new_column_names = ['id', 'score_math', 'score_english', 'attendance']

# 创建另一个副本以演示此方法
df_copy2 = df.copy()

# 将新列表赋值给 df.columns
df_copy2.columns = new_column_names

print("\n赋值给 df.columns 后的 DataFrame:")
print(df_copy2)
```

Output:

```
DataFrame after assigning to df.columns:
    id  score_math  score_english  attendance
0  101        85.0             76          90
1  102        92.0             88          95
2  103         NaN             95          85
3  104        78.0             80          92
```

这种方法更直接，但比 `.rename()` 灵活性更差。您必须为 *所有* 列提供名称，并且列表长度必须与 DataFrame 中的列数完全匹配，否则会报错。它通常更适合于创建 DataFrame 或全面修改列名的情况。对于有针对性的重命名，`.rename()` 通常是更推荐的方法。

重命名列是一种简单而有效的方法，可以提高 DataFrame 的清晰度和可用性，使后续分析步骤更顺畅，并使您的代码更易于阅读。

## 参考资料

- [pandas.DataFrame.rename](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html) — The pandas development team (2024)
  Pandas中`.rename()`方法的官方文档，详细说明其在列和索引标签上的应用。
- [pandas.DataFrame.columns](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.columns.html) — The pandas development team (2024)
  `DataFrame.columns`属性的官方文档，解释了直接赋值来重命名所有列的方法。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: Chapter 7: Data Cleaning and Preparation
  一本广泛参考的Pandas数据处理书籍，其中包含数据清洗和准备技术。

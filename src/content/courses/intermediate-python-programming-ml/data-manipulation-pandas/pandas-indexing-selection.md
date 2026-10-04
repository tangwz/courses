---
course: "intermediate-python-programming-ml"
chapter: "data-manipulation-pandas"
lesson: "pandas-indexing-selection"
sourceId: 2145
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-3-data-manipulation-pandas/pandas-indexing-selection"
title: "数据索引与选择"
description: "学习使用 loc、iloc 和布尔索引选择行、列和特定数据点的方法。"
order: 3
plots: []
sourceHash: "6919805a86da3ee5f10ce0682b73ce4e903548345a6aee7277a67c10dd28df01"
sourceCorrections: []
---

Pandas 利用 `Series` 和 `DataFrame` 等基本数据结构来组织和管理数据。为了有效处理这些数据，访问其特定部分——例如选取数据子集、行、列或单个单元格——几乎是所有数据分析和处理工作的基本操作。Pandas 提供了多种强大且优化的索引和选择方法，可根据您是想按标签还是按整数位置进行选择提供灵活性。

### 使用 `[]` 进行基本选择

选取数据最直接的方式是使用方括号操作符 `[]`。它的行为取决于您在方括号内传递的内容，以及您正在处理的是 `Series` 还是 `DataFrame`。

- **在 DataFrame 中选择列**：传递单个列名（字符串）或列名列表可选择一个或多个列。

  ```python
  import pandas as pd
  import numpy as np

  # 假设 df 是一个预定义的 DataFrame，例如：
  data = {
      'Temperature': [25.3, 26.1, 24.8, 27.0, 23.9, 25.5, 26.8],
      'Humidity': [65, 68, 62, 70, 60, 66, 69],
      'Pressure': [1012, 1010, 1015, 1009, 1013, 1011, 1008]
  }
  index_labels = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03',
                                 '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07'])
  df = pd.DataFrame(data, index=index_labels)

  # 选择单个列（返回一个 Series）
  temperatures = df['Temperature']
  print(type(temperatures))
  # 输出: <class 'pandas.core.series.Series'>
  print(temperatures)
  # 输出:
  # 2023-01-01    25.3
  # 2023-01-02    26.1
  # 2023-01-03    24.8
  # 2023-01-04    27.0
  # 2023-01-05    23.9
  # 2023-01-06    25.5
  # 2023-01-07    26.8
  # Name: Temperature, dtype: float64

  # 选择多个列（返回一个 DataFrame）
  weather_subset = df[['Temperature', 'Pressure']]
  print(type(weather_subset))
  # 输出: <class 'pandas.core.frame.DataFrame'>
  print(weather_subset.head(3))
  # 输出:
  #             Temperature  Pressure
  # 2023-01-01         25.3      1012
  # 2023-01-02         26.1      1010
  # 2023-01-03         24.8      1015
  ```
- **在 DataFrame 中选择行（切片）**：如果您传递一个切片（如 `0:3`），`[]` 会按位置选择行。这有时可能不明确，因为 `[]` 对于 DataFrame 主要用于列操作。

  ```python
  # 选择前 3 行
  first_three_days = df[0:3]
  print(first_three_days)
  # 输出:
  #             Temperature  Humidity  Pressure
  # 2023-01-01         25.3        65      1012
  # 2023-01-02         26.1        68      1010
  # 2023-01-03         24.8        62      1015
  ```

通过切片使用 `[]` 进行行选择是可行的，但为了清晰起见并避免可能出现的困扰，特别是在处理整数索引时，Pandas 提供了更明确的方法：`.loc` 和 `.iloc`。

### 基于标签的选择：`.loc`

`.loc` 索引器主要用于基于**标签**（索引名称和列名）进行选择。它提供了一种非常明确的数据选择方式。

- **语法**：`df.loc[row_label(s), column_label(s)]`
- **选择行**：

  - 通过标签选择单行：`df.loc['2023-01-02']`（返回一个 Series）
  - 通过标签列表选择多行：`df.loc[['2023-01-02', '2023-01-05']]`（返回一个 DataFrame）
  - 通过标签切片选择行：`df.loc['2023-01-02':'2023-01-04']`（返回一个 DataFrame）。请注意，基于标签的切片**包含**结束标签。
- **选择行和列**：

  - 单个单元格：`df.loc['2023-01-03', 'Humidity']`
  - 特定行和列：`df.loc[['2023-01-03', '2023-01-06'], ['Temperature', 'Pressure']]`
  - 行和列的切片：`df.loc['2023-01-02':'2023-01-04', 'Humidity':'Pressure']`
  - 特定列的所有行：`df.loc[:, ['Temperature', 'Humidity']]`（使用 `:` 选择所有行）
  - 特定行的所有列：`df.loc[['2023-01-05', '2023-01-07'], :]`（使用 `:` 选择所有列）

```python
# .loc 的示例

# 通过标签选择单行
print("2023-01-02 的行:")
print(df.loc['2023-01-02'])
# 输出:
# 2023-01-02 的行:
# Temperature     26.1
# Humidity        68.0
# Pressure      1010.0
# Name: 2023-01-02 00:00:00, dtype: float64

# 通过标签切片选择行（包含结束标签）
print("\n从 2023-01-02 到 2023-01-04 的行:")
print(df.loc['2023-01-02':'2023-01-04'])
# 输出:
# 从 2023-01-02 到 2023-01-04 的行:
#             Temperature  Humidity  Pressure
# 2023-01-02         26.1        68      1010
# 2023-01-03         24.8        62      1015
# 2023-01-04         27.0        70      1009

# 特定行和列
print("\n特定日期的温度和气压:")
print(df.loc[['2023-01-03', '2023-01-06'], ['Temperature', 'Pressure']])
# 输出:
# 特定日期的温度和气压:
#             Temperature  Pressure
# 2023-01-03         24.8      1015
# 2023-01-06         25.5      1011

# 所有行，特定列
print("\n湿度和气压列:")
print(df.loc[:, ['Humidity', 'Pressure']].head(3))
# 输出:
# 湿度和气压列:
#             Humidity  Pressure
# 2023-01-01        65      1012
# 2023-01-02        68      1010
# 2023-01-03        62      1015
```

### 基于位置的选择：`.iloc`

`.iloc` 索引器主要用于基于**整数位置**（从 $0$ 到 $长度-1$，如同标准 Python 列表）进行选择。它忽略索引标签和列名。

- **语法**：`df.iloc[row_position(s), column_position(s)]`
- **选择行**：

  - 通过位置选择单行：`df.iloc[1]`（返回对应于第二行的 Series）
  - 通过位置列表选择多行：`df.iloc[[1, 4]]`（返回包含第二行和第五行的 DataFrame）
  - 通过位置切片选择行：`df.iloc[1:4]`（返回包含位置 1、2 和 3 的行的 DataFrame）。请注意，基于整数的切片**不包含**结束位置，这与 Python 的切片行为一致。
- **选择行和列**：

  - 单个单元格：`df.iloc[2, 1]`（第三行、第二列的单元格）
  - 特定行和列：`df.iloc[[2, 5], [0, 2]]`（第三行和第六行，第一列和第三列）
  - 行和列的切片：`df.iloc[1:4, 0:2]`（行 1-3，列 0-1）
  - 特定列的所有行：`df.iloc[:, [0, 1]]`（第一列和第二列）
  - 特定行的所有列：`df.iloc[[4, 6], :]`（第五行和第七行）

```python
# .iloc 的示例

# 通过位置选择单行（第二行）
print("索引位置 1 的行:")
print(df.iloc[1])
# 输出:
# 索引位置 1 的行:
# Temperature     26.1
# Humidity        68.0
# Pressure      1010.0
# Name: 2023-01-02 00:00:00, dtype: float64

# 通过位置切片选择行（不包含结束位置）
print("\n索引位置 1 到 3 的行（不包含结束位置）:")
print(df.iloc[1:4])
# Output:
# 索引位置 1 到 3 的行（不包含结束位置）:
#             Temperature  Humidity  Pressure
# 2023-01-02         26.1        68      1010
# 2023-01-03         24.8        62      1015
# 2023-01-04         27.0        70      1009

# 通过位置选择特定行和列
print("\n特定位置的单元格:")
print(df.iloc[[2, 5], [0, 2]]) # Rows 2, 5 and Columns 0, 2
# Output:
# 特定位置的单元格:
#             Temperature  Pressure
# 2023-01-03         24.8      1015
# 2023-01-06         25.5      1011

# 通过位置切片选择行和列
print("\n行和列的切片:")
print(df.iloc[1:4, 0:2]) # Rows 1-3, Columns 0-1
# Output:
# 行和列的切片:
#             Temperature  Humidity
# 2023-01-02         26.1        68
# 2023-01-03         24.8        62
# 2023-01-04         27.0        70
```

### 布尔索引

最有用的选择方法之一是使用布尔（True/False）数组或 Series 来根据条件筛选数据。

1. **创建布尔条件**：通过将比较运算符应用于 Series（例如 DataFrame 的某一列）来生成一个布尔 Series。
2. **应用条件**：将此布尔 Series 传递给 `[]`、`.loc[]` 或 `.iloc[]`。只有布尔 Series 为 `True` 的行才会被选择。

```python
# 布尔索引示例

# 1. 创建条件：温度高于 25.0 摄氏度
high_temp_condition = df['Temperature'] > 25.0
print("布尔条件 Series:")
print(high_temp_condition)
# 输出:
# 布尔条件 Series:
# 2023-01-01     True
# 2023-01-02     True
# 2023-01-03    False
# 2023-01-04     True
# 2023-01-05    False
# 2023-01-06     True
# 2023-01-07     True
# Name: Temperature, dtype: bool

# 2. 使用 [] 应用条件
high_temp_days = df[high_temp_condition]
print("\n温度高于 25.0 的天数（使用 []）：")
print(high_temp_days)
# Output:
# 温度高于 25.0 的天数（使用 []）：
#             Temperature  Humidity  Pressure
# 2023-01-01         25.3        65      1012
# 2023-01-02         26.1        68      1010
# 2023-01-04         27.0        70      1009
# 2023-01-06         25.5        66      1011
# 2023-01-07         26.8        69      1008

# 使用 .loc 应用条件（为了清晰推荐此方式）
high_temp_days_loc = df.loc[high_temp_condition]
print("\n温度高于 25.0 的天数（使用 .loc）：")
print(high_temp_days_loc)
# 输出: (与上方相同)

# 根据条件选择特定列
high_temp_humidity = df.loc[df['Temperature'] > 26.0, 'Humidity']
print("\n温度高于 26.0 时对应的湿度:")
print(high_temp_humidity)
# Output:
# 湿度高于 26.0 的天数:
# 2023-01-02    68
# 2023-01-04    70
# 2023-01-07    69
# Name: Humidity, dtype: int64
```

您可以使用逻辑运算符组合多个条件：`&` 表示 AND，`|` 表示 OR，`~` 表示 NOT。请记住，由于 Python 的运算符优先级规则，需要将每个条件括在括号中。

```python
# 组合条件：温度 > 25 且 湿度 < 68
complex_condition = (df['Temperature'] > 25.0) & (df['Humidity'] < 68)
print("\n复杂条件（温度 > 25.0 且 湿度 < 68）:")
print(complex_condition)
# Output:
# 复杂条件（温度 > 25.0 且 湿度 < 68）：
# 2023-01-01     True
# 2023-01-02    False
# 2023-01-03    False
# 2023-01-04    False
# 2023-01-05    False
# 2023-01-06     True
# 2023-01-07    False
# dtype: bool

filtered_data = df.loc[complex_condition]
print("\n基于复杂条件筛选的数据:")
print(filtered_data)
# Output:
# 基于复杂条件筛选的数据：
#             Temperature  Humidity  Pressure
# 2023-01-01         25.3        65      1012
# 2023-01-06         25.5        66      1011
```

### 使用索引设置值

这些选择方法不仅用于读取数据，还可以用于就地修改数据。通过将选择置于赋值运算符的左侧，您可以更新 DataFrame 或 Series 的特定部分。

```python
# 设置值

# 创建副本以避免修改原始 df
df_copy = df.copy()

# 使用 .loc 将特定日期的湿度设置为 75
df_copy.loc['2023-01-03', 'Humidity'] = 75
print("\n2023-01-03 设定湿度后的 DataFrame:")
print(df_copy.loc['2023-01-01':'2023-01-04'])
# Output:
# 2023-01-03 设定湿度后的 DataFrame:
#             Temperature  Humidity  Pressure
# 2023-01-01         25.3        65      1012
# 2023-01-02         26.1        68      1010
# 2023-01-03         24.8        75      1015  # <-- 已更改值
# 2023-01-04         27.0        70      1009

# 使用布尔索引将气压 > 1012 的所有行的温度设置为 0
df_copy.loc[df_copy['Pressure'] > 1012, 'Temperature'] = 0
print("\n根据气压设定温度后的 DataFrame:")
print(df_copy)
# Output:
# 根据气压设定温度后的 DataFrame:
#             Temperature  Humidity  Pressure
# 2023-01-01         25.3        65      1012
# 2023-01-02         26.1        68      1010
# 2023-01-03          0.0        75      1015  # <-- 已更改值
# 2023-01-04         27.0        70      1009
# 2023-01-05          0.0        60      1013  # <-- 已更改值
# 2023-01-06         25.5        66      1011
# 2023-01-07         26.8        69      1008

# 使用 .iloc 为第一行设置多个列
df_copy.iloc[0, [0, 1]] = [20.0, 50] # 设置第一行的温度和湿度
print("\n使用 .iloc 设置多个值后的 DataFrame:")
print(df_copy.head(3))
# Output:
# 使用 .iloc 设置多个值后的 DataFrame:
#             Temperature  Humidity  Pressure
# 2023-01-01         20.0        50      1012  # <-- 已更改值
# 2023-01-02         26.1        68      1010
# 2023-01-03          0.0        75      1015
```

### `.loc` 和 `.iloc` 的选择

- 当您需要根据索引标签或列名选择数据时，请使用 `.loc`。如果您的索引标签有意义（如日期或 ID），它通常更明确，且不易出错。
- 当您需要根据整数位置选择数据时，请使用 `.iloc`，无论索引标签或列名如何。当您需要第 N 行或第 N 列时，或处理默认整数索引（0, 1, 2...）时，这会很有用。
- 如果可能，请避免使用 `[]` 进行行切片；为求清晰，请优先使用 `.loc` 或 `.iloc`。主要使用 `[]` 进行列选择是常见且可接受的。
- 当您的 DataFrame 具有整数索引标签时，请务必小心。在这种情况下，`df.loc[0]` 会选择标签为 `0` 的行，而 `df.iloc[0]` 会选择按位置排列的第一行。这种不确定性是一个重要的原因，促使我们优先使用 `.loc` 和 `.iloc`，而不是简单地使用 `[]` 索引进行除了基本列选择之外的任何操作。

掌握这些数据索引和选择技巧对于使用 Pandas 进行有效的数据处理至关重要。在为分析和机器学习 (machine learning)模型准备数据时，您会持续使用它们来筛选、子集化、查看和修改数据。

## 参考资料

- [Indexing and selecting data](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html) — The pandas development team (2024)
  Pandas数据索引和选择方法的官方、最全面指南，详细介绍了最佳实践和高级技术。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  本书由Pandas库的创建者撰写，深入讲解数据处理，包括重要的索引技术及其设计原理。
- [Hands-On Machine Learning with Scikit-Learn, Keras, & TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  本书提供了使用Pandas进行数据预处理和选择的实用示例，展示了索引对于为机器学习模型准备数据的重要性。

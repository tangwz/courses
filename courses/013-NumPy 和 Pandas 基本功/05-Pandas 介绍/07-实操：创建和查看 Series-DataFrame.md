# 实操：创建和查看 Series/DataFrame

来源：[原文](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-5-introduction-to-pandas/create-examine-pandas-practice)

[返回章节目录](README.md) · [返回课程目录](../README.md)

本实践旨在巩固您对 Pandas Series 和 DataFrame 的理解。您将通过不同方法创建这些基本数据结构，然后运用多种技术检查它们，从而了解它们的内容和属性。假定您在 Jupyter Notebook 环境中操作。

首先，请确保您已导入 Pandas。通常将其导入并取别名为 `pd`。我们还将导入 NumPy 并取别名为 `np`，因为 Pandas 经常与它配合使用。

```python
import pandas as pd
import numpy as np

print("Pandas 版本:", pd.__version__)
print("NumPy 版本:", np.__version__)
```

### 创建和查看 Pandas Series

Series 类似于带标签的一维数组。我们来创建几个。

**1. 从 Python 列表创建：**
如果您不指定，Pandas 会自动创建一个默认的整数索引。

```python
# 简单数字列表
data_list = [10, 20, 30, 40, 50]
series_from_list = pd.Series(data_list)

print("从列表创建的 Series:")
print(series_from_list)

# 访问元素（类似于列表/数组）
print("\n第一个元素:", series_from_list[0])
```

**2. 从带有自定义索引的 Python 列表创建：**
您可以通过分配标签来为索引提供含义。

```python
# 温度列表
temperatures = [22.5, 24.1, 19.8, 23.0]
# 对应的日期
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday']

series_with_index = pd.Series(temperatures, index=days)

print("\n带有自定义索引的 Series:")
print(series_with_index)

# 使用自定义索引访问
print("\n星期二的温度:", series_with_index['Tuesday'])
```

**3. 从 Python 字典创建：**
字典自然地将键（成为索引）映射到值。

```python
# 人口数据（单位：百万）
population_dict = {'California': 39.5, 'Texas': 29.1, 'Florida': 21.5, 'New York': 19.4}
series_from_dict = pd.Series(population_dict)

print("\n从字典创建的 Series:")
print(series_from_dict)

# 检查数据类型
print("\nSeries 的数据类型:", series_from_dict.dtype) # 在本例中是浮点数
```

**4. 从 NumPy 数组创建：**
您可以轻松地将 NumPy 数组转换为 Series。

```python
# 一个 NumPy 数组
np_array = np.array([100, 200, 300, 400])
series_from_numpy = pd.Series(np_array, index=['a', 'b', 'c', 'd'])

print("\n从 NumPy 数组创建的 Series:")
print(series_from_numpy)

# 检查其索引和值
print("\n索引:", series_from_numpy.index)
print("值:", series_from_numpy.values) # 返回一个 NumPy 数组
```

### 创建和查看 Pandas DataFrame

DataFrame 是 Pandas 的主力，表示表格数据。

**1. 从列表字典创建：**
这是创建 DataFrame 的一种非常常见的方法。每个字典键成为列名，与其关联的列表成为该列的数据。所有列表的长度必须相同。

```python
# 学生数据
student_data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [20, 21, 19, 22],
    'Major': ['CompSci', 'Physics', 'Math', 'CompSci'],
    'GPA': [3.8, 3.5, 3.9, 3.2]
}

df_students = pd.DataFrame(student_data)

print("从列表字典创建的 DataFrame:")
print(df_students)
```

**2. 从字典列表创建：**
列表中的每个字典代表一行。Pandas 从键推断列名。字典中缺失的键将导致 `NaN`（非数字）值。

```python
# 可能缺少部分信息的数据
sensor_readings = [
    {'sensor': 'A', 'temp': 25.5, 'humidity': 60},
    {'sensor': 'B', 'temp': 26.1}, # 湿度缺失
    {'sensor': 'A', 'temp': 25.8, 'humidity': 62},
    {'sensor': 'C', 'temp': 24.9, 'pressure': 1012} # 不同的字段
]

df_sensors = pd.DataFrame(sensor_readings)

print("\n从字典列表创建的 DataFrame:")
print(df_sensors)
```

请注意数据缺失处的 `NaN` 值。Pandas 会自动处理这种情况。

**3. 从二维 NumPy 数组创建：**
您可以从 NumPy 数组创建 DataFrame，并可选择提供列名和索引名。

```python
# 二维 NumPy 数组
data_np = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 创建带有默认索引/列的 DataFrame
df_from_np_default = pd.DataFrame(data_np)
print("\n从 NumPy 数组创建的 DataFrame（默认名称）：")
print(df_from_np_default)

# 创建带有自定义名称的 DataFrame
df_from_np_custom = pd.DataFrame(data_np,
                                 index=['Row1', 'Row2', 'Row3'],
                                 columns=['ColA', 'ColB', 'ColC'])
print("\n从 NumPy 数组创建的 DataFrame（自定义名称）：")
print(df_from_np_custom)
```

### 检查 DataFrame

现在，我们使用前面创建的 `df_students` DataFrame 来练习检查技术。

```python
print("用于检查的学生 DataFrame:")
print(df_students)

# 1. 使用 head() 查看顶部行
print("\n前 2 行 (head):")
print(df_students.head(2)) # 默认为 5 行

# 2. 使用 tail() 查看底部行
print("\n后 2 行 (tail):")
print(df_students.tail(2)) # 默认为 5 行

# 3. 使用 info() 获取简洁摘要
print("\nDataFrame 信息:")
df_students.info()
# 这会显示列名、非空计数和数据类型 (dtypes)。
# 请注意 'Major' 是 'object' 类型（通常是字符串），Age 是 int64，GPA 是 float64。

# 4. 使用 describe() 获取统计摘要
print("\n统计描述:")
print(df_students.describe())
# 为数值列提供计数、均值、标准差、最小值、最大值和四分位数。
# 'Name' 和 'Major'（对象类型）默认被排除。

# 5. 使用 shape 检查维度
print("\nDataFrame 形状（行，列）:", df_students.shape)

# 6. 列出列名
print("\n列名:", df_students.columns)

# 7. 查看索引
print("\n索引:", df_students.index) # 在此处显示默认的 RangeIndex
```

本次实操课程涵盖了从列表、字典和 NumPy 数组等各种常见数据结构创建 Series 和 DataFrame 的方法。我们还练习了使用 `head()`、`tail()`、`info()` 和 `describe()` 等重要方法，以及 `shape`、`columns` 和 `index` 等属性，以便快速了解数据的结构和内容。这些是您在使用 Pandas 时会持续用到的基本技能。

## 参考资料

- [Pandas User Guide: Series](https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#series) — The Pandas Development Team (2023)
  关于理解和创建 Pandas Series 对象的全面指南，包括各种构造方法和属性。
- [Pandas User Guide: DataFrame](https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#dataframe) — The Pandas Development Team (2023)
  详细解释了 DataFrame 的创建、属性以及 `head()`、`info()` 和 `describe()` 等基本检查方法。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  Pandas 创建者撰写的权威书籍，涵盖了 Series 和 DataFrame 的基础概念、创建和基本操作。
- [NumPy User Guide](https://numpy.org/doc/stable/user/whatisnumpy.html) — The NumPy Developers (2023)
  理解 NumPy 数组的核心资源，NumPy 数组是 Pandas Series 和 DataFrame 的底层数据结构。

---

[上一节](06-%E6%A3%80%E6%9F%A5%20DataFrames.md) · [下一节](../06-Pandas%20%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E4%B8%8E%E4%BF%9D%E5%AD%98/01-%E4%BB%8ECSV%E6%96%87%E4%BB%B6%E8%AF%BB%E5%8F%96%E6%95%B0%E6%8D%AE.md)

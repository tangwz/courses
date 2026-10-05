# 创建 Series

来源：[原文](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-5-introduction-to-pandas/creating-series)

[返回章节目录](README.md) · [返回课程目录](../README.md)

要创建 Pandas Series（一种带标签的一维数组），有几种常见方法可以使用。`pd.Series()` 构造函数是主要工具。

我们需要 Pandas 库，它通常使用别名 `pd` 导入。如果您还打算使用 NumPy 数组（它与 Pandas 配合得很好），则将其导入为 `np`。

```python
import pandas as pd
import numpy as np
```

### 从 Python 列表创建 Series

创建 Series 最直接的方法是使用标准的 Python 列表。如果您不指定索引，Pandas 会自动创建一个从 0 开始的默认整数索引。

```python
# 创建一个简单的 Python 列表
data_list = [10, 20, 30, 40, 50]

# 从列表创建 Pandas Series
series_from_list = pd.Series(data_list)

# 打印 Series
print(series_from_list)
```

**输出：**

```
0    10
1    20
2    30
3    40
4    50
dtype: int64
```

注意输出显示了两列：左侧是索引（0 到 4），右侧是数据值。Pandas 还会推断数据类型（在此例中是 `dtype: int64`，表示 64 位整数）。

### 创建带自定义索引的 Series

虽然默认的整数索引很有用，但 Series 真正的用处在于它能够为索引使用有意义的标签。您可以在创建时使用 `index` 参数 (parameter)提供一个标签列表或数组。索引列表的长度必须与数据列表相同。

```python
# 数据列表
data_list = [100, 200, 300, 400]

# 自定义索引标签
index_labels = ['alpha', 'beta', 'gamma', 'delta']

# 创建带自定义索引的 Series
series_custom_index = pd.Series(data=data_list, index=index_labels)

# 打印 Series
print(series_custom_index)
```

**输出：**

```
alpha    100
beta     200
gamma    300
delta    400
dtype: int64
```

现在，索引不再是 0、1、2、3，而是由字符串标签 'alpha'、'beta'、'gamma' 和 'delta' 组成。这使得访问特定数据点更直观，我们将在后面的章节中看到。

### 从 NumPy 数组创建 Series

您可以同样轻松地从 NumPy 数组创建 Series。这非常普遍，因为数据通常源自使用 NumPy 进行的数值计算。这个过程与使用列表相同。

```python
# 创建一个 NumPy 数组
numpy_array = np.array([5.5, 6.6, 7.7, 8.8])

# 从 NumPy 数组创建 Series
series_from_numpy = pd.Series(numpy_array)

# 打印 Series
print(series_from_numpy)
```

**输出：**

```
0    5.5
1    6.6
2    7.7
3    8.8
dtype: float64
```

同样，Pandas 会创建默认的整数索引并推断数据类型（在此例中是 float64）。您也可以在从 NumPy 数组创建 Series 时提供自定义索引，就像使用列表一样。

### 从 Python 字典创建 Series

另一种方便的方法是直接从 Python 字典创建 Series。在这种情况下，Pandas 会使用字典键作为索引标签，字典值作为 Series 数据。Series 元素的顺序通常会遵循字典的插入顺序（对于 Python 3.7+）。

```python
# 创建一个 Python 字典
data_dict = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}

# 从字典创建 Series
series_from_dict = pd.Series(data_dict)

# 打印 Series
print(series_from_dict)
```

**输出：**

```
Ohio      35000
Texas     71000
Oregon    16000
Utah       5000
dtype: int64
```

当您的数据已按键值对形式组织时，这种方法特别有用。

您也可以在从字典创建 Series 时显式指定索引。如果提供的索引标签在字典中不存在对应的键，Pandas 会插入一个 `NaN`（非数字）值，这是 Pandas 表示缺失数据的标准方式。如果字典包含未出现在指定索引中的键，则这些键值对将被忽略。

```python
# 字典
data_dict = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}

# 显式索引 - 包含 'California'（不在字典中）并排除 'Utah'
states = ['California', 'Ohio', 'Oregon', 'Texas']

# 创建带显式索引的 Series
series_explicit_index = pd.Series(data_dict, index=states)

# 打印 Series
print(series_explicit_index)
```

**输出：**

```
California        NaN
Ohio          35000.0
Oregon        16000.0
Texas         71000.0
dtype: float64
```

注意 'California' 具有 `NaN` 值，因为它不在 `data_dict` 中。此外，原始字典中的 'Utah' 被排除，因为它不在 `states` 索引列表中。`dtype` 变为 `float64` 是因为 `NaN` 被视为浮点值。

这些方法涵盖了实例化 Pandas Series 对象最常见的方式。当您处理数据时，您会经常发现自己从列表、字典或 NumPy 数组等现有数据结构创建 Series，作为您分析流程的第一步。

## 参考资料

- [pandas.Series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html) — pandas development team (2024)
  提供Series构造函数的详细API参考，包含所有参数和创建Series的示例。
- [Series](https://pandas.pydata.org/docs/user_guide/dsintro.html#series) — pandas development team (2024)
  官方用户指南中介绍Series数据结构及其常用创建方法的部分。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  Pandas创建者的权威著作，深入涵盖Series创建和操作的基础内容。
- [The N-dimensional array (ndarray)](https://numpy.org/doc/stable/user/absolute_beginners.html#the-n-dimensional-array-ndarray) — NumPy community (2024)
  Publisher: NumPy project
  介绍NumPy数组的基本结构，它是创建Pandas Series对象的常见数据源。

---

[上一节](02-Pandas%20%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%EF%BC%9ASeries.md) · [下一节](04-Pandas%20%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%EF%BC%9ADataFrame.md)

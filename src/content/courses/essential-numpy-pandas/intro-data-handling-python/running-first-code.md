---
course: "essential-numpy-pandas"
chapter: "intro-data-handling-python"
lesson: "running-first-code"
sourceId: 1683
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-1-intro-data-handling-python/running-first-code"
title: "运行你的第一个代码片段"
description: "编写并执行使用 NumPy 和 Pandas 的简单代码示例，以验证你的设置。"
order: 4
plots: []
sourceHash: "9d659d0ffa106a30b25bba386409cbefc1417e32b0f4d2218e7d312c16ce76ea"
sourceCorrections: []
---

编写并执行你的第一批 NumPy 和 Pandas 代码是一个重要步骤。这有助于确认库是否正确安装，并让你初步了解它们如何运作。

### 验证你的安装

使用任何 Python 库的第一步是将其 `import` 到你当前的脚本或 Notebook 会话中。这使得库的函数和对象可供你使用。对于 NumPy 和 Pandas，有一些标准的社区导入惯例：

- NumPy 通常以别名 `np` 导入。
- Pandas 通常以别名 `pd` 导入。

使用这些别名让你的代码更短、更易读，因为它们被广泛认可。

打开你的 Jupyter Notebook（或 Python 解释器）并在一个单元格中输入以下代码：

```python
# 导入 NumPy 库
import numpy as np

# 导入 Pandas 库
import pandas as pd

# 打印确认信息
print("NumPy 和 Pandas 导入成功！")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
```

要在 Jupyter 中运行此代码，请按 `Shift + Enter`。如果安装成功，你将在单元格下方看到打印的确认信息，以及 NumPy 和 Pandas 的已安装版本。如果你遇到 `ModuleNotFoundError` 这样的错误信息，请重新检查上一节（“设置你的环境”）中的安装步骤。

### 你的第一个 NumPy 数组

NumPy 的主要对象是 N 维数组，通常称为 `ndarray`。让我们从一个标准的 Python 列表创建一个简单的一维数组。

```python
# 创建一个 Python 列表
my_list = [1, 2, 3, 4, 5]

# 从列表创建一个 NumPy 数组
np_array = np.array(my_list)

# 打印数组
print("我的第一个 NumPy 数组：")
print(np_array)

# 打印数组对象的类型
print("对象的类型：")
print(type(np_array))
```

运行此单元格将输出：

```
我的第一个 NumPy 数组：
[1 2 3 4 5]
对象的类型：
<class 'numpy.ndarray'>
```

注意到输出 `[1 2 3 4 5]` 与 Python 列表 `[1, 2, 3, 4, 5]` 略有不同。这表明它是一个 NumPy 数组。`type()` 函数确认 `np_array` 确实是 `numpy.ndarray` 类的一个实例。

NumPy 数组支持高效的逐元素操作。让我们尝试一个简单的：将每个元素乘以 2。

```python
# 执行逐元素操作
doubled_array = np_array * 2

print("乘法运算后的数组：")
print(doubled_array)
```

输出：

```
乘法运算后的数组：
[ 2  4  6  8 10]
```

对于标准的 Python 列表，你需要使用循环或列表推导来完成此操作。NumPy 在后台更高效地处理了这一点。

### 你的第一个 Pandas 对象：Series 和 DataFrame

Pandas 提供了两种主要数据结构：`Series`（一维）和 `DataFrame`（二维）。

让我们创建一个 `Series`。Series 类似于一维 NumPy 数组，但带有关联的索引。

```python
# 从列表创建一个 Pandas Series
data_series = [10, 20, 30, 40, 50]
pd_series = pd.Series(data_series)

# 打印 Series
print("我的第一个 Pandas Series：")
print(pd_series)

# 打印 Series 对象的类型
print("\n对象的类型：")
print(type(pd_series))
```

运行此单元格将输出：

```
我的第一个 Pandas Series：
0    10
1    20
2    30
3    40
4    50
dtype: int64

对象的类型：
<class 'pandas.core.series.Series'>
```

注意到输出左侧包含值（10、20 等）和索引（0、1、2 等）。`dtype: int64` 表示 Series 中存储值的数据类型。

现在，让我们创建一个 `DataFrame`。DataFrame 是一种二维的、表格状的结构，其中列可以具有不同的数据类型。一种常见的方法是从 Python 字典创建它，其中键成为列名，值（列表或数组）成为列数据。

```python
# 创建一个数据字典
data_dict = {
    'StudentID': [101, 102, 103, 104],
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Score': [85, 92, 78, 88]
}

# 从字典创建一个 Pandas DataFrame
df = pd.DataFrame(data_dict)

# 打印 DataFrame
print("我的第一个 Pandas DataFrame：")
print(df)

# 打印 DataFrame 对象的类型
print("\n对象的类型：")
print(type(df))
```

运行此单元格会生成：

```
我的第一个 Pandas DataFrame：
   StudentID     Name  Score
0        101    Alice     85
1        102      Bob     92
2        103  Charlie     78
3        104    David     88

对象的类型：
<class 'pandas.core.frame.DataFrame'>
```

此输出清晰地类似于一个带有标签列（'StudentID'、'Name'、'Score'）的表格和一个自动生成的行索引（0、1、2、3）。

执行这些简单的代码片段确认你的环境已就绪，并让你初步了解如何在 NumPy 和 Pandas 中创建和操作基本数据结构。在接下来的章节中，我们将在此基础上，更详细地研究这些库的功能。

## 参考资料

- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html) — The pandas development team (2024)
  详述Pandas数据结构（Series和DataFrame）、创建方法和基本数据操作的官方文档。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104030/) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: 579
  Pandas创建者撰写的权威书籍，介绍NumPy和Pandas的基础知识、数据结构和实用的数据分析技术。第三版。

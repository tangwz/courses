# NumPy 数组的索引与切片

来源：[原文](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-2-numerical-computing-numpy/numpy-indexing-slicing)

[返回章节目录](README.md) · [返回课程目录](../README.md)

NumPy 数组创建完成后，访问其单个元素或元素子集是一项主要操作。与 Python 列表类似，NumPy 数组采用基于零的索引并支持切片。然而，NumPy 将这些理念扩展到多个维度，并提供了更强大的索引技术，这些技术在机器学习 (machine learning)中的数据处理中非常有用。

### 访问一维数组中的元素

访问一维 NumPy 数组中的元素，其方式很像访问 Python 列表中的元素。你使用方括号 `[]` 和所需元素的整数索引。请记住索引从 0 开始。

```python
import numpy as np

# 创建一个一维数组
arr1d = np.arange(10) # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 访问第一个元素
print(arr1d[0])
# 输出: 0

# 访问第五个元素
print(arr1d[4])
# 输出: 4

# 访问最后一个元素
print(arr1d[-1])
# 输出: 9
```

你也可以使用索引来修改元素：

```python
# 修改第一个元素
arr1d[0] = 100
print(arr1d)
# 输出: array([100,   1,   2,   3,   4,   5,   6,   7,   8,   9])
```

### 一维数组切片

切片允许你选择一个元素范围。语法是 `start:stop:step`，其中 `start` 是要包含的第一个元素的索引（包含），`stop` 是第一个 *不* 包含的元素的索引（不包含），`step` 定义元素间的步长。

```python
# 从索引 2 切片到（不包括）索引 5
print(arr1d[2:5])
# 输出: array([2, 3, 4])

# 从开头切片到索引 4
print(arr1d[:4])
# 输出: array([100,   1,   2,   3])

# 从索引 5 切片到末尾
print(arr1d[5:])
# 输出: array([5, 6, 7, 8, 9])

# 每隔一个元素切片
print(arr1d[::2])
# 输出: array([100,   2,   4,   6,   8])

# 从索引 1 到 7，步长为 3 的切片
print(arr1d[1:8:3])
# 输出: array([1, 4, 7])
```

#### 切片是视图，而非副本

NumPy 数组切片与 Python 列表切片之间的一个重要区别是，NumPy 数组切片是对原始数组数据的 *视图*。这意味着它们共享相同的底层数据缓冲区。修改切片会修改原始数组。这种行为旨在提升性能，避免不必要的数据复制，特别是对于机器学习 (machine learning)中常见的大型数据集。

```python
arr_original = np.arange(5) # array([0, 1, 2, 3, 4])
arr_slice = arr_original[1:4] # 获取一个切片（视图）

print(f"切片修改前原始数组: {arr_original}")
print(f"修改前切片: {arr_slice}")

# 修改切片的第一个元素
arr_slice[0] = 99

print(f"修改后切片: {arr_slice}")
print(f"切片修改后原始数组: {arr_original}")
# 输出:
# 切片修改前原始数组: [0 1 2 3 4]
# 修改前切片: [1 2 3]
# 修改后切片: [99  2  3]
# 切片修改后原始数组: [ 0 99  2  3  4]
```

注意 `arr_slice[0]` 的改变也改变了 `arr_original[1]`。如果你需要切片的一个独立的副本，请使用 `copy()` 方法：

```python
arr_original = np.arange(5)
arr_slice_copy = arr_original[1:4].copy() # 创建一个显式副本

arr_slice_copy[0] = 99 # 修改副本

print(f"切片副本: {arr_slice_copy}")
print(f"原始数组: {arr_original}") # 原始数组保持不变
# 输出:
# 切片副本: [99  2  3]
# 原始数组: [0 1 2 3 4]
```

了解视图与副本的行为对防止数据处理代码中出现意外的副作用很重要。

### 多维数组的索引和切片

NumPy 将索引和切片自然地扩展到多个维度的数组。你在方括号内为每个维度提供索引或切片，并用逗号分隔。

考虑一个二维数组（矩阵）：

```python
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
```

#### 访问单个元素

要访问单个元素，请提供行索引，然后是列索引。

```python
# 访问第 1 行、第 2 列的元素（值 6）
element = arr2d[1, 2]
print(element)
# 输出: 6

# 另一种语法（不常用）：arr2d[1][2]
# 这种方法可行，但效率较低，因为它会创建一个中间一维数组
```

#### 二维数组切片

你可以沿每个维度进行切片：

```python
# 获取第一行（行索引 0，所有列）
first_row = arr2d[0, :] # or simply arr2d[0]
print(first_row)
# 输出: array([1, 2, 3])

# 获取第二列（所有行，列索引 1）
second_col = arr2d[:, 1]
print(second_col)
# 输出: array([2, 5, 8])

# 获取一个子数组：行 0 和 1，列 1 和 2
sub_array = arr2d[0:2, 1:3]
print(sub_array)
# 输出:
# [[2 3]
#  [5 6]]

# 获取右下角的 2x2 矩阵
bottom_right = arr2d[1:, 1:]
print(bottom_right)
# 输出:
# [[5 6]
#  [8 9]]
```

像一维切片一样，多维切片也是原始数组的视图。修改切片会影响原始的 `arr2d`。

```python
# 修改 sub_array 切片
sub_array[0, 0] = 99 # 改变 arr2d[0, 1]

print(sub_array)
# 输出:
# [[99  3]
#  [ 5  6]]

print(arr2d) # 原始数组被修改
# 输出:
# [[ 1 99  3]
#  [ 4  5  6]
#  [ 7  8  9]]
```

如果你需要子数组的独立副本，请使用 `arr2d[0:2, 1:3].copy()`。

### 布尔索引（掩码）

布尔索引，通常称为掩码，是一种高效方法，你使用布尔数组（包含 `True` 或 `False` 值）从另一个数组中选择元素。布尔数组的形状必须与它所索引的维度相同。

通常，你基于应用于数组自身的条件来创建布尔数组。

```python
data = np.array([1, -2, 3, -4, 5, -6])

# 为正值创建一个布尔掩码
positive_mask = data > 0 # array([ True, False,  True, False,  True, False])
print(positive_mask)

# 使用掩码只选择正数元素
positive_values = data[positive_mask]
print(positive_values)
# 输出: array([1, 3, 5])
```

你可以直接在方括号内应用条件：

```python
negative_values = data[data < 0]
print(negative_values)
# 输出: array([-2, -4, -6])
```

布尔索引可以与赋值结合，来修改符合条件的元素：

```python
# 将所有负值设为 0
data[data < 0] = 0
print(data)
# 输出: array([1, 0, 3, 0, 5, 0])
```

#### 组合布尔条件

你可以使用 NumPy 的按位逻辑运算符组合多个条件：`&`（与）、`|`（或）和 `~`（非）。**注意：** 使用这些按位运算符，而不是 Python 关键字 `and`、`or`、`not`，因为 NumPy 需要执行元素级比较。由于运算符优先级，通常需要括号。

```python
arr = np.arange(12).reshape((3, 4))
print("原始数组:\n", arr)
# 输出:
# 原始数组:
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# 选择大于 3 且小于 9 的元素
mask = (arr > 3) & (arr < 9)
print("掩码 ( > 3 & < 9):\n", mask)
# 输出:
# 掩码 ( > 3 & < 9):
# [[False False False False]
#  [ True  True  True  True]
#  [ True False False False]]

print("选定元素:", arr[mask])
# 输出: 选定元素: [4 5 6 7 8]

# 选择小于 2 或大于 9 的元素
print("选定元素 (< 2 | > 9):", arr[(arr < 2) | (arr > 9)])
# 输出: 选定元素 (< 2 | > 9): [ 0  1 10 11]

# 选择不大于 5 的元素
print("选定元素 (~( > 5)):", arr[~(arr > 5)])
# 输出: 选定元素 (~( > 5)): [0 1 2 3 4 5]
```

#### 布尔索引创建副本

与基本切片不同，布尔索引总是创建数据的 *副本*，而不是视图。修改布尔索引操作的结果不会影响原始数组。

```python
arr_original = np.arange(5)
bool_selection = arr_original[arr_original > 2] # 元素 3, 4

print(f"原始数组: {arr_original}")
print(f"选定内容: {bool_selection}")

bool_selection[0] = 99 # 修改选定内容

print(f"修改后选定内容: {bool_selection}")
print(f"修改后原始数组: {arr_original}") # 原始数组保持不变
# 输出:
# 原始数组: [0 1 2 3 4]
# 选定内容: [3 4]
# 修改后选定内容: [99  4]
# 修改后原始数组: [0 1 2 3 4]
```

### 花式索引

花式索引允许你使用整数数组来指定要选择的索引。这为选择任意元素或重新排序元素提供了极大的灵活性。

#### 一维花式索引

传入一个索引列表或 NumPy 数组，以按所需顺序选择特定元素。

```python
arr1d = np.array([10, 20, 30, 40, 50, 60])

# 选择索引 1、3 和 4 处的元素
indices = [1, 3, 4]
selected = arr1d[indices]
print(selected)
# 输出: array([20, 40, 50])

# 以不同顺序选择元素，包括重复项
indices = np.array([5, 0, 5, 2])
selected_ordered = arr1d[indices]
print(selected_ordered)
# 输出: array([60, 10, 60, 30])
```

你还可以使用花式索引来修改特定元素：

```python
arr1d[[0, 2]] = -99 # 修改索引 0 和 2 处的元素
print(arr1d)
# 输出: array([-99,  20, -99,  40,  50,  60])
```

#### 多维花式索引

花式索引在多个维度中也适用。你可以为每个维度提供索引数组。

```python
arr2d = np.arange(12).reshape((3, 4))
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# 选择特定行（例如，第 0 行和第 2 行）
selected_rows = arr2d[[0, 2]]
print("选定行:\n", selected_rows)
# 输出:
# 选定行:
# [[ 0  1  2  3]
#  [ 8  9 10 11]]

# 选择特定列（例如，第 1 列和第 3 列）
selected_cols = arr2d[:, [1, 3]]
print("选定列:\n", selected_cols)
# 输出:
# 选定列:
# [[ 1  3]
#  [ 5  7]
#  [ 9 11]]
```

一个常见用法是使用行和列索引对选择特定元素。如果你提供两个索引数组 `rows` 和 `cols`，NumPy 将它们配对：`(rows[0], cols[0])`、`(rows[1], cols[1])` 等。结果数组的形状与索引数组的形状匹配。

```python
# 选择坐标 (0, 1)、(1, 2) 和 (2, 0) 处的元素
rows = np.array([0, 1, 2])
cols = np.array([1, 2, 0])

selected_elements = arr2d[rows, cols]
print("在 (0,1), (1,2), (2,0) 处选择的元素:", selected_elements)
# 输出: 在 (0,1), (1,2), (2,0) 处选择的元素: [1 6 8]
```

#### 花式索引也会创建副本

与布尔索引类似，花式索引通常创建数据的 *副本*，而不是视图。

```python
arr_original = np.arange(10)
fancy_selection = arr_original[[1, 5, 8]]

print(f"原始数组: {arr_original}")
print(f"选定内容: {fancy_selection}")

fancy_selection[0] = 99 # 修改选定内容

print(f"修改后选定内容: {fancy_selection}")
print(f"修改后原始数组: {arr_original}") # 原始数组保持不变
# 输出:
# 原始数组: [0 1 2 3 4 5 6 7 8 9]
# 选定内容: [1 5 8]
# 修改后选定内容: [99  5  8]
# 修改后原始数组: [0 1 2 3 4 5 6 7 8 9]
```

### 组合索引技术

你可以组合基本切片、布尔索引和花式索引来执行更复杂的选择。请记住关于视图与副本的规则，这取决于使用的 *主要* 索引类型。通常，如果涉及花式索引或布尔索引，你会得到一个副本。

```python
arr2d = np.arange(12).reshape((3, 4))
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# 选择第 0 行和第 2 行，以及第 1 到 3 列（不包含）
# 行使用花式索引，列使用切片
subset1 = arr2d[[0, 2], 1:3]
print("行 [0, 2], 列 1:3:\n", subset1)
# 输出:
# 行 [0, 2], 列 1:3:
# [[ 1  2]
#  [ 9 10]]

# 从第 1 行选择元素，其中该行中的元素大于 5
# 行使用基本索引，列使用布尔索引
subset2 = arr2d[1, arr2d[1] > 5]
print("第 1 行中大于 5 的元素:", subset2)
# 输出: 第 1 行中大于 5 的元素: [6 7]
```

掌握这些索引和切片技术是选择、过滤和修改数据中特定部分的基础，这些操作对数据分析和为机器学习 (machine learning)模型准备数据都是必要的。理解视图和副本之间的区别尤其重要，这有助于避免意外行为并高效管理内存。

## 参考资料

- [NumPy indexing](https://numpy.org/doc/stable/user/basics.indexing.html) — NumPy developers (2024)
  NumPy 索引和切片机制的权威指南，涵盖一维、多维、布尔和高级索引，并解释了视图与副本的区别。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  学习使用 NumPy 和 Pandas 进行数据操作的基础资源，提供了数组索引、切片及相关概念的详细解释和实用示例。
- [1.2. NumPy: powerful N-dimensional arrays](https://scipy-lectures.org/intro/numpy/array_object.html#indexing-and-slicing) — Emmanuelle Gouillart, Gaël Varoquaux, and contributors (2024)
  Journal: SciPy Lecture Notes
  一份全面的教育资源，介绍了 NumPy 数组对象，其中包括对各种索引和切片技术的清晰解释。

---

[上一节](02-%E6%95%B0%E7%BB%84%E5%88%9B%E5%BB%BA%E6%96%B9%E6%B3%95.md) · [下一节](04-%E6%95%B0%E7%BB%84%E8%AE%A1%E7%AE%97%E4%B8%8E%E9%80%9A%E7%94%A8%E5%87%BD%E6%95%B0.md)

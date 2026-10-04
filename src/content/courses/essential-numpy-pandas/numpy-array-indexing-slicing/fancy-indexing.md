---
course: "essential-numpy-pandas"
chapter: "numpy-array-indexing-slicing"
lesson: "fancy-indexing"
sourceId: 1720
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-3-numpy-array-indexing-slicing/fancy-indexing"
title: "花式索引"
description: "使用索引数组选择元素，访问非连续元素。"
order: 5
plots: []
sourceHash: "e7317f3da6dbc11a11e6c7bc96eb3b7eae7a69e54eb72d10d4a64e334687ca50"
sourceCorrections: []
---

花式索引提供了一种精确的方法，用于根据索引位置选择数组中的特定元素，尤其当这些位置不连续时。与选择连续元素块的切片不同，也与根据条件筛选的布尔索引不同，花式索引提供了一种灵活的方式来检索单独指定的项目。这种技术使用包含整数索引的NumPy数组（或Python列表）来从另一个数组中挑选元素。

### 使用索引数组选择元素

核心思路很简单：你将一个索引数组或列表传递给要从中选择的数组。

#### 一维数组中的花式索引

我们从一个简单的一维数组开始：

```python
import numpy as np

# 创建一个一维数组
arr = np.arange(10) * 5
print(f"Original array:\n{arr}")
# Output:
# Original array:
# [ 0  5 10 15 20 25 30 35 40 45]

# 定义我们要选择的索引
indices = np.array([1, 5, 8, 2])
print(f"Indices to select: {indices}")
# Output:
# Indices to select: [1 5 8 2]

# 使用花式索引
selected_elements = arr[indices]
print(f"Selected elements:\n{selected_elements}")
# Output:
# Selected elements:
# [ 5 25 40 10]
```

请注意以下几点：

1. 我们在方括号 `[]` 中使用了 `indices`，它是一个包含 `[1, 5, 8, 2]` 的NumPy数组。
2. NumPy 从 `arr` 中获取了这些特定索引位置的元素。
3. 结果数组 `selected_elements` 包含元素 `arr[1]`、`arr[5]`、`arr[8]` 和 `arr[2]`。
4. 结果数组的形状与*索引数组* (`indices`) 的形状匹配，而不是原始数组 (`arr`) 的形状。结果中元素的顺序也与索引数组中的顺序一致。

如果需要，你可以多次使用同一个索引：

```python
indices_repeated = np.array([3, 3, 1, 8, 1])
print(f"Indices with repetition: {indices_repeated}")
# Output:
# Indices with repetition: [3 3 1 8 1]

repeated_selection = arr[indices_repeated]
print(f"Selection with repeated indices:\n{repeated_selection}")
# Output:
# Selection with repeated indices:
# [15 15  5 40  5]
```

#### 多维数组中的花式索引

花式索引在多维数组中更为通用。你可以传递多个索引数组，根据不同轴上的坐标来选择元素。

考虑一个二维数组：

```python
arr2d = np.array([[ 0,  1,  2,  3],
                  [ 4,  5,  6,  7],
                  [ 8,  9, 10, 11],
                  [12, 13, 14, 15]])

print(f"Original 2D array:\n{arr2d}")
```

**选择特定行：**
如果你提供一个索引数组，它会选择与这些索引对应的整行：

```python
# 选择第0行和第2行
row_indices = np.array([0, 2])
selected_rows = arr2d[row_indices]

print(f"Selected rows (0 and 2):\n{selected_rows}")
# Output:
# Selected rows (0 and 2):
# [[ 0  1  2  3]
#  [ 8  9 10 11]]
```

**选择特定元素（坐标）：**
要使用坐标选择单个元素，你需要为*每个*维度提供一个索引数组。这些数组通常应具有相同的长度。NumPy 将索引逐个配对：选择的第一个元素在 `(row_indices[0], col_indices[0])`，第二个在 `(row_indices[1], col_indices[1])`，依此类推。

```python
# 行的索引
row_indices = np.array([0, 3, 1, 2])
# 列的索引
col_indices = np.array([1, 2, 0, 3])

# 选择坐标 (0,1), (3,2), (1,0), (2,3) 处的元素
selected_coords = arr2d[row_indices, col_indices]

print(f"Selected elements by coordinates:\n{selected_coords}")
# Output:
# Selected elements by coordinates:
# [ 1 14  4 11]
```

这里，选定的元素是 `arr2d[0, 1]`（即1），`arr2d[3, 2]`（即14），`arr2d[1, 0]`（即4）和 `arr2d[2, 3]`（即11）。结果是一个一维数组，因为我们提供了特定的坐标。

**花式索引与切片或基本索引的组合：**
你也可以将花式索引与切片或基本索引混合使用。例如，要选择特定行和特定列，你可以将它们组合起来：

```python
# 选择第0行和第2行，以及第1列和第3列
row_indices = np.array([0, 2])
col_indices = np.array([1, 3])

# 方法一：先选择行，然后从结果中选择列
selected_block_1 = arr2d[row_indices][:, col_indices]
print(f"Selected block (Method 1):\n{selected_block_1}")
# Output:
# Selected block (Method 1):
# [[ 1  3]
#  [ 9 11]]

# 方法二：使用 np.ix_ 进行索引广播（更高级）
# 这个辅助函数创建索引器，用于选择一个矩形区域
selected_block_2 = arr2d[np.ix_(row_indices, col_indices)]
print(f"Selected block (Method 2 with np.ix_):\n{selected_block_2}")
# Output:
# Selected block (Method 2 with np.ix_):
# [[ 1  3]
#  [ 9 11]]
```

两种方法都生成了一个子数组，其中包含行 0 和 2 与列 1 和 3 交汇处的元素。对于初学者来说，第一种方法通常更直观，它首先选择行，然后从中间结果中选择列。

你也可以选择特定行和*所有*列，或者从*所有*行中选择特定列：

```python
# 选择第1行和第3行，所有列
print(f"Rows 1 and 3, all columns:\n{arr2d[[1, 3]]}")
# Output:
# Rows 1 and 3, all columns:
# [[ 4  5  6  7]
#  [12 13 14 15]]

# 所有行，第0列和第2列
print(f"All rows, columns 0 and 2:\n{arr2d[:, [0, 2]]}")
# Output:
# All rows, columns 0 and 2:
# [[ 0  2]
#  [ 4  6]
#  [ 8 10]
#  [12 14]]
```

### 使用花式索引修改数组

就像基本索引和切片一样，花式索引也可用于修改数组元素。你将花式索引表达式放在赋值语句的左侧：

```python
arr = np.arange(10) * 5
print(f"Original array: {arr}")

indices = np.array([1, 5, 8, 2])
arr[indices] = 99 # 将单个值赋给所有选定的元素
print(f"Array after modifying elements at {indices}: {arr}")
# Output:
# Original array: [ 0  5 10 15 20 25 30 35 40 45]
# Array after modifying elements at [1 5 8 2]: [ 0 99 99 15 20 99 30 35 99 45]

# 你也可以赋值一个值数组，其形状与索引数组匹配
arr[indices] = np.array([-1, -2, -3, -4])
print(f"Array after assigning multiple values: {arr}")
# Output:
# Array after assigning multiple values: [ 0 -1 -4 15 20 -2 30 35 -3 45]
```

这同样适用于多维数组：

```python
print(f"Original 2D array:\n{arr2d}")

# 修改 (0,1), (3,2), (1,0), (2,3) 处的元素
row_indices = np.array([0, 3, 1, 2])
col_indices = np.array([1, 2, 0, 3])
arr2d[row_indices, col_indices] = 0

print(f"2D array after setting specific elements to 0:\n{arr2d}")
# Output:
# Original 2D array:
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]
#  [12 13 14 15]]
# 2D array after setting specific elements to 0:
# [[ 0  0  2  3]
#  [ 0  5  6  7]
#  [ 8  9 10  0]
#  [12 13  0 15]]
```

### 重要提示：副本与视图

切片和花式索引之间一个显著的区别是，**花式索引几乎总是返回数据的*副本*，而不是视图**。这意味着对花式索引返回的数组所做的更改*不会*影响原始数组。

```python
arr = np.arange(10)
print(f"Original array: {arr}")

# 切片（创建视图）
slice_view = arr[2:5]
print(f"Slice view: {slice_view}")
slice_view[0] = 99 # 修改视图
print(f"Original array after modifying slice view: {arr}") # 原始数组已改变

# 花式索引（创建副本）
indices = np.array([6, 8])
fancy_copy = arr[indices]
print(f"Fancy copy: {fancy_copy}")
fancy_copy[0] = -1 # 修改副本
print(f"Original array after modifying fancy copy: {arr}") # 原始数组未改变

# Output:
# Original array: [0 1 2 3 4 5 6 7 8 9]
# Slice view: [2 3 4]
# Original array after modifying slice view: [ 0  1 99  3  4  5  6  7  8  9]
# Fancy copy: [6 8]
# Original array after modifying fancy copy: [ 0  1 99  3  4  5  6  7  8  9]
```

记住这种副本行为很重要，特别是在你打算修改原始数组数据时。如果你需要使用索引数组修改原始数组，请像前面所示，将花式索引表达式直接放在赋值语句的左侧。

花式索引提供了一种强大且灵活的方法，可以根据元素的位置访问和操作数组元素，补充了基本切片和布尔索引的功能。

## 参考资料

- [Indexing](https://numpy.org/doc/stable/user/basics.indexing.html) — NumPy Developers (2023)
  这个官方文档章节详细解释了NumPy的索引功能,包括花式索引,并阐明了视图和副本之间的区别。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本广泛使用的Python数据科学教材,对NumPy数组索引(包括花式索引)及其数据副本行为提供了清晰的解释。
- [The NumPy array object (from SciPy Lecture Notes)](https://scipy-lectures.org/intro/numpy/array_object.html) — The SciPy Lecture Notes Team (2023)
  一个广泛使用的开源教育资源,提供NumPy的实用教程,包括花式索引的清晰示例和解释。

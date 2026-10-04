---
course: "essential-numpy-pandas"
chapter: "numpy-array-indexing-slicing"
lesson: "slicing-2d-arrays"
sourceId: 1714
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-3-numpy-array-indexing-slicing/slicing-2d-arrays"
title: "二维数组切片"
description: "从二维数组中选择特定行、列或矩形子区域。"
order: 3
plots: []
sourceHash: "760e373337ee00f60cc629470007cc4d1fdca5b03c35db54424757091c2d953d"
sourceCorrections: []
---

切片是一种强大的操作，适用于一维和多维数组。一维数组用于提取元素序列，而多维数组则允许选择特定的行、列或矩形数据块。这项功能对于处理NumPy中表示的矩阵式数据（如图像或表格）来说非常基本。

对于二维数组（矩阵），切片通过使用逗号分隔每个维度的切片规范，从而扩展了熟悉的 `start:stop:step` 符号。一般语法是：

`array[row_slice, column_slice]`

在这里，`row_slice` 指定要选择的行范围，`column_slice` 指定要选择的列范围。每个切片遵循与一维切片相同的规则：`start` 索引是包含的，`stop` 索引是不包含的，`step` 定义了步长。如果切片的任何部分（`start`、`stop` 或 `step`）被省略，NumPy 会使用默认值（维度开头、维度结尾、步长为1）。

我们使用一个示例二维数组来看看实际效果。

```python
import numpy as np

# 创建一个 4x5 数组（4 行，5 列）
arr2d = np.array([[ 1,  2,  3,  4,  5],
                  [ 6,  7,  8,  9, 10],
                  [11, 12, 13, 14, 15],
                  [16, 17, 18, 19, 20]])

print("原始二维数组:\n", arr2d)
```

### 选择行

要选择行范围同时保留所有列，你需要为行指定切片，并对列使用 `:`。

```python
# 选择前两行（索引 0 和 1）
first_two_rows = arr2d[0:2, :]
print("\n前两行:\n", first_two_rows)

# 选择从索引 1 开始（包含）到索引 3 结束（不包含）的行
rows_1_and_2 = arr2d[1:3, :]
print("\n索引 1 和 2 的行:\n", rows_1_and_2)
```

冒号 `:` 充当列维度的通配符，表明所选行应包含所有列。如果你选择整行，也可以省略逗号后的冒号：`arr2d[0:2]` 等同于 `arr2d[0:2, :]`。

### 选择列

类似地，要选择特定列同时保留所有行，你需要对行切片使用 `:` 并指定所需的列范围。

```python
# 选择前三列（索引 0, 1, 2）
first_three_cols = arr2d[:, 0:3]
print("\n前三列:\n", first_three_cols)

# 选择从索引 2 开始（包含）到索引 4 结束（不包含）的列
cols_2_and_3 = arr2d[:, 2:4]
print("\n索引 2 和 3 的列:\n", cols_2_and_3)
```

### 选择子数组（矩形区域）

真正的优势在于你结合行和列切片来选择数组的特定矩形子区域。

```python
# 选择从行 0, 1 和列 1, 2 组成的子数组
sub_array_1 = arr2d[0:2, 1:3]
print("\n子数组 (行 0-1, 列 1-2):\n", sub_array_1)
# 结果: [[2, 3], [7, 8]]

# 选择从行 1, 2 和列 2, 3, 4 组成的子数组
sub_array_2 = arr2d[1:3, 2:5] # or arr2d[1:3, 2:]
print("\n子数组 (行 1-2, 列 2-4):\n", sub_array_2)
# 结果: [[8, 9, 10], [13, 14, 15]]
```

以下图表展示了对 `arr2d[1:3, 2:4]` 定义的区域进行切片操作，该操作对应于索引为 1 和 2 的行，以及索引为 2 和 3 的列。

> 对 `arr2d[1:3, 2:4]` 进行切片的视觉表示。蓝色高亮显示的元素构成了结果 2x2 子数组，其中包含 `[[8, 9], [13, 14]]`。

### 切片是视图

需要记住的一个重要方面是，与一维数组切片一致，二维数组切片是原始数组数据的*视图*，而不是副本。这意味着如果你修改切片中的元素，这些更改将反映在原始数组中。

```python
# 获取一个切片（视图）
slice_view = arr2d[0:2, 0:2]
print("\n原始切片视图:\n", slice_view)

# 修改切片中的一个元素
slice_view[0, 0] = 99
print("\n修改后的切片视图:\n", slice_view)

# 检查原始数组
print("\n修改切片后的原始数组:\n", arr2d)
# 注意左上角元素（索引 0,0）现在是 99
```

如果你需要切片的独立副本，并且它独立于原始数组，你必须明确使用 `.copy()` 方法。

```python
# 创建切片的副本
slice_copy = arr2d[2:, 2:].copy()
print("\n切片副本:\n", slice_copy)

# 修改副本
slice_copy[0, 0] = -1
print("\n修改后的切片副本:\n", slice_copy)

# 再次检查原始数组 - 该区域保持不变
print("\n修改副本后的原始数组:\n", arr2d)
```

熟练掌握二维切片对于在 NumPy 中高效处理表格数据、图像块或任何网格状结构非常重要。它允许你隔离并处理数据的特定部分，而无需手动迭代元素。

## 参考资料

- [Indexing](https://numpy.org/doc/stable/user/basics.indexing.html) — NumPy Developers (2024)
  NumPy官方文档，介绍其索引和切片机制，包括多维数组以及视图与副本的区别。
- [The N-dimensional array (ndarray)](https://numpy.org/doc/stable/reference/arrays.ndarray.html) — NumPy Developers (2024)
  NumPy ndarray对象的参考文档，为数组数据结构和切片如何创建视图提供背景信息。

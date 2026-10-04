---
course: "essential-numpy-pandas"
chapter: "numpy-array-indexing-slicing"
lesson: "selecting-array-data-practice"
sourceId: 1725
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-3-numpy-array-indexing-slicing/selecting-array-data-practice"
title: "动手实践：从数组中选择数据"
description: "应用各种索引和切片技术来选择和修改 NumPy 数组中的数据。"
order: 7
plots: []
sourceHash: "3fc9ff3d31acfdc4986fe9bd832851546a687d441ff75e7e4e2d9a37bffa1a09"
sourceCorrections: []
---

NumPy 数组中数据访问和修改的各种选择方法的实际应用。这包括基本索引、切片、布尔索引和花式索引。

首先，请确保已导入 NumPy。如果您正在 Jupyter Notebook 或 Python 脚本中进行操作，请从这一行开始：

```python
import numpy as np
```

### 准备示例数组

我们需要一些数组来操作。让我们创建一个简单的一维数组和一个二维数组。

```python
# 一维数组
arr1d = np.arange(10) # 创建数组 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print("一维数组:")
print(arr1d)

# 二维数组 (3 行, 4 列)
arr2d = np.array([[ 1,  2,  3,  4],
                  [ 5,  6,  7,  8],
                  [ 9, 10, 11, 12]])
print("\n二维数组:")
print(arr2d)
```

### 基本索引和切片

请记住，索引用于访问单个元素，而切片用于提取子数组。

**练习 1：访问元素**

1. 从 `arr1d` 中获取索引为 3 的元素。
2. 从 `arr2d` 中获取第二行（索引 1）第三列（索引 2）的元素。

```python
# 1. arr1d 中索引为 3 的元素
element_1d = arr1d[3]
print(f"arr1d 中索引为 3 的元素: {element_1d}")

# 2. arr2d 中第 1 行第 2 列的元素
element_2d = arr2d[1, 2] # 或者 arr2d[1][2]
print(f"arr2d 中第 1 行第 2 列的元素: {element_2d}")
```

**练习 2：数组切片**

1. 从 `arr1d` 中获取前 5 个元素。
2. 从 `arr1d` 中获取从索引 5 开始的所有元素。
3. 从 `arr2d` 中获取前两行。
4. 从 `arr2d` 中获取前两列。
5. 从 `arr2d` 的右上角提取一个 2x2 的子数组（行 0-1，列 2-3）。

```python
# 1. arr1d 的前 5 个元素
slice1 = arr1d[:5]
print(f"\narr1d 的前 5 个元素: {slice1}")

# 2. arr1d 中从索引 5 开始的所有元素
slice2 = arr1d[5:]
print(f"arr1d 中从索引 5 开始的所有元素: {slice2}")

# 3. arr2d 的前两行
slice3 = arr2d[:2] # 行 0 和 1
print("\narr2d 的前两行:")
print(slice3)

# 4. arr2d 的前两列
slice4 = arr2d[:, :2] # 所有行，列 0 和 1
print("\narr2d 的前两列:")
print(slice4)

# 5. arr2d 的右上角 2x2 子数组
slice5 = arr2d[:2, 2:] # 行 0-1，列 2-3
print("\narr2d 的右上角 2x2 子数组:")
print(slice5)
```

### 布尔索引

布尔索引允许基于条件进行选择。

**练习 3：条件选择**

1. 从 `arr1d` 中选择所有大于 5 的元素。
2. 从 `arr2d` 中选择所有第一列（索引 0）的元素大于 4 的行。

```python
# 1. arr1d 中大于 5 的元素
bool_mask_1d = arr1d > 5
selected_1d = arr1d[bool_mask_1d]
print(f"\narr1d 中大于 5 的元素: {selected_1d}")
# 您也可以一步完成：arr1d[arr1d > 5]

# 2. arr2d 中第一个元素大于 4 的行
bool_mask_2d = arr2d[:, 0] > 4 # 对第一列施加条件
selected_rows_2d = arr2d[bool_mask_2d]
print("\narr2d 中第一个元素大于 4 的行:")
print(selected_rows_2d)
```

### 花式索引

花式索引使用索引数组来选择元素。

**练习 4：使用索引数组选择**

1. 从 `arr1d` 中选择索引为 1、3 和 7 的元素。
2. 从 `arr2d` 中选择索引为 0 和 2 的行。
3. 从 `arr2d` 中选择坐标为 (0, 1)、(1, 3) 和 (2, 0) 的元素。

```python
# 1. 从 arr1d 中选择特定索引
indices_1d = np.array([1, 3, 7])
selected_fancy_1d = arr1d[indices_1d]
print(f"\narr1d 中索引 {indices_1d} 处的元素: {selected_fancy_1d}")

# 2. 从 arr2d 中选择特定行
row_indices = np.array([0, 2])
selected_fancy_rows = arr2d[row_indices] # 选择整行
print("\narr2d 中索引 0 和 2 的行:")
print(selected_fancy_rows)

# 3. 使用坐标对从 arr2d 中选择特定元素
row_coords = np.array([0, 1, 2])
col_coords = np.array([1, 3, 0])
selected_fancy_elements = arr2d[row_coords, col_coords]
print(f"\narr2d 中坐标为 (0,1), (1,3), (2,0) 的元素: {selected_fancy_elements}")
```

### 修改数组子集

您可以使用任何这些索引方法对选定的部分赋值。请记住，切片通常返回*视图*，这意味着修改会影响原始数组，而花式索引通常返回*副本*。

**练习 5：更新数组值**

1. 将 `arr1d` 中索引为 0 的元素更改为 99。
2. 将 `arr1d` 的前 3 个元素更改为 100。
3. 将 `arr1d` 中所有偶数元素更改为 0。
4. 在 `arr2d` 中，将 (1, 1) 处的元素更改为 -5。
5. 在 `arr2d` 中，将整个最后一行（索引 2）更改为只包含值 13。

```python
print("\n修改前的原始 arr1d:", arr1d)
print("修改前的原始 arr2d:\n", arr2d)

# 1. 修改 arr1d 中的单个元素
arr1d_copy1 = arr1d.copy() # 在副本上操作，以保留原始数据供后续步骤使用
arr1d_copy1[0] = 99
print("\n1. arr1d 将索引 0 设置为 99 后的结果:", arr1d_copy1)

# 2. 修改 arr1d 中的切片
arr1d_copy2 = arr1d.copy()
arr1d_copy2[:3] = 100
print("2. arr1d 将前 3 个元素设置为 100 后的结果:", arr1d_copy2)

# 3. 根据条件修改 arr1d
arr1d_copy3 = arr1d.copy()
arr1d_copy3[arr1d_copy3 % 2 == 0] = 0 # 使用布尔索引进行赋值
print("3. arr1d 将偶数设置为 0 后的结果:", arr1d_copy3)

# 4. 修改 arr2d 中的单个元素
arr2d_copy1 = arr2d.copy()
arr2d_copy1[1, 1] = -5
print("\n4. arr2d 将元素 (1,1) 设置为 -5 后的结果:\n", arr2d_copy1)

# 5. 修改 arr2d 的整行
arr2d_copy2 = arr2d.copy()
arr2d_copy2[2] = 13 # 将标量赋值给整个行切片
# 或者：arr2d_copy2[2, :] = 13
print("5. arr2d 将最后一行设置为 13 后的结果:\n", arr2d_copy2)
```

这些练习涵盖了您在 NumPy 数组中操作数据的主要方式。熟练掌握这些选择和修改方法对于为数据分析或模型训练做准备至关重要。您还可以进一步尝试组合使用这些方法，例如，从通过布尔条件选择的行中选择特定列。

## 参考资料

- [NumPy User Guide: Indexing](https://numpy.org/doc/stable/user/basics.indexing.html) — NumPy Developers (2024)
  NumPy的官方用户指南，提供了所有索引和切片技术的详细信息。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本关于Python数据处理的权威书籍，其中包含专门解释NumPy数组、索引和切片的章节。
- [NumPy: the array object - Indexing and Slicing](https://scipy-lectures.org/intro/numpy/array_object.html#indexing-and-slicing) — Gaël Varoquaux, Emmanuelle Gouillart, and Olav Vahtras (2023)
  一个广泛使用的教育资源，提供了NumPy数组索引和切片清晰的解释和实际示例。

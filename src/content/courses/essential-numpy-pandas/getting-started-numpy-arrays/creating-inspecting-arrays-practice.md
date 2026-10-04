---
course: "essential-numpy-pandas"
chapter: "getting-started-numpy-arrays"
lesson: "creating-inspecting-arrays-practice"
sourceId: 1705
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-2-getting-started-numpy-arrays/creating-inspecting-arrays-practice"
title: "实践操作：创建和查看数组"
description: "一个实践环节，使用各种方法创建 NumPy 数组并检查它们的属性。"
order: 6
plots: ["plots/1705-0.json"]
sourceHash: "263ae0f6137ac9e8e66df9272d5f5a3e2d32d768167b897248b1ddb2142d37ac"
sourceCorrections: []
---

示例演示了使用多种方法创建多个 NumPy 数组。创建后，使用查看属性检查这些数组的结构。

首先，请确保已安装 NumPy 并将其导入。通常的约定是将其导入为别名 `np`。如果你在 Jupyter Notebook 或交互式 Python 环境中工作，请执行以下代码：

```python
import numpy as np
```

### 从 Python 列表创建数组

创建 NumPy 数组最直接的方法是从现有 Python 列表或元组创建。

**1. 一维数组：**
让我们从一个简单的整数列表开始。

```python
# 创建一个 Python 列表
list_1d = [1, 3, 5, 7, 9]

# 从列表中创建一个 NumPy 数组
array_1d = np.array(list_1d)

# 打印数组
print(array_1d)
```

输出：

```
[1 3 5 7 9]
```

现在，让我们查看其属性：

```python
print(f"类型: {type(array_1d)}")
print(f"数据类型 (dtype): {array_1d.dtype}")
print(f"维度 (ndim): {array_1d.ndim}")
print(f"形状: {array_1d.shape}")
print(f"大小: {array_1d.size}")
```

输出：

```
类型: <class 'numpy.ndarray'>
数据类型 (dtype): int64
维度 (ndim): 1
形状: (5,)
大小: 5
```

请注意，NumPy 将数据类型推断为 `int64`（一个64位整数，默认整数类型通常取决于你的系统）。它是一个1维数组（`ndim=1`），包含5个元素（`size=5`），其形状表示为 `(5,)`，表示沿着单个轴有5个元素。

**2. 二维数组：**
现在，让我们使用嵌套列表来创建一个2维数组（类似于矩阵）。

```python
# 创建一个嵌套的 Python 列表
list_2d = [[1, 2, 3], [4, 5, 6]]

# 创建一个 2D NumPy 数组
array_2d = np.array(list_2d)

# 打印数组
print(array_2d)
```

输出：

```
[[1 2 3]
 [4 5 6]]
```

让我们查看这个2维数组：

```python
print(f"数据类型 (dtype): {array_2d.dtype}")
print(f"维度 (ndim): {array_2d.ndim}")
print(f"形状: {array_2d.shape}")
print(f"大小: {array_2d.size}")
```

输出：

```
数据类型 (dtype): int64
维度 (ndim): 2
形状: (2, 3)
大小: 6
```

这次，NumPy 正确识别出它是一个2维数组（`ndim=2`）。形状 `(2, 3)` 告诉我们它有2行3列。元素总数（`size`）是 $2 \times 3 = 6$。

### 使用内置数组创建函数

NumPy 提供无需从 Python 列表开始创建数组的函数，这通常更高效。

**1. `np.zeros` 和 `np.ones`：**
这些函数分别创建填充零或一的数组。你需要将所需形状作为元组提供。

```python
# 创建一个 3x4 的全零数组
zeros_array = np.zeros((3, 4))
print("全零数组:
", zeros_array)
print(f"全零数组数据类型: {zeros_array.dtype}
") # 默认为 float64

# 创建一个包含 5 个一的 1D 数组，指定整数类型
ones_array = np.ones(5, dtype=np.int16)
print("全一数组:
", ones_array)
print(f"全一数组数据类型: {ones_array.dtype}")
print(f"全一数组形状: {ones_array.shape}")
```

输出：

```
全零数组:
 [[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
全零数组数据类型: float64

全一数组:
 [1 1 1 1 1]
全一数组数据类型: int16
全一数组形状: (5,)
```

注意 `np.zeros` 默认为浮点数（`float64`），而我们为 `np.ones` 明确要求了16位整数（`int16`）。

**2. `np.arange`：**
类似于 Python 的 `range`，但返回一个 NumPy 数组。

```python
# 创建一个数组，值从 0 开始到（但不包含）10，步长为 2
range_array = np.arange(0, 10, 2)
print("Arange 数组:
", range_array)
print(f"Arange 形状: {range_array.shape}")
print(f"Arange 数据类型: {range_array.dtype}")
```

输出：

```
Arange 数组:
 [0 2 4 6 8]
Arange 形状: (5,)
Arange 数据类型: int64
```

**3. `np.linspace`：**
创建一个数组，包含指定数量的元素，这些元素在起始值和结束值之间均匀间隔（包含两端）。

```python
# 创建一个数组，包含 6 个在 0 到 1 之间（包含两端）均匀间隔的值
linspace_array = np.linspace(0, 1, 6)
print("Linspace 数组:
", linspace_array)
print(f"Linspace 形状: {linspace_array.shape}")
print(f"Linspace 数据类型: {linspace_array.dtype}")
```

输出：

```
Linspace 数组:
 [0.  0.2 0.4 0.6 0.8 1. ]
Linspace 形状: (6,)
Linspace 数据类型: float64
```

### 理解数据类型 (dtypes)

NumPy 数组之所以高效，是因为所有元素通常共享相同的数据类型。让我们看看当我们创建数组并指定类型时会发生什么。

```python
# 从包含浮点数的列表创建数组
float_array = np.array([1.0, 2.5, 3.7, 4.2])
print("浮点数数组:", float_array)
print(f"数据类型: {float_array.dtype}
")

# 明确指定整数类型（会发生截断）
int_from_float = np.array([1.0, 2.5, 3.7, 4.2], dtype=np.int32)
print("从浮点数转换的整数数组:", int_from_float)
print(f"数据类型: {int_from_float.dtype}
")

# 从混合列表创建（NumPy 会向上转换到兼容类型）
mixed_array = np.array([1, 2.5, 3, 4.8])
print("混合数组:", mixed_array)
print(f"数据类型: {mixed_array.dtype}")
```

输出：

```
浮点数数组: [1.  2.5 3.7 4.2]
数据类型: float64

从浮点数转换的整数数组: [1 2 3 4]
数据类型: int32

混合数组: [1.  2.5 3.  4.8]
数据类型: float64
```

当我们强制将浮点数转换为 `int32` 数组时，小数部分被截断了。当从包含整数和浮点数的混合列表创建数组且未指定 `dtype` 时，NumPy 会智能地将所有元素向上转换为 `float64`，以容纳浮点数且不丢失信息。

### 数组结构可视化

有时，视觉表示有助于理解结构，特别是对于2维数组。让我们将之前创建的 `array_2d`（$[[1, 2, 3], [4, 5, 6]]$）可视化，它的形状是 `(2, 3)`。



![2x3 NumPy 数组的可视化](plots/1705-0.json)



> 热力图形象地表示了 `array_2d` 的2行3列。每个单元格对应数组中的一个元素，显示其值和位置。

本次实践环节涵盖了从列表创建 NumPy 数组，以及使用 `np.zeros`、`np.ones`、`np.arange` 和 `np.linspace` 等专用函数。我们还练习了查看数组的基本属性：`dtype`、`ndim`、`shape` 和 `size`。理解这些创建方法和属性是有效使用 NumPy 进行数值计算任务的重要前提。你现在已具备能力，可以创建 Python 中数据分析和科学计算的基本构成要素。

## 参考资料

- [NumPy User Guide: The Absolute Basics](https://numpy.org/doc/stable/user/absolute_beginners.html) — NumPy Developers (2024)
  提供 NumPy 的基础介绍，涵盖数组创建方法、形状、维度和数据类型等核心数组属性，以及基本操作。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本重要的数据处理参考书，其中关于 NumPy 的章节介绍数组创建、属性和基本操作。
- [Guide to NumPy](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEaHKCSIxoCbgLVtDshESAAGsqYfUhAbV4XptRPVQsc4emKm6GuRbL7_sFLcCJYvJBcJhQqLwA1YdMIcmadUvIZ8LUYKn5PXoR0tyI-dLqexRzSphuoiKGhakonN2qywAdjNjOdLULm-34shg==) — Travis E. Oliphant (2006)
  Publisher: Trelgol Publishing
  由 NumPy 的创建者撰写，本指南详细阐述该库的设计原则和 NumPy 数组的基本操作。

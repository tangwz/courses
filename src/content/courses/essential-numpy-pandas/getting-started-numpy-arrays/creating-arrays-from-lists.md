---
course: "essential-numpy-pandas"
chapter: "getting-started-numpy-arrays"
lesson: "creating-arrays-from-lists"
sourceId: 1695
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-2-getting-started-numpy-arrays/creating-arrays-from-lists"
title: "从 Python 列表创建数组"
description: "了解如何将 Python 列表和嵌套列表转换为 NumPy 数组。"
order: 2
plots: []
sourceHash: "75209165d87244076e62a1cfc58e5f0a0110b1e32263ded0aff3d2422d2feffb"
sourceCorrections: []
---

虽然 NumPy 提供了创建数组的专门函数（我们很快会看到），但一个非常常见的开始方法是将现有的 Python 数据结构，特别是列表，转换为 NumPy 数组。当你已经使用标准 Python 代码加载或生成了数据时，这通常是第一步。

用于此转换的主要函数是 `np.array()`。我们来看看它是如何工作的。

### 从列表创建一维数组

假设你有一个简单的 Python 数字列表：

```python
import numpy as np

# 一个标准的 Python 列表
python_list = [5, 10, 15, 20, 25]
print(f"原始 Python 列表: {python_list}")
print(f"类型: {type(python_list)}")

# 将列表转换为 NumPy 数组
numpy_array = np.array(python_list)

print(f"转换后的 NumPy 数组: {numpy_array}")
print(f"类型: {type(numpy_array)}")
print(f"数组的数据类型 (dtype): {numpy_array.dtype}")
```

当你运行这段代码时，你会注意到以下几点：

1. `np.array()` 函数将 Python 列表作为输入。
2. 输出的 `numpy_array` 不再是标准的 Python 列表，而是 `numpy.ndarray` 类型的一个对象。这就是我们提到的 N 维数组。
3. NumPy 自动推断了元素的数据类型。由于所有元素都是整数，因此生成的数组的 `dtype` 为 `int64`（具体的整数类型，如 `int32` 或 `int64`，可能会因你的系统而略有不同）。

这种自动类型检测很方便。如果你的列表包含浮点数，NumPy 将创建一个浮点数组：

```python
float_list = [1.0, 2.5, 3.7, 4.2]
float_array = np.array(float_list)
print(f"浮点数组: {float_array}")
print(f"浮点数组 dtype: {float_array.dtype}") # 可能是 float64
```

> 使用 `np.array()` 将一维 Python 列表转换为 NumPy ndarray。

### 从嵌套列表创建多维数组

NumPy 数组可以具有多个维度。你可以通过将嵌套的 Python 列表（包含其他列表的列表）传递给 `np.array()` 来创建多维数组。

```python
# 一个嵌套的 Python 列表（列表的列表）
nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(f"原始嵌套列表:\n{nested_list}")

# 转换为一个二维 NumPy 数组
numpy_2d_array = np.array(nested_list)

print(f"转换后的二维 NumPy 数组:\n{numpy_2d_array}")
print(f"数组形状: {numpy_2d_array.shape}") # 输出: (3, 3) -> 3 行, 3 列
print(f"数组维度: {numpy_2d_array.ndim}") # 输出: 2
print(f"数组 dtype: {numpy_2d_array.dtype}") # 输出: int64（通常）
```

这里，包含三个整数的三个列表组成的列表被转换为一个 3 行 3 列的二维数组（一个矩阵）。NumPy 会相应地排列数据。

> 将嵌套的 Python 列表转换为二维 NumPy ndarray。

为了让 NumPy 高效地创建标准的二维数组，内部列表必须具有相同的长度。如果它们不一致，NumPy 可能会创建一个 `dtype=object` 的数组，每个元素都是一个 Python 对象（如列表），这通常会丧失 NumPy 数组的性能优势。

```python
# 不规则列表的例子（谨慎使用）
uneven_list = [[1, 2], [3, 4, 5]]

# NumPy 可能会创建一个对象数组
# 注意：不同 NumPy 版本行为可能略有不同。
# 如有需要，显式设置 dtype=object 可确保此行为。
object_array = np.array(uneven_list, dtype=object)

print(f"不规则列表生成的数组:\n{object_array}")
print(f"数组 dtype: {object_array.dtype}") # 输出: object
```

### 指定数据类型 (`dtype`)

虽然 NumPy 的自动类型推断很有用，但有时你需要明确控制数组的数据类型。你可以使用 `np.array()` 函数中的 `dtype` 参数 (parameter)来指定所需的类型。这有助于：

- **内存管理：** 较小的数据类型（如 `float32` 而不是 `float64`，或 `int8` 而不是 `int64`）使用的内存更少。
- **精度：** 确保计算以足够的浮点精度执行。
- **兼容性：** 匹配其他库或函数所需的数据类型。

下面是如何从整数列表创建一个浮点数数组：

```python
integer_list = [1, 2, 3, 4]

# 从整数创建浮点数组
float_array = np.array(integer_list, dtype=float) # 或 np.float64, np.float32 等

print(f"原始整数列表: {integer_list}")
print(f"转换后的浮点数组: {float_array}")
print(f"数组 dtype: {float_array.dtype}") # 输出: float64（或指定的浮点类型）

# 创建一个较小整数类型的整数数组
int8_array = np.array(integer_list, dtype=np.int8)
print(f"转换后的 int8 数组: {int8_array}")
print(f"数组 dtype: {int8_array.dtype}") # 输出: int8
```

常见的 NumPy 数据类型包括 `np.int8`、`np.int16`、`np.int32`、`np.int64`、`np.uint8`（无符号整数）、`np.float32`、`np.float64`、`np.complex64`、`np.complex128`、`np.bool_` 和 `np.object_`。选择正确的 `dtype` 是一个重要考量，尤其是在处理大型数据集时。

转换 Python 列表是开始使用 NumPy 数组的一种基本方法，它提供了一条从标准 Python 数据结构到 NumPy 强大数值计算能力的途径。在下一节中，我们将查看 NumPy 提供的直接创建数组的函数，无需预先存在 Python 列表。

## 参考资料

- [numpy.array](https://numpy.org/doc/stable/reference/generated/numpy.array.html) — NumPy Developers (2023)
  `numpy.array` 函数的官方文档，详细说明其参数、用法以及创建数组的各种方法，包括从 Python 列表和嵌套列表创建。这是理解该函数行为的主要来源。
- [Array creation routines](https://numpy.org/doc/stable/user/basics.creation.html) — NumPy Developers (2023)
  NumPy 官方用户指南的一部分，本节概述了如何创建数组，将 `np.array()` 置于其他数组创建函数的背景中。它还讨论了数据类型推断和显式 `dtype` 规范。

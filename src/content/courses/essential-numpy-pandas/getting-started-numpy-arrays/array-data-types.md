---
course: "essential-numpy-pandas"
chapter: "getting-started-numpy-arrays"
lesson: "array-data-types"
sourceId: 1700
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-2-getting-started-numpy-arrays/array-data-types"
title: "理解数组数据类型"
description: "了解NumPy中可用的不同数值数据类型以及如何指定它们。"
order: 4
plots: []
sourceHash: "6667b13f9218e64fe0fceeea0e0f4db53e98dab8f56ef48693e130b3e379c89e"
sourceCorrections: []
---

NumPy数组的一个基本特点是它们是同质的；单个数组中的所有元素必须是相同的数据类型。这与标准的Python列表不同，后者可以在同一个列表中包含各种类型的元素（如整数、字符串和浮点数）。这种同质性是NumPy高效能与性能的基础，使得数值运算能够进行优化、低层次的实现。

数组元素的具体数据类型存储在一个名为 `dtype`（数据类型的缩写）的特殊属性中。理解并有时控制 `dtype` 对于以下几个方面很重要：

1. **内存占用：** 不同的数据类型需要不同的内存量。一个64位整数（`int64`）比32位整数（`int32`）或8位整数（`int8`）使用更多的内存。选择最小的合适类型可以显著减少大型数组的内存占用。
2. **性能：** 对特定类型数组（特别是数值类型）的操作可以由底层的C或Fortran代码更快地执行，因为这些操作无需标准Python中所需的类型检查开销。
3. **精度和范围：** 浮点类型（`float32`，`float64`）提供不同级别的精度。整数类型能表示的数值范围也不同。选择正确的类型可以确保计算准确并避免潜在的溢出错误。

### 常见NumPy数据类型

NumPy支持比标准Python更丰富的数值类型。以下是一些最常见的类型：

| 类型 | 字符串代码 | 说明 | 示例 |
| --- | --- | --- | --- |
| `int8`, `int16`, `int32`, `int64` | `i1`, `i2`, `i4`, `i8` | 有符号整数（8、16、32或64位） | `-128` 到 `127` (`i1`) |
| `uint8`, `uint16`, `uint32`, `uint64` | `u1`, `u2`, `u4`, `u8` | 无符号整数（非负数） | `0` 到 `255` (`u1`) |
| `float16`, `float32`, `float64` | `f2`, `f4`, `f8` | 浮点数（半精度、单精度、双精度） | `3.14159` |
| `complex64`, `complex128` | `c8`, `c16` | 复数（由两个32位或64位浮点数表示） | `1 + 2j` |
| `bool` | `?` | 布尔类型，存储 `True` 和 `False` 值 | `True` |
| `object` | `O` | Python对象类型 | `(1, 'a')`, `[1, 2]` |
| `string_` | `S` | 固定长度ASCII字符串类型（例如，`S10`表示长度为10的字符串） | `'numpy'` |
| `unicode_` | `U` | 固定长度Unicode类型（例如，`U10`表示长度为10的字符串） | `'你好'` |

*注意：* 默认的整数类型（`int_`）和浮点类型（`float_`）通常分别对应`int64`和`float64`，这取决于您的系统架构。但在需要特定精度或大小时，明确指定是一个好的做法。

### 查看数组的数据类型

您可以使用NumPy数组的`dtype`属性轻松查看其数据类型。

```python
import numpy as np

# 从整数列表创建一个数组
arr_int = np.array([1, 2, 3, 4])
print(f"Array: {arr_int}")
print(f"Data Type: {arr_int.dtype}")

# 从包含浮点数的列表创建一个数组
arr_float = np.array([1.0, 2.5, 3.0, 4.8])
print(f"\nArray: {arr_float}")
print(f"Data Type: {arr_float.dtype}")

# 如果类型混合，NumPy会自动进行类型提升
arr_mixed = np.array([1, 2, 3.5, 4]) # 包含整数和一个浮点数
print(f"\nArray: {arr_mixed}")
print(f"Data Type: {arr_mixed.dtype}") # 结果dtype是float64
```

Output:

```
Array: [1 2 3 4]
Data Type: int64

Array: [1.  2.5 3.  4.8]
Data Type: float64

Array: [1.  2.  3.5 4. ]
Data Type: float64
```

请注意最后一个例子，由于列表中既包含整数又包含浮点数，NumPy自动推断出可以容纳所有元素的最通用类型，在本例中是`float64`。

### 创建时指定数据类型

您不必依赖NumPy的自动推断。在创建数组时，可以使用`dtype`参数 (parameter)明确指定所需的数据类型。这对于控制内存占用或确保特定精度很有用。

```python
import numpy as np

# 指定float32
arr_float32 = np.array([1, 2, 3], dtype=np.float32)
print(f"Array: {arr_float32}")
print(f"Data Type: {arr_float32.dtype}")

# 指定int8（注意范围）
arr_int8 = np.array([10, 20, 127], dtype=np.int8)
print(f"\nArray: {arr_int8}")
print(f"Data Type: {arr_int8.dtype}")

# 使用字符串代码也有效
arr_complex = np.array([1+1j, 2+2j], dtype='c8') # complex64
print(f"\nArray: {arr_complex}")
print(f"Data Type: {arr_complex.dtype}")

# 使用创建函数
zeros_uint16 = np.zeros(5, dtype=np.uint16)
print(f"\nArray: {zeros_uint16}")
print(f"Data Type: {zeros_uint16.dtype}")
```

Output:

```
Array: [1. 2. 3.]
Data Type: float32

Array: [ 10  20 127]
Data Type: int8

Array: [1.+1.j 2.+2.j]
Data Type: complex64

Array: [0 0 0 0 0]
Data Type: uint16
```

### 使用`astype`更改数据类型

有时，您需要将现有数组转换为不同的数据类型。`astype()`方法会创建一个*新*数组，其类型为您指定的类型，同时复制原始数据并根据需要进行类型转换。它不会修改原始数组，除非您将结果重新赋值回原始变量名。

```python
import numpy as np

arr_float = np.array([1.1, 2.7, 3.5, 4.9])
print(f"Original Array: {arr_float}")
print(f"Original dtype: {arr_float.dtype}")

# 转换为整数类型（截断小数部分）
arr_int = arr_float.astype(np.int32)
print(f"\nConverted to int32: {arr_int}")
print(f"New dtype: {arr_int.dtype}")

# 将整数数组转换为布尔类型
arr_num = np.array([0, 1, 5, 0, -2])
print(f"\nNumeric Array: {arr_num}")
arr_bool = arr_num.astype(np.bool_) # 零变为False，非零变为True
print(f"Converted to bool: {arr_bool}")
print(f"New dtype: {arr_bool.dtype}")

# 将整数数组转换为字符串
arr_str = arr_int.astype(np.string_)
print(f"\nConverted to string: {arr_str}")
print(f"New dtype: {arr_str.dtype}")
```

Output:

```
Original Array: [1.1 2.7 3.5 4.9]
Original dtype: float64

Converted to int32: [1 2 3 4]
New dtype: int32

Numeric Array: [ 0  1  5  0 -2]
Converted to bool: [False  True  True False  True]
New dtype: bool

Converted to string: [b'1' b'2' b'3' b'4']
New dtype: |S11
```

**注意：** 使用 `astype()` 时请注意。将浮点数转换为整数会截断小数部分，而不是四舍五入。将高精度类型（如`float64`）转换为低精度类型（如`float32`）可能导致精度丢失。将数据转换为范围较小的类型（如`int64`转换为`int8`）可能导致意料之外的结果或错误，如果值超出目标类型的限制。

理解NumPy的数据类型是一个基本步骤。它让您能够通过做出明智的选择，来存储和处理您的数值数据，从而编写出更节省内存、运行更快的代码。随着您处理更大规模的数据集，选择正确 `dtype` 的影响会变得越来越重要。

## 参考资料

- [Data types](https://numpy.org/doc/stable/user/basics.html#data-types) — NumPy Developers (2023)
  官方且最具权威性的资源，用于理解 NumPy 数组的数据类型，详细说明其特性、内存使用情况以及如何有效地管理它们。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本关于使用 Python 进行数据分析的基准书籍，为 NumPy 数组及其数据类型在数据处理工作流中的实际应用和解释提供了指导。

---
course: "intermediate-python-programming-ml"
chapter: "numerical-computing-numpy"
lesson: "intro-numpy-arrays"
sourceId: 2109
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-2-numerical-computing-numpy/intro-numpy-arrays"
title: "NumPy 数组简介"
description: "理解 NumPy 的 ndarray 对象、其属性以及相对于标准 Python 列表的优势。"
order: 1
plots: []
sourceHash: "2be887ea1e853f7154a98c6ca3fcb2ff9828c2a566fa8f06bc42129149a8c077"
sourceCorrections: []
---

Python 列表虽然灵活，但在机器学习 (machine learning)中进行数值计算时，其性能表现不足。它们是可能分散在内存中的不同 Python 对象的集合，导致按元素进行的算术运算效率低下。NumPy 通过提供一种专门的数据结构——N维数组，即 `ndarray`——解决了这个问题。

`ndarray` 是一个值网格，所有值类型相同，并由非负整数元组索引。可将其视为同质数据的容器。这种同质性与 Python 列表有重要区别，是 NumPy 效率的核心。由于所有元素类型相同（例如，所有都是 64 位浮点数或 32 位整数），NumPy 可以将它们存储在连续的内存块中。这使得操作可以通过预编译的 C 代码快速执行，通常称之为*向量 (vector)化操作*，从而避免了 Python 循环和对每个元素进行类型检查带来的开销。

比较主要区别：

- **Python 列表：** 可容纳不同类型的元素（例如，`[1, "two", 3.0]`）。元素是 Python 对象，可能在内存中相距很远。操作通常需要显式的 Python 循环。
- **NumPy 数组 (`ndarray`)：** 容纳单一的指定数据类型元素（例如，`int64`、`float32`）。元素在内存中紧凑存储。操作通常是向量化 (quantization)的，以 C 语言的速度一次性作用于整个数组。

我们来看一个基本示例。可以从 Python 列表创建 NumPy 数组：

```python
import numpy as np

# 创建一个 Python 列表
my_list = [1, 2, 3, 4, 5]

# 从列表中创建一个 NumPy 数组
my_array = np.array(my_list)

print(my_array)
print(type(my_array))
```

执行此代码将输出：

```
[1 2 3 4 5]
<class 'numpy.ndarray'>
```

请注意，输出 `[1 2 3 4 5]` 没有像 Python 列表表示那样带有逗号。这是一个不明显的视觉提示，表明你正在处理 NumPy 数组。

### NumPy 数组的属性

每个 `ndarray` 实例都有描述其结构和数据类型的重要属性。理解这些属性是有效使用 NumPy 所必需的。我们来创建一个稍微复杂一点的数组来展示：

```python
# 创建一个二维数组（一个矩阵）
matrix_array = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]])

print(f"数组:\n{matrix_array}\n")
print(f"维度数量 (ndim): {matrix_array.ndim}")
print(f"数组形状 (shape): {matrix_array.shape}")
print(f"元素总数 (size): {matrix_array.size}")
print(f"元素数据类型 (dtype): {matrix_array.dtype}")
print(f"每个元素的字节大小 (itemsize): {matrix_array.itemsize}")
print(f"元素占用的总字节数 (nbytes): {matrix_array.nbytes}")
```

输出展示了这些属性：

```
Array:
[[1. 2. 3.]
 [4. 5. 6.]]

Number of dimensions (ndim): 2
Shape of the array (shape): (2, 3)
Total number of elements (size): 6
Data type of elements (dtype): float64
Size in bytes of each element (itemsize): 8
Total bytes consumed by elements (nbytes): 48
```

我们来分析这些属性的含义：

- `ndim`：数组的轴（维度）数量。我们的 `matrix_array` 是二维的（行和列）。像 `my_array` 这样的简单数组的 `ndim` 将为 1。
- `shape`：一个整数元组，指示数组沿每个维度的尺寸。对于 `matrix_array`，`(2, 3)` 表示 2 行 3 列。对于 `my_array`，形状将是 `(5,)`，表示一个包含 5 个元素的单维度。`shape` 是最常用的属性之一。
- `size`：数组中的元素总数。这是 `shape` 元组中元素的乘积（`matrix_array` 的结果是 `2 * 3 = 6`）。
- `dtype`：一个对象，描述数组中元素的数据类型。NumPy 支持广泛的数值类型（例如，`int32`、`int64`、`float32`、`float64`、`complex128`），以及布尔型（`bool`）和对象类型。`float64` 表示 64 位浮点数。NumPy 通常从输入推断数据类型，但你也可以在创建时明确指定（在下一节中介绍）。
- `itemsize`：数组中每个元素的字节大小。一个 `float64` 元素占用 8 字节（64 位 / 8 位/字节）。
- `nbytes`：数组元素占用的总字节数。这简单来说就是 `size * itemsize`（`matrix_array` 为 `6 * 8 = 48` 字节）。请注意，这不包括数组对象本身的开销，但它提供了数据内存使用情况的良好衡量。

> 一维和二维 NumPy 数组及其基本属性的视觉比较。

### 为何使用 NumPy 数组？

内存效率、向量 (vector)化操作和便捷功能的结合，使 NumPy 数组成为 Python 数据科学生态系统中数值数据的标准。

1. **性能：** NumPy 数组上的操作比 Python 列表上的等效操作明显更快，特别是对于大型数据集。这种速度来源于使用优化的预编译 C 代码以及在连续内存块中处理数据。
2. **内存占用：** 对于相同数量的数值元素，NumPy 数组比 Python 列表占用更少的内存，因为它们没有与为每个元素存储完整 Python 对象相关的开销。
3. **便捷性：** NumPy 提供了一个全面的高级函数库，用于数学运算、线性代数、傅里叶变换、随机数生成等，所有这些都旨在与 `ndarray` 对象高效配合。你可以用简洁的代码执行复杂的计算。

Pandas（我们将在下一章介绍）等库将其核心数据结构（Series 和 DataFrame）直接构建在 NumPy 数组之上。Scikit-learn 等机器学习 (machine learning)库期望数据以 NumPy 数组或兼容结构的形式提供。因此，熟练掌握 NumPy 数组不仅有帮助；它对于机器学习中的中级 Python 编程来说非常重要。

在接下来的章节中，我们将学习如何以各种方式创建数组、访问和修改其元素，并使用 NumPy 广泛的功能执行强大的计算。

## 参考资料

- [NumPy user guide](https://numpy.org/doc/stable/user/basics.html) — NumPy community (2024)
  NumPy 数组基础知识的官方文档，包括创建、基本属性和数据类型。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本广泛引用的书籍，详细介绍了 NumPy 作为 Python 数据科学生态系统中数据操作底层结构的重要作用。
- [NumPy: A Fast Array Language for Scientific Computing](https://ieeexplore.ieee.org/document/4164160) — Travis E. Oliphant (2007)
  Journal: Computing in Science & Engineering; Publisher: IEEE Computer Society; Volume: 9; Pages: 10-20; DOI: [10.1109/MCSE.2007.58](https://doi.org/10.1109/MCSE.2007.58)
  NumPy 创建者的原始学术论文，阐述了 `ndarray` 用于科学计算的设计和效率优势。

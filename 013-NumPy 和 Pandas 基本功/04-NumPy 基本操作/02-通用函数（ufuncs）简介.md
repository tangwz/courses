# 通用函数（ufuncs）简介

来源：[原文](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-4-fundamental-numpy-operations/intro-universal-functions)

[返回章节目录](README.md) · [返回课程目录](../README.md)

基本的算术运算（如加法和乘法）可以在 NumPy 数组上逐元素执行。虽然这些操作很方便，但使用标准 Python 循环在大型数据集上执行它们会相当慢。NumPy 提供了一种高效的方式来执行快速的逐元素操作：通用函数（Universal Functions），通常简称为 ufuncs。

ufunc 本质上是一个以逐元素方式操作 `ndarray` 对象的函数。可以将它们看作是简单函数的向量 (vector)化封装器，能够一次性处理整个数组，从而省去了显式的 Python 循环。这种向量化 (quantization)是 NumPy 高效的一个重要原因。ufunc 是通过编译的 C 代码实现的，这让它们的执行速度远超对应的 Python 代码。

### 为什么要使用 Ufuncs？

- **速度：** 通过 ufuncs 执行操作比使用 Python 循环遍历数组快得多。
- **方便性：** 它们提供简洁的语法来对数组应用函数。
- **广播：** Ufuncs 本身支持广播，允许对形状不同但兼容的数组执行操作（关于广播，本章后续会详细介绍）。

### Ufuncs 的类型

Ufuncs 通常可以根据其接受的输入数组数量进行分类：

1. **一元 Ufuncs：** 操作单个输入数组。例子包括计算每个元素的平方根、指数、对数或三角函数值的函数。
2. **二元 Ufuncs：** 操作两个输入数组。例子包括逐元素加法、减法、乘法、除法、比较运算符和逻辑运算。

### 一元 Ufuncs 示例

我们来看一些常用的一元 ufuncs。考虑以下数组：

```python
import numpy as np

arr = np.arange(1, 6)
print(arr)
# 输出: [1 2 3 4 5]
```

现在，我们来应用一些一元 ufuncs：

- **平方根（`np.sqrt`）：** 计算每个元素的非负平方根。

  ```python
  sqrt_arr = np.sqrt(arr)
  print(sqrt_arr)
  # 输出: [1.         1.41421356 1.73205081 2.         2.23606798]
  ```
- **指数（`np.exp`）：** 计算每个元素 $x$ 的指数（$e^x$）。

  ```python
  exp_arr = np.exp(arr)
  print(exp_arr)
  # 输出: [  2.71828183   7.3890561   20.08553692  54.59815003 148.4131591 ]
  ```
- **自然对数（`np.log`）：** 计算每个元素的自然对数（底数为 $e$）。

  ```python
  log_arr = np.log(arr)
  print(log_arr)
  # 输出: [0.         0.69314718 1.09861229 1.38629436 1.60943791]
  ```
- **正弦（`np.sin`）：** 计算每个元素的正弦（假定元素以弧度为单位）。

  ```python
  sin_arr = np.sin(arr)
  print(sin_arr)
  # 输出: [ 0.84147098  0.90929743  0.14112001 -0.7568025  -0.95892427]
  ```

### 二元 Ufuncs 和算术运算符

您实际上已经隐式地使用了二元 ufuncs！标准的算术运算符（`+`、`-`、`*`、`/`、`**`）对应于特定的 ufuncs：

- `arr1 + arr2` 等同于 `np.add(arr1, arr2)`
- `arr1 - arr2` 等同于 `np.subtract(arr1, arr2)`
- `arr1 * arr2` 等同于 `np.multiply(arr1, arr2)`
- `arr1 / arr2` 等同于 `np.divide(arr1, arr2)`
- `arr1 ** arr2` 等同于 `np.power(arr1, arr2)`

我们来看一个 `np.add` 的例子：

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([10, 20, 30])

sum_arr_op = arr1 + arr2
sum_arr_ufunc = np.add(arr1, arr2)

print(f"使用 + 运算符: {sum_arr_op}")
# 输出: 使用 + 运算符: [11 22 33]
print(f"使用 np.add ufunc: {sum_arr_ufunc}")
# 输出: 使用 np.add ufunc: [11 22 33]
```

如您所见，结果是相同的。对于简单的算术运算，使用运算符通常更易读，但了解底层的 ufunc（在此例中是 `np.add`）会很有用。其他二元 ufuncs 包括 `np.maximum`、`np.minimum`、`np.mod`、`np.copysign`、比较函数（如 `np.greater`、`np.less_equal` 等）和逻辑函数（如 `np.logical_and`、`np.logical_or`）。

Ufuncs 是 NumPy 性能和易用性的重要组成部分。它们提供了一个全面的优化函数库，用于对数组进行逐元素操作，为在数据分析和科学计算中进行的许多数值计算提供了支持。理解它们的工作方式对于编写高效的 NumPy 代码来说非常重要。

## 参考资料

- [Universal functions (ufunc)](https://numpy.org/doc/stable/reference/ufuncs.html) — NumPy Developers (2024)
  官方文档，解释了NumPy通用函数（ufuncs）的概念、类型和用法，是权威参考资料。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data-analysis/9781098104030/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  涵盖NumPy核心操作的综合指南，包含对ufuncs和向量化计算的详细说明。

---

[上一节](01-%E5%9F%BA%E6%9C%AC%E7%AE%97%E6%9C%AF%E8%BF%90%E7%AE%97.md) · [下一节](03-%E6%95%B0%E5%AD%A6%E5%92%8C%E7%BB%9F%E8%AE%A1%E5%87%BD%E6%95%B0.md)

# 了解 NumPy N 维数组

来源：[原文](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-2-getting-started-numpy-arrays/understanding-ndarrays)

[返回章节目录](README.md) · [返回课程目录](../README.md)

NumPy 的核心是其主要数据结构：**N 维数组**，常称为 `ndarray`。可以将其视为专为数值数据设计的高效、灵活的容器。尽管标准的 Python 列表功能多用，但它们未针对数据科学和机器学习 (machine learning)中常见的大规模数值运算进行优化。NumPy 数组解决了这些限制。

`ndarray` 有何独特之处？

1. **同质数据类型：** 与 Python 列表不同，Python 列表可以存储不同类型的元素（例如，整数、字符串和浮点数都可以在一个列表中），而单个 NumPy 数组中的所有元素必须是*相同*的数据类型。这种同质性是 NumPy 性能的重要因素。了解每个元素（例如）是 64 位浮点数，使 NumPy 能够使用高度优化的底层 C 代码进行计算，从而避免了与 Python 动态类型相关的许多额外开销。
2. **固定大小：** 创建 NumPy 数组时，其大小通常是固定的。尽管 NumPy 提供了改变数组大小的函数，但这些操作通常涉及创建新数组和复制数据，而非原地调整原始数组大小。这种固定大小的特点有助于内存分配和访问的效率。
3. **多维性：** 顾名思义，`ndarray` 可以有多个维度。

   - **一维数组** 类似于列表或向量 (vector)。
   - **二维数组** 类似于带有行和列的表格或矩阵。
   - 数组可以有 3、4 甚至更多维度，允许表示复杂数据集，例如彩色图像（维度 1：高度，维度 2：宽度，维度 3：颜色通道）或矩阵序列。每个维度都称为一个**轴**。
4. **高效性：** 由于 `ndarray` 将数据存储在连续的内存块中并使用编译过的 C 代码执行操作，对其进行的数学运算明显快于使用循环在 Python 列表上执行的等效操作。这种能力，常称为**向量化 (quantization)**，允许您对数据执行批量操作而无需编写显式循环，从而产生简洁且更快的代码。

我们来直观地展示一维数组和二维数组之间的区别：

> 一个简单的一维数组（序列）和二维数组（网格）的可视化表示。

快速看一下 `ndarray` 对象在 Python 代码中是什么样子。我们很快将详细介绍创建方法。

```python
# 导入 NumPy 库，通常别名为 'np'
import numpy as np

# 创建一个简单的 Python 列表
python_list = [10, 20, 30, 40, 50]

# 将列表转换为 NumPy ndarray
numpy_array = np.array(python_list)

# 打印数组及其类型
print("NumPy 数组:", numpy_array)
print("类型:", type(numpy_array))
```

运行此代码将输出：

```
NumPy Array: [10 20 30 40 50]
Type: <class 'numpy.ndarray'>
```

注意输出 `[10 20 30 40 50]` 看起来与列表相似，但没有逗号。这是 NumPy 一维数组的标准字符串表示形式。其类型证实我们现在正在使用 NumPy 的专用 `ndarray` 对象。

了解 `ndarray` 是运用 NumPy 能力进行高效数据处理和计算的第一步。随后的章节将引导您创建这些数组并更仔细地检查它们的属性。

## 参考资料

- [The N-dimensional array (ndarray)](https://numpy.org/doc/stable/user/basics.html) — NumPy Developers (2024)
  解释`ndarray`数据结构、其基本属性以及作为NumPy核心对象在高效数值计算中作用的官方指南。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: 85-132
  详细介绍了NumPy `ndarray`，包括其结构、创建和向量化操作原理，是Python数据分析的奠基之作。
- [Numerical Python: Scientific Computing and Data Science Applications with NumPy, SciPy, and Matplotlib](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH7pwmf74rydEEfNTbUoIk28sVLFvWg743x_r7t_2fbjTnpbwPfv75IMKpn4TC6du_06SKTOgYiLaWiQUzZOF4nKPUu6WAO0kILlzAX8F-oZkR1g9sbzSOtc4sLC6YNzFWJ0OoV1yXGH5keZ3XlGlTcQgEj0a8ZpGDWkq49dbr7SiI_d7bkek-8Gw==) — Robert Johansson (2018)
  Publisher: Apress; Pages: 23-77; DOI: [10.1007/978-1-4842-4235-5](https://doi.org/10.1007/978-1-4842-4235-5)
  探讨了`ndarray`作为科学计算的核心数据结构，强调其处理多维数值数据的效率和能力。

---

[上一节](../01-Python%20%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86%E4%BB%8B%E7%BB%8D/06-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%BE%E7%BD%AE%E4%B8%8E%E9%AA%8C%E8%AF%81.md) · [下一节](02-%E4%BB%8E%20Python%20%E5%88%97%E8%A1%A8%E5%88%9B%E5%BB%BA%E6%95%B0%E7%BB%84.md)

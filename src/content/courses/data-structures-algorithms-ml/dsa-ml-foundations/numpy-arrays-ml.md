---
course: "data-structures-algorithms-ml"
chapter: "dsa-ml-foundations"
lesson: "numpy-arrays-ml"
sourceId: 5237
sourceUrl: "https://apxml.com/zh/courses/data-structures-algorithms-ml/chapter-1-dsa-ml-foundations/numpy-arrays-ml"
title: "NumPy 数组：机器学习数值处理的根本"
description: "了解 NumPy 数组、广播和向量化，助力机器学习高效数值计算。"
order: 4
plots: ["plots/5237-0.json"]
sourceHash: "4fc5072d688167bdfb4f2b689b60a9cb78784e7961f146d828febc7b0d18a5af"
sourceCorrections: []
---

虽然 Python 内置的列表和字典用途广泛，但在处理机器学习 (machine learning)中常见的大型数值数据集时，它们通常显得不足。对列表进行逐元素的数学运算，使用 Python 循环会带来较高的计算成本。正因如此，NumPy（数值 Python）变得不可或缺。它提供功能强劲的 N 维数组对象和先进的函数，并针对数值计算进行了优化，是许多 Python 科学计算和机器学习库的核心组成部分。

### NumPy `ndarray`：实现高速运算的核心

NumPy 的核心是 `ndarray`（N 维数组）。与 Python 列表不同，Python 列表可以容纳不同类型的对象，并将其引用分散存储在内存中，而 NumPy 数组具有以下特性，使其在数值处理任务中表现出高效率：

1. **同构数据类型：** 单个 NumPy 数组中的所有元素必须是相同的数据类型（例如，全部为 32 位整数或全部为 64 位浮点数）。这种一致性使得 NumPy 能够紧凑地存储数据。
2. **固定大小：** NumPy 数组的大小在创建时就已经确定。虽然您可以基于现有数组创建新数组，但原始数组的大小不会像 Python 列表那样动态改变。
3. **连续内存：** NumPy 数组将其元素存储在连续的内存块中。这使得处理器能够更快地访问和处理数据，借助低级别优化（如 CPU 向量 (vector)指令）。Python 列表存储指向可能分散对象的指针，不具备此优势。

考虑创建一个简单数组：

```python
import numpy as np

# 从 Python 列表创建数组
my_list = [1, 2, 3, 4, 5]
my_array = np.array(my_list)

print(my_array)
# 输出: [1 2 3 4 5]

print(my_array.dtype)
# 输出: int64 (或 int32，取决于系统)
```

### 向量 (vector)化：消除慢速循环

NumPy 最主要的性能优势之一源于**向量化 (quantization)**。这意味着操作一次性应用于整个数组，而不是通过显式的 Python 循环逐元素进行。NumPy 通过将循环下推到高度优化、预编译的 C 代码来实现这一点。

让我们比较两种大型序列的加法，一种使用 Python 列表，另一种使用 NumPy 数组：

```python
import numpy as np
import time

size = 1_000_000

# 使用 Python 列表
list1 = list(range(size))
list2 = list(range(size))

start_time = time.time()
result_list = [x + y for x, y in zip(list1, list2)]
list_time = time.time() - start_time

# 使用 NumPy 数组
arr1 = np.arange(size)
arr2 = np.arange(size)

start_time = time.time()
result_array = arr1 + arr2
numpy_time = time.time() - start_time

print(f"Python 列表加法时间: {list_time:.4f} 秒")
print(f"NumPy 数组加法时间:  {numpy_time:.4f} 秒")

# 示例输出（时间会有所不同）:
# Python 列表加法时间: 0.1234 秒
# NumPy 数组加法时间:  0.0056 秒
```

NumPy 运算通常快几个数量级。这是因为 NumPy 数组上的 `+` 运算会在内部触发一个单一、优化的 C 循环，而列表推导式则涉及每个元素的许多较慢的 Python 级别操作。



![性能对比：列表与 NumPy 加法](plots/5237-0.json)



> 两种大型序列的元素相加的相对时间比较，分别使用 Python 循环和 NumPy 的向量化加法。NumPy 明显更快。

### 广播：对不同形状进行智能操作

NumPy 还具有**广播**功能，这是一套用于对不同形状的数组应用二元运算（例如加法、乘法）的规则。在可能的情况下，NumPy 避免创建显式的数据副本，而是像将较小的数组“拉伸”或“广播”以匹配较大数组的形状那样执行操作。

一个常见的例子是将标量（单个数字）添加到数组中：

```python
import numpy as np

arr = np.array([1, 2, 3])
scalar = 10

result = arr + scalar # 广播标量
print(result)
# 输出: [11 12 13]
```

这里，NumPy 将标量 `10` 视为一个数组 `[10, 10, 10]`，但实际上并没有在内存中创建该数组，从而节省了时间和空间。

广播也适用于更高维度。您可以将一维数组添加到二维数组的每一行：

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

row_vector = np.array([10, 20, 30])

# 通过广播将 row_vector 添加到矩阵的每一行
result = matrix + row_vector
print(result)
# 输出:
# [[11 22 33]
#  [14 25 36]
#  [17 28 39]]
```

广播通过避免手动平铺或重复数据，简化了代码并提高了效率。

### 机器学习 (machine learning)中的重要 NumPy 操作

NumPy 提供了大量函数库。以下是机器学习流程中常用的一些基本操作：

- **数组创建：** 除了 `np.array()`，`np.zeros()`、`np.ones()`、`np.arange()`、`np.linspace()` 和 `np.random.rand()` 等函数也常用于初始化数组。

  ```python
  zeros_arr = np.zeros((2, 3)) # 创建一个 2x3 的零数组
  print(zeros_arr)
  # 输出:
  # [[0. 0. 0.]
  #  [0. 0. 0.]]

  range_arr = np.arange(0, 10, 2) # 类似 Python 的 range，但创建的是数组
  print(range_arr)
  # 输出: [0 2 4 6 8]
  ```
- **索引和切片：** 访问元素和子数组的方式类似于 Python 列表，但很自然地扩展到多个维度。

  ```python
  arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
  print(arr_2d[0, 1])  # 访问第 0 行第 1 列的元素
  # 输出: 2

  print(arr_2d[:, 1]) # 访问所有行，第 1 列
  # 输出: [2 5]

  print(arr_2d[0, :])  # 访问第 0 行，所有列
  # 输出: [1 2 3]
  ```
- **数学运算：** 逐元素操作（`+`、`-`、`*`、`/`、`**`）是标准功能。更高级的线性代数操作，如点积（`np.dot` 或用于矩阵乘法的 `@` 运算符），对于线性回归、逻辑回归和神经网络 (neural network)等算法来说非常重要。

  ```python
  A = np.array([[1, 2], [3, 4]])
  B = np.array([[5, 6], [7, 8]])

  # 逐元素乘积
  print(A * B)
  # 输出: [[ 5 12]
  #          [21 32]]

  # 矩阵乘法（点积）
  print(A @ B)
  # 输出: [[19 22]
  #          [43 50]]
  # 等同于: np.dot(A, B)
  ```
- **聚合函数：** `np.sum()`、`np.mean()`、`np.std()`、`np.min()`、`np.max()` 等函数对数组执行计算，通常使用 `axis` 参数 (parameter)来指定是按行、按列还是对整个数组进行聚合。

  ```python
  data = np.array([[1, 5], [2, 3], [6, 4]])

  print(np.sum(data)) # 所有元素的和
  # 输出: 21

  print(np.mean(data, axis=0)) # 每列的平均值
  # 输出: [3. 4.]

  print(np.max(data, axis=1)) # 每行的最大值
  # 输出: [5 3 6]
  ```
- **重塑数组：** 使用 `reshape()` 改变数组的形状而不改变其数据。这对于根据特定机器学习模型输入要求准备数据很有用。

  ```python
  flat_arr = np.arange(6) # [0 1 2 3 4 5]
  reshaped_arr = flat_arr.reshape((2, 3))
  print(reshaped_arr)
  # 输出:
  # [[0 1 2]
  #  [3 4 5]]
  ```

### NumPy 在更广泛的机器学习 (machine learning)环境中的作用

NumPy 的效率和以数组为中心的设计使其成为许多其他重要 Python 库的构建底层。

- **Pandas：** 内部使用 NumPy 数组来支持其 Series 和 DataFrame 对象，提供更高级别的数据操作和分析工具。
- **Scikit-learn：** Python 中标准的机器学习库，其输入数据、模型参数 (parameter)和内部计算都高度依赖 NumPy 数组。
- **深度学习 (deep learning)框架（TensorFlow、PyTorch）：** 尽管它们有自己的张量对象，但它们与 NumPy 数组良好衔接，通常可以方便地在格式间进行转换。数据预处理通常在馈入这些框架之前会涉及 NumPy。

简而言之，机器学习流程中的数据通常从原始输入开始，可能使用 Pandas 加载和清洗，然后转换为 NumPy 数组进行数值处理，并馈入使用 Scikit-learn 或深度学习库构建的模型。因此，理解 NumPy 对于理解数据如何在大多数基于 Python 的机器学习工作流中高效流动和操作是必要的。

## 参考资料

- [NumPy documentation](https://numpy.org/doc/stable/) — NumPy Developers (2023)
  NumPy所有功能的官方且最全面的资源，涵盖ndarray结构、性能和API细节。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  Pandas创建者撰写的权威指南，通过实际示例和对向量化、广播等核心概念的解释，阐明NumPy在Python数据堆栈中的作用。
- [SciPy Lecture Notes](https://scipy-lectures.org/) — Gael Varoquaux, Emmanuelle Gouillart, and Olivier Martin (2022)
  免费、高质量的教育资源，涵盖NumPy、SciPy、Matplotlib及其他科学Python工具，解释了核心概念和实际应用。

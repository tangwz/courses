---
course: "essential-numpy-pandas"
chapter: "numpy-array-indexing-slicing"
lesson: "slicing-1d-arrays"
sourceId: 1711
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-3-numpy-array-indexing-slicing/slicing-1d-arrays"
title: "一维数组切片"
description: "使用切片语法从一维 NumPy 数组中提取子序列。"
order: 2
plots: []
sourceHash: "c90e8cbf4d83d76583466440fb32f13932583a08cb18e05a9fdb4049e2b1d2d9"
sourceCorrections: []
---

访问单个元素可以提供精确控制。然而，通常需要处理一维数组中的一个序列或一段元素。切片操作可以让你根据索引范围取数组的一部分。如果你用过 Python 列表，会觉得 NumPy 的切片操作很熟悉，但它针对 NumPy 的数组结构做了优化。

### 一维数组切片的基础

对一维数组 `arr` 进行切片的语法是 `arr[start:stop:step]`。我们来分析一下这些组成部分：

- `start`：切片开始的索引（包含）。如果省略，默认是数组的开头（索引 0）。
- `stop`：切片结束的索引（不包含）。此索引处的元素*不*在切片结果中。如果省略，默认是数组的末尾。
- `step`：选择元素之间的间隔。如果省略，默认是 1（即选择连续的元素）。

我们来创建一个简单的数组进行操作：

```python
import numpy as np

# 创建一个从0到9的一维数组
arr = np.arange(10)
print(arr)
# 输出： [0 1 2 3 4 5 6 7 8 9]
```

现在，我们来看一些切片示例：

1. **选择从索引 2 开始直到（但不包括）索引 5 的元素：**

   ```python
   slice1 = arr[2:5]
   print(slice1)
   # 输出： [2 3 4]
   ```

   这选择了索引 2、3 和 4 处的元素。
2. **选择从开头直到（但不包括）索引 4 的元素：**

   ```python
   slice2 = arr[:4] # 省略 'start' 默认是 0
   print(slice2)
   # 输出： [0 1 2 3]
   ```

   这选择了从索引 0 开始直到（但不包括）索引 4 的元素。
3. **选择从索引 5 开始到末尾的元素：**

   ```python
   slice3 = arr[5:] # 省略 'stop' 默认到末尾
   print(slice3)
   # 输出： [5 6 7 8 9]
   ```

   这选择了从索引 5 开始直到最后一个元素。
4. **选择整个数组：**

   ```python
   slice4 = arr[:] # 省略 'start' 和 'stop'
   print(slice4)
   # 输出： [0 1 2 3 4 5 6 7 8 9]
   ```

### 使用步长值

`step` 值让你能选择不连续的元素：

1. **选择每隔一个元素：**

   ```python
   slice5 = arr[::2] # 选择从头到尾每隔一个元素
   print(slice5)
   # 输出： [0 2 4 6 8]
   ```
2. **在指定范围（索引 1 到 7）内选择每隔一个元素：**

   ```python
   slice6 = arr[1:7:2] # 从 1 开始，在 7 之前停止，步长为 2
   print(slice6)
   # 输出： [1 3 5]
   ```

### 切片中的负索引

就像访问单个元素一样，你可以在切片中使用负索引。`-1` 指代最后一个元素，`-2` 指代倒数第二个，以此类推。

1. **选择最后 3 个元素：**

   ```python
   slice7 = arr[-3:]
   print(slice7)
   # 输出： [7 8 9]
   ```
2. **选择从开头直到最后 2 个元素之前的元素：**

   ```python
   slice8 = arr[:-2]
   print(slice8)
   # 输出： [0 1 2 3 4 5 6 7]
   ```
3. **使用负步长反转数组：**

   ```python
   reversed_arr = arr[::-1]
   print(reversed_arr)
   # 输出： [9 8 7 6 5 4 3 2 1 0]
   ```

### 切片是视图，不是副本

这是 NumPy 切片的一个重要方面：数组的切片返回的是数组数据的*视图*而不是*副本*。这意味着切片只是观察*相同*底层数据的一种不同方式。修改切片中的元素会影响到原始数组。

我们来看一下实际效果：

```python
print("原始数组：", arr)
# 原始数组： [0 1 2 3 4 5 6 7 8 9]

# 创建一个切片
arr_slice = arr[5:8]
print("切片：", arr_slice)
# 切片： [5 6 7]

# 修改切片中的一个元素
arr_slice[1] = 999
print("修改后的切片：", arr_slice)
# 修改后的切片： [  5 999   7]

# 检查原始数组
print("切片修改后的原始数组：", arr)
# 切片修改后的原始数组： [  0   1   2   3   4   5 999   7   8   9]
```

可以看到，修改 `arr_slice` 中索引 1 处的元素（它对应于原始 `arr` 中索引 6 处的元素）也改变了 `arr` 中的值。这种行为的目的是为了提高性能和内存效率，因为它避免了不必要的数据重复，特别是对于数据分析和科学计算中常见的大型数组。

如果你明确需要一个切片的副本，以便修改不会影响原始数据，可以使用 `copy()` 方法：

```python
arr_copy = arr[5:8].copy()
arr_copy[1] = 111 # 修改副本

print("原始数组：", arr)      # 不受副本修改的影响
# 原始数组： [  0   1   2   3   4   5 999   7   8   9]
print("复制的切片：", arr_copy)
# 复制的切片： [  5 111   7]
```

理解切片以及视图与副本的区别，对于有效地处理数据和避免 NumPy 代码中意料之外的副作用非常重要。

## 参考资料

- [Indexing](https://numpy.org/doc/stable/user/basics.indexing.html) — The NumPy Developers (2024)
  涵盖NumPy数组索引和切片以及关键的视图与副本行为的官方文档。
- [Python for Data Analysis](https://learning.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  Python数据科学的经典教材，详细说明了NumPy数组操作，包括一维切片语法和视图与副本的区别。
- [The NumPy array object: Indexing and Slicing](https://scipy-lectures.org/intro/numpy/array_object.html#indexing-and-slicing) — SciPy Lecture Notes Contributors (2024)
  广泛使用的资源，提供NumPy数组索引和切片的实用、以代码为中心的示例和解释，强化了步长值和负索引等概念。

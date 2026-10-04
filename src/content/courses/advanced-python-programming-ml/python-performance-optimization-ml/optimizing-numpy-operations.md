---
course: "advanced-python-programming-ml"
chapter: "python-performance-optimization-ml"
lesson: "optimizing-numpy-operations"
sourceId: 2265
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-2-python-performance-optimization-ml/optimizing-numpy-operations"
title: "优化 NumPy 操作"
description: "应用向量化和高级 NumPy 技术，加快机器学习中的数值计算速度。"
order: 2
plots: []
sourceHash: "38c42bebc1e1c41ae993c4df5c691d9bf47890d1b39a5ff5c88a597cb47562d2"
sourceCorrections: []
---

NumPy 位于科学 Python 生态系统的基础，提供了 `ndarray` 对象和大量函数库用于数值计算。因为许多机器学习 (machine learning)库直接建立在 NumPy 之上，或与 NumPy 数组有大量交互，所以了解如何高效使用它，对于优化机器学习工作流而言十分必要。纯 Python 循环在数值任务中出了名的慢，与 NumPy 的操作相比更是如此，后者通常用 C 或 Fortran 实现，并一次性处理整个数组。这里提供了确保您有效使用 NumPy 性能的方法。

### 使用向量 (vector)化

高效 NumPy 使用中最重要的原则是**向量化 (quantization)**。这意味着用 NumPy 函数替换 Python 中针对数组元素的显式循环，这些函数在整个数组或数组的某些部分上操作。这些函数执行编译过的、优化的 C 或 Fortran 代码，绕过 Python 解释器对每个元素的开销。

考虑一个简单示例：添加两个大向量。

```python
import numpy as np
import time

# 创建大数组
size = 1_000_000
a = np.random.rand(size)
b = np.random.rand(size)
c_loop = np.zeros(size)
c_vec = np.zeros(size)

# 方法 1: Python 循环 (处理大数组时避免使用！)
start_time = time.time()
for i in range(size):
    c_loop[i] = a[i] + b[i]
loop_time = time.time() - start_time
print(f"Python 循环时间: {loop_time:.4f} seconds")

# 方法 2: NumPy 向量化 (推荐)
start_time = time.time()
c_vec = a + b # 等同于 np.add(a, b)
vectorized_time = time.time() - start_time
print(f"向量化时间: {vectorized_time:.4f} seconds")

# 验证结果接近
assert np.allclose(c_loop, c_vec)

print(f"加速比: {loop_time / vectorized_time:.1f}x")
```

> 示例输出可能显示：
> Python 循环时间: 0.1850 seconds
> 向量化时间: 0.0015 seconds
> 加速比: 123.3x

差异通常是数量级的。向量化版本 `a + b` 将整个操作委派给 NumPy 的优化后端，比在 Python 中逐个元素迭代执行快得多。始终寻找将处理数组元素的 Python 循环替换为等效的向量化 NumPy 表达式的机会。

### 运用通用函数 (ufuncs)

NumPy 提供了通用函数，或称 ufuncs，它们在 ndarray 上逐元素操作。这些包含数学操作（`np.add`、`np.subtract`、`np.exp`、`np.log`、`np.sqrt`、`np.sin` 等）、比较函数（`np.greater`、`np.equal`）等。使用 ufuncs 是向量 (vector)化的一部分。

与其写成：

```python
import math
# 低效循环
result = np.zeros(a.shape)
for i in range(len(a)):
    result[i] = math.exp(a[i])
```

使用对应的 ufunc：

```python
# 高效向量化
result = np.exp(a)
```

这能高效地将指数函数应用于 `a` 的每个元素。查阅 NumPy 文档以找到适合您计算需求的 ufuncs。

### 借助广播机制

广播描述了 NumPy 在算术运算期间如何处理不同形状的数组。在特定约束下，较小的数组被“广播”到较大的数组上，使它们具有兼容的形状。这是一个强大的功能，它避免了显式复制数据的需要，节省了内存和计算时间。

广播规则如下：

1. 如果数组维度数量不同，在较小数组形状的前面添加 1，直到它们维度数量一致。
2. 如果两个数组在任何维度上的形状不匹配，则该维度形状为 1 的数组将被拉伸以匹配另一个形状。
3. 如果任何维度的大小不一致，且两者都不为 1，则会引发错误。

考虑将一维数组（向量 (vector)）添加到二维数组（矩阵）的每一行：

```python
# 示例矩阵（例如，数据集样本）
X = np.random.rand(5, 3) # 5 个样本，3 个特征

# 示例向量（例如，特征均值）
mean_vec = np.mean(X, axis=0) # 形状 (3,)

print("原始 X 形状:", X.shape)
print("均值向量形状:", mean_vec.shape)

# 广播机制：从 X 的每一行减去 mean_vec
# X 的形状是 (5, 3)
# mean_vec 的形状是 (3,)。NumPy 将其视为 (1, 3)。
# 规则 1: 形状变为 (5, 3) 和 (1, 3)。
# 规则 2: 大小为 1 的维度 (mean_vec 的第一个维度) 被拉伸以匹配 5。
# 结果操作如同 mean_vec 被复制了 5 次。
X_centered = X - mean_vec

print("中心化 X 形状:", X_centered.shape)
# 验证某一行是否已中心化
print("原始第一行:", X[0])
print("中心化第一行:", X_centered[0])
print("中心化数据的均值（应接近零）:", np.mean(X_centered, axis=0))
```

没有广播，您可能需要使用 `np.tile(mean_vec, (5, 1))` 手动平铺 `mean_vec`，从而创建一个更大的中间数组。广播更高效地达到相同结果。理解广播机制对于编写简洁且高性能的 NumPy 代码具有重要意义，尤其在特征缩放或距离计算中。

### 优化索引和切片

访问 NumPy 数组中的元素通常很快，但方法很重要：

- **基本切片：** 使用 `arr[1:5, :2]` 这样的切片尽可能创建原始数组的*视图*。视图是内存高效的，因为它们不复制数据；它们只引用原始数组内存的一部分。对视图的操作有时会修改原始数据。
- **花式索引：** 使用整数数组或布尔数组进行索引（例如，`arr[[0, 2, 4], :]` 或 `arr[arr > 0.5]`）总是创建数据的*副本*。这提供了灵活性，但可能更慢并消耗更多内存，尤其对于大数组。
- **布尔索引：** 根据条件选择元素（`arr[condition]`）是花式索引的常见形式。尽管它会创建副本，但这通常是根据向量 (vector)化条件过滤数据最易读且高效的方式。

```python
arr = np.arange(12).reshape(4, 3)

# 基本切片（视图）
view_slice = arr[1:3, 0:2]
print("基本切片（视图）：\n", view_slice)
view_slice[0, 0] = 99 # 修改视图会改变原始数组
print("修改视图后的原始数组：\n", arr)

# 重置数组
arr = np.arange(12).reshape(4, 3)

# 花式索引（副本）
fancy_indexed = arr[[0, 3], :] # 选择第 0 和第 3 行
print("花式索引（副本）：\n", fancy_indexed)
fancy_indexed[0, 0] = -1 # 修改副本不会改变原始数据
print("修改副本后的原始数组：\n", arr)

# 布尔索引（副本）
bool_indexed = arr[arr % 2 == 0] # 选择偶数
print("布尔索引（副本）：\n", bool_indexed)
```

请注意您获得的是视图还是副本。如果您需要修改数组的子集而不影响原始数组，请使用 `.copy()` 显式创建副本。如果内存效率重要，请尝试使用基本切片来组织操作，或者留意花式索引的开销。

### 在适当时候使用原地操作

NumPy 允许直接修改数组的操作，而不创建新的数组对象。这通过使用 `+=`、`-=`、`*=` 等运算符，或将现有数组提供给许多 ufuncs 中可用的 `out` 参数 (parameter)来完成。

```python
a = np.ones(5)
b = np.arange(5)

# 为结果创建一个新数组
c = a + b

# 直接修改 'a'（原地操作）
a += b # 等同于 np.add(a, b, out=a)

print("原地加法后的原始 'a':", a)
```

使用原地操作可以通过避免为中间结果分配临时数组来节省内存，这在处理大型数据集的循环或函数中很有用。然而，请谨慎使用：如果多个变量引用同一个数组（例如，通过视图），原地修改一个会影响其他变量。如果管理不慎，这可能导致难以察觉的错误。

### 考虑内存布局

NumPy 数组将数据存储在连续的内存块中。元素在这个块中的排列方式由数组的*顺序*决定。默认是 'C' 顺序（行优先），这意味着同一行中的元素彼此相邻存储。Fortran 顺序（'F'，列优先）则连续存储同一列中的元素。

```python
c_order_arr = np.arange(6).reshape(2, 3) # 默认 C 顺序
f_order_arr = np.array(c_order_arr, order='F')

print("C 顺序步长:", c_order_arr.strides) # 例如，64 位整数为 (24, 8)
print("F 顺序步长:", f_order_arr.strides) # 例如，64 位整数为 (8, 16)
```

按照内存布局顺序访问元素（例如，在 C 顺序数组中遍历行）由于更好的缓存局部性，通常更快。大多数 NumPy 操作都针对 C 顺序数组进行了优化，这与样本构成行的典型机器学习 (machine learning)数据非常吻合。虽然您通常不需要显式管理顺序，但请注意，涉及转置或切片的操作有时可能导致非连续数组。如果性能分析表明此类数组上的内存访问模式存在瓶颈，创建连续副本可能会有帮助：

```python
non_contiguous_view = c_order_arr[:, ::2] # 创建一个非连续视图
print("视图是连续的吗？", non_contiguous_view.flags['C_CONTIGUOUS'])

# 如果性能关键部分需要：
contiguous_copy = np.ascontiguousarray(non_contiguous_view)
# 或者简单地：contiguous_copy = non_contiguous_view.copy()
print("副本是连续的吗？", contiguous_copy.flags['C_CONTIGUOUS'])
```

### 避免不必要的副本

如索引部分所述，副本会消耗内存并需要时间来创建。在索引中，请注意某些 NumPy 函数可能会返回副本，而您可能期望视图，反之亦然。像 `np.flatten` 这样的函数总是返回副本，而 `np.ravel` 如果可能则返回视图。`reshape` 通常返回视图，但如果重塑与原始内存布局不兼容，则可能返回副本。有疑问时，查阅文档或使用 `np.shares_memory()` 来查看两个数组是否共享相同的底层数据。

### 借助专用函数

NumPy 包含了用于线性代数、傅里叶变换和随机数生成的常用操作的高度优化函数（例如 `numpy.linalg`、`numpy.fft`、`numpy.random`）。在适用时使用这些专用函数，因为它们通常比手动实现表现好很多。

一个特别强大的函数是 `numpy.einsum`（爱因斯坦求和约定）。它提供了表达复杂操作的简洁语法，涉及多维数组（张量）的乘法、求和和转置。虽然其语法最初可能看起来不熟悉，但 `einsum` 通常可以比 `dot`、`transpose` 和 `sum` 的组合更清晰、更高效地表达复杂操作。例如，计算矩阵的迹（`np.trace(A)`）可以写成 `np.einsum('ii->', A)`。批量矩阵乘法（`np.matmul(A, B)`）可以写成 `np.einsum('bij,bjk->bik', A, B)`。学习 `einsum` 对于优化深度学习 (deep learning)中常见的复杂张量代数很有价值。

### 综合应用：优化示例

让我们优化一个函数来计算两组点之间的成对欧几里得距离，这是在聚类或最近邻算法中常见的任务。

```python
def pairwise_distance_loop(X, Y):
    """使用 Python 循环进行低效计算。"""
    num_X = X.shape[0]
    num_Y = Y.shape[0]
    distances = np.zeros((num_X, num_Y))
    for i in range(num_X):
        for j in range(num_Y):
            # 计算平方欧几里得距离
            dist_sq = np.sum((X[i, :] - Y[j, :])**2)
            distances[i, j] = np.sqrt(dist_sq)
    return distances

def pairwise_distance_vectorized(X, Y):
    """使用广播和向量化进行高效计算。"""
    # 公式：dist(x, y)^2 = sum((x - y)^2) = sum(x^2 - 2xy + y^2)
    # 使用广播：(X[:, None, :] - Y[None, :, :]) 得到一个 (num_X, num_Y, num_dims) 数组
    # 然后对最后一个维度求和并取平方根。
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :] # 形状: (num_X, num_Y, 维度数量)
    distances_sq = np.sum(diff**2, axis=-1) # 形状: (num_X, num_Y)
    return np.sqrt(distances_sq)

# 使用 scipy 的另一种向量化方法（通常高度优化）
from scipy.spatial.distance import cdist
def pairwise_distance_scipy(X, Y):
    """使用专门的 SciPy 函数。"""
    return cdist(X, Y, metric='euclidean')

# 生成示例数据
num_samples_x = 100
num_samples_y = 150
num_features = 50
X_data = np.random.rand(num_samples_x, num_features)
Y_data = np.random.rand(num_samples_y, num_features)

# 时间比较（会因机器和数据大小而异）
start = time.time()
d_loop = pairwise_distance_loop(X_data, Y_data)
t_loop = time.time() - start

start = time.time()
d_vec = pairwise_distance_vectorized(X_data, Y_data)
t_vec = time.time() - start

start = time.time()
d_scipy = pairwise_distance_scipy(X_data, Y_data)
t_scipy = time.time() - start

print(f"循环时间: {t_loop:.4f} seconds")
print(f"向量化时间: {t_vec:.4f} seconds")
print(f"Scipy 时间: {t_scipy:.4f} seconds")
print(f"向量化相对循环的加速比: {t_loop / t_vec:.1f}x")
print(f"Scipy 相对循环的加速比: {t_loop / t_scipy:.1f}x")

# 验证结果接近
assert np.allclose(d_loop, d_vec)
assert np.allclose(d_loop, d_scipy)
```

> 示例输出可能显示：
> 循环时间: 0.4512 seconds
> 向量 (vector)化时间: 0.0150 seconds
> Scipy 时间: 0.0018 seconds
> 向量化 (quantization)相对循环的加速比: 30.1x
> Scipy 相对循环的加速比: 250.7x

此示例表明，用向量化的 NumPy 操作（使用广播和 ufuncs）替换显式 Python 循环会带来显著的性能提升。通常，使用 SciPy 等库中的函数（这些库建立在 NumPy 之上并包含高度优化的算法）可以提供更大的加速效果。

总而言之，优化 NumPy 的使用主要涉及以数组操作而非逐元素循环的方式思考。向量化、广播、高效索引以及借助专用函数是编写快速且内存高效的数值代码的重要技术，是优化基于 Python 的机器学习 (machine learning)应用程序的重要一步。

## 参考资料

- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data-analysis/9781098104030/) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: 579
  Python数据科学的开创性教材，详细解释了NumPy数组、向量化操作、广播和性能考量。第3版。
- [High Performance Python: Practical Performant Programming for Humans](https://books.google.com/books?id=i9gCEAAAQBAJ) — Micha Gorelick, Ian Ozsvald (2020)
  Publisher: O'Reilly Media; Pages: 450
  提供了优化Python代码的实用策略，其中专门的章节讨论了NumPy及其实现高性能的底层机制。第2版。
- [numpy.einsum documentation](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) — NumPy Developers (2024)
  官方文档详细介绍了`einsum`函数，包括其语法和复杂张量操作的示例。

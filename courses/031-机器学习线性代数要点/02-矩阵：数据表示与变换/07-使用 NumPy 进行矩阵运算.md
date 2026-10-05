# 使用 NumPy 进行矩阵运算

来源：[原文](https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-2-matrices-data-representation-transformations/implementing-matrix-operations-numpy)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然理解矩阵加法、乘法和转置的原理很重要，但对于机器学习 (machine learning)中常见的大型数据集来说，手动执行这些运算变得不切实际。这时，NumPy 这样的数值计算库就变得不可或缺。NumPy 提供了高度优化的函数，用于创建和处理多维数组，这些数组可以很好地作为矩阵使用。

让我们看看如何将我们讨论过的矩阵运算转换为使用 NumPy 的高效 Python 代码。

### 创建 NumPy 矩阵

首先，请确保你已安装 NumPy (`pip install numpy`) 并导入它，通常使用别名 `np`：

```python
import numpy as np
```

你可以从列表的列表中创建矩阵（表示为二维 NumPy 数组）：

```python
# 一个 2x3 矩阵
A = np.array([[1, 2, 3],
              [4, 5, 6]])

print("矩阵 A:\n", A)
print("A 的形状:", A.shape) # 输出: (2, 3) -> 2 行, 3 列
```

NumPy 也提供了创建特殊矩阵的函数：

```python
# 一个 3x3 的零矩阵
zeros_matrix = np.zeros((3, 3))
print("\n零矩阵:\n", zeros_matrix)

# 一个 2x4 的全一矩阵
ones_matrix = np.ones((2, 4))
print("\n全一矩阵:\n", ones_matrix)

# 一个 3x3 的单位矩阵
identity_matrix = np.eye(3)
print("\n单位矩阵:\n", identity_matrix)
```

### 基本逐元素运算

加法、减法和标量乘法等运算都是逐元素进行的，就像向量 (vector)一样，前提是矩阵的形状兼容。

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
scalar = 10

# 加法
C = A + B
# 另一种方式: C = np.add(A, B)
print("A + B:\n", C)

# 减法
D = A - B
# 另一种方式: D = np.subtract(A, B)
print("\nA - B:\n", D)

# 标量乘法
E = A * scalar
# 另一种方式: E = np.multiply(A, scalar)
print("\nA *", scalar, ":\n", E)

# 逐元素乘法（哈达玛积）
# 注意: 这不是标准的矩阵乘法
F = A * B
# 另一种方式: F = np.multiply(A, B)
print("\n逐元素 A * B:\n", F)
```

对形状不兼容的矩阵进行逐元素运算会导致 `ValueError`。

### 矩阵转置

矩阵转置是将其沿着主对角线翻转。在 NumPy 中，你可以使用 `.T` 属性或 `np.transpose()` 函数来实现这一点。

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

A_transpose = A.T
# 另一种方式: A_transpose = np.transpose(A)

print("矩阵 A:\n", A)
print("A 的形状:", A.shape) # 输出: (2, 3)

print("\nA 的转置:\n", A_transpose)
print("A_transpose 的形状:", A_transpose.shape) # 输出: (3, 2)
```

### 矩阵乘法（点积）

这可以说是机器学习 (machine learning)中最重要的一种矩阵运算，用于变换、方程求解等等。它*不是*逐元素乘法。NumPy 使用 `@` 运算符或 `np.matmul()` 函数（有时也用 `np.dot()`，对于二维数组其行为类似）。

请记住这个规则：要将矩阵 $A$ ($m \times n$) 乘以矩阵 $B$ ($n \times p$)，矩阵 $A$ 的列数必须等于矩阵 $B$ 的行数。结果矩阵 $C$ 的维度将是 $m \times p$。

```python
A = np.array([[1, 2],    # 2x2
              [3, 4]])
B = np.array([[5, 6, 7], # 2x3
              [8, 9, 0]])

# 使用 @ 进行矩阵乘法
C = A @ B
# 另一种方式: C = np.matmul(A, B)
# 另一种方式（对于 2D 数组）: C = np.dot(A, B)

print("矩阵 A (2x2):\n", A)
print("\n矩阵 B (2x3):\n", B)
print("\n矩阵乘法 A @ B (2x3):\n", C)
print("C 的形状:", C.shape) # 输出: (2, 3)

# 示例: 将变换矩阵应用于数据点
# 'data_points' 中的每一列是一个点 (x, y)
data_points = np.array([[1, 0, -1],  # x 坐标
                        [1, 2,  1]]) # y 坐标 (形状: 2x3)

# 一个旋转矩阵（例如，逆时针 90 度）
rotation_matrix = np.array([[0, -1], # 形状: 2x2
                           [1,  0]])

transformed_points = rotation_matrix @ data_points
print("\n原始点（列）:\n", data_points)
print("\n旋转矩阵:\n", rotation_matrix)
print("\n变换后的点（列）:\n", transformed_points)
```

> 该图显示了矩阵乘法 $C = A @ B$，并展示了涉及的维度。

尝试乘以不兼容的矩阵，例如上面例子中的 `B @ A`（一个 $2 \times 3$ 矩阵不能乘以一个 $2 \times 2$ 矩阵），将导致 `ValueError`。

### 效率考量

NumPy 运算是在高度优化的 C 代码中实现的，并且通常使用底层的基础线性代数子程序 (BLAS) 库。这使得计算比纯 Python 中使用循环编写的等效运算快得多。在机器学习 (machine learning)中使用大型数据集时，使用 NumPy 进行矩阵运算不仅方便，而且对性能来说是不可或缺的。

在接下来的章节中，我们将继续在这些 NumPy 根基上学习线性系统求解、向量 (vector)空间和分解。熟练掌握这些基本的 NumPy 运算是你实现更复杂机器学习算法的第一步。

## 参考资料

- [The N-dimensional array (ndarray)](https://numpy.org/doc/stable/reference/arrays.ndarray.html) — NumPy developers (2023)
  详细介绍了 NumPy 核心数组对象及其基本操作的官方文档，包括创建、元素级数学运算和转置。
- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  一本理解线性代数理论基础的经典教材，涵盖矩阵运算及其性质。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  一本关于 Python 科学计算的实用指南，其中 NumPy 章涵盖数组创建、操作和基本运算。
- [Deep Learning](https://www.deeplearningbook.org/contents/linear_algebra.html) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 25-56
  对线性代数概念（包括矩阵运算）进行了全面的数学介绍，这些概念对于理解深度学习算法至关重要。
- [18.06 Linear Algebra, Spring 2010](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) — Gilbert Strang (2010)
  Publisher: MIT OpenCourseWare
  一门全面的在线线性代数课程，提供讲座和学习材料，帮助读者掌握矩阵运算的基础知识。

---

[上一节](06-%E8%A1%A8%E7%A4%BA%E7%BA%BF%E6%80%A7%E6%96%B9%E7%A8%8B%E7%BB%84.md) · [下一节](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%95%B0%E6%8D%AE%E7%82%B9%E5%8F%98%E6%8D%A2.md)

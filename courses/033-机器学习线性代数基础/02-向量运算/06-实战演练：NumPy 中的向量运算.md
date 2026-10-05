# 实战演练：NumPy 中的向量运算

来源：[原文](https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-2-vector-operations/numpy-vector-operations-practice)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将向量 (vector)的基本运算转化为代码是实际应用中常见的需求。为了高效地完成这些运算，通常会使用 NumPy，它是 Python 中进行数值计算的标准库。

首先，我们来设置环境，导入 NumPy 库。我们通常遵循的惯例是将其导入并使用别名 `np`。

```python
import numpy as np
```

### 在 NumPy 中创建向量 (vector)

在 NumPy 中，向量表示为一维数组。你可以使用 `np.array()` 函数创建向量，传入一个 Python 列表即可。

```python
# 创建两个向量 v 和 w
v = np.array([2, 1])
w = np.array([1, 3])

print(f"向量 v: {v}")
print(f"向量 w: {w}")
```

**输出：**

```
向量 v: [2 1]
向量 w: [1 3]
```

### 向量 (vector)加法和减法

在 NumPy 中执行加法和减法很简单。该库重载了标准的 `+` 和 `-` 运算符，对相同形状的数组执行逐元素运算。

```python
# 向量加法
v_plus_w = v + w
print(f"v + w = {v_plus_w}")

# 向量减法
v_minus_w = v - w
print(f"v - w = {v_minus_w}")
```

**输出：**

```
v + w = [3 4]
v - w = [ 1 -2]
```

这与我们讨论的代数规则相符。将 `v` 和 `w` 相加会得到一个新向量 $[2+1, 1+3] = [3, 4]$。从几何角度看，这个运算可以想象成将一个向量的尾部放在另一个向量的头部。所得向量连接起点和终点，形成一个三角形。

> 向量加法的几何解释。结果向量 `v+w` 是由 `v` 和 `w` 形成的平行四边形的对角线。

### 标量乘法

要将向量 (vector)乘以标量，我们使用 `*` 运算符。这个运算会按比例改变向量的长度。乘以正标量会改变其大小，而负标量除了改变大小外，还会反转其方向。

```python
# 定义一个向量和一个标量
u = np.array([3, 2])
s = 2.5

# 执行标量乘法
scaled_u = s * u
print(f"向量 u: {u}")
print(f"标量 s: {s}")
print(f"s * u = {scaled_u}")
```

**输出：**

```
向量 u: [3 2]
标量 s: 2.5
s * u = [7.5 5. ]
```

可以看到，向量 `u` 的每个元素都乘以了标量 `s`。所得向量 `[7.5, 5]` 方向与 `u` 相同，但长度是 `u` 的 2.5 倍。

### 点积

点积是线性代数中的一个核心运算，特别是在机器学习 (machine learning)中。它给出一个单一的数值，与两个向量 (vector)之间的夹角有关。NumPy 提供了几种方法来计算它。

`np.dot()` 函数是常用的方法：

```python
v = np.array([2, 1])
w = np.array([1, 3])

# 使用 np.dot() 计算点积
dot_product = np.dot(v, w)
print(f"v 和 w 的点积是: {dot_product}")
```

**输出：**

```
v 和 w 的点积是: 5
```

计算方法是 $(2 \times 1) + (1 \times 3) = 2 + 3 = 5$。

另外，自 Python 3.5 起，你可以使用 `@` 运算符，它专门用于矩阵乘法，也适用于向量的点积。这个运算符通常使代码更易读。

```python
# 使用 @ 运算符计算点积
dot_product_operator = v @ w
print(f"使用 @ 运算符的点积是: {dot_product_operator}")
```

**输出：**

```
使用 @ 运算符的点积是: 5
```

### 向量 (vector)范数：测量长度

向量的范数是其长度或大小的量度。`np.linalg.norm()` 函数是完成此任务的工具。默认情况下，它计算 $L_2$ 范数（欧几里得范数）。

```python
u = np.array([3, 4])

# 计算 L2 范数（默认）
l2_norm = np.linalg.norm(u)
print(f"u 的 L2 范数是: {l2_norm}")
```

**输出：**

```
u 的 L2 范数是: 5.0
```

计算方法是 $\sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$。

要计算其他范数，例如 $L_1$ 范数（曼哈顿距离），你可以指定 `ord` 参数 (parameter)。

```python
# 计算 L1 范数
l1_norm = np.linalg.norm(u, ord=1)
print(f"u 的 L1 范数是: {l1_norm}")
```

**输出：**

```
u 的 L1 范数是: 7.0
```

这是元素绝对值的和：$|3| + |4| = 7$。

### 检查正交向量 (vector)

如果两个向量的点积为零，则它们是正交的。这是一个简单有效的方法。我们来定义两个应该是正交的向量并进行检查。

```python
# 定义两个正交向量
vec1 = np.array([1, 0])
vec2 = np.array([0, 1])

# 计算它们的点积
dot_prod = vec1 @ vec2
print(f"vec1 和 vec2 的点积是: {dot_prod}")
```

**输出：**

```
vec1 和 vec2 的点积是: 0
```

点积为 0 确认它们是正交的。现在，我们尝试使用两个非正交向量。

```python
# 定义两个非正交向量
vec3 = np.array([1, 1])
vec4 = np.array([1, -0.5])

# 计算它们的点积
dot_prod_non_ortho = vec3 @ vec4
print(f"vec3 和 vec4 的点积是: {dot_prod_non_ortho}")
```

**输出：**

```
vec3 和 vec4 的点积是: 0.5
```

由于结果不为零，向量 `vec3` 和 `vec4` 不是正交的。

### 应用：找出两个向量 (vector)之间的夹角

我们可以组合这些运算来解决有趣的问题。例如，我们可以找出两个向量之间的夹角 $\theta$，使用从点积定义中导出的公式：


$$
\cos(\theta) = \frac{\mathbf{v} \cdot \mathbf{w}}{\|\mathbf{v}\| \|\mathbf{w}\|}
$$


这意味着：


$$
\theta = \arccos\left(\frac{\mathbf{v} \cdot \mathbf{w}}{\|\mathbf{v}\| \|\mathbf{w}\|}\right)
$$


让我们在 NumPy 中实现这一点。

```python
v = np.array([2, 1])
w = np.array([1, 3])

# 计算点积
dot_vw = v @ w

# 计算范数
norm_v = np.linalg.norm(v)
norm_w = np.linalg.norm(w)

# 计算夹角的余弦值
cos_theta = dot_vw / (norm_v * norm_w)

# 计算弧度制下的夹角
angle_rad = np.arccos(cos_theta)

# 转换为度数以便于理解
angle_deg = np.degrees(angle_rad)

print(f"点积: {dot_vw:.2f}")
print(f"v 的范数: {norm_v:.2f}")
print(f"w 的范数: {norm_w:.2f}")
print(f"v 和 w 之间的夹角（弧度）: {angle_rad:.2f}")
print(f"v 和 w 之间的夹角（度数）: {angle_deg:.2f}")
```

**输出：**

```
点积: 5.00
v 的范数: 2.24
w 的范数: 3.16
v 和 w 之间的夹角（弧度）: 0.64
v 和 w 之间的夹角（度数）: 36.87
```

这个实战例子表明点积和范数如何一起作用，以提供有意义的几何信息。掌握这些 NumPy 运算是实现更高级机器学习 (machine learning)算法的第一步。

## 参考资料

- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html) — NumPy Developers (2024)
  这份官方指南详细介绍了NumPy数组，涵盖了创建、基本算术运算（加法、减法、标量乘法）、点积（`np.dot`和`@`运算符）以及`np.linalg.norm`等线性代数例程。
- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press; Pages: Chapters 1-2
  这本备受推崇的教材为向量运算、线性组合、点积和范数奠定了扎实的理论基础，对理解代码示例背后的数学原理至关重要。
- [Introduction to Applied Linear Algebra - Vectors, Matrices, and Least Squares](https://web.stanford.edu/~boyd/vmls/vmls.pdf) — Stephen Boyd and Lieven Vandenberghe (2018)
  Publisher: Cambridge University Press; Pages: Chapters 1-3
  本书介绍了线性代数在工程和数据科学（包括机器学习）中的应用。它提供了向量运算、范数和点积的应用视角。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: Chapters 2-4
  这本关于Python数据操作和分析的权威指南对NumPy进行了广泛的介绍。它提供了大量的代码示例，用于数组创建和各种向量操作，展示了如何高效地实现它们。

---

[上一节](05-%E6%AD%A3%E4%BA%A4%E5%90%91%E9%87%8F.md) · [下一节](../03-%E4%BD%BF%E7%94%A8%E7%9F%A9%E9%98%B5/01-%E7%9F%A9%E9%98%B5%E5%8A%A0%E6%B3%95%E5%92%8C%E5%87%8F%E6%B3%95.md)

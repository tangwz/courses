---
course: "julia-deep-learning"
chapter: "foundations-julia-ml"
lesson: "julia-numerical-computation-linear-algebra"
sourceId: 6868
sourceUrl: "https://apxml.com/zh/courses/julia-deep-learning/chapter-1-foundations-julia-ml/julia-numerical-computation-linear-algebra"
title: "Julia 数值计算：数组与线性代数"
description: "掌握 Julia 中的数组操作和线性代数运算，这些对深度学习算法是根本的。"
order: 4
plots: []
sourceHash: "6cfb7cb436a65e4396e644f21e396f7edc3352d69a5ba15e6b7d21b62a9b1923"
sourceCorrections: []
---

深度学习 (deep learning)算法的核心是数值计算，主要涉及数组（或张量）和线性代数运算。Julia 在这方面表现出色，提供高级且富有表达力的语法，同时性能可与低级语言媲美。对 Julia 数组操作和线性代数能力的掌握，对于构建和训练神经网络 (neural network)是必不可少的。

## 数组：数值数据的基本组成部分

在 Julia 中，数组是多维容器，可以保存同类型数据的集合。对于深度学习 (deep learning)而言，这些数组通常存储数字，例如输入特征、模型权重 (weight)或梯度。

### 创建数组

Julia 提供了几种创建数组的方法。你可能对基本的数组字面量很熟悉：

```julia
# Float64 类型的向量（一维数组）
vector_a = [1.0, 2.5, 3.0]

# Int64 类型的矩阵（二维数组）
matrix_b = [1 2 3; 4 5 6]

# 三维数组
tensor_c = rand(2, 3, 4) # 创建一个 2x3x4 的数组，其中包含 0 到 1 之间的随机 Float64 值
```

常用的数组初始化函数有：

- `zeros(T, dims...)` 或 `zeros(dims...)`：创建一个填充零的数组。`T` 指定元素类型（默认为 `Float64`）。

  ```julia
  zeros_matrix = zeros(Int8, 2, 3) # 一个 2x3 的 8 位整数矩阵，全部为零
  # 输出:
  # 2×3 Matrix{Int8}:
  #  0  0  0
  #  0  0  0
  ```
- `ones(T, dims...)` 或 `ones(dims...)`：创建一个填充一的数组。

  ```julia
  ones_vector = ones(3) # 一个 3 元素的 Float64 向量，全部为一
  # 输出:
  # 3-element Vector{Float64}:
  #  1.0
  #  1.0
  #  1.0
  ```
- `fill(value, dims...)`：创建一个填充特定 `value` 的数组。

  ```julia
  filled_array = fill(7.7, (2, 2)) # 一个 2x2 矩阵，所有元素都是 7.7
  # 输出:
  # 2×2 Matrix{Float64}:
  #  7.7  7.7
  #  7.7  7.7
  ```
- `rand(T, dims...)` 或 `rand(dims...)`：创建一个包含随机数（默认为 0 到 1 之间均匀分布）的数组。
- `randn(T, dims...)` 或 `randn(dims...)`：创建一个包含来自标准正态分布的随机数的数组。

### 数组属性

你可以使用以下函数检查数组特征：

- `size(A)`：返回一个包含数组 `A` 维度的元组。
- `length(A)`：返回 `A` 中的元素总数。
- `ndims(A)`：返回 `A` 的维度数量。

```julia
my_matrix = rand(4, 5)
println("大小: ", size(my_matrix))          # 输出: 大小: (4, 5)
println("长度: ", length(my_matrix))      # 输出: 长度: 20
println("维度: ", ndims(my_matrix))   # 输出: 维度: 2
```

### 索引和切片

通过索引访问和修改元素或子数组。Julia 使用 1-based 索引。

```julia
data_matrix = [10 20 30; 40 50 60; 70 80 90]
# 3×3 Matrix{Int64}:
#  10  20  30
#  40  50  60
#  70  80  90

first_element = data_matrix[1, 1]       # 10
second_row_third_col = data_matrix[2, 3]  # 60

# 切片
first_row = data_matrix[1, :]          # [10, 20, 30] (返回一个向量)
second_column = data_matrix[:, 2]      # [20, 50, 80] (返回一个向量)
sub_matrix = data_matrix[1:2, 2:3]     # [20 30; 50 60] (返回一个 2x2 矩阵)

# 使用 `end` 指代最后一个索引
last_element_first_row = data_matrix[1, end] # 30
```

### 元素级运算与广播

Julia 在数值计算方面的一个重要特性是广播。它允许你将函数按元素应用于数组，就像它们是标量一样，或者以兼容的方式组合不同形状的数组。这通过在运算符前或函数名后放置一个点 `.` 来调用。

```julia
A = [1 2; 3 4]
B = [10 20; 30 40]

# 元素级加法
C = A .+ B
# 输出:
# 2×2 Matrix{Int64}:
#  11  22
#  33  44

# 元素级乘法 (哈达玛积)
D = A .* B
# 输出:
# 2×2 Matrix{Int64}:
#  10   40
#  90  160

# 标量加法广播到所有元素
E = A .+ 5
# 输出:
# 2×2 Matrix{Int64}:
#  6  7
#  8  9

# 按元素应用函数
F = sin.(A) # 计算 A 中每个元素的正弦值
```

广播不仅仅是语法糖；它被高效实现，通常会融合操作以减少临时内存分配并提高性能。这在深度学习 (deep learning)中尤为实用，例如将偏置 (bias)向量 (vector)添加到激活矩阵的操作。

```julia
activations = rand(3, 4) # 一个 3x4 矩阵 (例如，3 个神经元，4 个样本)
bias_vector = [0.1, 0.2, 0.3] # 一个 3 元素向量 (每个神经元的偏置)

# 将 bias_vector 添加到 activations 的每一列
biased_activations = activations .+ bias_vector
# `bias_vector` 被视为 3x1 列向量并添加到 `activations` 的每一列
println(size(biased_activations)) # 输出: (3, 4)
```

## 线性代数：神经网络 (neural network)的语言

线性代数是大多数深度学习 (deep learning)算法所建立的数学依据。矩阵乘法、向量 (vector)点积和转置等运算随处可见。Julia 的 `LinearAlgebra` 标准库为这些任务提供了全面的工具集，通常使用 BLAS（基本线性代数子程序）和 LAPACK（线性代数包）等高度优化的后端库。

要使用这些函数，通常需要从以下代码开始：

```julia
using LinearAlgebra
```

### 常见运算

1. **点积**：两个向量 (vector) $u$ 和 $v$ 的点积是 $u^T v = \sum_i u_i v_i$。

   ```julia
   u = [1.0, 2.0, 3.0]
   v = [4.0, 5.0, 6.0]

   dot_product_uv = dot(u, v)      # 1.0*4.0 + 2.0*5.0 + 3.0*6.0 = 4.0 + 10.0 + 18.0 = 32.0
   # 或者，使用 Unicode 符号 ⋅ (输入方式为 \cdot<TAB>)
   dot_product_uv_alt = u ⋅ v      # 32.0
   ```
2. **矩阵乘法**：如果 $A$ 是一个 $m \times n$ 矩阵，而 $B$ 是一个 $n \times p$ 矩阵，它们的乘积 $C = AB$ 是一个 $m \times p$ 矩阵。

   ```julia
   M1 = [1 2; 3 4]      # 2x2 矩阵
   M2 = [5 6 7; 8 9 10] # 2x3 矩阵

   P = M1 * M2          # 结果是一个 2x3 矩阵
   # 输出:
   # 2×3 Matrix{Int64}:
   #  21  24  27
   #  47  54  61
   ```

   在深度学习 (deep learning)中，矩阵乘法是根本。例如，在全连接层中，输入会乘以一个权重 (weight)矩阵。
3. **转置**：矩阵 $A$ 的转置 $A^T$ 交换了它的行和列。

   ```julia
   A = [1 2 3; 4 5 6]
   A_transpose = transpose(A) # 或者 A'
   # 输出 (A_transpose):
   # 3×2 transpose(::Matrix{Int64}) with eltype Int64:
   #  1  4
   #  2  5
   #  3  6
   ```

   请注意，`A'` 会创建一个惰性的 `Transpose` 包装器。要获得一个新的密集矩阵，可以使用 `collect(A')` 或 `copy(transpose(A))`。
4. **单位矩阵**：单位矩阵 $I$ 是一个对角线上为一，其余为零的方阵。

   ```julia
   I3 = I(3) # 创建一个 3x3 单位矩阵 (UniformScaling 对象)
   # 要获得密集矩阵:
   dense_I3 = Matrix(I3)
   # 输出:
   # 3×3 Matrix{Bool} (在运算中提升为其他类型):
   #  1  0  0
   #  0  1  0
   #  0  0  1
   ```
5. **矩阵逆**：对于方阵 $A$，其逆矩阵 $A^{-1}$ 满足 $A A^{-1} = A^{-1} A = I$。

   ```julia
   S = [3.0 1.0; 1.0 2.0]
   S_inv = inv(S)
   # 输出:
   # 2×2 Matrix{Float64}:
   #   0.4  -0.2
   #  -0.2   0.6

   # 验证: S * S_inv 应该接近单位矩阵
   println(S * S_inv)
   # 输出:
   # 2×2 Matrix{Float64}:
   #  1.0          5.55112e-17
   #  5.55112e-17  1.0
   ```

   虽然 `inv()` 可用，但对于求解像 $Ax = b$ 这样的线性系统，使用反斜杠运算符 `x = A \ b` 通常在数值上更稳定、更高效。
6. **求解线性系统**：为了求解 $Ax = b$ 中的 $x$：

   ```julia
   A_sys = [2.0 1.0; 1.0 3.0]
   b_sys = [1.0, 2.0]
   x_sol = A_sys \ b_sys
   # 输出:
   # 2-element Vector{Float64}:
   #  0.2
   #  0.6

   # 验证: A_sys * x_sol 应该接近 b_sys
   println(A_sys * x_sol)
   # 输出:
   # [1.0, 2.0]
   ```

### 与深度学习 (deep learning)计算的关联

这些数组和线性代数运算不仅仅是抽象的数学工具；它们是深度学习的驱动力。思考一下神经网络 (neural network)中单个神经元的输出或全连接层的输出，通常表示为 $y = f(W x + b)$，这里的：

- $x$ 是输入特征向量 (vector)。
- $W$ 是权重 (weight)矩阵。
- $b$ 是偏置 (bias)向量。
- $W x$ 是矩阵-向量乘法。
- $W x + b$ 涉及将偏置向量 $b$ 添加到 $W x$ 的结果中（如果 $x$ 是批量输入，通常使用广播）。
- $f$ 是一个元素级激活函数 (activation function)（如 ReLU 或 sigmoid）。

下图说明了典型层计算的流程：

> 此图显示了输入特征 (X) 如何通过权重 (W) 和偏置 (b) 经过矩阵乘法和加法转换，然后通过激活函数，以生成层的输出 (Y)。

数据本身，无论是简单的特征集、图像还是文本序列，都表示为数组（在深度学习中常称为张量）。灰度图像可以是二维数组（高 x 宽），彩色图像可以是三维数组（高 x 宽 x 通道），而彩色图像的 mini-batch 则可以是四维数组（高 x 宽 x 通道 x 批量大小）。

因此，掌握 Julia 中的数组操作和线性代数是有效实现、理解和调试深度学习模型的直接先决条件。当你开始使用 Flux.jl 时，你会不断看到这些操作，尽管有时它们会抽象到更高级的层定义中。

## 参考资料

- [Multi-dimensional Arrays](https://docs.julialang.org/en/v1/manual/arrays/) — The Julia Language (2024)
  Julia 官方的多维数组文档，涵盖创建、索引和广播功能。
- [Linear Algebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) — The Julia Language (2024)
  Publisher: The Julia Language
  Julia 标准 LinearAlgebra 库的官方文档，详细介绍了数值计算的基本功能。
- [Deep Learning](https://www.deeplearningbook.org/contents/linear_algebra.html) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 2: Linear Algebra
  一本基础教材，全面概述了线性代数概念在深度学习中的应用。

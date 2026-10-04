# 在 Julia 中使用数组和矩阵

来源：[原文](https://apxml.com/zh/courses/julia-for-machine-learning/chapter-1-julia-ml-getting-started/julia-arrays-matrices)

[返回章节目录](README.md) · [返回课程目录](../README.md)

数组和矩阵是 Julia 数值计算的根本，进而也是大多数机器学习 (machine learning)任务的根本。它们提供了一种高效方式来存储和处理数据集合，例如特征集、模型参数 (parameter)或图像像素。Julia 对数组的实现尤为强大，兼具高性能和灵活、富有表现力的语法。假定您对编程有一定了解，Julia 对这些数据结构的处理方法成为关注的焦点。

您会发现 Julia 的数组是基于1的索引，这意味着第一个元素位于索引1，而非0，这在 Python 或 C++ 等语言中很常见。Julia 中的数组也是可变的，它们的元素可以是任意类型，但指定具体的元素类型（例如 `Float64`）通常会因为 Julia 的类型系统和方法专门化而带来更好的性能。

### 创建数组和矩阵

Julia 提供多种方式来创建数组，满足不同需求。

#### 向量 (vector)（一维数组）

向量是一维数组。它们本质上是元素的列表。

```julia
# 整数向量
vec1 = [10, 20, 30]
println(vec1) # 输出: [10, 20, 30]
println(typeof(vec1)) # 输出: Vector{Int64} (or Array{Int64, 1})

# 浮点数向量
vec2 = [1.5, 2.5, 3.5]
println(vec2) # 输出: [1.5, 2.5, 3.5]
println(typeof(vec2)) # 输出: Vector{Float64}

# 显式类型向量
vec3 = Float32[1, 2, 3] # 元素将为 Float32 类型
println(vec3) # 输出: Float32[1.0, 2.0, 3.0]
println(eltype(vec3)) # 输出: Float32

# 在 Julia 中，一维数组通常被视为列向量。
# 要创建 1xN 行向量（这是一个二维数组），可以使用：
row_vector = [1 2 3] # 注意是空格分隔，没有逗号
println(row_vector) # 输出: 1×3 Matrix{Int64}:
                    #          1  2  3
println(typeof(row_vector)) # 输出: Matrix{Int64} (or Array{Int64, 2})
```

#### 矩阵（二维数组）

矩阵是二维数组，是表示数据集的核心，其中行表示样本、列表示特征，或用于存储神经网络 (neural network)中的权重 (weight)矩阵。

```julia
# 创建一个 2x3 矩阵（2行，3列）
matrix1 = [1 2 3; 4 5 6]
println(matrix1)
# 输出:
# 2×3 Matrix{Int64}:
#  1  2  3
#  4  5  6

# 行中的元素用空格分隔，行与行之间用分号分隔。
# 如果需要，也可以用逗号分隔行中的元素，
# 但对于矩阵字面量，空格更常见。
matrix2 = Float64[1.0 2.0; 3.0 4.0]
println(matrix2)
# 输出:
# 2×2 Matrix{Float64}:
#  1.0  2.0
#  3.0  4.0
```

#### 高维数组

Julia 支持任意维度的数组。

```julia
# 一个三维数组 (2x2x2)
array3d = Array{Int, 3}(undef, 2, 2, 2) # 创建一个未初始化的三维数组
# 接下来您可以填充数值。
# 示例：
array3d[1,1,1] = 10
println(array3d[1,1,1]) # 输出: 10
```

`undef` 关键字表示创建数组时不会将其元素初始化为任何特定值；它们将包含该内存位置中的任意数据。

#### 初始化函数

Julia 提供便捷函数来创建具有特定初始值的数组：

```julia
# 一个未初始化的 2x3 Float64 矩阵
uninit_matrix = Matrix{Float64}(undef, 2, 3)

# 一个 3x2 的零矩阵
zeros_matrix = zeros(3, 2)
println(zeros_matrix)
# 输出:
# 3×2 Matrix{Float64}:
#  0.0  0.0
#  0.0  0.0
#  0.0  0.0

# 一个指定类型且全为1的向量
ones_vector = ones(Int8, 4)
println(ones_vector) # 输出: Int8[1, 1, 1, 1]

# 一个填充了值7的 2x2 矩阵
fill_matrix = fill(7, (2, 2))
println(fill_matrix)
# 输出:
# 2×2 Matrix{Int64}:
#  7  7
#  7  7

# 具有随机值的数组
rand_vector = rand(3)         # 包含3个0到1之间 Float64 值的向量
rand_matrix = rand(2, 2)      # 包含0到1之间 Float64 值的 2x2 矩阵
randn_matrix = randn(2,3)     # 包含来自标准正态分布的 Float64 值的 2x3 矩阵
```

#### 范围和推导式

数组也可以从范围构建，或通过使用推导式创建，这是一种基于迭代集合来创建数组的简洁方式。

```julia
# 从范围创建
range_array = collect(1:5) # 创建一个向量: [1, 2, 3, 4, 5]
println(range_array)

# 数组推导式
squares = [i^2 for i in 1:4] # 向量: [1, 4, 9, 16]
println(squares)

# 矩阵推导式
matrix_comp = [i * j for i in 1:2, j in 1:3]
println(matrix_comp)
# 输出:
# 2×3 Matrix{Int64}:
#  1  2  3
#  2  4  6
```

### 检查数组属性

一旦有了数组，您常常需要查询其属性：

```julia
my_matrix = [10 20; 30 40; 50 60]
# 3×2 Matrix{Int64}:
#  10  20
#  30  40
#  50  60

println(eltype(my_matrix))  # 输出: Int64 (元素类型)
println(size(my_matrix))    # 输出: (3, 2) (以元组表示的维度)
println(size(my_matrix, 1)) # 输出: 3 (第一维的大小 - 行)
println(size(my_matrix, 2)) # 输出: 2 (第二维的大小 - 列)
println(length(my_matrix))  # 输出: 6 (元素总数)
println(ndims(my_matrix))   # 输出: 2 (维度数量)

# axes 提供每个维度的有效索引范围
println(axes(my_matrix))    # 输出: (Base.OneTo(3), Base.OneTo(2))
println(axes(my_matrix, 1)) # 输出: Base.OneTo(3) (行的索引: 1到3)
```

### 索引和切片

访问和修改数组的元素或子部分是常用操作。请记住，Julia 是基于1的索引。

```julia
A = [1, 2, 3, 4, 5]
M = [10 20 30; 40 50 60]

# 访问单个元素
println(A[1])       # 输出: 1
println(M[2, 3])    # 输出: 60 (第2行，第3列)

# 使用 `end` 指代最后一个索引
println(A[end])     # 输出: 5
println(M[1, end])  # 输出: 30

# 切片获取子数组（切片默认创建副本）
sub_A = A[2:4]      # 索引 2, 3, 4 处的元素。输出: [2, 3, 4]
println(sub_A)

col_2 = M[:, 2]     # 所有行，第2列。输出: [20, 50] (一个向量)
println(col_2)

row_1 = M[1, :]     # 第1行，所有列。输出: [10, 20, 30] (一个向量)
println(row_1)

sub_M = M[1:2, [1, 3]] # 第1-2行，第1和第3列
println(sub_M)
# 输出:
# 2×2 Matrix{Int64}:
#  10  30
#  40  60

# 修改元素
A[1] = 100
M[2, 3] = 600
println(A)      # 输出: [100, 2, 3, 4, 5]
println(M)
# 输出:
# 2×3 Matrix{Int64}:
#  10  20   30
#  40  50  600
```

#### 逻辑索引

您可以使用布尔数组来选择元素：

```julia
data = [1.2, -0.5, 2.3, 0.0, -1.8]
positive_data = data[data .> 0] # 注意元素级比较 '.>'
println(positive_data) # 输出: [1.2, 2.3]
```

### 数组操作

#### 元素级操作（广播）

对于数组的逐元素操作，Julia 使用简洁的点（`.`）语法。这称为广播。如果两个数组大小相同，`A .+ B` 会将它们逐元素相加。如果有一个数组 `A` 和一个标量 `s`，`A .+ s` 会将 `s` 加到 `A` 的每个元素。广播非常高效，是数值代码中的常见模式。

```julia
X = [1 2; 3 4]
Y = [5 6; 7 8]
scalar = 10

# 逐元素相加
Z_add = X .+ Y
println(Z_add)
# 输出:
# 2×2 Matrix{Int64}:
#   6   8
#  10  12

# 广播标量
Z_scalar_add = X .+ scalar
println(Z_scalar_add)
# 输出:
# 2×2 Matrix{Int64}:
#  11  12
#  13  14

# 逐元素相乘
Z_mult = X .* Y
println(Z_mult)
# 输出:
# 2×2 Matrix{Int64}:
#   5  12
#  21  32

# 逐元素函数应用
Z_sin = sin.(X) # 将 sin 应用于 X 的每个元素
println(Z_sin)
# 输出（近似值）:
# 2×2 Matrix{Float64}:
#  0.841471  0.909297
#  0.14112  -0.756802
```

如果数组的维度兼容（例如，矩阵和向量 (vector)，其中向量可能被视为行或列），广播会智能处理不同形状数组之间的操作。

> 将向量 `V` 广播到矩阵 `M`，并将标量 `S` 广播到矩阵 `M`。维度会扩展或值会重复以匹配。

您也可以使用 `.=` 执行原地广播，它直接修改左侧的数组，可能节省内存分配：

```julia
A = [1.0, 2.0, 3.0]
B = [0.5, 0.5, 0.5]
A .+= B # A 现在是 [1.5, 2.5, 3.5]
println(A)
```

#### 线性代数

Julia 自带一个全面的 `LinearAlgebra` 标准库。您常常需要 `using LinearAlgebra` 来使用其函数。

```julia
using LinearAlgebra

mat_A = [1 2; 3 4]
mat_B = [5 0; 0 5] # 一个缩放矩阵
vec_v = [10, 20]

# 矩阵乘法
mat_C = mat_A * mat_B
println(mat_C)
# 输出:
# 2×2 Matrix{Int64}:
#   5  10
#  15  20

# 矩阵-向量乘法
result_vec = mat_A * vec_v
println(result_vec) # 输出: [50, 110]

# 转置
# A' 执行递归转置（对复数而言是共轭转置）
# transpose(A) 执行非递归转置
mat_A_T = mat_A'
println(mat_A_T)
# 输出:
# 2×2 adjoint(::Matrix{Int64}) with eltype Int64:
#  1  3
#  2  4

mat_A_transpose = transpose(mat_A)
println(mat_A_transpose)
# 输出:
# 2×2 transpose(::Matrix{Int64}) with eltype Int64:
#  1  3
#  2  4

# 点积（内积）
u = [1, 2, 3]
v = [4, 5, 6]
dot_prod = dot(u, v) # 或者对于实向量使用 u' * v
println(dot_prod) # 输出: 32 (1*4 + 2*5 + 3*6)

# 矩阵的逆
square_mat = [3.0 1.0; 1.0 2.0]
inv_mat = inv(square_mat)
println(inv_mat)
# 输出（近似值）:
# 2×2 Matrix{Float64}:
#   0.4  -0.2
#  -0.2   0.6

# 行列式
det_val = det(square_mat)
println(det_val) # 输出: 5.0

# 特征值和特征向量
eigen_decomp = eigen(square_mat)
eigenvalues = eigen_decomp.values
eigenvectors = eigen_decomp.vectors
println("特征值: ", eigenvalues)       # 输出: Eigenvalues: [1.58579, 3.41421]
println("特征向量:\n", eigenvectors)
# Output:
# Eigenvectors:
#  -0.850651  -0.525731
#   0.525731  -0.850651

# 奇异值分解 (SVD)
svd_decomp = svd(mat_A)
U, S_vals, V_mat = svd_decomp.U, svd_decomp.S, svd_decomp.V
# U 和 V 是正交矩阵，S_vals 是奇异值向量。
println("奇异值: ", S_vals) # 输出: Singular values: [5.46499, 0.365966]
```

这些线性代数操作是许多机器学习 (machine learning)算法的核心工具，从线性回归到主成分分析 (PCA) 以及深度学习 (deep learning)模型的内部机制。

### 修改数组

#### 重塑

您可以在不改变数组内容的情况下改变其维度，只要元素总数不变。`reshape` 通常返回原始数组内存的*视图*。

```julia
original_vec = collect(1:6) # [1, 2, 3, 4, 5, 6]
reshaped_mat = reshape(original_vec, (2, 3)) # 2行，3列
println(reshaped_mat)
# Output:
# 2×3 reshape(::Vector{Int64}, 2, 3) with eltype Int64:
#  1  3  5
#  2  4  6
# 注意列主序：元素先按列填充。

reshaped_mat[1,1] = 100
println(original_vec[1]) # 输出: 100，因为 reshape 返回一个视图
```

#### 扁平化

要将多维数组转换为一维向量 (vector)（列向量），请使用 `vec()`：

```julia
M = [1 2; 3 4]
V = vec(M)
println(V) # 输出: [1, 3, 2, 4] (列主序扁平化)
V[2] = 300
println(M[2,1]) # 输出: 300，vec 也返回一个视图
```

#### 连接

Julia 提供函数和语法来组合数组：

```julia
A = [1, 2]
B = [3, 4]
M1 = [1 2; 3 4]
M2 = [5 6; 7 8]

# 垂直连接
v_concat = vcat(A, B)       # 对于向量: [1, 2, 3, 4]
v_concat_mat = vcat(M1, M2)
println(v_concat_mat)
# Output:
# 4×2 Matrix{Int64}:
#  1  2
#  3  4
#  5  6
#  7  8

# 垂直连接的字面量语法
v_concat_literal = [A; B]
v_concat_mat_literal = [M1; M2]
println(v_concat_mat_literal == v_concat_mat) # 输出: true

# 水平连接
h_concat_mat = hcat(M1, M2)
println(h_concat_mat)
# Output:
# 2×4 Matrix{Int64}:
#  1  2  5  6
#  3  4  7  8

# 水平连接的字面量语法
# 确保维度兼容；A 和 B 是列向量。
A_col = [1; 2]
B_col = [3; 4]
h_concat_literal_vec = [A_col B_col] # 结果为 2x2 矩阵: [1 3; 2 4]
h_concat_mat_literal = [M1 M2]
println(h_concat_mat_literal == h_concat_mat) # 输出: true

# 使用 cat() 进行通用连接
C1 = ones(2,2)
C2 = zeros(2,2)
cat_dim3 = cat(C1, C2; dims=3) # 沿着新的第三维度连接
println(size(cat_dim3)) # 输出: (2, 2, 2)
```

对于一维 `Vector`，您还可以使用 `push!`、`pop!`、`append!`、`prepend!` 进行修改操作。

### 视图：高效处理子数组

当您切片数组时，例如 `sub_array = M[1:2, :]`，Julia 默认情况下会创建该数组部分的*副本*。如果您处理大型数据集或执行许多切片操作，这种复制在内存和时间方面效率不高。

另一种方法是创建*视图*。视图是一个 `SubArray` 对象，它引用原始数组的数据而不进行复制。修改视图*会*修改原始数组。

```julia
data_matrix = rand(5, 5)

# 切片创建副本
slice_copy = data_matrix[1:2, 1:2]
slice_copy[1,1] = 999.0
println(data_matrix[1,1]) # 原始矩阵未改变

# 创建视图
view_of_data = @view data_matrix[1:2, 1:2]
# or: view_of_data = view(data_matrix, 1:2, 1:2)

view_of_data[1,1] = -100.0
println(data_matrix[1,1]) # 输出: -100.0 (原始矩阵*已*改变)

# 视图在将数组部分传递给函数时很有用，可以避免复制开销
function process_row!(row_view)
    row_view .*= 2 # 原地修改行
end

first_row_view = @view data_matrix[1, :]
process_row!(first_row_view)
println(data_matrix[1,:]) # data_matrix 的第一行现在翻倍了。
```

当您需要操作数组的一部分时，尤其是在打算修改它或数组很大时，请使用视图以避免不必要的内存分配。如果您需要一个独立的副本，那么标准切片是合适的。

### 数组类型和性能说明

Julia 的优势之一是其类型系统。当您创建一个像 `Float64[1.0, 2.0, 3.0]` 这样的数组时，Julia 知道每个元素都是 `Float64` 类型。这使得编译器可以为该数组的操作生成高度专门化和优化的机器码。如果您创建一个像 `Any[1, "hello", 3.0]` 这样的数组，Julia 在运行时需要做更多工作来确定类型，这可能较慢。

对于机器学习 (machine learning)，您几乎总是使用具体类型的数组，通常是像 `Float64`、`Float32` 或 `Int` 这样的数字。这是 Julia 在数值任务中表现出高效率的一个重要因素。

Julia 还为数组定义了一个抽象层次结构：

- `AbstractArray{T,N}`: 所有 N 维数组的超类型，元素类型为 `T`。
- `AbstractVector{T}`: `AbstractArray{T,1}` 的别名。
- `AbstractMatrix{T}`: `AbstractArray{T,2}` 的别名。

编写接受 `AbstractArray`（或 `AbstractVector`、`AbstractMatrix`）的函数可以使您的代码通用，并与各种类似数组的结构一起工作，包括像稀疏数组或静态大小数组这样的专门化结构，我们在此处不会详细介绍它们，但它们值得了解。

### 数组和矩阵与机器学习 (machine learning)的关系

数组和矩阵是机器学习实现的通用语言：

- **特征矩阵 (X)**: 通常，数据集表示为矩阵，其中行是单个样本（或实例），列是特征（或属性）。如果您有 $m$ 个样本和 $n$ 个特征，您的特征矩阵 $X$ 将是 $m \times n$。
- **目标向量 (vector) (y)**: 在监督学习 (supervised learning)中，对应每个样本的标签或目标值通常存储在向量中。如果您有 $m$ 个样本，您的目标向量 $y$ 将有 $m$ 个元素。
- **模型参数 (parameter)（权重 (weight)/系数）**: 许多模型，如线性回归或神经网络 (neural network)，都具有以向量或矩阵形式存储和处理的参数（权重、偏置 (bias)）。例如，在线性回归 $y = Xw + b$ 中，$w$ 是一个权重向量。
- **中间计算**: 神经网络中的激活、优化过程中的梯度以及许多其他量都表示为数组。

让我们考虑一个简单示例：计算特征矩阵中每个特征的平均值。

```julia
# 示例特征矩阵（3个样本，2个特征）
X_features = [1.0 10.0;
              2.0 12.0;
              3.0 14.0]

# 计算每列（特征）的平均值
# Julia 中的 mean 函数可以沿着指定维度操作
using Statistics # 用于 mean 函数

mean_feature1 = mean(X_features[:, 1]) # 第一列的平均值
mean_feature2 = mean(X_features[:, 2]) # 第二列的平均值

println("特征 1 的平均值: ", mean_feature1) # 输出: 2.0
println("特征 2 的平均值: ", mean_feature2) # 输出: 12.0

# 更一般地，获取所有列的平均值：
feature_means = mean(X_features, dims=1) # dims=1 表示对每列沿着行进行操作
println("特征平均值（1x2 行矩阵）:\n", feature_means)
# Output:
# 特征平均值（1x2 行矩阵）:
#  2.0  12.0
```

熟练掌握 Julia 中的数组和矩阵操作是高效实现和理解机器学习算法的根本。富有表现力的语法、广播功能以及强大的线性代数支持相结合，使 Julia 成为处理这些任务的引人入胜的环境。随着学习深入，您将看到这些结构被广泛使用。

## 参考资料

- [Multi-dimensional Arrays](https://docs.julialang.org/en/v1/manual/arrays/) — The Julia Language Developers (2024)
  Julia数组数据结构官方指南，涵盖创建、索引、类型及性能方面。
- [Linear Algebra](https://docs.julialang.org/en/stable/manual/linear-algebra/) — The Julia Language Developers (2024)
  Publisher: The Julia Language
  Julia标准库中用于数组和矩阵线性代数运算函数的官方文档。
- [Introduction to Linear Algebra](https://math.mit.edu/linearalgebra/) — Gilbert Strang (2023)
  Publisher: Wellesley-Cambridge Press; Pages: 440
  一本经典教材，为数值计算中重要的线性代数概念构建了坚实的数学基础。
- [Julia for Data Analysis](https://www.manning.com/books/julia-for-data-analysis) — Bogumił Kamiński (2022)
  Publisher: Manning Publications
  提供使用Julia的数组和矩阵功能进行数据分析的实际示例和方法。

---

[上一节](04-%E6%95%B0%E6%8D%AE%E6%93%8D%E4%BD%9C%E6%89%80%E9%9C%80%E7%9A%84%E6%A0%B8%E5%BF%83%20Julia%20%E8%AF%AD%E6%B3%95.md) · [下一节](06-DataFrames.jl%20%E7%AE%80%E4%BB%8B.md)

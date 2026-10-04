---
course: "julia-for-machine-learning"
chapter: "julia-ml-getting-started"
lesson: "julia-setup-data-ops-practice"
sourceId: 6845
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-1-julia-ml-getting-started/julia-setup-data-ops-practice"
title: "实践操作：环境配置与基本数据处理"
description: "通过设置 Julia 机器学习环境并使用数组和 DataFrames 执行基本数据操作来应用您的知识。"
order: 8
plots: []
sourceHash: "f2bfed0f7eb0dd8dc568ca7144738f0c0e1ba83fa8c5a5bc6913c2b3f5edd305"
sourceCorrections: []
---

为了有效利用 Julia 进行机器学习 (machine learning)，建立项目专用环境和执行基本数据操作必不可少。本文将指导您完成这些主要步骤，演示如何使用 Julia 数组和 `DataFrames.jl` 包。这些技能是未来更复杂数据处理和模型构建任务的构成要素。

## 启动 Julia 并管理包

在开始之前，请确保您已安装并能访问 Julia。通常，您可以通过在终端或命令提示符中输入 `julia` 来启动 Julia 会话。这将打开 Julia 的读取-求值-打印循环 (REPL)，这是一个可以执行 Julia 代码的交互式环境。

```
$ julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version x.y.z (YYYY-MM-DD)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia>
```

对于任何机器学习 (machine learning)项目，管理依赖项都非常重要。Julia 内置的包管理器 `Pkg.jl` 可以帮助您为项目创建独立环境。现在，让我们为本次练习创建一个环境。

1. **进入 Pkg 模式**：在 Julia REPL 中，输入 `]` 切换到 Pkg REPL。提示符将变为 `pkg>`。
2. **激活环境**：要在当前工作目录中（例如，本课程练习的文件夹）创建并激活一个环境，请使用 `activate` 命令。

   ```pkg
   pkg> activate .
   ```

   如果当前目录中不存在 `Project.toml` 和 `Manifest.toml` 文件，此命令将创建它们。`Project.toml` 列出您的直接依赖项，而 `Manifest.toml` 记录所有依赖项的准确版本，以保证可复现性。
3. **添加所需包**：对于本次实践，我们将需要 `DataFrames.jl` 来处理表格数据，以及 `CSV.jl` 来读取逗号分隔值文件（尽管我们在这里会很少用到它）。

   ```pkg
   pkg> add DataFrames CSV
   ```

   此命令将下载这些包及其依赖项，并更新您的项目文件。
4. **检查状态**：您可以使用 `st`（`status` 的缩写）查看当前环境中已安装的包。

   ```pkg
   pkg> st
   ```

   这将列出 `CSV` 和 `DataFrames` 以及其他任何包。
5. **退出 Pkg 模式**：按下 `Backspace` 键返回标准 `julia>` REPL 提示符。

现在，我们的环境已准备就绪，让我们加载 `DataFrames` 以供本次会话使用。（注意：`CSV` 将在稍后涉及文件加载时使用。）

```julia
julia> using DataFrames
```

## 在 Julia 中使用数组和矩阵

数组是 Julia 中数值计算的根本。让我们来了解一些基本操作。请记住，Julia 使用 1-基于索引，这意味着数组的第一个元素在索引 1。

### 创建数组

您可以很方便地创建向量 (vector)（一维数组）和矩阵（二维数组）。

```julia
# 创建一个向量（一维数组）
vector_a = [10, 20, 30, 40, 50]
println("向量 A: ", vector_a)

# 创建一个具有特定元素类型的向量
vector_b = Float64[1.5, 2.5, 3.5]
println("向量 B (Float64): ", vector_b)

# 创建一个矩阵（二维数组）
# 空格分隔行内元素，分号开始新的一行
matrix_1 = [1 2 3; 4 5 6]
println("矩阵 1:\n", matrix_1)

# 创建一个用零初始化的矩阵
# zeros(类型, 行数, 列数)
matrix_zeros = zeros(Int, 2, 3) # 一个 2x3 的整数零矩阵
println("零矩阵:\n", matrix_zeros)

# 创建一个用一初始化的矩阵
matrix_ones = ones(2, 2) # 如果未指定类型，则默认为 Float64
println("一矩阵:\n", matrix_ones)
```

### 访问元素和切片

访问数组中的单个元素或子部分（切片）是直接的。

```julia
# 访问元素（1-基于索引）
println("vector_a 的第一个元素: ", vector_a[1])       # 输出: 10
println("matrix_1 第 2 行第 3 列的元素: ", matrix_1[2, 3]) # 输出: 6

# 切片
# 获取 vector_a 中索引 2 到 4 的元素
slice_vector_a = vector_a[2:4]
println("vector_a 的切片（第 2 到第 4 个）: ", slice_vector_a) # 输出: [20, 30, 40]

# 获取 matrix_1 的第一行
row_1 = matrix_1[1, :] # 冒号 ':' 表示该维度中的所有元素
println("matrix_1 的第一行: ", row_1) # 输出: [1, 2, 3]

# 获取 matrix_1 的第二列
col_2 = matrix_1[:, 2]
println("matrix_1 的第二列: ", col_2) # 输出: [2, 5]
```

### 数组操作和广播

Julia 在数值操作方面表现出色。对于数组上的元素级操作，Julia 使用“点”语法进行广播。

```julia
# 与标量进行元素级加法（广播）
vector_plus_5 = vector_a .+ 5 # 将 5 添加到 vector_a 的每个元素
println("向量 A + 5: ", vector_plus_5) # 输出: [15, 25, 35, 45, 55]

matrix_plus_10 = matrix_1 .+ 10 # 将 10 添加到 matrix_1 的每个元素
println("矩阵 1 + 10:\n", matrix_plus_10)

# 两个相同大小数组之间的元素级乘法
vector_mult = vector_a .* [2, 2, 2, 2, 2]
println("向量 A 元素级乘以 2: ", vector_mult)

matrix_b = [10 20 30; 40 50 60]
element_wise_prod = matrix_1 .* matrix_b # 必须具有相同维度
println("matrix_1 和 matrix_b 的元素级乘积:\n", element_wise_prod)
```

运算符（如 `.+`, `.*`）之前的点 `.` 告诉 Julia 将该操作应用于数组的每个元素，或应用于两个数组的对应元素之间。这是编写简洁高效数值代码的一个重要特点。

## `DataFrames.jl` 数据操作简介

`DataFrames.jl` 是 Julia 处理表格数据的主要工具，很像 Python 中的 Pandas 或 R 中的 data.frames。

### 创建 DataFrame

您可以通过多种方式创建 DataFrame，例如从数组或字典的集合创建。

```julia
# 确保 DataFrames 已加载（如果您的会话中尚未加载）
# using DataFrames

# 从命名列（向量）创建 DataFrame
df_contacts = DataFrame(
    ID = [1, 2, 3, 4],
    Name = ["Alice", "Bob", "Charlie", "Diana"],
    Age = [30, 24, 39, 28],
    City = ["New York", "Paris", "London", "Berlin"]
)

println("DataFrame 'df_contacts':")
display(df_contacts) # display() 在某些环境中通常提供更好的格式化
```

### 检查 DataFrame

一旦您有了 DataFrame，您就会希望检查其结构和内容。

```julia
# 获取维度（行数，列数）
println("df_contacts 的大小: ", size(df_contacts)) # 例如，(4, 4)

# 获取列名
println("列名: ", names(df_contacts))

# 显示前几行
println("df_contacts 的前 2 行:")
display(first(df_contacts, 2))

# 获取数值列的摘要统计信息
println("df_contacts 的摘要统计信息:")
display(describe(df_contacts))
```

`describe` 函数为每列提供计数、均值、最小值、最大值以及其他有用的统计信息。

### 选择数据

您可以根据多种条件选择列或行。

```julia
# 选择单列（返回向量）
ages_vector = df_contacts.Age
println("年龄（作为向量）: ", ages_vector)

# 另一种选择单列的方式（返回包含一列的 DataFrame）
cities_df = df_contacts[:, :City] # 注意：这可能因上下文/版本而异
# 作为向量的单列：df_contacts[!, :City]
# 作为 DataFrame 的单列：df_contacts[:, [:City]]
cities_as_df_column = df_contacts[:, [:City]]
println("城市（作为 DataFrame）:")
display(cities_as_df_column)

# 选择多列（返回新的 DataFrame）
name_and_city_df = df_contacts[:, [:Name, :City]]
println("姓名和城市:")
display(name_and_city_df)

# 按索引选择行
first_two_rows = df_contacts[1:2, :]
println("按索引的前两行:")
display(first_two_rows)
```

### 过滤行

基于条件过滤行是一种常见操作。

```julia
# 年龄大于 30 岁的人
older_than_30 = df_contacts[df_contacts.Age .> 30, :] # 注意用于比较的广播点
println("年龄大于 30 岁的联系人:")
display(older_than_30)

# 使用 filter 函数筛选来自纽约的联系人
# filter(列 => 条件函数, 数据框)
from_new_york = filter(:City => city -> city == "New York", df_contacts)
println("来自纽约的联系人（使用 filter）:")
display(from_new_york)
```

### 关于加载外部数据的一点说明

虽然我们是以编程方式创建 `df_contacts`，但您将经常处理来自外部文件的数据，例如 CSV 文件。我们已添加到环境中的 `CSV.jl` 包就是为此目的而用。下面快速展示如何加载 CSV 文件：

```julia
# 文件 'sample_data.csv' 的内容：
# product_id,category,price
# P101,Electronics,299.99
# P102,Books,24.50
# P103,Electronics,49.00

# 加载此文件（假设它在您的工作目录中）：
# using CSV # 确保 CSV.jl 已加载
# df_products = CSV.read("sample_data.csv", DataFrame)
# display(df_products)
```

我们提供此作为常见做法的示例。详细的数据加载和保存技术将在第 2 章中介绍。目前，熟练掌握对以编程方式创建的 DataFrame 的操作已足够。

## 实践操作总结

在本次实践部分中，您迈出了重要的第一步：

- 您已学会使用 `Pkg.jl` 初始化 Julia 项目环境并添加如 `DataFrames.jl` 和 `CSV.jl` 等所需包。
- 您已练习创建和操作 Julia 数组及矩阵，执行元素访问、切片和广播操作。
- 您已创建了您的第一个 DataFrames，检查了它们的结构，选择了特定列和行，并根据条件过滤了数据。

这些针对数组和 DataFrame 的基本操作是您在任何基于 Julia 的机器学习 (machine learning)项目中将反复使用的技能。它们为后续章节中您将遇到的更高级数据准备、模型构建和评估方法提供了前提。掌握了这些基本内容后，您已能很好地进行更精细的数据处理任务。

## 参考资料

- [The Julia Language Documentation](https://docs.julialang.org/) — The Julia Language (2024)
  Julia 编程语言的官方文档，涵盖了 REPL、用于环境管理的 Pkg.jl、数组和广播等基础概念。
- [DataFrames.jl Documentation](https://dataframes.juliadata.org/stable/) — The DataFrames.jl Contributors (2024)
  DataFrames.jl 包的官方文档，详细介绍了在 Julia 中创建、检查、选择和过滤表格数据的方法。
- [Julia for Data Analysis](https://www.manning.com/books/julia-for-data-analysis) — Bogumił Kamiński (2022)
  Publisher: Manning Publications
  一本内容全面的书籍，提供了使用 Julia 进行数据分析的实用指导，广泛涵盖了数组、DataFrames.jl 和 CSV.jl。

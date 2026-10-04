---
course: "julia-for-machine-learning"
chapter: "julia-ml-getting-started"
lesson: "intro-dataframes-jl"
sourceId: 6841
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-1-julia-ml-getting-started/intro-dataframes-jl"
title: "DataFrames.jl 简介"
description: "开始使用 DataFrames.jl，这是 Julia 用于处理表格数据的主要包，功能类似于 Python 中的 Pandas。"
order: 6
plots: []
sourceHash: "2b05b6852b516aca025fc3edf6bb15a4d729c325ad97ccd3980badda5431cef1"
sourceCorrections: []
---

虽然 Julia 的数组和矩阵在数值计算方面功能强大，但机器学习 (machine learning)任务通常涉及结构更清晰、类型更多样的数据集。你将经常看到以表格形式组织的数据，很像电子表格或 SQL 数据库表，其中各列可以保存不同类型的信息，如数字、文本或日期。为了在 Julia 中有效处理这类表格数据，`DataFrames.jl` 包是必不可少的。

`DataFrames.jl` 提供 `DataFrame` 类型，这是一种二维、大小可变、表格状的数据结构，其中各列有名称，并且通常存储特定类型的数据。如果你有使用 Python 的 Pandas 库或 R 的数据框的经验，你会觉得 `DataFrames.jl` 的设计理念非常熟悉。它旨在让数据操作直观且高效，为 Julia 中的许多数据分析和机器学习工作奠定根基。

> `DataFrame` 将数据组织成命名列，每列都有特定的数据类型。请注意，`Age` 列如何可以同时包含 `Missing` 值和 `Float64`。

### 创建 DataFrame

在使用 `DataFrames.jl` 之前，你需要将其安装并加载到你的 Julia 会话中。如果你遵循了“设置 Julia 机器学习 (machine learning)环境”和“使用 Pkg.jl 进行包管理”中的设置，你可能已经安装了它。如果没有，你可以使用 Julia 的包管理器来添加它：

```julia
# 按 ] 进入 Pkg 模式
# pkg> add DataFrames
# 按 Backspace 退出 Pkg 模式
```

安装后，使用 `using` 将其引入当前 Julia 会话：

```julia
using DataFrames
```

有几种方式可以构建 `DataFrame`。一种常见方法是提供一个命名列的集合，其中每列都是一个向量 (vector)：

```julia
df = DataFrame(
    ID = [1, 2, 3, 4],
    Name = ["Alice", "Bob", "Charlie", "David"],
    Age = [25, 30, 22, 35],
    Score = [88.5, 92.0, 77.3, 85.0]
)
```

当你在 Julia REPL 或 Jupyter notebook 中执行这段代码时，`DataFrame` 将以表格形式显示：

```
4×4 DataFrame
 Row │ ID     Name     Age    Score   
     │ Int64  String   Int64  Float64 
─────┼─────────────────────────────────
   1 │     1  Alice       25     88.5
   2 │     2  Bob         30     92.0
   3 │     3  Charlie     22     77.3
   4 │     4  David       35     85.0
```

你也可以从矩阵创建 `DataFrame`，单独提供列名：

```julia
using Random # 用于生成随机数据
Random.seed!(42); # 用于结果重现
matrix_data = rand(3, 2)
df_from_matrix = DataFrame(matrix_data, [:Feature1, :Feature2])
```

这会产生：

```
3×2 DataFrame
 Row │ Feature1  Feature2 
     │ Float64   Float64  
─────┼────────────────────
   1 │ 0.51387   0.701026
   2 │ 0.175891  0.223533
   3 │ 0.673325  0.493094
```

### 检查你的 DataFrame

一旦你有了 `DataFrame`，你会想查看它的结构和内容。以下是一些基本函数：

- `size(df)`: 返回一个元组 `(行数, 列数)`。
- `nrow(df)` 和 `ncol(df)`: 分别给出行数和列数。
- `names(df)`: 返回列名数组（字符串形式）。
- `eltype.(eachcol(df))`: 显示每列的数据类型。这对于确认 Julia 如何解释你的数据很有用。
- `first(df, n)`: 显示 DataFrame 的前 `n` 行（类似于其他系统中的 `head()` 函数）。
- `last(df, n)`: 显示最后 `n` 行。
- `describe(df)`: 为每列提供汇总统计，例如数值列的均值、中位数、最小值、最大值和缺失值数量，以及分类列的计数。

让我们在 `df` 上试试这些函数：

```julia
println("大小: ", size(df))
println("行数: ", nrow(df))
println("列名: ", names(df))
println("列类型: ", eltype.(eachcol(df)))

println("\n前 2 行:")
show(stdout, "text/plain", first(df, 2)) # 使用 show 以获得更好的 REPL 样式输出
println("\n\n汇总统计:")
show(stdout, "text/plain", describe(df, :mean, :min, :max, :nmissing)) # 选择特定统计量
println()
```

Output:

```
Size: (4, 4)
Number of rows: 4
Column names: ["ID", "Name", "Age", "Score"]
Column types: [Int64, String, Int64, Float64]

First 2 rows:
2×4 DataFrame
 Row │ ID     Name   Age    Score   
     │ Int64  String Int64  Float64 
─────┼───────────────────────────────
   1 │     1  Alice     25     88.5
   2 │     2  Bob       30     92.0

Summary statistics:
4×5 DataFrame
 Row │ variable  mean     min      max      nmissing 
     │ Symbol    Union…   Any      Any      Int64    
─────┼────────────────────────────────────────────────
   1 │ ID        2.5      1        4               0
   2 │ Name               Alice    David           0
   3 │ Age       28.0     22       35              0
   4 │ Score     85.7     77.3     92.0            0
```

### 访问数据

你可以通过多种方式访问 `DataFrame` 中的数据，通过列名（`Symbol` 或字符串）和行整数索引来引用。

**访问列：**
要选择一列或多列：

- 作为向量 (vector)（如果选择一列）：`df.ColumnName` 或 `df[!, :ColumnName]`。`!` 表示你想要实际的列向量，而不是一个新的单列 DataFrame。

  ```julia
  ages = df.Age  # 将 'Age' 列作为向量访问
  scores = df[!, :Score] # 也将 'Score' 作为向量访问
  println("年龄: $ages")
  ```
- 作为新的 DataFrame（如果选择一列或多列）：`df[:, :ColumnName]` 或 `df[:, [:Col1, :Col2]]`。

  ```julia
  name_and_score_df = df[:, [:Name, :Score]]
  println("\n姓名和分数 DataFrame:")
  show(stdout, "text/plain", name_and_score_df)
  println()
  ```

**访问行：**
通过整数位置（1-索引）选择行：

- 单行：`df[row_index, :]`。这会返回一个 `DataFrameRow`，其行为类似于单行 DataFrame。

  ```julia
  first_row = df[1, :]
  println("\n第一行: $first_row")
  ```
- 多行（切片）：`df[start_index:end_index, :]`。这会返回一个新的 DataFrame。

  ```julia
  first_two_rows = df[1:2, :]
  println("\n前两行 DataFrame:")
  show(stdout, "text/plain", first_two_rows)
  println()
  ```

**访问特定元素：**
要获取单个值，请同时指定行和列：

```julia
alices_score = df[1, :Score]       # Alice 在第 1 行
bobs_age = df[df.Name .== "Bob", :Age][1] # 更高级：条件选择然后索引
println("\nAlice 的分数: $alices_score")
println("Bob 的年龄: $bobs_age")
```

表达式 `df.Name .== "Bob"` 创建一个布尔向量，然后用于过滤行。由于这会返回一个包含 Bob 年龄的单元素向量，我们使用 `[1]` 来提取该值。

### 为什么机器学习 (machine learning)要用 DataFrames.jl？

`DataFrames.jl` 不仅仅是一个表格。它的设计为机器学习任务提供了多项优势：

1. **自然的数据表示：** 大多数监督学习 (supervised learning)数据集（特征和目标变量）与 DataFrame 结构完美匹配。
2. **高效操作：** 由于列带有类型，Julia 编译器可以对列上的操作进行高度优化，从而获得良好的性能。
3. **处理缺失数据：** 数据通常很混乱，包含缺失值。`DataFrames.jl` 支持 Julia 的 `missing` 值，使你能够表示和处理不完整的数据集。你将在第 2 章看到更多相关内容。
4. **与生态系统集成：** `DataFrames.jl` 可以与 Julia 生态系统中其他对机器学习很重要的包顺畅结合，例如 `MLJ.jl`（用于模型构建）、`Plots.jl`（用于数据可视化）以及各种统计包。

本简介为你提供了 `DataFrames.jl` 的基本信息以及如何执行基本操作。随着学习的深入，你会看到 `DataFrame` 对象在为 Julia 中的机器学习模型准备数据方面居于核心地位。下一章，“Julia 中的数据操作与准备”，将大幅扩展这些功能，涵盖如何从文件加载数据、清理数据、转换特征以及执行更复杂的操作。

## 参考资料

- [DataFrames.jl Documentation](https://dataframes.juliadata.org/stable/) — The DataFrames.jl Contributors (2024)
  DataFrames.jl 包的综合指南，涵盖了 Julia 中表格数据操作的安装、数据结构和所有功能。
- [Julia for Machine Learning: Master the art of machine learning with Julia to solve real-world problems](https://www.packtpub.com/product/julia-for-machine-learning/9781838647037) — Erik Engheim (2020)
  Publisher: Packt Publishing
  这本书提供了使用 Julia 进行机器学习的实用指南，其中包含专门介绍 DataFrames.jl 数据操作的相关章节。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  尽管专注于 Python 的 Pandas 库，这本基础书籍详细阐述了表格数据结构的设计原则和实际应用，这些原则和应用与 DataFrames.jl 直接相似。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  这本广泛使用的机器学习书籍提供了数据准备和操作的实际示例，强化了结构化表格数据（通常使用 Pandas）对机器学习工作流程的普遍重要性。

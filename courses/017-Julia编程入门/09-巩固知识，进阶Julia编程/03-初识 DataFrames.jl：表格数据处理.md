# 初识 DataFrames.jl：表格数据处理

来源：[原文](https://apxml.com/zh/courses/getting-started-julia-programming/chapter-9-consolidating-knowledge-advancing-julia/first-look-dataframes-jl-tabular-data)

[返回章节目录](README.md) · [返回课程目录](../README.md)

您会在科学计算、数据分析和机器学习 (machine learning)项目中遇到很多以表格形式组织的数据，其中行表示单个观测值，列表示不同的属性或变量。可以想想您使用过的电子表格；这种结构正是我们所指的。为了在 Julia 中高效处理这类数据，社区开发了一个功能强大且广泛使用的包：`DataFrames.jl`。

`DataFrames.jl` 提供专门的数据结构和函数，专门用于高效、易用地处理表格数据。它让您能够以一种既能胜任复杂任务，又对常见操作简单明了的方式加载、操作、清理和分析数据。如果您接触过 Python 中的 Pandas 或 R 中的数据框，您会发现 `DataFrames.jl` 在 Julia 生态系统中扮演着类似的角色。它是 Julia 中许多以数据为中心的工作流程的重要组成部分。

### 开始使用 DataFrames.jl

在使用 `DataFrames.jl` 之前，您需要将其添加到您的 Julia 环境中。Julia 中的“包”是预先编写好的代码集合，可以增强 Julia 的功能。您可以使用 Julia 内置的包管理器 `Pkg` 来添加 `DataFrames.jl`。如果您尚未安装它，请打开您的 Julia REPL（交互式命令行）并输入：

```julia
using Pkg
Pkg.add("DataFrames")
```

此命令会下载 `DataFrames.jl` 及其依赖项，使其可用于您的项目。您只需在 Julia 安装中执行此操作一次。

安装完成后，您可以在任何 Julia 会话或脚本中通过编写以下内容来开始使用它：

```julia
using DataFrames
```

此行加载 `DataFrames` 模块，使其函数和类型（例如 `DataFrame` 类型本身）在当前作用域中可用。

### 创建您的第一个 DataFrame

让我们创建一个简单的 DataFrame 来了解其运作方式。`DataFrame` 可以通过多种方式构建，但一种常见的方法是为您的列提供名称，并为每列提供相应的矢量数据（类似于 Julia 数组）。

假设我们有几名学生的数据：他们的 ID、姓名、年龄和考试分数。我们可以这样表示：

```julia
# 确保已加载 DataFrames
using DataFrames

# 创建一个 DataFrame
df = DataFrame(
    ID = [101, 102, 103, 104],
    Name = ["Alice", "Bob", "Charlie", "Diana"],
    Age = [23, 21, 24, 22],
    Score = [88.5, 92.0, 77.5, 95.0]
)

# 显示 DataFrame
println(df)
```

运行此代码时，Julia 会在您的控制台打印出一个格式整齐的表格：

```
4×4 DataFrame
 行 │ ID     姓名     年龄    分数
     │ Int64  String   Int64  Float64
─────┼───────────────────────────────────
   1 │   101  Alice       23     88.5
   2 │   102  Bob         21     92.0
   3 │   103  Charlie     24     77.5
   4 │   104  Diana       22     95.0
```

请注意输出如何显示 DataFrame 的尺寸（4 行 × 4 列）、列名、每列的数据类型，然后是数据本身。这种即时视觉反馈对掌握数据结构非常有益。

### 查看 DataFrame 的基本方法

拥有 DataFrame 后，您会想要查看其内容。以下是一些基本操作：

- **查看尺寸**：获取行数和列数：

  ```julia
  println(size(df))  # 输出: (4, 4)
  println(nrow(df))  # 输出: 4 (行数)
  println(ncol(df))  # 输出: 4 (列数)
  ```
- **查看前几行**：如果您的 DataFrame 很大，您可能只想看开头或结尾部分。

  ```julia
  println(first(df, 2)) # 显示前 2 行
  ```

  这会输出：

  ```
  2×4 DataFrame
   行 │ ID     姓名   年龄    分数
       │ Int64  String Int64  Float64
  ─────┼───────────────────────────────────
     1 │   101  Alice     23     88.5
     2 │   102  Bob       21     92.0
  ```

  类似地，`last(df, 2)` 会显示最后两行。
- **获取列名**：

  ```julia
  println(names(df)) # 输出: ["ID", "姓名", "年龄", "分数"]
  ```
- **统计概要**：`describe` 函数提供每列的快速统计概要。

  ```julia
  println(describe(df))
  ```

  这会为您提供诸如数值列的平均值、最小值、最大值、中位数和缺失值数量等信息，以及不同类型的其他相关信息。这是初步了解数据集的一种好方法。
- **访问列**：您可以使用列名将单个列作为矢量获取。有几种方法可以做到这一点：

  ```julia
  ages = df.Age
  println(ages)
  # 输出: [23, 21, 24, 22]

  scores = df[!, :Score] # ! 表示“所有行”，:Score 是列名符号
  println(scores)
  # 输出: [88.5, 92.0, 77.5, 95.0]
  ```

  `df.Age` 和 `df[!, :Score]` 都会将列作为 Julia 矢量返回。`Score` 前的冒号 `:`（即 `:Score`）会创建一个 `Symbol`，这是 `DataFrames.jl` 内部通常引用列名的方式。
- **访问行**：您可以通过索引访问特定行。例如，要获取第一行：

  ```julia
  first_row = df[1, :] # 1 是行索引，: 表示“所有列”
  println(first_row)
  ```

  这会返回一个 `DataFrameRow` 对象，它表示单行，但仍然知道其列名。

本次简要介绍仅展示了 `DataFrames.jl` 功能的很小一部分。它提供了丰富的功能集，用于数据清洗（处理缺失值、转换数据类型）、根据条件筛选行、选择特定列、数据分组、合并多个 DataFrames，等等。

随着您在 Julia 方面的进步，特别是在数据分析、机器学习 (machine learning)或任何涉及结构化数据集的方面，`DataFrames.jl` 可能会成为您工具箱中不可或缺的工具。我们鼓励您查阅其详细文档，并在遇到不同数据难题时尝试其功能。有效管理和准备表格数据的能力是一项基本技能，`DataFrames.jl` 在 Julia 中为此提供了出色的支持。

## 参考资料

- [DataFrames.jl Documentation](https://dataframes.juliadata.org/stable/) — The DataFrames.jl Developers (2024)
  DataFrames.jl 的权威和最新资源，涵盖其所有功能和最佳实践。
- [Julia for Data Analysis](https://www.packtpub.com/product/julia-for-data-analysis/9781800561501) — Jose Storopoli, Rik Huijzer, and Long Quan (2021)
  Publisher: Packt Publishing
  提供使用 DataFrames.jl 在 Julia 生态系统中进行数据操作、清理和分析的实践指南。
- [Pkg.jl Documentation](https://pkgdocs.julialang.org/v1/) — The Julia Language Developers (2024)
  Julia 内置包管理器 Pkg.jl 的文档，对于添加和管理 DataFrames.jl 等包至关重要。

---

[上一节](02-%E5%A4%9A%E9%87%8D%E6%B4%BE%E5%8F%91%E7%9A%84%E5%AE%9E%E9%99%85%E8%BF%90%E7%94%A8%EF%BC%9A%E7%AE%80%E5%8D%95%E7%A4%BA%E4%BE%8B.md) · [下一节](04-%E4%BD%BF%E7%94%A8%20Plots.jl%20%E8%BF%9B%E8%A1%8C%E6%95%B0%E6%8D%AE%E5%8F%AF%E8%A7%86%E5%8C%96%EF%BC%9A%E6%A6%82%E8%A7%88.md)

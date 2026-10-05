# 数据科学所需的核心Julia包

来源：[原文](https://apxml.com/zh/courses/julia-deep-learning/chapter-1-foundations-julia-ml/julia-data-science-packages)

[返回章节目录](README.md) · [返回课程目录](../README.md)

有效的机器学习 (machine learning)始于结构良好且干净的数据。Julia的生态系统提供了一系列强大的包，旨在高效处理数据和I/O，在您开始构建模型之前，它们就已成为您工具集的重要组成部分。这些工具不仅性能良好，充分利用了Julia的速度优势，还与语言的类型系统和多重分派等特性良好结合，从而生成富有表现力且高效的数据处理代码。

### 处理表格数据：DataFrames.jl

`DataFrames.jl` 是Julia中大多数数据分析和准备工作流程的核心。如果您来自Python或R，您会发现它的作用很熟悉：它提供了一个`DataFrame`对象，这是一个二维、大小可变且可能包含不同类型数据的表格结构，带有标签轴（行和列）。可以把它想象成电子表格或SQL表，但直接在您的Julia环境中操作。

`DataFrames.jl` 允许您：

- 从各种来源加载数据（通常与`CSV.jl`等其他包配合使用）。
- 选择、筛选和转换列。
- 对数据进行分组并执行聚合操作。
- 妥善处理缺失值。
- 连接和合并多个数据集。

让我们看一个快速示例。首先，请确保您已安装该包。在Julia REPL中，您可以键入`]`进入Pkg模式，然后输入`add DataFrames`来添加它。

```julia
using DataFrames

# 从数组字典创建DataFrame
data = Dict(
    "ID" => [1, 2, 3, 4, 5],
    "Age" => [25, 30, 22, 35, 28],
    "Salary" => [50000, 65000, 45000, 75000, 62000],
    "Department" => ["HR", "Engineering", "Marketing", "Engineering", "HR"]
)
df = DataFrame(data)

println("原始DataFrame:")
println(df)
```

这将输出：

```
原始DataFrame:
5×4 DataFrame
 行 │ ID     年龄   薪水    部门
     │ Int64  Int64  Int64   String
─────┼──────────────────────────────────
   1 │     1     25   50000  HR
   2 │     2     30   65000  Engineering
   3 │     3     22   45000  Marketing
   4 │     4     35   75000  Engineering
   5 │     5     28   62000  HR
```

您可以轻松执行选择特定列或筛选行等操作：

```julia
# 选择特定列
selected_cols = df[!, [:ID, :Salary]]
println("\n选择的列 (ID和薪水):")
println(selected_cols)

# 筛选行：工程部门的员工或年龄大于30岁的员工
filtered_df = filter(row -> row.Department == "Engineering" || row.Age > 30, df)
println("\n筛选后的DataFrame (工程部门或年龄 > 30):")
println(filtered_df)
```

`df[!, [:ID, :Salary]]` 中的 `!` 用于选择所有行。`DataFrames.jl` 提供了一种用于数据操作的丰富迷你语言，该语言有详细文档，在为机器学习 (machine learning)模型准备数据集方面非常强大。

### 读取和写入数据：CSV.jl

尽管`DataFrames.jl`可以帮助您在数据进入内存后对其进行处理，但您首先需要加载它。逗号分隔值（CSV）文件是存储表格数据的一种普遍使用的格式。Julia的`CSV.jl`包是一个快速灵活的工具，用于读写CSV文件。

`CSV.jl`的一个主要特点是它与`DataFrames.jl`直接集成。将CSV文件读取到DataFrame中是简单直接的：

```julia
using CSV
using DataFrames

# 假设您有一个名为 'employees.csv' 的文件，内容如下：
# ID,Name,Age,Salary
# 1,Alice,30,70000
# 2,Bob,24,50000
# 3,Charlie,35,80000

# 为了模拟这种情况，我们先写入一个临时CSV文件
csv_content = """
ID,Name,Age,Salary
1,Alice,30,70000
2,Bob,24,50000
3,Charlie,35,80000
"""
open("employees.csv", "w") do f
    write(f, csv_content)
end

# 将CSV文件读取到DataFrame中
df_from_csv = CSV.read("employees.csv", DataFrame)

println("\n从CSV加载的DataFrame:")
println(df_from_csv)

# 清理临时文件
rm("employees.csv")
```

`CSV.read("employees.csv", DataFrame)` 命令会自动推断列类型并高效地解析文件。`CSV.jl` 还可以处理各种分隔符、编码以及CSV文件中常见的其他复杂情况。同样地，`CSV.write("output.csv", df)` 会将DataFrame `df` 保存到CSV文件中。

`CSV.jl` 的性能是一个显著优势，尤其是在处理深度学习 (deep learning)中常见的大型数据集时。

### 数据处理工作流程

通常，您在Julia机器学习 (machine learning)项目中的初始数据处理步骤将涉及使用`CSV.jl`将数据加载到`DataFrame`中，然后使用`DataFrames.jl`的丰富功能来检查、清理、转换和进行数据集的特征工程。这些准备好的数据随后成为模型训练管道的输入。

以下图表说明了这种常见的工作流程：

> 使用`CSV.jl`和`DataFrames.jl`的典型数据摄取和准备管道。

### 生态系统中其他有用的包

尽管 `DataFrames.jl` 和 `CSV.jl` 是处理表格数据的重要部分，但Julia的数据科学生态系统还包括许多其他专用包：

- **`JSON3.jl`**: 用于读写JSON格式数据，这在Web API和配置文件中很常见。
- **`Arrow.jl`**: 实现了Apache Arrow列式格式，可在Julia与其他系统（如Spark、Pandas或R）之间实现高性能数据交换。
- **`Missings.jl`**: 提供处理缺失数据的工具，补充了`DataFrames.jl`中已有的功能。
- **`Statistics.jl`**: Julia标准库的一部分，提供基本的统计函数（均值、中位数、标准差、相关性），这些函数在对DataFrame进行探索性数据分析时经常使用。
- **`Chain.jl` 或 `Pipe.jl`**: 这些包提供了宏（`@chain` 或 `@pipe`），允许您以更具可读性的管道方式编写数据转换序列，类似于R中的`dplyr`或Pandas中的方法链。例如：

```julia
using DataFrames
using Chain # 假设 Chain.jl 已安装

# 示例DataFrame（与之前相同）
data = Dict(
    "ID" => [1, 2, 3, 4, 5],
    "Age" => [25, 30, 22, 35, 28],
    "Salary" => [50000, 65000, 45000, 75000, 62000],
    "Department" => ["HR", "Engineering", "Marketing", "Engineering", "HR"]
)
df = DataFrame(data)

@chain df begin
    filter(row -> row.Department == "Engineering", _)
    groupby(:Department)
    combine(nrow => :count, :Salary => mean => :average_salary)
end
```

这段代码首先筛选出"Engineering"部门，然后按部门分组（本例中只会是"Engineering"），最后计算该组的计数和平均薪水。`_` 作为链中前一个操作结果的占位符。

### 使用Plots.jl和DataFrames.jl进行数据可视化

探索性数据分析（EDA）是模型构建前的重要步骤。`Plots.jl`是Julia中一个流行的绘图元包，它与`DataFrames.jl`集成良好。例如，您可以快速从DataFrame的某一列生成直方图。

让我们假设我们的`df_from_csv`（来自`CSV.jl`示例）已加载，并且我们想查看“年龄”的分布。



[交互图表：员工年龄分布](https://apxml.com/zh/courses/julia-deep-learning/chapter-1-foundations-julia-ml/julia-data-science-packages#plot-55mgi)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "员工年龄分布",
    "xaxis": {
      "title": "年龄"
    },
    "yaxis": {
      "title": "频率"
    },
    "bargap": 0.1,
    "plot_bgcolor": "#e9ecef",
    "paper_bgcolor": "white"
  },
  "data": [
    {
      "type": "histogram",
      "x": [
        30,
        24,
        35
      ],
      "marker": {
        "color": "#339af0"
      },
      "name": "年龄"
    }
  ]
}
```

</details>



> 直方图显示了示例数据集中年龄的分布。使用`Plots.jl`结合`DataFrames.jl`的数据可以简单地生成此类可视化。

在Julia中生成此类图表（安装`Plots.jl`后）：

```julia
using Plots
# 假设 df_from_csv 已根据 CSV.jl 示例填充
# histogram(df_from_csv.Age, title="年龄分布", xlabel="年龄", ylabel="频率", legend=false, color=:blue)
```

这一行（已注释掉，以使文档自包含，在生成时无需运行Julia绘图代码）将生成一个与上面JSON描述的类似的直方图。

掌握这些数据处理包是您在Julia中进行任何认真的机器学习 (machine learning)工作的前提。它们提供了加载、清理、转换和理解数据的手段，确保您馈入深度学习 (deep learning)模型的数据质量高且格式正确。随着您学习的进展，您会发现这些工具的性能和表达能力会显著促进高效的开发工作流程。您在这里培养的技能将在使用`Flux.jl`构建神经网络 (neural network)时直接用于数据集的准备。

## 参考资料

- [DataFrames.jl Documentation](https://dataframes.juliadata.org/stable/) — The DataFrames.jl Developers (2024)
  在Julia中使用DataFrames.jl的权威指南，涵盖了其用于表格数据处理的所有功能。
- [CSV.jl Documentation](https://csv.juliadata.org/stable/) — The CSV.jl Developers (2024)
  Publisher: JuliaData
  CSV.jl包的官方文档，详细介绍了CSV文件的高效读写。
- [Hands-On Julia for Data Science: Learn how to analyze data, implement machine learning algorithms, and develop interactive dashboards with Julia](https://learning.oreilly.com/library/view/hands-on-julia-for/9781803248677/) — Paul Bouquin, Antoine Balley (2023)
  Publisher: Packt Publishing
  一本实用指南，涵盖Julia中的数据分析、机器学习和可视化，包括DataFrames.jl及其他数据科学包的使用。
- [The Julia Language Documentation](https://docs.julialang.org/en/v1/) — The Julia Project (2024)
  Julia编程语言的官方文档，提供了其功能、设计和包管理的基础知识。

---

[上一节](02-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%83%8C%E6%99%AF%E4%B8%8B%E7%9A%84%E7%B1%BB%E5%9E%8B%E7%B3%BB%E7%BB%9F%E4%B8%8E%E5%A4%9A%E9%87%8D%E5%88%86%E6%B4%BE.md) · [下一节](04-Julia%20%E6%95%B0%E5%80%BC%E8%AE%A1%E7%AE%97%EF%BC%9A%E6%95%B0%E7%BB%84%E4%B8%8E%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B0.md)

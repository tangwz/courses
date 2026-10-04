---
course: "julia-for-machine-learning"
chapter: "julia-data-manipulation-preparation"
lesson: "loading-saving-data-dataframes"
sourceId: 6847
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-2-julia-data-manipulation-preparation/loading-saving-data-dataframes"
title: "使用 DataFrames.jl 加载和保存数据"
description: "学习如何将各种文件格式（CSV、Excel）的数据读取到 DataFrames 中，并将处理后的数据保存回磁盘。"
order: 1
plots: []
sourceHash: "dd9670cad50dac628b3558a05935fb3cc2a68d89a47fe0e4fa896e1192842be0"
sourceCorrections: []
---

任何机器学习 (machine learning)项目的过程都从数据开始，高效地将数据导入 Julia 环境是第一个实际步骤。`DataFrames.jl` 是 Julia 处理表格数据的基础包，类似于 Python 中的 Pandas 或 R 中的数据框。常见任务包括将各种文件格式的数据加载到 Julia DataFrames 中以及将处理后的 DataFrames 保存回磁盘，这些是任何数据准备流程都不可或缺的技能。

### 将数据读取到 DataFrames 中

您会遇到的大多数数据集将存储在平面文件中，其中逗号分隔值 (CSV) 文件很常见。Julia 在 `CSV.jl` 包的帮助下，提供了高效的工具，可以直接将这些文件读取到 `DataFrame` 中。

#### 使用 CSV 文件

要开始使用，您首先需要在环境中安装并可用 `CSV` 和 `DataFrames` 包。如果您遵循了第 1 章的设置，您可能已经拥有它们。如果尚未安装，可以使用 Julia 的包管理器添加它们：

```julia
using Pkg
Pkg.add(["CSV", "DataFrames"])
```

安装后，您可以将它们引入当前的 Julia 会话：

```julia
using CSV
using DataFrames
```

读取 CSV 文件的主要函数是 `CSV.read`。其最基本的用法需要文件路径，并指定您希望输出为 `DataFrame`：

```julia
# 假设您的工作目录中有一个名为 'iris.csv' 的文件
# iris.csv 的内容：
# sepal_length,sepal_width,petal_length,petal_width,species
# 5.1,3.5,1.4,0.2,setosa
# 4.9,3.0,1.4,0.2,setosa
# ... (更多行)

df_iris = CSV.read("iris.csv", DataFrame)
```

这个简单的命令读取 `iris.csv` 文件，并将其内容加载到一个名为 `df_iris` 的 `DataFrame` 对象中。`CSV.read` 函数非常智能，通常会推断数据类型、分隔符以及是否存在标题行。然而，数据很少如此简单。您经常需要提供更具体的指令。

让我们考虑一个稍微复杂一点的场景。想象一个名为 `student_grades.csv` 的文件，内容如下：

```
student_id;name;subject;grade_percent;attendance_rate
S001;Alice;Math;85;0.95
S002;Bob;Science;N/A;0.88
S003;Charlie;Math;72;0.90
```

请注意分号分隔符和表示缺失分数的“N/A”字符串。以下是您可能如何读取它的方法：

```julia
# student_grades.csv 的内容（如上）
file_path = "student_grades.csv"

# 为使示例运行而创建的虚拟文件
open(file_path, "w") do f
    write(f, "student_id;name;subject;grade_percent;attendance_rate\n")
    write(f, "S001;Alice;Math;85;0.95\n")
    write(f, "S002;Bob;Science;N/A;0.88\n")
    write(f, "S003;Charlie;Math;72;0.90\n")
end

df_students = CSV.read(
    file_path,
    DataFrame;
    delim=';',
    missingstrings=["N/A", ""], # 将 "N/A" 和空字符串视作缺失值
    types=Dict(:grade_percent => Union{Missing, Float64}, :attendance_rate => Float64),
    header=true # 明确声明存在标题行
)

# 显示前几行以验证
println(first(df_students, 5))
```

> 这将输出：
>
> ```
> 3×5 DataFrame
>  Row │ student_id  name     subject  grade_percent  attendance_rate
>      │ String      String   String   Float64?       Float64
> ─────┼───────────────────────────────────────────────────────────────
>    1 │ S001        Alice    Math             85.0             0.95
>    2 │ S002        Bob      Science         missing           0.88
>    3 │ S003        Charlie  Math             72.0             0.9
> ```

让我们分解 `CSV.read` 的一些常见参数 (parameter)：

- `delim`：指定用于分隔值的字符。逗号的常见替代包括分号 (`;`)、制表符 (`\t`) 或空格。
- `header`：可以是表示标题行号的整数，用于多行标题的 `Range`，或布尔值（如果第一行是标题则为 `true`，否则为 `false`）。您还可以传递字符串数组以提供自定义列名。
- `missingstrings`：应解释为缺失值的字符串向量 (vector)。`CSV.jl` 会将这些字符串转换为 Julia 的 `missing` 值。默认情况下，它通常将空字段视为缺失值。
- `types`：允许您按名称或索引为每列或特定列指定数据类型。这对于确保数据以正确的格式读取很有用，特别是对于可能被模糊解释的列（例如，应该是字符串的数字，或确保列可以包含带有 `Union{Missing, YourType}` 的 `missing` 值）。
- `select`：要读取的列名（作为 `Symbol` 或 `String`）或索引的向量。对于仅加载部分列很有用。
- `drop`：类似于 `select`，但指定要排除的列。
- `datarow`：指定实际数据第一行的整数，如果标题或数据之前有介绍性行，则此参数很有用。
- `normalizenames`：如果为 `true`（默认值），则列名会被规范化为有效的 Julia 标识符（例如，“Column Name”变为 `Column_Name`）。

加载数据后，立即检查它是一个好的习惯：

```julia
# 假设 df_students 已如上所示加载

# 显示前 5 行
println("前 5 行：")
println(first(df_students, 5))

# 获取维度（行，列）
println("\n维度：", size(df_students))

# 获取列名
println("\n列名：", names(df_students))

# 获取列的数据类型
println("\n列类型：")
println(eltype.(eachcol(df_students)))

# 获取统计摘要（计数、平均值、最小值、最大值、缺失计数等）
println("\n统计摘要：")
println(describe(df_students))
```

这种初步检查有助于您验证数据是否按预期加载，并在进行数据清洗和转换之前，让您初步了解其结构和内容。

#### 读取其他文件格式

虽然 CSV 很常见，但您可能会遇到其他格式。

- **Excel 文件 (.xlsx, .xls)**：`XLSX.jl` 包很常用。它允许您从 Excel 工作簿中的特定工作表读取数据。

  ```julia
  using XLSX
  # 从 Excel 文件读取第一个工作表
  # xf = XLSX.readxlsx("my_data.xlsx")
  # sheet = xf[XLSX.sheetnames(xf)[1]]
  # df_excel = DataFrame(XLSX.gettable(sheet)...)

  # 或者直接将工作表读取到 DataFrame 中
  # df_excel_direct = DataFrame(XLSX.readtable("my_data.xlsx", "Sheet1")...)
  ```

  注意：`XLSX.readtable` 函数返回数据列和标题名称的元组，这些元组可以展开（`...`）到 `DataFrame` 构造函数中。
- **JSON 及其他格式**：对于 JSON，`JSON3.jl` 或 `JSON.jl` 是流行的选择。您通常会解析 JSON 结构，然后将相关部分（通常是对象数组）转换为 `DataFrame`。对于 Parquet (`Parquet.jl`)、Arrow (`Arrow.jl`) 等格式以及数据库连接（`JDBC.jl`、`ODBC.jl`、用于 PostgreSQL 的 `LibPQ.jl` 等），存在其他专用包。一般模式是使用特定包将数据读取到 Julia 结构中，然后，如有必要，将其转换为 `DataFrame`。

### 将 DataFrames 保存到磁盘

在处理和准备数据后，您通常需要将结果 `DataFrame` 保存回文件。同样，在处理 CSV 文件时，`CSV.jl` 是您的主要工具。

函数 `CSV.write` 用于此目的：

```julia
# 假设 df_processed 是您处理过的一个 DataFrame
# 例如，我们创建一个示例：
df_processed = DataFrame(
    ID = [1, 2, 3],
    Name = ["Processed_Alice", "Processed_Bob", "Processed_Charlie"],
    Score = [95.0, 88.5, 70.0]
)

CSV.write("processed_data.csv", df_processed)
```

这会将 `df_processed` 保存到当前工作目录中名为 `processed_data.csv` 的文件中。`CSV.write` 还接受几个可选参数 (parameter)：

- `delim`：指定要使用的分隔符（默认为 ','）。
- `header`：一个布尔值，指示是否将列名作为第一行写入（默认为 `true`）。
- `append`：如果为 `true`，则将 DataFrame 追加到现有文件。如果为 `false`（默认），则覆盖文件。
- `bom`：（字节顺序标记 (token)）如果为 `true`，则写入 BOM 字符，这有时有助于与 Excel 等软件对 UTF-8 编码文件的兼容性。
- `missingstring`：用于在输出文件中表示 `missing` 值的字符串（默认为空字符串）。

例如，使用制表符分隔符并将缺失值表示为“MISSING”来保存：

```julia
# 让我们为演示目的向 df_processed 添加一个缺失值
df_processed_with_missing = copy(df_processed)
df_processed_with_missing.Score[2] = missing

CSV.write(
    "processed_data_tab.tsv",
    df_processed_with_missing;
    delim='\t',
    missingstring="MISSING"
)
```

与读取类似，如果您需要保存为 Excel 等其他格式，您将使用相关包（例如，`XLSX.jl` 提供了 `XLSX.writetable` 和其他函数来构建和保存 Excel 文件）。

```julia
# 保存到 Excel 的示例（需要 XLSX.jl）
# using XLSX
# data_to_save = [
#     "ID" => df_processed.ID,
#     "Name" => df_processed.Name,
#     "Score" => df_processed.Score
# ]
# XLSX.writetable("processed_output.xlsx", data_to_save, sheetname="results", anchor_cell="A1")
```

熟练掌握数据的加载和保存是一项基本技能。通过 `DataFrames.jl` 和 `CSV.jl` 等辅助包，Julia 为这些初始的 I/O 操作提供了一个强大而灵活的环境，为接下来您将学习的数据清洗、转换和特征工程步骤奠定了基础。请务必记住在加载数据后检查它，以确保其符合您的预期，然后再进行更复杂的操作。

## 参考资料

- [DataFrames.jl Documentation](https://dataframes.juliadata.org/stable/) — The DataFrames.jl Contributors (2024)
  使用 DataFrames.jl 的指南，包括其设计原则、数据处理函数以及与其他 Julia 包的集成。
- [CSV.jl Documentation](https://csv.juliadata.org/stable/) — The JuliaData Contributors (2024)
  在 Julia 中高效读写 CSV 文件的官方资源，详细说明了各种解析和写入选项。
- [XLSX.jl Documentation](https://felipenoris.github.io/XLSX.jl/stable) — The JuliaData Contributors (2024)
  处理 Excel 文件的资源，包含如何从电子表格读取数据以及将 DataFrames 保存到 Excel 工作簿。

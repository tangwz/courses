# 实践：Julia 用于数据处理与基本算法

来源：[原文](https://apxml.com/zh/courses/julia-deep-learning/chapter-1-foundations-julia-ml/practice-julia-data-manipulation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

动手练习旨在巩固您在数据处理和简单算法组件实现方面的 Julia 技能。这些练习涵盖了对深度学习 (deep learning)必不可少的编程基础。您将使用常见的数据科学包，并应用数值计算技术。我们假定您已设置好 Julia 环境，并安装了 `DataFrames.jl`、`CSV.jl` 和 `Plots.jl` 等包。如果尚未设置，请查看“设置 Julia 深度学习环境”指南。

### 1. 加载和查看数据

深度学习 (deep learning)以及通用机器学习 (machine learning)都从数据开始。Julia 提供处理各种数据格式的优秀工具。我们将首先创建一个小型示例数据集，将其保存为 CSV（逗号分隔值）文件，然后使用 `CSV.jl` 和 `DataFrames.jl` 加载它。

让我们设想一个来自农业实验的数据集，它记录了基于日照和水分的植物生长情况。

首先，确保您拥有必要的包。如果您在 Julia REPL 中，可以添加它们：

```julia
using Pkg
Pkg.add("DataFrames")
Pkg.add("CSV")
Pkg.add("Plots") # 我们稍后会用到这个
```

现在，让我们将示例数据创建为字符串并写入文件。

```julia
using CSV
using DataFrames

# 将数据定义为多行字符串
csv_data = """
plant_id,sunlight_hours,water_ml,growth_cm
1,4.5,100,5.2
2,5.0,110,5.8
3,3.5,90,4.1
4,6.1,120,7.3
5,4.2,95,4.9
6,5.5,115,6.5
"""

# 将这些数据写入临时 CSV 文件
file_path = "plant_data.csv"
open(file_path, "w") do io
    write(io, csv_data)
end

# 使用 CSV.jl 将数据加载到 DataFrame 中
df = CSV.read(file_path, DataFrame)

# 显示前几行
println("DataFrame 的前 3 行：")
println(first(df, 3))
```

`CSV.read` 函数很直接。它接受文件路径和期望的接收器类型，在我们的例子中是 `DataFrame`。`DataFrames.jl` 提供了一种强大且灵活的方式来处理表格数据，类似于 Python 中的 Pandas 或 R 的数据帧。

让我们进一步查看一下我们的 `DataFrame`：

```julia
# 获取 DataFrame 的统计概览
println("\n统计概览：")
println(describe(df))

# 获取列名
println("\n列名：")
println(names(df))

# 选择特定列（例如 growth_cm）
growth_data = df.growth_cm
println("\n生长数据（前 5 个值）：")
println(first(growth_data, 5))

# 筛选行：日照超过 5 小时的植物
high_sunlight_plants = filter(row -> row.sunlight_hours > 5, df)
println("\n日照超过 5 小时的植物：")
println(high_sunlight_plants)
```

`describe` 函数可让您快速概览每列，包括数值列的均值、最小值、最大值和其他统计量。您可以使用点表示法（例如 `df.sunlight_hours`）或字符串索引（例如 `df[!, "sunlight_hours"]`）访问列。`filter` 函数在根据条件选择数据子集时很灵活。

### 2. 数据处理和转换

通常，原始数据并非处于分析或模型训练的理想状态。您可能需要创建新特征或规范化现有特征。

#### 创建新特征

让我们创建一个新特征，比如 `water_per_sunlight_hour`，这可以让我们了解相对于日照的浇水效率。

```julia
# 创建新列：water_ml / sunlight_hours
# 使用 . (点) 进行广播，实现元素级操作
df.water_per_sunlight_hour = df.water_ml ./ df.sunlight_hours

println("\n带有新特征的 DataFrame：")
println(first(df, 3))
```

Julia 的广播语法（在运算符前使用点 `.`，例如 `./` 或函数调用 `foo.(...)`）将操作逐元素应用于数组或列。这在 Julia 中是高效且符合惯用法的。

#### 规范化列

规范化是常见的预处理步骤。让我们对 `growth_cm` 列应用最小-最大缩放。公式为：


$$
X_{缩放后} = \frac{X - X_{最小值}}{X_{最大值} - X_{最小值}}
$$


这会将数据缩放到一个范围，通常是 [0, 1]。

```julia
# 对 'growth_cm' 进行最小-最大规范化
growth_col = df.growth_cm
min_growth = minimum(growth_col)
max_growth = maximum(growth_col)

# 使用广播应用公式
df.growth_cm_normalized = (growth_col .- min_growth) ./ (max_growth - min_growth)

println("\n带有规范化生长数据的 DataFrame：")
println(select(df, :plant_id, :growth_cm, :growth_cm_normalized)) # 显示相关列
```

在这里，`minimum` 和 `maximum` 是作用于数组的标准 Julia 函数。算术运算 `-` 和 `./` 被广播应用于 `growth_col` 数组。

### 3. 基本算法片段：一个简单的线性预测

虽然本章是基本内容，但我们来实施一个非常简单的预测计算。设想我们有一个假设，即植物生长可以通过日照小时数和水量进行线性组合预测：


$$
\text{预测生长} = w_1 \times \text{日照小时数} + w_2 \times \text{水量毫升} + b
$$


$w_1, w_2$ 是权重 (weight)，$b$ 是偏置 (bias)项。在完整的机器学习 (machine learning)流程中，我们会*学习*这些参数 (parameter)。现在，我们假定有一些预定义（可能任意）的值并计算预测结果。

```julia
# 为我们的简单线性模型定义一个函数
function predict_growth(sunlight, water, w1, w2, b)
    return w1 * sunlight + w2 * water + b
end

# 假设一些权重和偏置
w1_hypothetical = 0.8
w2_hypothetical = 0.05
b_hypothetical = -1.0

# 将此预测应用于 DataFrame 的每一行
# 我们可以用这些预测创建一个新列
# 使用匿名函数和按行迭代（对于大数据来说不常见，但在这里很清楚）
df.predicted_growth = [predict_growth(row.sunlight_hours, row.water_ml, w1_hypothetical, w2_hypothetical, b_hypothetical) for row in eachrow(df)]

# 或者，使用广播实现更“Julia”风格的向量化方法：
# df.predicted_growth = predict_growth.(df.sunlight_hours, df.water_ml, w1_hypothetical, w2_hypothetical, b_hypothetical)
# 注意：对于上述向量化调用，predict_growth 需要正确处理带有数组输入的标量 w1, w2, b，或者在其中使用广播。
# 对于初学者应用按行逻辑时，`eachrow` 方法通常更清晰。

println("\n带有预测生长数据的 DataFrame：")
println(select(df, :plant_id, :growth_cm, :predicted_growth))
```

这个练习显示了如何在 Julia 中定义函数并将其应用于数据。计算本身是线性模型和神经网络 (neural network)中线性层的重要组成部分。稍后，您将了解到自动微分如何帮助找到 `w1`、`w2` 和 `b` 的最佳值。

### 4. 数据可视化

一图胜千言。`Plots.jl` 是 Julia 中一个受欢迎的绘图元包，它允许您使用各种绘图后端。让我们创建一个简单的散点图，以可视化实际 `growth_cm` 与我们的 `predicted_growth` 之间的关系。

```julia
using Plots
gr() # 使用 Plots.jl 的 GR 后端，您也可以选择其他后端，如 pyplot() 或 plotlyjs()

# 实际生长与预测生长的散点图
scatter_plot = scatter(df.growth_cm, df.predicted_growth,
                    xlabel="实际生长 (cm)",
                    ylabel="预测生长 (cm)",
                    title="实际生长与预测生长",
                    legend=false,
                    aspect_ratio=:equal, # 使 x 轴和 y 轴的刻度在视觉上具有可比性
                    color=:blue,
                    markersize=5)

# 添加一条 y=x 的参考线（完美预测）
plot!(df.growth_cm, df.growth_cm, line=:dash, color=:red)

# 在 VS Code 或 Jupyter 等环境中显示图表：
# display(scatter_plot)
# 或保存到文件：
# savefig(scatter_plot, "actual_vs_predicted_growth.png")

# 对于本课程网页，以下是此类图表的表示：
println("\n正在生成绘图数据（用于网页显示示例）...")
```



[交互图表：实际生长与预测生长](https://apxml.com/zh/courses/julia-deep-learning/chapter-1-foundations-julia-ml/practice-julia-data-manipulation#plot-xdhdfr)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "实际生长与预测生长",
    "xaxis": {
      "title": "实际生长 (cm)"
    },
    "yaxis": {
      "title": "预测生长 (cm)"
    },
    "showlegend": false,
    "aspectratio": {
      "x": 1,
      "y": 1
    }
  },
  "data": [
    {
      "type": "scatter",
      "mode": "markers",
      "x": [
        5.2,
        5.8,
        4.1,
        7.3,
        4.9,
        6.5
      ],
      "y": [
        3.1,
        3.9,
        2.05,
        5.38,
        3.01,
        4.650000000000001
      ],
      "marker": {
        "color": "#228be6",
        "size": 8
      }
    },
    {
      "type": "scatter",
      "mode": "lines",
      "x": [
        4.1,
        7.3
      ],
      "y": [
        4.1,
        7.3
      ],
      "line": {
        "dash": "dash",
        "color": "#f03e3e"
      }
    }
  ]
}
```

</details>



> 实际植物生长与简单线性模型（带有任意权重 (weight)）预测生长之间的关系。虚线代表完美预测。

视觉检查是任何数据分析或建模任务中的重要步骤。它能帮助您了解关系、识别异常值并判断模型性能。

### 5. 保存处理后的数据

在处理完 `DataFrame` 后，您可能希望保存结果。`CSV.jl` 也可以将 `DataFrame` 写入 CSV 文件。

```julia
# 定义输出文件路径
output_file_path = "plant_data_processed.csv"

# 将 DataFrame 写入新的 CSV 文件
CSV.write(output_file_path, df)

println("\n处理后的 DataFrame 已保存到：$output_file_path")

# 清理创建的文件（可选，为本示例的整洁性）
# rm(file_path)
# rm(output_file_path)
```

本次实践课带您完成了 Julia 中重要的数据处理任务：加载、查看、转换、执行基本计算、可视化和保存数据。这些技能是您在后续章节中构建和训练更复杂深度学习 (deep learning)模型的根本。随着学习的深入，您将看到 Julia 的性能和富有表达力的语法如何使这些操作既高效又令人愉悦。下一章将介绍 `Flux.jl`，您将从那里开始构建神经网络 (neural network)，并运用到许多这些基本的数据处理能力。

## 参考资料

- [The Julia Language Documentation](https://docs.julialang.org/en/v1/) — The Julia Language (2024)
  朱莉娅核心语法、类型、函数和高效数组操作（如广播）的重要参考资料。
- [DataFrames.jl Documentation](https://dataframes.juliadata.org/stable/) — The DataFrames.jl Developers (2024)
  使用DataFrames在Julia中进行表格数据操作、检查和转换的指南。
- [CSV.jl Documentation](https://csv.juliadata.org/stable/) — The JuliaData Community (2024)
  关于加载和保存CSV文件的详细信息，以及读写结构化数据的接口。
- [Plots.jl Documentation](https://docs.juliaplots.org/latest/) — Thomas Breloff (2024)
  在Julia中创建各种图表和可视化的参考资料，涵盖不同的后端和图表类型。

---

[上一节](07-%E9%85%8D%E7%BD%AE%E6%82%A8%E7%9A%84Julia%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%8E%AF%E5%A2%83.md) · [下一节](../02-Flux.jl%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%85%A5%E9%97%A8/01-Flux.jl%EF%BC%9A%E8%AE%BE%E8%AE%A1%E5%8E%9F%E5%88%99%E4%B8%8E%E6%9E%B6%E6%9E%84.md)

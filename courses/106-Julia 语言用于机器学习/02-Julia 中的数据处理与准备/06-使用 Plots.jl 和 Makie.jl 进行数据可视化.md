# 使用 Plots.jl 和 Makie.jl 进行数据可视化

来源：[原文](https://apxml.com/zh/courses/julia-for-machine-learning/chapter-2-julia-data-manipulation-preparation/julia-data-visualization)

[返回章节目录](README.md) · [返回课程目录](../README.md)

数据可视化是数据准备流程中必不可少的一步。可视化能够直观地帮助理解数据分布、变量间的关系、异常值的存在以及预处理步骤的效果。Julia 提供了功能强大且灵活的绘图库，其中 `Plots.jl` 和 `Makie.jl` 是创建各种静态和交互式可视化的主要选择。

### 使用 Plots.jl 查看数据

`Plots.jl` 是 Julia 中一个多功能绘图元包。它作为各种绘图后端（如 GR、Plotly、PyPlot）的统一接口，允许您轻松地在它们之间切换。这使其成为一个出色的通用工具，用于快速查看数据和生成出版质量的图表。

开始使用 `Plots.jl`，通常需要安装它和一个后端，例如 GR，这是一个快速且受欢迎的选择：

```julia
# 如果尚未安装，请在 Julia REPL 中运行此代码
# using Pkg
# Pkg.add("Plots")
# Pkg.add("GR")

using Plots, DataFrames
gr() # 将 GR 设置为后端

# 假设 'df' 是之前步骤中的 DataFrame
# 例如：
df = DataFrame(
    Age = rand(25:65, 100),
    Income = rand(30000:120000, 100),
    Category = rand(["A", "B", "C"], 100)
)
```

#### 直方图和密度图

直方图和密度图是理解单个连续变量分布的基础。它们有助于识别偏度、众数和潜在异常值，这可以为数据转换或异常值处理提供决策依据。

```julia
histogram(df.Age,
          bins=10,
          label="年龄分布",
          xlabel="年龄",
          ylabel="频率",
          title="客户年龄分布",
          fillalpha=0.7,
          linecolor=:auto)
```

此代码为 `Age` 列生成一个直方图。调整 `bins` 参数 (parameter)可以帮助呈现分布的不同方面。

以下是这种分布可能的样子，直接表示：



[交互图表：样本人群年龄分布](https://apxml.com/zh/courses/julia-for-machine-learning/chapter-2-julia-data-manipulation-preparation/julia-data-visualization#plot-xy1285)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "histogram",
      "x": [
        25,
        30,
        35,
        28,
        42,
        50,
        33,
        37,
        45,
        22,
        29,
        31,
        38,
        40,
        48,
        55,
        26,
        32,
        36,
        41,
        60,
        28,
        33,
        47,
        52
      ],
      "xbins": {
        "start": 20,
        "end": 65,
        "size": 5
      },
      "marker": {
        "color": "#228be6"
      },
      "name": "年龄"
    }
  ],
  "layout": {
    "title": "样本人群年龄分布",
    "xaxis": {
      "title": "年龄"
    },
    "yaxis": {
      "title": "频率"
    },
    "bargap": 0.1,
    "paper_bgcolor": "white",
    "plot_bgcolor": "#e9ecef"
  }
}
```

</details>



> 年龄分布。这样的图可以快速显示年龄的分布和集中情况。

密度图提供了直方图的平滑版本：

```julia
density(df.Income,
        label="收入密度",
        xlabel="收入",
        ylabel="密度",
        title="收入分布",
        linewidth=2,
        fill=(0, 0.3, :blue))
```

#### 散点图

散点图非常适合可视化两个连续变量之间的关系。它们可以显示相关性、聚类和模式，这可能暗示特征交互或指导模型选择。

```julia
scatter(df.Age, df.Income,
        xlabel="年龄",
        ylabel="收入",
        title="收入对比年龄",
        label="客户",
        markersize=5,
        markerstrokewidth=0,
        alpha=0.6)
```

这个图将显示您的数据集中年龄和收入之间是否存在可辨别的趋势。

#### 箱线图

箱线图可以有效总结数值变量的分布，并可按类别变量进行分组。它们清晰地显示中位数、四分位数和潜在异常值。

```julia
boxplot(df.Category, df.Income,
        xlabel="类别",
        ylabel="收入",
        title="按类别划分的收入分布",
        legend=false,
        linewidth=2)
```

这有助于比较不同类别间的收入分布，突出显示各组内的差异和异常值。

### 使用 Makie.jl 进行交互式和高级可视化

`Makie.jl` 是 Julia 中另一个功能强大的绘图生态系统，专为高性能和交互式可视化而设计。它在大数据集、3D 绘图以及创建复杂、出版质量的图表方面表现出色。`Makie.jl` 有多个后端，例如用于交互式桌面窗口的 `GLMakie.jl`、用于静态矢量图形（SVG、PDF）和栅格图像（PNG）的 `CairoMakie.jl`，以及用于浏览器中基于 WebGL 的图表的 `WGLMakie.jl`。

使用 `CairoMakie.jl`（用于静态输出）的简单散点图可能如下所示：

```julia
# 如果尚未安装，请在 Julia REPL 中运行此代码
# using Pkg
# Pkg.add("CairoMakie") # 或使用 GLMakie 进行交互式绘图

using CairoMakie, DataFrames

# 重复使用 DataFrame 'df'
f = Figure(resolution = (600, 400))
ax = Axis(f[1, 1],
          xlabel="年龄",
          ylabel="收入",
          title="使用 Makie 的收入对比年龄")

scatter!(ax, df.Age, df.Income,
         markersize=10,
         color=(:teal, 0.7), # 使用调色板中的一种颜色并带透明度
         strokecolor=:black,
         strokewidth=1)

# 保存图表：
# save("income_vs_age_makie.png", f)
# 在支持的环境中显示（如 VS Code Julia 扩展绘图窗格）：
# f
```

`Makie.jl` 提供了一种基于场景图的方法，对绘图元素进行精细控制。虽然 `Plots.jl` 通常用于标准查看图更快速，但当您需要高度定制或交互式输出，或者处理其性能优势得以体现的超大型数据集时，`Makie.jl` 变得特别有用。

### 在数据准备工作流中应用可视化

可视化在各种数据准备任务中发挥直接作用：

1. **理解特征分布：** 在应用缩放或归一化 (normalization)等转换之前，绘制数值特征的直方图或密度图。例如，高度偏斜的分布可能受益于对数转换。对转换前后进行可视化可确认其效果。
2. **检测异常和离群点：** 箱线图是异常值检测的标准工具。散点图也可以显示异常数据点，它们明显偏离一般模式。例如，绘制 `age` 对 `income` 的图可能显示某个个体在同年龄组中收入异常高，这会促使进一步检查。
3. **评估缺失数据：** 尽管在数据清洗中已讨论，但可视化缺失情况可能富有启发性。显示跨特征和样本缺失值的热图可以显示模式，例如某个特征大部分为空或某些样本有许多缺失条目。`Plots.jl` 可以创建热图：

   ```julia
   # 示例：创建一个指示缺失值的矩阵
   # 假设 'raw_data' 是一个可能包含缺失值的 DataFrame
   # M = ismissing.(Matrix(raw_data))
   # heatmap(M, title="缺失数据模式", xlabel="特征", ylabel="样本")
   ```
4. **指导特征工程：** 可视化变量间的关系可以启发创建新特征。如果两个变量 `X` 和 `Y` 的散点图显示出明显的非线性关系，您可能会考虑创建多项式特征（例如，$X^2$, $Y^2$）或交互项（例如，$X \times Y$）。
5. **评估分类数据：** 条形图有助于理解特征内不同类别的频率。这对于决定编码策略或识别不平衡类别很重要。

   ```julia
   # 使用 Plots.jl 和带有 'Category' 列的 DataFrame 'df'
   category_counts = combine(groupby(df, :Category), nrow => :Count)
   bar(category_counts.Category, category_counts.Count,
       xlabel="类别",
       ylabel="频率",
       title="类别分布",
       legend=false)
   ```

### 选择合适的工具和图表

- **对于快速、日常的数据查看图：** `Plots.jl` 凭借其简单的语法和多后端支持，通常是最方便的选择。
- **对于高性能需求、大型数据集、交互性或复杂的二维/三维场景：** `Makie.jl` 是一个强有力的选择。
- **选择图表类型：**
  - **单个变量的分布：** 直方图、密度图、箱线图。
  - **两个连续变量之间的关系：** 散点图。
  - **连续变量与类别变量之间的关系：** 箱线图（每个类别一个）、小提琴图。
  - **两个类别变量之间的关系：** 堆叠条形图、分组条形图或计数热图。

有效的数据可视化不仅仅是生成图表；它在于提出正确的问题并选择能够清晰回答它们的可视化方式。始终标记 (token)轴，提供标题，并在必要时使用图例。目标是获得理解，从而实现更好的数据准备，并最终得到更有效的机器学习 (machine learning)模型。在处理数据集时，将可视化视为分析中的迭代伙伴，帮助您在每一步优化数据。

## 参考资料

- [Plots.jl Documentation](https://docs.juliaplots.org/latest/) — Thomas Breloff (Current)
  了解 Plots.jl 元包在 Julia 中进行数据可视化的功能、后端和语法的必备资源。
- [Makie.jl Documentation](https://doi.org/10.21105/joss.03349) — Simon Danisch, Julius Krumbiegel (2021)
  Journal: Journal of Open Source Software; Publisher: The Open Journal; Volume: 6; Pages: 3349; DOI: [10.21105/joss.03349](https://doi.org/10.21105/joss.03349)
  提供 Makie.jl 高性能、交互式和 3D 绘图功能的全面信息，包括其场景图方法和各种后端。
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media
  涵盖有效数据可视化原理和最佳实践的基础性文本，对于设计图表至关重要，无论使用何种特定工具。
- [Julia for Data Analysis](https://www.manning.com/books/julia-for-data-analysis) — Bogumił Kamiński (2022)
  Publisher: Manning Publications; DOI: [10.1007/978-3-031-28511-9](https://doi.org/10.1007/978-3-031-28511-9)
  提供使用 Julia 进行数据分析的实用指导，包括在数据科学工作流程中应用 Plots.jl 的可视化技术。

---

[上一节](05-%E5%9C%A8%20Julia%20%E4%B8%AD%E5%BA%94%E7%94%A8%E7%89%B9%E5%BE%81%E5%B7%A5%E7%A8%8B.md) · [下一节](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%95%B0%E6%8D%AE%E6%B8%85%E6%B4%97%E4%B8%8E%E7%89%B9%E5%BE%81%E5%88%9B%E5%BB%BA.md)

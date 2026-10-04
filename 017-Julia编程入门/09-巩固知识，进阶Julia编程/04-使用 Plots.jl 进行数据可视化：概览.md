# 使用 Plots.jl 进行数据可视化：概览

来源：[原文](https://apxml.com/zh/courses/getting-started-julia-programming/chapter-9-consolidating-knowledge-advancing-julia/visualizing-data-plots-jl-overview)

[返回章节目录](README.md) · [返回课程目录](../README.md)

数据可视化是理解信息的一种常见做法。以图形形式查看数据能比直接查看原始数字或表格，更有效地帮助您理解模式、趋势和异常值。`Plots.jl` 是 Julia 生态系统中为此目的最常用且灵活的工具之一。这里将介绍如何使用此包来开始创建可视化。

`Plots.jl` 本身不是一个绘图库，而是一个绘图元包或应用程序编程接口（API）。它为许多不同的底层绘图库（称为“后端”）提供了一个通用接口。这意味着您可以学习一套命令，然后选择由哪个后端实际生成图表，例如使用 `GR` 以便快速生成静态图表，或使用 `PlotlyJS` 来制作交互式基于网页的图表。这种灵活性是一个显著的优点，它让您可以在绘图引擎之间切换，而无需大幅重写您的绘图代码。

### Plots.jl 入门

在创建任何图表之前，如果您的 Julia 环境中尚未添加 `Plots.jl` 包，则需要先添加它。您可以使用 Julia 内置的包管理器 Pkg 来完成此操作。打开 Julia REPL 并输入：

```julia
import Pkg
Pkg.add("Plots")
```

安装完成后，您可以使用 `using` 关键字将包加载到当前的 Julia 会话中：

```julia
using Plots
```

默认情况下，`Plots.jl` 会尝试使用合适的后端，通常是 GR，这是一个快速轻量级的选项，适用于快速可视化。对于 GR 的基本使用，您通常不需要单独安装后端，因为 `Plots.jl` 通常会处理它。

### 创建您的第一个图表

让我们从一个简单的折线图开始。假设您有一些表示曲线上点的数据，例如 $x$ 范围从 -5 到 5 时 $y = x^2$ 的值。

首先，我们生成数据：

```julia
x = -5:0.5:5 # 创建一个从 -5 到 5，步长为 0.5 的数字范围
y = x.^2     # 将 x 的每个元素平方。注意点运算符表示元素级操作。
```

现在，绘图很简单：

```julia
plot(x, y)
```

当您执行此命令时，应该会出现一个绘图窗口，显示一个抛物线。这个基本的 `plot` 命令功能相当强。如果您提供两个向量 (vector)，`Plots.jl` 会假定第一个用于 x 轴，第二个用于 y 轴，并且默认情况下，它会创建一个折线图。

### 美化您的图表

原始的图表通常不够用。您会希望添加标题、标记 (token)坐标轴，如果您绘制多个系列，可能还会添加图例。`Plots.jl` 允许您使用各种关键字参数 (parameter)或通过调用“修改”函数（以 `!` 结尾的函数）来完成此操作。

让我们美化之前的图表：

```julia
plot(x, y,
    label="y = x^2",
    linewidth=2,
    color=:blue
)
title!("我的第一个 Julia 图表")
xlabel!("X轴值")
ylabel!("Y轴值 (x 的平方)")
```

这里：

- `label` 为我们的数据系列分配一个名称，该名称将用于图例。
- `linewidth` 调整线条的粗细。
- `color` 设置线条的颜色。`Plots.jl` 能够识别许多颜色名称作为符号（例如，`:blue`、`:red`、`:green`）。
- `title!`、`xlabel!` 和 `ylabel!` 是修改当前图表以分别添加标题和坐标轴标签的函数。

### 了解不同图表类型

`Plots.jl` 支持多种图表类型。以下是一些常见的：

- **散点图**：适用于展示两个变量之间的关系，其中单个数据点很重要。使用 `scatter(x, y)`。
- **条形图**：适用于比较不同类别之间的数量。使用 `bar(categories, heights)`。
- **直方图**：非常适合显示单个变量的分布情况。使用 `histogram(data)`。

让我们尝试一个简单的散点图。

```julia
# 生成一些略带噪音的数据
x_scatter = 1:10
y_scatter = x_scatter .+ randn(10) # y = x + 一些随机噪音

scatter(x_scatter, y_scatter,
    label="带噪音数据点",
    color=:red,
    markershape=:circle,
    title="简单散点图",
    xlabel="X 数据",
    ylabel="Y 数据"
)
```

这将为每个数据点创建一个带有独立圆形标记 (token)的图表。`markershape` 参数 (parameter)允许您控制这些标记的外观。

### 简要查看后端

如前所述，`Plots.jl` 使用后端来渲染图表。虽然默认（通常是 GR）对许多目的来说已经很棒，但您可能希望使用不同的后端，例如创建交互式图表。`PlotlyJS` 是为此目的而常用的后端。

要使用特定的后端，您通常会在绘制图表之前将其名称作为函数调用，例如 `plotlyjs()`。`Plots.jl` 随后将尝试在会话中为后续图表使用该后端。您可能需要先添加后端包（例如，`Pkg.add("PlotlyJS")`）。

```julia
# Pkg.add("PlotlyJS") # 如果尚未添加
using Plots
plotlyjs() # 切换到 PlotlyJS 后端

# 条形图的示例数据
categories = ["A", "B", "C", "D"]
heights = [10, 24, 15, 7]

bar(categories, heights,
    title="示例条形图 (PlotlyJS)",
    label="数值",
    color=:teal
)
```

如果您执行此操作，您可能会在浏览器或 IDE 的绘图面板中得到一个交互式图表，您可以在其中将鼠标悬停在条形图上以查看值、缩放和平移。

下面是一个简单条形图可能的样子，以可由 PlotlyJS 等后端生成的形式呈现。



[交互图表：类别比较](https://apxml.com/zh/courses/getting-started-julia-programming/chapter-9-consolidating-knowledge-advancing-julia/visualizing-data-plots-jl-overview#plot-1ps55hz)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "类别比较"
    },
    "xaxis": {
      "title": "类别"
    },
    "yaxis": {
      "title": "数量"
    },
    "font": {
      "family": "Arial, sans-serif"
    },
    "paper_bgcolor": "#f8f9fa"
  },
  "data": [
    {
      "type": "bar",
      "x": [
        "苹果",
        "香蕉",
        "樱桃",
        "枣"
      ],
      "y": [
        20,
        14,
        23,
        10
      ],
      "marker": {
        "color": [
          "#ff6b6b",
          "#fcc419",
          "#748ffc",
          "#51cf66"
        ]
      },
      "name": "水果数量"
    }
  ]
}
```

</details>



> 一个条形图，比较四个不同类别的数量。

### 后续方向

这只是对 `Plots.jl` 的简要介绍。它提供了丰富的功能，包括：

- 在同一图表上绘制多个系列。
- 创建子图（一个图中包含多个图表的布局）。
- 对颜色、标记 (token)、线条样式、注释等进行广泛的自定义选项。
- 针对统计、金融和其他方面/专业类别的专用图表类型。
- 使用 `savefig("filename.png")` 命令将图表保存为各种格式（PNG、PDF、SVG 等）。

`Plots.jl` 的官方文档是了解其全部能力的极佳来源。当您进行本课程末尾建议的小项目时，请考虑可视化如何帮助您理解数据或呈现结果。能够快速绘制数据在任何编程或数据分析工作中都是一项宝贵的技能。

## 参考资料

- [Plots.jl Documentation](https://docs.juliaplots.org/latest/) — Thomas Breloff (2023)
  Plots.jl元包的综合指南，详细介绍了安装、各种后端、绘图函数以及Julia数据可视化的广泛自定义选项。
- [Julia for Data Analysis](https://www.manning.com/books/julia-for-data-analysis) — Bogumił Kamiński (2023)
  Publisher: Manning Publications; Pages: 472
  一本将Julia应用于数据处理、分析和可视化任务的实用指南，其中包含使用Plots.jl创建各种图表的详细示例。
- [Storytelling with Data: A Data Visualization Guide for Business Professionals](https://www.wiley.com/en-us/Storytelling+with+Data%3A+A+Data+Visualization+Guide+for+Business+Professionals-p-9781119002253) — Cole Nussbaumer Knaflic (2015)
  Publisher: John Wiley & Sons; Pages: 288
  一本广受好评的著作，介绍了创建有效且引人入胜的数据可视化的基本原则和最佳实践，对于理解绘图背后的‘为什么’至关重要。

---

[上一节](03-%E5%88%9D%E8%AF%86%20DataFrames.jl%EF%BC%9A%E8%A1%A8%E6%A0%BC%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86.md) · [下一节](05-Julia%20%E8%BF%9B%E9%98%B6%E5%AD%A6%E4%B9%A0%E6%8C%87%E5%BC%95.md)

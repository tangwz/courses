---
course: "getting-started-julia-programming"
chapter: "modules-packages-project-organization"
lesson: "incorporating-external-libraries-projects"
sourceId: 6807
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-6-modules-packages-project-organization/incorporating-external-libraries-projects"
title: "在项目中集成外部库"
description: "如何在 Julia 脚本和模块中引用（`using` 或 `import`）已安装的外部包。"
order: 6
plots: []
sourceHash: "8b4659314892a36445617e13ec86645382ca7547eecfa281f78064665163ac6a"
sourceCorrections: []
---

外部包扩展了 Julia 项目的功能。为了在 Julia 脚本或模块中利用这些包的功能，需要通过特定方法使其可用。这提供了一种途径，让你能够使用 Julia 社区开发的一系列工具，从而执行数据分析、绘图或网页开发等特定任务，而无需从头编写所有代码。实现这一目标的主要方法是通过 `using` 和 `import` 语句。

### 将包代码引入作用域

可以将每个 Julia 包视为一个独立的工具箱，组织在其自己的模块中。Julia 中的模块提供了一个独立的命名空间，这意味着在一个模块中定义的函数、类型和变量的名称不会与其他模块或你自己的代码中的名称冲突，除非你明确决定合并它们。要使用包工具箱中的工具，你首先需要告诉 Julia 你打算访问其内容。

### `using` 语句：轻松访问导出内容

使包功能可用的最常用方法是使用 `using` 语句。当你编写 `using PackageName` 时，Julia 会查找 `PackageName` 模块（Pkg 已经帮助定位并提供了它），并将其所有*导出*的名称引入到你当前的作用域中。导出的名称是包作者指定供公共使用的函数、类型和常量。

让我们通过一个例子来呈现这一点。Julia 自带一个名为 `Statistics` 的标准库，它提供 `mean`、`median` 和 `std` 等函数。虽然它是一个标准库（这意味着你不需要显式地 `Pkg.add` 它），但使用其内容的方式与使用你添加的任何外部包的方式相同。

```julia
# 使用 Statistics 模块中的函数
using Statistics

data_points = [15, 25, 35, 45, 55]
average_value = mean(data_points) # 'mean' 现在可以直接使用
println("平均值是: $average_value")

median_value = median(data_points) # 'median' 也可以直接使用
println("中位数是: $median_value")
```

在这段代码中，在 `using Statistics` 之后，我们可以直接调用 `mean()` 和 `median()`，就像它们定义在我们自己的脚本中一样。这很方便，因为它减少了输入。

`using` 的主要优点是其直接性。然而，如果你的代码，或者你正在 `using` 的另一个包，定义了与 `Statistics` 导出的名称相同的函数或变量，则可能会出现问题。例如，如果你有自己的名为 `mean` 的函数，Julia 可能会混淆要使用哪一个。这被称为名称冲突。

### `import` 语句：限定访问

`using` 的另一种选择是 `import` 语句。当你编写 `import PackageName` 时，Julia 仍然会使包可用，但它并*不*将其所有导出的名称直接引入当前作用域。相反，它只引入包的模块名本身。要访问包中的任何函数或类型，你必须使用包名进行限定，例如 `PackageName.functionName`。

以下是如何使用 `import` 来使用 `Statistics`：

```julia
import Statistics # 只将 'Statistics' 模块名引入作用域

data_points = [15, 25, 35, 45, 55]

# 要使用函数，必须在其前面加上 'Statistics.'
average_value = Statistics.mean(data_points)
median_value = Statistics.median(data_points)

println("平均值（通过 import）是: $average_value")
println("中位数（通过 import）是: $median_value")
```

`import PackageName` 的主要优点是它完全避免了名称冲突。由于你总是编写 `Statistics.mean`，因此无论其他地方是否存在其他 `mean` 函数，你打算调用哪个 `mean` 函数都是明确的。缺点是你的代码会变得有点冗长。

### 选择性 `import`：一种折衷方案

Julia 提供了一种方式，可以获得 `using` 的部分简洁性，同时保留 `import` 的部分明确性。你可以将包中的特定名称直接导入到你的作用域中，而其他名称仍然需要限定。这通过 `import PackageName: name1, name2, ...` 来完成。

假设你经常使用 `Statistics` 中的 `mean` 和 `std`，但 `median` 使用频率较低。你可以这样做：

```julia
import Statistics: mean, std # 只直接导入 mean 和 std

data_points = [15, 25, 35, 45, 55]

average_value = mean(data_points)   # 'mean' 可以直接使用
std_dev = std(data_points)        # 'std' 可以直接使用

# 如果 'median' 未被选择性导入，则需要使用限定访问：
# (首先确保模块 'Statistics' 本身是已知的，例如通过 'import Statistics' 或上述的选择性导入)
median_value = Statistics.median(data_points) 

println("平均值: $average_value, 标准差: $std_dev, 中位数: $median_value")
```

在此场景中，`mean` 和 `std` 可以直接调用，但 `Statistics` 中的任何其他函数（例如本例中的 `median`，假设它没有在冒号后列出）仍然需要 `Statistics.` 前缀。这种方法可以是一个很好的折衷方案，它减少了常用函数的输入量，同时使你的主命名空间更整洁。

你可以将这些机制想象成邀请专家到你的工作室：

- `using PackageName`: 专家 (PackageName) 抵达，他们所有常用工具（导出的名称）都陈列在你的主工作台上，供立即使用。
- `import PackageName`: 专家 (PackageName) 抵达，但将他们的工具放在自己的工具箱里。你必须询问：“PackageName，我可以使用你的锤子吗？”（例如，`PackageName.hammer()`）。
- `import PackageName: tool1, tool2`: 专家只带来你要求的特定工具（`tool1`，`tool2`），并将它们放在你的主工作台上。

### `using` 和 `import` 的放置位置

按照惯例，`using` 和 `import` 语句应放在你的 Julia (`.jl`) 文件的顶部，或者如果你正在编写自己的模块，则放在模块定义的开头。这使得包的内容可供该文件或模块中的所有后续代码使用。

```julia
# 在你的 .jl 文件的顶部
using Statistics
using DataFrames # 假设 DataFrames.jl 已添加到你的项目

# ... 其余代码 ...
```

### Julia 如何查找你的包

当你编写 `using SomePackage` 或 `import SomePackage` 时，Julia 需要找到 `SomePackage` 的代码。这就是你在前面章节中使用 Pkg（Julia 的包管理器）所做工作发挥作用的地方。Pkg 为你的项目环境维护 `Project.toml` 和 `Manifest.toml` 文件。这些文件记录了你的项目依赖哪些包（以及版本）。Julia 会查阅这些信息来找到并加载正确的代码。因此，Pkg 处理“使其可用”的部分，而 `using`/`import` 处理“使其在当前代码中可访问”的部分。

### 如果 Julia 找不到包怎么办？

如果你尝试 `using` 或 `import` 一个 Julia 在当前项目环境（或 Julia 默认可用包集合）中找不到的包，你会遇到一个错误，通常是 `ArgumentError`。它可能看起来像这样：

```
ERROR: ArgumentError: Package PackageName not found in current path.
- Run `import Pkg; Pkg.add("PackageName")` to install the PackageName package.
```

这条消息很有帮助。它通常表明两件事之一：

1. 包未安装在当前活动的 Julia 环境中。
2. 你可能拼错了包名。

正如错误消息所建议的，解决方法是将包添加到你当前项目的环境中。你可以在 Julia REPL 中通过进入包管理模式（输入 `]`）然后 `add PackageName` 来完成，或者直接在 Julia 脚本或 REPL 中运行 `import Pkg; Pkg.add("PackageName")`。

### 选择你的方法

对于初学者，`using PackageName` 通常完全可以，特别是对于你会大量使用的核心包（例如用于表格数据的 `DataFrames` 或用于数据可视化的 `Plots`）。它使你的代码简洁。

如果你遇到名称冲突（例如，Julia 警告你一个名称在多个地方定义），或者如果你想非常明确地说明每个函数来自哪里（这在大型项目或协作时很有用），那么请切换到 `import PackageName` 并使用 `PackageName.functionName` 这样的限定名称。选择性导入 `import PackageName: specificName` 提供了一个折衷方案。

随着你编写更多的 Julia 代码，你会逐渐体会到哪种方法在不同情况下最有效。重点是 Julia 提供了灵活的机制来管理外部代码如何与你的代码集成，帮助你基于 Julia 社区的贡献构建功能强大的应用程序。

## 参考资料

- [Modules](https://docs.julialang.org/en/v1/manual/modules/) — The Julia Language Contributors (2024)
  Publisher: The Julia Language
  详细介绍了Julia的模块系统，包括`using`和`import`语句、命名空间管理以及代码组织。
- [The Pkg.jl Guide](https://pkgdocs.julialang.org/v1/) — The Julia Language Contributors (2024)
  Julia包管理器Pkg的权威指南，解释了如何添加、管理和理解包的依赖项及环境。

---
course: "julia-for-machine-learning"
chapter: "julia-ml-getting-started"
lesson: "julia-ml-environment-setup"
sourceId: 6834
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-1-julia-ml-getting-started/julia-ml-environment-setup"
title: "配置您的 Julia 机器学习环境"
description: "学习如何安装 Julia 并配置您的开发环境，为机器学习项目准备基本软件包。"
order: 2
plots: []
sourceHash: "aaccbb1589958b62fbdc5949038395cf68ba4cdc8b83d01af92355f38eff19d6"
sourceCorrections: []
---

正确配置您的 Julia 环境是顺畅进行机器学习 (machine learning)开发的一个必要步骤。首先，您需要安装 Julia，熟悉其交互式命令行，选择编辑器，并安装一些基本软件包。

## 安装 Julia

首先，您需要下载并安装 Julia。
Julia 语言官方网站 [julialang.org](https://julialang.org) 是下载的权威来源。您会找到适用于所有主流操作系统的安装程序：Windows、macOS 和 Linux，以及详细的安装说明。

Julia 提供不同版本：

- **长期支持 (LTS) 版：** 这些版本会在较长时间内获得错误修复支持，使其成为需要长期稳定性的项目的稳妥选择。
- **稳定版：** 这是最新正式发布的版本，包含最新功能。对于学习和一般开发，通常推荐使用最新的稳定版。

下载适用于您系统的安装程序，并按照提供的说明操作。在 Linux 和 macOS 上，您可能还会通过 `apt`、`yum` 或 `brew` 等软件包管理器找到 Julia，但直接从网站下载可确保您获得最新版本。

安装完成后，通过打开系统终端（或 Windows 上的命令提示符）并输入以下命令进行验证：

```bash
julia
```

此命令应启动 Julia 的交互式会话，即 REPL（Read-Eval-Print Loop，读取-评估-打印循环），您将看到显示 Julia 版本的横幅，后跟 `julia>` 提示符。

```
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

要退出 REPL，请键入 `exit()` 或按 `Ctrl+D`。

## Julia REPL

REPL 对任何 Julia 开发者来说都是一个非常有用的工具。它让您可以交互式执行 Julia 代码、快速验证想法、检查对象和管理软件包。当您在终端中输入 `julia` 时，就进入了 REPL。

`julia>` 提示符是您输入 Julia 表达式的地方。按下 Enter，Julia 将评估表达式并输出结果。

```julia
julia> 1 + 1
2

julia> println("Hello, Julia for ML!")
Hello, Julia for ML!
```

REPL 有几种有用的模式：

- **帮助模式 (`?`)：** 在 `julia>` 提示符下键入 `?`，它会变为 `help?>`。然后您可以键入任何函数或宏的名称以获取其文档。按退格键可返回 Julia 提示符。

  ```julia
  julia> ?println
  search: println printstyled print print_shortest sprint

  println([io::IO], xs...)

  Print (using print) xs followed by a newline.

  If io is not supplied, prints to stdout.

  Examples
  ≡≡≡≡≡≡≡≡≡≡

  julia> println("Hello, world")
  Hello

  julia> io = IOBuffer();

  julia> println(io, "Hello")

  julia> String(take!(io))
  Hello, \n
  ```
- **软件包模式 (`]`)：** 在 `julia>` 提示符下键入 `]`，它会变为 `(@vX.Y) pkg>`。此模式用于 Julia 的内置软件包管理器 Pkg。您可以在此添加、删除、更新软件包，以及管理项目环境。我们将在下面稍微提及此模式，并在本章稍后专门的部分中详细介绍 `Pkg.jl`。按退格键可返回 Julia 提示符。

在 REPL 中尝试是学习 Julia 语法和功能的好方法。

## 选择您的开发工具

尽管 REPL 非常适合快速任务，但对于大型项目，您会需要一个更强大的编辑器或集成开发环境 (IDE)。以下是 Julia 开发的一些流行选择：

- **Visual Studio Code (VS Code) 及 Julia 扩展：** 这是目前最受欢迎、功能最完善的 Julia 环境。VS Code 的官方 Julia 扩展提供了语法高亮、代码补全（IntelliSense）、集成 REPL、绘图库、调试能力和静态代码检查。如果您是 Julia IDE 的新手，VS Code 是一个很好的开始。
- **Jupyter Notebooks 及 IJulia.jl：** 如果您习惯于 Python 或 R 中的 Jupyter Notebooks，您会很高兴得知 Julia 通过 `IJulia.jl` 软件包提供了很好的支持。Notebooks 非常适合进行数据分析，将代码、文本、公式和可视化内容整合在一个文档中。
- **其他文本编辑器：** Sublime Text、Atom、Emacs 和 Vim 等编辑器也通过社区维护的插件提供 Julia 语言支持，集成程度各异。

对于机器学习 (machine learning)任务，VS Code 或 Jupyter Notebooks 尤其适合，因为它们能够轻松运行代码、可视化数据和管理复杂项目。

> 配置 Julia 开发环境的主要步骤和工具。详细的软件包管理将在后续讨论。

## 安装您的首批软件包

Julia 的强大功能通过其丰富的软件包生态系统得到很大扩展。您可以使用 REPL 中的 Pkg 模式安装软件包。如果您打算使用 Jupyter Notebooks，我们来安装 `IJulia.jl`；并安装 `Plots.jl` 以获得基本绘图功能，这在早期快速检查时会很有用。

1. 打开 Julia REPL。
2. 键入 `]` 进入 Pkg 模式。提示符将变为 `(@vX.Y) pkg>`。
3. 要添加 `IJulia.jl`，请键入：

   ```
   (@vX.Y) pkg> add IJulia
   ```

   按下 Enter。Pkg 将下载并安装 `IJulia.jl` 及其依赖项。
4. 同样地，要添加 `Plots.jl`，请键入：

   ```
   (@vX.Y) pkg> add Plots
   ```

   如果您更喜欢用 `Makie.jl` 进行绘图，也可以类似地使用 `add Makie` 进行安装。
5. 按退格键退出 Pkg 模式，返回 `julia>` 提示符。

您现在已经安装了您的首批 Julia 软件包！我们将在“使用 Pkg.jl 进行软件包管理”一节中更详细地介绍软件包管理、项目特定环境以及 `Project.toml` 和 `Manifest.toml` 文件的作用。目前，这个基本的 `add` 命令足以让您开始。其他机器学习 (machine learning)专用软件包，例如 `DataFrames.jl`、`MLJ.jl` 和 `Flux.jl`，将在后续章节中按需介绍和安装。

## 快速测试

让我们确保一切正常。在您的 Julia REPL 中：

```julia
julia> println("My Julia environment is ready for machine learning!")
我的 Julia 环境已为机器学习做好准备！

julia> my_variable = 42
42

julia> result = my_variable / 2 + 5
26.0

julia> println("The result of a simple calculation is: ", result)
简单计算的结果是：26.0
```

如果您安装了 `Plots.jl` 并想测试它（首次运行时可能需要一些时间，因为它会进行预编译）：

```julia
julia> using Plots
julia> plot(rand(5, 5), title="我的第一个 Julia 图表", legend=false)
```

这应该会打开一个窗口，显示一个包含 5 个随机序列的简单图表。绘图的具体细节将在后面介绍，但这证实了一个具备图形功能的软件包可以加载和使用。

Julia 已安装，REPL 可访问，编辑器已选择，并且能够添加软件包，您的基本环境已配置好。您现在已准备好学习 Julia 用于数据操作和机器学习 (machine learning)的功能。

## 参考资料

- [The Julia Language Documentation](https://docs.julialang.org/) — The Julia Project developers (2024)
  Publisher: The Julia Project
  提供关于Julia安装、REPL使用和包管理的全面指导。
- [Julia Extension for Visual Studio Code Documentation](https://www.julia-vscode.org/) — Julia Computing and contributors (2024)
  Visual Studio Code中Julia扩展的官方文档，详细介绍了其开发功能和配置方法。
- [IJulia.jl Documentation](https://julialang.github.io/IJulia.jl/stable/) — The IJulia Project and contributors (2025)
  提供将Julia与Jupyter Notebook集成所需的说明，对于交互式和文学化编程环境至关重要。

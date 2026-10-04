---
course: "getting-started-julia-programming"
chapter: "modules-packages-project-organization"
lesson: "introduction-pkg-julia-package-manager"
sourceId: 6805
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-6-modules-packages-project-organization/introduction-pkg-julia-package-manager"
title: "Pkg 介绍：Julia 的包管理器"
description: "开始使用 Pkg，Julia 的内置包管理器，用于处理项目依赖项和环境。"
order: 4
plots: []
sourceHash: "0e5b3cf9425faa2d792b6b6f1c18f1afc595adf6ebb4927cc8dcc34afd812934"
sourceCorrections: []
---

模块帮助您将*自己的*代码组织成易于管理的部分，而许多编程任务可以从使用他人编写的代码中受益。Julia 环境提供了丰富的外部库集合，称为包，它们提供从绘制数据到构建复杂机器学习 (machine learning)模型等各种专用功能。为了帮助您管理和使用此资源，Julia 配备了一个强大的内置工具：`Pkg`，即 Julia 包管理器。

可以把 `Pkg` 看作是处理所有外部 Julia 软件的助手。它的主要作用是查找、安装、更新和管理项目所依赖的包。如果没有像 `Pkg` 这样的包管理器，您将面临手动下载库文件、确保它们彼此兼容以及与您的 Julia 版本兼容，并将它们放置在代码可以找到的位置的繁琐且容易出错的任务。`Pkg` 自动化了所有这些操作，使得扩展 Julia 的核心功能变得简单明了。

对于初学者来说，`Pkg` 尤其有价值，因为它大大降低了使用复杂工具的门槛。需要处理 CSV 文件？有相应的包。想创建可视化？也有相应的包。`Pkg` 允许您通过简单的命令将这些工具集成到您的项目中。

`Pkg` 的核心是通过与包注册表交互来工作的。注册表就像一个可用的 Julia 包目录。当您要求 `Pkg` 添加一个包时，它会查询一个注册表（默认是“General”注册表，由 Julia 社区维护），以查找关于该包及其依赖项的信息。然后它下载必要的文件，并使它们可供您的项目使用。

> 此图说明了您（开发者）如何指示 `Pkg` 添加一个包。`Pkg` 随后与包注册表通信，查找并下载该包，使其功能可供您的 Julia 项目使用。

`Pkg` 的一个重要特点是其项目环境的理念。您处理的每个项目都可以拥有自己独立的包集合及其特定版本。这非常有用，因为不同的项目可能需要同一包的不同版本，或者一个项目可能使用另一个项目不需要的包。环境可以防止它们相互干扰，从而实现更稳定和可预测的开发。

这种项目特定包的管理主要通过 `Pkg` 在您的项目主目录中创建和维护的两个文件来完成：

- **`Project.toml`**：此文件列出了您的项目直接依赖的包。您可以将其视为项目的包采购清单。它还可能指定版本兼容性规则，例如“我需要此包的 1.0 或更新版本。”
- **`Manifest.toml`**：此文件更详细。它记录了您的项目使用的*所有*包的精确版本，包括来自 `Project.toml` 的直接依赖项和任何间接依赖项（您选择的包本身所依赖的包）。`Manifest.toml` 确保如果您与他人共享您的项目，或者您以后再使用它，您都可以使用完全相同的包版本重现完全相同的环境，从而获得可重复的结果。它就像一张完整的、逐项列出的收据，确保您每次都能得到相同的物品。

您通常会在 Julia REPL（读取-求值-打印循环）中使用 `Pkg` 的特殊模式与其交互。要进入此模式，您只需在 Julia 提示符下键入右方括号 `]`：

```julia
julia> ]
pkg>
```

请注意提示符如何从 `julia>` 变为 `pkg>`。您现在键入的任何命令都将由包管理器解释。要退出 `pkg>` 模式并返回到标准 Julia 提示符，您可以在空行上按 Backspace 键或 Ctrl+C。

在接下来的部分中，我们将查看您在 `pkg>` 模式下将使用的特定命令，以便查找、添加、更新和管理您的 Julia 项目的包。了解 `Pkg` 是在 Julia 中开发更复杂应用程序的根本，因为它提供了访问全球 Julia 社区创建的各种工具和功能的途径。

## 参考资料

- [Pkg.jl Documentation](https://pkgdocs.julialang.org/v1/) — The Julia Language Developers (2024)
  这份官方文档解释了包管理器的命令、功能以及项目环境文件（Project.toml, Manifest.toml）。

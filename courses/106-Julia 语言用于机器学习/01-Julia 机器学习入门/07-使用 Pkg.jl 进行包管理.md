# 使用 Pkg.jl 进行包管理

来源：[原文](https://apxml.com/zh/courses/julia-for-machine-learning/chapter-1-julia-ml-getting-started/julia-pkg-management)

[返回章节目录](README.md) · [返回课程目录](../README.md)

当您开始使用 Julia 构建机器学习 (machine learning)模型时，管理项目所需的软件组件就成为您工作流程中的一个重要方面。Julia 配备了内置的复杂包管理器 `Pkg.jl`，旨在处理依赖关系、管理项目特有的环境，并保证您的机器学习项目可重现和可共享。了解 `Pkg.jl` 对于高效、有条理的开发流程非常必要。

`Pkg.jl` 管理项目需要的包，例如用于数据处理的 `DataFrames.jl` 或用于机器学习任务的 `MLJ.jl`。它保证您拥有这些包的正确版本，并且它们自身的依赖也得到满足，从而避免冲突和臭名昭著的“在我的机器上能跑”的问题。这在机器学习中尤其重要，因为库的精确版本可能影响模型行为和结果。

### 与 Pkg.jl 交互：Pkg REPL 模式

与 `Pkg.jl` 交互最常用的方式是通过其特殊的 REPL（读取-评估-打印循环）模式。您可以通过在标准 Julia REPL 中输入右方括号 `]` 来进入此模式。

```julia
julia> ]
pkg>
```

您的提示符将从 `julia>` 变为 `pkg>`，表示您现在处于包管理器环境。在这里，您可以发布命令来管理项目中的包。要退出 Pkg REPL 并返回 Julia REPL，请按 Backspace 或 Ctrl+C（在空行上）。

### 依赖管理的核心：Project.toml 和 Manifest.toml

当您开始为项目管理包时，`Pkg.jl` 会在项目根目录中创建并使用两个重要文件：

1. **`Project.toml`**：此文件列出了项目的直接依赖。可以把它看作是项目需求的高级声明。对于您明确添加的每个包，`Project.toml` 会记录其名称和一组兼容版本。例如，如果您添加 `DataFrames.jl`，此处将出现一个对应条目。此文件可读性强，可以编辑（尽管通常通过 Pkg 命令间接编辑），并应提交到版本控制系统。
2. **`Manifest.toml`**：此文件是项目运行所需*所有*包的完整详细列表，包括直接依赖（`Project.toml` 中的）和间接依赖（您的直接依赖所依赖的包）。重要之处在于，`Manifest.toml` 记录了依赖图中每个包的*精确*版本。此文件保证任何使用您项目的人都能精确重现环境，细致到每个组件的特定版本。`Manifest.toml` 由 `Pkg.jl` 自动生成和更新；通常不应手动编辑它。它也应该提交到版本控制系统。

这两个文件共同提供了一个定义和重现项目环境的系统。

> 此图展示了 Pkg.jl 命令，通过 Pkg REPL 模式访问，如何与项目目录中的 `Project.toml` 和 `Manifest.toml` 文件进行交互，以及这些文件如何与外部包注册表和您自己的代码相关联。

### 管理项目环境

默认情况下，如果您不明确激活项目环境，`Pkg.jl` 会在针对您的 Julia 版本的全局环境上运行。尽管这对于快速实验来说没问题，但对于任何严肃的项目，特别是在机器学习 (machine learning)中，您应该使用项目专属环境。这可以隔离每个项目的依赖项，防止不同项目间的版本冲突。

为新项目创建并激活环境：

1. 在终端中进入项目根目录。
2. 启动 Julia REPL。
3. 输入 `]` 进入 Pkg 模式。
4. 激活当前目录的环境：

```
pkg> activate .
  激活新项目于 `/path/to/your/project`
```

`.` 表示当前目录。如果 `Project.toml` 不存在，`Pkg.jl` 会创建一个。现在，任何包操作都只会影响此项目的环境。`pkg>` 提示符也可能改变以反映活动项目，例如 `(your-project) pkg>`。

### Pkg.jl 的常用命令

以下是 Pkg REPL 模式中最常用的一些命令：

- **`add PackageName`**：将 `PackageName` 的最新版本（及其依赖）添加到当前项目环境。它会更新 `Project.toml` 和 `Manifest.toml`。

  ```
  (your-project) pkg> add DataFrames
  ```

  要添加特定版本的包，使用 `add PackageName@version`，例如 `add DataFrames@1.3`。您也可以直接从 Git 仓库或本地路径添加包。
- **`status` (或 `st`)**：显示当前活动环境中已安装的包，及其版本和简要状态。

  ```
  (your-project) pkg> st
  Project.toml 状态
    [a93c6f00] DataFrames v1.3.6
    ...
  ```
- **`update` (或 `up`)**：将 `Project.toml` 中的所有包更新到其最新兼容版本，同时遵守定义的兼容性限制。如果您想更新特定包，使用 `up PackageName`。

  ```
  (your-project) pkg> up
  ```
- **`rm PackageName`**：从项目的直接依赖中移除 `PackageName`。`Pkg.jl` 也会移除任何不再需要的间接依赖。

  ```
  (your-project) pkg> rm DataFrames
  ```
- **`instantiate`**：这是一个非常有用的命令，特别是在协作工作或在新机器上设置项目时。它会读取 `Manifest.toml` 文件（如果 `Manifest.toml` 不存在，则读取 `Project.toml`）并下载和安装所有列出的包到它们指定的版本。这可以确保您获得项目定义的精确环境。

  ```
  (your-project) pkg> instantiate
  ```

  如果您从包含 `Project.toml` 和 `Manifest.toml` 的仓库克隆了一个 Julia 项目，那么在进入项目目录并激活其环境后，通常的第一步就是运行 `instantiate`。

### 保证机器学习 (machine learning)工作流程的可重现性

可重现性是良好科学实践的根基，在机器学习中同样重要。`Pkg.jl` 通过 `Project.toml` 和 `Manifest.toml` 提供了实现此目标的方法：

1. **始终使用项目专属环境**：为每个新项目激活一个环境（`activate .`）。
2. **将 `Project.toml` 和 `Manifest.toml` 提交到版本控制系统**（例如 Git）以及您的源代码。这些文件完整描述了您的项目依赖。
3. 当协作者（或您未来在另一台机器上的自己）克隆仓库时，他们可以 `cd` 到项目目录，启动 Julia，进入 Pkg 模式，然后运行 `activate .` 后跟 `instantiate`。这将重现精确的软件环境，保证代码按预期运行且结果一致。

### Pkg.jl 的编程使用

虽然 Pkg REPL 模式方便交互使用，`Pkg.jl` 也可以在您的 Julia 脚本中以编程方式使用。这对于自动化环境设置或管理 Julia 项目的工具很有用。

要使用 `Pkg.jl` 函数，您首先需要导入它：

```julia
import Pkg  # 或使用 using Pkg 导入所有导出名称

# 示例：以编程方式添加包
Pkg.add("JSON")

# 示例：实例化当前项目的环境
Pkg.instantiate()
```

这种编程访问允许对项目和包管理任务进行强大的脚本编写。

掌握 `Pkg.jl`，您将对项目依赖项获得细致的控制，从而使 Julia 中的机器学习 (machine learning)项目更稳定、可重现、易于管理。随着您学习本课程并开始使用各种机器学习专用包，您会发现 `Pkg.jl` 是您的 Julia 工具包中不可或缺的工具。

## 参考资料

- [Pkg.jl Documentation](https://pkgdocs.julialang.org/v1/) — The Julia Language Developers (2024)
  这是 `Pkg.jl` 的主要全面来源，详细介绍了其功能、REPL 模式、配置文件（`Project.toml`、`Manifest.toml`）和基本命令。
- [Environments](https://pkg.julialang.org/v1/manual/environments/) — The Julia Language Developers (2025)
  Publisher: The Julia Language
  Julia 官方手册的此部分阐述了 Julia 中的环境概念，解释了其在管理项目依赖和保证可复现设置方面的作用。
- [Julia Programming for Operations Research](https://www.chkwon.net/julia) — Changhyun Kwon (2023)
  Publisher: Springer; DOI: [10.1007/978-3-031-28526-7](https://doi.org/10.1007/978-3-031-28526-7)
  一本现代教材，涵盖了基础的 Julia 编程，包括对 `Pkg.jl` 和项目管理实践的详细解释，以构建稳定的应用程序。

---

[上一节](06-DataFrames.jl%20%E7%AE%80%E4%BB%8B.md) · [下一节](08-%E5%AE%9E%E8%B7%B5%E6%93%8D%E4%BD%9C%EF%BC%9A%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE%E4%B8%8E%E5%9F%BA%E6%9C%AC%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86.md)

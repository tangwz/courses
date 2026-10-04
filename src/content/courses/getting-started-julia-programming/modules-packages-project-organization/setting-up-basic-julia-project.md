---
course: "getting-started-julia-programming"
chapter: "modules-packages-project-organization"
lesson: "setting-up-basic-julia-project"
sourceId: 6808
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-6-modules-packages-project-organization/setting-up-basic-julia-project"
title: "设置一个基本的 Julia 项目"
description: "使用 Pkg 创建新的 Julia 项目并设置其独立环境和依赖清单的步骤。"
order: 7
plots: []
sourceHash: "2a1b0074896feae0fc23176f4c53962fab028e023b5354403fb43c3b69aaa3fc"
sourceCorrections: []
---

随着你的 Julia 程序变大，保持代码组织有序不只是好习惯，更是代码易于维护和共享的必需。Julia 提供了其包管理器 Pkg，用于为新的 Julia 项目创建结构化环境。这种做法有助于有效地管理依赖项，并确保你的项目可以复现。

### 为什么需要专用项目环境？

设想你同时进行多个 Julia 项目。项目 A 可能需要某个包的 1.2 版本，而项目 B 则需要同一包的 2.0 版本。如果所有包都安装在同一个全局位置，你很快就会遇到冲突。

Julia 的项目环境由 Pkg 管理，解决了这个问题。每个项目都有自己的独立空间，记录其特定的依赖项及其版本。这带来了多项好处：

- **可复现性**：你可以确信你的代码今天、明天或在不同的机器上都会以相同的方式运行，因为精确的包版本已被锁定。
- **依赖隔离**：项目之间互不干扰其包需求。
- **清晰结构**：它有利于你的项目文件保持整洁有序的布局。

### 创建你的第一个项目

让我们创建一个新的 Julia 项目。启动一个新的、可共享的项目或包的推荐方式是使用 Pkg 中的 `generate` 命令。

1. **打开 Julia REPL**：在终端中输入 `julia` 来启动 Julia。

   ```
   $ julia
                  _
      _       _ _(_)_     |  Documentation: https://docs.julialang.org
     (_)     | (_) (_)    |
      _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
     | | | | | | |/ _` |  |
     | | |_| | | | (_| |  |  Version 1.x.y (YYYY-MM-DD)
    _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
   |__/                   |

   julia>
   ```
2. **进入 Pkg 模式**：输入 `]` 将 Julia 提示符 (`julia>`) 切换到 Pkg 提示符 (`pkg>`)。

   ```
   julia> ]
   pkg>
   ```
3. **生成项目**：使用 `generate` 命令，后跟你期望的项目名称。我们将项目命名为 "DataAnalyzer"。

   ```
   pkg> generate DataAnalyzer
    Generating  project DataAnalyzer:
       DataAnalyzer/Project.toml
       DataAnalyzer/src/DataAnalyzer.jl
   ```

   Pkg 创建了一个名为 `DataAnalyzer` 的新目录。在其中，它放置了一个 `Project.toml` 文件（我们稍后会讨论）和一个 `src` 目录，该目录包含一个基本的 Julia 文件 `DataAnalyzer.jl`。此文件已设置为模块，你可以开始添加代码。
4. **进入你的项目目录**：按下 Backspace 或 Ctrl+C 退出 Pkg 模式。然后，从你的操作系统终端中，进入新创建的项目目录。

   ```
   pkg> # Press Backspace
   julia>
   ```

   现在，在你的系统终端中：

   ```
   $ cd DataAnalyzer
   ```

### 激活项目环境

一旦你进入了项目目录 (`DataAnalyzer`)，你需要告诉 Julia 使用该项目的特定环境。这被称为“激活”环境。

有两种常用方法：

1. **启动 Julia 时使用 `--project` 标志**：
   在终端中，当你在 `DataAnalyzer` 目录中时，像这样启动 Julia：

   ```
   $ julia --project=.
   ```

   `.` 指当前目录。Julia 将检测到 `Project.toml` 文件并使用它。你的 Julia REPL 提示符现在将以项目名称作为前缀：

   ```
   (DataAnalyzer) julia>
   ```
2. **在 Julia REPL 内部激活**：
   如果你已经在使用 Julia 或者更喜欢正常启动 Julia，你可以在进入项目目录后激活环境：

   - 启动 Julia：`julia`
   - 进入 Pkg 模式：`]`
   - 使用 `activate` 命令。由于你在项目目录中，`.` 指当前目录的项目。

     ```
     julia> # (确保你的终端在 DataAnalyzer 目录中)
     julia> ]
     (v1.x) pkg> activate .
       Activating project at `~/path/to/your/DataAnalyzer`

     (DataAnalyzer) pkg>
     ```

   注意 Pkg 提示符如何从 `(v1.x) pkg>`（代表默认全局环境）变为 `(DataAnalyzer) pkg>`。这确认你的项目环境已激活。

### Julia 项目的构成：`Project.toml` 和 `Manifest.toml`

当你生成或激活一个项目时，Pkg 依赖于两个重要文件：`Project.toml` 和 `Manifest.toml`。

#### `Project.toml` - 你项目的身份证明

此文件是项目的主要配置文件。它包含元数据，例如项目名称、其唯一标识符 (UUID)、作者信息，以及重要的是，其*直接*依赖项的列表。这些是项目明确声明所需的包。

如果你在生成 `DataAnalyzer/Project.toml` 后打开它，你会看到类似以下内容：

```toml
name = "DataAnalyzer"
uuid = "1234abcd-5678-efgh-ijkl-mnopqrstuvwx" # This will be a unique UUID
authors = ["Your Name <your.email@example.com>"] # Might be prefilled or you add it
version = "0.1.0"

[deps]
# Direct dependencies will be listed here later
```

- `name`：你的项目名称。
- `uuid`：你的项目的通用唯一标识符。当你的项目成为其他项目可能依赖的包时，这很重要。
- `authors`：关于项目创建者的信息。
- `version`：你的项目版本，遵循语义版本控制（例如，`0.1.0`）。
- `[deps]`：此部分将列出项目直接使用的包的名称和 UUID。

你通常会编辑 `authors` 字段，并可能在项目发展过程中调整 `version`。`[deps]` 部分通常由 Pkg 命令管理。

#### `Manifest.toml` - 可复现的精确方案

除了 `Project.toml`，你还会找到一个 `Manifest.toml` 文件（它可能在你第一次添加包或 `实例化` 环境时创建）。此文件是 Pkg 确保完全可复现性的一种方式。

尽管 `Project.toml` 列出你想要的*哪些*直接包（例如，“我需要一个 JSON 解析包”），`Manifest.toml` 则记录了*所有*实际用于满足这些需求的包的*精确版本*。这包括直接依赖项以及任何间接依赖项（你选择的包本身依赖的包）。

例如，如果 `DataAnalyzer` 依赖于 `PackageA`，而 `PackageA` 又依赖于 `PackageB` 和 `PackageC`，那么你的 `Project.toml` 只会列出 `PackageA`。然而，你的 `Manifest.toml` 将列出被解析和安装的 `PackageA`、`PackageB` 和 `PackageC` 的特定版本。

你通常**不手动编辑 `Manifest.toml`**。当你添加、移除或更新包时，Pkg 会自动更新它。此文件对于确保其他人（或你本人，在不同计算机上或未来）可以设置与完全相同包版本的环境非常重要，从而带来一致的行为。

### 为你的项目添加依赖项

假设你的 `DataAnalyzer` 项目需要处理 JSON 数据。为此你需要一个包，例如 `JSON.jl`。

1. **确保你的项目环境已激活**。你的 Pkg 提示符应显示为 `(DataAnalyzer) pkg>`。
2. **添加包**：

   ```
   (DataAnalyzer) pkg> add JSON
      Updating registry at `~/.julia/registries/General.toml`
     Resolving package versions...
    Installing JSON3 v1.1.2
     Installed JSON3 ─ v1.1.2
    Updating `~/path/to/your/DataAnalyzer/Project.toml`
     [3c869059] + JSON3 v1.1.2
    Updating `~/path/to/your/DataAnalyzer/Manifest.toml`
     [3c869059] + JSON3 v1.1.2
     # ... other potential indirect dependencies
   ```

   *(注意：Pkg 可能会根据注册表状态选择 `JSON3.jl` 或其他 JSON 包。过程是相同的。)*

现在，如果你检查你的 `Project.toml`，你会看到 `JSON3`（或类似名称）已添加到 `[deps]` 部分下：

```toml
[deps]
JSON3 = "3c869059-04de-5ccb-9dba-057ba395f190"
```

你的 `Manifest.toml` 将为 `JSON3` 及其依赖的任何包提供更详细的条目，指定精确版本和 git 树哈希值以实现完美的可复现性。

### 基本项目结构和你的代码

`generate` 命令创建了你的项目的标准布局：

> Pkg 生成的新 Julia 项目的典型目录结构。

- `DataAnalyzer/`：你的项目根目录。
- `Project.toml`：定义项目元数据和直接依赖项。
- `Manifest.toml`：记录所有依赖项的精确版本以实现可复现性。
- `src/`：此目录通常是你项目主要 Julia 源代码的所在地。
  - `DataAnalyzer.jl`：Pkg 创建了具有基本模块结构的此文件：

    ```julia
    module DataAnalyzer

    greet() = print("Hello World!")

    end # 模块 DataAnalyzer
    ```

    你可以修改此文件以包含项目的函数、类型和逻辑。对于大型项目，你可以在 `src` 目录中添加更多 `.jl` 文件，并在 `DataAnalyzer.jl` 中使用 `include()` 语句来组织你的代码。

### 在你的项目环境中工作

每当你处理 `DataAnalyzer` 项目时，无论是编写代码、运行脚本还是管理包，请确保其环境已激活。这保证 Julia 使用项目中 `Project.toml` 和 `Manifest.toml` 文件中定义的正确版本包。如果你的 REPL 提示符显示 `(DataAnalyzer) julia>` 或 `(DataAnalyzer) pkg>`，则表示一切正常。

以这种方式设置项目最初可能看起来有点额外工作，但它很快就会变得得心应手。在组织、协作以及确保你的代码在不同设置下可靠运行方面的好处，都是非常值得的。你现在拥有了一个坚实基础，可以构建更复杂的 Julia 应用程序。

## 参考资料

- [Pkg.jl Documentation](https://pkgdocs.julialang.org/v1/) — The Julia Language Developers (2023)
  Julia包管理器的综合指南，涵盖项目环境、依赖管理和项目结构，对可复现的开发至关重要。
- [Modules and Packages](https://docs.julialang.org/en/v1/manual/modules/) — The Julia Language Developers (2023)
  解释Julia的模块系统和包的结构方式，为项目内的代码组织提供基础理解。

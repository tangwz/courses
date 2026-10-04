---
course: "getting-started-julia-programming"
chapter: "modules-packages-project-organization"
lesson: "practice-organizing-code-dependencies"
sourceId: 6809
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-6-modules-packages-project-organization/practice-organizing-code-dependencies"
title: "实践：组织代码与管理依赖"
description: "动手操作，在 Julia 中创建模块、管理包并设置简单的项目结构。"
order: 8
plots: []
sourceHash: "ac656c91486407ecf648706ee53b23cd82c1d6b2793ebd6fa09e57001d8a6ff8"
sourceCorrections: []
---

本内容提供了模块在代码组织中的作用以及 Pkg 包管理器在处理依赖项方面的实践经验。通过这些练习，您将首先创建并使用一个简单模块，然后建立一个新的 Julia 项目并引入一个外部包。这些步骤将帮助您更好地构建更大的 Julia 应用程序。

### 练习一：创建和使用本地模块

本练习侧重于在单个项目目录内创建和使用模块的基础知识。模块能帮助您捆绑相关函数和常量，从而避免命名冲突，并让您的代码更易于维护。

1. **设置您的工作区：**
   为本次练习创建一个新目录，命名为 `MyModuleProject`。使用终端进入该目录。

   ```bash
   mkdir MyModuleProject
   cd MyModuleProject
   ```
2. **定义您的模块：**
   在 `MyModuleProject` 中，创建一个名为 `StringUtils.jl` 的 Julia 文件。此文件将包含我们的模块。将以下代码添加到 `StringUtils.jl` 中：

   ```julia
   # MyModuleProject/StringUtils.jl
   module StringUtils

   export greet, count_characters

   """
       greet(name::String)

   返回个性化的问候字符串。
   """
   function greet(name::String)
       return "Hello, $name! Greet to your custom module."
   end

   """
       count_characters(text::String)

   计算给定字符串中的字符数。
   """
   function count_characters(text::String)
       return length(text)
   end

   end # module StringUtils
   ```

   在此 `StringUtils` 模块中，我们定义了两个函数：`greet` 和 `count_characters`。请注意 `export` 关键字；这使得 `greet` 和 `count_characters` 在模块导入时可用。还包含了文档字符串，这是说明函数用途的好做法。
3. **在脚本中使用模块：**
   现在，在 `MyModuleProject` 目录中创建另一个 Julia 文件。将此文件命名为 `main.jl`。此脚本将使用 `StringUtils` 模块。

   ```julia
   # MyModuleProject/main.jl

   # `include` 函数执行 StringUtils.jl 文件，
   # 使其中定义的 StringUtils 模块可用。
   include("StringUtils.jl")

   # `using` 关键字将 StringUtils 中导出的名称
   # 引入当前作用域。StringUtils 前的点表示
   # 它是在当前作用域/工作区中定义的模块。
   using .StringUtils

   # 调用模块中的函数
   println(greet("Julia Developer"))

   message = "Learning Julia modules is fun!"
   char_count = count_characters(message)
   println("The message '$message' has $char_count characters.")
   ```
4. **运行您的脚本：**
   从终端执行 `main.jl`，确保您仍在 `MyModuleProject` 目录中：

   ```bash
   julia main.jl
   ```

   您应该看到以下输出：

   ```
   Hello, Julia Developer! Greet your custom module.
   The message 'Learning Julia modules is fun!' has 30 characters.
   ```

这个第一个练习演示了将代码组织成模块并在同一目录中使用的简单方法。在这种情况下，`include` 函数是用于从另一个文件加载模块定义的简单机制。

> 本地模块练习的目录结构和文件交互。`main.jl` 脚本包含并使用了在 `StringUtils.jl` 中定义的 `StringUtils` 模块。

### 练习二：使用 Pkg 设置项目并使用包

在此练习中，您将学习使用 Julia 内置的包管理器 Pkg 来创建一个专用的项目环境并添加一个外部包。这是管理更结构化的 Julia 项目依赖项的标准方式。

1. **创建一个新的项目目录：**
   在 `MyModuleProject` 目录之外，为本项目创建一个新文件夹，例如 `DataAnalysisProject`。进入该目录。

   ```bash
   mkdir DataAnalysisProject
   cd DataAnalysisProject
   ```
2. **初始化并激活项目环境：**
   在 `DataAnalysisProject` 目录内启动 Julia REPL。

   ```bash
   julia
   ```

   进入 REPL 后，输入 `]` 进入 Pkg 模式。然后，为当前目录（`.`）激活一个新环境：

   ```julia
   (@v1.x) pkg> activate .
     Activating new project at `/path/to/your/DataAnalysisProject` # 实际路径会有所不同
   ```

   您的 REPL 提示符现在应变为 `(DataAnalysisProject) pkg>`。这表明您正在此项目专用的环境中工作。任何添加的包都将列在此项目的 `Project.toml` 和 `Manifest.toml` 文件中，而不是全局列表。
3. **添加一个外部包：**
   让我们添加 `Dates` 包，该包是 Julia 标准库的一部分，但仍在项目环境中进行显式管理会有好处。此包提供了用于处理日期和时间的函数。

   ```julia
   (DataAnalysisProject) pkg> add Dates
     Resolving package versions...
      Updating `/path/to/your/DataAnalysisProject/Project.toml`
     [Dates]
      Updating `/path/to/your/DataAnalysisProject/Manifest.toml`
     [Dates, Printf, Unicode] # 实际依赖项可能略有不同
   ```

   命令完成后，Pkg 将会下载有关该包及其依赖项的信息（如果需要），并将 `Dates` 记录为项目的 `Project.toml` 文件中的依赖项。`Manifest.toml` 文件将存储所有已安装包的精确版本，确保您的项目可复现。

   按 `Backspace` 或 `Ctrl+C` 退出 Pkg 模式并返回到 `julia>` 提示符。
4. **在脚本中使用包：**
   在您的 `DataAnalysisProject` 目录中创建一个名为 `report_generator.jl` 的新 Julia 脚本文件。添加以下代码：

   ```julia
   # DataAnalysisProject/report_generator.jl
   using Dates # 使 Dates 包中的函数可用

   function generate_daily_report_filename()
       current_date = today() # 获取当前日期
       formatted_date = Dates.format(current_date, "yyyy-mm-dd")
       return "Report_$(formatted_date).txt"
   end

   report_file = generate_daily_report_filename()
   println("Today's report filename: $report_file")

   # 另一个 Dates 函数的示例
   println("Current day of the week: ", dayname(today()))
   ```
5. **运行您的项目脚本：**
   您可以从系统的终端运行此脚本。确保您在 `DataAnalysisProject` 目录中。使用 `--project` 标志告诉 Julia 使用当前目录中定义的环境：

   ```bash
   julia --project report_generator.jl
   ```

   或者，如果您的 Julia REPL 仍在 `DataAnalysisProject` 目录中打开并处于活动状态（即 `Project.toml` 所在位置），您可以运行：

   ```julia
   include("report_generator.jl")
   ```

   输出将类似于此（日期将是当前日期）：

   ```
   Today's report filename: Report_2023-10-27.txt
   Current day of the week: Friday
   ```

通过完成本练习，您已成功创建了一个独立的（isolated）项目环境，添加了一个包依赖项，并在您的代码中使用了该包。您还将在 `DataAnalysisProject` 目录中找到两个新文件：

- `Project.toml`：此文件列出了项目的直接依赖项（例如 `Dates`）。
- `Manifest.toml`：此文件记录了项目中使用的所有包（直接和间接）的精确版本。这对于确保您的代码在与他人共享或在不同机器上运行时能保持一致性非常重要。

> `DataAnalysisProject` 的结构，显示了 Pkg 如何管理像 `Dates` 包这样的依赖项，这些依赖项随后由 `report_generator.jl` 脚本使用。

这些练习为您提供了使用模块组织代码以及使用 Pkg 管理项目环境和依赖项的概览。随着您的项目增长，这些工具将变得必不可少，以帮助您的 Julia 编程工作保持清晰、可重用和可复现。

## 参考资料

- [Modules](https://docs.julialang.org/en/v1/manual/modules/) — The Julia Language Developers (2024)
  官方文档，解释Julia用于代码组织的模块系统，包括`module`、`export`、`using`和`include`指令。
- [Pkg.jl: The Package Manager](https://docs.julialang.org/en/v1/stdlib/Pkg/) — The Julia Language Developers (2024)
  官方文档，详细说明Julia包管理器在依赖管理、项目环境、`Project.toml`和`Manifest.toml`方面的用法。
- [Code Loading](https://docs.julialang.org/en/v1/manual/code-loading/) — The Julia Language Developers (2024)
  官方文档，阐明Julia如何加载代码，包括`include`和`using`在模块和项目组织上下文中的语义。

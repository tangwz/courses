# 编写和运行你的第一个 Julia 脚本

来源：[原文](https://apxml.com/zh/courses/getting-started-julia-programming/chapter-1-introducing-julia-setup-first-steps/writing-running-first-julia-script)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Julia REPL（读取-求值-打印 循环）是一个用于尝试代码片段和获取即时反馈的出色工具。然而，对于大型程序或可重用工具，保存你的工作变得非常重要。Julia 脚本正为此目的而存在。Julia 脚本是一个简单的纯文本文件，包含 Julia 代码，通常以 `.jl` 扩展名保存。

### 创建你的第一个脚本文件

要编写 Julia 脚本，你需要一个文本编辑器。在上一节“选择你的 Julia 开发环境”中，我们讨论了各种选项。如果你已经设置了带有 Julia 扩展的集成开发环境 (IDE)，例如 Visual Studio Code，或者带有 Juno 的编辑器，例如 Atom，你可以使用它们。目前，即使是像记事本（Windows）、文本编辑（macOS，确保处于纯文本模式）或 Gedit（Linux）这样的基本文本编辑器也足够了。

1. **打开你选择的文本编辑器。** 这将为你提供一个空白区域来编写代码。
2. **输入以下代码行：**

   ```julia
   println("Hello, Julia!")
   ```

   这一行使用了 `println()` 函数。`println`（“print line”的缩写）函数是一个基本的 Julia 命令，用于在控制台显示文本或值。你放在括号和引号内的任何内容都将被打印出来，后面跟着一个换行符，它会将光标移动到控制台的下一行。
3. **保存文件。**

   - 在文本编辑器中，选择 `文件 > 另存为...`。
   - 选择一个你容易找到的位置，例如你的桌面，或者一个你可能创建的新文件夹，名为 `julia_projects`。
   - 将文件命名为 `hello.jl`。使用 `.jl` 扩展名很重要，因为它会告诉系统（以及 Julia 解释器）这是一个 Julia 文件。

你现在已经编写了你的第一个 Julia 脚本！它很简单，但对于任何复杂程度的脚本，过程都是相同的。

### 运行你的 Julia 脚本

保存了 `hello.jl` 脚本后，下一步是运行它。这通常通过你的系统命令行界面 (CLI) 来完成，CLI 也被称为终端或命令提示符。

1. **打开你的终端或命令提示符：**

   - **Windows：** 搜索“命令提示符”或“PowerShell”。
   - **macOS：** 打开“终端”（你可以在“应用程序 > 实用工具”中找到它，或通过 Spotlight 搜索）。
   - **Linux：** 这因发行版而异，但通常你可以按 `Ctrl+Alt+T` 或在应用程序菜单中找到“终端”。
2. **导航到你保存文件的目录。** 终端通常会在你的主目录中打开。你需要使用 `cd`（更改目录）命令来移动到包含 `hello.jl` 的文件夹。

   - 例如，如果你将其保存在桌面上，你可以输入：
     `cd Desktop`
   - 如果你将其保存在“文档”文件夹中名为 `julia_projects` 的文件夹内，你可以输入（根据需要调整路径）：
     `cd Documents/julia_projects`（在 macOS/Linux 上）
     `cd Documents\julia_projects`（在 Windows 上）

   你可以使用 `ls`（在 macOS/Linux 上）或 `dir`（在 Windows 上）命令来列出当前目录中的文件，以确认 `hello.jl` 在其中。
3. **运行脚本。** 一旦你进入正确的目录，输入以下命令并按 Enter 键：

   ```bash
   julia hello.jl
   ```

   此命令指示 Julia 解释器（`julia` 部分）执行 `hello.jl` 文件中包含的代码。

你的终端上应该会显示以下输出：

```text
Hello, Julia!
```

恭喜！你已成功编写并执行了你的第一个 Julia 脚本。

如果你正在使用像 VS Code 这样带有 Julia 扩展的 IDE，通常会有一个“运行”按钮（通常是三角形图标）或菜单中的命令（例如，“Julia：执行文件”），可以直接在 IDE 中运行脚本，为你处理终端命令。这对于开发来说可能更方便。

### 一个稍微复杂一点的例子

让我们在第一个脚本的基础上进行扩展，了解更多内容。打开你的 `hello.jl` 文件（或者创建一个新文件，例如 `myscript.jl`），然后修改它或输入以下内容：

```julia
# 这是一个注释。Julia 会忽略以 # 开头的行。

# 将消息存储在变量中
message = "开始 Julia 脚本编程！"
println(message)

# 执行一个简单的计算
base_value = 15
multiplier = 3
result = base_value * multiplier

println("The result of the calculation is: ", result)
println("We multiplied ", base_value, " by ", multiplier, ".")
```

保存此文件，并使用 `julia <你的文件名>.jl` 从终端运行它。
输出将是：

```text
欢迎开始 Julia 脚本编程！
The result of the calculation is: 45
We multiplied 15 by 3.
```

在此示例中：

- 以 `#` 开头的行是注释。它们是为了方便人们理解代码；Julia 会忽略它们。
- 我们创建了变量（`message`、`base_value`、`multiplier`、`result`）来存储数据（文本和数字）。我们将在下一章详细介绍变量。
- 我们执行了一个简单的乘法运算（`*` 是乘法运算符）。
- `println` 可以打印变量的内容，如果用逗号分隔，也可以打印多个项目。

### 编辑-保存-运行 循环

你刚刚遵循的过程是：

1. 在文本编辑器中**编辑**你的代码。
2. **保存**文件。
3. 从终端（或 IDE）**运行**脚本。
4. 观察输出，并在必要时重复。

这是大多数编程的基本工作流程。当你编写更复杂的程序时，你会多次重复这个循环，一步步地构建和完善你的代码。

通过编写和运行这些简单的脚本，你已经迈出了重要一步。你现在知道如何创建可以执行、保存和共享的独立 Julia 程序。在接下来的章节中，我们将在现有学习成果之上继续学习，更详细地查看 Julia 的语法、数据类型和功能。

## 参考资料

- [The Julia Language Manual - Getting Started](https://docs.julialang.org/en/v1/manual/getting-started/) — The Julia Language Developers (2025)
  安装Julia和运行基本命令（包括从命令行执行脚本）的官方指南。
- [Think Julia: How to Think Like a Computer Scientist](https://benlauwens.github.io/ThinkJulia.jl/latest/book.html) — Ben Lauwens, Allen B. Downey (2019)
  Publisher: O'Reilly Media
  一本系统地教授使用Julia进行编程基础概念的入门教材，涵盖脚本的创建和执行。

---

[上一节](06-%E9%80%89%E6%8B%A9%E6%82%A8%E7%9A%84Julia%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83.md) · [下一节](08-Julia%20%E4%B8%AD%E7%9A%84%E6%A0%B8%E5%BF%83%E8%AF%AD%E6%B3%95%E5%85%83%E7%B4%A0.md)

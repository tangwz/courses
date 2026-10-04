---
course: "getting-started-julia-programming"
chapter: "introducing-julia-setup-first-steps"
lesson: "hands-on-environment-initial-scripting"
sourceId: 6766
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-1-introducing-julia-setup-first-steps/hands-on-environment-initial-scripting"
title: "实践：环境配置与初始脚本编写"
description: "配置 Julia 环境并编写基本脚本的实际步骤，以巩固你的理解。"
order: 10
plots: []
sourceHash: "64d0eada362b5e94e7d60ee52e8e1afc3c48c175f9f2a03439559a41435892e8"
sourceCorrections: []
---

巩固你的 Julia 设置，并练习编写和运行 Julia 代码的最初步骤是内容重点。这包括逐步检查你的安装，整理工作区，并运行一个比传统“Hello, World!”例子更进一步的简单脚本。这种实践经验对后续主题具有主要作用。

### 验证你的 Julia 安装

在编写更多代码之前，一个好的做法是确认 Julia 已正确安装，并且可以从你的系统命令行界面（例如 macOS/Linux 上的 Terminal 或 Windows 上的 Command Prompt/PowerShell）访问。

打开你的终端并输入以下命令：

```
julia --version
```

按回车键。你将看到类似于这样的输出，表明你安装的 Julia 版本：

```
julia version 1.x.y
```

（其中 `1.x.y` 是你安装的具体版本号，例如 `1.9.3`。）

如果你看到类似“julia: command not found”的错误消息，这意味着 Julia 可执行文件不在你系统的 PATH 中。你可能需要重新查看针对你的操作系统的安装说明（在“在你的系统上安装 Julia”中讲到），以确保选择了“将 Julia 添加到 PATH”选项，或者手动添加它。能够从终端的任何目录运行 `julia` 对流畅的开发体验很重要。

### 整理你的工作区和第一个脚本

良好的组织习惯应尽早培养。让我们为你的 Julia 项目创建一个专用空间并编写一个小型脚本。

1. **创建项目目录**：
   导航到你电脑上喜欢存放项目的位置（例如，你的文档文件夹）。创建一个新目录。对于本课程，你可以将其命名为 `julia_course_work` 或类似名称。

   ```bash
   # 在你的终端中，导航到你喜欢的位置
   # 例如，如果你在你的主目录中：
   cd Documents    # 或你喜欢的位置
   mkdir julia_course_work
   cd julia_course_work
   ```

   这个新目录将是你本课程练习的主要工作区。
2. **再次使用 REPL 进行快速测试**：
   打开你的终端（如果尚未打开并进入你的新项目目录）并启动 Julia REPL：

   ```
   julia
   ```

   你会看到 `julia>` 提示符。让我们尝试一个快速计算：

   ```julia
   julia> 25 * 4 + 15
   115

   julia> exit() # 或按 Ctrl+D (macOS 上按 Cmd+D)
   ```

   REPL 非常适合这类快速实验和计算。
3. **创建你的第一个计算脚本**：
   现在，让我们创建一个脚本文件。使用你选择的文本编辑器（无论是安装了 Julia 扩展的 VS Code、安装了 Juno 的 Atom，还是像 Notepad++、Sublime Text 或 Gedit 这样的简单编辑器），在你的 `julia_course_work` 目录中创建一个名为 `simple_calc.jl` 的新文件。

   将以下代码输入到 `simple_calc.jl` 中：

   ```julia
   # simple_calc.jl
   # 一个执行并显示一些计算的脚本

   # 定义几个变量
   principal_amount = 1000 # 初始金额
   annual_interest_rate = 0.05 # 5% 利率

   # 计算一年的简单利息
   interest_earned = principal_amount * annual_interest_rate

   # 计算一年后的总金额
   total_amount = principal_amount + interest_earned

   # 显示结果
   println("Initial Principal: \$$principal_amount") # 使用字符串插值和一个字面量 $
   println("Annual Interest Rate: $(annual_interest_rate * 100)%") # 使用表达式插值
   println("Interest Earned in One Year: \$$interest_earned")
   println("Total Amount after One Year: \$$total_amount")
   ```

   这个脚本中有几点需要注意：

   - 以 `#` 开头的行是注释，Julia 会忽略它们。它们是为人阅读的。
   - 我们为 `principal_amount` 和 `annual_interest_rate` 等变量赋值。
   - 我们执行计算并将结果存储在新变量中。
   - `println()` 用于向控制台显示输出。注意在文本字符串中直接使用变量名前的 `$`（例如 `$principal_amount`）或表达式周围的 `$(...)`（例如 `$(annual_interest_rate * 100)`）。这称为字符串插值，是一种将值嵌入 (embedding)字符串的便捷方式。要打印字面量美元符号，我们用反斜杠对其进行“转义”，例如 `\$`。
4. **运行你的脚本**：
   保存 `simple_calc.jl` 文件。回到你的终端，确保你仍在 `julia_course_work` 目录中（你保存文件的地方）。通过输入以下内容执行脚本：

   ```
   julia simple_calc.jl
   ```

   你将看到以下输出：

   ```
   Initial Principal: $1000
   Annual Interest Rate: 5.0%
   Interest Earned in One Year: $50.0
   Total Amount after One Year: $1050.0
   ```

   恭喜！你已经配置了一个基本工作区并运行了一个执行某些操作的 Julia 脚本。

### 了解 REPL 与脚本工作流程

你现在已经使用了 Julia REPL 和脚本文件。

- **REPL** 对于交互式使用、测试小型代码片段、快速计算以及获得即时反馈非常棒。可以把它当作你的 Julia 草稿本。
- \*\*脚本文件（.jl 文件）\*\*是你编写更持久、更大程序的地方。脚本允许你保存工作、组织复杂的逻辑，并反复运行你的代码。

随着你的学习进展，你会发现自己在两种模式之间切换。通常，你可能会在 REPL 中进行初步尝试，然后将完善后的代码转移到脚本中。

### 常见障碍快速检查

如果事情没有像预期的那样顺利，以下是几个常见的初始障碍：

- **`julia: command not found`（或类似错误）**：这通常意味着 Julia 的安装目录不在你的系统 PATH 中。重新检查你的安装步骤，特别是任何与将 Julia 添加到 PATH 相关的选项。安装或修改 PATH 后，你可能需要重启终端甚至电脑才能使更改生效。
- **`ERROR: could not open file simple_calc.jl`**：这通常意味着你的终端不在你保存 `simple_calc.jl` 的同一目录中。在运行 `julia simple_calc.jl` 之前，使用 `cd`（改变目录）等命令导航到正确的文件夹（在我们的例子中是 `julia_course_work`）。你可以在 Linux/macOS 中使用 `pwd`（打印工作目录）或在 Windows 命令提示符中使用 `cd`（不带参数 (parameter)）来查看当前目录。
- **语法错误**：如果 Julia 在运行脚本时报告错误，请仔细将你的脚本内容与示例进行比较。逗号位置不对、缺少括号或拼写错误都可能导致错误。Julia 的错误消息通常会指向它遇到问题的行号。

不要被初始错误吓倒。它们是编程的正常部分，学会理解和修正它们是一项有价值的技能。

### 使用 REPL 的帮助模式

REPL 有一个非常有用的内置帮助模式。你曾在“访问 Julia 文档和支持”中简要遇到过它。让我们再试一次。启动 Julia REPL：

```
julia
```

在 `julia>` 提示符下，输入 `?` 并按回车键。提示符将变为 `help?>`。现在，你可以输入函数名或说明来获取文档。例如，要了解 `sqrt` 函数（用于计算平方根）：

```
help?> sqrt
```

按回车键。你将看到 `sqrt` 的文档。

```julia
搜索: sqrt CbrtIEuclidean Isqrt

  sqrt(x)

  返回 $\sqrt{x}$。对于负实数参数会抛出 DomainError。使用 complex(x) 可获得复数结果。

  如果 x 是矩阵，则计算矩阵平方根。

  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

  示例
  ≡≡≡≡≡≡≡≡≡≡

  julia> sqrt(4.0)
  2.0

  julia> sqrt(big(81))
  9.0

  julia> sqrt(-1)
  错误: DomainError with -1.0:
  sqrt 只有在以复数参数调用时才会返回复数结果。尝试 sqrt(Complex(x))。
  堆栈跟踪:
  [...]
```

按退格键或 `q` 退出帮助分页器（如果它占据了整个屏幕），将你返回到 `help?>` 提示符。再按一次退格键（或 Ctrl+C）返回 `julia>` 提示符。现在你可以使用该函数了：

```julia
julia> sqrt(16)
4.0

julia> sqrt(2)
1.4142135623730951
```

这种交互式帮助是一个强大的工具，可让你直接在工作环境中了解 Julia 的功能。

### 展望未来

在你的环境配置完毕并完成了这些初始脚本编写步骤后，你已经为后续章节做好了充分准备。我们鼓励你随着学习进展，在 `julia_course_work` 文件夹中为每个章节或主题创建子目录。实验将帮助你熟练掌握 Julia，所以当你好奇或遇到困难时，不要犹豫修改脚本、在 REPL 中尝试新事物以及查阅文档。这种实践将提高你的熟练度。

## 参考资料

- [The Julia Language Manual](https://docs.julialang.org/en/v1/) — The Julia Language Developers (2024)
  提供关于 Julia 安装、交互式 REPL、脚本执行和基本编程语法的权威信息。
- [Julia Extension for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia) — Julia Computing and contributors (2024)
  在 Visual Studio Code 中设置和使用 Julia 扩展的文档，这是一个流行的 Julia 开发环境。

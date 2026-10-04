# 使用 Julia REPL

来源：[原文](https://apxml.com/zh/courses/getting-started-julia-programming/chapter-1-introducing-julia-setup-first-steps/navigating-julia-repl)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Julia REPL（读取-评估-打印循环）是您直接与 Julia 交互的主要工具。它是一个环境，您可以在其中输入 Julia 代码，让代码立即执行，并查看结果。这使得它非常适合尝试小段代码、学习语言以及进行快速计算。可以将其视为与 Julia 解释器直接对话。

要启动 REPL，请打开您的终端或命令提示符，输入 `julia`，然后按 Enter 键。如果 Julia 安装正确，并且其 `bin` 目录已添加到您的系统 PATH 中，您应该会看到一个包含 Julia 版本信息的横幅，后面跟着 Julia 提示符：

```
julia>
```

这个 `julia>` 提示符就是您输入命令的地方。让我们看看您可以做什么。

### 基本交互

REPL 已准备好评估 Julia 表达式。尝试一些简单的算术运算：

```
julia> 2 + 3
5

julia> 10 * 4 - 5
35
```

Julia 读取您的输入（`2 + 3`），评估它（计算和），打印结果（`5`），然后循环回显示提示符，准备好接收您的下一个命令。

您也可以定义变量：

```
julia> message = "Hello, Julia learners!"
"Hello, Julia learners!"

julia> x = 100
100

julia> y = x / 4
25.0
```

要查看变量的值，只需输入其名称即可：

```
julia> message
"Hello, Julia learners!"
```

REPL 方便地打印最后一次评估表达式的结果。如果一个表达式没有明确返回值（例如赋值，它在技术上会返回赋的值），REPL 会显示它。如果使用像 `println` 这样的函数，它主要执行一个动作（打印到控制台）并返回 `nothing`，REPL 不会为 `nothing` 额外打印一行，除非它是唯一的结果。

```
julia> println("This is a test.")
This is a test.
```

### 切换 REPL 特殊模式

Julia REPL 有几种特殊模式，通过在空行的开头输入特定字符即可进入。这些模式提供了强大的内置功能。

> Julia REPL 中可用的不同模式以及如何切换它们。

1. **帮助模式 (`?`)**
   如果您想了解更多关于某个函数、类型或模块的信息，请在空的 `julia>` 提示符下输入 `?` 并按 Enter 键。提示符会变为 `help?>`。现在，输入您感兴趣的名称：

   ```
   julia> ?
   help?> println
   search: println print printstyled sprint isprint islink issticky Sprintln

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
     Hello\n

   help?>
   ```

   这会显示 `println` 的文档（docstring）。要退出帮助模式并返回到正常的 `julia>` 提示符，请在空的 `help?>` 提示符下按 Backspace 键，或者直接在空行上按 Enter 键。
2. **Shell 模式 (`;`)**
   要直接从 REPL 运行 shell 命令（例如在 Linux/macOS 上是 `ls`，在 Windows 上是 `dir`），请在空的 `julia>` 提示符下输入 `;` 并按 Enter 键。提示符会变为 `shell>`。

   在 Linux 或 macOS 上：

   ```
   julia> ;
   shell> ls
   Desktop   Documents Downloads Music     Pictures  ProjectA
   shell> pwd
   /Users/yourusername
   shell>
   ```

   在 Windows 上：

   ```
   julia> ;
   shell> dir
    Volume in drive C is OS
    Volume Serial Number is XXXX-XXXX

    Directory of C:\Users\yourusername

   01/10/2023  09:00 AM    <DIR>          .
   01/10/2023  09:00 AM    <DIR>          ..
   ...
   shell>
   ```

   要退出 shell 模式，请在空的 `shell?>` 提示符下按 Backspace 键，或者输入 `exit` 并按 Enter 键。
3. **包管理模式 (`]`)**
   在空的 `julia>` 提示符下输入 `]` 以进入包管理模式。提示符会变为 `pkg>`。此模式用于管理 Julia 包（库），例如添加新包或更新现有包。

   ```
   julia> ]
   pkg> status
   Status `~/.julia/environments/v1.9/Project.toml` (empty)
   ```

   我们将在第 6 章详细介绍包管理。目前，您只需了解有这种模式即可。要退出包管理模式，请在空的 `pkg>` 提示符下按 Backspace 键或按 `Ctrl+C`。

### REPL 的实用功能

除了这些模式，REPL 还具有几个功能，使您的交互式会话更高效：

- **`ans` 变量**：最后一次成功评估表达式的结果始终存储在一个名为 `ans` 的特殊变量中。

  ```
  julia> 21 * 2
  42

  julia> ans
  42

  julia> ans + 8
  50
  ```
- **Tab 补全**：这是一个非常实用的功能。开始输入变量名、函数名、模块名，甚至是文件路径，然后按下 `Tab` 键。REPL 会尝试为您完成输入。如果有多种可能性，它会显示出来。

  ```
  julia> prin<Tab>
  print   print_shortest   print_unescaped   println   printstyled

  julia> pri<Tab>ntln("Tab completion is great!")
  Tab completion is great!
  ```
- **历史命令导航**：您可以使用向上 (`⬆`) 和向下 (`⬇`) 箭头键访问之前输入的命令。这对于重新运行命令或更正错别字而无需重新输入整行非常有用。

  - 您还可以通过输入 `Ctrl+R`，然后输入您正在查找的命令的一部分来搜索历史记录。
- **清屏**：要清除终端屏幕，通常可以使用 `Ctrl+L`。
- **多行输入**：如果您正在输入一个跨越多行的长命令，例如函数定义或循环，REPL 会自动检测并允许您继续在下一行输入，直到表达式完成。提示符可能会略微改变或缩进，以表示您正在多行块中。

  ```
  julia> for i in 1:3
             println("Iteration: ", i)
         end
  Iteration: 1
  Iteration: 2
  Iteration: 3
  ```

### 退出 REPL

当您完成 Julia 会话后，您可以通过几种方式退出 REPL：

- 输入 `exit()` 并按 Enter 键。
- 在 Linux 和 macOS 上，在空行上按 `Ctrl+D`。
- 在 Windows 上，在空行上按 `Ctrl+Z` 然后按 Enter 键（尽管在某些现代终端，如 Windows Terminal 中，`Ctrl+D` 也可能有效）。

REPL 是一个学习 Julia 的优秀环境。它提供即时反馈，允许快速测试想法，并帮助您熟悉 Julia 的语法和函数。在学习本课程的过程中，请务必多尝试不同的命令，并查看其功能。

## 参考资料

- [Interactive Use](https://docs.julialang.org/en/v1/manual/getting-started/#Interactive-Use) — The Julia Language Developers (2025)
  Julia官方交互式环境文档，涵盖其功能和模式。
- [Think Julia: How to Think Like a Computer Scientist](https://www.oreilly.com/library/view/think-julia/9781492045021/) — Ben Lauwens, Allen Downey (2019)
  Publisher: O'Reilly Media
  一本教授使用Julia编程的入门教材，早期章节专门介绍REPL和基本交互。

---

[上一节](04-%E5%9C%A8%E6%82%A8%E7%9A%84%E7%B3%BB%E7%BB%9F%E4%B8%8A%E5%AE%89%E8%A3%85%20Julia.md) · [下一节](06-%E9%80%89%E6%8B%A9%E6%82%A8%E7%9A%84Julia%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83.md)

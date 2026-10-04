---
course: "python-for-beginners"
chapter: "getting-started-with-python"
lesson: "using-python-interpreter-repl"
sourceId: 2249
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-1-getting-started-with-python/using-python-interpreter-repl"
title: "Python 解释器 (REPL) 的使用"
description: "学习如何使用 Python 读取-执行-打印-循环 (REPL) 来交互式运行 Python 代码片段。"
order: 7
plots: []
sourceHash: "38907c91564feb71ef9d436e46c4adc3462000046c341ebb6b309ceb62418898"
sourceCorrections: []
---

Python 的交互式解释器，通常称为 REPL，提供了一种直接与 Python 交互的方法。REPL 是 Read-Eval-Print Loop（读取-执行-打印-循环）的缩写，这准确描述了它的作用：它读取你输入的 Python 代码，执行它，（如有结果）打印出来，然后循环等待你的下一次输入。这是一个很好的环境，用于试验代码片段、检查语法或熟悉 Python 的功能，而无需创建文件。

### 什么是 REPL？

可以把 REPL 想象成你与 Python 解释器之间的直接对话。你输入一个命令，按下回车，Python 会立即响应。这种即时反馈循环对于学习和快速测试非常有帮助。

- **读取 (Read)：** 它读取你输入的代码行。
- **执行 (Eval)：** 它执行该代码。
- **打印 (Print)：** 它将执行结果打印到控制台。
- **循环 (Loop)：** 它等待你的下一次输入。

### 启动 REPL

通常，你可以通过打开终端或命令提示符，然后输入 `python`（或有时是 `python3`，取决于你的安装和操作系统）并按下回车键来启动 Python REPL。

在 **Windows** 上：

1. 打开命令提示符（搜索 `cmd`）或 PowerShell。
2. 输入 `python` 并按下回车。

在 **macOS** 上：

1. 打开终端（应用程序 > 实用工具 > 终端或使用聚焦搜索）。
2. 输入 `python3`（在 macOS 上通常更推荐，以区分旧的系统 Python 2）并按下回车。

在 **Linux** 上：

1. 打开终端应用程序。
2. 输入 `python3`（如果 `python3` 链接到 `python`，也可以输入 `python`）并按下回车。

如果一切设置正确，你应该会看到类似以下内容（具体的版本号和细节可能有所不同）：

```text
Python 3.10.4 (main, Mar 31 2022, 08:41:55) [GCC 7.5.0] on linux
输入 "help", "copyright", "credits" 或 "license" 以获取更多信息。
>>>
```

`>>>` 是 Python 提示符。它表示解释器已准备好接受你的命令。

### 使用 REPL

让我们直接在 `>>>` 提示符下尝试一些简单命令。

1. **基本算术：** 输入一个数学表达式并按下回车。Python 会立即计算它。

   ```python
   >>> 2 + 3
   5
   >>> 100 - 5 * 10
   50
   >>> (50 + 50) * 2 / 4
   50.0
   ```

   注意 Python 如何计算结果并在下一行显示它们。
2. **使用 `print()`：** `print()` 函数用于显示输出。

   ```python
   >>> print("Hello, Python!")
   Hello, Python!
   ```
3. **赋值变量：** 你可以像在脚本中一样创建变量。

   ```python
   >>> message = "Learning Python is fun"
   >>> print(message)
   Learning Python is fun
   >>> count = 10
   >>> count * 5
   50
   ```

   请注意，简单地赋值一个变量（`message = ...`）不会产生输出。但如果你直接输入变量名本身，REPL 通常会打印它的值：

   ```python
   >>> message
   'Learning Python is fun'
   >>> count
   10
   ```

   字符串 `'Learning Python is fun'` 周围的单引号表示其数据类型（一个字符串）。
4. **多行语句：** 对于像 `if` 语句或循环这样的结构，提示符会变为 `...`，表示 Python 期望更多输入来完成该代码块。在空行（只有 `...` 提示符）上按下回车，即可结束该代码块。

   ```python
   >>> name = "Alice"
   >>> if name == "Alice":
   ...     print("Hello, Alice!")
   ... else:
   ...     print("Hello, stranger!")
   ...
   Hello, Alice!
   >>>
   ```

### 退出 REPL

当你使用完交互式解释器后，可以通过以下几种方式退出它：

- 输入 `exit()` 并按下回车。
- 输入 `quit()` 并按下回车。
- 按下 `Ctrl+D`（在 Linux/macOS 上）或 `Ctrl+Z` 后再按下回车（在 Windows 上）。

你将返回到你的常规终端或命令提示符。

### 为什么要使用 REPL？

REPL 特别适合用于：

- **快速试验：** 无需创建文件即可测试小段代码或语法。
- **了解对象：** 使用 `dir()` 和 `help()` 等函数检查变量或对象上可用的方法或属性。（我们稍后会介绍这些）。
- **调试：** 快速运行一两行代码以了解其行为。
- **学习：** 获得 Python 语法和命令的即时反馈。

尽管你大部分程序都将写在脚本文件（`.py` 文件）中，但 REPL 在你的 Python 开发过程中仍然是一个有用的辅助工具。它是你尝试新事物的试验场。在下一节中，我们将继续编写并运行你的第一个实际 Python 脚本文件。

## 参考资料

- [The Python Tutorial](https://docs.python.org/3/tutorial/index.html) — Guido van Rossum and the Python development team (2024)
  官方指南，涵盖Python基础概念，包括对交互式解释器的详细介绍。
- [Python Crash Course, 3rd Edition: A Hands-On, Project-Based Introduction to Programming](https://nostarch.com/python-crash-course-3rd-edition/) — Eric Matthes (2022)
  Publisher: No Starch Press
  一本广受好评的入门书籍，利用REPL进行初步代码探索和测试。

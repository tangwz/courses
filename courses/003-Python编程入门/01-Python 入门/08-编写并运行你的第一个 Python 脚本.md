# 编写并运行你的第一个 Python 脚本

来源：[原文](https://apxml.com/zh/courses/python-for-beginners/chapter-1-getting-started-with-python/first-python-script)

[返回章节目录](README.md) · [返回课程目录](../README.md)

对于复杂的 Python 编程任务，将代码写入文件（通常称为脚本）是基础。脚本能够保存工作、多次运行代码，并构建更复杂的应用程序。虽然交互式解释器 (REPL) 对于快速测试和查看命令功能很有用，但它不适用于大型项目。

让我们来创建并运行你的第一个 Python 脚本。

### 什么是 Python 脚本？

Python 脚本就是一个普通的文本文件，包含 Python 代码。按照惯例，这些文件都带有 `.py` 扩展名（例如，`my_program.py` 或 `data_processor.py`）。Python 解释器会读取这个文件，并从上到下执行其中编写的命令。

### 创建你的脚本文件

你可以使用任何纯文本编辑器来编写 Python 脚本。简单的编辑器，如记事本（Windows）、TextEdit（macOS）或 gedit（Linux）都可以用。然而，正如在关于 IDE 和代码编辑器的章节中提到的，使用专门的代码编辑器（如 VS Code、Sublime Text、Atom）或集成开发环境 IDE（如 PyCharm、Spyder）提供了有用的功能，比如语法高亮和代码补全，当你的程序变得更复杂时，这些功能会非常有用。

对于这个第一个例子，让我们保持简单直接。

1. **打开你选择的文本编辑器或 IDE。**
2. **创建一个新文件。**
3. **将下面这行代码输入到文件中：**

   ```python
   print("Hello, Python!")
   ```

我们来分析一下这行代码：

- `print` 是一个内置的 Python 函数。函数是可重用的代码块，用来完成特定任务。`print()` 函数的任务是将输出显示到控制台或终端窗口。
- 文本 `"Hello, Python!"` 被称为字符串。字符串是用引号（单引号 `'` 或双引号 `"`）括起来的字符序列。我们将这个字符串作为 *参数 (parameter)* 传递给 `print()` 函数，告诉它我们想要显示什么内容。
- `print` 后面的括号 `()` 用来调用函数，并包含传递给它的任何参数。

### 保存脚本

现在，保存文件。

1. 选择 `文件 -> 保存` 或使用保存快捷键（如 Ctrl+S 或 Cmd+S）。
2. 选择一个你容易找到文件的地方。你的桌面或一个专门用于 Python 练习的文件夹（例如，`python_scripts`）都是不错的选择。
3. **将文件命名为 `hello.py`。** 确保扩展名是 `.py`。有些简单的文本编辑器可能会自动添加 `.txt`；确保它被准确保存为 `hello.py`。

### 从终端运行脚本

你不能通过双击 `.py` 文件来运行脚本。相反，你需要通过命令行或终端使用 Python 解释器。

1. **打开你的终端或命令提示符。**

   - **Windows：** 搜索 `cmd` 或 `PowerShell`。
   - **macOS：** 打开 `终端`（通常在“应用程序”->“实用工具”中）。
   - **Linux：** 打开你的发行版终端应用程序（例如，`gnome-terminal`，`konsole`）。
2. **切换到你保存 `hello.py` 的目录。** 使用 `cd`（change directory，更改目录）命令。例如：

   - 如果你保存到桌面：`cd Desktop`
   - 如果你保存到用户主目录中一个名为 `python_scripts` 的文件夹里：`cd python_scripts`（macOS/Linux 上）或 `cd Documents\python_scripts`（在 Windows 上根据需要调整路径）。
   - 你可以使用 `ls`（macOS/Linux）或 `dir`（Windows）命令列出当前目录中的文件，以确认 `hello.py` 是否存在。
3. **运行脚本。** 输入以下命令并按回车键：

   ```bash
   python hello.py
   ```

   *注意：* 根据你的安装情况（特别是如果你安装了多个 Python 版本），你可能需要改用 `python3`：

   ```bash
   python3 hello.py
   ```
4. **观察输出。** 你应该会在终端中你的命令下方直接看到文本 "Hello, Python!"：

   ```
   Hello, Python!
   ```

### 发生了什么？

当你执行 `python hello.py` 时，你指示 Python 解释器（`python` 或 `python3` 程序）执行以下操作：

1. 找到名为 `hello.py` 的文件。
2. 读取文件内容。
3. 逐行执行 Python 代码。在这个例子中，它执行了 `print()` 函数，将指定的字符串显示到你的终端。

恭喜！你已经编写并运行了你的第一个 Python 脚本。这个简单的“Hello, Python!”程序是学习许多编程语言的传统第一步。它确认你的设置工作正常，并介绍了从文件创建和运行代码的基本过程，这是大多数 Python 应用程序的开发方式。你现在已经准备好了解更多关于 Python 编程的组成部分了。

## 参考资料

- [The Python Tutorial](https://docs.python.org/3/tutorial/) — Guido van Rossum and the Python Development Team (2024)
  Publisher: Python Software Foundation
  介绍Python语言及其核心功能的官方指南，包括如何从命令行运行脚本和使用解释器。
- [Think Python: How to Think Like a Computer Scientist, 3rd Edition](https://greenteapress.com/wp/think-python/) — Allen B. Downey (2024)
  Publisher: O'Reilly Media
  一本全面易懂的Python编程基础教材，涵盖脚本创建、函数和基本数据类型等主题。

---

[上一节](07-Python%20%E8%A7%A3%E9%87%8A%E5%99%A8%20%28REPL%29%20%E7%9A%84%E4%BD%BF%E7%94%A8.md) · [下一节](09-%E9%9B%86%E6%88%90%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E4%B8%8E%E4%BB%A3%E7%A0%81%E7%BC%96%E8%BE%91%E5%99%A8%E7%AE%80%E4%BB%8B.md)

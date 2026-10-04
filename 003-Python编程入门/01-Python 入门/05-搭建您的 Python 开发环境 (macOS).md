# 搭建您的 Python 开发环境 (macOS)

来源：[原文](https://apxml.com/zh/courses/python-for-beginners/chapter-1-getting-started-with-python/python-setup-macos)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在您的 macOS 电脑上设置 Python 环境是开发的基础。建立一个合适的开发环境是编写和运行您自己的 Python 程序的第一项实际步骤。

macOS 通常预装了较旧版本的 Python，主要是为了兼容旧版系统脚本。尽管这很方便，但此版本可能不是您进行日常开发所需要的。我们强烈建议直接从 Python 官方网站安装最新的稳定版本，以确保您可以使用最新功能、安全更新，并让不同项目保持一致的行为。

### 查看您当前的 Python 版本（可选）

安装之前，您可以查看系统上是否已安装 Python 3 以及其版本。

1. **打开终端：** 您可以在 `Applications` 文件夹中的 `Utilities` 子文件夹中找到终端应用程序。快速打开它的方法是使用聚焦搜索（Cmd + 空格键）并输入“Terminal”。
2. **检查 Python 3：** 在终端窗口中，输入以下命令并按回车键：

   ```bash
   python3 --version
   ```

   如果 Python 3 已正确安装并配置，您将看到类似 `Python 3.x.y` 的输出（其中 `x` 和 `y` 是版本号）。如果您收到“command not found”错误，或者显示的版本非常旧，请继续执行下面的安装步骤。
3. **检查 Python 2（旧版）：** 您可能还安装了 Python 2。您可以通过以下命令检查：

   ```bash
   python --version
   ```

   您可能会看到 `Python 2.7.x`。对于现代 Python 开发，请务必 *不要* 使用此版本。始终使用 `python3` 命令（或者在安装官方版本后只使用 `python`，正如我们将看到的）。

即使 `python3 --version` 显示的是最新版本，使用官方安装程序也能确保您拥有许多工具和教程所预期的标准配置。

### 下载 Python 官方安装程序

1. **访问官方网站：** 打开您的网页浏览器，访问 Python 官方网站：<https://www.python.org>
2. **前往下载页面：** 将鼠标悬停在“Downloads”菜单上。它应该会自动检测您正在使用 macOS，并建议下载最新的稳定版本。
3. **下载安装程序：** 点击标有“Download Python 3.x.y”字样的按钮。这将下载一个 `.pkg` 文件（例如 `python-3.11.5-macos11.pkg`）。下载最新的稳定版本；除非有特定原因，否则请避免使用预发布或测试版本。

### 运行安装程序

1. **打开 .pkg 文件：** 找到下载的 `.pkg` 文件（通常在您的 `Downloads` 文件夹中），双击它以启动安装向导。
2. **遵循安装步骤：**
   - 您将看到一个介绍屏幕。点击 **继续**。
   - 阅读“重要信息”（Read Me）部分。点击 **继续**。
   - 查看软件许可协议。点击 **继续**，如果您接受条款，则点击 **同意**。
   - 在“安装类型”屏幕上，您通常只需点击 **安装**。它会显示所需的磁盘空间和安装位置（通常是 `/Library/Frameworks/Python.framework`）。您通常无需自定义位置。
   - 系统可能会提示您输入管理员密码以授权安装。输入密码并点击 **安装软件**。
   - 安装程序将复制必要的文件。完成后，您应该会看到一个确认屏幕，表示安装成功。它通常包含有关已安装组件的信息，包括 IDLE（一个简单的集成开发环境）以及关于 PATH 的说明。点击 **关闭**。如果系统提示，您可以将安装程序文件移至废纸篓。

安装程序通常会配置您的系统，使 `python3` 命令指向新安装的版本。

### 验证安装

确认安装成功并且系统能识别新的 Python 版本，这很要紧。

1. **打开一个 *新的* 终端窗口：** 关闭所有现有终端窗口，然后打开一个新的。这可确保对系统配置（例如更新 PATH）所做的任何更改都能正确加载。
2. **检查 Python 3 版本：** 输入以下命令并按回车键：

   ```bash
   python3 --version
   ```

   您现在应该能看到刚安装的特定版本号（例如 `Python 3.11.5`）。如果您仍然看到旧版本或遇到错误，请仔细检查安装步骤或查阅 Python 文档进行故障排除。
3. **检查 pip 版本：** `pip` 是 Python 的包安装程序，用于添加库和工具。它包含在 Python 官方安装程序中。通过运行以下命令来验证它：

   ```bash
   pip3 --version
   ```

   这应该会输出您的 Python 安装捆绑的 pip 版本及其位置。我们将在课程后面使用 `pip` 安装外部包。

### 使用 Python 解释器 (REPL)

Python 安装完成后，您现在可以使用其“读取-评估-打印”循环 (REPL) 直接与其交互。这是尝试简单代码片段的好方法。

1. **启动 REPL：** 在终端窗口中，只需输入 `python3` 并按回车键。

   ```bash
   python3
   ```
2. **查看提示符：** 您将看到一些关于 Python 版本的信息，然后是解释器提示符：`>>>`。这表示 Python 正在等待您输入命令。
3. **运行命令：** 输入一个简单的 Python 命令，例如打印消息，然后按回车键：

   ```python
   >>> print("Hello from Python on macOS!")
   ```

   Python 将立即执行该命令并显示输出：

   ```
   Hello from Python on macOS!
   >>>
   ```

   `>>>` 提示符再次出现，等待您的下一个命令。
4. **退出 REPL：** 要离开交互式解释器并返回常规终端提示符，您可以输入 `exit()` 并按回车键，或按 `Ctrl+D`。

   ```python
   >>> exit()
   ```

恭喜！您已成功在 macOS 系统上安装 Python，验证了安装，并与 Python REPL 进行了交互。您现在拥有了一个可用的环境，可以准备编写和运行您的第一个 Python 脚本了，我们很快就会讲到这些。

## 参考资料

- [Python Downloads for macOS](https://www.python.org/downloads/macos/) — Python Software Foundation (2024)
  下载macOS最新稳定版Python安装程序及相关安装信息的官方来源。
- [pip documentation](https://pip.pypa.io/en/stable/) — The pip developers (2024)
  pip的官方文档，它是Python的包安装程序，对于管理Python库和工具至关重要。
- [Python Crash Course, 3rd Edition](https://nostarch.com/python-crash-course-3rd-edition/) — Eric Matthes (2022)
  Publisher: No Starch Press
  一本广受欢迎的Python编程入门书籍，包含有关搭建开发环境的实用指南。

---

[上一节](04-%E8%AE%BE%E7%BD%AE%E6%82%A8%E7%9A%84%20Python%20%E7%8E%AF%E5%A2%83%20%28Windows%29.md) · [下一节](06-%E6%90%AD%E5%BB%BA%20Python%20%E7%8E%AF%E5%A2%83%20%28Linux%29.md)

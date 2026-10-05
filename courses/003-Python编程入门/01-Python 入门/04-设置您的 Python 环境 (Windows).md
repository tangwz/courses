# 设置您的 Python 环境 (Windows)

来源：[原文](https://apxml.com/zh/courses/python-for-beginners/chapter-1-getting-started-with-python/python-setup-windows)

[返回章节目录](README.md) · [返回课程目录](../README.md)

要开始 Python 编程，在您的计算机上设置 Python 环境是至关重要的第一步。本文将指导您如何在 Windows 操作系统上安装 Python。得益于 Python 软件基金会提供的用户友好型安装程序，整个过程非常简单。

### 下载 Python 安装程序

首先，您需要获取 Windows 系统的官方 Python 安装程序。

1. **访问 Python 官方网站：** 打开您的网络浏览器并访问 [python.org](https://www.python.org/)。这里是所有 Python 相关内容的中心。
2. **找到下载区：** 将鼠标悬停在“Downloads”（下载）菜单上。您应该会看到一个专门针对 Windows 的按钮或链接，通常会推荐最新的稳定版本（例如：“Download for Windows Python 3.x.x”）。
3. **下载安装程序：** 点击按钮下载推荐的安装程序。这通常是一个 `.exe` 文件。网站通常会检测您的 Windows 是 32 位还是 64 位版本，并提供合适的安装程序。如果有选择，请为现代系统选择 64 位安装程序，除非您有特殊原因需要使用 32 位版本。

### 运行安装过程

下载完成后，找到 `.exe` 文件（通常在您的“下载”文件夹中），然后双击它以开始安装。

1. **启动安装程序：** Windows 用户账户控制可能会提示您允许此应用对您的设备进行更改。点击“是”。
2. **配置安装选项：** 安装程序的第一个屏幕会显示一些重要的选择。

   - **将 Python 添加到 PATH：** 查找底部标有“Add Python 3.x to PATH”或“Add python.exe to PATH”之类的复选框。**强烈建议您勾选此框。** 选中此选项后，您可以直接从 Windows 命令提示符或 PowerShell 的任何目录运行 Python 命令。如果不勾选此项，每次都需要输入 Python 可执行文件的完整路径，这很不方便。我们将在下面简要说明 PATH 是什么。
   - **选择安装类型：** 您通常会看到两个主要选项：
     - `立即安装`：这是推荐给初学者的选项。它会将 Python 以及默认设置安装到标准用户目录中，并包含 IDLE（一个简单的开发环境）、pip（包安装程序）和文档。
     - `自定义安装`：这允许您选择特定功能、更改安装位置以及配置其他高级选项。对于本课程，默认的 `立即安装` 就足够了。
   > **什么是 PATH？**
   > PATH 是 Windows（以及其他操作系统）上的一个环境变量。它包含一个目录列表。当您在命令提示符中输入 `python` 等命令时，Windows 会在 PATH 变量列出的目录中查找名为 `python.exe` 的可执行文件。在安装过程中勾选“Add Python to PATH”会自动将 Python 的安装目录添加到此列表中，从而使 Python 易于使用。
3. **继续安装：** 点击 `立即安装`（请确保已勾选“Add Python to PATH”复选框）。安装程序在复制文件和设置 Python 时会显示进度条。
4. **设置成功：** 安装完成后，您应该会看到“Setup was successful”（设置成功）消息。您可能还会看到一个“Disable path length limit”（禁用路径长度限制）的选项。点击此选项有助于避免在某些开发场景中与长文件路径相关的潜在问题。如果有此选项，通常建议禁用此限制，这样做是安全的。
5. **关闭安装程序：** 您现在可以关闭安装程序窗口了。

### 验证您的 Python 安装

为确保 Python 已正确安装并可访问，您需要使用命令提示符或 PowerShell 进行验证。

1. **打开命令提示符或 PowerShell：**
   - 按下 Windows 键，输入 `cmd`，然后按 Enter 键打开命令提示符。
   - 或者，按下 Windows 键，输入 `powershell`，然后按 Enter 键打开 PowerShell。两者都可用于验证。
2. **检查 Python 版本：** 在终端窗口中，输入以下命令并按 Enter 键：

   ```bash
   python --version
   ```

   您也可以尝试：

   ```bash
   py --version
   ```

   如果安装成功且 Python 已添加到 PATH，您应该会看到类似以下的输出（版本号可能有所不同）：

   ```
   Python 3.11.4
   ```
3. **检查 pip 版本：** Pip 是 Python 的包管理器，用于安装额外的库。它会随 Python 自动安装。通过输入以下命令进行验证：

   ```bash
   pip --version
   ```

   您应该会看到显示 pip 版本及其位置的输出，例如：

   ```
   pip 23.1.2 from C:\Users\YourUsername\AppData\Local\Programs\Python\Python311\Lib\site-packages\pip (python 3.11)
   ```

**故障排除：** 如果您输入 `python --version` 并收到类似“'python' 不是内部或外部命令...”的错误消息，最常见的原因是安装时未将 Python 添加到 PATH 环境变量中。最简单的办法通常是通过 Windows 设置 > 应用卸载 Python，然后重新安装，并确保这次勾选“Add Python to PATH”框。

恭喜！您已成功在 Windows 系统上安装了 Python，并验证了其可用性。您现在可以开始与 Python 解释器交互或编写您的第一个脚本了，这将在下一部分介绍。

## 参考资料

- [Using Python on Windows](https://docs.python.org/3/using/windows.html) — Python Software Foundation (2024)
  在Windows上安装和配置Python的官方指南，详细介绍了安装步骤、环境设置及其他重要信息。
- [Installing packages](https://packaging.python.org/en/latest/guides/installing-packages/) — Python Packaging Authority (2025)
  Publisher: Python Packaging Authority
  解释如何使用pip安装和管理Python包的标准资源，pip是Python开发中不可或缺的工具。
- [Python Crash Course, 2nd Edition: A Hands-On, Project-Based Introduction to Programming](https://nostarch.com/pythoncrashcourse2e/) — Eric Matthes (2022)
  Publisher: No Starch Press
  一本面向初学者的易懂编程书籍，其中包含专门介绍设置Python环境的部分，也覆盖了Windows系统。
- [Add an environment variable to your PATH for command-line control](https://learn.microsoft.com/en-us/windows/terminal/tutorials/add-to-path) — Microsoft (2023)
  Publisher: Microsoft
  微软官方的说明，解释Windows环境变量，特别是PATH变量的功能及调整方法。

---

[上一节](03-%E7%90%86%E8%A7%A3%E8%A7%A3%E9%87%8A%E5%9E%8B%E8%AF%AD%E8%A8%80%E4%B8%8E%E7%BC%96%E8%AF%91%E5%9E%8B%E8%AF%AD%E8%A8%80.md) · [下一节](05-%E6%90%AD%E5%BB%BA%E6%82%A8%E7%9A%84%20Python%20%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%20%28macOS%29.md)

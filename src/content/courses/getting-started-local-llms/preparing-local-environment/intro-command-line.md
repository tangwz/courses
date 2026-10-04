---
course: "getting-started-local-llms"
chapter: "preparing-local-environment"
lesson: "intro-command-line"
sourceId: 4227
sourceUrl: "https://apxml.com/zh/courses/getting-started-local-llms/chapter-2-preparing-local-environment/intro-command-line"
title: "命令行/终端简介"
description: "对命令行或终端的简要介绍，这对于一些配置步骤是必需的。"
order: 7
plots: []
sourceHash: "fc8624cbba17166e4914e06da2e009b99b8bdd4a12edfbc5d1d003accc73ee9e"
sourceCorrections: []
---

虽然许多现代应用程序提供图形用户界面（GUI）进行交互，但命令行界面（CLI），通常被称为“终端”或“shell”，对于开发者以及越来越多与更高级软件（包括一些本地大型语言模型工具）交互的用户来说，仍然是一个重要工具。掌握其基本知识可以大大简化本地大型语言模型环境的配置和管理过程。

可以将命令行看作是与计算机进行基于文本的对话。您无需点击图标和菜单，而是输入特定的指令，计算机则会以文本形式给出回应或执行相应动作。尽管初看起来可能有些令人生畏，但掌握一些基本指令就能获得对系统的直接而强大的控制。

### 为什么需要命令行？

运行本地大型语言模型时，熟悉命令行有诸多益处：

1. **安装:** 许多大型语言模型工具、框架（如 `llama.cpp`）以及支持软件（如 Python 及其包）都是通过命令行指令安装或管理的。Ollama 等工具在模型首次下载和运行方面，非常依赖它。
2. **直接操控:** 命令行通常提供比图形用户界面更直接的程序功能和配置访问方式。
3. **自动化:** 重复性任务常可利用简单的命令行脚本实现自动化。
4. **问题排查:** 错误消息和日志常在终端显示，提供重要的诊断信息。
5. **普遍性:** 它是跨不同操作系统（Windows、macOS、Linux）的通用界面，因此在一个系统上学到的技能通常可在其他系统上使用。

即使您计划主要使用像 LM Studio 这样的图形用户界面工具，在配置或排查问题时遇到命令行指令也是常见情况。

### 查找您的终端

访问命令行的方法因操作系统的不同而略有差异：

- **Windows:** 您有几种选择。
  - **命令提示符 (Command Prompt):** 在开始菜单中搜索 `cmd` 或“命令提示符”。
  - **PowerShell:** 在开始菜单中搜索 `PowerShell`。它是命令提示符的更现代、更强大的替代品。
  - **Windows 终端 (Windows Terminal):** 微软推出的一款新应用程序（通常预装在 Windows 11 中，或可从 Microsoft Store 获取），它可以在选项卡中托管命令提示符、PowerShell 和 WSL（适用于 Linux 的 Windows 子系统）会话。这通常是推荐的选择。
- **macOS:** 终端应用程序就叫做 **终端 (Terminal)**。
  - 前往 `应用程序` > `实用工具` 文件夹，双击 `终端.app`。
  - 或者，使用 Spotlight 搜索 (Cmd + 空格)，输入 `Terminal`，然后按回车。
- **Linux:** 名称因桌面环境而异，但常见的名称包括：
  - **终端 (Terminal)**
  - **Gnome 终端 (Gnome Terminal)** (适用于 GNOME 环境)
  - **Konsole** (适用于 KDE 环境)
  - **xterm** (一个更基础的选择)
  - 您通常可以在应用程序菜单的“系统工具”或类似类别中找到它，或者通过搜索“终端”找到。

打开后，您通常会看到一个窗口，其中有一个闪烁光标，旁边是所谓的**提示符**。提示符表明系统已准备好接收您的指令。它通常会显示您的用户名、计算机名称（主机名）以及您在文件系统中的当前位置（目录），例如 Windows 上的 `C:\Users\您的用户名>` 或 macOS/Linux 上的 `username@hostname:~$`。

### 主要组成部分：指令、参数 (parameter)和路径

与终端交互涉及输入指令并按回车键。指令就是程序或内置 shell 功能的名称。

- **指令 (Commands):** 您希望执行的操作（例如 `cd`、`ls`、`ollama`）。
- **参数 (Arguments):** 指令运行所需附加信息（例如文件名、目录名、URL）。
- **选项 (Options，又称标志/开关):** 改变指令行为的修饰符，通常以连字符（`-`）或双连字符（`--`）开头（例如 `ls -l` 以显示详细列表）。

文件系统导航是一个重要内容。您的计算机将文件组织在目录（文件夹）中。

- **`pwd` (Print Working Directory，显示当前工作目录):** 在 macOS 和 Linux 上，此指令会告诉您当前所在的目录位置。（在 Windows 命令提示符中，提示符本身通常会显示路径，或者您可以不带参数键入 `cd`）。
- **`ls` (List，列出):** 在 macOS 和 Linux 上，这会列出当前目录中的文件和子目录。
  - Example: `ls`
- **`dir` (Directory，目录):** 在 Windows 命令提示符中，这会列出文件和子目录。
  - Example: `dir`
- **`cd` (Change Directory，更改目录):** 此指令可让您进入不同目录。您需要将目录路径作为参数提供。
  - Example (macOS/Linux): `cd Documents` (进入 `Documents` 子目录)
  - Example (Windows): `cd Documents`
  - Example (向上移动一级): `cd ..` （`..` 表示父目录）

让我们快速尝试一个例子。打开您的终端，输入指令以列出当前目录中的文件（macOS/Linux 为 `ls`，Windows 为 `dir`），然后按回车键。

```bash
# 在 macOS 或 Linux 上
ls
```

```batch
:: 在 Windows 命令提示符上
dir
```

您应该会看到屏幕上打印出文件和文件夹的列表。现在尝试进入一个目录，例如您的 Documents 目录（假设它直接位于您当前位置下）：

```bash
# 在 macOS 或 Linux 上
cd Documents
pwd # 确认位置已改变
ls  # 列出 Documents 中的文件
```

```batch
:: 在 Windows 命令提示符上
cd Documents
cd  # 确认位置已改变（打印当前路径）
dir # 列出 Documents 中的文件
```

如果这看起来不熟悉，请不用担心。本课程中，您无需成为命令行专家。我们将为安装 Ollama 或 Python 等任务提供所需的具体指令。这里的目的只是向您介绍这个重要工具，并让未来的步骤不那么陌生。可以把它看作是在开始一个项目之前，先了解工作室里基本工具有哪些。

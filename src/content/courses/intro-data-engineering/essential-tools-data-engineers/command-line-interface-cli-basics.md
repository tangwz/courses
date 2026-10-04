---
course: "intro-data-engineering"
chapter: "essential-tools-data-engineers"
lesson: "command-line-interface-cli-basics"
sourceId: 5597
sourceUrl: "https://apxml.com/zh/courses/intro-data-engineering/chapter-6-essential-tools-data-engineers/command-line-interface-cli-basics"
title: "命令行界面 (CLI) 基本操作"
description: "学习使用命令行与系统交互的基本命令。"
order: 3
plots: []
sourceHash: "b22bd06d12f83b9cd162c0b421d7e6d31de9b9a0a72d1d671e431d8a0f37df72"
sourceCorrections: []
---

尽管图形用户界面 (GUI) 提供了通过窗口、图标和菜单与计算机进行视觉化交互的方式，但数据工程中的许多工作是在幕后通过一种基于文本的方法进行的，这便是命令行界面 (CLI)。可以将 CLI 视为与操作系统进行的直接对话，您输入命令，系统则以文本形式给出回应。

### 为何使用命令行？

对于数据工程师来说，CLI 因以下几个原因而不可或缺：

1. **自动化与脚本编写：** 处理日志文件或部署代码等重复性任务，可以通过编写执行一系列 CLI 命令的脚本来轻松实现自动化。
2. **远程服务器访问：** 数据基础设施通常运行在远程服务器上（位于其他地方的机器，可能在数据中心或云端）。连接和管理这些服务器的标准方式是通过 CLI，通常会用到像 SSH (安全外壳协议) 这样的工具。
3. **资源效率：** CLI 通常比 GUI 消耗更少的系统资源（内存、CPU），这在处理可能资源受限的服务器时很重要。
4. **工具可用性：** 许多强大的数据处理工具和实用程序主要设计为从命令行使用，有时甚至完全没有图形用户界面。
5. **精确与控制：** CLI 对操作提供细粒度控制，其程度通常超出视觉界面所能提供的。

### 访问您的终端

您访问 CLI 的方式取决于您的操作系统：

- **macOS：** 使用内置的 **终端 (Terminal)** 应用程序（可在“应用程序”>“实用工具”中找到）。
- **Linux：** 有各种终端模拟器可用，通常命名为 **Terminal**、**Konsole** 或 **xterm**。
- **Windows：** 您可以使用 **命令提示符 (Command Prompt)** (cmd.exe) 或更现代、功能更强的 **PowerShell**。适用于 Linux 的 Windows 子系统 (WSL) 也是在 Windows 上直接运行 Linux 环境（包括其 CLI 工具）的常用选项。

当您打开终端窗口时，会看到一个**提示符**。这是您输入命令的地方。它通常包含您的用户名和当前目录等信息，通常以 `$`、`%` 或 `>` 等符号结尾（如果您拥有管理权限，则为 `#`）。

```text
username@hostname:~$ _
```

### 文件系统导航

您的计算机以分层结构组织文件和文件夹（也称为目录），就像树上的分支一样。CLI 提供了在此结构中移动的命令。

> 典型的 Linux/macOS 目录结构简化视图。

以下是基本导航命令：

- **`pwd` (打印工作目录):** 显示您当前所在目录的完整路径。

  ```bash
  pwd
  ```

  输出可能为 `/home/username` 或 `/Users/username`。
- **`ls` (列出):** 列出您当前目录中的文件和目录。

  ```bash
  ls
  ```

  常用有用选项（以 `-` 开头的参数 (parameter)）：

  - `ls -l`：显示详细（“长”）列表，包括权限、所有者、大小和修改日期。
  - `ls -a`：显示所有文件，包括隐藏文件（以点 `.` 开头的文件）。您可以组合选项：`ls -la`。
- **`cd` (更改目录):** 用于进入不同的目录。

  ```bash
  # 进入名为 'documents' 的目录
  cd documents

  # 返回上一级目录（父目录）
  cd ..

  # 直接进入您的主目录
  cd ~
  # 或者直接输入 cd，不带任何参数
  cd

  # 进入根目录
  cd /
  ```

### 处理文件和目录

数据工程师经常处理文件，并需要组织它们。

- **`mkdir` (创建目录):** 创建一个新目录。

  ```bash
  mkdir my_new_project
  ```
- **`touch`：** 创建一个空文件或更新现有文件的修改时间。

  ```bash
  touch script.py
  ```
- **`cp` (复制):** 复制文件或目录。

  ```bash
  # 复制文件
  cp source_file.txt destination_file.txt

  # 复制文件到目录
  cp important_data.csv data_backup/

  # 复制整个目录（需要 -r 选项进行递归复制）
  cp -r project_folder project_folder_backup
  ```
- **`mv` (移动):** 将文件或目录移动到不同位置，如果目标在同一目录中，则对其进行重命名。

  ```bash
  # 重命名文件
  mv old_name.txt new_name.txt

  # 移动文件到目录
  mv report.pdf documents/
  ```
- **`rm` (删除):** 删除文件。**谨慎使用！从 CLI 删除的文件通常无法恢复。**

  ```bash
  # 删除文件
  rm temporary_file.txt

  # 删除空目录
  rmdir empty_directory

  # 删除目录及其所有内容（需要 -r 进行递归删除，请小心使用！）
  rm -r directory_to_delete
  ```

### 查看文件内容

快速查看文件内容是一项常见任务。

- **`cat` (连接):** 显示一个或多个文件的全部内容。最适合小文件。

  ```bash
  cat config.txt
  ```
- **`less`：** 一次显示一屏幕的文件内容。使用箭头键或 Page Up/Down 键进行导航，按 `q` 退出。非常适合大文件。

  ```bash
  less large_log_file.log
  ```
- **`head`：** 显示文件的前几行（默认 10 行）。

  ```bash
  head data.csv
  ```
- **`tail`：** 显示文件的最后几行（默认 10 行）。用于查看最近的日志条目。使用 `tail -f` 可实时查看文件新增内容。

  ```bash
  tail error.log
  ```

### 组合命令：管道与重定向

CLI 的两个强大功能是管道和重定向：

- **管道 (`|`)：** 将一个命令的输出作为另一个命令的输入。这允许您将命令串联起来。

  ```bash
  # 列出文件，然后过滤包含 '.py' 的行
  ls -l | grep .py
  ```
- **重定向 (`>` 和 `>>`)：** 将命令的输出发送到文件，而不是屏幕。

  ```bash
  # 用 ls 的输出覆盖 file_list.txt
  ls > file_list.txt

  # 将 ls 的输出追加到 file_list.txt
  ls >> file_list.txt
  ```

### 为何这对数据工程很重要

尽管这些命令看起来很基础，但它们是许多数据工程任务的构建模块。您将使用 CLI 来：

- 导航包含数据或代码的服务器目录。
- 运行用于数据处理的 Python 脚本或编译程序。
- 检查日志文件中的错误。
- 管理配置文件。
- 与云提供商的 CLI（如 AWS CLI、gcloud、az cli）交互以管理云资源。
- 自动化部署和工作流执行。

掌握命令行是一项基本技能。它提供对数据所在和处理系统的直接访问，所实现的效率和自动化远远超出 GUI 单独能提供的。从练习这些基本命令开始吧；它们将很快变得得心应手。

## 参考资料

- [The Missing Semester of Your CS Education](https://missing.csail.mit.edu/) — Anish Athalye, Jon Gjengset, and Jose Javier Gonzalez Ortiz (2020)
  Publisher: MIT
  一门介绍实用计算机科学工具的课程，包含对shell的广泛讲解。
- [GNU Bash Reference Manual](https://www.gnu.org/software/bash/manual/bash.html) — Chet Ramey and Brian Fox (2025)
  Publisher: GNU
  GNU Bash shell的官方详细参考手册，该shell在许多操作系统中得到广泛使用。

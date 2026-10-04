# 搭建 Python 环境 (Linux)

来源：[原文](https://apxml.com/zh/courses/python-for-beginners/chapter-1-getting-started-with-python/python-setup-linux)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Linux 发行版通常预装了 Python，因为许多系统工具都依赖它运行。但是，预装版本可能不是最新版，或者你的项目可能需要特定版本。检查你的 Python 安装，并在必要时安装或更新 Python 是重要的。

### 检查你当前的 Python 版本

首先，打开你的终端应用程序。这是你与系统进行交互的命令行界面。你通常可以在应用程序菜单中搜索“终端”来找到它。

终端打开后，你可以检查 Python 3（当前标准）是否已安装以及你拥有的版本。输入以下命令并按回车键：

```bash
python3 --version
```

如果 Python 3 已安装，你将看到类似这样的输出（确切的版本号可能不同）：

```
Python 3.10.4
```

如果出现“命令未找到”之类的错误，则 Python 3 很可能未安装，或者不在你的系统 PATH 中（系统查找可执行程序的目录列表）。

你也可以尝试：

```bash
python --version
```

在一些较旧的系统或配置不同的系统上，`python` 可能指向 Python 2，它已不再受支持，不应用于新的开发。如果此命令显示的版本以 `2.x.x` 开头，你的工作应明确使用 `python3` 命令。如果显示 `3.x.x` 版本，那么 `python` 和 `python3` 可能在你系统上是同一安装的别名。为了确保一致性和清晰度，尤其是在初学阶段，通常最稳妥的做法是明确使用 `python3` 命令。

### 使用发行版包管理器安装 Python

在 Linux 上安装或更新 Python 的推荐方式是通过你的发行版内置的包管理器。这可以确保 Python 与系统其他部分正确集成，并妥善处理依赖项。以下是一些常用 Linux 发行版的命令：

**对于 Debian、Ubuntu 及其衍生版（如 Linux Mint）：**

1. 首先，更新你的软件包列表以获得最新的可用版本：

   ```bash
   sudo apt update
   ```

   `sudo` 表示“superuser do”（超级用户执行），它赋予后续命令管理员权限。你很可能会被要求输入密码。`apt` 是包管理器。`update` 用于刷新可用软件包列表。
2. 安装 Python 3、pip（Python 包安装器）和 venv（用于创建虚拟环境）：

   ```bash
   sudo apt install python3 python3-pip python3-venv
   ```

   此命令指示 `apt` `install` 指定的软件包。`python3` 是解释器本身，`python3-pip` 让你能够安装额外的 Python 库，而 `python3-venv` 协助处理项目特定的依赖项（我们稍后会提到）。

**对于 Fedora：**

1. 更新你的软件包列表：

   ```bash
   sudo dnf check-update
   ```

   Fedora 使用 `dnf` 包管理器。
2. 安装 Python 3 和 pip：

   ```bash
   sudo dnf install python3 python3-pip
   ```

   （`python3-venv` 的功能通常随 Fedora 上的主 `python3` 软件包一同提供。）

**对于 Arch Linux 及其衍生版（如 Manjaro）：**

1. 同步软件包数据库并更新系统：

   ```bash
   sudo pacman -Syu
   ```

   Arch 使用 `pacman` 包管理器。`-Syu` 刷新软件包列表并升级已安装的软件包。
2. 安装 Python 3 和 pip：

   ```bash
   sudo pacman -S python python-pip
   ```

   在 Arch 上，软件包名称通常是 `python`（表示最新的 Python 3）和 `python-pip`。

如果你使用的是不同的 Linux 发行版，请查阅其文档以获取相应的包管理器命令（例如，openSUSE 使用 `zypper`，较旧的 CentOS/RHEL 使用 `yum`）。软件包名称通常很相近（`python3`、`python3-pip`）。

### 验证安装

运行安装命令后，验证 Python 3 和 pip 是否已正确安装。打开一个新的终端窗口或再次输入命令：

```bash
python3 --version
pip3 --version
```

你现在应该会看到安装的版本号无误地打印到控制台。看到这些版本号就表明你的 Python 开发环境已在 Linux 机器上准备就绪。

Python 安装后，你现在可以开始与 Python 解释器进行交互或编写你的第一个脚本，这些内容我们将在接下来的部分中介绍。使用发行版的包管理器通常是在 Linux 上管理主要 Python 安装的最简单且非常有效的方法。

## 参考资料

- [Python Setup and Usage: Using Python on Unix platforms](https://docs.python.org/3/using/unix.html) — Python Software Foundation (2024)
  这份官方文档详细介绍了在类Unix系统（包括Linux）上安装和管理Python的方法，其中包含使用系统包管理器，与Linux环境设置直接相关。
- [Installation](https://pip.pypa.io/en/stable/installation/) — The pip developers (2024)
  pip（Python包安装器）的官方安装指南，涵盖了多种安装方法和包管理的重要信息。
- [venv - Creation of virtual environments](https://docs.python.org/3/library/venv.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  解释如何使用`venv`创建隔离的Python虚拟环境的官方文档，这是依赖项管理的基本实践。

---

[上一节](05-%E6%90%AD%E5%BB%BA%E6%82%A8%E7%9A%84%20Python%20%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%20%28macOS%29.md) · [下一节](07-Python%20%E8%A7%A3%E9%87%8A%E5%99%A8%20%28REPL%29%20%E7%9A%84%E4%BD%BF%E7%94%A8.md)

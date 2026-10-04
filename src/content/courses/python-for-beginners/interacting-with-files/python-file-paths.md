---
course: "python-for-beginners"
chapter: "interacting-with-files"
lesson: "python-file-paths"
sourceId: 2341
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-6-interacting-with-files/python-file-paths"
title: "理解文件路径"
description: "了解文件路径在不同操作系统上的作用以及绝对路径和相对路径的区别。"
order: 1
plots: []
sourceHash: "b3fa3571222fbb2549ec6daa5697968c37d90a4677595feaa8e63095c8a031ed"
sourceCorrections: []
---

当你的 Python 脚本需要从文件读取数据或将结果写回时，它首先需要知道文件在计算机存储（例如硬盘或固态硬盘）上的*位置*。就像你需要一个地址来找到特定的房屋一样，你的程序需要一个**文件路径**来找到特定的文件。这个路径本质上是文件在系统文件夹（也称为目录）层级结构中的地址。

可以将计算机的文件系统想象成一个大型文件柜。主文件柜是根目录，抽屉是主文件夹或驱动器，抽屉内的文件夹是子文件夹，而最终抽屉内的文档就是文件。文件路径会精确地告诉 Python 要访问哪个抽屉、该抽屉内的哪个文件夹以及哪个文档（文件）。

### 路径组成部分

常见的文件路径包含：

1. **目录/文件夹名称：** 你需要通过这些文件夹来找到文件。
2. **分隔符：** 用于分隔目录名称和最终文件名的特殊字符。
3. **文件名：** 文件的实际名称，通常包含扩展名（如 `.txt`、`.csv`、`.py`）。

具体格式，特别是分隔符，取决于你的操作系统。

### 操作系统差异

文件路径的表示方式因你使用的 Windows、macOS 或 Linux 系统而略有不同。

- **Windows：** 路径通常以驱动器盘符（如 `C:` 或 `D:`）开头，后跟冒号。目录使用反斜杠 (`\`) 分隔。

  - 示例：`C:\Users\YourUsername\Documents\report.txt`
  - 注意：Python 在 Windows 上相当灵活，通常也能识别路径中的正斜杠 (`/`)，这使得编写跨平台代码更简单。
- **macOS 和 Linux（类 Unix 系统）：** 这些系统使用单一根目录，由正斜杠 (`/`) 表示。目录总是使用正斜杠 (`/`) 分隔。没有像 Windows 那样的驱动器盘符。

  - 示例 (macOS)：`/Users/yourusername/Documents/report.txt`
  - 示例 (Linux)：`/home/yourusername/documents/report.txt`

理解这些差异很重要，特别是如果你打算与可能使用不同操作系统的人分享你的脚本。

### 绝对路径

**绝对路径**提供文件或目录从文件系统最顶层开始的完整位置。

- 在 Windows 上，它以驱动器盘符开头（例如，`C:\...`）。
- 在 macOS/Linux 上，它以根斜杠 (`/...`) 开头。

示例：

- Windows: `C:\Program Files\Python310\python.exe`
- macOS: `/Applications/Calculator.app`
- Linux: `/usr/bin/python3`

绝对路径是明确的；无论你的脚本当前在哪里运行，它们都指向一个特定位置。然而，将绝对路径硬编码到你的脚本中会降低其可移植性。如果你将项目移动到不同的位置或与他人分享，该绝对路径可能就不再正确了。

### 相对路径

**相对路径**指定文件或目录相对于当前工作目录（CWD）的位置。CWD 是你的 Python 脚本执行时所在的目录。

可以把它想象成从你当前所在位置给出方向：“进入 `data` 文件夹并找到 `results.csv`”或“向上一个级别，然后进入 `config` 文件夹”。

相对路径中使用了特殊符号：

- `.`（一个点）：表示当前目录。
- `..`（两个点）：表示父目录（向上一个级别）。

示例（假设你的脚本在 macOS/Linux 上从 `/Users/alice/project/` 运行，或在 Windows 上从 `C:\Users\Alice\project\` 运行）：

- `data.txt`：在 CWD（`project` 文件夹）中寻找 `data.txt`。
- `input/config.json`：在 CWD 中寻找一个名为 `input` 的文件夹，然后在该 `input` 文件夹中寻找 `config.json`。完整路径将是 `/Users/alice/project/input/config.json`。
- `../scripts/utility.py`：寻找 CWD 的父目录（即 `/Users/alice/` 或 `C:\Users\Alice\`），然后在该父目录中寻找一个 `scripts` 文件夹，最后在 `scripts` 中寻找 `utility.py`。完整路径将是 `/Users/alice/scripts/utility.py`。

> 此示例文件结构说明了从 `project` 目录（即当前工作目录 CWD）出发的相对路径和绝对路径。

相对路径使你的项目更加独立和可移植。如果你在项目目录中组织你的数据文件，你可以使用相对路径，即使整个项目文件夹被移动到其他位置，脚本仍然能正常运行。

You可以使用 Python 内置的 `os` 模块获取正在运行脚本的 CWD：

```python
import os
current_directory = os.getcwd()
print(f"当前工作目录是: {current_directory}")
```

运行这段代码会显示你的终端或 IDE 视为执行脚本起始点的目录的绝对路径。

### 绝对路径与相对路径的选择

对于初学者和大多数项目工作，**通常建议使用相对路径**。使用子目录组织你的项目，用于存放数据、脚本等，并使用 `data/my_file.txt` 或 `../config/settings.ini` 等相对路径。这使你的代码更易于分享并在不同计算机上运行。

主要在你需要访问项目结构之外的固定、已知位置的文件时使用绝对路径，例如系统配置文件或跨项目共享的资源。请注意，这可能会使你的脚本与特定的机器设置绑定。

Python 的文件处理函数（如接下来会看到的 `open()`）和 `os.path` 以及更新的 `pathlib` 等标准库模块提供了处理两种类型路径的工具，并帮助管理操作系统之间的差异。目前，重要的一点是理解你的计算机以及 Python 如何定位你想要处理的文件的原理。

## 参考资料

- [\`os\` - Miscellaneous operating system interfaces](https://docs.python.org/3/library/os.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  对于理解Python中基本的操作系统交互和路径操作功能至关重要。
- [\`pathlib\` - Object-oriented filesystem paths](https://docs.python.org/3/library/pathlib.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  介绍了一种现代、面向对象的文件系统路径处理方法，提供了更好的可读性和鲁棒性。
- [Reading and Writing Files](https://automatetheboringstuff.com/chapter10/) — Al Sweigart (2019)
  Publisher: No Starch Press; Pages: Chapter 10
  通过清晰的示例，为Python中的文件I/O操作和路径处理提供了实用且适合初学者的介绍。

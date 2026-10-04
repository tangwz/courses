---
course: "essential-numpy-pandas"
chapter: "intro-data-handling-python"
lesson: "setup-verification-practice"
sourceId: 1689
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-1-intro-data-handling-python/setup-verification-practice"
title: "动手实践：设置与验证"
description: "安装库、启动 Jupyter 并运行初始 NumPy/Pandas 命令的实际步骤。"
order: 6
plots: []
sourceHash: "3148b624a11ee68f488437656d5f6985bd1fb120c8e80e9835a89c0d38e657ec"
sourceCorrections: []
---

这个动手练习将引导你安装 NumPy 和 Pandas 库（如果尚未安装），并通过在 Jupyter Notebook 中运行一些基础代码来确认一切正常运行。

### 确认安装

根据你选择直接使用 Anaconda 还是 `pip`，安装步骤会略有不同。

#### 选项一：使用 Anaconda

Anaconda 简化了包管理。如果你按照“设置你的环境”部分的建议安装了 Anaconda，你很可能已经安装了 NumPy、Pandas 和 Jupyter。不过，我们还是来明确地确认或安装它们。

1. **打开 Anaconda Prompt（或 macOS/Linux 上的终端）：**

   - 在 Windows 上，在“开始”菜单中搜索“Anaconda Prompt”。
   - 在 macOS 或 Linux 上，打开你的标准终端应用程序。
2. **安装/更新库：** 确保你拥有最新版本是个好习惯。执行以下命令：

   ```bash
   conda install numpy pandas jupyterlab
   ```

   Conda 会检查这些包是否已安装，如果必要则更新它们，或者如果它们缺失则安装它们。它还会自动处理安装任何所需的依赖项。你可能会被提示确认安装计划；如果是，请输入 `y` 并按回车键。
3. **验证安装（可选）：** 你可以列出 Conda 管理的已安装包来检查：

   ```bash
   conda list numpy pandas
   ```

   此命令应显示 NumPy 和 Pandas 的已安装版本。

#### 选项二：使用 pip

如果你使用 `pip`（Python 的标准包安装器）直接管理你的 Python 环境，请按照以下步骤操作。

1. **打开终端或命令提示符：**

   - 在 Windows 上，打开命令提示符 (cmd) 或 PowerShell。
   - 在 macOS 或 Linux 上，打开你的终端。
2. **安装库：** 使用 `pip` 安装 NumPy、Pandas 和 JupyterLab（包含 Jupyter Notebook）：

   ```bash
   pip install numpy pandas jupyterlab
   ```

   *注意：* 根据你的系统配置，你可能需要使用 `pip3` 而不是 `pip`，特别是如果你同时安装了 Python 2 和 Python 3。在某些系统上，特别是 Linux 和 macOS，使用 `python -m pip install ...` 是一种更可靠的方式，可以确保你使用的是与你预期 Python 解释器关联的 `pip`。
3. **验证安装（可选）：** 你可以使用 `pip` 显示已安装包的详细信息：

   ```bash
   pip show numpy pandas
   ```

   如果 NumPy 和 Pandas 包成功安装，此命令将显示它们的信息。

### 启动 JupyterLab

库安装好后，让我们启动 JupyterLab，这是我们将在整个课程中使用的交互式环境。

1. **导航到你的项目目录（推荐）：** 打开你的终端或 Anaconda Prompt。使用 `cd`（更改目录）命令导航到你希望保存本课程笔记本的文件夹。例如：

   ```bash
   cd path/to/your/projects/essential-numpy-pandas
   ```

   将 `path/to/your/projects/essential-numpy-pandas` 替换为你计算机上的实际路径。在特定的项目文件夹中工作有助于使你的文件井井有条。
2. **启动 JupyterLab：** 输入以下命令并按回车键：

   ```bash
   jupyter lab
   ```

   此命令将启动 JupyterLab 服务器。你的默认网页浏览器应自动打开，显示 JupyterLab 界面。如果它没有自动打开，终端将提供一个 URL（通常以 `http://localhost:8888/lab` 开头），你可以复制并粘贴到浏览器的地址栏中。

保持终端窗口运行；关闭它将关闭 Jupyter 服务器。

### 创建并运行你的第一个笔记本

现在，让我们创建一个笔记本并运行一些代码来确认 NumPy 和 Pandas 已经准备就绪。

1. **创建新笔记本：** 在 JupyterLab 界面中（它在你的浏览器中打开），查找“启动器”选项卡。在“笔记本”下，点击“Python 3”内核图标（根据你的设置，它可能有一个稍微不同的名称，如“Python [conda env:base]”）。这将创建一个并打开一个新的、未命名的笔记本文件（`.ipynb`）。
2. **重命名笔记本（可选但推荐）：** 点击笔记本区域顶部的“Untitled.ipynb”名称，并将其重命名为有描述性的名称，例如 `01-Setup-Verification.ipynb`。
3. **输入并运行 NumPy 代码：** 在第一个代码单元格（旁边带有 `[ ]:` 的框）中，输入以下 Python 代码：

   ```python
   import numpy as np

   # Create a simple NumPy array
   my_array = np.array([1, 2, 3, 4, 5])

   # Print the array
   print("My first NumPy array:")
   print(my_array)

   # Print the shape of the array
   print("Array shape:")
   print(my_array.shape)
   ```

   要运行此单元格中的代码，请点击单元格内部并按下 `Shift + Enter`。
4. **验证 NumPy 输出：** 在单元格下方，你应该看到输出：

   ```
   My first NumPy array:
   [1 2 3 4 5]
   Array shape:
   (5,)
   ```

   看到此输出确认 NumPy 已正确安装并正常运行。`import numpy as np` 行导入该库，通常为其赋予别名 `np`。然后我们创建了一个简单的 1 维数组并打印它，以及它的形状（沿着一个维度有 5 个元素）。
5. **输入并运行 Pandas 代码：** 输出下方应出现一个新的代码单元格。如果没有，点击笔记本工具栏中的 `+` 按钮。在这个新单元格中，输入以下代码：

   ```python
   import pandas as pd

   # Create a simple Pandas Series
   my_series = pd.Series({'a': 10, 'b': 20, 'c': 30})

   # Print the Series
   print("My first Pandas Series:")
   print(my_series)
   ```

   按下 `Shift + Enter` 运行此单元格。
6. **验证 Pandas 输出：** 你应该看到以下输出：

   ```
   My first Pandas Series:
   a    10
   b    20
   c    30
   dtype: int64
   ```

   此输出确认 Pandas 也已安装并正常运行。我们使用约定俗成的别名 `pd` 导入它，并从 Python 字典创建了一个基本的 Series（一维带标签数组）。

### 故障排除提示

如果你在安装或运行代码时遇到错误：

- **命令未找到：** 如果你的终端无法识别 `conda` 或 `pip`，这可能意味着 Anaconda 或 Python 在安装过程中没有添加到你系统的 PATH 环境变量中。请重新查看你操作系统的安装说明，或尝试再次运行安装，确保你选择将它添加到 PATH 的选项（如果可用且适合你的设置）。
- **ImportError：** 如果 Python 提示找不到模块（`No module named 'numpy'` 或 `No module named 'pandas'`），则安装可能失败，或者你可能正在使用与安装库时不同的 Python 环境运行笔记本。确保你在执行安装的相同环境（例如，相同的 Anaconda Prompt/终端会话）中运行 `jupyter lab`。请再次尝试安装命令。
- **检查版本：** 有时会出现兼容性问题。在笔记本单元格中导入它们后，你可以使用 `np.__version__` 和 `pd.__version__` 检查版本。

成功运行这些简单的代码片段意味着你的环境已正确配置 NumPy 和 Pandas，并且你熟悉在 Jupyter Notebook 中执行代码的基本知识。你现在可以继续学习后续章节，并开始使用这些功能强大的库了。记住保存你的笔记本（文件 -> 保存笔记本或 Ctrl+S/Cmd+S），你也可以通过回到运行 `jupyter lab` 的终端窗口并按两次 `Ctrl + C` 来关闭 Jupyter 服务器。

## 参考资料

- [NumPy Documentation](https://numpy.org/doc/stable/) — The NumPy Development Team (2024)
  提供关于安装NumPy和理解其基本数组对象的全面指南，这对于验证至关重要。
- [pandas documentation](https://pandas.pydata.org/docs/) — The pandas Development Team (2025)
  提供关于安装pandas和开始使用其核心数据结构（如Series）的完整信息，有助于设置验证。
- [JupyterLab Documentation](https://jupyterlab.readthedocs.io/en/stable/) — The Project Jupyter Community (2024)
  详细说明JupyterLab的安装和使用，它是课程中用于运行代码的交互式开发环境。
- [Conda Documentation](https://docs.conda.io/en/latest/) — Anaconda, Inc. (2024)
  Publisher: Anaconda, Inc.
  解释如何使用Conda包管理器管理包和环境，包括安装和列出命令。
- [pip documentation](https://pip.pypa.io/en/stable/) — The pip developers (2024)
  涵盖使用pip（Python的标准包安装程序）安装和管理Python包，这与直接的库设置相关。

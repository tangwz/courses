---
course: "data-visualization-matplotlib-seaborn"
chapter: "intro-data-visualization-python"
lesson: "setup-python-environment-plotting"
sourceId: 1143
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-1-intro-data-visualization-python/setup-python-environment-plotting"
title: "设置您的Python环境"
description: "使用Anaconda或pip设置Python环境进行数据可视化工作的指南。"
order: 6
plots: []
sourceHash: "4a7c8be18f84d25483de8b23b0ad988c6abba0243c3f6badf1d06ff4c2bcb536"
sourceCorrections: []
---

在我们开始绘制图表之前，需要确保您的电脑已备好所需工具。这包括设置一个Python环境，其中包含Matplotlib和Seaborn库，以及它们常见的配套库如NumPy和Pandas。

可以将Python环境视为您项目的专用工作区。它能将项目所需的库版本与其他项目隔离开，从而避免冲突。对于数据科学和可视化工作，管理这些库及其依赖项有时会比较复杂，但幸运的是，我们有工具来简化此过程。

### 使用Anaconda进行环境管理

管理数据科学Python环境和包的流行方式之一是**Anaconda**（或其轻量级版本**Miniconda**）。Anaconda是一个发行版，捆绑有Python，一个名为`conda`的强大环境和包管理器，以及许多预安装的常用数据科学库。这使得设置相对简单。

**为何选择Anaconda？**

- **简便性：** 一次性安装Python和必要的数据科学库。
- **依赖项处理：** `conda`在管理库之间复杂的依赖项方面表现出色，这在科学计算中很常见。
- **环境管理：** 轻松为不同项目创建隔离环境。

**步骤（一般指南）：**

1. **下载与安装：** 访问[Anaconda发行版网站](https://www.anaconda.com/products/distribution)并下载适合您操作系统的安装程序（Windows、macOS或Linux）。遵循安装说明。我们建议使用提供的最新Python 3版本。
2. **创建专用环境（推荐）：** 打开您的终端（Windows上的Anaconda Prompt，macOS/Linux上的Terminal），并专门为本课程创建一个新环境。这有助于保持文件整洁。我们将它命名为`viz_env`，并包含Python、Matplotlib、Seaborn、Pandas、NumPy和JupyterLab（一个用于交互式绘图的实用工具）：

   ```bash
   conda create --name viz_env python=3.9 matplotlib seaborn pandas numpy jupyterlab
   ```

   您可能会被提示是否继续（`y/n`）；输入`y`并按回车。*注意：如果需要，您可以选择不同的Python版本，但通常建议使用3.9或更高版本。*
3. **激活环境：** 在为本课程安装包或运行代码之前，您需要激活刚创建的环境：

   ```bash
   conda activate viz_env
   ```

   您的终端提示符现在应在开头显示`(viz_env)`，表示环境已激活。

### 使用pip和venv（替代方案）

如果您不想使用Anaconda，可以使用Python的内置工具：`venv`用于创建虚拟环境，`pip`用于安装包。这种方法常受软件开发者青睐，并能保持安装最小化。

**为何选择pip和venv？**

- **内置：** `venv`作为Python 3.3+的标准组件提供。`pip`是标准的Python包安装器。
- **轻量级：** 创建只包含您明确安装内容的最小环境。

**步骤（一般指南）：**

1. **确保Python和pip已安装：** 您需要安装Python 3。`pip`通常随Python一同安装。您可以通过在终端中输入`python --version`或`python3 --version`来检查您的Python版本。
2. **创建虚拟环境：** 在终端中导航到您的项目目录并运行：

   ```bash
   python -m venv viz_env
   ```

   这会创建一个名为`viz_env`的文件夹，其中包含Python解释器副本和安装库的位置。
3. **激活环境：**
   - **macOS/Linux：** `source viz_env/bin/activate`
   - **Windows：** `viz_env\Scripts\activate`
     您的终端提示符应更改以指示活动环境（通常显示`(viz_env)`）。
4. **安装库：** 使用`pip`安装所需包：

   ```bash
   pip install matplotlib seaborn pandas numpy jupyterlab
   ```

### 给初学者的建议

虽然两种方法都可行，但本课程**我们推荐使用Anaconda**，特别是如果您是Python环境的新手。它处理数据科学库中常见复杂相互依赖关系的能力，使设置过程更顺畅。

### 验证您的设置

一旦您使用任一方法安装了库并激活了您的环境（`viz_env`），就可以进行快速检查。在已激活的终端中键入`python`启动Python解释器，或键入`jupyter lab`启动JupyterLab。

在Python提示符或Jupyter Notebook单元格中，尝试导入这些库：

```python
import matplotlib
import seaborn
import pandas
import numpy

print("库导入成功！")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Seaborn version: {seaborn.__version__}")
print(f"Pandas version: {pandas.__version__}")
print(f"NumPy version: {numpy.__version__}")
```

如果这段代码运行没有`ImportError`消息并打印出版本，则您的环境已正确设置，您可以继续将这些库导入到您的脚本中并创建您的第一个图表！

## 参考资料

- [Getting Started with Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) — Anaconda (2024)
  Publisher: Anaconda
  conda官方文档，涵盖环境和包管理。
- [\`venv\` - Creation of virtual environments](https://docs.python.org/3/library/venv.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  Python内置工具venv创建独立环境的官方指南。
- [\`pip\` User Guide](https://pip.pypa.io/en/stable/user_guide/) — PyPA (2024)
  安装和管理Python包的官方指南。

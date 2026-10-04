---
course: "probability-statistics-fundamentals-ml"
chapter: "intro-prob-stats-ml"
lesson: "setting-up-python"
sourceId: 2404
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-1-intro-prob-stats-ml/setting-up-python"
title: "搭建Python环境"
description: "关于搭建包含NumPy和Pandas等必要库的Python环境，以进行统计工作的说明。"
order: 5
plots: []
sourceHash: "72aaa699e5d6ffd5bed14e439f0fd8d32d28620bebc745257492b095711d1b37"
sourceCorrections: []
---

一个功能正常的Python环境，配备在数据科学和机器学习 (machine learning)中广泛使用的标准库，是进行实践操作的基础。Python因其易读性、丰富的库和活跃的社区支持，已成为这些方面的首选语言。提供设置这些所需工具的指导。

我们强烈推荐使用**Anaconda发行版**或其轻量级版本**Miniconda**。Anaconda捆绑了Python与许多流行的数据科学软件包，并包含`conda`——一个强大的包和环境管理器。使用`conda`有助于管理依赖项并隔离你的项目环境，从而避免不同项目可能需要同一库的不同版本所引起的冲突。这比直接将包安装到系统基础Python环境中要好得多。

### 安装Anaconda/Miniconda

1. **下载：** 访问Anaconda官方网站 (<https://www.anaconda.com/products/distribution>) 或Miniconda页面 (<https://docs.conda.io/en/latest/miniconda.html>)。下载适合你操作系统的安装程序（Windows、macOS或Linux）。我们建议选择最新的Python 3版本。
2. **安装：** 运行安装程序并按照屏幕上的说明操作。在Windows上，通常最好选择“Just Me”安装，并接受默认安装位置，除非你有特殊原因不这样做。在macOS或Linux上安装时，可能会问你是否希望安装程序通过运行`conda init`来初始化Anaconda。一般建议选择“是”，以便`conda`在你的终端中随时可用。在Windows上，Anaconda Prompt将添加到你的“开始”菜单。
3. **验证（可选）：** 打开你的终端（或Windows上的Anaconda Prompt）并输入`conda --version`。如果安装成功，它会显示已安装的conda版本。

### 核心库

虽然Anaconda默认安装了许多包，但我们将在本课程中大量使用的核心库是：

- **NumPy：** Python中进行数值计算的基本包。它提供高效的多维数组对象和处理这些对象的工具。许多统计计算都依赖于NumPy的功能。
- **Pandas：** 基于NumPy构建，Pandas提供高性能、易于使用的数据结构（如DataFrame）和数据分析工具。它对于加载、清洗、处理和分析结构化数据是不可缺少的。
- **Matplotlib & Seaborn：** 用于数据可视化的库。Matplotlib是基础绘图库，而Seaborn在此基础上提供更偏向统计学且更美观的图表，同时代码量更少。我们将在下一章中使用它们来可视化数据汇总。
- **JupyterLab/Jupyter Notebook：** 一个交互式计算环境，它允许你创建和共享包含实时代码、公式、可视化和叙述文本的文档。这非常适合学习、实验和呈现数据分析流程。Anaconda通常会自动安装这些。

### 使用Conda安装库

如果你安装了Miniconda，或者由于某些原因你的Anaconda安装中缺少这些库，你可以使用终端或Anaconda Prompt中的`conda`命令进行安装：

```bash
conda install numpy pandas matplotlib seaborn jupyterlab
```

Conda将确定这些包及其依赖项的兼容版本，并在安装前征求你的确认。

### 使用JupyterLab

库安装完成后，你可以启动JupyterLab环境。打开你的终端或Anaconda Prompt，切换到你希望存放课程文件的目录（你可以使用`cd path/to/your/directory`之类的命令），然后运行：

```bash
jupyter lab
```

此命令应在你的网页浏览器中打开一个新标签页，显示JupyterLab界面。在此处，你可以创建新的notebook（`.ipynb`文件），在单元格中编写和执行Python代码，使用Markdown添加解释性文本，并在notebook中直接查看结果和可视化内容。这个交互式环境非常适合本课程中的动手实践练习。

### 检查你的设置

为了确认核心库已正确安装，请创建一个新的Jupyter notebook并在一个单元格中运行以下代码：

```python
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns

print(f"NumPy 版本: {np.__version__}")
print(f"Pandas 版本: {pd.__version__}")
print(f"Matplotlib 版本: {mpl.__version__}")
print(f"Seaborn 版本: {sns.__version__}")
```

如果代码无错误运行并显示每个库的版本号，则你的环境已准备就绪。你现在拥有了加载数据并执行我们将在后续章节中讲解的统计分析所需的工具。

## 参考资料

- [Installing Miniconda](https://docs.conda.io/en/latest/miniconda.html) — Anaconda, Inc. (2024)
  Publisher: Anaconda, Inc.
  Miniconda的官方安装指南，以及如何使用其包和环境管理功能的说明。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104030/) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: 579
  一本使用Python的Pandas、NumPy和Jupyter工具进行数据处理和分析的全面指南，由Pandas的创建者撰写。
- [JupyterLab Documentation](https://jupyterlab.readthedocs.io/en/stable/) — Project Jupyter (2024)
  JupyterLab的官方文档，提供关于其界面、功能以及如何有效使用这个交互式计算环境的指南。
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html) — The NumPy Developers (2024)
  NumPy的官方用户指南，涵盖了其强大的数组对象、各种例程以及基本的数值计算功能。

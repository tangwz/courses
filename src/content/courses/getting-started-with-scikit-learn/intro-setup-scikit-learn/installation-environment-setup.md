---
course: "getting-started-with-scikit-learn"
chapter: "intro-setup-scikit-learn"
lesson: "installation-environment-setup"
sourceId: 1826
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-1-intro-setup-scikit-learn/installation-environment-setup"
title: "安装与环境设置"
description: "Scikit-learn 及其 NumPy 和 SciPy 等所需依赖的逐步安装指南。"
order: 2
plots: []
sourceHash: "db8be65e2e9811b7cc0f5890212a3e619693bbd772b75f193cc81f067b0106ba"
sourceCorrections: []
---

正确设置环境是使用 Scikit-learn 的必要步骤。Scikit-learn 旨在与现有 Python 科学计算体系顺利协作。恰当的设置能够保证你跟进示例，并将这些技术应用到你自己的项目中，避免兼容性问题。

如课程先决条件中所述，你应该已经安装了 Python，并对 NumPy 和 Pandas 有基本熟悉。Scikit-learn 直接基于这些库，特别是利用 NumPy 的数组结构和数学函数，并与 Pandas DataFrames 良好集成。

### 使用虚拟环境（推荐）

强烈建议为你的 Python 项目，包括本课程，使用虚拟环境。虚拟环境隔离项目依赖，避免了不同项目之间可能需要的同一库的不同版本造成的冲突。

**使用 `venv`（Python 内置工具）：**

1. **创建环境：** 打开你的终端或命令提示符，切换到你的项目目录，然后运行：

   ```bash
   python -m venv sklearn-env
   ```

   （将 `sklearn-env` 替换为你偏好的环境名称）。
2. **激活环境：**

   - 在 macOS 和 Linux 上：

     ```bash
     source sklearn-env/bin/activate
     ```
   - 在 Windows 上：

     ```bash
     sklearn-env\Scripts\activate
     ```

   你的终端提示符现在应该显示虚拟环境已激活。

**使用 `conda`（如果你使用 Anaconda 或 Miniconda）：**

1. **创建环境：**

   ```bash
   conda create --name sklearn-env python=3.9
   ```

   （将 `sklearn-env` 替换为你选择的名称，如果需要，指定一个 Python 版本）。Conda 将要求你确认要安装的软件包。
2. **激活环境：**

   ```bash
   conda activate sklearn-env
   ```

在已激活的虚拟环境中工作，可以保证你安装的软件包独立存在，并且不会影响你的全局 Python 安装或其他项目。

### 安装 Scikit-learn

虚拟环境激活后，你就可以安装 Scikit-learn。该库本身依赖于 NumPy 和 SciPy。使用 `pip` 或 `conda` 等标准软件包管理器安装 Scikit-learn，通常会自动处理这些依赖关系。

**使用 `pip`：**

这是标准 Python 软件包安装器。如果你不使用 Anaconda 分发版，这通常是最简单的方法。

```bash
pip install scikit-learn
```

此命令将下载并安装最新稳定版本的 Scikit-learn，如果你的环境中尚未存在 NumPy 和 SciPy，它还会一并安装所需的版本。

**使用 `conda`：**

如果你正在使用 Anaconda 或 Miniconda，`conda` 是推荐的软件包管理器。

```bash
conda install scikit-learn
```

Conda 管理软件包和环境，包括可能复杂的二进制依赖，这有时会带来好处，尤其是在 Windows 上。

### 核心依赖

Scikit-learn 的主要依赖是：

- **NumPy：** 用于高效的数组操作、线性代数和数值计算（`<https://numpy.org/>`）。Scikit-learn 的主要数据结构通常是 NumPy 数组。
- **SciPy：** 提供用于科学计算的基础算法，Scikit-learn 将其用于部分实现（`<https://scipy.org/>`）。

这些库构成了科学 Python 生态系统的基础，并且几乎总是与 Scikit-learn 一同安装。

### 增强功能的可选依赖

尽管对于基本的 Scikit-learn 使用并非严格必需，但在典型的机器学习 (machine learning)工作流程中，你经常会用到其他库：

- **Pandas：** 是数据操作和分析的重要工具，尤其适用于加载和准备存储在表格格式（如 CSV 文件）中的数据（`<https://pandas.pydata.org/>`）。Scikit-learn 与 Pandas DataFrames 集成度很高。
- **Matplotlib：** 用于创建静态、动画和交互式可视化图表的标准库（`<https://matplotlib.org/>`）。有助于数据分析和模型评估。
- **Seaborn：** 基于 Matplotlib 构建，Seaborn 提供高级接口，用于绘制美观且信息丰富的统计图形（`<https://seaborn.pydata.org/>`）。

如果你的环境中尚未安装这些库，你可以使用 `pip` 或 `conda` 添加它们：

**使用 `pip`：**

```bash
pip install pandas matplotlib seaborn jupyterlab
```

（我们在这里加入了 `jupyterlab`，因为它提供了一个方便的交互式数据科学工作环境，你可能会觉得它有用）。

**使用 `conda`：**

```bash
conda install pandas matplotlib seaborn jupyterlab
```

### 验证安装

为确认 Scikit-learn 已正确安装，你可以在已激活的虚拟环境中启动 Python 解释器或 Jupyter Notebook 会话，然后运行以下命令：

```python
import sklearn
import numpy
import scipy
import pandas # 可选，如果已安装
import matplotlib # 可选，如果已安装

# 打印 scikit-learn 版本
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"NumPy version: {numpy.__version__}")
print(f"SciPy version: {scipy.__version__}")
# print(f"Pandas version: {pandas.__version__}") # 如果已安装，请取消注释
# print(f"Matplotlib version: {matplotlib.__version__}") # 如果已安装，请取消注释
```

如果这些命令执行时没有引发 `ImportError` 并打印库版本，那么你的环境就设置正确了，可以继续学习 Scikit-learn 的功能了。我们将在本章末的动手实践中进行更正式的验证。

## 参考资料

- [Installing scikit-learn](https://scikit-learn.org/stable/install.html) — The scikit-learn developers (2024)
  Scikit-learn库及其依赖项的官方安装指南，涵盖多种安装方法。
- [\`venv\` - Creation of virtual environments](https://docs.python.org/3/library/venv.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  提供有关使用Python内置`venv`模块创建和管理隔离项目环境的全面信息。
- [Managing environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) — conda (2024)
  Conda包和环境管理器创建、激活和管理环境的官方指南。
- [NumPy documentation](https://numpy.org/doc/stable/) — The NumPy developers (2024)
  NumPy的官方文档，它是Python中数值计算的基础库，Scikit-learn在其之上构建。

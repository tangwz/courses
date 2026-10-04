---
course: "building-ml-recommendation-system"
chapter: "foundations-of-recommendation-systems"
lesson: "preparing-development-environment"
sourceId: 7453
sourceUrl: "https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-1-foundations-of-recommendation-systems/preparing-development-environment"
title: "准备开发环境"
description: "关于如何安装和配置 pandas、scikit-learn 和 surprise 等 Python 库的指导手册。"
order: 6
plots: []
sourceHash: "514c1b6cb701716be64ac986fd11d09bfa66ba3e31e34742b6637c249d1f3ba4"
sourceCorrections: []
---

要构建可运行的推荐系统，首先需要一个正确配置的开发环境。这种设置可以确保代码按预期运行，并为你提供数据处理、建模和评估所需的工具。你将学习如何创建独立环境并安装构建这些系统所需的 Python 库。

### 虚拟环境的必要性

在安装软件包之前，创建独立的 Python 环境是标准做法。虚拟环境是一个自包含的目录树，其中包含特定的 Python 安装程序和若干额外的软件包。使用虚拟环境可以避免不同项目所需的依赖项之间发生冲突，并保持全局 Python 安装环境的整洁。我们将使用 Python 3 自带的 `venv` 工具。

要创建并激活虚拟环境，请打开终端并运行以下命令。我们将环境命名为 `rec-env`，但你也可以选择任何名称。

首先，创建环境：

```bash
python3 -m venv rec-env
```

接下来，激活它。具体命令因操作系统而异。

**在 macOS 和 Linux 上：**

```bash
source rec-env/bin/activate
```

**在 Windows 上：**

```bash
rec-env\Scripts\activate
```

激活后，终端提示符通常会显示当前环境的名称，这表明你安装的所有软件包都将包含在该环境中。

> 开发环境的结构，从操作系统到虚拟环境中隔离的具体库。

### 安装必要的库

我们的工作将依靠 Python 数据科学体系中的一组库，以及一个专门用于构建推荐系统的库。

- **pandas**：用于加载、操作和清洗表格数据的主要工具。用户-物品交互数据通常在 pandas DataFrame 中进行管理。
- **NumPy**：提供高效的 N 维数组和数学函数，是许多数值算法的计算支柱。
- **scikit-learn**：一个全面的机器学习 (machine learning)库。我们将使用其模块来完成文本向量 (vector)化（使用 `TfidfVectorizer`）和计算相似度（使用 `cosine_similarity`）等任务。
- **JupyterLab**：一个基于 Web 的交互式环境，允许将代码、可视化图表和文本结合起来，非常适合数据分析和模型开发。
- **Surprise**：专门为构建和分析推荐系统而设计的 Python 库。它提供了几种主要算法（包括 SVD 和 k-NN）的预置实现，以及我们将在后续章节中频繁使用的评估工具。

在激活虚拟环境的情况下，使用 `pip` 安装这些包：

```bash
pip install pandas numpy scikit-learn jupyterlab scikit-surprise
```

安装过程可能需要几分钟，因为 `pip` 会下载并安装每个软件包及其依赖项。

### 验证安装

为了确认所有组件都已正确安装，你可以运行一个简短的 Python 脚本。创建一个名为 `verify_install.py` 的新文件，并添加以下代码：

```python
import pandas as pd
import numpy as np
import sklearn
import surprise

print("所有库均已成功导入！")
print("-" * 30)
print(f"pandas 版本: {pd.__version__}")
print(f"numpy 版本: {np.__version__}")
print(f"scikit-learn 版本: {sklearn.__version__}")
print(f"surprise 版本: {surprise.__version__}")
```

在终端执行该脚本：

```bash
python verify_install.py
```

如果环境设置正确，你将看到成功消息以及已安装库的版本号。此阶段出现的任何错误通常意味着安装过程中存在需要解决的问题。

环境配置完成后，你就可以获取并检查数据了，这些数据将为推荐模型提供支持。下一节将通过加载数据集进行实际练习，这是任何数据驱动项目的起点。

## 参考资料

- [venv - Creation of virtual environments](https://docs.python.org/3/library/venv.html) — Python Software Foundation (2023)
  Publisher: Python Software Foundation
  venv 官方文档，详细说明如何创建和管理独立的 Python 环境。
- [Python for Data Analysis: Data Wrangling with pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本指导使用 Python 数据分析库（包括 pandas 和 NumPy）的全面书籍。
- [scikit-learn: Machine Learning in Python](https://scikit-learn.org/stable/) — Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, and Vincent Michel (2023)
  scikit-learn 机器学习库的官方文档，涵盖其使用方法和算法。
- [JupyterLab Documentation](https://jupyterlab.readthedocs.io/en/stable/) — Project Jupyter (2023)
  JupyterLab 的官方指南，一个用于数据科学的交互式开发环境。
- [scikit-surprise documentation](https://surprise.readthedocs.io/en/stable/) — Nicolas Hug (2023)
  Surprise 库的官方文档，专门用于构建和分析推荐系统。

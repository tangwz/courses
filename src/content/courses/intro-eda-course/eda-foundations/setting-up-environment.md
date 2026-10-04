---
course: "intro-eda-course"
chapter: "eda-foundations"
lesson: "setting-up-environment"
sourceId: 1133
sourceUrl: "https://apxml.com/zh/courses/intro-eda-course/chapter-1-eda-foundations/setting-up-environment"
title: "设置您的环境"
description: "设置Python环境，包含数据分析工作所需库的指南。"
order: 5
plots: []
sourceHash: "eadef6d23c569b4f117b051e786ade487dd9b3e5109fc865cf4da522313c45cb"
sourceCorrections: []
---

简要回顾了EDA的目的以及相关标准Python库之后，下一步是准备您的本地设备。保持一致且独立的开发环境对管理项目依赖项和确保可复现性是必要的。这可以避免不同项目所需的包之间发生冲突。我们将概述两种常见方法：使用Python内置的`venv`模块配合`pip`，或使用Anaconda发行版。

### 使用 `venv` 和 `pip`

Python 3包含用于创建轻量级虚拟环境的`venv`模块。如果您已经有一个自己管理的Python安装，这种方法通常更受青睐。

1. **创建虚拟环境：**
   打开您的终端或命令提示符，导航到目标项目目录，然后运行：

   ```bash
   # 在 macOS/Linux 上
   python3 -m venv eda-env

   # 在 Windows 上
   python -m venv eda-env
   ```

   此命令会创建一个名为`eda-env`的目录（您可以选择任何名称），其中包含Python解释器的副本以及安装库的位置。
2. **激活环境：**
   在安装包之前，您需要激活环境：

   ```bash
   # 在 macOS/Linux 上
   source eda-env/bin/activate

   # 在 Windows (命令提示符) 上
   eda-env\Scripts\activate.bat

   # 在 Windows (PowerShell) 上
   eda-env\Scripts\Activate.ps1
   ```

   您的命令提示符现在应该表明您已进入`eda-env`环境。
3. **安装所需库：**
   在环境已激活的情况下，使用Python的包安装工具`pip`来安装本课程所需的核心库：

   ```bash
   pip install pandas numpy matplotlib seaborn jupyterlab
   ```

   此命令会下载并安装Pandas（用于数据处理）、NumPy（用于数值运算）、Matplotlib和Seaborn（用于数据可视化），以及JupyterLab（一个流行的数据科学交互式开发环境）。如果您喜欢Jupyter Notebook，它也是一个可行替代方案（`pip install notebook`）。

### 使用 Anaconda/Miniconda

Anaconda是一个专为数据科学设计的流行Python发行版。它预打包了许多常用库，并包含其自己的环境和包管理器，名为`conda`。Miniconda是`conda`的一个精简安装程序。如果您喜欢这个生态系统：

1. **安装 Anaconda 或 Miniconda：**
   如果您尚未安装，请从Anaconda或Miniconda的官方网站下载并安装它们。按照为您操作系统提供的说明操作。
2. **创建 Conda 环境：**
   打开您的终端或Anaconda Prompt并创建一个新环境：

   ```bash
   conda create --name eda-env python=3.9 pandas numpy matplotlib seaborn jupyterlab -y
   ```

   此命令会创建一个名为`eda-env`的环境，其中包含特定的Python版本（例如3.9，根据需要调整），并同时安装列出的包。`-y`标志会自动确认安装提示。
3. **激活环境：**
   激活新创建的环境：

   ```bash
   conda activate eda-env
   ```

   您的提示符将变为显示`(eda-env)`。

### 验证安装

无论选择哪种方法，您都可以验证所需库是否已正确安装。激活虚拟环境后，在终端中输入`python`来启动Python解释器，或者输入`jupyter lab`来启动JupyterLab并打开新笔记本。

在Python解释器或Jupyter笔记本单元格中，尝试导入这些库：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 打印库版本（可选，但推荐）
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
# 注意：Matplotlib和Seaborn的版本打印需要访问其特定版本属性
import matplotlib
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Seaborn version: {sns.__version__}")

# 测试一个简单命令
df_test = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
print("\n测试 DataFrame 创建成功：")
print(df_test.head())
```

如果这些命令在没有`ImportError`消息的情况下执行，则您的环境已正确配置所需库。

您现在已具备工具和专用环境，可以在后续章节中进行数据加载、清洗、分析和可视化。

## 参考资料

- [\`venv\` - Creation of virtual environments](https://docs.python.org/3/library/venv.html) — Python Software Foundation (2024)
  Python内置模块创建独立环境的官方指南。
- [pip documentation](https://pip.pypa.io/en/stable/) — The pip developers (2024)
  安装和管理Python包的官方指南。
- [Managing environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) — Anaconda, Inc. (2024)
  创建和管理Conda环境的官方说明。

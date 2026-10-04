---
course: "python-llm-workflows"
chapter: "python-environment-setup-llm"
lesson: "env-config-practice"
sourceId: 5111
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-2-python-environment-setup-llm/env-config-practice"
title: "开发环境配置实践"
description: "为本课程配置本地开发环境的实践步骤。"
order: 6
plots: []
sourceHash: "0a582257f3d01b2b28036b60e760d2e6a4c63e4b55664b46fa22ae4b9f3ff2fd"
sourceCorrections: []
---

为LLM工作流配置本地Python开发环境，包括实现良好组织的具体步骤。此内容详细说明了虚拟环境、依赖管理和安全API密钥处理。遵循这些步骤将为您的项目提供一个清晰且可重现的基础。

我们假设您的系统上安装了合适版本的Python，如“选择Python版本”一节中所述。

### 1. 创建项目目录

首先，为您的LLM项目创建一个专用文件夹。打开您的终端或命令提示符，导航到您通常存储开发项目的位置。然后，创建并进入新目录：

```bash
mkdir python-llm-course-project
cd python-llm-course-project
```

将项目保存在独立目录中是标准做法，有助于保持条理。

### 2. 初始化虚拟环境

在项目目录内，使用Python内置的`venv`模块创建一个虚拟环境。我们将环境目录命名为`.venv`（开头的点通常表示隐藏文件/文件夹，这在环境目录中很常见）。

```bash
python -m venv .venv
```

此命令会创建`.venv`子目录，其中包含Python解释器的副本以及管理本项目独立包所需的工具。

### 3. 激活虚拟环境

在安装包之前，您需要激活环境。激活命令在不同操作系统之间略有不同：

- **macOS / Linux (bash/zsh)：**

source .venv/bin/activate
\`\`\`

- **Windows (命令提示符)：**

.venv\Scripts\activate.bat
\`\`\`

- **Windows (PowerShell)：**

.venv\Scripts\Activate.ps1
\`\`\`
（您可能需要调整PowerShell的执行策略：`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`）

激活后，您的终端提示符通常会变为显示环境名称（例如，`(.venv) your-user@hostname:~/python-llm-course-project$`），表明`pip`命令现在将在该隔离环境中运行。

### 4. 安装核心库

虚拟环境激活后，我们来安装一些本课程中将要使用的核心库，使用`pip`。我们将从`python-dotenv`开始，用于管理环境变量（如API密钥），以及LangChain核心包。

```bash
python -m pip install python-dotenv langchain openai
```

*注意：根据您之后计划使用的LLM提供商，您可能需要其他包，如`anthropic`、`google-generativeai`等。我们将在需要时安装这些。*

### 5. 创建`requirements.txt`文件

为确保您的环境可重现并方便共享依赖，生成一个`requirements.txt`文件。此文件列出了您当前环境中安装的所有包及其具体版本。

```bash
python -m pip freeze > requirements.txt
```

您可以查看生成的`requirements.txt`文件。它将包含`langchain==X.Y.Z`、`openai==A.B.C`及其依赖项等条目。如果其他人想设置相同的环境，他们可以运行`pip install -r requirements.txt`。

### 6. 安全设置API密钥存储（`.env`文件）

如“安全设置API密钥”中所述，我们将使用`.env`文件来存储敏感凭据。在\*\*项目目录的根目录（`python-llm-course-project`）\*\*中创建一个名为`.env`的文件。以以下格式添加您的API密钥：

```plaintext
# .env 文件
OPENAI_API_KEY='your_actual_openai_api_key_here'
# ANTHROPIC_API_KEY='如果您需要Anthropic密钥'
# 根据需要添加其他密钥
```

将`'your_actual_openai_api_key_here'`替换为您的真实密钥。\*\*重要提示：\*\*切勿将此`.env`文件提交到Git等版本控制系统。

### 7. 配置`.gitignore`

为避免意外提交您的虚拟环境目录和敏感的`.env`文件，请在项目根目录中创建一个`.gitignore`文件。添加以下行：

```plaintext
# .gitignore 文件

# 虚拟环境
.venv/

# 环境变量
.env

# Python缓存文件
__pycache__/
*.pyc
```

这会告诉Git忽略这些文件和目录。

### 8. 通过简单脚本验证设置

我们来在项目根目录中创建一个小型Python脚本（`verify_setup.py`），以确认环境已激活、包已安装，并且API密钥可以加载。

```python
# verify_setup.py
import os
from dotenv import load_dotenv

print("尝试加载环境变量...")
# 从.env文件加载变量到环境变量
load_success = load_dotenv()

if load_success:
    print(".env 文件加载成功。")
else:
    print("警告：.env 文件未找到或加载失败。")

# 尝试从环境变量中获取API密钥
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    # 简单检查，不直接打印主密钥
    print("在环境变量中找到OpenAI API密钥。")
    print(f"长度: {len(api_key)}")
else:
    print("错误：在环境变量中未找到OpenAI API密钥。")
    print("请确保它在您的.env文件中设置正确并且文件已加载。")

# 验证包导入
try:
    from langchain_core.prompts import PromptTemplate
    print("成功从langchain_core导入PromptTemplate。")
except ImportError as e:
    print(f"从LangChain导入时出错: {e}")
    print("请确保LangChain已正确安装在您的虚拟环境中。")
```

在您的终端中运行此脚本（请确保您的虚拟环境仍处于激活状态）：

```bash
python verify_setup.py
```

如果一切配置正确，您应该看到确认`.env`文件已加载、API密钥已找到（以及其长度，而非密钥本身）以及LangChain组件成功导入的输出。如果您遇到错误，请仔细检查之前的步骤，确保`.env`文件存在于正确位置、虚拟环境已激活且库已无问题地安装。

### 项目结构概述

此时，您的项目目录应类似于以下结构：

> 您的项目目录包含虚拟环境（`.venv`）、API密钥存储（`.env`）、被忽略的文件列表（`.gitignore`）、依赖项列表（`requirements.txt`）以及您的Python脚本（`verify_setup.py`）。

您现在拥有了一个功能完善且隔离的Python环境，已准备好开发LLM应用。此设置提供了一个稳定的基础，供您在整个课程中安装更多库、编写代码和管理依赖项，而不会干扰其他项目或您的全局Python安装。

## 参考资料

- [venv - Creation of virtual environments](https://docs.python.org/3/library/venv.html) — Python Software Foundation (2023)
  Publisher: Python Software Foundation
  Python官方文档，解释了如何创建和管理独立的Python虚拟环境。
- [Installing Python Packages](https://packaging.python.org/en/latest/guides/installing-packages/) — The Python Packaging Authority (PyPA) (2023)
  一份关于`pip`使用、`requirements.txt`进行依赖管理以及安装Python包方法的指南。
- [python-dotenv](https://pypi.org/project/python-dotenv/) — Saurabh Kumar (2025)
  `python-dotenv`的官方PyPI页面，提供了使用`.env`文件加载环境变量进行安全凭证管理的信息。
- [gitignore(5) Manual Page](https://git-scm.com/docs/gitignore) — Git Project (2023)
  Git官方文档，描述了如何配置`.gitignore`文件以阻止Git跟踪特定的文件和目录（例如虚拟环境和敏感凭据）。

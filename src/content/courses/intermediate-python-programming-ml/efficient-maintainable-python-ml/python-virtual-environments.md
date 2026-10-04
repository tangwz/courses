---
course: "intermediate-python-programming-ml"
chapter: "efficient-maintainable-python-ml"
lesson: "python-virtual-environments"
sourceId: 2218
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-6-efficient-maintainable-python-ml/python-virtual-environments"
title: "虚拟环境简介"
description: "了解如何使用虚拟环境（如 venv 或 conda）有效地管理项目依赖项。"
order: 4
plots: []
sourceHash: "4ecd967eaad2db976dc86e052fb0c7347937c0ddace87f98f8e36dd7a0568ea6"
sourceCorrections: []
---

随着您的机器学习 (machine learning)项目不断发展，您会发现自己经常需要使用各种 Python 库，例如 NumPy、Pandas、Scikit-learn、Matplotlib，以及可能的深度学习 (deep learning)框架如 TensorFlow 或 PyTorch。这些库中的每一个都有其自己的依赖项和特定的版本要求。当项目 A 需要 NumPy 的 1.20 版本，而项目 B 却需要只有在 1.22 版本中才有的新功能时，会发生什么呢？将库直接安装到您的主 Python 安装中，很快就会导致冲突，并使您的项目难以在不同设置或不同时间稳定运行。在这种情况下，虚拟环境变得不可或缺。

### 什么是虚拟环境？

虚拟环境本质上是一个独立的目录，包含一个特定的 Python 解释器及其自己的一套已安装库。可以将其视为每个 Python 项目的独立作业区。当您激活一个虚拟环境时，您安装或卸载的任何软件包都仅限于该环境，而您的全局 Python 安装和其他项目环境则不受影响。

这种隔离避免了不同项目依赖项之间的冲突。您的系统上可以有多个虚拟环境，每个环境都根据特定项目的具体需求量身定制。

> 每个虚拟环境都维护自己独立的一套已安装软件包，从而防止项目间的版本冲突。

### 为什么在机器学习 (machine learning)中使用虚拟环境？

使用虚拟环境是 Python 开发中一项基本且良好的实践，在机器学习工作流程中它尤其有益，原因如下：

1. **依赖管理：** 这是主要优点。机器学习项目通常依赖于特定版本的库，如 Scikit-learn、TensorFlow、PyTorch、NumPy 或 Pandas。使用虚拟环境可以确保项目 A（可能使用较旧、稳定的 Scikit-learn 版本）在您为项目 B 安装最新版本时不会损坏。
2. **可重现性：** 对于研究和生产部署，确保您的代码运行可预期非常重要。虚拟环境允许您准确记录所需的软件包及其版本（通常在 `requirements.txt` 或 `environment.yml` 文件中）。其他人（或者您自己，在不同的机器上或稍后的时间）可以重新创建完全相同的环境，大大减少了“在我的机器上能运行”的问题。
3. **协作：** 在团队中工作时，虚拟环境确保每个人都使用相同的依赖项集合。共享 `requirements` 文件使协作者能够快速设置相同的环境，从而简化开发流程。
4. **整洁的系统安装：** 它使您的全局 Python 安装保持整洁，避免了各种项目积累的可能冲突的软件包集合。这使得您的基础系统更加稳定。

### 常用工具：`venv` 和 `conda`

Python 提供了几种管理虚拟环境的方式。其中两种特别常用：

1. **`venv`：** 这个模块包含在 Python 标准库中（从 Python 3.3 开始）。它轻量且通常足以满足许多主要依赖通过 `pip`（Python 包安装器）安装的软件包的项目。它创建的环境包含 Python 解释器的副本或符号链接，以及一个用于新库的 `site-packages` 目录。
2. **`conda`：** 这是一个包和环境管理器，随 Anaconda 和 Miniconda 发行版一同提供，在数据科学界很受欢迎。`conda` 可以管理 Python 包，也可以管理非 Python 软件依赖项（如 C 库）以及 Python 解释器本身。当项目具有仅靠 `pip` 难以管理的复杂依赖项，或者需要轻松切换不同 Python 版本时，它特别有用。

对于大多数标准 Python 机器学习 (machine learning)项目，如果依赖项可通过 `pip` 获取，`venv` 通常是更简单且推荐的起始选择。如果您的项目涉及复杂的非 Python 依赖项，或者您已经在使用 Anaconda 生态系统，`conda` 是一个功能强大的替代方案。

### `venv` 的基本工作流程

以下是使用 `venv` 在命令行上的典型工作流程：

1. **创建环境：** 导航到您的项目目录并运行：

   ```bash
   # 在 macOS/Linux 上
   python3 -m venv my_ml_env

   # 在 Windows 上
   python -m venv my_ml_env
   ```

   这会创建一个名为 `my_ml_env` 的目录（您可以选择任何名称），其中包含环境文件。
2. **激活环境：**

   ```bash
   # 在 macOS/Linux (bash/zsh) 上
   source my_ml_env/bin/activate

   # 在 Windows (命令提示符) 上
   my_ml_env\Scripts\activate.bat

   # 在 Windows (PowerShell) 上
   my_ml_env\Scripts\Activate.ps1
   ```

   您的命令行提示符应会改变，以表明环境已激活（例如，`(my_ml_env) your_user@machine:...$`）。
3. **安装软件包：** 现在，使用 `pip` 安装项目所需的库。这些库将安装在激活的环境*内部*。

   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```
4. **冻结依赖项：** 为了使您的环境可重现，将已安装软件包及其确切版本的列表保存到文件中，通常命名为 `requirements.txt`。

   ```bash
   pip freeze > requirements.txt
   ```

   此文件可以与他人共享，或稍后用于重新创建环境。
5. **从 requirements 文件安装：** 如果您收到一个包含 `requirements.txt` 文件的项目，您可以创建一个新的虚拟环境，激活它，然后使用以下命令安装所有依赖项：

   ```bash
   pip install -r requirements.txt
   ```
6. **停用环境：** 完成项目工作后，您可以停用环境：

   ```bash
   deactivate
   ```

   这会将您返回到系统的全局 Python 上下文 (context)。

### 与您的机器学习 (machine learning)项目结构集成

正如在项目结构化部分讨论的，您通常应该在主项目文件夹内或旁边创建虚拟环境。将环境目录（上述示例中的 `my_ml_env/`）添加到项目的 `.gitignore` 文件中是一种常见做法，以防止将环境本身提交到版本控制；只需跟踪 `requirements.txt` 文件。

从机器学习项目开始时就采用虚拟环境是一个简单而有效的方法，可以创建更易于维护和协作的代码库。它消除了常见的错误源，并确保精心制作的数据管道和模型在不同设置中表现一致。

## 参考资料

- [\`venv\` - Creation of virtual environments](https://docs.python.org/3/library/venv.html) — Python Documentation (2023)
  Python内置`venv`模块的官方文档，详细说明了其用于创建隔离Python环境的用法。
- [Managing environments (Conda documentation)](https://docs.conda.io/en/latest/user-guide/tasks/manage-environments.html) — Conda documentation team (2023)
  Conda管理虚拟环境的官方指南，涵盖了创建、激活以及管理各类依赖包。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本广泛使用的书籍，介绍了Python进行数据处理和分析，为数据科学项目中依赖管理的重要性提供了背景。(第三版)
- [The Hitchhiker's Guide to Python! - Virtual Environments](https://docs.python-guide.org/dev/virtualenvs/) — Kenneth Reitz and Tanya Schlusser (2024)
  一份由社区维护的指南，提供了Python开发的实用建议和最佳实践，包含虚拟环境的专门章节。

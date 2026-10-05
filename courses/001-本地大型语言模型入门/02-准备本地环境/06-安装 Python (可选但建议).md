# 安装 Python (可选但建议)

来源：[原文](https://apxml.com/zh/courses/getting-started-local-llms/chapter-2-preparing-local-environment/installing-python)

[返回章节目录](README.md) · [返回课程目录](../README.md)

运行本地大型语言模型 (LLM) 的一些流行工具，如 Ollama 或 LM Studio (我们稍后会介绍)，通常会打包所有必需组件。但即便如此，在你的系统上安装 Python 也常常很有用，尤其当你计划进行超出最基本配置的操作时。可以将 Python 看作不仅仅是一种编程语言，更是人工智能和机器学习 (machine learning)中大量工具和库的依托。

许多高级 LLM 框架、用于模型微调 (fine-tuning)的库，或自动化任务的脚本都使用 Python 编写。安装 Python 之后，为日后使用这些资源提供了可能。即使你开始使用的特定工具没有明确要求 Python，现在安装它也会为将来的使用提供灵活性。

### 为何考虑安装 Python？

- **工具兼容性广：** 许多重要的 AI 库，如 Hugging Face 的 `transformers`、`PyTorch` 和 `TensorFlow`，都高度依赖 Python。
- **脚本编写：** Python 让你能够编写简单的脚本，以与 LLM 交互、处理文本或自动化工作流程。
- **社区支持：** Python 拥有一个庞大且活跃的社区，这意味着有大量的教程、支持论坛和预构建的代码片段可用。
- **未来应用：** 如果你决定尝试训练小型模型、微调 (fine-tuning)现有模型或使用更专业的 LLM 工具，Python 环境通常是先决条件。

### 检查 Python 是否已安装

在安装之前，值得检查 Python 是否已存在于你的系统上，可能由其他软件安装而来。打开你的命令行或终端 (我们将在下一节简要介绍)，并尝试以下命令：

- **在 Windows 上：**
  打开命令提示符或 PowerShell 并输入：

  ```bash
  python --version
  ```

  你可能还需要尝试：

  ```bash
  py --version
  ```
- **在 macOS 或 Linux 上：**
  打开你的终端并输入：

  ```bash
  python3 --version
  ```

如果你看到版本号 (例如 `Python 3.9.7` 或类似)，则表示 Python 3 已安装。如果你收到“command not found”之类的错误消息，或者报告的版本以 `2.` 开头 (例如 `Python 2.7.18`)，则应安装当前版本的 Python 3。大多数 LLM 工具与 Python 3.8 或更高版本配合最佳。

### 如何安装 Python

安装 Python 最可靠的方法是使用官方来源：

1. **访问官方网站：** 前往 Python 下载页面 [python.org/downloads/](https://python.org/downloads/)。网站通常会检测你的操作系统并推荐合适的安装程序。
2. **下载安装程序：** 下载适用于你操作系统 (Windows、macOS 或 Linux) 的最新稳定版 Python 3 安装程序。
3. **运行安装程序：** 执行下载的文件。
   - **Windows 用户须知：** 在安装过程中，请务必勾选标有 **“将 Python [版本] 添加到 PATH”** 或 **“将 python.exe 添加到 Path”** 的框。这将使 Python 可以方便地从命令行访问。
   - **macOS/Linux：** 按照安装程序的提示操作。在这些系统上的安装通常很简单。也可以使用 `brew` (macOS) 或 `apt`/`yum` (Linux) 等包管理器，但官方安装程序通常更适合初学者。
4. **验证安装：** 安装完成后，关闭并重新打开你的终端或命令提示符。然后，再次运行版本检查命令 (Windows 上为 `python --version` 或 `py --version`，macOS/Linux 上为 `python3 --version`)。现在你应该能看到刚刚安装的版本号。

### 快速了解 `pip`：Python 包安装器

当你安装 Python 时，它通常会附带一个名为 `pip` 的工具。这是 Python 的标准包管理器，用于安装和管理其他开发者创建的额外库或软件包。你不需要单独安装 `pip`。

虽然我们暂时不会安装特定的 Python 包，但你经常会看到这样的说明：

```bash
pip install some-library-name
```

此命令会指示 `pip` 从 Python 包索引 (PyPI) 下载并安装指定的库 (`some-library-name`)，PyPI 是一个全面的 Python 软件库。如果你选择使用基于 Python 的 LLM 工具，那么了解 `pip` 是如何为你的 Python 环境添加额外功能的，对后续步骤很有帮助。

总而言之，虽然对于使用 Ollama 或 LM Studio 等工具的第一步来说，安装 Python 可能并非严格必要，但这是一个建议的准备步骤。它能让你的系统适应本地 LLM 方面的更广泛用途，并符合更广泛的 AI 和机器学习 (machine learning)方面的常见做法。

## 参考资料

- [The Python Tutorial (Python 3.12.2 documentation)](https://docs.python.org/3/tutorial/) — Guido van Rossum and the Python Development Team (2024)
  Publisher: Python Software Foundation
  官方文档提供了关于Python的全面指导，包括安装和语言基础知识。
- [pip documentation](https://pip.pypa.io/en/stable/) — The pip developers (2023)
  Publisher: Python Packaging Authority
  `pip`是Python的标准包安装程序，本官方指南对其进行了介绍，用于管理库和包。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd Edition](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  展示了Python在机器学习和人工智能中的实际应用，利用了TensorFlow和PyTorch等框架。

---

[上一节](05-%E6%93%8D%E4%BD%9C%E7%B3%BB%E7%BB%9F%E5%85%BC%E5%AE%B9%E6%80%A7.md) · [下一节](07-%E5%91%BD%E4%BB%A4%E8%A1%8C-%E7%BB%88%E7%AB%AF%E7%AE%80%E4%BB%8B.md)

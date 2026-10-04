---
course: "python-for-beginners"
chapter: "putting-it-all-together"
lesson: "python-project-structure"
sourceId: 2392
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-10-putting-it-all-together/python-project-structure"
title: "组织你的项目文件"
description: "学习在小型 Python 项目中组织文件和目录的基本原则。"
order: 4
plots: []
sourceHash: "1c7adfcb520faebc50209473738b5d3c38fe43322137cd1be27a82fc758afd50"
sourceCorrections: []
---

即使是小项目，花时间思考如何组织文件也会带来很大益处。良好的结构能让你的代码更容易理解、维护和与他人（甚至是你未来的自己！）分享。当你休息一段时间后回到一个项目时，清晰的布局能帮助你快速找到内容。

### 从项目目录开始

为你的项目创建一个专门的目录（文件夹）非常重要。请避免将Python脚本和相关文件随意放置在桌面或文档文件夹中。

例如，如果你正在构建本章讨论的简单命令行工具，为其创建一个目录：

```bash
mkdir my_simple_tool
cd my_simple_tool
```

所有与此特定工具相关的文件都将位于 `my_simple_tool` 目录内。

### 基本结构

对于只包含一个脚本的非常简单的应用程序，你的项目结构可能就是脚本本身，或许再加一个说明文件。

> 包含主脚本和 README 文件的基本项目结构。

这里：

- `tool.py`：这是你的主 Python 脚本，其中包含应用程序的代码。你可能将其命名为诸如 `calculator.py` 或 `task_manager.py` 这样有说明性的名字。
- `README.md`：这是一个用 Markdown 格式编写的文本文件。包含 `README.md` 是标准做法，用以说明项目功能、如何配置（如有必要）以及如何运行。即使是简单的描述也很有帮助。

### 职责分离：多文件

随着你的应用程序稍有增长，你可能会发现自己在编写辅助函数或定义特定的类。与其将所有内容都放入一个庞大的脚本中，不如将逻辑上不同的部分分离到不同的文件（模块）中。这提高了组织性和可重用性。

想象你的命令行工具需要一些实用工具函数。你可以将它们放在一个单独的文件中，比如命名为 `utils.py`。

> 包含主脚本、实用工具模块和 README 的项目结构。

在这种结构中：

- `tool.py`：仍然是主脚本，是你应用程序的入口点。它会从 `utils.py` 导入函数或类。
- `utils.py`：包含供 `tool.py` 使用的辅助函数或类。
- `README.md`：仍然用作项目描述。

你可以使用导入语句将 `utils.py` 中的函数导入到 `tool.py` 中，例如 `import utils` 或 `from utils import specific_function`。

### 跟踪依赖项：`requirements.txt`

如果你的项目开始使用外部包（通过 `pip` 安装的库），记录它们非常重要。标准方式是使用名为 `requirements.txt` 的文件。此文件列出了项目运行所需的外部包及其版本。

Example `requirements.txt`:

```
requests==2.28.1
numpy>=1.21.0
```

即使你的初始项目不使用外部库，了解 `requirements.txt` 对未来的项目或分享你的代码也很有用。其他人可以使用 `pip install -r requirements.txt` 轻松安装所有必需的依赖项。

### 大型项目的常见约定

随着项目变得明显更大、更复杂，通常会采用更精细的结构。你可能会遇到以下目录：

- `src/` 或 `app/`：包含应用程序的主要源代码，通常会进一步划分为子包。
- `tests/`：包含用于测试主应用程序代码的文件。
- `docs/`：用于存放文档文件。
- `data/`：用于存放应用程序使用或生成的数据文件。
- `scripts/`：用于存放与项目相关的实用工具或辅助脚本。

对于本课程中我们正在构建的简单命令行工具，前面展示的基本结构通常足够。主要原则是让项目组织方式与当前复杂程度相符，并在需要时允许其合理扩展。从一个专门的目录和 `README.md` 开始始终是一个好的做法。随着复杂性略有增加，为不同逻辑添加独立文件（如 `utils.py`）是下一个合理步骤。

## 参考资料

- [Tutorial on packaging projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/) — Python Packaging Authority (PyPA) (2024)
  Publisher: Python Packaging Authority (PyPA)
  着重于设置用于分发的软件包，这本身就涉及健壮的项目结构，包括 `src/` 目录和依赖管理。
- [Structuring Your Project](https://docs.python-guide.org/writing/structure/) — The Hitchhiker's Guide to Python Contributors (2024)
  为不同复杂度的 Python 项目组织提供了实用指南和常见约定。
- [Requirements Files](https://pip.pypa.io/en/stable/user_guide/#requirements-files) — The pip developers (2024)
  官方文档，详细说明了 `requirements.txt` 文件在管理项目依赖方面的创建和使用。
- [Clean Code: A Handbook of Agile Software Craftsmanship](https://www.pearson.com/us/higher-education/program/Martin-Clean-Code-A-Handbook-of-Agile-Software-Craftsmanship-1st-Edition/PGM1047608.html) — Robert C. Martin (2008)
  Publisher: Pearson Education
  一本关于软件工程的奠基性书籍，倡导的整洁代码和良好设计原则是有效项目结构和模块化的基础。

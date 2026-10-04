# 虚拟环境 (venv, conda)

来源：[原文](https://apxml.com/zh/courses/python-llm-workflows/chapter-2-python-environment-setup-llm/virtual-environments-venv-conda)

[返回章节目录](README.md) · [返回课程目录](../README.md)

随着您开始构建更复杂的 Python 应用程序，尤其是在像 LLM 开发这样快速变化的方面，您会很快遇到一个常见问题：管理项目依赖。想象一下，您正在处理两个不同的 LLM 项目。项目 A 需要某个特定库（例如 `langchain`）的 1.5 版本，而项目 B 需要更新的 1.8 版本。如果您将这些库全局安装在您的系统上，当您切换到另一个项目时，一个项目将不可避免地出现问题。这是因为全局 Python 安装通常一次只能支持给定库的一个版本。

此时，虚拟环境变得不可或缺。虚拟环境本质上是一个隔离的目录，其中包含一个特定的 Python 解释器及其自己的一组已安装的包，独立于您系统的全局 Python 安装或其他虚拟环境。可以将其视为每个项目的独立工作区。

使用虚拟环境有几个显著的优点：

1. **依赖隔离：** 每个项目都可以有自己特定的一组依赖项（和版本），而不会干扰其他项目。这避免了前面描述的“依赖地狱”情况。
2. **可复现性：** 通过列出项目的准确依赖（通常在 `requirements.txt` 文件中，接下来会讲到），您或合作者可以轻松地在其他地方重新创建相同的环境。
3. **整洁性：** 它使您的全局 Python 安装保持整洁，不受项目特定包的影响。

两个主要工具常用于创建和管理 Python 虚拟环境：`venv` 和 `conda`。

### 使用 `venv`

`venv` 是 Python 3.3 及更高版本中包含的、用于创建虚拟环境的标准轻量级工具。对于许多 Python 项目，它通常足够了，包括主要依赖通过 `pip` 安装的 Python 包的典型 LLM 应用程序。

**使用 `venv` 创建环境**

要使用 `venv` 创建虚拟环境，在您的终端或命令提示符中，进入项目根目录并运行以下命令：

```bash
python -m venv .venv
```

- `python -m venv` 调用 `venv` 模块。
- `.venv` 是为存放虚拟环境的目录选择的名称。使用 `.venv` 或 `venv` 是一个常见做法，在类 Unix 系统上，以点号开头通常会默认隐藏该目录。您可以选择任何名称，但保持一致性会有帮助。

此命令会创建 `.venv` 目录。该目录包含用于创建它的 Python 解释器的一个副本或符号链接，以及像 `lib/pythonX.Y/site-packages`（在 Linux/macOS 上）或 `Lib\\site-packages`（在 Windows 上）这样的子目录，项目特定的包将安装在那里。

**激活环境**

在使用环境之前，您需要激活它。激活会修改您 shell 的路径设置，以便 `python` 和 `pip` 等命令指向虚拟环境目录中的版本，而不是全局版本。激活命令根据您的操作系统的不同而略有差异：

- **在 macOS 和 Linux 上 (bash/zsh)：**

  ```bash
  source .venv/bin/activate
  ```
- **在 Windows 上 (命令提示符)：**

  ```cmd
  .venv\\Scripts\\activate.bat
  ```
- **在 Windows 上 (PowerShell)：**

  ```powershell
  .venv\\Scripts\\Activate.ps1
  ```

  （注意：您可能需要先调整 PowerShell 执行策略：`Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`）

激活后，您的 shell 提示符通常会改变以表明当前激活的环境，通常会以 `(.venv)` 作为提示符前缀。现在，任何使用 `pip` 安装的包都将放入 `.venv` 目录中，专属于此项目。

**停用环境**

当您完成项目工作或想切换到另一个项目时，只需运行：

```bash
deactivate
```

此命令会将您的 shell 恢复到之前的状态，再次使用全局 Python 安装。

### 使用 `conda`

`conda` 是一个跨平台的包和环境管理系统，随 Anaconda 和 Miniconda 发行版提供。尽管 `venv` 主要处理 Python 包，但 `conda` 与语言无关，擅长管理复杂的依赖，包括非 Python 库（如用于 GPU 加速的 CUDA，这与本地运行 LLM 相关）甚至 Python 解释器本身。

**使用 `conda` 创建环境**

如果您已安装 Anaconda 或 Miniconda，可以使用以下命令创建新环境：

```bash
conda create --name myenv python=3.10
```

- `conda create --name myenv` 创建一个名为 `myenv` 的新环境。选择一个描述性的名称。
- `python=3.10` 指定在此环境中安装的 Python 版本。`conda` 可以在您的系统上同时管理多个 Python 版本，这些版本在不同的环境中隔离。您可以省略版本以获取与您的 `conda` 安装关联的默认版本。

`conda` 将确定必要的依赖项，向您显示一个计划，并在创建环境之前请求确认。

**激活环境**

要激活 `conda` 环境：

```bash
conda activate myenv
```

与 `venv` 类似，您的提示符通常会改变以表明当前激活的环境，例如 `(myenv)`。

**停用环境**

要停用当前的 `conda` 环境并返回到默认环境（或之前激活的环境）：

```bash
deactivate
```

### 您应该使用哪个工具？

- **`venv`：** 对于大多数依赖项可通过 `pip` 获取的纯 Python 项目，通常更推荐使用。它内置、轻量，并且紧密遵循标准的 Python 实践。如果您的 LLM 工作流程主要涉及像 `langchain`、`openai`、`llamaindex` 等库以及标准数据科学工具（`pandas`、`numpy`），那么 `venv` 通常是更简单、更直接的选择。
- **`conda`：** 如果您的项目在 Python 包中包含复杂的依赖项（例如 C 库、特定版本的系统工具），或者如果您需要轻松管理不同的 Python 版本，或者如果您已经在使用 Anaconda 生态系统（在数据科学中很常见），它是一个更好的选择。某些涉及本地模型运行或特定硬件加速的 LLM 任务可能会从 `conda` 管理系统级依赖的能力中受益。

对于本课程，除非您有指向 `conda` 的特定需求，推荐使用 **`venv`**，因为它简单且与 Python 直接集成。核心要点是：**每个项目选择一个工具并持续使用**。无论您选择哪个工具，在开始项目工作时，激活环境都应该是您采取的第一步。这确保您始终将包安装到预期隔离空间中并在其中运行代码，从而防止冲突并使您的 LLM 开发过程更加顺畅。

## 参考资料

- [venv - Creation of virtual environments](https://docs.python.org/3/library/venv.html) — Python Software Foundation (2023)
  Publisher: Python Software Foundation
  Python 内置 `venv` 模块的官方文档，解释了如何创建和管理虚拟环境。
- [Managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) — Anaconda, Inc. (2024)
  官方文档，提供了使用 `conda` 创建、激活和管理独立环境及软件包的说明。
- [The Hitchhiker's Guide to Python! - Virtual Environments](https://docs.python-guide.org/dev/virtualenvs/) — Kenneth Reitz and Tanya Schlusser (2023)
  一份广受认可的社区风格指南和资源，涵盖 Python 开发实践，其中包括关于虚拟环境的详细部分。

---

[上一节](01-%E9%80%89%E6%8B%A9%E4%BD%A0%E7%9A%84Python%E7%89%88%E6%9C%AC.md) · [下一节](03-%E4%BD%BF%E7%94%A8pip%E5%92%8Crequirements.txt%E7%AE%A1%E7%90%86%E4%BE%9D%E8%B5%96.md)

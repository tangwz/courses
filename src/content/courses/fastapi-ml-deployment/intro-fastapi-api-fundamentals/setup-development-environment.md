---
course: "fastapi-ml-deployment"
chapter: "intro-fastapi-api-fundamentals"
lesson: "setup-development-environment"
sourceId: 5238
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-1-intro-fastapi-api-fundamentals/setup-development-environment"
title: "设置开发环境"
description: "安装 Python、FastAPI、Uvicorn 及设置开发虚拟环境的指南。"
order: 5
plots: []
sourceHash: "9a0830d31438aa6b71fc8aeada39ac98d98d1020be9eba2d5a01b29567e60195"
sourceCorrections: []
---

设置开发环境是使用 FastAPI 构建 API 的一个基础步骤。这包括确保您安装了正确的 Python 版本，并安装了必要的库：FastAPI 本身以及一个像 Uvicorn 这样的 ASGI 服务器来运行您的应用程序。我们还将采用使用虚拟环境来清晰地管理项目依赖的良好做法。

本课程假定您已安装 Python 3.7 或更高版本。FastAPI 使用了现代 Python 特性，例如类型提示和异步编程，这些功能需要较新的版本。您可以通过打开终端或命令提示符并运行以下命令来检查您的 Python 版本：

```bash
python --version
# 或有时是 python3 --version
```

如果您需要安装或更新 Python，请访问 Python 官方网站 (<https://www.python.org/>) 并按照您操作系统的说明进行操作。

### 使用虚拟环境

强烈建议为每个 Python 项目使用虚拟环境。虚拟环境会生成独立的区域，您可以在其中安装特定版本的包，而不会影响全局 Python 安装或其他项目。这可以避免依赖冲突，并使您的项目可复现。

Python 附带 `venv` 模块，用于生成虚拟环境。

1. **进入您的项目目录：** 打开终端，切换到您想创建 FastAPI 项目的目录。

   ```bash
   mkdir fastapi-ml-intro
   cd fastapi-ml-intro
   ```
2. **创建虚拟环境：** 运行 `venv` 模块，为您的环境指定一个名称（常见约定是 `.venv` 或 `venv`）。

   ```bash
   python -m venv .venv
   # 或 python3 -m venv .venv
   ```

   此命令会创建一个 `.venv` 目录，其中包含 Python 解释器的副本和用于安装项目特定包的位置。
3. **激活虚拟环境：** 激活会修改您 shell 的路径，以优先使用该环境的 Python 解释器和包。

   - **在 macOS 和 Linux 上：**

     ```bash
     source .venv/bin/activate
     ```
   - **在 Windows 上（命令提示符）：**

     ```bash
     .venv\\Scripts\\activate.bat
     ```
   - **在 Windows 上（PowerShell）：**

     ```bash
     .venv\\Scripts\\Activate.ps1
     ```

     （您可能需要先调整执行策略：`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`）

   激活后，您的终端提示符通常会显示环境的名称（例如，`(.venv) your-user@your-machine:...$`）。所有后续的 `pip` 安装都将在此独立环境中进行。要取消激活，只需键入 `deactivate`。

### 安装 FastAPI 和 Uvicorn

虚拟环境激活后，您现在可以使用 Python 的包安装器 `pip` 来安装 FastAPI 和 Uvicorn。

- **FastAPI：** 核心 Web 框架。
- **Uvicorn：** 一个 ASGI（异步服务器网关接口）服务器。FastAPI 基于 ASGI 标准构建，Uvicorn 是 FastAPI 作者推荐的，用于开发和生产的快速服务器程序。

通过一个命令安装两者：

```bash
pip install "fastapi[all]"
```

`[all]` 选项是 FastAPI 提供的一种便捷方式。它会安装 `fastapi` 本身、用于服务的 `uvicorn`、用于数据验证的 `pydantic`（FastAPI 大量依赖此库），以及其他一些通常有用的可选依赖项（如用于表单数据的 `python-multipart`、用于模板的 `jinja2` 等）。这可以保证您有一个全面的起始点。

如果您偏好最小化安装，可以单独安装它们：

```bash
pip install fastapi uvicorn pydantic
```

然而，对于初学者来说，通常建议使用 `fastapi[all]`。

### 验证安装

要确认 FastAPI 和 Uvicorn 已正确安装，您可以让 `pip` 显示它们的详细信息：

```bash
pip show fastapi uvicorn
```

此命令将显示有关已安装包的信息，包含它们的版本。请确保没有报告任何错误。

### 开发工具

虽然不是严格要求，但使用一个好的代码编辑器或 IDE（集成开发环境）将大大提升您的开发体验。Visual Studio Code (VS Code)、PyCharm、Sublime Text 或其他选项都提供了语法高亮、代码补全、调试和终端集成等功能，这些在与 FastAPI 等框架配合使用时非常实用。

您的开发环境现已准备就绪。您已拥有 Python、FastAPI、Uvicorn 和一个独立的虚拟环境。在下一节中，您将使用此配置来创建并运行您的第一个简单 FastAPI 应用程序。

## 参考资料

- [venv - Creation of virtual environments](https://docs.python.org/3/library/venv.html) — Python Software Foundation (2024)
  Python 内置模块创建独立虚拟环境的官方文档。
- [Installation](https://fastapi.tiangolo.com/install/) — Sebastián Ramírez (2024)
  Publisher: FastAPI Project
  安装 FastAPI 及其推荐的 ASGI 服务器 Uvicorn 的官方说明。
- [Installing Packages - Creating virtual environments](https://packaging.python.org/en/latest/guides/installing-packages/#creating-virtual-environments) — Python Packaging Authority (PyPA) (2024)
  Python 包安装的权威信息，包括虚拟环境的使用。

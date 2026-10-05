# 打包您的 Python LLM 应用程序

来源：[原文](https://apxml.com/zh/courses/python-llm-workflows/chapter-10-deployment-operational-practices/packaging-python-llm-app)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将您的大型语言模型 (LLM) 应用程序从本地开发机器迁移到生产环境需要仔细准备。仅仅复制 Python 脚本不足以实现可靠且可扩展的部署。应用程序打包涉及将代码、依赖项和配置组织成一种标准格式，以便于在其他地方安装和运行。适当的打包确保一致性，简化部署，并使您的应用程序更容易管理。

### 组织您的项目

在您打包应用程序之前，它需要一个逻辑结构。一个组织良好的项目更容易理解、维护和打包。虽然没有单一的强制性结构，但一种常见且有效的布局会将应用程序代码、测试、配置和文档分开。

考虑为典型的 LLM 应用程序（例如使用 FastAPI 和 LangChain 构建的应用程序）采用这样的结构：

```
my_llm_app/
├── app/                     # 主应用程序源代码
│   ├── __init__.py
│   ├── main.py              # FastAPI 应用程序定义
│   ├── chains.py            # LangChain 或工作流逻辑
│   ├── models.py            # 用于 API I/O 的 Pydantic 模型
│   └── utils.py             # 辅助函数
├── tests/                   # 单元和集成测试
│   ├── __init__.py
│   ├── test_chains.py
│   └── test_api.py
├── .env.example             # 示例环境变量
├── config.yaml              # 静态配置（可选）
├── Dockerfile               # 构建 Docker 镜像的说明
├── pyproject.toml           # 项目元数据和构建配置 (PEP 517/518)
├── README.md                # 项目文档
└── requirements.txt         # 应用程序依赖项
```

> 适用于 Python LLM 应用程序的典型项目结构，将源代码 (`app`)、测试 (`tests`)、配置和打包文件分开。

这种结构清楚地表明了项目的不同部分所在位置。`app` 目录包含核心逻辑，`tests` 保存测试，而像 `.env.example` 和 `config.yaml` 这样的配置文件管理设置。`pyproject.toml` 文件在定义项目元数据和构建系统要求方面越来越成为标准，而 `requirements.txt` 则列出了直接依赖项。

### 管理依赖项

您的 LLM 应用程序可能依赖于像 `langchain`、`llamaindex`、`openai`、`fastapi`、`pydantic` 等外部库。这些依赖项必须明确声明，以便它们可以在任何环境中保持一致地安装。

管理依赖项的标准方法是使用 `requirements.txt` 文件。此文件仅仅列出所需的包，理想情况下会固定特定版本。固定版本（例如 `langchain==0.1.16`）可以防止在依赖项的新版本发布时出现不兼容更改而导致的意外中断。

您可以根据当前的开发环境（最好是虚拟环境）使用 `pip` 生成此文件：

```bash
# 首先激活您的虚拟环境
# 例如：source venv/bin/activate

pip freeze > requirements.txt
```

尽管 `pip freeze` 会捕获环境中的所有内容，但通常更好的做法是手动整理 `requirements.txt`，只列出应用程序需要的直接依赖项。让 `pip` 解析子依赖项。对于更复杂的依赖项管理，特别是管理开发、测试和生产的不同依赖项集合时，像 `pip-tools` 或 `Poetry` 这样的工具会很有益，它们通常与 `pyproject.toml` 集成。

### 处理配置

LLM 应用程序通常需要 API 密钥等敏感信息，或模型名称、端点和阈值等配置细节。将这些信息直接硬编码到源代码中是不安全且不灵活的。

最佳实践要求将配置与代码分离：

1. **环境变量：** 使用环境变量在运行时注入配置。这在容器化和云环境中很常见。像 `python-dotenv` 这样的工具可以在开发期间从 `.env` 文件加载变量。请记住将 `.env` 添加到您的 `.gitignore` 文件中，以避免提交秘密信息。
2. **配置文件：** 使用 YAML (`config.yaml`) 或 TOML 等格式存储敏感度较低或结构更规整的配置。在应用程序代码中加载这些文件。

您的应用程序代码应该在启动时读取这些配置。例如，使用 Pydantic 的设置管理或像 `os.getenv()` 这样的库来读取环境变量。

### 创建构建制品：Wheel 文件

分发 Python 包的标准方法是使用“wheel”格式（`.whl` 文件）。Wheel 文件是一种预构建的包制品，它比源发行版 (`sdist`) 安装快得多，因为它通常可以跳过带有 C 扩展的包所需的编译步骤。它将您的代码和元数据打包成标准化结构。

要构建 Wheel 文件，您首先需要定义项目的元数据。现代标准是使用 `pyproject.toml` 文件 (PEP 621)。这是一个最小示例：

```toml
# pyproject.toml

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "my_llm_app"
version = "0.1.0"
authors = [
  { name="Your Name", email="your.email@example.com" },
]
description = "一个示例 LLM 应用程序。"
readme = "README.md"
requires-python = ">=3.9"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License", # 选择您的许可证
    "Operating System :: OS Independent",
]
dependencies = [
    "fastapi",
    "uvicorn[standard]",
    "langchain",
    "openai",
    "python-dotenv",
    # 在此处添加 requirements.txt 中的其他直接依赖项
]

[project.urls]
Homepage = "https://github.com/yourusername/my_llm_app" # 可选

# 定义命令行脚本（入口点）
[project.scripts]
my-llm-cli = "app.cli:main" # 如果您有 CLI 界面，则为示例
```

此文件指定了项目名称、版本、作者、描述、Python 版本要求，并且重要地，还指定了依赖项（应与您的 `requirements.txt` 文件保持一致）。它还定义了构建系统（在此例中为 `setuptools`）。

有了 `pyproject.toml` 文件，您可以使用标准 `build` 工具构建 Wheel 文件：

```bash
# 如果您尚未安装构建工具，请先安装
pip install build

# 构建 wheel 和源发行版
python -m build
```

此命令将创建一个 `dist/` 目录，其中包含 `.whl` 文件（即 wheel 文件）和 `.tar.gz` 文件（即源发行版）。Wheel 文件 (`dist/my_llm_app-0.1.0-py3-none-any.whl`) 是您通常用于部署的制品。它封装了您的应用程序代码和元数据，可以随时通过 `pip install dist/my_llm_app-0.1.0-py3-none-any.whl` 进行安装。

### 定义入口点

如果您的应用程序包含命令行界面 (CLI) 或需要作为特定脚本运行，您可以在 `pyproject.toml` 文件的 `[project.scripts]` 表下定义“入口点”。

例如，`my-llm-cli = "app.cli:main"` 意味着在安装 Wheel 文件后，环境中将有一个名为 `my-llm-cli` 的命令可用，该命令会执行 `app/cli.py` 模块中的 `main` 函数。这对于管理任务或与您的 LLM 应用程序进行简单的命令行交互很有用。对于使用 Uvicorn 等工具提供服务的 Web 应用程序，入口点可能是 Web 服务器命令本身，指向您的 FastAPI 或 Flask 应用程序实例（例如 `uvicorn app.main:app`）。

通过遵循这些打包步骤、组织您的项目、管理依赖项、安全地处理配置以及构建一个可分发的 Wheel 文件，您将创建一个标准化且可重现的制品。这个打包好的应用程序为部署的后续步骤奠定了根基，例如使用 Docker 进行容器化，我们将在随后进行介绍。

## 参考资料

- [Packaging Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/) — Python Packaging Authority (PyPA) (2024)
  PyPA官方教程，详细介绍了现代Python项目打包，包括pyproject.toml元数据、构建wheel包和依赖声明。
- [PEP 621 -- Storing project metadata in pyproject.toml](https://peps.python.org/pep-0621/) — Brett Cannon, Dustin Ingram, Paul Ganssle, Pradyun Gedam, Sébastien Eustace, Thomas Kluyver, Tzu-ping Chung (2020)
  Journal: Python Enhancement Proposals
  定义了在pyproject.toml中指定Python项目元数据的标准，是现代打包和构建系统的核心。
- [Structuring Your Project](https://docs.python-guide.org/writing/structure/) — Kenneth Reitz and Tanya Schlusser (2024)
  提供关于如何组织Python项目以形成可维护且可扩展结构的实用指导。

---

[上一节](../09-LLM%20%E5%BA%94%E7%94%A8%E7%9A%84%E6%B5%8B%E8%AF%95%E4%B8%8E%E8%AF%84%E4%BC%B0/07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%B8%BALLM%E9%93%BE%E8%AE%BE%E7%BD%AE%E5%9F%BA%E6%9C%AC%E6%B5%8B%E8%AF%95.md) · [下一节](02-%E4%BD%BF%E7%94%A8%20Docker%20%E8%BF%9B%E8%A1%8C%E5%AE%B9%E5%99%A8%E5%8C%96.md)

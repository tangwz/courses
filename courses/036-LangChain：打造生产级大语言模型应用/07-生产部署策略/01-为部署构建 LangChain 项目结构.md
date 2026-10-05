# 为部署构建 LangChain 项目结构

来源：[原文](https://apxml.com/zh/courses/langchain-production-llm/chapter-7-deployment-strategies-production/structuring-projects-deployment)

[返回章节目录](README.md) · [返回课程目录](../README.md)

### 推荐的项目布局

尽管理想结构可能因应用复杂程度和团队偏好而异，但对于包含 LangGraph 和 LangServe 等现代工具的生产级 LangChain 应用，一种常见且高效的布局通常如下所示：

> 这是一个可部署 LangChain 应用的典型目录结构，强调职责分离和现代组件。

我们来审视每个主要目录的用途：

- **`src/` (或 `app/`)**: 这是你应用的核心。它包含定义 LangChain 逻辑的 Python 模块和包。
  - **模块化**: 在 `src/` 内部，根据功能将代码组织到子目录中。常见文件夹包含 `graphs/` (用于 LangGraph 有状态工作流)、`chains/` (用于 LCEL 可运行组件)、`tools/`、`prompts/` 和 `utils/`。这种结构有助于隔离逻辑，便于重用和测试。
  - **API 入口**: 对于生产部署，你通常需要一个 API 服务器。像 `server.py` 这样的文件通常放置在此处，用于定义 FastAPI 应用和 LangServe 路由。
  - 在 `src/` 及其子目录中使用 `__init__.py` 文件，将它们标记 (token)为 Python 包。
- **`config/`**: 在这里存放配置文件，按环境分离（例如，`default.yaml`, `production.yaml`）。这对于模型参数 (parameter)或提示模板等复杂的非敏感设置很有用。然而，现代应用越来越倾向于依赖代码中定义的 Pydantic Settings 类，这些类直接从环境变量读取，从而减少了对外部 YAML 文件的依赖。
- **`tests/`**: 包含所有自动化测试。
  - **`unit/`**: 针对独立函数、工具或图节点的测试。
  - **`integration/`**: 验证组件之间交互的测试，例如完整的链式执行或代理循环。
- **`scripts/`**: 存放用于非主应用流程任务的实用脚本，例如数据摄取、向量 (vector)存储索引或评估运行。
- **`notebooks/`**: 用于试验和原型设计的 Jupyter Notebook。将它们分开能确保实验性代码不与生产逻辑混淆。
- **`deploy/` (或 `infra/`)**: 包含与部署基础设施相关的文件，例如 Kubernetes 清单 (`kubernetes/`)、Terraform 配置 (`terraform/`) 或 Helm 图表。
- **`Dockerfile`**: 定义如何为你的应用构建容器镜像。它通常位于根目录中，以便构建上下文 (context)可以包含所有项目文件。
- **依赖管理文件**:
  - `requirements.txt` 或 `pyproject.toml`: 定义运行时依赖项。对于 Poetry 或 PDM 等现代工具，使用 `pyproject.toml` 是标准做法，允许严格的版本锁定以及更便捷的开发依赖管理。
- **环境变量**:
  - `.env`: *应在 `.gitignore` 中列出*。包含秘密信息（API 密钥）和本地设置。
  - `.env.example`: 一个模板文件，展示所需变量但没有具体值。
- **`.gitignore`**: 指定 Git 应该忽略的文件（例如，`.env`, `__pycache__/`, 本地数据）。
- **`README.md`**: 提供关于设置、测试和部署流程的基本文档。

## 参考资料

- [Packaging Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/) — Python Packaging Authority (2024)
  Python 应用程序和库结构化、打包和分发的官方指南，涵盖布局和依赖文件。
- [The Twelve-Factor App](https://12factor.net/) — Adam Wiggins (2011)
  构建健壮且可扩展的软件即服务应用程序的奠基性方法论，涵盖配置管理和部署原则。
- [Poetry Documentation](https://python-poetry.org/docs/) — Sébastien Eustace and Contributors (2024)
  Poetry 的官方文档，这是一种 Python 依赖管理和打包工具，通过清晰的依赖定义帮助确保可重现的构建。

---

[上一节](../06-%E4%BC%98%E5%8C%96%E5%92%8C%E6%89%A9%E5%B1%95%20LangChain%20%E5%BA%94%E7%94%A8/07-%E5%AE%9E%E8%B7%B5%EF%BC%9ALangChain%20%E9%93%BE%E7%9A%84%E6%80%A7%E8%83%BD%E8%B0%83%E4%BC%98.md) · [下一节](02-%E4%BD%BF%E7%94%A8%20Docker%20%E5%B0%86%20LangChain%20%E5%BA%94%E7%94%A8%E5%AE%B9%E5%99%A8%E5%8C%96.md)

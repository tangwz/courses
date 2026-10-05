# 设置 Flask

来源：[原文](https://apxml.com/zh/courses/basics-ml-deployment/chapter-3-prediction-service-flask/setting-up-flask)

[返回章节目录](README.md) · [返回课程目录](../README.md)

为了为模型创建 API，我们选择的工具 Flask 将在开发环境中进行配置。Flask 是一个针对 Python 的微框架，这意味着它提供了构建 Web 应用的基本功能，而不强加过多的结构或依赖。这使其成为创建简单预测服务的理想选择。

### Python 和 pip

在安装 Flask 之前，请确保您的系统上已安装 Python。Flask 要求 Python 3.7 或更高版本。除 Python 外，您还应该拥有 `pip`，它是 Python 的标准包安装程序。您通常可以通过打开终端或命令提示符并运行以下命令来检查您的 Python 版本：

```bash
python --version
# 或者可能是 python3 --version
```

并使用以下命令检查您的 pip 版本：

```bash
pip --version
# 或者可能是 pip3 --version
```

如果您已安装 Python，`pip` 通常会随其一起安装。

### 使用虚拟环境

强烈建议在虚拟环境中安装 Flask 和其他项目依赖。虚拟环境为您的项目提供了一个独立的 Python 环境。这意味着您为该项目安装的包不会干扰全局安装的包或为其他项目安装的包，从而避免潜在的版本冲突。

1. **创建虚拟环境：**
   在您的终端或命令提示符中，导航到您的项目目录。然后，运行以下命令来创建一个虚拟环境。我们将环境命名为 `venv`（一种常见约定），但您可以选择不同的名称。

   ```bash
   python -m venv venv
   ```

   此命令会在您的项目目录中创建一个 `venv` 文件夹，其中包含 Python 解释器的副本以及一个用于安装项目特定库的位置。
2. **激活虚拟环境：**
   在安装 Flask 之前，您需要激活该环境。激活命令因您的操作系统而略有不同：

   - **在 macOS 和 Linux 上：**

     ```bash
     source venv/bin/activate
     ```
   - **在 Windows 上（命令提示符）：**

     ```bash
     .\venv\Scripts\activate
     ```
   - **在 Windows 上（PowerShell）：**

     ```bash
     .\venv\Scripts\Activate.ps1
     ```

     （您可能需要调整 PowerShell 执行策略：`Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`）

   激活后，您的终端提示符通常会改变，显示活动环境的名称（例如，`(venv) your-prompt$`），这表明 `pip` 等命令现在将在此独立环境中运行。

### 安装 Flask

在您的虚拟环境激活后，使用 `pip` 安装 Flask 非常简单：

```bash
pip install Flask
```

此命令指示 `pip` 从 Python 包索引 (PyPI) 下载 Flask 及其所需的依赖项（例如用于处理 HTTP 请求和响应的 Werkzeug，以及用于模板的 Jinja2，尽管我们在此不重点介绍模板），并将它们安装到您当前的 `venv` 中。

### 验证安装

您可以快速验证 Flask 是否正确安装。一种方法是询问 Flask 的版本：

```bash
python -m flask --version
```

或者，您可以在已激活的虚拟环境中启动 Python 解释器并尝试导入 Flask：

```bash
python
```

然后，在 Python 解释器内部：

```python
>>> import flask
>>> print(flask.__version__)
# 这应该会打印已安装的 Flask 版本，例如 2.3.2
>>> exit()
```

如果这些命令运行没有错误并显示版本号，则 Flask 已成功安装在您的虚拟环境中并可供使用。

Flask 安装完成后，我们便具备了构建第一个简单 Web 应用所需的一切，该应用最终将提供我们已保存机器学习 (machine learning)模型的预测服务。在下一节中，我们将编写 Flask 应用的基础代码。

## 参考资料

- [Flask Installation](https://flask.palletsprojects.com/en/latest/installation/) — Pallets (2010)
  Publisher: Pallets
  Flask Web 框架的官方安装指南，包含设置虚拟环境和使用 pip 安装 Flask 的说明。
- [\`venv\` - Creation of virtual environments](https://docs.python.org/3/library/venv.html) — Python Software Foundation (2024)
  Python 内置模块 venv 的官方文档，该模块提供了为项目创建独立 Python 环境的标准方法。
- [Installing Python Modules](https://docs.python.org/3/installing/) — Python Software Foundation (2024)
  描述如何安装 Python 模块和包的官方指南，主要使用 pip 包安装程序。

---

[上一节](02-Web%E6%A1%86%E6%9E%B6%E7%AE%80%E4%BB%8B.md) · [下一节](04-%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E5%9F%BA%E7%A1%80%E7%9A%84%20Flask%20%E5%BA%94%E7%94%A8.md)

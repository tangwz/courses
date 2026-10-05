# 安全配置 API 密钥

来源：[原文](https://apxml.com/zh/courses/python-llm-workflows/chapter-2-python-environment-setup-llm/setting-up-api-keys-securely)

[返回章节目录](README.md) · [返回课程目录](../README.md)

与大型语言模型交互通常需要 API 密钥，它们本质上是密码，允许访问可能产生费用的服务。随意处理这些密钥会带来很大的安全隐患。强烈不建议将它们直接嵌入 (embedding)到源代码中。如果您的代码被共享、发布到版本控制系统（如 Git），甚至意外泄露，您的密钥可能会被泄露，导致未经授权的使用和意料之外的费用。

思考这个示例，关于**不应**做的事情：

```python
# 警告：不安全做法 - 请勿硬编码 API 密钥！
import openai

# 切勿像这样将您的秘密密钥直接嵌入代码中。
API_KEY = "sk- दिस इज ए सीक्रेट की" # 示例占位符 - 切勿这样做

client = openai.OpenAI(api_key=API_KEY)

# ... 您的其余代码
```

将这样的代码提交到仓库，即使是私有仓库，也存在风险。一种更安全、更灵活的方法是使用环境变量。

### 使用环境变量

环境变量是由操作系统维护的变量，存在于应用程序源代码之外。这种分离使它们非常适合存储敏感的配置信息，如 API 密钥。

大多数大型语言模型提供商的客户端库（例如 OpenAI、Anthropic、Cohere 或 Hugging Face 的库）都设计成在初始化时未直接传入 API 密钥的情况下，会自动查找特定的环境变量。常见的变量名包括 `OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`COHERE_API_KEY` 等。请务必查阅您正在使用的特定库的文档。

您可以在终端会话中直接设置环境变量：

```bash
# 在 Linux/macOS 上
export OPENAI_API_KEY="your_actual_api_key_here"

# 在 Windows 命令提示符中
set OPENAI_API_KEY="your_actual_api_key_here"

# 在 Windows PowerShell 中
$Env:OPENAI_API_KEY="your_actual_api_key_here"
```

设置后，您可以在 Python 代码中使用 `os` 模块访问这些变量：

```python
import os
import openai

# 从环境变量加载 API 密钥
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    print("错误：OPENAI_API_KEY 环境变量未设置。")
    # 适当处理错误，例如退出程序或抛出异常
else:
    # 安全地使用
    client = openai.OpenAI(api_key=api_key)
    # 或者通常，如果环境变量已设置，客户端库可以自动找到它：
    # client = openai.OpenAI() # 如果环境变量已设置，这通常会起作用

    print("API 密钥加载成功。")
    # ... 使用客户端的其余代码
```

使用 `os.getenv("VARIABLE_NAME")` 通常比 `os.environ["VARIABLE_NAME"]` 更受推荐，因为 `getenv` 在变量未找到时会返回 `None`，这让您能够妥善处理缺失的密钥。直接访问 `os.environ["VARIABLE_NAME"]` 如果变量未设置，将引发 `KeyError`。

虽然直接在终端设置变量可行，但它是临时的（通常只在当前会话中有效），而且如果您有许多变量或需要在不同项目之间切换，这可能会变得繁琐。

### 在本地开发中使用 `.env` 文件

在本地开发期间管理环境变量的一种常见且便捷的做法是使用 `.env`（点环境）文件。这些是简单的文本文件，您可以在其中将环境变量定义为键值对。

1. **安装辅助库：** 您的脚本运行时，您需要一个库将这些文件加载到您的环境中。一个常用的选择是 `python-dotenv`。使用 pip 安装它：

   ```bash
   pip install python-dotenv
   ```
2. **创建 `.env` 文件：** 在您项目的根目录中，创建一个名为 `.env` 的文件（注意开头的点）。添加您的 API 密钥和其他配置变量，每行一个：

   ```plaintext
   # .env 文件内容
   OPENAI_API_KEY="sk-your_openai_api_key_here"
   ANTHROPIC_API_KEY="sk-ant-your_anthropic_api_key_here"
   # 如果需要，添加其他配置变量
   # MODEL_NAME="gpt-4"
   ```
3. **在您的代码中加载 `.env` 文件：** 在 Python 脚本的开头（或在中央配置模块中），使用 `python-dotenv` 将 `.env` 文件中的变量加载到实际环境中：

   ```python
   import os
   import openai
   from dotenv import load_dotenv

   # 从 .env 文件加载环境变量
   load_dotenv()

   # 现在您可以像正常设置一样访问变量
   openai_api_key = os.getenv("OPENAI_API_KEY")
   anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

   if openai_api_key:
       print("找到 OpenAI 密钥。")
       # client_openai = openai.OpenAI() # 通常会自动找到
       # ... 使用
   else:
       print("警告：在环境变量或 .env 文件中未找到 OPENAI_API_KEY。")

   if anthropic_api_key:
       print("找到 Anthropic 密钥。")
       # ... 使用
   else:
       print("警告：在环境变量或 .env 文件中未找到 ANTHROPIC_API_KEY。")

   # ... 您的应用程序的其余逻辑
   ```

   `load_dotenv()` 函数读取 `.env` 文件，并将其键值对加载到可以通过 `os.getenv()` 访问的环境变量中。

### 最重要的一步：`.gitignore`

您的 `.env` 文件包含秘密信息。**它绝不能提交到版本控制（如 Git）。** 为防止意外提交，请将 `.env` 添加到您项目的 `.gitignore` 文件中。

如果您没有 `.gitignore` 文件，请在您项目的根目录中创建一个。添加以下行（以及您希望 Git 忽略的其他文件/目录，例如虚拟环境文件夹）：

```plaintext
# .gitignore 文件内容

# 忽略虚拟环境目录
venv/
__pycache__/

# 忽略敏感配置文件
.env

# 忽略 IDE/编辑器文件
.vscode/
.idea/
```

这会告诉 Git 忽略 `.env` 文件，防止您的敏感 API 密钥被意外推送到您的仓库。

> 不安全的硬编码与使用从被 Git 忽略的 `.env` 文件加载的环境变量来管理 API 密钥的对比。

### 从本地开发到后续部署

虽然 `.env` 文件非常适合本地开发，但它们通常不直接用于生产环境。在部署应用程序时，您通常会通过部署平台的机制配置环境变量（例如 Heroku 配置变量、AWS Systems Manager Parameter Store、Docker 环境变量、Kubernetes Secrets）。其原理保持不变：将秘密信息置于代码库之外，并通过环境注入它们。对于高度敏感的应用程序，专门的秘密管理工具（如 HashiCorp Vault）或云提供商服务提供更高级的功能，但环境变量提供了一个可靠的起点。

从一开始就使用环境变量和 `.env` 文件（并被 Git 正确忽略）安全地管理 API 密钥，是构建可靠、安全的大型语言模型应用程序的一项基本做法。这可以防止意外泄露，并使应用程序的配置在不同的开发和部署阶段更易于管理。

## 参考资料

- [\`os\` - Miscellaneous operating system interfaces](https://docs.python.org/3/library/os.html) — Python core developers (2024)
  Python官方文档，说明如何与操作系统交互，包括使用`os.getenv()`方法获取环境变量。
- [How to manage your API keys](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety) — OpenAI (2025)
  Publisher: OpenAI
  OpenAI关于安全处理API密钥的建议，强调使用环境变量进行密钥存储和提供通用安全指导。
- [gitignore(5) Manual Page](https://git-scm.com/docs/gitignore) — Git project contributors (2023)
  Git官方文档，解释如何使用`.gitignore`文件阻止特定文件（例如包含敏感信息的`.env`文件）提交到版本控制。
- [OWASP Top 10 Web Application Security Risks](https://owasp.org/www-project-top-ten/) — OWASP Foundation (2021)
  Publisher: OWASP Foundation
  概述了关键的Web应用程序安全风险，包括敏感数据暴露和安全配置错误，强化了安全管理API密钥实践的必要性。

---

[上一节](04-%E6%A0%B8%E5%BF%83%E5%BA%93%E5%AE%89%E8%A3%85.md) · [下一节](06-%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE%E5%AE%9E%E8%B7%B5.md)

---
course: "getting-started-model-context-protocol"
chapter: "debugging-and-client-integration"
lesson: "hands-on-practice-end-to-end-integration"
sourceId: 7860
sourceUrl: "https://apxml.com/zh/courses/getting-started-model-context-protocol/chapter-4-debugging-and-client-integration/hands-on-practice-end-to-end-integration"
title: "动手实践：端到端集成"
description: "最终实践，将定制服务器连接到客户端并执行复杂任务。"
order: 6
plots: ["plots/7860-0.json"]
sourceHash: "f1106adbee184986f3232428435c30c79a9367e05f1a5bad7c9ebc892a045308"
sourceCorrections: []
---

我们已构建了一个能够查询SQLite数据库并处理工具执行请求的服务器。然而，一个独立运行的服务器，如果没有客户端来使用其资源，用处不大。在这个最后的集成练习中，我们将把您的基于Python的MCP服务器连接到Claude桌面版。此过程包括定义传输配置、安全管理环境变量，以及验证大型语言模型与您的本地数据之间的互动流程。

### 集成架构

此集成依赖于标准输入/输出 (stdio) 传输机制。与监听特定端口的HTTP服务器不同，本地MCP服务器通常由客户端应用程序作为子进程启动。客户端管理服务器的生命周期，向服务器的`stdin`发送JSON-RPC消息，并从`stdout`读取响应。

这种架构最大程度减少了本地工具的网络开销，并简化了身份验证，因为服务器在用户的本地环境下运行。以下图表展示了Claude桌面版与您的定制服务器建立会话时的层级结构和通信流程。

> 该集成拓扑展示了客户端如何读取配置以启动服务器进程并维持通信通道。

### 配置客户端

要注册您的服务器，您必须修改Claude桌面版使用的配置文件。此文件告诉应用程序要启动哪些服务器以及执行哪些命令。此文件的位置因操作系统而异：

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

用您喜欢的文本编辑器打开此文件。如果文件不存在，请创建它。您将在 `mcpServers` 键下定义一个JSON对象。每个条目代表一个独立的服务器连接。

配置要求运行的 `command`（通常是 Python 可执行文件）以及指向您脚本的 `args`。如果您的服务器依赖于 `uv` 等外部库或严格的虚拟环境，您必须指向该环境中 Python 可执行文件的绝对路径，以确保正确解决依赖问题。

假设您的项目目录中有一个名为 `database_server.py` 的服务器脚本。配置条目如下所示：

```json
{
  "mcpServers": {
    "sqlite-explorer": {
      "command": "/absolute/path/to/your/venv/bin/python",
      "args": [
        "/absolute/path/to/project/database_server.py"
      ],
      "env": {
        "DB_PATH": "/absolute/path/to/data.db",
        "LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

请注意使用绝对路径。客户端应用程序从其自身的工作目录执行命令，而不是从您脚本的目录执行。使用相对路径常常导致在连接时立即出现 `FileNotFoundError`。

### 保护环境变量

在上述配置中，`env` 字典对于传递配置数据而无需将其硬编码到源代码中非常重要。对于基于API的工具，例如天气获取器或股价分析器，您会将API密钥放在此处。

当客户端启动服务器进程时，它会将这些变量注入到进程环境中。您的Python代码使用标准的 `os` 模块访问它们：

```python
import os

db_path = os.environ.get("DB_PATH")
if not db_path:
    raise ValueError("DB_PATH environment variable is required")
```

这种分离确保敏感凭据保留在您的本地配置文件中，并且不会随您的服务器代码提交到版本控制系统。

### 验证连接生命周期

配置保存后，重启Claude桌面版。初始化时，应用程序会解析配置文件并尝试与您的服务器进行握手。这包括发送一个 `initialize` JSON-RPC请求。如果您的服务器正确响应其能力和工具定义，则连接建立。

您可以通过查看集成图标来验证Claude桌面版界面中的连接状态。如果连接失败，客户端通常会创建一个日志文件。检查这些日志是调试启动问题的主要方式。

这种互动的效率由请求-响应循环的延迟决定。工具执行的总时间 $T_{total}$ 可以建模为：

$T_{total} = T_{proc} + T_{exec} + T_{model}$

其中 $T_{proc}$ 是 JSON 序列化/反序列化开销，$T_{exec}$ 是您的 Python 函数运行所需的时间（例如，查询数据库），而 $T_{model}$ 是 LLM 处理结果所需的时间。

下表可视化了本地工具调用的典型延迟分布。请注意，虽然在本地 stdio 环境下网络延迟可忽略不计，但模型处理时间通常是互动中的主要部分。



![本地MCP工具调用延迟分布](plots/7860-0.json)



> 单个同步工具执行周期中延迟组成部分的细分。

### 执行首次查询

服务器运行后，您现在可以使用自然语言与其互动。MCP协议向用户隐藏了工具执行的细节。您无需输入特定命令；相反，您只需表达意图。

在Claude中输入以下提示：
*“请检查数据库 schema，并列出 'products' 表中的前五个条目。”*

在幕后，会发生以下序列：

1. **工具选择：** LLM 分析您的请求，并将其与您的服务器提供的工具定义进行匹配（例如，`list_tables`，`query_db`）。
2. **调用请求：** 客户端向您的服务器发送一个 `tools/call` 请求，其中包含必要的参数 (parameter)（例如，`SELECT * FROM products LIMIT 5`）。
3. **执行：** 您的服务器接收请求，对 SQLite 文件执行 SQL，并格式化结果。
4. **结果返回：** JSON 数据被发送回客户端。
5. **合成：** LLM 解释原始 JSON 数据，并生成一个描述产品的可读响应。

### 常见集成问题

当集成未能立即生效时，原因通常在以下三个方面之一：

1. **路径解析：** 最常见的错误。确保 Python 可执行文件的路径是绝对路径，并且脚本路径正确。
2. **模块导入：** 如果您的脚本导入了 `mcp` 或 `pydantic`，但 Python 命令指向的是系统 Python 而非您的虚拟环境，服务器将在启动时无声崩溃。
3. **Stdio 污染：** 您的服务器必须严格通过 `stdout` 上的 JSON-RPC 进行通信。如果您的代码中为了调试目的而包含 `print()` 语句，它们将破坏 JSON 流并导致连接失败。调试输出应始终使用 `logging` 模块写入 `stderr` 或文件。

通过成功完成此集成，您已将一个独立的脚本转变为一个上下文 (context)提供者，它扩展了通用 LLM 的能力。这种模式为构建更复杂的助手奠定了根基，使其能够与专有数据集和内部 API 互动。

## 参考资料

- [subprocess - Subprocess management](https://docs.python.org/3/library/subprocess.html) — Python Software Foundation (2024)
  Python `subprocess` 模块的官方文档，解释了如何启动进程、连接其输入/输出管道以及管理其生命周期，这是标准输入输出集成架构的核心。
- [os - Miscellaneous operating system interfaces](https://docs.python.org/3/library/os.html) — Python Software Foundation (2024)
  Python `os` 模块的官方文档，涵盖了 `os.environ` 和访问环境变量的方法，该方法用于安全管理配置数据和 API 密钥。
- [Toolformer: Language Models That Can Use Tools](https://arxiv.org/abs/2302.04761) — Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom (2023)
  Journal: arXiv preprint arXiv:2302.04761; DOI: [10.48550/arXiv.2302.04761](https://doi.org/10.48550/arXiv.2302.04761)
  一篇基础研究论文，介绍了一种通过简单 API 调用训练语言模型使用外部工具的方法，为大型语言模型的工具选择和调用请求过程提供了背景。

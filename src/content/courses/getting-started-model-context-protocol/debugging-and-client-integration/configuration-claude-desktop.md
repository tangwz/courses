---
course: "getting-started-model-context-protocol"
chapter: "debugging-and-client-integration"
lesson: "configuration-claude-desktop"
sourceId: 7857
sourceUrl: "https://apxml.com/zh/courses/getting-started-model-context-protocol/chapter-4-debugging-and-client-integration/configuration-claude-desktop"
title: "Claude Desktop 的配置"
description: "修改配置文件，以便在 Claude Desktop 中登记本地服务器。"
order: 3
plots: []
sourceHash: "e0b0aa0ce04c916408b81384d5ebcbfb1be63b4dcc9f67b19e4c7ce0d59387fb"
sourceCorrections: []
---

使用 MCP 检查器确认服务器逻辑后，接下来就是将其与生产级客户端结合。对于本地开发和日常使用，Claude Desktop 作为主要主机应用。它管理 MCP 服务器的生命周期，将其作为子进程启动，并通过标准输入输出 (stdio) 进行通信。

要登记服务器，您必须修改位于您主机上的一个特定配置文件。这个 JSON 文件作为登记簿，告知客户端运行哪个可执行文件、传递哪些参数 (parameter)，以及向服务器环境注入哪些环境变量。

### 查找配置文件

该配置文件名为 `claude_desktop_config.json`。其位置取决于您的操作系统。如果文件和目录结构尚不存在，您可能需要创建它们。

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

您可以通过 Claude Desktop 界面快速访问此文件，方法是选择设置菜单（通常由首字母或通用图标表示），然后选择 **设置** > **开发者** > **编辑配置**。此操作会在您的默认文本编辑器或文件管理器中打开该文件。

### mcpServers 定义

配置的核心是 `mcpServers` 对象。此对象中的每个属性代表一个不同的服务器连接。您选择的属性名称作为该服务器在 Claude 界面中的标识符。

与每个服务器关联的值是一个包含 `command`、一个 `args` 数组和一个可选的 `env` 字典的对象。

以下是一个基本 Python 服务器配置的结构：

```json
{
  "mcpServers": {
    "my-local-data": {
      "command": "python3",
      "args": [
        "/Users/username/projects/mcp-server/main.py"
      ]
    }
  }
}
```

Claude Desktop 初始化时，会解析此文件并尝试执行 `python3 /Users/username/projects/mcp-server/main.py`。客户端随后会监听该进程的 `stdout`，以获取 JSON-RPC 消息。

> Claude Desktop 中的配置文件解析和进程启动流程。

### 绝对路径和虚拟环境

配置中一个常见的失败之处是使用相对路径。启动进程的工作目录通常是应用包位置，而不是您的项目文件夹。因此，您必须始终为可执行文件和脚本使用绝对路径。

如果您使用 Python 进行开发，最佳做法是使用位于您的虚拟环境中的 python 可执行文件。这确保服务器能访问已安装的依赖项，如 `mcp` 或 `pydantic`。

对于使用 `uv` 或 `venv` 且位于 `.venv` 中的项目，配置应直接指向二进制文件：

```json
{
  "mcpServers": {
    "sqlite-explorer": {
      "command": "/Users/username/projects/sqlite-tool/.venv/bin/python",
      "args": [
        "/Users/username/projects/sqlite-tool/server.py",
        "--db-path",
        "/Users/username/data/Chinook.db"
      ]
    }
  }
}
```

对于通过 `npm` 发布或使用 `npx` 运行的 TypeScript 服务器，命令需要 node 可执行文件的完整路径，或一个可以解析该命令的 shell 包装器。然而，通常通过使用 `node` 的绝对路径和构建输出可以实现更简单的执行：

```json
{
  "mcpServers": {
    "filesystem-tool": {
      "command": "node",
      "args": [
        "/Users/username/projects/fs-mcp/dist/index.js",
        "/Users/username/allowed-directory"
      ]
    }
  }
}
```

### 注入环境变量

服务器通常需要身份验证凭据，例如用于外部服务（例如 GitHub、Brave Search 或云数据库）的 API 密钥。虽然您可以将这些硬编码到源代码中，但这样做存在安全风险。MCP 配置允许您通过 `env` 对象安全地传递这些信息。

此处定义的变量在运行时注入到进程环境中。它们在 Python 中可通过 `os.environ` 访问，在 Node.js 中可通过 `process.env` 访问。

```json
{
  "mcpServers": {
    "github-integration": {
      "command": "uvx",
      "args": [
        "mcp-server-github"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxxxxxxxxxxx",
        "LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

在此示例中，我们使用 `uvx`（`uv` 包管理器的一部分）立即下载并运行远程服务器包，提供必要的令牌来授权请求。

### 应用更改和问题处理

Claude Desktop 只在启动时读取配置文件。对 JSON 文件所做的任何更改都要求您完全退出并重新启动应用。关闭窗口通常不够；您必须终止应用进程。

重新启动后，请在输入界面中查找连接指示器，通常是一个类似插头或服务器机架的图标。点击此图标会显示您已连接服务器的状态。

#### 常见配置错误

1. **JSON 语法错误：** 尾随逗号或缺少括号将阻止文件解析。如果 Claude Desktop 无法加载服务器，请使用 Linter 或在线验证器验证您的 JSON。
2. **无声故障：** 如果服务器脚本立即崩溃（例如，由于 Python 中缺少导入或语法错误），Claude 可能不会显示明显的错误。它只是无法连接。
3. **Stdio 污染：** 如架构部分所述，该协议依赖于标准输出的干净使用。如果您的代码使用 `print()` 进行调试，这些消息会干扰 JSON-RPC 消息结构。请确保所有调试信息都写入标准错误 (`stderr`)。

要诊断无法连接的服务器，您可以在您自己的终端中，完全按照配置文件中的定义手动运行 `command` 和 `args`。

```bash
# 手动测试命令，看是否崩溃
/Users/username/projects/sqlite-tool/.venv/bin/python /Users/username/projects/sqlite-tool/server.py
```

如果命令运行并挂起（等待输入）而未抛出异常，则环境设置很可能正确。您随后可以终止它，并相信 Claude Desktop 将能成功启动它。

## 参考资料

- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification) — JSON-RPC Working Group (2010)
  定义了通过包括标准I/O在内的各种传输层进行远程过程调用的消息格式和协议标准，与MCP服务器通信密切相关。
- [\`subprocess\` - Subprocess management](https://docs.python.org/3/library/subprocess.html) — Python Software Foundation (2024)
  Journal: Python Documentation
  Python官方库文档，详细说明了如何与操作系统进程交互，包括执行、标准I/O管理以及为生成的子进程配置环境变量。
- [\`venv\` - Creation of virtual environments](https://docs.python.org/3/library/venv.html) — Python Software Foundation (2024)
  Journal: Python Documentation
  Python官方文档，提供了创建和使用虚拟环境以隔离项目依赖的说明，有助于保持服务器配置的整洁和可重复性。

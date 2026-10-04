---
course: "getting-started-model-context-protocol"
chapter: "debugging-and-client-integration"
lesson: "using-mcp-inspector"
sourceId: 7855
sourceUrl: "https://apxml.com/zh/courses/getting-started-model-context-protocol/chapter-4-debugging-and-client-integration/using-mcp-inspector"
title: "使用 MCP 检查器"
description: "使用基于网络的检查器来可视化和测试服务器端点。"
order: 1
plots: ["plots/7855-0.json"]
sourceHash: "ae96fcece85a83ae65c02f8ceff5a757b66d0d8e020ae191b76c09aeb66ff7df"
sourceCorrections: []
---

开发 MCP 服务器需要处理不可见的数据流。您的 Python 或 TypeScript 代码通过标准输入输出 (stdio) 进行通信，交换 JSON-RPC 消息，这些消息在没有拦截的情况下很难追踪。在尝试将服务器连接到像 Claude Desktop 这样的 LLM 客户端之前，您必须验证您的实现是否遵循协议并按预期运行。MCP 检查器就是这个验证层。

MCP 检查器是一个以网页界面形式运行的开发工具。它充当一个代理客户端，连接到您的服务器，允许您浏览资源、执行工具和手动调整提示。这种隔离消除了 LLM 的不确定性。如果在检查器内部工具执行失败，问题就出在您的代码中。如果它在检查器中成功但在 LLM 中失败，问题很可能出在工具描述或模式清晰度上。

### 安装和运行检查器

检查器以 Node.js 包的形式分发。您无需全局安装它；最好使用 `npx` 来执行，以确保您使用的是调试协议的最新版本。该命令结构要求您将通常用于运行服务器的命令作为参数 (parameter)传递给检查器。

对于使用 `uv` 的 Python 服务器，命令遵循以下模式：

```bash
npx @modelcontextprotocol/inspector uv run server.py
```

对于直接构建或运行的 TypeScript 服务器：

```bash
npx @modelcontextprotocol/inspector node build/index.js
```

执行后，检查器将您的服务器作为子进程启动。它附加到服务器的 `stdin` 和 `stdout` 流，充当传输层。同时，它会启动一个本地 Web 服务器（通常在 `localhost:5173` 上），您可以在其中与用户界面进行交互。

> 调试会话的架构，显示了用户界面、检查器代理和目标服务器之间的分离。

### 验证资源

网页界面加载后，检查器会启动 `initialize` 握手。如果成功，"资源" 选项卡将填充您的服务器暴露的数据端点列表。此视图验证了您的实现的两个重要方面。首先，它确认 `resources/list` 处理程序返回了正确的元数据结构。其次，它允许您测试内容获取逻辑。

点击特定资源会发送带有特定 URI 的 `resources/read` 请求。检查器会在主视图中渲染返回的内容（文本或二进制）。这对于调试动态资源特别有用。例如，如果您的资源 URI 包含 `file:///logs/{date}` 这样的参数 (parameter)，检查器允许您验证您的模式匹配逻辑是否正确路由请求并提取变量。

### 测试工具执行

工具通常包含 MCP 服务器中最复杂的逻辑。检查器会为您的模式中定义的每个工具生成一个动态表单。此表单严格源自您在代码中提供的 `inputSchema` JSON 模式。

如果表单中缺少必填字段，或者某个字段本应是整数却允许字符串，这表明您的 Pydantic 模型或 Zod 定义存在错误。您可以将测试值输入这些表单并执行工具。检查器会发送 `tools/call` 请求并显示结果。

这种手动执行是必要的，用于捕获不会触发异常的逻辑错误。例如，一个数据库查询工具可能在技术上成功（返回 200 OK 状态），但由于 SQL 语句格式错误而返回空列表。通过检查检查器中的原始 JSON 输出，您可以区分“未找到数据”的场景和静默失败。

> 
> 
> ![调试会话中的流量类型分布](plots/7855-0.json)
> 
> 
>
> 服务器调试会话期间观察到的典型请求类型分类。

### 追踪 JSON-RPC 消息

检查器最强大的功能是连接日志。该日志位于底部面板或专用选项卡中，记录了检查器与您的服务器之间交换的每个 JSON-RPC 消息。

调试时，您会在日志流中寻找三个特定组件：

1. **请求结构：** 验证方法名称（例如 `tools/call`）和 `params` 对象。确保参数 (parameter)正确嵌套。
2. **响应匹配：** 每个请求都带有一个唯一的 `id`（整数或字符串）。服务器的响应必须带有相同的 `id`。如果 ID 不匹配，客户端将丢弃该响应。
3. **错误传播：** 如果您的服务器崩溃或抛出未处理的异常，检查器应接收一个标准的 JSON-RPC 错误对象。

请看标准错误格式：


$$
Error_{response} = \{ "code": \text{整数}, "message": \text{字符串}, "data": \text{可选} \}
$$


如果您的服务器直接将 Python 堆栈跟踪打印到 `stdout`，它会违反 JSON 结构，检查器将报告解析错误。日志允许您精准定位服务器输出偏离 JSON 规范的位置。

### 处理标准输入输出噪音

集成过程中一个常见问题是“标准输入输出噪音”。当您的服务器代码或依赖项将调试信息（例如 `print("Connecting to DB...")`）打印到标准输出时，就会发生这种情况。由于 MCP 使用标准输出进行协议通信，这些纯文本字符串会破坏 JSON 流。

检查器分离了 `stdout`（协议）和 `stderr`（日志）。要修复检查器中出现的解析错误，请确保您的所有应用程序日志都导向 `stderr`。在 Python 中，使用配置为写入 `sys.stderr` 的 `logging` 模块。在 Node.js 中，使用 `console.error()` 用于日志，并将 `console.log()` 严格保留给 MCP 消息，如果您正在手动管理传输（尽管推荐使用 SDK 抽象）。

### 管理环境变量

检查器在其自己的进程环境中运行。除非特定环境变量（如 API 密钥）存在于您当前的 shell 会话中，否则它不会自动继承。如果您的服务器在检查器下运行时无法与外部服务认证，检查环境配置是第一步。

您可以在启动检查器时内联传递环境变量：

```bash
# Linux/macOS
OPENAI_API_KEY=sk-... npx @modelcontextprotocol/inspector uv run server.py

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."; npx @modelcontextprotocol/inspector uv run server.py
```

通过使用检查器系统地验证资源、工具和消息完整性，您可以确保服务器足够强大，可以应对后续集成阶段中 LLM 驱动交互的不可预测性。

## 参考资料

- [Model Context Protocol (MCP) Official Website and Specification](https://www.modelcontextprotocol.org/) — Model Context Protocol Community (2024)
  提供官方规范、工具信息以及实现 MCP 服务器和客户端的指南，包括与 Inspector 相关细节。
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification) — JSON-RPC Working Group (2010)
  定义了使用 JSON 的远程过程调用协议，这是 MCP Inspector 与服务器之间通信的基础。
- [JSON Schema: A vocabulary for validating JSON documents](https://json-schema.org/specification/2020-12/) — JSON Schema Working Group (2020)
  JSON Schema 的官方文档，用于定义 MCP 服务器中工具输入的结构和验证规则。
- [\`logging\` - Logging facility for Python | Python Documentation](https://docs.python.org/3/library/logging.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  Python `logging` 模块的官方文档，提供配置日志的指导，包括将输出定向到标准错误以避免损坏标准输出流。

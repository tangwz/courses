---
course: "getting-started-model-context-protocol"
chapter: "debugging-and-client-integration"
lesson: "analyzing-log-output"
sourceId: 7856
sourceUrl: "https://apxml.com/zh/courses/getting-started-model-context-protocol/chapter-4-debugging-and-client-integration/analyzing-log-output"
title: "分析日志输出"
description: "在开发周期中解读服务器日志和错误追踪。"
order: 2
plots: ["plots/7856-0.json"]
sourceHash: "0ec9ce571661c10b7ac0dd6670fdf3a0c6d53c1fe47094a8bda4d09d7935f211"
sourceCorrections: []
---

MCP 中有效的日志记录策略与标准 Web 服务器开发有很大不同。在传统的 REST API 环境中，开发者通常查看控制台输出或追踪日志文件来追踪执行过程。在通过标准输入/输出 (stdio) 运行的 MCP 服务器中，标准输出流本身就是网络连接。将原始文本或调试语句写入标准输出会破坏 JSON-RPC 流，导致连接立即中断。

为了在获取诊断信息的同时保持连接稳定，您必须将日志导向不干扰协议传输的特定通道。这包括使用标准错误 (stderr) 进行系统级诊断，以及使用 MCP 协议自身的日志记录功能处理面向客户端的信息。

### 传输限制

开发过程中服务器立即终止的最常见原因是意外使用了打印语句。当 MCP 客户端建立连接时，它期望从服务器标准输出接收的每一行都是有效的 JSON-RPC 消息。

如果您的代码执行 `print("Starting server...")`，客户端将接收到这个原始字符串，无法将其解析为 JSON，并引发验证错误。为避免这种情况，所有非协议输出都必须重定向。

以下图表说明了在 MCP 架构中数据流必须如何分离。JSON-RPC 流量独占标准输出通道，而日志则通过标准错误或封装的通知传输。

> 通信通道的分离可以防止协议损坏。协议消息和封装的日志通知共享标准输出，而原始调试文本使用标准错误。

### 使用标准错误 (stderr)

标准错误流 ($stderr$) 是低级别系统日志、堆栈追踪和未处理异常的主要接收端。MCP 客户端通常会捕获此流并将其写入主机上的专用日志文件。

在 Python 中，您可以配置根日志记录器自动写入 $stderr$。这可以确保即使您使用的库尝试记录信息，也不会意外写入 $stdout$。

设置日志记录配置时，请确保流处理器明确指向 $stderr$：

```python
import logging
import sys

# 配置日志写入 stderr
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

logger = logging.getLogger("mcp_server")
logger.info("Server initialization complete") # 安全
```

分析 $stderr$ 中的日志时，您经常会看到应用程序的原始内部状态。当服务器启动失败或静默崩溃时，您就应该查看这里。由于这些日志绕过了 JSON-RPC 解析器，即使协议握手未能完成，它们也仍然可见。

### 协议自带日志记录

对于旨在由客户端界面内的用户或 AI 助手查看的消息，MCP 规定了一种特定的通知方法：`notifications/message`。

与直接写入 $stderr$ 不同，这些日志是经由 $stdout$ 发送的结构化 JSON 对象。它们被封装在有效的 JSON-RPC 信封中，因此不会破坏协议。客户端会处理这些通知，并可以在调试控制台或检查器窗口中显示它们。

日志通知的结构包括一个严重级别和要记录的数据：

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/message",
  "params": {
    "level": "info",
    "logger": "database_tool",
    "data": "Query executed successfully: SELECT * FROM users"
  }
}
```

使用 MCP SDK，您通过服务器上下文 (context)而非标准打印函数发送这些日志。此方法使得客户端能够根据严重性过滤消息，类似于 syslog 处理日志级别的方式。

### 解读 JSON-RPC 错误

当传输正常但请求失败时，服务器会返回 JSON-RPC 错误响应。分析这些错误需要了解规范中定义的错误代码。

一个标准错误响应如下所示：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": "Missing required argument: 'query'"
  }
}
```

`code` 整型数值表明了失败的类别。标准范围包括：

- **-32700**：解析错误。服务器收到了无效的 JSON。
- **-32600**：无效请求。JSON 结构不是有效的 Request 对象。
- **-32601**：方法未找到。客户端请求的工具或资源不存在。
- **-32602**：参数 (parameter)无效。提供给工具的参数与 Pydantic 模式不匹配。
- **-32603**：内部错误。工具处理器内部发生未处理的异常。

调试时，请查看 `data` 字段。良好实现的服务器会用具体信息填充此字段，例如缺少哪个参数或发生的异常文本。

### 调试连接生命周期

连接问题常出现在会话的边界，即启动和关闭时。由于客户端管理服务器进程，与进程生成相关的日志位于客户端自己的应用程序日志中，而非服务器的输出。

如果您的服务器未出现在客户端集成列表中，请检查客户端日志中的退出代码。

- **退出代码 0**：服务器正常关闭。
- **退出代码 1**：服务器崩溃并出现错误。请检查捕获到的 $stderr$ 输出。
- **退出代码 127/126**：命令未找到或不可执行。这通常表明您的配置文件中存在路径问题，客户端无法找到 Python 或 Node 可执行文件。

以下图表呈现了在工具失败的典型调试会话中，您可能遇到的日志消息的数量和类型。请注意传输消息（心跳）如何占据主导流量，而实际错误则稀疏但值得注意。



![按消息类型划分的日志量分布（调试会话）](plots/7856-0.json)



> 在典型会话中，传输流量（蓝色）是恒定的。协议错误（红色）以与应用程序日志事件（粉色）同时出现的峰值形式呈现，有助于准确找出失败的确切时刻。

### 追踪异步操作

MCP 服务器通常会并发处理多个请求，特别是在使用 SSE（Server-Sent Events）传输或按顺序调用多个工具时。在这些情况下，线性的日志文件可能会令人困惑，因为来自不同请求的日志条目会交错出现。

为了在异步环境中高效地分析日志，您应该追踪请求 ID。每个 JSON-RPC 请求都包含一个独一无二的 `id` 字段。在工具处理器或资源读取器内部记录日志时，请在日志消息中包含此 ID。

如果您使用 Python 的 `contextvars`，可以在请求的入口点存储请求 ID，并将其自动注入到该执行上下文 (context)中创建的每个日志记录中。这使得您可以针对特定 ID 过滤日志文件（例如，`grep "req-123" server.log`），并重构单个交互的完整流程，同时忽略来自其他并发操作的干扰信息。

通过强制严格分离流并了解协议返回的错误代码，您可以将不透明的连接失败转变为可解决的逻辑问题。下一步是配置客户端应用程序，使其在正确的环境设置下执行您的服务器。

## 参考资料

- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification) — JSON-RPC Working Group (2010)
  描述 JSON-RPC 协议，包含请求/响应格式、错误代码和通知机制。
- [logging - Logging facility for Python](https://docs.python.org/3/library/logging.html) — Python Software Foundation (2024)
  Python 官方日志模块配置指南，包含流处理程序和指向 `sys.stderr`。
- [contextvars - Context Variables](https://docs.python.org/3/library/contextvars.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  解释 `contextvars` 在异步 Python 应用程序中管理上下文局部状态。

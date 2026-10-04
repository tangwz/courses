---
course: "getting-started-model-context-protocol"
chapter: "architecture-and-fundamentals"
lesson: "capabilities-negotiation"
sourceId: 7841
sourceUrl: "https://apxml.com/zh/courses/getting-started-model-context-protocol/chapter-1-architecture-and-fundamentals/capabilities-negotiation"
title: "能力协商"
description: "客户端和服务器如何在初始化阶段交换能力。"
order: 5
plots: []
sourceHash: "315dd4402cb7ccc13ca2c8d65a3fd6a8ea74583deab50582e8b7c35e9f851dd7"
sourceCorrections: []
---

建立MCP客户端和服务器之间的连接，不仅需要传输层。由于协议允许客户端（如IDE或AI桌面应用）和服务器（如数据库连接器）独立发展，双方通常具有不同的功能集合。一个服务器可能支持活动资源订阅，而另一个只支持静态读取。一个客户端可能允许服务器请求抽样（通过LLM生成内容），而另一个可能出于安全原因限制此操作。

为应对这种差异，协议规定了一个严格的初始化阶段。此阶段如同合同协商，双方在此声明各自的功能。在此握手完成之前，任何功能性请求，如列出工具或读取资源，都无法进行。

### 初始化流程

MCP连接的生命周期在传输链接（Stdio或SSE）建立后立即开始。握手由一个同步的请求-响应对，以及随后的一个通知组成。这个严格有序的流程确保在应用逻辑启动前，配置已确定。

1. **请求：** 客户端发送一个 `initialize` 请求，其中包含其协议版本和能力。
2. **响应：** 服务器回复其协议版本和自身能力。
3. **通知：** 客户端发送 `notifications/initialized` 消息，表明它已接受服务器的响应并准备好操作。

如果此流程中任何步骤失败，或协议版本不兼容，连接将中断。

> 建立活跃MCP会话所需的三步初始化流程。

### 声明能力

初始化有效载荷中的 `capabilities` 属性是功能检测的核心机制。它是一个JSON对象，其中键代表标准化功能。某个键的存在意味着对该功能的支持。如果某个键缺失，另一方必须假定该功能不可用，并禁用相应的逻辑。

#### 客户端能力

当客户端发起连接时，它会声明自己能处理什么。这可以防止服务器发送客户端无法处理的通知或请求。常见的客户端能力包括：

- **抽样：** 表明客户端允许服务器通过 `sampling/createMessage` 请求LLM补全（使用客户端的推理 (inference)提供程序）。
- **根目录：** 表明客户端可以向服务器提供文件系统根目录，这对于需要知道允许访问哪些目录的服务器很有用。
- **实验性：** 一个保留区域，用于正在测试的非标准功能。

#### 服务器能力

服务器的响应决定了LLM可用的API表面。标准服务器能力包括：

- **日志：** 表明服务器会发出日志消息，客户端应捕获并显示。
- **提示：** 表明服务器提供提示模板（例如，`prompts/list`）。
- **资源：** 表明服务器提供数据资源。此对象可以进一步指定是否支持 `subscribe`（活动更新）。
- **工具：** 表明服务器提供可执行功能（例如，`tools/call`）。

### 能力数据结构

能力对象的结构是分层的。简单的布尔值通常不够用，因为特定功能常有自己的子配置。

考虑以下服务器在握手期间发送的JSON-RPC有效载荷示例：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "resources": {
        "subscribe": true,
        "listChanged": true
      },
      "tools": {},
      "logging": {}
    },
    "serverInfo": {
      "name": "sqlite-explorer",
      "version": "1.0.2"
    }
  }
}
```

在此响应中，服务器明确声明支持**资源**、**工具**和**日志**。请注意，`prompts` 未出现。客户端必须据此推断该服务器不提供提示模板。此外，在 `resources` 能力中，服务器将 `"subscribe": true`。这表明客户端发送 `resources/subscribe` 请求是有效的。如果此标志为false或缺失，客户端将把资源视为静态。

### 版本协商

MCP规范会随着时间演变。为保持稳定性，初始化请求中包含版本协商机制。

1. 客户端在 `protocolVersion` 字段中发送其支持的版本（例如，`"2024-11-05"`）。
2. 服务器将其与自身实现进行比较。
3. 服务器回复协商后的版本。

数学上，如果 $V_c$ 是客户端版本，而 $V_s$ 是服务器最大支持版本，则商定的协议版本 $V_{final}$ 通常由以下公式确定：

$V_{最终版本} = \text{选择兼容版本}(V_c, V_s)$

通常，如果服务器能够支持客户端请求的版本，它将接受连接。如果客户端请求的版本比服务器已知的新，服务器可能会拒绝连接或回复其最高的已知版本，并由客户端处理向后兼容性。

### 处理功能不匹配

健壮的MCP实现必须处理“优雅降级”。你不能仅仅因为你的代码预期某个能力存在就假定它存在。

例如，如果你正在构建一个可视化服务器日志的客户端，你必须首先检查服务器的握手响应中是否存在 `logging` 能力。

```python
# 客户端处理协商的伪代码逻辑
server_caps = response.result.capabilities

if "logging" in server_caps:
    enable_log_panel()
else:
    hide_log_panel()

if "resources" in server_caps and server_caps["resources"].get("subscribe"):
    enable_live_updates()
else:
    enable_poll_button()
```

这种模式将UI/UX与后端实现分离。它让先进的客户端仍能从简单的服务器中获取价值，也让功能强大的服务器在连接到基本客户端时仍能运作（尽管功能受限）。

### 故障模式

初始化阶段是连接问题的主要筛选器。此阶段常见的故障情况包括：

- **传输超时：** 客户端通过Stdio连接，但在设定的时间内（通常10-30秒）未收到 `initialize` 请求的响应。这通常表明服务器进程在启动时崩溃，或正在等待永不抵达的输入。
- **JSON-RPC解析错误：** 服务器在握手完成前向 `stdout` 发出非JSON调试文本。这会损坏传输流，导致客户端解析失败。
- **版本不兼容：** 客户端需要服务器实现不支持的较新协议功能。

通过严格执行此协商阶段，MCP得以保证后续的交互（可能涉及传递敏感数据或执行副作用）在一个明确且商定的上下文 (context)环境中发生。

## 参考资料

- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification) — JSON-RPC Working Group (2010)
  定义了使用 JSON 进行远程过程调用通信的标准，这是 MCP 描述的消息结构、请求/响应和通知模式的基础。
- [Language Server Protocol Specification](https://microsoft.github.io/language-server-protocol/specifications/lsp/latest/specification/) — Microsoft and Red Hat (Latest version)
  提供了一个广泛采用的客户端-服务器协议的实际示例，该协议像 MCP 一样，严重依赖于“初始化”请求和能力协商。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491927029/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  提供了构建可靠分布式系统的基本原则，涵盖了数据模型演进、向后和向前兼容性以及 API 版本控制策略等主题，这些对于有效的能力协商至关重要。

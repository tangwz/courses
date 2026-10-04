# JSON-RPC 消息结构

来源：[原文](https://apxml.com/zh/courses/getting-started-model-context-protocol/chapter-1-architecture-and-fundamentals/json-rpc-message-structure)

[返回章节目录](README.md) · [返回课程目录](../README.md)

模型上下文 (context)协议 (Model Context Protocol) 不同组成部分之间的高效沟通，需要一套严格且商定的语言。传输层负责比特数据通过套接字或标准输入流的传输，而应用层则依赖结构化格式来理解意图。MCP 使用 JSON-RPC 2.0，这是一种无状态、轻量级的远程过程调用 (RPC) 协议，采用 JSON 编码。

此格式定义了客户端如何请求数据、服务器如何响应以及如何报告错误。由于 JSON-RPC 是传输无关的，因此无论您是通过 HTTP 与服务器发送事件通信，还是通过本地标准 I/O 管道传输数据，消息结构都保持一致。理解这种结构对实现任何 MCP 服务器都很重要，因为标准的调试工具和检查器会以这种原始格式显示流量。

### JSON-RPC 2.0 标准

JSON-RPC 通过交换数据对象来运作。MCP 连接中发送的每条消息都是一个 JSON 对象，包含特定字段，这些字段确定其用途。主要有三种消息类型：请求 (Requests)、响应 (Responses) 和通知 (Notifications)。

所有消息都必须遵循 2.0 版本规范。这是通过在每个对象中包含一个特定成员来强制执行的：

```json
"jsonrpc": "2.0"
```

如果此字段缺失或包含不同的版本号，接收方通常会拒绝该消息。此版本标签确保了协议在演进时的未来兼容性。

### 请求的构成

请求是由客户端发起的一次调用（或偶尔由服务器在双向采样中发起），用于在远程方调用特定方法。一个有效的请求对象必须包含四个属性：

1. **jsonrpc**: 必须精确为 `"2.0"`。
2. **method**: 一个字符串，包含要调用的方法的名称。MCP 定义了标准方法，例如 `resources/list` 或 `tools/call`。
3. **params**: 一个结构化值（对象或数组），用于保存方法调用期间要使用的参数 (parameter)值。如果方法不接受任何参数，则可以省略此项。
4. **id**: 发送方建立的标识符。这可以是一个字符串、一个数字或 null。

`id` 成员在功能上很重要。因为 MCP 连接通常是异步的，多个请求可能同时进行。`id` 允许发送方将传入的响应与触发它的原始请求匹配起来。

设想一个场景，MCP 客户端请求服务器执行一个名为“get-weather”的工具。其负载将如下所示：

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "get-weather",
    "arguments": {
      "city": "San Francisco",
      "unit": "celsius"
    }
  },
  "id": 1
}
```

在这种结构中，`method` 指示服务器执行工具逻辑，`params` 提供该工具模式定义的必要参数。

> 这种交换模式依赖于标识符，以在无状态 JSON 传输中保持状态。通知允许单向数据流，而不会阻塞。

### 响应的构成

当服务器收到请求时，它必须用一个响应对象进行回复。响应表明操作是成功还是失败。响应对象必须包含 `jsonrpc` 版本以及与其应答请求相同的 `id`。

很重要的一点是，响应对象必须且只能包含两个可能的结果成员之一：`result` 或 `error`。它们不能同时作为同一响应对象的成员。

#### 成功的响应

如果方法执行成功完成，响应对象会包含一个 `result` 成员。此成员的值由调用的方法决定。

继续天气工具的例子，一个成功的响应将如下所示：

```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Current temperature in San Francisco is 18°C"
      }
    ]
  },
  "id": 1
}
```

客户端收到此对象，注意到 `id` 为 `1`，将其与其内部对原始请求的跟踪进行匹配，并处理 `result`。

#### 错误响应

如果方法失败，可能是工具名称不存在或参数 (parameter)无效，服务器会返回一个 `error` 对象而不是 `result`。

`error` 对象有以下必需字段：

- **code**: 一个整数，指示错误类型。
- **message**: 错误的简明字符串描述。
- **data** (可选): 额外的原始或结构化数据，包含详细的错误信息。

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": "Missing required argument: 'city'"
  },
  "id": 1
}
```

JSON-RPC 定义了一系列保留错误代码，范围从 -32768 到 -32000。例如，-32700 表示解析错误（收到了无效的 JSON），-32601 表示方法未找到。在构建您的 MCP 服务器时，您应酌情将内部应用程序异常映射到这些标准化代码，或对特定于实现的故障使用通用代码。

### 通知与单向通信

并非所有交互都需要响应。在许多 MCP 工作流中，一方需要发送信息，而无需等待确认。这通过通知来处理。

通知在结构上与请求相同，但有一个明显区别：它不包含 `id` 成员。当接收方处理一个缺少标识符的对象时，它将其视为通知。接收方执行该方法，但不回复。

通知在 MCP 中常用于日志记录和状态更新。例如，如果服务器想向客户端控制台记录调试消息，它会发送一个 `notifications/message`：

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/message",
  "params": {
    "level": "info",
    "data": "Database connection established"
  }
}
```

因为没有 `id`，服务器不会阻塞等待客户端确认接收。这对于可以接受消息丢失的大量遥测或进度更新来说是高效的。然而，开发人员必须注意，不要将通知用于需要确认以确保数据完整性的操作。

### 消息批处理

JSON-RPC 2.0 支持批处理，允许客户端同时发送请求对象数组。服务器随后会用一个响应对象数组进行回复。尽管 MCP 规范主要依赖单个消息交互，以简化典型的 LLM 聊天循环，但如果传输层聚合了对象，服务器实现应准备好解析对象数组。

处理批处理时，服务器独立处理每个请求。响应数组中元素的顺序不保证与请求数组匹配。相反，客户端必须完全依赖 `id` 字段将响应映射回其请求。如果批处理包含通知，这些项将不会在响应数组中生成对应的条目。

理解这些结构提供了编写后续章节中讨论的逻辑层所需的语法。随着拓扑结构和消息格式的明确，下一个要考虑的是在进程间传输这些 JSON 对象的物理传输机制。

## 参考资料

- [Building Microservices: Designing Fine-Grained Systems](https://www.oreilly.com/library/view/building-microservices-2nd/9781492070329/) — Sam Newman (2021)
  Publisher: O'Reilly Media
  讨论多种服务间通信模式，包括 RPC，并有助于理解分布式系统中轻量级协议的架构选择。

---

[上一节](02-%E5%AE%A2%E6%88%B7%E7%AB%AF-%E5%AE%BF%E4%B8%BB-%E6%9C%8D%E5%8A%A1%E5%99%A8%E6%8B%93%E6%89%91.md) · [下一节](04-%E4%BC%A0%E8%BE%93%E6%9C%BA%E5%88%B6%EF%BC%9A%E6%A0%87%E5%87%86%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA%EF%BC%88Stdio%EF%BC%89%E5%92%8C%E6%9C%8D%E5%8A%A1%E5%99%A8%E5%8F%91%E9%80%81%E4%BA%8B%E4%BB%B6%EF%BC%88SSE%EF%BC%89.md)

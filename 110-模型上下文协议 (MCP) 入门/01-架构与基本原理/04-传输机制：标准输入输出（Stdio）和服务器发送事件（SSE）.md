# 传输机制：标准输入输出（Stdio）和服务器发送事件（SSE）

来源：[原文](https://apxml.com/zh/courses/getting-started-model-context-protocol/chapter-1-architecture-and-fundamentals/transport-mechanisms-stdio-sse)

[返回章节目录](README.md) · [返回课程目录](../README.md)

模型上下文 (context)协议依赖于消息格式与传输方式的明确分离。JSON-RPC 2.0 定义了各方之间交换数据的结构，而传输层则决定了这些消息如何在网络线路或系统边界间传递。

理解传输机制很有必要，因为它会影响您的服务器部署方式、认证管理以及连接故障的排查。MCP目前规定了两种主要传输类型：标准输入输出（Stdio），用于本地进程整合；以及服务器发送事件（SSE），用于远程基于HTTP的通信。

### Stdio 传输：本地进程通信

Stdio传输是连接MCP服务器到本地客户端的默认机制，例如Claude桌面应用程序或本地运行的IDE。这种模式下，客户端作为父进程，启动MCP服务器作为子进程。

通信直接通过操作系统的标准流进行。这建立了客户端与服务器实例之间的一对一关系。当客户端需要发送请求时，它将JSON-RPC消息写入服务器的标准输入（`stdin`）。服务器处理此消息并将响应写入其标准输出（`stdout`）。

由于协议消息严格通过`stdin`和`stdout`传输，任何打印到`stdout`的不相关文本都会损坏JSON流并导致协议错误。因此，开发者必须使用标准错误（`stderr`）来记录所有日志和诊断信息。

> Stdio传输中的消息流将协议数据与诊断日志隔离，以避免流损坏。

#### Stdio的特点

这种传输方法为本地开发和单用户工具提供了特定优势：

1. **安全性：** 服务器在用户本地机器上运行。它不向本地网络或互联网暴露端口，减少了攻击面。
2. **认证：** 由于客户端直接启动进程，很少需要明确的认证令牌。操作系统的用户权限处理访问控制。
3. **生命周期管理：** 服务器的生命周期与客户端绑定。当客户端应用程序关闭时，连接断开，服务器进程自动终止。

### SSE 传输：远程通信

服务器发送事件（SSE）允许MCP服务器远程运行或在与客户端分离的容器化环境中运行。在构建分布式系统时，这种传输方式是必需的，即一个中心服务器为多个客户端提供上下文 (context)，或者当服务器需要客户端本地机器上没有的资源（如高性能GPU或特定数据库访问）时。

与提供单一双向套接字的WebSocket不同，MCP SSE传输利用了两种独立的HTTP交互模式：

1. **服务器到客户端：** 客户端向SSE端点（例如`/sse`）打开一个持久的HTTP连接。服务器将JSON-RPC响应和通知作为文本事件推送到此流中。
2. **客户端到服务器：** 客户端通过标准HTTP POST调用，向服务器指定的另一个端点（例如`/messages`）发送请求。

这种解耦与标准HTTP架构保持一致。它使得POST请求可以通过标准Web工具进行负载均衡或检查，而SSE通道则处理AI交互的异步特性。

> SSE传输将流量分为用于服务器输出的持久事件流和用于客户端输入的无状态HTTP POST请求。

#### 处理远程限制

实施SSE引入了Stdio所没有的复杂情况：

- **无状态性：** HTTP本质上是无状态的。MCP服务器必须维护SSE连接ID与后续POST请求之间的映射，以确保响应被路由到正确的客户端。
- **安全性：** 与Stdio不同，服务器暴露了一个网络端口。您必须实施认证（例如API密钥或OAuth）和传输层安全（TLS）以保护传输中的数据。
- **跨域资源共享（CORS）：** 如果客户端（例如Web浏览器）在与服务器不同的域上运行，必须正确配置跨域资源共享（CORS）头，以允许连接。

### 选择合适的传输方式

选择Stdio还是SSE取决于部署架构和目标用户群体。对于个人工具、本地文件操作或与IDE配合运行的脚本，Stdio几乎总是正确的选择。对于企业级集成、共享知识库或在云环境中运行的服务，SSE是必需的。

以下图表显示了两种传输方式在实现难度和部署灵活性之间的权衡。



[交互图表：传输机制权衡分析](https://apxml.com/zh/courses/getting-started-model-context-protocol/chapter-1-architecture-and-fundamentals/transport-mechanisms-stdio-sse#plot-1h8rzfs)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "传输机制权衡分析",
    "xaxis": {
      "title": "实现难度",
      "range": [
        0,
        10
      ],
      "showgrid": true,
      "gridcolor": "#e9ecef"
    },
    "yaxis": {
      "title": "网络灵活性",
      "range": [
        0,
        10
      ],
      "showgrid": true,
      "gridcolor": "#e9ecef"
    },
    "showlegend": true,
    "plot_bgcolor": "#ffffff",
    "paper_bgcolor": "#ffffff",
    "font": {
      "family": "Helvetica"
    }
  },
  "data": [
    {
      "x": [
        2
      ],
      "y": [
        2
      ],
      "mode": "markers+text",
      "type": "scatter",
      "name": "Stdio",
      "text": [
        "Stdio"
      ],
      "textposition": "top right",
      "marker": {
        "size": 18,
        "color": "#228be6"
      }
    },
    {
      "x": [
        7
      ],
      "y": [
        9
      ],
      "mode": "markers+text",
      "type": "scatter",
      "name": "SSE",
      "text": [
        "SSE"
      ],
      "textposition": "bottom right",
      "marker": {
        "size": 18,
        "color": "#fa5252"
      }
    }
  ]
}
```

</details>



> Stdio实现难度较低，但限于本地机器使用。SSE为远程网络提供了高灵活性，但需要更复杂的（状态和安全）处理。

### 消息封装

无论选择哪种传输方式，载荷内容保持不变。MCP消息是一个JSON-RPC 2.0对象。

在Stdio中，此JSON对象被序列化为字符串，并后跟一个换行符（`\n`）以分隔消息。

$\text{消息}_{\text{stdio}} = \text{JSON}(\text{RPC}) + \text{`\n`}$

在SSE中，消息被封装在标准事件格式中。服务器发送一个事件类型（通常是`message`），后跟数据载荷。

```text
event: message
data: {"jsonrpc": "2.0", "method": "notifications/resources/list_changed", "params": ...}
```

这种一致性确保您在后续章节中构建的服务器逻辑、资源、提示和工具与传输层无关。您可以一次编写核心逻辑，并通过Stdio、SSE或两者同时暴露它，使用MCP SDK提供的相应适配器。

## 参考资料

- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification) — JSON-RPC Working Group (2010)
  定义通过各种传输协议进行远程过程调用的标准消息格式的权威规范。

---

[上一节](03-JSON-RPC%20%E6%B6%88%E6%81%AF%E7%BB%93%E6%9E%84.md) · [下一节](05-%E8%83%BD%E5%8A%9B%E5%8D%8F%E5%95%86.md)

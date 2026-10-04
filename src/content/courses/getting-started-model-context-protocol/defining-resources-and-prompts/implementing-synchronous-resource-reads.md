---
course: "getting-started-model-context-protocol"
chapter: "defining-resources-and-prompts"
lesson: "implementing-synchronous-resource-reads"
sourceId: 7845
sourceUrl: "https://apxml.com/zh/courses/getting-started-model-context-protocol/chapter-2-defining-resources-and-prompts/implementing-synchronous-resource-reads"
title: "实现同步资源读取"
description: "实现同步返回数据的处理器的代码示例。"
order: 3
plots: []
sourceHash: "032fa755d2d0710fb14c5e4b64c1eb73c40e39d86b088839fe072e7635afd748"
sourceCorrections: []
---

资源读取操作的实现是一种机制，它将静态统一资源标识符 (URI) 转换为大型语言模型 (LLM) 可访问的内容。资源列表提供了可用数据点的目录，而读取处理器则执行实际的数据获取。在同步环境下，此过程涉及接收请求、定位底层数据、正确格式化数据，并立即将其返回给客户端。

### 资源读取生命周期

当 LLM 选择要检查的资源时，客户端会发送一个使用 `resources/read` 方法的 JSON-RPC 请求。此请求包含模型希望访问的特定 URI。服务器必须将此 URI 路由到正确的内部处理器函数。

同步读取的生命周期包含三个不同的阶段：

1. **路由：** 服务器将传入的 URI 与已注册的模板或方案进行匹配。
2. **获取：** 处理器函数执行从内存、文件系统或数据库获取数据的逻辑。
3. **序列化：** 原始数据被封装在一个标准内容对象中，通常包含文本载荷和 MIME 类型。

以下图表描绘了客户端发起读取请求时的数据流。

> 同步资源读取请求的数据流，说明了路由和数据获取步骤。

### 注册读取处理器

在 Python SDK 中，处理资源读取涉及使用装饰器修饰一个接受 URI 作为参数 (parameter)的函数。SDK 管理底层的 JSON-RPC 通信，让您专注于数据获取逻辑。

处理器必须以特定格式返回内容。协议定义了一个 `ReadResourceResult`，其中包含一个内容项列表。通常，您将返回 `TextContent`，它包含 `uri`、`text` 正文和 `mimeType`。

考虑一个旨在公开系统日志的服务器。URI 方案可能遵循 `logs://system/{log_level}` 模式。实现要求定义一个函数来解析此 URI 并相应地筛选日志数据。

```python
from mcp.server.fastmcp import FastMCP

# 初始化服务器
mcp = FastMCP("LogServer")

# 模拟数据存储
LOG_DATA = {
    "error": "模块 X 发生严重故障\n数据库连接丢失",
    "info": "服务在端口 8080 启动\n健康检查通过",
    "debug": "变量状态转储: {x: 1, y: 2}"
}

@mcp.resource("logs://system/{level}")
def read_log(level: str) -> str:
    """
    读取特定严重级别的系统日志。
    """
    # 同步访问请求的数据
    content = LOG_DATA.get(level)
    
    if content is None:
        # 在内容中返回错误消息通常更安全 
        # 比为简单查询抛出异常更好
        return f"未找到级别为: {level} 的日志"
    
    return content
```

在此示例中，模式 `logs://system/{level}` 自动从传入的 URI 中提取 `level` 变量。如果客户端请求 `logs://system/error`，SDK 会调用 `read_log("error")` 并自动将返回的字符串封装在有效的资源响应对象中。

### 处理二进制和 JSON 数据

虽然文本是 LLM 上下文 (context)最常见的格式，但资源通常需要表示结构化数据或二进制内容。

对于结构化数据，最佳实践是在返回对象之前将其序列化为 JSON。这确保 LLM 收到一个语法正确的字符串，它可以轻松解析。您应将 MIME 类型设置为 `application/json`，以为模型提供关于内容结构的提示。

```python
import json

@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: str) -> str:
    # 模拟数据库查询
    user_data = {
        "id": user_id,
        "role": "admin",
        "last_login": "2023-10-27T10:00:00Z"
    }
    
    # 返回序列化的 JSON
    return json.dumps(user_data, indent=2)
```

在 JSON 输出中提供缩进会稍微增加令牌计数，但显著提高了模型的可读性，有助于模型更准确地处理数据结构。

### 动态 URI 匹配

同步读取通常需要处理未在资源目录中明确列出的 URI。您可能明确列出 `file://project/readme.md`，但您可能希望支持使用通配符或模式读取目录中的任何文件。

在实现处理器时，您必须确保 URI 的动态部分指向有效数据。这引入了一个安全考虑：路径遍历。当资源处理器接受一个作为文件路径或数据库标识符的动态参数 (parameter)时，验证是防止未经授权访问所必需的。

以下图表阐明了处理动态 URI 时所需的决策逻辑。

> 用于验证和处理动态资源参数的逻辑流。

### 同步读取中的错误处理

当资源无法读取时，服务器必须清楚地传达此故障。在 MCP 架构中，您在读取操作期间处理错误有两个主要选项：

1. **协议级错误：** 抛出异常，SDK 会将其转换为 JSON-RPC 错误响应。这向客户端表示请求本身失败或格式不正确。
2. **内容级错误：** 返回包含错误文本描述的资源。

对于 LLM 交互，第二种方法通常更佳。如果模型请求了一个不存在的文件，收到 JSON-RPC 错误可能会导致工具使用链中断，或者模型虚构失败原因。返回文本结果，例如“错误：在路径 X 未找到文件”，允许模型将错误作为上下文 (context)读取，并可能通过请求不同路径来纠正自己的错误。

```python
@mcp.resource("file://{path}")
def read_file(path: str) -> str:
    try:
        # 验证路径是否相对于安全目录
        if ".." in path or path.startswith("/"):
            raise ValueError("访问被拒绝")
            
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"错误：文件 '{path}' 不存在。"
    except ValueError as e:
        return f"错误：{str(e)}"
    except Exception:
        return "错误：读取文件时发生内部错误。"
```

### 性能

同步读取会阻塞请求处理线程。当服务器正在读取文件或查询数据库时，如果实现是单线程或阻塞的，它将无法处理该特定连接上的其他消息。

对于本地文件读取，延迟可以忽略不计。但是，如果您的资源从缓慢的外部 API 获取数据，延迟会直接增加用户等待响应的时间。

如果数据获取时间预计会很长（例如，超过 500 毫秒），您应考虑将结果缓存在内存中。由于资源是通过特定 URI 请求的，这些 URI 是极佳的缓存键。

$T_{响应} = T_{网络} + T_{处理} + T_{查找}$

通过缓存最小化 $T_{查找}$ 确保 LLM 交互的上下文 (context)组装阶段保持流畅。简单的 Python 字典或 LRU（最近最少使用）缓存装饰器是对于不经常变化的资源来说的有效策略。

## 参考资料

- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification) — JSON-RPC Working Group (2010)
  定义了 MCP 中用于通信的协议，包括方法调用和错误处理。
- [RESTful Web Services](https://www.oreilly.com/library/view/restful-web-services/9780596529260/) — Leonard Richardson and Sam Ruby (2007)
  Publisher: O'Reilly Media
  描述了使用 URI 和标准化内容类型进行资源识别、表示和交互的原则。
- [Path Traversal Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Path_Traversal_Cheat_Sheet.html) — OWASP Foundation (2023)
  Journal: OWASP Cheat Sheet Series
  解释了路径遍历漏洞及与动态 URI 处理相关的防御措施。
- [Computer Systems: A Programmer's Perspective](https://csapp.cs.cmu.edu/) — Randal E. Bryant and David R. O'Hallaron (2015)
  Publisher: Pearson
  提供了内存层次结构和缓存原则（包括 LRU）的全面解释，有助于提高读取性能。

---
course: "getting-started-model-context-protocol"
chapter: "defining-resources-and-prompts"
lesson: "uri-schemes-patterns"
sourceId: 7844
sourceUrl: "https://apxml.com/zh/courses/getting-started-model-context-protocol/chapter-2-defining-resources-and-prompts/uri-schemes-patterns"
title: "URI 方案和模式"
description: "构建统一资源标识符 (URI) 以唯一标识和路由数据。"
order: 2
plots: []
sourceHash: "c88d44fa5005e629d378d39e9064d66b17dd498757f8f3d44ca302cca77686d1"
sourceCorrections: []
---

统一资源标识符 (URI) 是模型上下文 (context)协议的寻址系统。客户端-宿主-服务器拓扑搭建了通信链路，而 URI 则提供了定位和获取数据所需的具体坐标。当大型语言模型（LLM）或客户端应用需要获取特定的上下文内容，如文件、数据库中的一行或系统日志时，它会使用一个结构化字符串来唯一标识该项。

在 MCP 中，URI 不仅是静态地址。它们与 **URI 模板**一同使用，服务器可借此公布可获取资源的类别，而非逐一列出每个项目。这一区别对性能非常要紧。一个封装了数百万行数据库的服务器，在初始化握手时无法向客户端发送数百万个 URI 列表。相反，它会发送一个模板，展示访问这些行的模式。

### MCP URI 的构成

MCP URI 遵循 RFC 3986 中定义的标准语法。掌握这种结构对于设计直观且无冲突的资源名称来说很有必要。一个典型的 URI 包含 **方案**、可选的 **授权信息** 和 **路径**。

URI = 方案 ":" ["//" 授权信息] 路径 ["?" 查询] ["#" 片段]

在 MCP 服务器中，各组成部分的功能如下：

1. **方案：** 标识协议或具体的数据源类型。虽然 `file`、`http` 和 `https` 普遍使用，但 MCP 服务器常定义自定义方案，如 `linear`、`postgres` 或 `app-logs`，来对资源进行命名空间划分。
2. **授权信息：** 在本地 MCP 服务器中常被省略，但如果服务器管理多个外部系统的连接，则可用来区分不同的实例或主机名。
3. **路径：** 用于标识具体资源的分层标识符。

下图说明了 MCP 服务器如何解析这些组成部分，从而将请求路由到正确的内部处理程序。

> URI 请求通过服务器解析流程，送达对应函数处理程序的示意。

### URI 模板和动态资源

静态资源表示一个单一、不变的 URI，例如 `config://application/settings`。这适用于单个资产。然而，多数应用处理的是动态数据集合。为了应对这种情况，MCP 使用了 URI 模板 (RFC 6570)。

模板含有用大括号括起来的变量，例如 `file:///{path}`。当服务器注册此资源时，它会告诉客户端：“我能处理任何与此模式相符的 URI。”

当客户端请求 `file:///Users/jdoe/project/readme.md` 时，MCP SDK 会自动执行模式匹配。它从 URI 中取出 `/Users/jdoe/project/readme.md`，并将其作为参数 (parameter)传递给您注册的函数。这种将*接口*（模板）与*实现*（函数）分开的做法，使得资源公开可扩展。

#### 在 Python 中定义模式

使用 Python SDK 时，您通过装饰器定义这些模式。框架会进行路由处理。

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("FileSystemServer")

# 静态资源：无变量
@mcp.resource("app://metadata")
def get_metadata() -> str:
    return "Version 1.0.0 - File Reader System"

# 动态资源：使用模板变量 {path}
@mcp.resource("file:///{path}")
def read_file(path: str) -> str:
    """从本地文件系统读取文件。"""
    try:
        # 在实际实现中，需验证 'path' 以防止目录遍历
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return "Error: File not found."
```

在此示例中，SDK 注册了模式 `file:///{path}`。如果 LLM 生成了 `file:///etc/hosts` 的请求，变量 `path` 将获取值 `/etc/hosts`。

### 方案设计策略

选取正确的方案是一个架构决定，它会作用于 LLM 如何解读数据上下文 (context)。一个清晰、语义化的方案能让模型在读取内容之前就把握数据的性质。

1. **标准数据使用标准方案：** 如果您从磁盘提供实际文件，请使用 `file://`。这向客户端传达，路径很可能符合文件系统约定。
2. **为自定义数据设定命名空间：** 如果您正在为某个应用构建服务器，请使用该应用的名称作为方案。例如，`slack://channels/{id}` 比 `app://slack/channels/{id}` 更具描述性。
3. **代理避免使用 `http/https`：** 除非您的资源确实是要获取的网页，否则请避免使用 `http` 或 `https`。如果您的服务器从 REST API 获取数据并返回 JSON 对象，请定义一个自定义方案，如 `api://`。使用 `https` 可能会让模型误以为可以直接浏览网页，而非与您的特定 API 包装器交互。

### 模式匹配处理

MCP 服务器内部的匹配过程遵循特定的优先级。掌握此点有助于避免路由冲突，尤其当存在重叠模式时。

通常，精确匹配优先于模板匹配。如果您同时定义了一个通用文件读取器和一个用于 readme 文件的具体处理程序，那么当请求该精确 URI 时，具体处理程序应被触发。

考虑一个公布用户记录的数据库服务器的路由处理方式：

> 将传入 URI 路由到相应处理程序的决策树。

### 处理查询参数 (parameter)

MCP URI 支持查询参数，这些参数在不改变资源身份的情况下，可用于筛选或修改资源表示。例如，您可能只想获取日志文件的最后 50 行。

URI 可能看起来像 `logs://system/error.log?lines=50`。

在您的资源处理程序中，解析这些参数取决于具体的 SDK 实现细节，但通常，完整的 URI 可供查看。在定义模式时，通常的做法是在模板中定义路径 (`logs://{type}/{name}`)，并将查询字符串视为在函数内部处理的可选元数据。

### URI 设计的最佳实践

- **稳定性：** URI 应保持稳定。如果资源的 URI 改变，LLM 的上下文 (context)窗口可能含有过时引用，从而引发幻觉 (hallucination)或错误。
- **可读性：** 尽管主要供机器使用，但可读的 URI 在调试时会有便利。`users://123/profile` 优于 `data://obj?id=5921&type=u`。
- **安全性：** 切勿盲目信任从 URI 模板派生出的输入。就像网络开发一样，如果 ID 直接传递给数据库查询，“路径”变量的输入必须经过清理，以避免目录遍历攻击或 SQL 注入。

通过严谨地确立您的 URI 方案和模式，您创建了一个可预测的数据图。这使得 LLM 能够充当智能导航器，准确请求其回答用户查询所需的上下文。

## 参考资料

- [Uniform Resource Identifier (URI): Generic Syntax](https://datatracker.ietf.org/doc/html/rfc3986) — T. Berners-Lee, R. Fielding, L. Masinter (2005)
  Publisher: The Internet Society
  定义了URI的通用语法，包括方案、授权、路径、查询和片段组件，是资源识别的基础。
- [Designing Web APIs: Building APIs That Developers Love](https://www.oreilly.com/library/view/designing-web-apis/9781492037911/) — Brenda Jin, Saurabh Sahni, and Amir Shevat (2018)
  Publisher: O'Reilly Media
  为设计稳定、可读且安全的URI和API结构提供了指导，这与URI设计原则和最佳实践直接相关。

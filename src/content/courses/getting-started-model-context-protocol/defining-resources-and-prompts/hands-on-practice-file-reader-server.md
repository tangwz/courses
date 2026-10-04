---
course: "getting-started-model-context-protocol"
chapter: "defining-resources-and-prompts"
lesson: "hands-on-practice-file-reader-server"
sourceId: 7848
sourceUrl: "https://apxml.com/zh/courses/getting-started-model-context-protocol/chapter-2-defining-resources-and-prompts/hands-on-practice-file-reader-server"
title: "动手实践：构建文件读取服务器"
description: "分步指南，指导您构建一个读取并提供本地文件服务的MCP服务器。"
order: 6
plots: []
sourceHash: "7ca3f5c0e154b99540075c1c553e4df2e4963e3c401506602b0454582039af8d"
sourceCorrections: []
---

我们现在从架构定义转向具体实现。在本节中，我们将构建一个能工作的MCP服务器，它能够从本地目录读取文件并将其内容提供给客户端。此练习巩固了之前讨论的URI方案、资源注册和同步读取模式的理解。

我们将构建一个名为`filesystem-reader`的服务器，它将对外提供一个包含日志文件的特定目录。这模拟了一种常见的生产情况，其中LLM需要访问服务器日志进行错误排查，而无需直接的机器shell访问权限。

### 环境与依赖

确保您的开发环境已安装MCP Python SDK。我们将使用`FastMCP`类，它为定义资源和工具提供了一个简化的接口，无需样板式的事件循环处理。

```bash
pip install mcp
```

创建一个名为`mcp-file-server`的目录，并在其中添加一个名为`logs`的子目录。在`logs`目录中放入两个简单的文本文件：`app.log`和`error.log`，其中包含虚拟数据。这个结构将作为我们对外提供资源的目标。

### 服务器初始化

创建一个名为`server.py`的文件。入口点涉及导入所需的类并实例化服务器。我们将服务器名称定义为`filesystem-reader`，客户端将使用它来识别上下文 (context)的来源。

```python
from mcp.server.fastmcp import FastMCP
import os

# 初始化MCP服务器
mcp = FastMCP("filesystem-reader")

# 定义我们资源的基准路径
# 在生产环境中，请使用绝对路径或环境变量
BASE_DIR = os.path.abspath("./logs")
```

### 实现资源列表功能

MCP服务器的第一个要求是告知客户端可用数据。客户端通过`resources/list`功能获取数据信息。我们通过扫描目标目录并将每个文件转换为`Resource`对象来实现这一点。

URI是这里的核心标识符。我们将使用自定义方案`logs://`来区分我们的内部日志资源和通用文件路径。这种抽象使我们能够将内部文件系统结构与外部资源标识符分离。

```python
@mcp.resource("logs://{filename}")
def get_log_file(filename: str) -> str:
    """
    根据文件名读取日志文件。
    """
    # 安全检查：确保文件名不包含路径遍历字符
    if ".." in filename or "/" in filename or "\\" in filename:
        raise ValueError("无效文件名：不允许路径遍历")

    file_path = os.path.join(BASE_DIR, filename)
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise ValueError(f"日志文件{filename}未找到")
    except Exception as e:
        raise RuntimeError(f"无法读取文件：{str(e)}")
```

在上面的代码中，装饰器`@mcp.resource`有两个作用。首先，它将该函数注册为处理匹配`logs://{filename}`模式的读取请求的处理器。其次，它隐式地设置了列表逻辑。当客户端请求资源列表时，SDK会使用此定义来了解可用的端点。

然而，对于像文件目录这样可能会出现新文件的动态资源，我们通常需要一个明确的列出器来告诉客户端当前*确切*存在哪些文件。模式匹配适用于读取，但列表功能需要遍历目录。

```python
from mcp.types import Resource

@mcp.resource_listing
def list_resources() -> list[Resource]:
    """
    扫描日志目录并返回可用日志资源的列表。
    """
    resources = []
    if not os.path.exists(BASE_DIR):
        return []

    for filename in os.listdir(BASE_DIR):
        if filename.endswith(".log"):
            resources.append(Resource(
                uri=f"logs://{filename}",
                name=f"系统日志：{filename}",
                description=f"服务器应用程序日志文件：{filename}",
                mimeType="text/plain"
            ))
    return resources
```

这种关注点分离很有意义。**列表处理器**充当目录，返回元数据（URI、名称、MIME类型）。**读取处理器**在请求特定URI时执行I/O操作。

下图说明了客户端与这些处理器交互时的消息流。

> 操作顺序将资源查找与内容获取分开，使客户端能够在请求大量数据包之前整理可用上下文 (context)。

### 添加上下文 (context)提示

虽然直接访问文件有益，但提示使我们能够将这些数据打包成可操作的指令。我们可以定义一个提示，它会自动获取日志文件并要求LLM进行分析。这减轻了最终用户进行提示工程 (prompt engineering)的负担。

我们定义了一个名为`analyze-log`的提示，它接受一个`filename`参数 (parameter)。

```python
@mcp.prompt("analyze-log")
def analyze_log_prompt(filename: str) -> list:
    """
    创建一个提示上下文来分析特定的日志文件。
    """
    return [
        {
            "role": "user",
            "content": {
                "type": "text",
                "text": f"请分析以下日志文件中的错误和警告模式：logs://{filename}"
            }
        }
    ]
```

当在客户端中选择此提示时，它会注入一条包含URI的用户消息。客户端识别URI后，将使用我们之前定义的资源处理器获取内容，并将其嵌入 (embedding)上下文窗口，然后发送给LLM。

### 运行服务器

`FastMCP`类处理通过标准输入/输出（stdio）进行的底层JSON-RPC通信。要运行服务器，您只需调用该脚本。

```python
if __name__ == "__main__":
    mcp.run()
```

要测试此集成，您不能单独直接运行此Python脚本。相反，您需要配置一个MCP客户端（如MCP Inspector或Claude Desktop）来执行此脚本。客户端管理进程生命周期，向`stdin`发送请求并从`stdout`读取响应。

### 安全措施

向LLM公开文件系统需要实行严格的界限执行。在我们的`get_log_file`函数中，我们实现了一个基本的路径遍历检查（`..`）。在生产环境中，您应该使用更严格的路径检查：

1. **路径规范化：** 将请求的路径解析为绝对形式。
2. **前缀检查：** 验证已解析的路径是否以`BASE_DIR`路径开头。
3. **只读模式：** 确保服务器进程的文件系统权限被限制为对数据目录的只读访问。

通过遵循这些做法，您可以为本地数据进入模型的上下文 (context)创建一个安全的通道，将静态文件转变为动态信息资源。

## 参考资料

- [Model Context Protocol (MCP) Documentation](https://docs.anthropic.com/en/docs/mcp) — Anthropic (2024)
  Publisher: Anthropic
  针对模型上下文协议的官方文档，详细介绍了架构、SDK 使用以及使用 `FastMCP` 和资源定义实现服务器。
- [FastAPI Documentation](https://fastapi.tiangolo.com/) — Sebastián Ramírez (2024)
  为 `FastMCP` 使用的基于装饰器的 API 定义风格提供了背景，展示了现代 Python 服务器开发实践。
- [Uniform Resource Identifier (URI): Generic Syntax](https://datatracker.ietf.org/doc/html/rfc3986) — T. Berners-Lee, R. Fielding, L. Masinter (2005)
  Journal: IETF RFC 3986; Publisher: The Internet Society; DOI: [10.17487/RFC3986](https://doi.org/10.17487/RFC3986)
  定义统一资源标识符 (URI) 通用语法的基础规范，包括方案等组件，这对于 `logs://` URI 设计至关重要。

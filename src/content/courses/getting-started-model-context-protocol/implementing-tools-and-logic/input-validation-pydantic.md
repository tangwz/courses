---
course: "getting-started-model-context-protocol"
chapter: "implementing-tools-and-logic"
lesson: "input-validation-pydantic"
sourceId: 7850
sourceUrl: "https://apxml.com/zh/courses/getting-started-model-context-protocol/chapter-3-implementing-tools-and-logic/input-validation-pydantic"
title: "使用 Pydantic 进行输入校验"
description: "使用 Pydantic 模型确保工具参数的类型安全和数据完整性。"
order: 2
plots: []
sourceHash: "adea6baac78be28e8481a6b0799fb75440991fc2f6c51728a61f3a957172385a"
sourceCorrections: []
---

工具的可靠运行有赖于严格遵循已定义的接口。当大型语言模型 (LLM) 调用工具时，它会生成一个 JSON 格式的文本字符串。尽管 LLM 生成有效语法的能力不断增强，但它们是概率性运作的。它们可能会凭空捏造参数 (parameter)，以错误的格式提供数据（例如在需要整数时发送字符串），或者完全遗漏必要的参数。

为了弥合模型的概率性与代码的确定性要求之间的差距，我们采用严密的输入校验。在 Python 生态系统中，Pydantic 是数据解析和校验的标准。在模型上下文 (context)协议 (MCP) 中，Pydantic 扮演两个不同的作用：它对传入的工具请求强制执行类型安全，并自动生成 `tools/list` 能力定义所需的 JSON Schema。

### 定义工具接口

工具的根本是表示其参数 (parameter)的数据模型。通过继承 `pydantic.BaseModel`，您可以定义输入的预期结构。此模型充当 MCP 服务器和客户端之间的协议。

设想一个场景，我们正在构建一个用于查询产品数据库的工具。一个简单的实现可能会接受一个通用字典。然而，使用 Pydantic 模型可以提供即时校验逻辑。

```python
from pydantic import BaseModel, Field, PositiveInt
from typing import Optional

class InventoryQuery(BaseModel):
    category: str = Field(
        ...,
        description="用于筛选的产品类别，例如“电子产品”或“服装”。"
    )
    limit: PositiveInt = Field(
        10,
        description="要返回的最大结果数。默认为 10。"
    )
    warehouse_id: Optional[str] = Field(
        None,
        description="可选的特定仓库 ID，用于缩小搜索范围。"
    )
```

在这个例子中，`InventoryQuery` 类执行了多项重要任务。它强制规定 `category` 为字符串，`limit` 为正整数。它还把 `warehouse_id` 标记 (token)为可选。如果 LLM 尝试使用 `limit=-5` 调用此工具，Pydantic 会在您的逻辑运行之前拦截请求并引发 `ValidationError`。

### 字段描述的作用

您会注意到上面代码中大量使用了 `Field` 函数。在标准的 Python 开发中，这些描述用于开发人员的文档目的。在 MCP 的背景下，这些描述具有功能性作用：它们是提示的一部分。

当 MCP 服务器向客户端（如 Claude Desktop）通告其工具时，它会将 Pydantic 模型序列化为 JSON Schema。您的 `Field` 定义中的 `description` 参数 (parameter)会直接传递给 LLM。这些描述指导模型如何正确填充参数。

如果校验逻辑定义为函数 $f(x) \to \{0, 1\}$，其中 $1$ 代表有效，$0$ 代表无效，那么描述会增加 $P(f(x_{LLM}) = 1)$ 的概率。Pydantic 模型中清晰、详细的描述能大幅降低 `ValidationError` 事件的发生率。

### 校验流程

当请求到达 MCP 服务器时，它会经历一系列特定的检查。了解这个流程对于调试客户端和您的工具逻辑之间的交互很有帮助。

> 该图概述了工具请求的顺序处理过程，强调了校验步骤的把关作用。

如果校验步骤失败，MCP 服务器会捕获异常。服务器不会崩溃，而是将校验错误格式化为文本响应并发送回客户端。这种反馈循环让 LLM 能够发现自己的错误（“参数 (parameter) 'limit' 必须为正数”），并使用修正后的值再次尝试工具调用。

### 实现自定义校验器

标准类型检查涵盖大多数使用场景，但复杂工具通常需要超出简单类型的逻辑。例如，如果您正在构建一个 SQL 查询工具，您可能希望限制允许的操作类型（例如，允许 `SELECT` 但禁止 `DROP`）。

Pydantic 允许使用 `@field_validator` 装饰器实现自定义校验器。这让您可以将领域特定逻辑注入到校验层中。

```python
from pydantic import BaseModel, Field, field_validator

class SQLQuery(BaseModel):
    query: str = Field(..., description="要执行的 SQL 查询。")

    @field_validator('query')
    @classmethod
    def prevent_destructive_commands(cls, v: str) -> str:
        forbidden = ["DROP", "DELETE", "TRUNCATE", "ALTER"]
        upper_query = v.upper()
        for cmd in forbidden:
            if cmd in upper_query:
                raise ValueError(f"不允许执行破坏性命令 '{cmd}'。")
        return v
```

通过将此逻辑放置在 Pydantic 模型内部，您可以将安全规则与执行逻辑解耦。`SQLQuery` 模型确保您的执行处理器只接收已清理的字符串。这种关注点分离使代码库更易于测试和维护。

### 与 MCP 服务器集成

一旦您的 Pydantic 模型被定义，它们必须注册到 MCP 服务器实例。SDK 使用类型提示来检查函数签名，并将参数 (parameter)映射到 JSON Schema。

当使用标准 MCP Python SDK（或像 FastMCP 这样的包装器）时，您将 Pydantic 模型作为工具函数的第一个参数的类型提示进行传递。

```python
# 假设 'mcp' 是一个已实例化的 FastMCP 或 Server 对象

@mcp.tool()
def query_inventory(args: InventoryQuery) -> str:
    """
    根据类别筛选条件查询仓库库存系统。
    """
    # 此时，'args' 保证是一个有效的 InventoryQuery 对象
    # 具有正确的类型并满足所有约束。

    results = database.search(
        category=args.category,
        limit=args.limit
    )

    return format_results(results)
```

在此实现中，`query_inventory` 函数的文档字符串成为工具的顶级描述，而 `InventoryQuery` 模型则提供参数的 schema。

### 处理复杂的嵌套结构

工具偶尔需要复杂的嵌套数据结构。例如，一个配置图表库的工具可能需要一个数据点列表，其中每个点包含 $x$ 和 $y$ 坐标。Pydantic 通过模型嵌套来处理这种情况。

```python
class DataPoint(BaseModel):
    x: float
    y: float

class ChartConfig(BaseModel):
    title: str
    points: list[DataPoint]
    color: str = "blue"
```

当 LLM 与使用 `ChartConfig` 的工具交互时，MCP 服务器会校验整个树。如果五十个点列表中的第三个点，其 $y$ 值预期为浮点数，但实际却是一个字符串，那么校验会恰好针对该元素失败。这种粒度确保您的工具逻辑永远不需要在深度嵌套的列表或字典中手动解析或转换类型。

### 对 LLM 性能的影响

您校验的严密性直接影响代理的可靠性。尽管使用松散类型（如 `dict` 或 `Any`）以避免校验错误可能很诱人，但这会将错误处理的负担转移到您的运行时逻辑，而这更难反馈给 LLM。

当 LLM 接收到源自 Pydantic 模型的清晰、结构化 schema 时，它的表现会更好，因为“动作空间”定义明确。如果它确实犯了错误，来自 Pydantic 的结构化错误消息会作为纠正性提示，指导模型在后续轮次中走向正确的使用方式。这种自我修正行为是使用正式 schema 相比非结构化字符串解析的一个重要优势。

## 参考资料

- [Pydantic Documentation](https://docs.pydantic.dev/latest/) — Samuel Colvin and Pydantic contributors (2024)
  Pydantic功能（数据模型定义、类型验证、自定义验证器和JSON Schema生成）的官方资源。
- [JSON Schema](https://json-schema.org/) — IETF and JSON Schema community (2020)
  提供了声明JSON数据结构和验证约束的标准，是Pydantic模型序列化为LLM工具的基础。
- [FastAPI and Pydantic Cookbook](https://www.packtpub.com/product/fastapi-and-pydantic-cookbook/) — Alessandro Giusti (2024)
  Publisher: Packt Publishing
  展示了Pydantic在实际应用中（特别是FastAPI等API开发框架内）如何用于定义请求模式、执行验证和生成OpenAPI（JSON）Schema。

---
course: "building-advanced-llm-agent-tools"
chapter: "developing-custom-python-tools"
lesson: "structuring-tool-outputs"
sourceId: 6563
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-2-developing-custom-python-tools/structuring-tool-outputs"
title: "大型语言模型复杂工具输出的结构化方法"
description: "学习如何以LLM智能体易于解析和使用的方式，组织工具返回的复杂数据。"
order: 5
plots: []
sourceHash: "c79fbe2f77ebfd8070d8ae2a3ac4342c1189b2381b1a16dc627a8b3aabc6499d"
sourceCorrections: []
---

当你的Python工具完成任务后，它如何将结果呈现给大型语言模型（LLM），这与任务本身一样重要。LLM虽然擅长处理自然语言，但在接收外部数据时，更偏好可预测性和结构。简单地返回一段未区分的文本，可能导致误解，需要更复杂的提示才能解析，甚至让LLM忽略重要细节。有效地组织工具输出，对于构建可靠高效的智能体系统而言非常重要。

主要目标是让LLM理解和使用工具输出的工作尽可能简单。这通常意味着以机器易于解析且LLM便于“推理 (inference)”的格式提供数据。

### 选择输出格式

虽然工具在技术上可以输出纯文本，LLM通常也能理解，但结构化格式在一致性和可靠性方面更优。

- **JSON (JavaScript 对象表示法)**: 这是为LLM准备的工具输出中最受推荐和广泛采用的格式。

  - 它轻量且易于阅读。
  - 包括Python在内的几乎所有编程语言，都对其解析和生成有出色的内置支持。
  - 其键值对结构，以及对数组和嵌套对象的支持，允许丰富的分层数据表示。
  - LLM通常经过良好训练，能够理解并从中提取信息。许多LLM智能体框架和函数/工具调用API明确要求或最适合使用JSON。
- **XML (可扩展标记 (token)语言)**: 如果你的工具与只提供XML的旧系统或API交互，你可能需要返回XML或进行转换。然而，如果可以选择，由于其简洁性以及在现代Web API中更普遍的使用，JSON通常是新工具开发的优选。
- **纯文本（附带注意事项）**: 对于结构极其简单的输出（例如，一个只返回单个数字或布尔状态的工具），纯文本*可能*足够。然而，一旦涉及多条信息，或者信息之间存在内在关联，JSON就会成为更好的选择。如果必须使用纯文本来处理复杂数据，请建立一个非常清晰、一致且易于解析的基于分隔符的系统，但请注意，这比JSON更容易导致LLM误解。

对于大多数自定义Python工具，**JSON是推荐的输出格式。**

### 设计有效的输出模式

“模式”（schema）指的是结构化输出的设计。一个精心设计的模式能确保LLM以可预测且有用的方式接收数据。

#### 清晰性和描述性

字段名（JSON中的键）应该一目了然。避免使用隐晦的缩写。如果一个字段代表用户的电子邮件地址，`user_email`或`email_address`远优于`u_em`或`dat1`。请记住，LLM会使用这些名称来理解值的含义。

#### 一致性

在命名约定（例如，键使用`snake_case`或`camelCase`）和类似信息的数据类型方面保持一致性，无论是跨不同工具，还是在同一工具的各种输出中。如果用户ID在系统某部分是整数，那么它在任何地方都应该是整数。

#### 数据类型

使用适当的JSON数据类型:

- **字符串**: 用于文本数据。
- **数字**: 用于整数和浮点值。
- **布尔值**: 用于`true`/`false`状态。
- **数组**: 用于项目列表（例如，产品名称列表、搜索结果）。
- **对象**: 用于分组相关的键值对（嵌套）。
- **空值**: 用于明确表示字段没有值。

#### 细节与简洁的平衡

包含LLM当前任务或后续步骤可能需要的所有信息。但是，请注意冗长。过大的输出可能会占用LLM上下文 (context)窗口中宝贵的token，并可能使LLM更难高效处理。如果工具生成大量数据，请考虑工具本身是否可以执行一些摘要或过滤，或者分页是否合适。

#### 错误表示

如果工具遇到可以优雅处理的错误，或者需要传达失败状态，输出结构应适应这种情况。工具可以返回一个指示成功或失败的JSON对象，而不是抛出导致智能体停止的异常（除非这是期望的行为）。

例如，一个成功的输出可能如下所示:

```json
{
  "status": "success",
  "data": {
    "user_id": "12345",
    "username": "llm_dev"
  }
}
```

错误输出则可能如下:

```json
{
  "status": "error",
  "error_code": "USER_NOT_FOUND",
  "message": "User with ID '67890' could not be found."
}
```

这使得LLM能够理解工具已执行但遇到了特定问题，然后LLM可以在其推理 (inference)中使用这些信息。

### 结构化输出示例

让我们来看一个获取产品信息的工具。

**一个不太理想、有些非结构化的文本输出:**

```text
Product: SuperWidget, Price: $29.99, In Stock: Yes, Features: Durable, Lightweight, Color: Blue.
```

LLM*可能*会解析这段文本，但其可靠性较低。如果“In Stock”是“No”怎么办？或者货币符号改变了呢？

**同一产品的良好结构化JSON输出:**

```json
{
  "product_name": "SuperWidget",
  "price": 29.99,
  "currency": "USD",
  "in_stock": true,
  "features": ["Durable", "Lightweight"],
  "specifications": {
    "color": "Blue",
    "weight_grams": 150
  }
}
```

此JSON输出明确。LLM可以轻松提取价格、检查库存状态或列出功能。

**对于返回多个项目（如搜索结果）的工具:**

```json
{
  "query_terms": ["python", "async", "web server"],
  "results_count": 2,
  "items": [
    {
      "title": "Building Async Web Servers with Python",
      "url": "https://example.com/async-servers",
      "snippet": "An in-depth guide to using asyncio and libraries like aiohttp...",
      "relevance_score": 0.85
    },
    {
      "title": "Understanding Python's Async/Await",
      "url": "https://another-site.org/python-async",
      "snippet": "A foundational explanation of asynchronous programming concepts in Python.",
      "relevance_score": 0.72
    }
  ]
}
```

在此，结构清楚地将每个搜索结果划分为数组中的一个对象，每个对象都拥有自己的一组属性。

### 在Python中实现结构化输出

Python的标准库使得生成JSON变得简单直接。`json`模块是你的主要工具。

```python
import json

def get_product_info_tool(product_id: str):
    # 假设从数据库或API获取数据
    if product_id == "SW001":
        product_data = {
            "product_name": "SuperWidget",
            "price": 29.99,
            "currency": "USD",
            "in_stock": True, # Python布尔值
            "features": ["Durable", "Lightweight"],
            "specifications": {
                "color": "Blue",
                "weight_grams": 150
            }
        }
        # 将Python字典序列化为JSON字符串
        return json.dumps(product_data)
    else:
        error_data = {
            "status": "error",
            "error_code": "PRODUCT_NOT_FOUND",
            "message": f"Product with ID '{product_id}' not found."
        }
        return json.dumps(error_data)

# 使用示例:
output_json = get_product_info_tool("SW001")
print(output_json)
# 输出:
# {"product_name": "SuperWidget", "price": 29.99, "currency": "USD", "in_stock": true, "features": ["Durable", "Lightweight"], "specifications": {"color": "Blue", "weight_grams": 150}}

output_error_json = get_product_info_tool("XYZ789")
print(output_error_json)
# 输出:
# {"status": "error", "error_code": "PRODUCT_NOT_FOUND", "message": "Product with ID 'XYZ789' not found."}
```

请注意Python的`True`是如何转换为JSON的`true`的。`json.dumps()`函数处理这些转换。

对于更复杂的情况，特别是当处理许多不同的输出结构或在序列化之前需要验证输出数据时，像**Pydantic**这样的库非常有帮助。Pydantic允许你使用Python类型提示定义数据模型。这些模型随后可以解析输入数据并序列化为JSON，确保你的输出符合预定义的模式。

```python
from typing import List, Dict, Optional
from pydantic import BaseModel
import json

# 为我们的产品结构定义Pydantic模型
class ProductSpecifications(BaseModel):
    color: str
    weight_grams: int

class ProductInfo(BaseModel):
    product_name: str
    price: float
    currency: str
    in_stock: bool
    features: List[str]
    specifications: ProductSpecifications

class ErrorResponse(BaseModel):
    status: str = "error"
    error_code: str
    message: str

def get_product_info_tool_pydantic(product_id: str) -> str:
    if product_id == "SW001":
        product_data = ProductInfo(
            product_name="SuperWidget",
            price=29.99,
            currency="USD",
            in_stock=True,
            features=["Durable", "Lightweight"],
            specifications=ProductSpecifications(color="Blue", weight_grams=150)
        )
        # Pydantic模型具有.model_dump_json()方法（或Pydantic v1中的.json()）
        return product_data.model_dump_json()
    else:
        error_data = ErrorResponse(
            error_code="PRODUCT_NOT_FOUND",
            message=f"Product with ID '{product_id}' not found."
        )
        return error_data.model_dump_json()

# print(get_product_info_tool_pydantic("SW001"))
```

使用Pydantic通过验证你即将序列化的数据是否符合预期的结构和类型，增加了一层健壮性。

以下图表说明了使用结构化输出时，数据从Python工具到LLM的典型流程:

> 当使用结构化JSON输出时，从工具执行到LLM接收的数据流。

### 指导LLM使用结构化输出

仅仅返回JSON并不总是足够的。LLM需要知道期待什么以及如何使用它。这通常通过以下方式实现:

1. **工具描述**: 当你为LLM智能体定义工具时（如第1章“LLM智能体工具基础”中讨论的），你会描述其目的、输入以及输出格式。明确说明“此工具返回一个包含X、Y和Z字段的JSON对象”可以为LLM做好准备。
2. **函数/工具调用模式**: 现代LLM API（如OpenAI的）具有明确的“函数调用”或“工具调用”功能。在此，你为你工具的*输入参数 (parameter)*提供JSON模式定义，有时也为其*输出*提供定义。LLM随后会尝试生成与你的输入模式匹配的参数，并且能更好地准备解析它知道应符合特定结构的输出。
3. **提示工程 (prompt engineering)**: 在你指示LLM使用工具的提示中，你可以重申预期的输出结构或提供示例。

### 处理大型或复杂输出

如果你的工具可能返回大量数据:

- **工具内摘要/过滤**: 工具本身能否在返回数据前智能地进行摘要或过滤？例如，与其返回1000个搜索结果，不如返回最相关的10个，或提供发现的摘要。
- **指示截断或分页**: 如果必须截断，请在输出中明确说明（例如，`"results_truncated": true, "total_available_results": 1000`）。如果你的工具支持分页，输出可以包含如何获取下一页的信息（例如，`next_page_token: "nextToken123"`）。
- **流式传输（高级）**: 对于非常大的数据集或连续数据，可以考虑流式输出，尽管这会增加工具本身和智能体处理它的复杂性。

通过精心组织工具输出，你可以显著提升LLM智能体的可靠性和能力。清晰、可预测的JSON输出使得LLM更像是精确的程序化组件，而不仅仅是一个文本处理器，从而带来更强大、更可靠的应用。在你持续开发自定义Python工具时，请始终考虑LLM将如何最好地接收你的工具提供的信息。

## 参考资料

- [Function calling](https://platform.openai.com/docs/guides/function-calling) — OpenAI (2024)
  本指南详细介绍了如何将自定义工具与OpenAI模型集成，说明了工具模式的定义以及大型语言模型如何消费结构化工具输出。
- [Pydantic Documentation](https://pydantic.dev/) — Samuel Colvin (2024)
  Pydantic的官方文档，这是一个Python库，有助于定义、验证和序列化数据模式，对于健壮地构建工具输出结构特别有用。
- [RFC 8259: The JavaScript Object Notation (JSON) Data Interchange Format](https://www.rfc-editor.org/rfc/rfc8259) — T. Bray (2017)
  Publisher: IETF; DOI: [https://doi.org/10.17487/RFC8259](https://doi.org/10.17487/RFC8259)
  定义JSON格式的官方规范，为表示结构化数据提供了一个通用标准，这对于工具输出的一致性至关重要。

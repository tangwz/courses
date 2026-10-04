---
course: "prompt-engineering-agentic-workflows"
chapter: "prompt-engineering-tool-use"
lesson: "formatting-tool-specifications-llm"
sourceId: 6727
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-agentic-workflows/chapter-3-prompt-engineering-tool-use/formatting-tool-specifications-llm"
title: "为LLM理解工具规范的格式化处理"
description: "学习如何格式化工具描述和API，以便LLM能够明白并运用它们。"
order: 3
plots: []
sourceHash: "845f94a00c09f2d36181ec9ce6b5fb340d3e69416cb9cdd26c26db966e3dbe02"
sourceCorrections: []
---

为了让AI代理可靠地运用外部工具，它首先必须明白工具是什么，它的功能，如何调用，以及预期返回什么。这种明白并非源于先天知识，而是来自我们精心准备的描述。可以这样想：你不会在没有API文档的情况下，将一个新的软件库交给开发人员。同样，LLM需要一份它可能运用的每个工具的“手册”，这份手册就是工具规范。如何组织这些规范，直接影响着LLM能否有效明白并运用其可用的工具。

LLM“读取”工具规范的方式与人类程序员截然不同。LLM将信息作为文本序列处理。该规范成为LLM决定是否运用工具以及如何格式化其请求时考虑的上下文 (context)的一部分。一个格式良好的规范使LLM能够准确识别工具的用途，将用户意图对应到工具输入，并正确组织它需要发送给工具的信息。缺少这一点，代理可能会选错工具，提供格式错误的输入，或误读工具的输出，从而导致错误和任务失败。

### 工具规范的基本组成部分

为确保LLM能够有效运用工具，其规范应清楚定义几项重要信息。这些组成部分是LLM推理 (inference)过程与工具功能之间的主要接口。

1. **工具名称**：

   - 工具的独特且一致的标识符。LLM将运用此名称表明它运用工具的意图。
   - **良好做法**：运用清晰、描述性的名称。常见惯例包括 `snake_case`（例如，`get_stock_price`）或 `camelCase`（例如，`fetchWeatherData`）。避免可能难以让LLM解析或一致生成空格或特殊字符。
   - 例子：`calculator`、`web_search_engine`、`product_database_lookup`
2. **描述**：

   - 对工具的用途、其能力以及（重要地）何时运用它进行简洁的自然语言说明。此描述不仅是为了明白*如何*运用工具，也是为了LLM*选定*工具的决策过程。
   - `unit_converter`工具的例子：“在不同度量单位之间转换值（例如，摄氏度到华氏度，千克到磅）。当请求在指定单位之间进行转换时运用此工具。”
3. **输入参数 (parameter)（参数）**：

   - 这可以说是工具正确调用的最重要的部分。工具接受的每个参数都需要清楚定义。
     - **名称**：参数的描述性名称（例如，`query`、`unit_from`、`value_to_convert`）。
     - **类型**：预期数据类型（例如，`string`、`number`、`boolean`、`array`、`object`）。指定类型有助于LLM正确格式化数据。运用类似JSON Schema的类型定义通常有益。
     - **描述**：清楚说明参数代表什么以及任何特定约束或格式（例如，“要在网上查询的搜索词”，或“温度值，必须是数字类型”）。
     - **必填/可选**：表明参数对于工具运行是否必需。
   - `send_email`工具参数的示例片段：

     ```json
     {
       "recipient_address": {
         "type": "string",
         "description": "收件人邮箱地址，例如'user@example.com'。"
       },
       "subject_line": {
         "type": "string",
         "description": "邮件主题。"
       },
       "email_body": {
         "type": "string",
         "description": "邮件主要内容。"
       }
     }
     ```
4. **输出结构（返回值）**：

   - 工具在成功运行后预期返回的数据结构和类型。这有助于LLM预期工具响应的格式并正确解析它。
   - **类型**：输出的数据类型（例如，`string`、`number`、`object`、`array`）。
   - **描述**：输出代表什么。如果是对象，描述其属性。
   - `unit_converter`工具的例子：

     ```json
     {
       "converted_value": {
         "type": "number",
         "description": "转换的数值结果。"
       },
       "new_unit": {
         "type": "string",
         "description": "值被转换到的单位。"
       }
     }
     ```
5. **运用例子（可选但强烈推荐）**：

   - 提供一些工具被调用的例子（输入参数）以及典型输出的样子，可以显著提升LLM正确运用工具的能力。这符合少样本提示原则。
   - `calculator.add`的例子：

     ```
     用户请求: "15加27是多少？"
     代理操作: { "tool_name": "calculator", "operation": "add", "parameters": { "num1": 15, "num2": 27 } }
     工具响应: { "result": 42 }
     ```

### 工具规范的常见格式结构

尽管规范的内容是主要的，但其格式也同样重要。LLM需要高效解析这些信息。以下是几种常见方法：

#### JSON（JavaScript 对象表示法）

JSON是一种广泛运用的数据交换格式，通常被LLM很好地明白。运用JSON结构（通常借鉴JSON Schema来定义参数 (parameter)和输出）来描述工具是一种方法。

```json
{
  "tool_name": "get_current_weather",
  "description": "获取指定地点的当前天气情况。",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "城市名称（例如，'伦敦'，'东京'）。"
      },
      "unit": {
        "type": "string",
        "description": "温度单位：'celsius'或'fahrenheit'。默认为'celsius'。",
        "enum": ["celsius", "fahrenheit"]
      }
    },
    "required": ["location"]
  },
  "returns": {
    "type": "object",
    "properties": {
      "temperature": {
        "type": "number",
        "description": "指定单位下的当前温度。"
      },
      "condition": {
        "type": "string",
        "description": "天气情况的简要描述（例如，'晴朗'，'多云'）。"
      },
      "humidity_percent": {
        "type": "integer",
        "description": "当前湿度百分比。"
      }
    }
  }
}
```

> 一个定义`get_current_weather`工具的JSON对象，包括其描述、带有类型和约束的参数，以及预期的返回结构。

#### 带分隔符的结构化自然语言

对于更简单的工具，或者如果你想在提示中以更易读的方式呈现信息，带有清晰分隔符的结构化自然语言是可行的。一致性在这里很重要。

```text
工具: find_city_timezone
描述: 获取给定城市的当前IANA时区。
参数:
  city_name (string, required): 城市名称（例如，“New York”，“Paris”）。
  country_code (string, optional): 用于区分同名城市的两位ISO国家代码（例如，“US”，“FR”）。
返回值:
  timezone (string): IANA时区标识符（例如，“America/New_York”）。
  error (string, optional): 如果找不到时区的错误消息。
```

> 一个运用结构化自然语言并为每个组成部分提供清晰标签的工具规范。

#### 专用工具/函数调用格式

许多现代LLM API，例如OpenAI的API，提供特定的“函数调用”或“工具运用”能力。这些API通常期望工具规范采用预定义的JSON格式。在运用其专用工具运用功能时，遵循这些供应商特定格式是必需的。

例如，OpenAI的API可能期望工具定义列表如下：

```json
[
  {
    "type": "function",
    "function": {
      "name": "get_flight_booking_status",
      "description": "根据预订ID获取航班预订状态。",
      "parameters": {
        "type": "object",
        "properties": {
          "booking_id": {
            "type": "string",
            "description": "航班预订的独特标识符，例如'FGH789'。"
          }
        },
        "required": ["booking_id"]
      }
    }
  }
]
```

> 一个工具可能如何为支持专用函数调用的LLM API定义的例子，显示通常需要的嵌套结构。

查阅你正在运用的特定LLM或框架的文档很重要，因为它们通常规定工具规范的首选格式。

### 编写有效工具规范的良好做法

- **清晰和简洁**：在描述中避免运用行话或过于复杂的句子。规范会增加整体提示长度，因此请注意token限制。
- **一致性**：在所有工具定义中运用统一的命名规范（例如，所有参数 (parameter)名称都采用`snake_case`）和结构。这有助于LLM学习模式。
- **描述性命名**：为工具和参数选择清晰表明其用途的名称（例如，`user_query`优于`inp1`）。
- **明确列出枚举值**：如果参数只接受特定值集合（枚举），请清楚列出它们（例如，`status: "pending" | "completed" | "failed"`）。
- **预期常见变体**：在描述中，你可以暗示如何处理用户请求中的微小变体。例如，对于搜索工具，你可能会指定“运用此工具查找关于主题、人物或事件的信息。”
- **版本控制**：如果你的工具随时间变化（例如，新参数、不同输出），请考虑在规范中包含`version`字段。你的代理逻辑可能需要处理不同工具版本。
- **迭代改进**：你的工具规范的第一个版本可能不完美。测试LLM明白和运用工具的程度。观察其生成的工具调用并根据任何误解或错误改进规范。这是提示工程 (prompt engineering)工作流的常见部分。

> 为LLM明白而设计的全面工具规范的组成部分。该规范充当架构，指导LLM解析其细节并形成准确的工具调用。

通过投入时间创建清晰、全面且格式良好的工具规范，你为LLM提供了必要知识，使其能够在你的代理系统中充当有能力且可靠的组成部分。这个基础很重要，才能接着管理实际工具运行时输入和输出的动态流动，我们将在接下来讨论这一点。

## 参考资料

- [Function calling](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEHb7SvTlcL5mF8XBbCSRCJiSiIhrC588dS1Mg-j567UBU4Y_AgIDmnLO2zUBszyklTEo4FOA-w56bx5Pi2EaCDudauhRCRq8tNBS2j8aofhAIG7ryzu4_rhXA-nYqpI6zikevfAgsT-FXbiGJtCqL25pw=) — OpenAI (2023)
  Publisher: OpenAI
  OpenAI官方文档，详细介绍了如何使用OpenAI大型语言模型定义和调用函数（工具），提供了工具规范的实用示例和最佳实践。
- [Tool calling in Gemini](https://ai.google.dev/docs/guides/function_calling) — Google AI (2024)
  Publisher: Google AI
  Google AI的文档，阐述了将自定义工具与Gemini模型集成和使用的过程，展示了他们定义工具和进行交互的具体方法。
- [JSON Schema](https://json-schema.org/) — JSON Schema community (2020)
  JSON Schema的官方网站和规范，是定义JSON数据结构、内容和验证规则的标准，非常适用于构建LLM工具参数和输出。

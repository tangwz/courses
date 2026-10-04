---
course: "building-advanced-llm-agent-tools"
chapter: "llm-agent-tooling-foundations"
lesson: "practice-first-tool-definition"
sourceId: 6552
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-1-llm-agent-tooling-foundations/practice-first-tool-definition"
title: "实作：编写你的第一个工具定义"
description: "一个关于定义简单工具的实践环节，涵盖其描述、输入参数和预期输出格式。"
order: 7
plots: []
sourceHash: "c5d91e32b527bd6cc468232f53928f8df5fb8b5fcfbdc9fa07b3d467e3384f44"
sourceCorrections: []
---

编写工具定义有助于扩展LLM代理的能力，并要求清晰的规范。下面将指导完成第一个工具定义的编写过程。一个定义完善的工具是LLM有效交互的基础，它确保代理明白该工具的作用、所需信息以及预期的结果。

在此，我们将侧重于*定义*本身。实现工具逻辑的Python代码将在后续章节介绍。目前，我们的目标是创建一个LLM可以使用的清晰约定。

### 工具定义的构成

在开始之前，我们先简要回顾一下工具定义的主要构成部分。这些部分共同作用，为LLM代理提供对工具的全面理解。

> LLM代理工具定义中常见的构成部分。

### 场景：一个简单的天气信息工具

我们来设计一个能获取指定地点当前天气的工具。这是一个常见且实用的例子。LLM代理可以使用此类工具来回答用户的问题，例如“伦敦天气怎么样？”或“我明天在巴黎需要带伞吗？”（尽管我们的第一个版本只提供当前天气）。

我们的工具需要：

1. 一个清晰的名称。
2. 一个LLM能理解的描述，以便判断何时使用此工具。
3. 输入参数 (parameter)的定义（例如，城市）。
4. 输出样式的定义（例如，温度、天气情况）。

### 步骤1：为你的工具命名

名称应具有描述性且简洁。它常被程序化使用，因此蛇形命名法（snake\_case）或驼峰命名法（camelCase）很常见。我们将工具命名为：`get_current_weather`。

### 步骤2：编写描述

描述主要面向LLM。它应清晰说明：

- 工具的作用。
- 何时应使用它。
- 关于其能力或限制的任何细节。

我们的`get_current_weather`工具的一个良好描述可能是：
“获取指定城市的当前天气情况，包括温度、简要描述和湿度。当用户询问特定地点的天气时，请使用此工具。它可选择性地接收州或国家代码，以帮助区分同名城市。”

请注意，此描述如何指导LLM了解其用途（“当用户询问天气时”）并暗示其参数 (parameter)。

### 步骤3：定义输入参数 (parameter)（输入规范）

工具需要输入才能运行。我们需要定义`get_current_weather`工具预期的每个参数。对于每个参数，我们应指定：

- **名称**：参数的识别方式（例如，`city`）。
- **类型**：数据类型（例如，`string`、`number`、`boolean`）。
- **描述**：该参数用途的人类可读和LLM可读的解释。这对于LLM正确填充参数非常重要。
- **必填/可选**：LLM是否*必须*提供此参数。

许多LLM框架使用类似JSON Schema的结构来定义参数。我们来定义天气工具的输入：

- **`city`**：
  - 类型：`string`
  - 描述：“要获取天气的城市名称，例如‘San Francisco’、‘Tokyo’。”
  - 必填：是
- **`state_or_country_code`**：
  - 类型：`string`
  - 描述：“可选。州缩写（例如，加利福尼亚州的‘CA’）或两字母ISO国家代码（例如，大不列颠的‘GB’），必要时用于区分城市名称。”
  - 必填：否

这告诉LLM它绝对需要一个`city`，但`state_or_country_code`是可选的，应在可用或需要时用于提供更具体的地点信息。

### 步骤4：定义输出结构（输出规范）

定义工具将返回的数据结构与定义输入同样重要。这有助于LLM（以及你，开发者）明白工具执行后预期的结果。

对于我们的`get_current_weather`工具，成功执行可能会返回：

- **`location_found`**：
  - 类型：`string`
  - 描述：“已解析出天气信息的完整地点，例如‘London, GB’。”
- **`temperature`**：
  - 类型：`number`（可以是整数或浮点数）
  - 描述：“当前温度。”
- **`unit`**：
  - 类型：`string`
  - 描述：“提供的温度单位（例如，‘Celsius’、‘Fahrenheit’）。”
- **`condition`**：
  - 类型：`string`
  - 描述：“天气情况的简要文字描述（例如，‘Sunny’、‘Cloudy with showers’）。”
- **`humidity_percent`**：
  - 类型：`integer`
  - 描述：“当前湿度百分比（例如，65 表示 65%）。”
- **`error`**：
  - 类型：`string`（或 `null`）
  - 描述：“如果无法获取天气数据，则显示错误消息（例如，‘City not found’）。如果请求成功，此字段将不存在或为null。”

定义输出，特别是包含潜在的错误字段，使工具更可预测，并更方便LLM代理处理各种结果。

### 步骤5：组装完整的工具定义

现在，让我们将所有这些部分整合为一个单一的、结构化的定义。具体格式可能因你使用的LLM框架（如LangChain、LlamaIndex，或直接使用OpenAI的GPT等模型）而异。一种常见做法是使用JSON对象。

这里是我们`get_current_weather`工具定义的可能样式：

```json
{
  "name": "get_current_weather",
  "description": "获取指定城市的当前天气情况，包括温度、简要描述和湿度。当用户询问特定地点的天气时，请使用此工具。它可选择性地接收州或国家代码，以帮助区分同名城市。",
  "input_schema": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "要获取天气的城市名称，例如‘San Francisco’、‘Tokyo’。"
      },
      "state_or_country_code": {
        "type": "string",
        "description": "可选。州缩写（例如，加利福尼亚州的‘CA’）或两字母ISO国家代码（例如，大不列颠的‘GB’），必要时用于区分城市名称。"
      }
    },
    "required": ["city"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "location_found": {
        "type": "string",
        "description": "已解析出天气信息的完整地点，例如‘London, GB’。"
      },
      "temperature": {
        "type": "number",
        "description": "当前温度。"
      },
      "unit": {
        "type": "string",
        "description": "提供的温度单位（例如，‘Celsius’、‘Fahrenheit’）。"
      },
      "condition": {
        "type": "string",
        "description": "天气情况的简要文字描述（例如，‘Sunny’、‘Cloudy with showers’）。"
      },
      "humidity_percent": {
        "type": "integer",
        "description": "当前湿度百分比（例如，65 表示 65%）。"
      },
      "error": {
        "type": "string",
        "description": "如果无法获取天气数据，则显示错误消息（例如，‘City not found’）。如果请求成功，此字段将不存在或为null。"
      }
    },
    "required": ["location_found", "temperature", "unit", "condition", "humidity_percent"]
  }
}
```

> 注意：某些框架可能使用“parameters”而非“input\_schema”。这是输入和输出的结构化定义。`output_schema`中的`required`列表表示成功时预期出现的字段；`error`字段通常只在出现问题时出现。

### 该定义为何有效

这种结构化定义有几个重要作用：

- **LLM理解**：`name`和`description`帮助LLM识别针对给定用户查询或任务的正确工具。
- **参数 (parameter)构成**：`input_schema`，特别是`properties`及其各自的`description`字段，让LLM明白需要哪些参数、它们的类型以及如何格式化。例如，如果用户说“巴黎天气”，LLM就知道从`city`参数中提取“Paris”。
- **输出解释**：尽管并非所有框架都要求LLM直接使用严格的`output_schema`（与`input_schema`方式相同），但定义它是一个良好实践。它为实现工具的开发者明确了约定，并有助于设计LLM如何处理工具结果。它对于测试和验证也很有价值。

### 本次练习的考量点

当你编写自己的工具定义时，请记住我们例子中得出的这些要点：

- **清晰性极为重要**：工具本身及其参数 (parameter)的描述对LLM来说必须明确无歧义。从LLM的角度思考：“如果我读到这个，我能准确知道要提供什么以及这个工具的作用吗？”
- **类型要具体**：使用正确的数据类型（`string`、`number`、`boolean`、`object`、`array`）有助于验证并确保工具接收到预期格式的数据。
- **区分必填与可选**：这有助于LLM形成有效的调用并允许更灵活的工具。
- **规划输出**：了解你的工具将返回什么，包括潜在的错误状态，与定义其输入同样重要。

本次定义工具的实作练习为下一步做好了准备：实现其背后的实际逻辑。通过扎实的定义，你为LLM代理提供了必要信息，使其能够有效使用你的自定义功能。在后续章节中，我们将探讨如何用Python代码实现这些定义。

## 参考资料

- [Function calling](https://platform.openai.com/docs/guides/function-calling) — OpenAI (2023)
  这份官方指南提供了定义OpenAI模型可使用的工具（函数）的实用示例和最佳实践，直接反映了本节讨论的类似JSON Schema的结构。
- [JSON Schema](https://json-schema.org/) — Austin Wright, Henry Andrews, Ben Hutton, Greg Dennis (2020)
  Publisher: IETF Trust
  JSON Schema的官方网站和规范，它是定义JSON数据结构和验证的基础标准，被LLM框架广泛用于指定工具的输入和输出。
- [LangChain Documentation: Tools](https://python.langchain.com/docs/modules/agents/tools/) — LangChain (2024)
  为LangChain框架中定义和集成工具提供了实用指导和示例，LangChain是构建由LLM驱动的应用程序和代理的广泛使用的库。

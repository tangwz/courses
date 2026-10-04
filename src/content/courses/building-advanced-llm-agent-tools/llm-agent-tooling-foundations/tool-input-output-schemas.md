---
course: "building-advanced-llm-agent-tools"
chapter: "llm-agent-tooling-foundations"
lesson: "tool-input-output-schemas"
sourceId: 6545
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-1-llm-agent-tooling-foundations/tool-input-output-schemas"
title: "工具输入和输出模式的优良做法"
description: "了解定义工具输入和输出模式的优良做法，确保LLM的清晰度和可用性。"
order: 4
plots: []
sourceHash: "fafe8f59cc8ac29ce0f514439331ce0bb1684e20abada0a3eb0409f136aa0b66"
sourceCorrections: []
---

为使LLM代理可靠地使用工具，它需要清晰地了解工具需要哪些信息以及将返回哪些信息。这时，输入和输出模式就发挥作用了。可以将模式视为LLM代理与其工具之间的正式约定。明确定义的模式能消除歧义，减少错误，并使您的代理更可靠。以下是定义这些模式的优良做法，确保您的LLM能有效地与工具互动。

正如我们已经讨论了清晰的工具描述对于LLM理解的重要性一样，模式是支撑这些描述的结构性骨干。它们提供了数据交换的精确格式。

## 定义输入模式

输入模式规定了您的工具执行其功能所需的数据结构和类型。清晰定义的输入模式使LLM能够正确地组织其请求，显著增加工具成功执行的可能性。

### 输入模式的要素

1. **数据类型**：明确指定每个输入参数 (parameter)的数据类型。常见类型包括：

   - `string`：用于文本数据。
   - `number`：用于数值数据（整数或浮点数）。
   - `integer`：用于整数。
   - `boolean`：用于真/假值。
   - `array`：用于项的列表（指定数组中项的类型）。
   - `object`：用于包含键值对的结构化数据。

   为端口指定“数值”类型而不是通用的“字符串”类型，有助于LLM提供有效输入，并使您的工具更容易进行类型检查。
2. **必选与可选参数**：明确区分哪些参数是工具运行所需的强制性参数，哪些是可选的。这有助于LLM做出明智的决定，了解它绝对必须收集或生成哪些信息。
3. **参数描述**：就像工具本身需要良好描述一样，输入模式中的每个参数也应有清晰、简洁的描述。此描述引导LLM理解每个参数的用途以及预期值的类型。例如，对于名为`user_id`的参数，像“用户的唯一标识符”这样的描述比没有任何描述有帮助得多。
4. **枚举 (Enums)**：如果参数只能接受有限的预定义值集，请使用枚举。例如，天气工具的`unit`参数可能接受`"celsius"`或`"fahrenheit"`。将其定义为枚举可避免LLM尝试使用不受支持的单位，例如`"kelvin"`。
5. **嵌套结构**：对于需要复杂输入的工具，您可以使用嵌套对象或对象数组。例如，订单处理工具可能预期`line_item`对象的数组，其中每个`line_item`包含`product_id`和`quantity`。

### 使用JSON Schema定义输入

JSON Schema是描述JSON数据结构的广泛采用标准。它是定义工具输入模式的极佳选择，因为它富有表现力、机器可读且人类可读。许多LLM框架和代理系统设计为与使用JSON Schema指定参数 (parameter)的工具定义配合使用。

以下是使用JSON Schema为简单“获取天气”工具定义的输入模式示例：

```json
{
  "type": "object",
  "properties": {
    "location": {
      "type": "string",
      "description": "城市和州，例如：旧金山，加利福尼亚州"
    },
    "unit": {
      "type": "string",
      "description": "温度单位",
      "enum": ["celsius", "fahrenheit"],
      "default": "celsius"
    }
  },
  "required": ["location"]
}
```

在此模式中：

- 整体输入是一个`object`。
- 它有两个属性：`location`（字符串）和`unit`（字符串）。
- `location`是一个必选参数。
- `unit`是可选的，默认值为`"celsius"`，并且只能是两个指定的枚举值之一。
- 每个属性都有一个`description`来帮助LLM理解其用途。

## 定义输出模式

正如输入模式定义工具所需内容一样，输出模式定义工具在成功执行后将返回的数据结构和类型。这种可预测性对于LLM有效地解析工具的响应并将其用于后续推断或形成答案很重要。

### 输出模式的要素

1. **可预测的结构**：LLM需要知道预期哪些字段以及它们的数据类型。输出模式提供此指导。
2. **一致性**：工具必须始终返回符合其定义输出模式的数据。如果一个字段定义为数值类型，它应该始终返回一个数值，而不是有时返回一个表示数值的字符串。
3. **足够多的细节**：输出应提供足够的信息以帮助LLM达成目标，但避免用不必要的数据使其负担过重。模式有助于确定这种平衡。
4. **错误信息**：尽管详细的错误处理将在专门章节中讨论，但您的输出模式设计也应考虑如何传达错误。常见做法是有一个一致的方式来指示成功或失败，如果失败，则提供结构化的错误消息。例如，成功时可能存在主要输出字段，或者失败时可能存在`error`对象。

### 使用JSON Schema定义输出

JSON Schema同样有效于定义输出结构。它使LLM能够预知其将接收数据的格式。

继续以“获取天气”工具为例，以下是其输出模式的示例：

```json
{
  "type": "object",
  "properties": {
    "temperature": {
      "type": "number",
      "description": "指定单位的当前温度。"
    },
    "condition": {
      "type": "string",
      "description": "天气状况的简要描述（例如：晴朗、多云、有雨）。"
    },
    "humidity_percent": {
      "type": "number",
      "description": "当前湿度百分比。"
    },
    "requested_location": {
        "type": "string",
        "description": "获取天气信息的地点。"
    }
  },
  "required": ["temperature", "condition", "humidity_percent", "requested_location"]
}
```

此模式告诉LLM预期一个包含`temperature`、`condition`、`humidity_percent`和`requested_location`的对象，以及它们的类型和描述。了解此结构后，LLM可以可靠地提取所需的信息。

以下图表说明了输入和输出模式如何促进LLM代理与工具之间的互动：

> 模式作为约定，确保LLM和工具了解彼此的数据预期。

## 模式的通用优良做法

无论是定义输入模式还是输出模式，一些优良做法将帮助您为您的工具创建有效且可维护的约定：

- **明确且无歧义**：避免模糊定义。LLM依赖于模式的精确性。例如，如果字段表示日期，请指定它是日期字符串（最好注明格式，例如ISO 8601）还是Unix时间戳。
- **使用特定数据类型**：如果数据本质上是数字，优先使用`integer`或`number`而不是`string`。对布尔值使用`boolean`类型。这有助于更好的验证和更明确的意图。
- **描述性命名和描述**：为参数 (parameter)和字段选择清晰、不言自明的名称。用面向LLM的详细描述来补充它们。这些描述通常比代码中的注释更重要，因为LLM直接使用它们进行推断。
- **力求简洁**：设计尽可能简单的模式，同时仍能传达必要信息。如果扁平、简单的设计能达到相同目的，请避免过度嵌套或复杂结构。复杂性会增加LLM误解或工具实现中出错的可能性。
- **考虑LLM的视角**：始终以LLM作为主要使用者来设计您的模式。如何让LLM最容易理解要提供什么以及期待什么？
- **迭代和测试**：工具和模式设计通常是一个迭代过程。使用LLM测试您的工具，观察它如何理解模式和使用工具，并根据这些观察调整您的模式。
- **必要时包含单位或上下文 (context)**：如果数值带有单位（例如：米、秒、摄氏度），要么将单位作为参数的一部分（例如`duration_seconds`），要么为单位提供一个单独的参数，如天气示例所示。这可以避免误解。
- **规划模式演进**：随着工具的演进，其模式可能需要更改。考虑如何管理模式版本，以与可能使用旧定义的代理保持兼容性。（我们将在第6章更详细地介绍版本管理。）

遵循这些做法，您将为LLM代理与其工具之间的沟通建立起点。明确定义的输入和输出模式不仅仅是形式；它们是构建可靠、可预测、智能的代理系统的重要组成部分。它们将工具从黑盒 (black box)转变为LLM可以理解并自信地使用的清晰定义的组件。

## 参考资料

- [JSON Schema: The home of JSON Schema](https://json-schema.org/) — The JSON Schema team (2024)
  JSON Schema的官方网站和文档，这是一个广泛用于定义数据结构的规范，提供了模式定义的权威参考。
- [Function calling](https://platform.openai.com/docs/guides/function-calling) — OpenAI (2024)
  提供使用JSON Schema为OpenAI语言模型定义工具的实用指南，展示了代理工具和模式设计的实际应用。

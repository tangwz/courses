---
course: "getting-started-model-context-protocol"
chapter: "implementing-tools-and-logic"
lesson: "tool-definition-schema"
sourceId: 7849
sourceUrl: "https://apxml.com/zh/courses/getting-started-model-context-protocol/chapter-3-implementing-tools-and-logic/tool-definition-schema"
title: "工具定义规范"
description: "使用JSON Schema构建工具定义，以向LLM描述功能。"
order: 1
plots: []
sourceHash: "45b89586786b72a6b3575b89bb667a3a1036a824807b57486a2d121bdbb723a5"
sourceCorrections: []
---

任何MCP工具的根本是其规范。当客户端连接到您的服务器时，它会请求一份可用工具列表。您的服务器会返回一个定义，描述函数签名，其格式是大语言模型（LLM）可以解析和理解的。此定义规定了模型如何构建其请求。如果规范松散或定义不清，模型将难以正确执行该工具，或者可能会尝试使用格式不当的参数 (parameter)来调用它。

### 工具定义的构成

MCP规范要求每个工具定义必须包含三个主要组成部分：唯一的名称、功能描述和输入规范。这些部分使宿主应用程序能够将工具作为可用函数提供给模型。

`name`字段作为工具的标识符。它类似于Python或TypeScript中的函数名称。推荐做法是使用snake\_case或camelCase字符串，这些字符串能清楚地归类操作，例如`database_query`或`file_write`。当模型选择调用工具时，它将在其决策输出中引用此名称。

`description`字段与标准代码注释明显不同。注释是为开发人员准备的，而此描述是发送给LLM的提示上下文 (context)的一部分。它充当AI的操作指南。高质量的描述不仅说明工具的功能，还说明应在何种情况下使用它。例如，与其简单地写“获取天气”，更好的描述将是“获取特定城市的当前天气情况。当用户询问温度、降雨或预报时使用此工具。”

`inputSchema`定义工具接受的参数 (parameter)。MCP为此采用JSON Schema标准。这允许您为每个参数定义明确的类型，确保服务器接收到可以处理且不会立即出错的数据。

> MCP工具定义的结构，展示了标识符、模型指令和明确数据验证层之间的关系。

### 了解输入规范

`inputSchema`属性必须是一个JSON对象，描述预期的参数 (parameter)。此对象遵循JSON Schema草案标准。在顶层，类型几乎总是`object`，表示函数预期接收的参数字典。

在此对象中，您定义`properties`。`properties`对象中的每个键都对应一个参数名称。对于每个参数，您必须指定一个`type`，例如`string`、`integer`、`boolean`或`array`。这种类型定义为防范虚构输入提供了第一道防线。如果工具期望端口号为整数，JSON Schema验证可确保模型不会传递“eighty-eighty”之类的字符串。

考虑一个旨在搜索产品数据库的工具。通过MCP协议发送的JSON结构将如下所示：

```json
{
  "name": "search_products",
  "description": "按名称或类别在目录中搜索产品。返回价格和库存状态。",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "搜索词，例如，“无线耳机”"
      },
      "category": {
        "type": "string",
        "enum": ["electronics", "clothing", "home"],
        "description": "按产品类别筛选"
      },
      "max_price": {
        "type": "integer",
        "description": "最高价格（美元）"
      }
    },
    "required": ["query"]
  }
}
```

在此示例中，`query`是唯一的必填参数。`category`参数使用`enum`关键字将输入限制为一组特定的允许字符串。这减少了模型虚构数据库中不存在的类别的可能性。

### 为模型描述参数 (parameter)

正如工具本身需要描述一样，规范中的每个属性都受益于一个描述字段。这是您进行微提示工程 (prompt engineering)的地方。参数的描述准确告诉模型，要从用户的对话历史中提取哪些信息来填充该字段。

如果您有一个名为`sql_query`的参数，像“要执行的SQL命令”这样的描述会有帮助。然而，像“用于查询users表的只读SELECT语句。请勿使用DROP或DELETE”这样的描述既是指令，也是安全防护。模型在生成工具调用之前，会将这些指令融入其推理 (inference)过程。

### 必选参数 (parameter)与可选参数

JSON Schema中的`required`字段是一个字符串列表，指示哪些属性必须存在。必选参数和可选参数之间的区别明显影响模型行为。

当参数被列入`required`时，模型将优先寻找或生成其值。如果对话上下文 (context)中缺少信息，训练有素的模型（如Claude）通常会停止执行并向用户请求澄清，而不是猜测。例如，如果天气工具需要`city`参数，而用户问“天气怎么样？”，模型会回应“哪个城市？”，而不是使用虚构的位置来调用工具。

可选参数允许模型保持灵活。如果`max_results`是可选的，模型可以省略它，您的服务器实现应处理默认值逻辑。

### 规范的复杂度与性能

虽然JSON Schema支持深层嵌套和复杂的验证逻辑（如`oneOf`或`allOf`），但建议尽可能保持工具规范扁平化。深层嵌套的结构会增加LLM的token计数和认知负担，这可能导致更高的延迟或解析错误。

如果工具需要复杂的对象层级，请考虑将其分解为更简单的参数 (parameter)，或将功能拆分为多个更具体的工具。目标是最小化接口中的模糊性，以便模型能够以高可靠性将用户意图映射到函数参数。

### 验证规范

在MCP架构中，`ListToolsRequest`是将这些定义从服务器传输到客户端的机制。

$R_{tools} = \{ T_1, T_2, \dots, T_n \}$

其中每个$T$都是一个完整的工具定义对象。当客户端收到此列表时，它不执行代码，只是注册该功能。规范的验证发生两次：

1. **客户端：** LLM提供方通常会在将请求发回您的服务器之前，验证生成的JSON是否与规范匹配。
2. **服务器端：** 您的服务器必须根据您定义的规范验证传入的`CallToolRequest`。

这种双重验证确保您的工具处理器内的实现逻辑收到干净、类型化的数据，让您能够专注于业务逻辑，而不是对基本数据类型进行大量错误检查。在下一节中，我们将了解如何在Python中使用Pydantic模型自动化此验证。

## 参考资料

- [Function calling - OpenAI API](https://platform.openai.com/docs/guides/function-calling) — OpenAI (2024)
  提供定义大型语言模型可调用函数的实用指南和示例，展示了模式设计和描述编写的最佳实践。

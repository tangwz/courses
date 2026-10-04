---
course: "building-advanced-llm-agent-tools"
chapter: "integrating-external-apis-tools"
lesson: "natural-language-to-api-calls"
sourceId: 6592
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-4-integrating-external-apis-tools/natural-language-to-api-calls"
title: "自然语言到API调用的映射方法"
description: "探讨使LLM能够将自然语言请求翻译成结构化API工具调用的方法。"
order: 4
plots: []
sourceHash: "f5e2552e8bb91cc4ae1ff4d930c4d7370d2dd68a6f00526f90fe9a6229e764bb"
sourceCorrections: []
---

弥合人类语言的流动性、常有歧义的特点与API精确、结构化要求之间的差异，是为LLM代理构建工具时面临的常见难题。代理可能理解用户的请求，例如“伦敦明天天气如何？”，但天气API需要一个像`GET /api/weather?city=London&date=2024-07-15`这样的调用。这里有几种执行这种转换的方法，使LLM代理能够有效地调用外部API。

### 直接由LLM驱动的映射

最直接的方法是运用LLM固有的语言理解能力。通过向LLM提供API工具的详细描述，包括其目的、参数 (parameter)、类型和必填字段（如第1章“理解工具规范和描述”中所讨论），您可以指示LLM直接生成结构化的API调用。

许多现代LLM支持“函数调用”或“工具使用”模式，它们可以输出一个结构化的JSON对象，表示要调用的函数（您的API工具）以及要使用的参数。

**示例：**
假设有一个`get_weather` API的工具描述，它接收`location`（字符串）、`date`（字符串，`YYYY-MM-DD`）和`units`（枚举：`celsius`、`fahrenheit`）作为参数，以及一个用户查询：“下周五奥斯陆会冷吗？请给我摄氏温度。”

LLM在工具描述的指导下，会尝试生成如下输出：

```json
{
  "tool_name": "get_weather",
  "parameters": {
    "location": "Oslo",
    "date": "2024-07-19", // Assuming 'next Friday' is resolved
    "units": "celsius"
  }
}
```

您的代理框架随后会获取此JSON并执行实际的API调用。

**优点：**

- 如果LLM支持结构化输出或函数调用，实现起来相对简单。
- 运用LLM对语言的高级理解能力，包括解决一些歧义。

**注意事项：**

- 映射的质量很大程度上取决于您的工具描述的清晰度和完整性以及LLM的能力。
- LLM可能会“幻觉 (hallucination)”参数值或误解复杂请求。为了获得高可靠性，可能需要大量的提示工程 (prompt engineering)或微调 (fine-tuning)。
- 日期和时间解析（例如“下周五”）可能很棘手。有时，最好让LLM提取原始的时间表达式，然后让您的代码将其解析为特定的日期/时间格式。

### LLM用于实体提取，代码用于映射逻辑

一种更全面的方法，特别是对于复杂API或当您需要更精细的控制时，涉及一个两步过程：

1. **LLM用于实体和意图提取**：使用LLM解析自然语言查询并提取相关实体（如位置、日期、名称、搜索词）和用户意图。LLM在此处的输出是中间结构化表示，而非最终的API调用。
2. **程序化映射**：您的代码获取这些提取的实体，并根据识别出的意图，以编程方式构建精确的API调用，应用任何必要的业务逻辑、验证或数据转换。

**示例：**
用户查询：“我需要找一个从波士顿到圣地亚哥的航班，下周某个时间出发，两周后返回。”

LLM可能会提取：

```json
{
  "intent": "find_flight",
  "origin": "Boston",
  "destination": "San Diego",
  "departure_period": "next week",
  "return_time_relative": "two weeks after departure"
}
```

您的Python代码随后会：

- 将“下周”解析为日期范围（例如，`2024-07-22`到`2024-07-28`）。
- 根据出发时间计算返回期间。
- 将`origin`映射到API的`departureAirportCode`参数 (parameter)（可能涉及从城市名称到机场代码的查找）。
- 将`destination`映射到`arrivalAirportCode`。
- 构建`GET /flights/search`的API调用，并使用正确格式化的参数。

**优点：**

- 拥有更大的控制力和可靠性，因为您的代码处理最终的映射和验证。
- 更容易实现复杂的业务逻辑（例如，“如果用户要求X但未指定Y，则将Y默认设置为Z”）。
- 更易于测试，因为您可以单独测试提取和映射步骤。

**注意事项：**

- 需要更多的开发工作来编写映射逻辑。
- 您需要为LLM的输出设计中间结构化格式。

### 用于API端点或参数 (parameter)选择的语义搜索

当代理能够调用大量API工具，或者单个API具有众多复杂且重叠的参数时，LLM可能难以选择正确的API端点或最适合的参数集。在这种情况下，语义搜索可以作为初步步骤。

1. **嵌入 (embedding)描述**：为您API工具及其参数的描述创建向量 (vector)嵌入。
2. **嵌入查询**：嵌入用户的自然语言查询。
3. **相似性搜索**：在向量空间中执行相似性搜索，以找到其描述与用户查询语义上最相似的API工具或参数。
4. **优化提示**：将匹配度最高的k个工具/参数描述作为上下文 (context)提供给LLM。这缩小了LLM的选择范围，提高了其生成正确API调用结构的准确性。

这种方法对于在复杂的API环境中的发现和消歧特别有用。例如，如果用户说“更新客户记录”，而您有`update_customer_address`、`update_customer_contact_info`和`update_customer_preferences`等API，语义搜索可以根据查询和API描述中更细致的细节，帮助识别其中哪些（或全部）与查询最相关。

**优点：**

- 提高了从大量选项中选择正确工具或参数的准确性。
- 可以处理用户查询中更多样化的措辞。

**注意事项：**

- 增加了复杂性，需要嵌入模型和向量存储。
- 嵌入和描述的质量对于获得良好结果很重要。

### 少量样本提示（Few-Shot Prompting）

LLM是优秀的模式识别器。您可以通过在提示中直接提供几个自然语言查询及其相应的期望API调用结构的示例（样本），来指导LLM的映射行为。这是一种上下文 (context)学习的形式。

**提示示例片段：**

```
Here are some examples of how to map user requests to the 'create_calendar_event' API tool:

User request: "Schedule a meeting with John for tomorrow at 3 PM about the project update."
API call: { "tool_name": "create_calendar_event", "parameters": { "title": "Meeting with John: project update", "attendees": ["john@example.com"], "start_time": "YYYY-MM-DDTH15:00:00", "duration_minutes": 60 } }

User request: "Block out my calendar for a focus session next Monday morning from 9 to 11."
API call: { "tool_name": "create_calendar_event", "parameters": { "title": "Focus Session", "start_time": "YYYY-MM-DDT09:00:00", "end_time": "YYYY-MM-DDT11:00:00" } }

User request: {{user_actual_query}}
API call:
```

LLM随后会尝试遵循示例中显示的模式来处理`{{user_actual_query}}`。

**优点：**

- 可以显著提高特定、重复模式的映射准确性。
- 如果您有代表性的示例，实现起来相对容易。

**注意事项：**

- 效果取决于示例的质量和相关性。
- 示例的数量受LLM上下文窗口大小的限制。
- 对于与所提供示例显著偏离的查询，可能无法很好地泛化。

以下图表阐明了将自然语言查询转换为结构化API调用的一般过程。

> 该图表显示了一个自然语言查询由映射引擎处理，该引擎可以采用一个或多个所描述的方法，以生成一个准备执行的结构化API调用。

### 有效映射的重要考量

无论您选择何种方法，有几个因素对于成功的自然语言到API调用映射都很重要：

- **高质量工具描述**：正如第1章所强调的，LLM需要关于API的全面信息：其目的、每个参数 (parameter)的名称、数据类型（字符串、整数、布尔、枚举）、格式（例如，日期为`YYYY-MM-DD`）、是否必需或可选，以及每个参数代表什么的清晰描述。对于`enum`类型，列出所有可能的值。这是LLM构建其理解的依据。
- **参数值格式化**：确保LLM被引导以API期望的精确格式提供值。例如，如果API期望ISO Alpha-2格式的国家代码（例如，“US”、“GB”），则应在参数描述中指定。如果LLM提取到“United States”，您的映射逻辑（或经过良好指示的LLM）需要进行转换。
- **处理歧义和缺失信息**：自然语言通常不精确。
  - **澄清**：设计您的代理，在参数缺失或请求有歧义时提出澄清问题。例如，如果用户询问“Springfield的天气”，而您的API需要州名，代理可以询问：“您指的是哪个Springfield？”
  - **默认值**：在您的工具规范或映射逻辑中，为可选参数定义合理的默认值。
  - **置信度评分**：如果LLM生成了多个可能的解释，您的系统可能会使用置信度评分（如果LLM提供）或启发式方法来选择最有可能的一个，或者向用户呈现选项。
- **迭代改进**：将自然语言映射到API调用很少能一次就完美。计划使用多样化的用户查询进行大量测试，观察LLM的行为，并迭代改进您的工具描述、提示、示例集或自定义映射逻辑。记录交互和失败的映射对于此过程非常宝贵（在第6章中将进一步讨论）。

最终，将自然语言转换为API调用需要找到恰当的平衡点。您希望发挥LLM的灵活性和理解能力，同时保持可靠API交互所需的精确性和控制。方法的选择将取决于您API的复杂性、期望的可靠性水平以及可用的开发资源。

## 参考资料

- [Function calling](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH1TKzgA14QDfG1OoVaU22T0WHLCxAe8OfIWiNprHF7wVgMf_UxtQt_1Xs1FYLLD6mjHwAEp8Yz-U4sgwUC7SLzUFEOyXzJtyA-bB1I8jWgYEG8o1uMFcMx1Xeca9_53uiyubZhLiFhEq9-0ovV62ugxoZQLnltXY1-DKUzCxoH-j1HQE2NLJwE) — OpenAI (2023)
  使用OpenAI函数调用功能的官方指南，其允许大型语言模型输出表示工具调用的结构化JSON。
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Rewon Child, Mark Polański, Alan Mok, Edward Tabish, Cameron Gray, Alec Radford, Ilya Sutskever (2020)
  Journal: Advances in Neural Information Processing Systems 33 (NeurIPS 2020); DOI: [10.48550/arXiv.2005.14165](https://doi.org/10.48550/arXiv.2005.14165)
  这篇开创性论文展示了大型语言模型少样本学习的有效性，这是通过示例指导大型语言模型的核心原则。
- [OpenAPI Specification Version 3.1.0](https://spec.openapis.org/oas/v3.1.0) — OpenAPI Initiative (2021)
  描述RESTful API的标准，提供了一种结构化方式来定义API端点、参数和数据类型，这对于创建强大的工具规范是不可或缺的。

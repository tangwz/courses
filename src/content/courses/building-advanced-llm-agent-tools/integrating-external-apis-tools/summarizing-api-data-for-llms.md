---
course: "building-advanced-llm-agent-tools"
chapter: "integrating-external-apis-tools"
lesson: "summarizing-api-data-for-llms"
sourceId: 6594
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-4-integrating-external-apis-tools/summarizing-api-data-for-llms"
title: "将API数据汇总并呈现给LLM"
description: "学习如何处理和汇总API响应数据，为LLM提供简洁且相关的信息。"
order: 5
plots: []
sourceHash: "fc5ced12597cc48413f79cb944d084ae7f4c8cb1901904b352cd7bd284b71137"
sourceCorrections: []
---

API响应通常包含大量信息，有时远超LLM完成特定任务所需或能有效处理的范围。原始数据倾泻可能使LLM的上下文 (context)窗口不堪重负，引入干扰，并使代理难以提取所需的准确信息。因此，有效汇总和呈现API数据是构建实用API工具的重要环节。这需要将冗长或复杂的API输出转换为LLM易于理解的简洁、相关的信息。

在将API数据呈现给LLM之前进行汇总的主要原因包括：

- \*\*上下文窗口管理：\*\*LLM在有限的上下文窗口中运行。未经处理的大型API响应会迅速耗尽此限制，可能导致重要信息被截断，或LLM无法考量完整数据集。
- \*\*减少干扰：\*\*API经常返回大量字段或数据点，这些可能与代理的当前目标无关。汇总有助于过滤掉这些干扰，让LLM专注于相关信息。
- \*\*提高处理效率：\*\*LLM可以更快地处理更小、更有针对性的信息片段，且计算开销更低。
- \*\*提升准确性和可靠性：\*\*当LLM接收到清晰、简洁、相关的数据时，它更有可能正确理解信息，从而做出更好的判断或生成更准确的输出。

## 汇总API数据的方法

你的工具中可以使用几种方法来压缩和精简API响应。方法的选择通常取决于API数据的性质和LLM代理的需求。

### 选择性字段提取

最直接的汇总方法是从API响应中只提取基本字段。如果你的工具查询一个用户信息API，该API返回二十个不同的字段，但LLM代理只要求用户的姓名、电子邮件地址和账户状态，那么你的工具应被编程为解析完整响应，然后构建一个只包含这三项信息的新、更小的数据结构。

例如，考虑一个返回全面JSON对象的 天气API：

```json
{
  "location": {
    "name": "San Francisco",
    "region": "California",
    "country": "USA",
    "lat": 37.78,
    "lon": -122.42,
    "tz_id": "America/Los_Angeles",
    "localtime_epoch": 1678886400,
    "localtime": "2023-03-15 10:00"
  },
  "current": {
    "last_updated_epoch": 1678886100,
    "last_updated": "2023-03-15 09:55",
    "temp_c": 12.0,
    "temp_f": 53.6,
    "is_day": 1,
    "condition": {
      "text": "Partly cloudy",
      "icon": "//cdn.weatherapi.com/weather/64x64/day/116.png",
      "code": 1003
    },
    "wind_mph": 5.6,
    "wind_kph": 9.0,
    // ... 更多字段 ...
  }
  // ... 可能包括天气预报数据、空气质量等 ...
}
```

如果LLM代理的任务是“获取旧金山的当前温度和天气状况”，你的工具应处理这个冗长的响应，并返回一个专注的结构：

```json
{
  "city": "San Francisco",
  "temperature_celsius": 12.0,
  "condition": "Partly cloudy"
}
```

这显著减少了LLM需要处理的数据量。这种选择和转换的逻辑存在于你的Python工具代码中。

### 数据聚合

当API响应包含产品、文章或交易记录等列表项时，返回整个列表可能不切实际。你的工具可以聚合这些信息：

- 提供项目**数量**（例如，“找到37个符合你条件的产品。”）。
- 计算数值数据的**汇总统计数据**（例如，“平均客户评分：4.5星，总销售单位：1,280”）。
- 返回**前N个**最相关的项目，可能根据对查询重要的特定属性进行排序。

设想一个API返回一个近期客户订单列表，每个订单都有金额：

```json
[
  {"order_id": "ORD1001", "amount": 50.75, "date": "2023-10-25"},
  {"order_id": "ORD1002", "amount": 120.00, "date": "2023-10-25"},
  {"order_id": "ORD1003", "amount": 35.50, "date": "2023-10-26"}
  // ... 可能还有数百个条目 ...
]
```

如果LLM代理询问“10月25日的总销售额是多少？”，该工具可以执行聚合：

```json
{
  "date_queried": "2023-10-25",
  "total_sales_amount": 170.75,
  "number_of_orders": 2
}
```

### 内容转换与简化

API数据有时需要转换为LLM更容易理解或更符合自然语言的格式。

- \*\*代码转文本：\*\*将数字或模糊的状态码（例如，`payment_status: 2`）转换为描述性文本（例如，`payment_status_description: "Completed"`）。
- \*\*布尔值转描述性文本：\*\*将`is_premium_user: true`更改为`user_tier: "Premium"`。
- \*\*扁平化嵌套结构：\*\*如果API返回深度嵌套的JSON对象，但层次关系对LLM的任务不重要，你的工具可以扁平化部分结构以简化。

例如，任务管理API可能会返回一个带有`status_id`的任务对象：

```json
{
  "task_id": "PROJ-42",
  "title": "Refine API data presentation layer",
  "assigned_user_id": "DEV-03",
  "status_id": 4, // 4 在内部指代“审核中”
  "creation_timestamp": 1678880000
}
```

你的工具可以将其转换为LLM更易于理解的形式：

```json
{
  "task_name": "Refine API data presentation layer",
  "current_status": "Under Review",
  "created_on": "2023-03-15"
}
```

这个内置在你的工具中的转换层直接改善了API数据对LLM的可用性。

### 基于LLM的汇总

对于包含大段非结构化文本（例如，完整文章、详细产品描述、大量用户评论）的API响应，你可能会考虑在数据到达主代理之前，使用LLM进行汇总作为预处理步骤。

工作流程通常包括：

1. 你的API工具调用外部API并获取冗长文本 (long context)。
2. 该工具随后对LLM进行二次调用（这可以是驱动你代理的相同模型，也可以是针对汇总进行了优化的不同模型）。用于此汇总LLM的提示会指示它压缩文本，可能专注于与代理更广泛任务相关的方面，并遵守指定长度（例如，句子数量或token数量）。
3. 由这个次级LLM生成的简洁汇总随后作为主工具的输出返回给代理。

> 该数据流图示了一个API工具如何使用二次LLM调用，对来自外部API的冗长文本进行预汇总。

**基于LLM的汇总的重要考量：**

- \*\*延迟：\*\*这种方法引入了额外的LLM调用，这会增加工具返回结果的总耗时。
- \*\*成本：\*\*每次LLM调用都有相关成本。使用LLM进行预汇总会增加运营开销。
- \*\*汇总质量：\*\*汇总的有效性取决于汇总LLM的能力和所用提示的清晰度。
- \*\*信息保真度：\*\*必须注意确保汇总过程不会无意中遗漏对代理后续推理 (inference)或行动重要的信息。

这种方法对于处理大量文本内容特别有用，但应审慎实施，平衡简洁性益处与增加延迟和成本的潜在缺点。

## 向LLM呈现汇总数据

汇总API数据后，将其呈现给LLM是工具内的最后一步。目标是以明确、LLM（或代理框架）易于解析且直接有助于其任务的格式提供信息。

- **使用清晰一致的数据格式：**

  - \*\*JSON：\*\*这通常是首选格式，特别是当LLM擅长处理结构化数据，或工具的输出需要由代理框架进行编程访问时。确保JSON格式良好。
  - \*\*纯文本：\*\*对于非常简单的信息，经过仔细格式化的自然语言字符串可以很有效。例如：“巴黎明日天气预报：晴，最高气温22°C。”
  - **键值对：**“键: 值”的简单列表适合呈现一小部分不同数据点。
  - \*\*Markdown：\*\*对于文本汇总，Markdown可用于添加轻量级结构，如列表、粗体强调或标题，这可以提高LLM的可读性。
- \*\*提供足够的上下文 (context)：\*\*工具返回的数据应是可理解的。这意味着不仅要返回值，还要解释这些值代表什么，特别是当其并非本身显而易见时。

  - 与其返回 `{"value": 75}`，不如考虑 `{"query_parameter": "disk_space_percentage_used", "current_value": 75, "unit": "%"}`。
  - 如果返回列表，请描述列表包含什么：`{"user_id": "U123", "recent_orders_summary": [{"order_id": "X789", "status": "Shipped"}, ...]}`。
    工具的描述，即LLM在决定使用工具前查阅的内容，也应设定对输出格式和内容的预期。
- \*\*与LLM预期和任务要求保持一致：\*\*工具输出的结构应根据LLM预期的使用方式进行设计。如果LLM需要从输出中提取特定实体，请使这些实体独特且易于识别。如果LLM的目的是将信息合成自然语言响应以供最终用户使用，则工具的输出应促进这一点。
- \*\*简洁地管理输出模式描述：\*\*当你为LLM代理定义工具时，其输出模式的描述（即LLM应预期工具返回什么）需要清晰而简洁，以避免在代理的提示中消耗过多token。工具返回的实际数据实例可以更丰富，但模式描述本身应是高效的汇总。
- \*\*优雅地处理“无数据”情况和错误：\*\*你的工具的汇总和呈现逻辑还必须考虑外部API对查询未返回相关数据或报告错误的情况。你的工具不应直接将空响应或原始API错误消息（这可能模糊或过于技术化）传递给LLM，而应将其转换为清晰、信息丰富的消息。

  - API返回类似于 `{"errorCode": 503, "errorMessage": "Service Unavailable"}` 的错误。你的工具可以将其翻译为：`{"status": "Error", "details": "外部数据服务暂时不可用。请稍后重试。"}`
  - API对搜索返回空列表 `[]`。你的工具可以返回：`{"status": "Success", "data": null, "message": "未找到符合你搜索条件的条目。"}`

通过投入精力审慎汇总API响应，并以结构化和可理解的方式呈现，你可以显著提高LLM代理使用外部数据的能力。这个精简层通常是区分基本API集成与高效、可靠工具的因素，后者能提升代理性能。始终考虑代理的视角：工具的输出应直接帮助LLM完成导致该工具调用的请求。这种量身定制的方法是构建高级且可靠的LLM代理工具不可或缺的一环。

## 参考资料

- [Function Calling](https://platform.openai.com/docs/guides/function-calling) — OpenAI (2023)
  提供了关于大型语言模型如何与结构化数据输入和输出（包括JSON格式和模式定义）进行工具交互的官方指南。
- [A Survey on Large Language Model based Text Summarization](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGrSLwOfP6r_f5lZr4nn52uugmg7m0vu1ELbR6xH_zfLeSzDa93MAp1oFoE7HNONLqFK02FouN_MYNGmw9b_uW-OnKh2X1hWmzfQdSDujq6SgxDonwsFL6cTUPE9rE=) — Xiaofei Sun, Jiawei Li, Xinze Li, Yutong Zhang, Jiawei Fan, Huifeng Li, Zhaofei Yu, Ming Zhou (2023)
  Journal: arXiv preprint arXiv:2308.08272
  介绍了大型语言模型在文本总结中的应用，涵盖了该领域内的各种技术、挑战和研究方向。

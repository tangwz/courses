---
course: "prompt-engineering-agentic-workflows"
chapter: "prompt-engineering-tool-use"
lesson: "managing-tool-inputs-outputs-prompts"
sourceId: 6728
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-agentic-workflows/chapter-3-prompt-engineering-tool-use/managing-tool-inputs-outputs-prompts"
title: "通过提示词管理工具输入和输出"
description: "编写提示词来管理代理与其工具之间的数据流，包括输入和输出。"
order: 4
plots: []
sourceHash: "ea8f1f02df055c661c5fb835ad5f1b78dded2d1af9f4dbadb6470afca7e3ba30"
sourceCorrections: []
---

有效使用外部工具，不仅仅是指示AI代理调用某个特定功能；这需要细致管理流入和流出这些工具的数据。你的提示词 (prompt)是实现这种控制的主要界面，它们指引代理如何组织输入以确保工具准确运行，以及如何理解输出以继续完成任务或告知用户。鉴于大型语言模型（LLMs）基于文本运行，而工具常期望JSON等结构化数据或返回复杂响应，你的提示词必须弥合这一差异。

### 通过提示词 (prompt)组织工具输入

为了让代理正确使用工具，它需要以工具期望的准确格式提供输入。你的提示词必须指引代理准备这些输入。

#### 明确的指令和参数 (parameter)定义

指引输入格式最直接的方式，是在提示词中明确详细说明所需的参数、它们的数据类型以及任何特定的格式规则。这减少了模糊性，并增加了工具接收到有效数据的可能性。

例如，如果你有一个工具 `calculate_loan_payment`，它需要 `principal`（数字）、`annual_interest_rate`（数字，例如5%用0.05表示）和 `loan_term_years`（整数），你的提示词可能包含：

```text
要使用 `calculate_loan_payment` 工具，你必须提供以下参数：
- `principal`：贷款总额（例如，20000）。
- `annual_interest_rate`：年利率，以小数形式表示（例如，6.5%的利率，提供0.065）。
- `loan_term_years`：贷款的年限（例如，5）。

用户希望计算一笔30,000美元贷款在10年内以4.2%年利率的还款。为 `calculate_loan_payment` 工具准备输入。
```

代理随后应该能够确定工具调用的正确值和结构。

#### 生成结构化输入（JSON/XML）

许多工具，特别是与API交互的工具，期望JSON或XML等结构化格式的输入。你可以指示代理直接生成这些结构。当工具输入本身复杂或包含嵌套元素时，这特别有用。

考虑一个代理需要使用 `cart_add_item` 工具向电子商务购物车添加一个物品的场景。

提示词片段：

```text
`cart_add_item` 工具期望一个JSON对象，其中包含 `product_id`（字符串）和 `quantity`（整数）。
用户希望将2件产品“A-B-123”添加到购物车。为 `cart_add_item` 工具生成JSON输入。
```

代理的内部推理 (inference)（或如果这样设计的话，实际输出）应该生成类似以下的内容：

```json
{
  "tool_name": "cart_add_item",
  "tool_input": {
    "product_id": "A-B-123",
    "quantity": 2
  }
}
```

你的代理框架随后会解析此JSON并执行工具调用。

#### 从上下文 (context)动态生成输入

代理常需要从当前对话、先前的工具输出或其内部记忆中提取输入值。提示词可以指引这个提取过程。

例子：

```text
用户：我想预订下周五从伦敦到纽约的航班。
代理的提示词：用户想要预订航班。使用 `flight_search` 工具。
它需要 `origin_city`、`destination_city` 和 `departure_date`。
从用户的请求和今天的日期（假设今天是2023-10-20）中提取这些信息，以确定“下周五”。然后，为 `flight_search` 组织输入。
```

代理需要解析“伦敦”、“纽约”，计算“下周五”（例如，2023-10-27），然后构建输入。

#### 处理可选参数和默认值

工具常有带默认值的可选参数。你的提示词应让代理了解这些，以便它能决定是提供这些参数，还是让默认值生效。

`image_resize` 工具的提示词片段：

```text
`image_resize` 工具接受 `image_url`（字符串，必需）、`width`（整数，可选）和 `height`（整数，可选）。
如果只提供 `width`，则保持 `height` 的宽高比。如果只提供 `height`，则保持 `width` 的宽高比。
如果 `width` 和 `height` 都未提供，则将较大维度调整为默认的512像素，同时保持宽高比。
将“http://example.com/image.jpg”处的图片调整为800像素的宽度。
```

这指引代理提供 `image_url` 和 `width`，并且理解 `height` 将由工具或由提示词指引的后续处理步骤进行适当处理。

### 通过提示词 (prompt)处理工具输出

工具一旦执行，就会返回一个输出。此输出可能是一个简单值、一个复杂的JSON对象、一个文本块或一条错误消息。代理需要指引，再次通过提示词，来理解并运用这些信息。

#### 总结和信息提取

工具输出，如冗长的API响应或网页内容，可能非常详细。你常需要代理总结这些信息，或提取与其当前任务有关的具体细节。

例子：一个 `web_search` 工具返回了关于一家公司的较长文本 (long context)片段。
提示词片段：

```text
`web_search` 工具返回了关于“Innovatech Corp.”的以下信息：
“[包含公司历史、产品和近期新闻的冗长文本...]”
提取首席执行官（CEO）的姓名，并总结Innovatech的主要产品线。
```

代理随后应处理文本并提供简洁的回答。

#### 格式转换和呈现

工具输出可能不是用户友好的格式。提示词可以指示代理将输出转换为自然语言，或转换为更适合后续步骤或用户展示的结构化格式。

例子：一个 `database_query` 工具返回了一列产品记录。
工具输出：

```json
[
  {"id": "P101", "name": "Wireless Mouse", "price": 25.99, "stock": 150},
  {"id": "P102", "name": "USB Keyboard", "price": 45.00, "stock": 80}
]
```

提示词片段：

```text
`database_query` 工具返回了上述产品列表。将这些信息以可读的摘要形式呈现给用户，突出显示库存低于100的产品。
```

代理随后可能生成：“我们找到了这些产品：无线鼠标售价25.99美元，库存150个。USB键盘售价45.00美元，库存80个，低于我们100的阈值。”

#### 基于输出的条件逻辑

代理常需要基于工具输出做出决策。提示词可以为代理定义它要遵循的条件逻辑。

例子：一个 `payment_process` 工具。
工具输出（情景1 - 成功）：`{"status": "success", "transaction_id": "txn_123abc"}`
工具输出（情景2 - 失败）：`{"status": "failed", "error_message": "Insufficient funds"}`

提示词片段：

```text
`payment_process` 工具已完成。
如果 `status` 是“success”，向用户确认交易ID，并继续进行 `shipping_安排` 工具的操作。
如果 `status` 是“failed”，告知用户 `error_message`，并询问他们是否想尝试不同的支付方式。
工具返回：[在此处插入工具输出]
你的下一步是什么？
```

这指引了代理的分支逻辑。

#### 理解错误消息

工具可能会失败，并返回错误消息。提示词必须使代理具备能力去理解常见错误，并建议合适的行动，例如重试、向用户请求澄清，或尝试另一种方法。

例子：一个 `api_call` 工具返回一个错误。
工具输出：`{"error_code": 401, "message": "Unauthorized: API key invalid or expired."}`

提示词片段：

```text
`api_call` 工具返回了以下错误：
`{"error_code": 401, "message": "Unauthorized: API key invalid or expired."}`
这通常表示提供的API密钥不再有效。告知用户此具体问题，并请求他们提供一个更新的API密钥。
```

这使得代理能够优雅地处理错误，而不是停止运行或提供无用的响应。

### 在多工具交互中管理数据流

在许多代理工作流程中，一个工具的输出会成为另一个工具的输入，或影响代理的后续决策。提示词 (prompt)对于管理这种数据流以及在多个步骤中保持上下文 (context)很重要。

考虑一个负责规划旅行的代理：

1. **用户请求**：“计划一次到阳光海滩目的地的周末旅行，预算在500美元以内。”
2. **提示词1（目标拆解与工具选择）**：代理识别出子任务：查找目的地、查询航班、查询酒店、核实预算。
3. **工具1（`find_beach_destinations`）**：输入来源于“阳光海滩”。
   - 工具1输入提示词：“使用 `find_beach_destinations`，条件是：‘阳光充足’。输出应为城市名称列表。”
   - 输出1：`["Miami", "Cancun", "Malibu"]`
4. **提示词2（迭代与工具2选择）**：“对于列表 `["Miami", "Cancun", "Malibu"]` 中的每个目的地，使用 `flight_search` 和 `hotel_search` 工具查找下个周末的选项。跟踪费用。”
5. **工具2与3（`flight_search`、`hotel_search`）**：输入包含来自输出1的目的地。
   - 工具2/3输入提示词：“对于‘Miami’，查找[日期]的航班和[日期]的酒店。提取价格。”
   - 输出2与3：Miami的航班和酒店选项及价格。
6. **提示词3（聚合与决策）**：“汇总Miami的费用。它是否在500美元预算内？如果是，呈现出来。如果不是，或需要更多选项，则对‘Cancun’重复此操作。”

这个序列显示了提示词如何持续指引代理，确保数据（如目的地名称和价格）在工作流程中被正确传递和处理。

> 这张图表阐明了一个多工具数据流，其中提示词指引代理如何使用一个工具的输出来作为后续工具的输入，逐步构建出一个解决方案。

### 提示工具输入/输出的最佳做法

掌握通过提示词 (prompt)管理工具输入和输出，是一个持续完善的过程。以下是一些指引你的最佳做法：

- **明确且无歧义**：清楚定义数据格式、必需字段、可选字段和可能的值。你的指令越具体，代理误解的空间就越小。
- **提供提示词内示例（少样本学习 (few-shot learning)）**：向代理展示你期望的具体内容。对于输入生成，包含一个期望的JSON或参数 (parameter)结构示例。对于输出处理，展示一个原始输出的例子和期望的处理形式。

  ```text
  // 输入格式示例
  工具：`add_event_to_calendar`
  期望的输入格式：{"event_title": "string", "date": "YYYY-MM-DD", "time": "HH:MM"}
  用户说：“将‘团队会议’添加到我明天的日程中，时间是下午2点。”
  （假设今天是2023-11-15）
  生成工具输入。
  ```
- **指导错误理解与恢复**：不要假设工具总是成功。提供关于代理应如何理解常见错误代码或消息，以及可以采取哪些后备行动的指导（例如：“如果你从 `get_document` 工具收到404错误，告知用户文档未找到，并询问他们是否有替代ID。”）。
- **迭代并严格测试**：你的提示词第一个版本可能不完美。在各种场景下测试你的代理，观察它如何处理工具输入和输出，并相应地完善你的提示词。在此处，记录代理的推理 (inference)以及与工具交换的数据非常有价值。
- **鼓励大型语言模型（LLM）生成结构化输出供内部使用**：当大型语言模型处理工具的输出来供进一步内部使用时（例如，输入给另一个工具或决策过程），提示它将该输出总结或转换为一致的结构化格式（如JSON）。这使得你的代理代码能够更可靠地使用这些信息。
- **考虑上下文 (context)窗口**：如果工具输出非常大，请注意大型语言模型的上下文窗口。你可能需要提示代理总结或只提取最相关的信息，以避免超出限制或丢失旧的上下文。信息压缩的技术，在关于记忆的章节中会讲到，在此处也相关。

通过精心设计提示词来管理工具输入和输出的细节，你帮助你的AI代理更有效、更可靠地与外部环境交互，从而显著增强其执行复杂任务的能力。

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2022)
  Journal: arXiv preprint arXiv:2210.03629; Publisher: arXiv; DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  本文介绍了ReAct框架，展示了语言模型如何结合推理（如规划、解释观察）和行动（如使用工具）来解决复杂任务。这对于理解LLM如何解释工具输出以指导后续行动至关重要。
- [Function calling](https://platform.openai.com/docs/guides/function-calling) — OpenAI (2024)
  OpenAI的官方文档提供了关于如何定义工具模式以及指导大型语言模型调用函数、生成结构化输入（如JSON）并在代理工作流中处理工具响应的实践指南。
- [Enabling language models to use tools](https://www.deepmind.com/blog/enabling-language-models-to-use-tools) — Google DeepMind (2023)
  Publisher: Google DeepMind Blog
  一篇来自Google AI的博客文章，讨论了语言模型与外部工具交互的能力和实现，包括为有效工具使用设计提示以及在代理应用中管理数据流的见解。

---
course: "prompt-engineering-llm-application-development"
chapter: "output-parsing-validation-reliability"
lesson: "prompting-structured-data-revisited"
sourceId: 3075
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-7-output-parsing-validation-reliability/prompting-structured-data-revisited"
title: "提示生成结构化数据（再谈）"
description: "强化引导 LLM 生成 JSON 等结构化输出的方法。"
order: 2
plots: []
sourceHash: "2f4b2492edb9587d7456e0b8c9e28c2e55637f2e49129e5a4f0184a6e4d5b6c8"
sourceCorrections: []
---

大语言模型 (LLM) 响应的自由形式在将其集成到软件应用中时构成了一大障碍。尽管人类能轻易理解对话文本，但程序通常需要可预测的、结构化数据格式（例如 JSON 或 XML）以便进行可靠的处理。仅仅向 LLM 提问并寄希望于最好的结果，通常会导致难以或无法自动解析的输出，从而引起应用错误。

我们在第2章中提到了如何引导 LLM 生成特定输出格式。现在，当我们专注于构建可靠的应用时，让我们重新审视并强化那些方法。让 LLM 在源头正确生成结构化数据是处理输出的第一步。尽管这并非完美保证，但有效的提示能大大增加获得应用可直接使用数据的可能性，从而减轻后续解析和验证逻辑的负担。

### 优化结构化指令

最直接的方法仍然是明确指示模型所需的格式。然而，仅仅说“使用 JSON”可能不足以获得一致的结果，特别是在要求复杂的情况下。指令的精确性很重要。

- **明确格式：** 不要只说“格式为 JSON，”要指定*确切的结构*。例如：“*仅*回复一个 JSON 对象，其中包含键 'summary'（字符串）和 'keywords'（字符串列表）。”
- **使用清晰的分隔符：** 指示模型将结构化输出包裹在特定标记 (token)中。这有助于后续的程序化提取，即使模型添加了对话性的填充文本。示例：“提供用三个反引号包裹的 JSON 输出，示例如下：`json { ... }` ”
- **将指令与角色提示结合：** 分配一个角色有时可以隐式地引导格式。“你是一个有用的助手，负责提取信息并始终以 JSON 格式回复。从文本中提取姓名和电子邮件。*仅*回复 JSON 对象。”

### 采用少样本示例来规定结构

演示您期望的确切输出格式通常比描述它更有效。少样本提示，即在提示本身中提供示例，对于强制执行结构非常有效。

考虑要求 LLM 提取联系信息：

**零样本（结构化输出可靠性较低）：**

```text
从以下文本中提取姓名和城市，并将其作为 JSON 对象提供：
"John Doe 住在 Springfield，可以通过 john.d@email.com 联系到他。"

仅回复 JSON 对象。
```

*潜在问题：* 模型可能会生成有效的 JSON，但确切的键名或结构在不同运行之间可能略有差异。

**少样本（结构化输出更可靠）：**

````text
从文本中提取姓名和城市，并将其作为 JSON 对象提供，键名为 "contact_name" 和 "location_city"。

文本: "Alice Smith works in Metropolis. Her email is alice.s@email.com."
JSON:
```json
{
  "contact_name": "Alice Smith",
  "location_city": "Metropolis"
}
````

文本: "Bob Johnson resides in Gotham."
JSON:

```json
{
  "contact_name": "Bob Johnson",
  "location_city": "Gotham"
}
```

文本: "John Doe lives in Springfield and can be reached at [john.d@email.com](mailto:john.d@email.com)."
JSON:

```json
{
  "contact_name": "John Doe",
  "location_city": "Springfield"
}
```

````
通过提供具体示例（`{"contact_name": "...", "location_city": "..."}`），您可以大大限制 LLM 的输出范围，使其更有可能遵循您期望的结构。这些示例隐含地定义了所需的键和期望的值类型（在此例中为字符串）。

### 在提示中定义结构

对于更复杂的结构，特别是涉及嵌套对象、不同数据类型或可选字段的结构，在提示中明确定义一个结构会很有益。这比单独的示例提供了更强的约束。您不一定需要形式化的结构语言；清晰的描述通常就足够了。

**带有结构描述的提示示例：**

```text
从下面的用户评论中提取产品信息。将输出格式化为符合以下结构的 JSON 对象：

- "product_name": 字符串（必需）- 提及的产品名称。
- "rating": 整数（可选）- 如果提及，给出的星级评分（1-5）。
- "sentiment": 字符串（必需）- 整体情绪（“积极”、“消极”、“中性”）。
- "features_mentioned": 字符串列表（可选）- 讨论的任何具体产品功能。

*仅*回复用三个反引号包裹的 JSON 对象。如果可选字段未在评论中出现，则从 JSON 中省略它。

评论: "This SuperWidget is amazing! Setup was easy and it works perfectly. 5 stars! The battery life is great."

JSON 输出:
````

指示模型关于必填与可选字段以及期望的数据类型（字符串、整数、列表）有助于防止诸如缺少必需数据或将评分作为字符串（“5 stars”）而非整数（5）返回等错误。

### 常见挑战与提示策略

即使仔细提示，您仍可能遇到问题：

1. **不完整或无效的 JSON：** 模型可能在生成中途停止，或者生成语法不正确的 JSON（例如，缺少逗号、括号不匹配）。
   - **提示缓解方法：** 确保您的 `max_tokens` 设置足以应对预期的输出大小。将非常复杂的提取任务分解为更小、按序的提示有时会有帮助。要求模型在输出前再次检查其 JSON 有效性，偶尔能改善结果，但并非万无一失。
2. **多余文本：** LLM 在结构化块前后包含对话文本（例如，“好的，这是您请求的 JSON：...”）。
   - **提示缓解方法：** 强化诸如“*仅*回复 JSON 对象”之类的指令。使用清晰的分隔符（`json ...` ）可以使您的代码更容易隔离 JSON，即使存在多余文本。
3. **结构或值不正确：** 模型使用了错误的键名，包含意料之外的字段，或者以错误的格式放置数据（例如，字符串而不是数字）。
   - **提示缓解方法：** 精确的指令、少样本示例和结构定义是这里的主要方法。对您的要求尽量明确。

尽管这些提示方法能显著增加获得良好格式化结构化数据的几率，但它们并非百分百的解决方案。LLM 仍然可能犯错、误解指令或生成意料之外的变体。这就是为什么本章中讨论的后续步骤——输出解析和验证——是构建真正可靠的 LLM 驱动应用的重要组成部分。可以将提示以获取结构视为最大化成功的可能性，而将解析/验证视为必要的安全网，以应对仅凭提示不足以解决的情况。

## 参考资料

- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2020); DOI: [10.48550/arXiv.2005.14165](https://doi.org/10.48550/arXiv.2005.14165)
  介绍了上下文学习和少样本提示的概念，这是一种指导大型语言模型的技术，对于通过示例强制输出结构尤其适用。
- [OpenAI Prompt engineering guide](https://platform.openai.com/docs/guides/prompt-engineering) — OpenAI (2024)
  Publisher: OpenAI
  提供了提示大型语言模型的实用策略和最佳实践，包括获取JSON等结构化输出的明确指令和技术。

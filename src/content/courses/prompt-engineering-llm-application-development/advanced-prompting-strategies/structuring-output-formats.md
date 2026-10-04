---
course: "prompt-engineering-llm-application-development"
chapter: "advanced-prompting-strategies"
lesson: "structuring-output-formats"
sourceId: 2984
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-2-advanced-prompting-strategies/structuring-output-formats"
title: "结构化输出格式 (JSON, Markdown)"
description: "引导LLM以JSON或Markdown等特定格式生成回复的提示技巧。"
order: 5
plots: []
sourceHash: "02fbc83b1e8077a87e22b49d552933f511fe4f36372c59df2bc27b31052c8b86"
sourceCorrections: []
---

虽然大型语言模型擅长生成人类可读的文本，但许多应用需要更可预测、机器可解析的输出格式。直接处理自由格式文本可能脆弱且易出错。请求结构化输出，例如JSON或Markdown，是构建更可靠应用的一个基本步骤，这些应用能够以编程方式使用模型生成的信息。引导LLM生成这些特定格式回复的方法将得到介绍。

### 结构化的必要性

设想构建一个从电子邮件中提取联系信息并将其添加到数据库的应用。如果LLM返回一个句子，例如“联系人是John Doe，他的电子邮件是[john.doe@example.com](mailto:john.doe@example.com)，电话是123-456-7890”，你的应用代码需要解析这个字符串以找到相关部分。这种解析逻辑可能变得复杂，并且如果LLM的措辞略有改变，可能会出现问题。

然而，如果你可以指示LLM返回：

```json
{
  "name": "John Doe",
  "email": "john.doe@example.com",
  "phone": "123-456-7890"
}
```

处理变得非常简单。你可以直接将JSON字符串解析为原生数据结构（例如Python字典），并可预测地访问字段。类似地，请求Markdown输出对于生成旨在支持富文本格式的用户界面中呈现的内容也很有用。

优点包括：

- **更易集成：** 结构化数据与其他软件组件、数据库或API顺畅对接。
- **提高可靠性：** 解析定义明确的格式比从自然语言中提取信息更不容易出错。
- **程序化使用：** 使得LLM输出可直接用于配置、数据录入或后续处理步骤。

### 结构化输出的提示技巧

实现结构化输出很大程度上依赖于清晰的指令，有时还包括提示中的示例。以下是常用策略：

1. **明确指令：** 最直接的方法是在提示指令中清晰说明所需格式。具体说明格式（例如，“JSON对象”、“Markdown列表”），如果可能，还要说明该格式内的结构。
2. **模式描述（特别是针对JSON）：** 对于JSON，明确描述预期的键、每个键关联的数据类型（字符串、数字、布尔值、列表），以及字段是必需还是可选。你甚至可以描述嵌套结构。
3. **提供示例（少样本学习 (few-shot learning)）：** 在提示中包含一个或多个你期望的确切输出格式示例。这能强化指令，并为模型提供清晰的遵循模板。
4. **使用分隔符：** 指示模型将结构化输出置于特定分隔符内，例如三个反引号。这有助于将所需输出与模型可能生成的任何对话前言或解释区分开。例如：“提取实体并将它们作为JSON对象返回，用 `json ...`  包裹。”

### 示例：提示生成JSON输出

让我们细化联系信息提取任务。我们希望LLM处理一段文本并返回一个包含姓名、电子邮件和公司的JSON对象。

**提示：**

```text
从以下文本中提取姓名、电子邮件地址和公司名称。将信息作为JSON对象返回，键为“contact_name”、“contact_email”和“company”。如果缺少任何信息，则使用null作为其值。

文本：
“Reach out to Jane Smith from TechCorp Inc. at jane.s@techcorp.com regarding the project update.”

输出JSON：
```

**预期LLM回复：**

```json
{
  "contact_name": "Jane Smith",
  "contact_email": "jane.s@techcorp.com",
  "company": "TechCorp Inc."
}
```

**变体与注意事项：**

- **信息缺失：** 如果文本只是“Jane Smith可以通过[jane.s@techcorp.com](mailto:jane.s@techcorp.com)联系”，模型按照指令，理想情况下应返回：

  ```json
  {
    "contact_name": "Jane Smith",
    "contact_email": "jane.s@techcorp.com",
    "company": null
  }
  ```
- **格式错误的JSON：** LLM有时会犯错，生成略有错误的JSON（例如，缺少逗号、引用不正确）。应用需要下游验证和错误处理（第7章将详细介绍）。
- **清晰性很重要：** 模糊的指令，如“给我JSON格式的联系信息”，比上面更具体的提示成功可能性更小。

### 示例：提示生成Markdown输出

Markdown对于生成格式化文本很有用，例如摘要、列表或简单文档。

**提示：**

```text
根据提供的上下文，总结大型语言模型用于客户支持的主要优点。按照以下格式总结：
- 一个主标题（H2级别），标题为“LLM在客户支持中的优点”。
- 一个无序列表，详细列出至少三个不同的优点。
- 将每个项目符号中的主要要点加粗。

上下文：
[在此插入描述LLM优点的文本：更快的响应时间、24/7可用性、处理常见查询、多语言支持、一致的语气等。]

格式化摘要：
```

**预期LLM回复：**

```markdown
## LLM在客户支持中的优点

*   **更快的响应时间**：LLM可以立即回复客户咨询，显著减少等待时间。
*   **全天候可用**：与人工客服不同，由LLM驱动的机器人可以持续运行，提供全天候支持。
*   **可扩展处理常见查询**：模型高效处理大量重复性问题，使人工客服得以处理复杂事务。
*   **一致的沟通**：LLM在所有互动中保持既定的品牌声音和语气。
```

**变体与注意事项：**

- **格式具体性：** 你可以请求各种Markdown元素，如标题（`#`、`##`）、列表（`*`、`-`、`1.`）、粗体（`**文本**`）、斜体（`*文本*`）、代码块（\`\`\`\`\` \`\`\`\`\`\`\`）和表格。
- **渲染：** Markdown的实用性来自于它能够被渲染成HTML或在支持它的平台（如GitHub、Slack或定制网页界面）上良好显示。

### 挑战与最佳实践

- **模型依赖性：** 一些LLM在遵循格式指令方面比其他模型训练得更好或微调 (fine-tuning)得更优。可能需要进行实验。
- **复杂性与可靠性：** 过于复杂的模式或格式规则会增加LLM犯错的可能性。从简单开始，逐步增加复杂性。
- **勿略验证：** 始终将LLM生成的结构化输出视为可能无效。在使用数据之前，在你的应用中实施解析和验证逻辑。我们将在第7章详细介绍这一点。
- **指令位置：** 将格式指令放在提示的末尾，紧接在预期输出之前，通常会产生好的结果。
- **迭代：** 如果模型无法持续生成所需格式，请优化你的指令、添加更清晰的示例或简化请求的结构。

掌握结构化输出生成是LLM应用开发者的实用技能。通过精心设计指定所需格式（如JSON或Markdown）并提供清晰示例的提示，可以显著提高LLM回复在软件系统中的可靠性和实用性。

## 参考资料

- [Prompt Engineering Guide](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFvrYBdcLLhd3d_CV_F-iy3_XB0uDOTBmsz-Dy-bkC4efKxWS9Gha3fv-1fbgvi0QSYyouVNwcHpVphLPgTgxcJk6ui2Et_btU7N19Ia_cPwWM0OmuQ0MfFjw==) — OpenAI (2024)
  Publisher: OpenAI
  官方指南，提供制作有效提示的全面技术，包括从LLM生成结构化输出的方法。
- [Introduction to Prompt Design](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/introduction-prompt-design) — Google Cloud (2024)
  Publisher: Google Cloud
  大型语言模型提示设计指南，涵盖控制模型响应和实现特定输出格式的最佳实践。
- [Building Large Language Model Applications](https://www.oreilly.com/library/view/building-large-language/9781098144215/) — Reza Shabani, Jonathon Riley (2023)
  Publisher: O'Reilly Media
  面向开发人员的实践书籍，关于将LLM集成到应用程序中，其中包含管理和解析模型输出以获取结构化数据的章节。
- [Output Parsers](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH8E4hkj_i-jiVFvJIK6U7_XDtwRMNR9eGSMXkb4Kvel2d8CdwzEi5Z6ouOLMO0sQN8UWsbphub0bxtXqMitIpa99BsFkF0Ko_BUTAG_B0gQVJD1puxINfjIs22f9fDCRRYP3fA-6rpW6Z9AOTIt3D_0A4ptk=) — LangChain (2024)
  Publisher: LangChain
  官方文档，解释如何使用输出解析器将LLM生成的文本结构化并验证为JSON等格式，常与Pydantic结合使用。

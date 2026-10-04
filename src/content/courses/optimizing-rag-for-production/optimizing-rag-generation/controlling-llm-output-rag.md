---
course: "optimizing-rag-for-production"
chapter: "optimizing-rag-generation"
lesson: "controlling-llm-output-rag"
sourceId: 6190
sourceUrl: "https://apxml.com/zh/courses/optimizing-rag-for-production/chapter-3-optimizing-rag-generation/controlling-llm-output-rag"
title: "控制LLM输出：风格、语气和真实性"
description: "实施方法来控制RAG系统中LLM生成的文本的风格、语气和事实性，以获得期望的输出特点。"
order: 2
plots: []
sourceHash: "02ed30480b5c8c37182af875cb9d60dc2920f979605ae92dd58b0da27eee9e49"
sourceCorrections: []
---

一旦您的RAG系统检索到相关文档，下一个难题是引导大型语言模型（LLM）生成回答，使其不仅提供信息，还要符合风格、语气以及最重要的事实性方面的具体要求。原始的LLM输出虽然通常流畅，但在呈现方式和对所给上下文 (context)的遵守程度上可能会有差异。在生产环境中，这种差异通常是不可接受的。提供了更好地管理LLM生成过程的技术，以确保输出满足您应用的需求。

### 引导LLM输出特点

控制LLM的输出涉及精巧的提示工程 (prompt engineering)、参数 (parameter)调整，有时还包括您与模型交互时的架构选择。目标是使生成步骤可预测，并与您应用的特点及真实性要求保持一致。

下图描绘了这些控制机制通常在RAG系统的生成部分中的位置：

> 交互流程图，展示控制机制如何利用检索到的上下文 (context)影响LLM生成器，从而产生受控输出。

### 管理输出风格

LLM输出的风格指的是其语言特点，例如正式性、复杂性和文学性。调整风格对于符合应用程序的“声音”和用户预期非常重要。

#### 针对风格的提示工程 (prompt engineering)

使用提示是影响风格最直接的方式。

- **系统提示**: 对于支持系统提示的模型（如GPT-4或Claude），系统提示可以在整个对话中为LLM设定一个全局行为或角色。例如：“您是一位专攻古罗马的博学历史学家。您的语言应正式且详细。”
- **语境学习（少量样本提示）**: 在提示中提供所需风格的示例是一种有效方法。

  ```
  User: 用一个简洁专业的段落总结这份文档。
  Context: [检索到的文档文本...]
  Example of desired output: 该研究表明X和Y之间存在显著关联，这表明Z。建议进行进一步研究。
  Assistant: [LLM生成类似专业风格的摘要]
  ```
- **指令提示**: 直接向LLM发出指令。
  - “以自然纪录片解说员的风格，解释光合作用的原理。”
  - “将以下段落改写成适合法律文件的形式。”
    指令的细节程度与您获得的控制能力成正比。

#### 为保持风格一致而进行微调 (fine-tuning)

当需要特定且一致的风格，而提示工程产生的结果不一致或需要过于复杂的提示时，对LLM进行微调，使用包含该风格示例的数据集，便成为一个可行的选择。这是一个更复杂的过程，稍后将详细介绍，但它是实现深度风格调整的有力方法。

#### 用于隐含风格控制的结构化输出

要求LLM以JSON或XML等结构化格式生成输出，可以间接强制某种风格。例如，如果您要求一个带有特定键的JSON对象，LLM更有可能为这些值生成简洁、数据化的文本，这与更技术或分析性的风格相符。许多现代LLM提供“函数调用”或“工具使用”能力，这些能力在生成结构化数据方面表现出色。

```json
{
  "request": "从提供的文本中提取主要发现，并以项目符号列表形式列在'findings'键下，并在'summary'键下提供一句话摘要。",
  "context": "科学论文摘要...",
  "desired_output_structure": {
    "summary": "字符串",
    "findings": ["字符串"]
  }
}
```

指导模型遵守这种结构，可以同时引导内容和隐含的风格。

### 管理输出语气

语气指的是文本传达的情感色彩或态度。LLM可能需要对客户支持表现出同理心，对技术说明保持中立，或对营销文案具有说服力。

#### 用于语气调整的提示技术

与风格类似，提示中的直接指令是有效的：

- **明确指令**: “以鼓励和支持的语气回应。”
- **角色扮演**: “你是一位友好的助手。你的目标是让用户感到舒适和被理解。”
- **对抗性提示**: 为避免不想要的语气：“如实回应。不要使用过于热情或随意的语言。”

#### 情感引导

通过在提示中包含与所需情感相关的词语或短语来巧妙地引导语气（但这需要仔细设计和测试）：

- 为了积极的语气，您可以在示例或指令中包含“我们很乐意提供帮助”或“好问题”等短语，即使它们不是用户直接查询的一部分。

#### 示例：通过提示修改语气

设想一个RAG系统回答用户关于产品功能的问题。

**中性语气提示:**
“User: 功能X如何运作？
Context: [功能X的文档]
Assistant: 功能X的运作方式是...”

**同理心语气提示（适用于用户可能感到沮丧的支持场景）:**
“User: 功能X我弄不明白，太 confusing 了！
Context: [功能X的文档，常见故障排除步骤]
Assistant (as a patient support agent): 我理解功能X一开始可能有点复杂，但我在这里提供帮助。让我们一步步来。功能X的运作方式是...”

传达的核心信息可能相同，但由提示引导的框架和措辞会大幅改变语气。

### 确保事实一致性和根据

对于RAG系统，事实性不容妥协。生成的回答必须忠实于检索到的上下文 (context)。尽管有一个专门章节讨论幻觉 (hallucination)消除，但对事实性的初步控制始于您如何指导LLM使用所提供的信息。

#### 强有力根据提示

这些指令明确限制LLM只能依据所提供的文档。

- **专一性**: “*仅*根据所提供文档中的信息，回答以下问题。”
- **处理缺失信息**: “如果答案未在提供的上下文中找到，请说明‘所提供的信息不包含此问题的答案’。”这是为了防止臆测。

  ```
  Prompt:
  仅使用以下上下文中提供的信息来回答问题。
  请勿使用任何外部知识。如果答案不在上下文中，请说“我无法根据所提供的信息回答。”

  Context:
  <document id="doc1">
  空客A380飞机在全经济舱配置下，最大载客量为853人。
  其典型巡航速度为0.85马赫。
  </document>
  <document id="doc2">
  超音速运输机协和号的巡航速度为2.04马赫。
  它最多可搭载128名乘客。
  </document>

  Question: A380的翼展是多少？

  Expected LLM Output:
  我无法根据所提供的信息回答。
  ```

#### 上下文范围设定

在提示中清晰地划分上下文。使用类似XML的标签（`<context>`, `</context>`, `<document>`, `</document>`) 有助于LLM区分指令、用户查询以及其应使用的知识库。

#### 引用来源指令

提高根据性的一个有效方法是要求LLM引用其主张的来源，提及上下文中的特定文档ID或段落。

- “回答问题并引用支持您答案的文档ID。例如：天空是蓝色的（doc1）。”
  这不仅促使LLM仔细查看文本，也为用户提供了可验证性。

#### 自我纠正和思维链提示

对于复杂查询，您可以指导LLM“一步步思考”，或者生成答案后再对照上下文进行严谨审阅，以确保准确性，最后再呈现最终输出。
示例（简化版）：
“Question: [用户问题]
Context: [检索到的上下文]
Instruction: 首先，从上下文中找出相关句子。其次，*仅*根据这些句子组织答案。第三，验证您的答案未引入外部信息。只提供已验证的答案。”

#### 生成参数 (parameter)：Temperature 和 Top\_p

尽管提示工程 (prompt engineering)是主要的，但LLM生成参数也发挥作用：

- **Temperature**: 控制随机性。降低Temperature（例如，$T < 0.5$，通常$T \approx 0.1 - 0.3$）使输出更具确定性和集中性，通常通过减少创意性（和可能不正确）偏差的可能性来提高事实性。
- **Top\_p（核采样）**: 控制用于采样的token的累积概率质量。较低的`top_p`（例如，$p < 0.9$）也会将LLM限制在更可能、通常更真实可靠的token上。

这里有一个权衡：极低的Temperature可能导致重复或过于刻板的输出。实验对于为您的应用找到恰当的平衡非常重要。

控制策略总结:

| 控制方面 | 主要技术 | 辅助技术 | 主要考量 |
| --- | --- | --- | --- |
| **风格** | 系统提示、语境学习、指令提示 | 结构化输出（JSON/XML）、微调 (fine-tuning) | 与品牌“声音”保持一致、用户预期 |
| **语气** | 明确指令、角色扮演、情感引导 | 对抗性提示 | 语境适宜性（例如，支持 vs. 信息） |
| **事实性** | 强有力根据提示、上下文范围设定、引用来源 | 低 temperature/top\_p、自我纠正 | 严格遵守所提供上下文、避免编造 |

> 控制LLM输出风格、语气和事实性的策略。

有效地控制风格、语气和事实性是一个迭代过程。这需要仔细的提示设计、使用多样化输入进行全面测试，以及尝试不同技术的意愿。掌握这些控制方法，您可以显著提高RAG系统生成回答的质量和可靠性，使其适用于要求严格的生产应用。这为后续更高级的讨论，例如针对性幻觉消除和专业生成任务的微调，提供了准备。

## 参考资料

- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) — Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi (2023)
  Journal: arXiv preprint arXiv:2310.11511; DOI: [10.48550/arXiv.2310.11511](https://doi.org/10.48550/arXiv.2310.11511)
  介绍了一种框架，通过使LLM能够根据检索到的文档批判和改进其输出，从而提高RAG系统的事实准确性和生成质量。
- [The Hugging Face Course: Chapter 7 - Advanced Techniques for Text Generation](https://huggingface.co/docs/transformers/main/en/llm_tutorial#generation-strategies) — Hugging Face (2024)
  Publisher: Hugging Face
  一份权威的在线资源，详细介绍了文本生成的各种解码策略，包括温度和top_p等参数对控制输出随机性和连贯性的解释和实际影响。
- [Function calling and tool use](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGo1GbxYMn6v78i__Ugy0uxPiRGEqJM33LagER-mb-F8qdL8vQQBcqVjihozb4HZdCCCiOI-eubHopj-9VRgfpnPnMPVpRXx9PVlo4JMP7jb_Wxssck7Rhos4uMCsmrwlYDWUQymHmbL7JuKHkztjSLAlQ=) — OpenAI (2024)
  Publisher: OpenAI
  官方文档，提供了使用OpenAI模型进行函数调用以实现结构化输出和集成外部工具的实用指南，这是一种隐式风格和内容控制技术。

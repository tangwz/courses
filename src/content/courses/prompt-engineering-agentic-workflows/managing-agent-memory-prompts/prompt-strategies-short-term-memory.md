---
course: "prompt-engineering-agentic-workflows"
chapter: "managing-agent-memory-prompts"
lesson: "prompt-strategies-short-term-memory"
sourceId: 6742
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-agentic-workflows/chapter-5-managing-agent-memory-prompts/prompt-strategies-short-term-memory"
title: "提示策略：短期记忆与上下文窗口管理"
description: "学习如何通过提示设计管理代理上下文窗口中的信息。"
order: 2
plots: []
sourceHash: "d2cd02ff6c3f48874818bc30c8397e9ff10015e78dff5686cb0e4e2797f99b42"
sourceCorrections: []
---

AI代理的即时回想能力，即它记住刚刚发生的事情或其当前具体指令的能力，很大程度上依赖于大语言模型（LLM）的上下文 (context)窗口。这个窗口充当代理的短期或工作记忆，存放LLM可以直接访问的信息，以生成其下一个回应或行动。有效的提示工程 (prompt engineering)对于管理这种有限的资源不可或缺，它确保代理在多步操作中保持连贯性并专注于相关细节。

### 上下文 (context)窗口：代理的工作记忆

可以把LLM的上下文窗口想象成计算机的内存（RAM）：它是快速、可直接访问的内存，但容量有限。在单次交互中提供给LLM的每一条信息，包括定义代理角色的系统指令、用户的当前查询、最近对话轮次的记录、可用工具的描述，以及任何中间思考或“草稿板”笔记，都必须适配到此窗口内。

上下文窗口的大小以token（标记 (token)）衡量，大致可以认为是词或词的一部分。模型有不同的上下文窗口大小，例如4,096个token、8,192个token、32,768个token，甚至更大。尽管更大的窗口提供更多空间，但它们总归是有限的。如果信息流超过此限制，旧信息通常会被挤出，导致代理“遗忘”交互的早期部分。这会中断任务的连贯性，并可能导致错误或不相关的回应。

例如，如果代理的上下文窗口为4096个token：

- 系统指令和角色定义可能占用200个token。
- 可用工具的描述可能占用500个token。
- 这样就剩下 $4096 - 200 - 500 = 3396$ 个token，用于实际的对话历史、用户查询以及当前轮次的动态工作笔记。

通过提示设计有效管理此预算是构建能干代理的一项重要技能。

### 管理上下文 (context)窗口的主要提示策略

为了充分利用代理的短期记忆，您可以采用多种提示工程 (prompt engineering)策略。这些方法有助于确保最相关的信息保持在LLM的注意力范围内。

#### 1. 信息策略性放置

LLM并非总是对提示的所有部分给予同等权重 (weight)。通常，提示开头（首因效应）或结尾（近因效应）的信息对输出有更强的影响。

- **首因效应**：将基础信息，例如代理的核心角色、总体目标或重要限制，放置在提示结构的开头。如果底层LLM API支持，这通常是“系统提示”的一部分，或者是用户提示的初始固定部分。

  ```
  System: 你是一位一丝不苟的研究助理。你的目标是收集关于可再生能源的信息。总是引用你的来源。不要提供观点，只提供事实数据。
  ```
- **近因效应**：最直接的任务、用户的最新问题或当前步骤的具体指令应放置在输入的末尾。这确保这是模型在生成回应前最后处理的信息。

  ```
  User: ... [之前的对话历史已总结] ...
  User: 现在，根据目前为止的发现，找出城市环境中太阳能电池板采用面临的三个主要挑战。
  ```

#### 2. 通过历史管理保持对话流畅性

为了使代理进行连贯的对话或执行一系列相关任务，它需要记住之前说过或做过什么。

- **滚动历史窗口**：在提示中包含最近 $N$ 轮对话的记录。这提供即时历史上下文。 $N$ 的值取决于token预算和任务的性质。
- **总结指令**：随着对话进行，完整历史会超出上下文窗口。您可以指导代理（或使用单独的LLM调用）总结对话的早期部分。然后将此压缩摘要包含在后续提示中。

  ```xml
  <system_prompt>
  你是一个乐于助人的助手。
  </system_prompt>

  <user_prompt>
  <conversation_history_summary>
  用户正在计划春季巴黎7日游，2位成人。预算适中。兴趣包括博物馆和历史遗迹。
  </conversation_history_summary>

  <recent_exchanges>
  User: 晚间娱乐有什么选择？
  Agent: 巴黎的晚间娱乐有古典音乐会、戏剧表演和塞纳河游船。这些听起来有吸引力吗？
  User: 塞纳河游船听起来不错。你能找到包含晚餐的选项吗？
  </recent_exchanges>

  Current question: 寻找塞纳河上两人晚餐游船的选项。
  </user_prompt>
  ```

  在此示例中，`conversation_history_summary` 是早期对话的浓缩版本，而 `recent_exchanges` 则保留了最近几次交互的原文。

#### 3. “草稿板”或“工作笔记”技术

您可以指导代理直接在其提示中维护一个“草稿板”或“工作笔记”部分。这个部分作为一个可变空间，代理可以在其中记下中间思考、计算、观察结果或精细化计划。

代理被提示读取并更新此草稿板，作为其推理 (inference)过程的一部分。在代理生成其回应（其中包含更新后的草稿板）后，您的控制应用程序代码会提取此草稿板并将其包含在下一轮的提示中。

**优点**：

- 使代理的推理过程更透明。
- 帮助代理跟踪复杂的当前状态信息或部分结果。
- 减少遗忘中间步骤的可能性。

**示例提示结构**：

```
System: 你是一个负责解决多步问题的AI助手。使用 <scratchpad> 进行逐步思考并记录重要信息。

User:
<problem_description>
[在此处填写详细问题说明]
</problem_description>

<scratchpad>
[之前的草稿板内容，或首次轮次时的“在此处逐步思考...”]
</scratchpad>

Based on the problem and your scratchpad, what is the next step or your final answer? Update your scratchpad.
```

The agent的输出将包含其推理以及 `<scratchpad>` 的更新内容，这些内容随后会被反馈到下一个提示中。

#### 4. 结构化数据以增强清晰度

使用清晰的分隔符，例如类似XML的标签（如 `<history>`、`<user_query>`、`<available_tools>`）或Markdown标题，有助于LLM区分复杂提示中的各种信息类型。这可以提高模型准确解析提示的能力，并专注于与其当前子任务最相关的部分。

```
### 指令
你是一个乐于助人的助手。

### 可用工具
- search_web(query: string)
- get_weather(city: string)

### 对话历史
User: 伦敦天气怎么样？
Agent: 伦敦目前天气15°C，多云。
User: 谢谢！现在你能查找那里的热门旅游景点吗？

### 当前任务
根据对话，使用适当的工具查找伦敦的热门旅游景点。
```

这种结构化方法有助于LLM理解其输入的不同组成部分及其作用。

#### 5. 显式上下文注入

虽然上述策略管理的是交互中已经引入的信息，但有时代理需要它尚未遇到的特定信息，或者这些信息是早期已被总结的交互的一部分。您的应用程序逻辑可以主动将相关的上下文片段注入到提示中。例如，如果代理提到某个特定实体，系统可以从知识库中检索关于该实体的简短定义或重要事实，并将其添加到当前轮次的提示中，从而丰富代理的短期记忆。

### 上下文 (context)窗口组件的可视化

以下图表说明了各种信息如何构成提示，进而填充LLM在给定处理轮次中的上下文窗口。

> 该图显示，包含系统指令、用户请求、历史记录、工具数据和工作笔记的提示，填充LLM的上下文窗口。该窗口随后告知LLM的处理单元生成输出。

### 主动处理上下文 (context)溢出

这些提示策略的主要目标是有效利用有限的上下文窗口，并*防止*重要信息丢失。当信息量（系统提示、历史、工具描述、当前查询）超出LLM的token限制时，模型将不可避免地“遗忘”其中一些信息，通常在简单的截断情况下是旧的信息。这种丢失可能导致代理失去对目标的关注、重复之前的行动，或未能使用相关的历史信息。

通过主动管理每轮进入提示的内容，进行总结、选择和结构化，您可以减少此类上下文溢出的可能性。这些方法是您在多轮交互中保持代理连贯且有状态行为的第一道防线。

虽然这些方法对于管理短期记忆很有效，但它们并非解决长期知识保留或从数据集中召回信息的完整方案。对于这些需求，涉及外部知识库和检索机制的策略（本章后面会讨论）变得必要。然而，管理得当的上下文窗口是所有其他记忆技术得以发挥作用的前提。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: arXiv preprint arXiv:1706.03762; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍Transformer架构的基础论文，该架构构成了当前大型语言模型及其上下文窗口机制的基础。
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou (2022)
  Journal: arXiv; Publisher: arXiv; DOI: [10.48550/arXiv.2201.11903](https://doi.org/10.48550/arXiv.2201.11903)
  介绍了思维链提示，这是一种与便笺簿技术直接相关的方法，用于增强模型推理和在上下文窗口内跟踪中间思考。
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang (2023)
  Journal: Transactions of the Association for Computational Linguistics (TACL); DOI: [10.48550/arXiv.2307.03172](https://doi.org/10.48550/arXiv.2307.03172)
  研究大型语言模型如何处理长上下文中的信息，详细阐述了影响提示中信息策略性放置的首因效应和近因效应等现象。
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2023)
  Journal: arXiv preprint arXiv:2210.03629; DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍了ReAct，一个通过结构化便笺簿方法结合推理和行动的智能体框架，展示了多步骤智能体任务中有效的上下文管理。
- [Prompt Engineering Guide](https://docs.anthropic.com/claude/docs/prompt-engineering) — Anthropic (2024)
  Publisher: Anthropic
  官方指南，提供了大型语言模型提示工程的实用策略和最佳实践，包括关于上下文管理、提示结构和令牌感知的建议。

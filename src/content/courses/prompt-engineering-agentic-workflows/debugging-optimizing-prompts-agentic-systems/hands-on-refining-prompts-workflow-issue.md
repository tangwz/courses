---
course: "prompt-engineering-agentic-workflows"
chapter: "debugging-optimizing-prompts-agentic-systems"
lesson: "hands-on-refining-prompts-workflow-issue"
sourceId: 6756
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-agentic-workflows/chapter-6-debugging-optimizing-prompts-agentic-systems/hands-on-refining-prompts-workflow-issue"
title: "实践：优化提示词以解决智能体工作流程问题"
description: "参与实践调试环节，修改提示词以修复出现故障的智能体工作流程。"
order: 8
plots: []
sourceHash: "259e0178d7a95545acc5d93f3d412d40ca33dfce2b7ab1394f3717a3ce20c7b8"
sourceCorrections: []
---

现在，让我们将理论付诸实践。您已经了解了智能体提示词 (prompt)的常见问题、系统化迭代和分析方法。本次实践练习旨在引导您完成一个简化智能体工作流程的调试和优化过程，该流程目前表现不佳。我们将发现问题，提出提示词修改建议，并讨论这些修改如何带来更好结果。

假设我们有一个智能体 `ReportCraftAI`，旨在帮助生成特定主题的近期新闻文章摘要。其预期工作流程如下：

1. 接收用户提供的主题。
2. 使用网页搜索工具查找过去一周内发布的三篇相关新闻文章。
3. 对每篇文章，提取主要标题和简短摘要。
4. 将这些内容整理成一份报告。

然而，用户反映 `ReportCraftAI` 通常只返回一篇文章，有时文章不相关，并且摘要提取质量不佳。

我们的任务是通过检查其提示词来诊断问题，并优化它们以提升其性能。我们假设该智能体可使用 `search_web(query: str, time_period: str, num_results: int)` 工具和 `extract_summary(article_text: str)` 工具。

### 初始问题工作流程和提示词 (prompt)

假设当 `ReportCraftAI` 被要求处理某个主题（例如“可再生能源的进展”）时，其核心系统提示词包含以下内容：

**初始系统提示词片段：**

```
You are ReportCraftAI, a helpful assistant. -> 你是 ReportCraftAI，一个有用的助手。
Your goal is to create a report on recent news. -> 你的目标是撰写一份关于近期新闻的报告。
Current Task: User wants a report on "advancements in renewable energy". -> 当前任务：用户想要一份关于“可再生能源进展”的报告。

Available Tools: -> 可用工具：
1. search_web(query: str, time_period: str, num_results: int): Searches the web for news. time_period can be "past_week", "past_month". -> 1. search_web(query: str, time_period: str, num_results: int): 搜索网络新闻。time_period 可以是 "past_week"（过去一周）或 "past_month"（过去一月）。
2. extract_summary(article_text: str): Extracts a concise summary from an article. -> 2. extract_summary(article_text: str): 从文章中提取简洁摘要。

Plan: -> 计划：
1. Search for news on the topic. -> 1. 搜索该主题的新闻。
2. Extract information. -> 2. 提取信息。
3. Compile the report. -> 3. 整理报告。

Begin. -> 开始。
```

智能体随后会以此为基础为自身生成后续提示或决定如何使用工具。步骤1（搜索）中可能的第一个内部提示或思考过程可能受到以下内容影响：

**智能体初始搜索“思考”：**
“我需要查找关于‘可再生能源进展’的新闻。我应该使用 `search_web` 工具。”

**由此产生的工具调用提示（隐式或显式）：**
`search_web(query="advancements in renewable energy news", time_period="recent", num_results=1)`

**观察到的问题：**

- **问题1（文章数量）：** 如果没有明确指示，智能体通常默认 `num_results=1` 或其他较低的数字；或者如果工具期望特定值，“recent”这个时间段可能过于模糊。
- **问题2（相关性）：** 查询“advancements in renewable energy news”可能过于宽泛。
- **问题3（摘要提取）：** 如果 `extract_summary` 的提示词仅仅是“总结这篇文章”，则摘要质量可能不稳定。

### 分析智能体行动序列（模拟）

如果我们有日志，可能会看到类似这样的内容（简化）：

```
Thought: User wants a report on "advancements in renewable energy". I need to find articles. -> 思考：用户想要一份关于“可再生能源进展”的报告。我需要查找文章。
Action: search_web(query="advancements in renewable energy news", time_period="recent", num_results=1) -> 动作：search_web(query="advancements in renewable energy news", time_period="recent", num_results=1)
Observation: [Article1: "Solar Power Breakthrough...", Content: "Detailed text..."] -> 观察：[文章1：“太阳能突破...”，内容：“详细文本...”]
Thought: I have one article. Now I need to summarize it. -> 思考：我有一篇文章。现在我需要总结它。
Action: extract_summary(article_text="Detailed text...") -> 动作：extract_summary(article_text="Detailed text...")
Observation: "Solar power is advancing." -> 观察：“太阳能正在发展。”
Thought: I have the summary. I will compile the report. -> 思考：我有了摘要。我将整理报告。
Output: Report: Article 1: Solar Power Breakthrough. Summary: Solar power is advancing. -> 输出：报告：文章1：太阳能突破。摘要：太阳能正在发展。
```

这个追踪清楚地表明智能体只获取了一篇文章并生成了一个非常简短的摘要。

### 步骤1：优化搜索指引

初始系统提示词 (prompt)过于笼统，没有为搜索任务提供足够的约束。我们需要更具体。

**修订后的系统提示词片段（侧重任务定义）：**

```
You are ReportCraftAI, a helpful assistant. -> 你是 ReportCraftAI，一个有用的助手。
Your goal is to create a report on recent news. -> 你的目标是撰写一份关于近期新闻的报告。
Current Task: User wants a report on "advancements in renewable energy". -> 当前任务：用户想要一份关于“可再生能源进展”的报告。
Specific Instructions: -> 具体指示：
- Find exactly three (3) relevant news articles. -> - 查找恰好三 (3) 篇相关新闻文章。
- Articles must be published within the "past_week". -> - 文章必须在“过去一周”内发布。
- Focus on significant developments or announcements. -> - 侧重于重要进展或公告。

Available Tools: -> 可用工具：
1. search_web(query: str, time_period: str, num_results: int): Searches the web for news. time_period must be "past_week" or "past_month". -> 1. search_web(query: str, time_period: str, num_results: int): 搜索网络新闻。time_period 必须是“past_week”（过去一周）或“past_month”（过去一月）。
2. extract_summary(article_text: str, desired_length_words: int): Extracts a concise summary from an article to a desired word length. -> 2. extract_summary(article_text: str, desired_length_words: int): 从文章中提取简洁摘要，达到期望的字数长度。

Plan: -> 计划：
1. Formulate a precise search query based on the topic and instructions. -> 1. 根据主题和指示，制定精确的搜索查询。
2. Use search_web to find 3 articles from the past_week. -> 2. 使用 search_web 查找过去一周的 3 篇文章。
3. For each article, use extract_summary to get a 50-word summary. -> 3. 对每篇文章，使用 extract_summary 获取一份 50 字的摘要。
4. Compile the headlines and summaries into a report. -> 4. 将标题和摘要整理成一份报告。

Begin. -> 开始。
```

**修改理由：**

- **指示的明确性：** 我们明确说明了“查找恰好三 (3) 篇相关新闻文章”和“文章必须在‘过去一周’内发布”。这直接对应 `search_web` 工具的 `num_results` 和 `time_period` 参数 (parameter)。
- **查询制定指引：** “侧重于重要进展或公告”促使智能体创建比仅主题加“新闻”更好的搜索查询。
- **工具描述更新：** 我们通过添加 `desired_length_words` 参数，使 `extract_summary` 更可控。

经过这些修改，智能体内部的搜索“思考”过程变得更受约束，从而实现更好的工具调用：

**智能体改进后的搜索“思考”：**
“我需要查找 3 篇关于‘可再生能源重大进展’的近期文章（过去一周内）。我将使用 `search_web`。”

**由此产生的工具调用提示（隐式或显式）：**
`search_web(query="significant advancements in renewable energy", time_period="past_week", num_results=3)`

这是一个显著改进，直接解决了问题1并有助于解决问题2。

### 步骤2：改进摘要提取

此前，`extract_summary` 工具可能只收到了极少指令。修订后的系统提示词 (prompt)现在指导智能体使用新的 `desired_length_words` 参数 (parameter)。

**最初对 `extract_summary` 的隐式提示词（源自“提取信息”）：**
“总结这篇文章内容：[文章内容]”

**修订后的 `extract_summary` 提示词（源自“使用 extract\_summary 获取一份 50 字的摘要”）：**
“从以下文本中提取一份约 50 字的摘要，侧重于主要发现：[文章内容]”
或者，如果智能体根据计划直接调用工具：
`extract_summary(article_text="[article content]", desired_length_words=50)`

**修改理由：**

- **明确的长度控制：** 要求“一份 50 字的摘要”为工具背后的 LLM 提供了一个明确的目标。
- **侧重指示：** 添加“侧重于主要发现”（如果我们对摘要步骤本身进行提示）有助于 LLM 优先处理重要信息。

这直接解决了问题3，从而生成更一致且有用的摘要。

### 步骤3：迭代和测试变体

假设在这些修改之后，智能体现在能稳定地获取三篇文章，摘要也更佳，但有时返回的文章中有一篇是观点文章而非新闻报道。最初的“侧重于重要进展或公告”指示是一个好的开始，但我们可以进一步优化查询生成的提示词 (prompt)。

我们可以在系统提示词的搜索指示中尝试一种变体：

**系统提示词变体（搜索指示）：**
“...侧重于关于重要进展或公告的事实性新闻报道，避免观点文章或博客文章。”

**比较变体：**
为了测试这一点，您将使用之前的提示词和这个新变体，在几个不同主题上运行智能体。然后您将比较输出结果：

- **指标1：相关新闻文章数量（目标：3）。**
- **指标2：返回的观点文章/博客数量（目标：0）。**
- **指标3：摘要的主观质量。**

此时，提示词A/B测试等方法就变得有用。如果这种变体在不损害其他方面的情况下持续减少不相关的文章，那么它是一个很好的采用选择。

一个简单的图表可以说明智能体流程的转变：

> 此图表展示了初始效果较差的工作流程与通过改进智能体指导性提示词实现的优化后工作流程之间的对比。

### 日志记录以持续改进

在整个调试过程中，详细的日志记录将非常宝贵。设想一下日志记录了：

- 每次工具调用所使用的精确提示词 (prompt)。
- 每个工具的原始输出。
- 智能体的“内部独白”或推理 (inference)步骤（如果您的架构支持此功能，例如在 ReAct 中）。

例如，如果 `search_web` 工具返回错误消息或意外数据，日志将有助于准确定位问题是工具本身还是智能体如何提示它。如果 `extract_summary` 尽管要求 50 字却持续生成过短的摘要，日志将有助于调查工具是否遵守了参数 (parameter)或输入文本是否过短。

通过审查这些日志，您可以系统地识别提示链中哪些部分薄弱并需要进一步优化。使用版本控制（例如，使用 Git 管理提示词文件或专用提示词管理系统）来组织您的提示词，可以跟踪更改，在新提示词表现更差时回滚，并管理不同版本以进行 A/B 测试。

本次实践练习模拟了开发智能体工作流程中的常见情况。核心要点不仅仅是编写提示词，而是将其视为您系统的核心部分，需要进行测试、分析和迭代优化。通过应用本章的原则，您可以显著提升 AI 智能体的可靠性和性能。

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2022)
  Journal: arXiv preprint arXiv:2210.03629; DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍了ReAct框架，通过明确的思维步骤和工具交互，结合推理和行动来提升智能体性能。

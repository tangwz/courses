---
course: "prompt-engineering-agentic-workflows"
chapter: "advanced-prompting-agent-control"
lesson: "prompting-self-correction-error-handling"
sourceId: 6723
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-agentic-workflows/chapter-2-advanced-prompting-agent-control/prompting-self-correction-error-handling"
title: "提示代理进行自我修正和错误处理"
description: "编写提示，使代理能够在运行过程中识别并从错误中恢复。"
order: 7
plots: []
sourceHash: "64389225fa3e79a865eaee6d0ad17bc611a0604f4e1a82e749d2812ab6962a0c"
sourceCorrections: []
---

随着AI代理承担更复杂的任务，它们自主运行并从错误中恢复的能力变得越来越重要。构建真正可靠的代理，需要它们能够妥善处理意外情况和错误。这包括设计提示，这些提示不仅能指导代理成功执行和思考过程，还能为其提供自我修正和错误处理的机制。通过将这些能力融入提示，对人工持续监督的需求得以减少，并提升代理工作流程的整体韧性。

核心理念是超越简单地指示代理 *做什么*，还要指导它 *在事情出错时该怎么做*。这通常涉及提示代理反思其行为、输出或已处理的信息，然后在识别出差异或失败时采取纠正措施。

## 提示代理进行自我修正的技巧

有几种提示策略可以促使代理识别并处理错误。这些技巧通常涉及为代理提供自我评估的框架，或针对常见失败模式提供明确的指示。

### 明确的错误检测和处理指令

一种直接的方法是在提示中为代理提供条件逻辑。这类似于编写代理可以遵循的 `if-then-else` 语句。

例如，当代理需要与外部工具或API交互时，您可以指定它应如何响应不同的错误代码或回复：

```text
System: 你是一个使用 `getStockPrice` 工具检索股票价格的助手。
可用工具: `getStockPrice(ticker_symbol)`

指令:
1. 当被问及股票价格时，使用 `getStockPrice` 工具并提供股票代码。
2. 如果 `getStockPrice` 返回“Symbol Not Found”错误，请告知用户股票代码无效，并请求一个正确的代码。
3. 如果 `getStockPrice` 返回“Service Unavailable”错误，等待5秒并再次调用该工具。
4. 如果重试也失败，或发生任何其他错误，请告知用户当前无法检索价格，并说明遇到的错误。

User: GOGL 的价格是多少？
```

在这种情况下，代理被明确指导如何解释工具输出以及应采取哪些备用操作。这降低了代理卡住或提供无用的通用错误消息的可能性。

### 用于自我审视的反思性提示

您可以提示代理在最终确定之前检查其生成的内容或行动计划。这鼓励了一种自我审视的形式，即代理根据既定标准评估其工作。

考虑一个负责总结一份文档的代理：

```text
System: 你是一个总结文章的AI助手。生成总结后，请根据以下清单进行检查:
- 总结是否捕获了文章的主要论点？
- 总结的长度是否约为原始文章的10%？
- 总结是否不含原文中没有的个人观点或解读？
- 是否有任何重复的短语或句子？

如果你发现问题，请提供一份修改后的总结。否则，请确认总结符合所有标准。

User: 这是文章：[长文章文本...] 请总结它。
```

代理首先生成总结，然后，在提示的指导下，执行自我评估。这一反思步骤可以发现遗漏、风格问题或不符合要求的地方。

### 带修改的重试

如果失败的根本原因仍然存在，简单地重试一个操作可能无效。一种更先进的方法是提示代理在失败时修改其方法。

例如，如果代理的搜索查询没有返回结果：

```text
System: 你是一名研究助手。当搜索信息时，如果你的初始查询没有返回相关结果，请遵循以下步骤:
1. 分析你的初始查询。它是否过于具体或过于狭窄？
2. 扩大你的搜索词或尝试同义词。例如，如果“nanoparticle drug delivery systems”（纳米粒子药物递送系统）失败了，请尝试“nanomedicine drug transport”（纳米药物运输）或“nanocarriers for therapeutics”（治疗用纳米载体）。
3. 使用修改后的查询再次执行搜索。
4. 如果在2次修改尝试后仍不成功，请报告未能找到信息。

User: 查找关于“量子光子纠缠稳定器”的研究论文。
```

这指导代理不要轻易放弃，而是根据之前尝试的结果，通过迭代改进其方法来解决问题。

### 使用工具进行验证

可以提示代理使用其他工具（甚至是一个具有特定“检查者”角色的独立LLM调用）来验证其工作或中间结果。例如，可以提示编写代码的代理在提交代码之前使用代码检查工具或代码执行环境来检查错误。

```text
System: 你是一名Python编程助手。
1. 编写Python代码来解决用户的请求。
2. 在展示代码之前，你必须通过在脑海中运行它来验证，或者如果可用，使用 `python_code_checker` 工具。
3. 如果你检测到错误，请纠正它们并重新验证。
4. 仅呈现已验证和纠正后的代码。解释你所做的任何修正。

User: 编写一个Python函数，该函数接受一个数字列表并返回偶数的总和。
```

### 结构化错误报告

当代理无法自行解决错误时，以清晰、结构化的方式报告失败非常重要。这使更广范围的系统（或人工监督者）能够了解出了什么问题并可能进行干预。

您可以在提示中指定错误报告的格式：

```text
System: ...
如果在处理请求时发生不可恢复的错误，请以以下JSON格式提供您的响应:
{
  "status": "error",
  "error_type": "<错误类型>",
  "error_message": "<关于错误的详细信息>",
  "last_action_attempted": "<你正在执行的操作描述>"
}
请勿将此格式用于成功的响应。

User: 处理附加的财务报告。
```

这种结构化报告有助于日志记录、调试，并可能自动化更高层次的恢复过程。

## 自我修正循环的可视化

代理执行动作、检查工作并在必要时进行修正的过程可以被可视化为一个循环。提示工程 (prompt engineering)师的角色是定义此循环中每个步骤的条件和指导原则。

> 一个融入了提示式自我修正的代理操作周期。代理执行一个动作，根据提示指导审视其输出，如果检测到问题，在继续之前尝试修正措施。

## 提示代理进行自我修正的注意事项

实施自我修正能力需要仔细的提示设计：

- **特殊性与通用性**：高度具体的错误处理指令对已知常见错误有效，但可能使提示变得冗长。更通用的自我反思指令可以覆盖更广范围的未知错误，但在解决特定问题时可能效果较差。
- **过度修正的风险**：代理可能变得过于谨慎或误解指令，导致它“修正”并非真正错误的事物，或者陷入尝试修正不存在问题的循环。
- **错误诊断的复杂性**：LLM有时可能会凭空捏造对错误或其原因的理解。提示需要引导它们进行准确诊断或安全回退。
- **迭代完善**：编写有效的自我修正提示是一个迭代过程。您可能需要观察代理行为，识别失败模式，并随着时间的推移完善您的提示以改进错误处理。这是提示工程 (prompt engineering)中的一个常见主题，在这里尤其重要。

通过深思熟虑地将自我修正和错误处理策略融入您的提示中，您可以构建更自主、可靠和智能的代理。这些技巧是创建能够以更强的韧性处理任务的系统的根本。管理和从错误中恢复的能力是向更高级代理行为迈出的重要一步，您将有机会在实际练习中实现。

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://doi.org/10.48550/arXiv.2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2022)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍了ReAct范式，一种结合推理和行动的LLM代理基础方法，包括处理工具交互及其潜在结果。该方法对于代理工作流中的优雅错误恢复很有价值。
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) — Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao (2023)
  Journal: arXiv preprint arXiv:2303.11366; DOI: [10.48550/arXiv.2303.11366](https://doi.org/10.48550/arXiv.2303.11366)
  提出了一种LLM代理通过反思过去行动序列并生成自我反思提示来指导未来行动的口头强化学习框架。这与引导自我批评和失败后迭代修改直接相关。
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2209.07858) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Anna Chen, Andy Jones, Anna Goldieberger, Aziza Mirrashed, Cameron McKinnon, Carol Chen, Catherine Olsson, Chris Conly, David Drain, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jackson Kernion, Jasmine Sittler, Jennifer Glowinski, Jeremy Scheurer, Jessica Kerr, Josh Jacobson, Kristen Lee, Liane Lovitt, Lisa Wang, Michael Sellitto, Mo Mukherjee, Nicholas Joseph, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Shauna Gordon-McKeon, Simon Lefevre, Tristan Hume, Zac Hatfield-Dodds, Danny Hernandez, Daniela Amodei, Dario Amodei, Jack Clark, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2209.07858; DOI: [10.48550/arXiv.2209.07858](https://doi.org/10.48550/arXiv.2209.07858)
  虽然主要关注安全性，但本文介绍了一种AI模型根据一组原则批评和修订自身输出的方法，展示了一种稳健的AI驱动的自我评估和纠正形式。

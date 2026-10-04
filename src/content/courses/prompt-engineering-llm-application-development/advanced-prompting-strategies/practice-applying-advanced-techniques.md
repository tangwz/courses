---
course: "prompt-engineering-llm-application-development"
chapter: "advanced-prompting-strategies"
lesson: "practice-applying-advanced-techniques"
sourceId: 2993
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-2-advanced-prompting-strategies/practice-applying-advanced-techniques"
title: "练习：应用高级技巧"
description: "动手实践少量样本、角色和思维链提示的实现。"
order: 8
plots: []
sourceHash: "43beab3cb72e80063d2fc51b3c2c072862857c1d8c0c80122c9b2dd9cbd0d122"
sourceCorrections: []
---

动手练习提供了运用少量样本提示、角色提示、结构化输出请求和思维链的机会。这些技术能更有效地引导大型语言模型（LLM）的行为。

对于这些练习，您需要访问一个LLM，可以是通过网页界面（如在线演示平台），也可以是通过API以编程方式访问。这里的重点是构建*提示本身*；具体的API调用细节对于本次练习不那么重要，但在第1章已有所讲解，并将在第4章中进一步说明。

### 练习1：使用少量样本提示改进分类

零样本提示依赖于LLM的通用知识。虽然功能强大，但对于某些特定分类任务，它有时可能存在歧义。少量样本提示在提示中提供示例以阐明您的意图。

**任务：** 将客户反馈分类为“积极”、“消极”或“中立”。

**1. 零样本尝试：**

考虑这个简单的提示：

```text
分类以下客户反馈的情绪：

反馈: "The user interface is quite confusing."
情感:
```

LLM可能会正确输出`Negative`。然而，对于更模糊的反馈，它可能会遇到困难。

**2. 少量样本提示构建：**

现在，让我们提供示例（样本）来引导模型：

```text
将客户反馈的情绪分类为积极、消极或中立。

反馈: "I love the new features, they work great!"
情感: Positive

反馈: "The documentation is okay, but could be clearer."
情感: Neutral

反馈: "The app crashes every time I try to save."
情感: Negative

反馈: "The user interface is quite confusing."
情感:
```

**轮到您了：**

- 将这个少量样本提示发送给LLM。观察输出。它是否一致地是`Negative`？
- 尝试使用相同的少量样本提示分类一段新的反馈，例如：`"Response times are acceptable."`
- 尝试改变示例或样本数量（$k$）。这如何影响对“它有效。”这类模糊反馈的分类？

少量样本学习通过提供所需输入-输出模式的具体示例，显著提升了特定、定制任务的可靠性。

### 练习2：结合角色提示和结构化输出

有时您需要LLM扮演特定角色，*并*以结构化的方式（如JSON）格式化其输出，以便更方便地进行程序化使用。

**任务：** 生成一个历史事件的简短、事实性总结，扮演历史学家，并将其输出为JSON对象。

**1. 基本提示（潜在问题）：**

```text
Summarize the moon landing.
```

这可能会生成一个不错的总结，但格式不可预测（纯文本、段落、项目符号列表），并且语气可能不同。

**2. 角色与结构提示：**

让我们指示LLM它应该扮演*谁*以及*如何*响应。

```text
扮演一个中立的历史学家。提供阿波罗11号登月的简明总结。
将输出格式化为JSON对象，包含以下键："event_name", "date", "key_figures", "brief_summary"。

示例输出格式：
{
  "event_name": "...",
  "date": "...",
  "key_figures": ["...", "..."],
  "brief_summary": "..."
}

请提供阿波罗11号登月的总结：
```

**轮到您了：**

- 将这个详细的提示发送给LLM。
- LLM是否采用了历史学家角色（中立、事实性的语气）？
- 它是否生成了与所请求结构匹配的有效JSON？
- 尝试使用相同的提示结构请求其他事件的总结。格式如何？

这种组合对于将LLM输出集成到期望可预测数据结构的应用程序中非常有用。

### 练习3：使用思维链（CoT）鼓励推理 (inference)

对于需要多步推理的问题，仅仅要求答案可能导致LLM猜测或进行逻辑跳跃。思维链提示鼓励模型展示其思考过程，通常能提升准确性。

**任务：** 解决一个简单的多步应用题。

**问题：** 一家杂货店最初有50个苹果。他们上午卖出15个苹果，下午又收到30个苹果。一天结束时他们有多少个苹果？

**1. 直接提示：**

```text
A grocery store starts with 50 apples. They sell 15 apples in the morning and receive a shipment of 30 more apples in the afternoon. How many apples do they have at the end of the day?
```

对于这类简单问题，许多LLM都能正确解答。然而，对于更复杂的逻辑，它们可能会出错。

**2. 思维链提示：**

让我们明确要求LLM逐步进行推理。

```text
A grocery store starts with 50 apples. They sell 15 apples in the morning and receive a shipment of 30 more apples in the afternoon. How many apples do they have at the end of the day?

让我们一步一步思考：
1. 从苹果的初始数量开始。
2. 计算卖出的苹果数量。
3. 计算收到的苹果数量。
4. 计算最终数量。
```

此外，您可以使用少量样本方法来演示CoT：

```text
问: 罗杰有5个网球。他又买了2罐网球。每罐有3个网球。他现在有多少个网球？
答: 让我们一步一步思考。罗杰最初有5个球。他买了2罐，每罐有3个球，所以他买了2 * 3 = 6个球。总共，他现在有5 + 6 = 11个球。答案是11。

问: 一家杂货店最初有50个苹果。他们上午卖出15个苹果，下午又收到30个苹果。一天结束时他们有多少个苹果？
答: 让我们一步一步思考。
```

**轮到您了：**

- 将直接提示和CoT提示（任一版本）发送给您的LLM。
- 比较输出。CoT提示是否生成了清晰的推理路径？
- 两种情况下的最终答案都正确吗？（对于这个简单问题，很可能正确）。
- 尝试一个稍微复杂一点的问题。在直接提示可能失败的情况下，CoT是否有助于保持准确性？示例：“一列火车在2小时内行驶了120公里。然后它以50公里/小时的速度又行驶了150公里。整个行程总共花费了多少时间？”

CoT对于算术、常识和符号推理任务特别有效。

### 实验与下一步

这些练习体现了高级提示技巧的实际运用。尽管进一步尝试：

- 结合技巧：将角色提示*与*少量样本示例和CoT指令一起使用。
- 完善指令：让您的指令更清晰或更具体。
- 调整参数 (parameter)：除了这些提示结构，调整温度或其他生成参数（如第1章所述）。

掌握这些策略为您提供了强大的工具包，用于引导LLM行为，从而构建更复杂、更可靠的应用程序。下一章将考察设计、测试和完善提示的系统化过程。

## 参考资料

- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Anna Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei (2020)
  Journal: Advances in Neural Information Processing Systems 33; DOI: [10.48550/arXiv.2005.14165](https://doi.org/10.48550/arXiv.2005.14165)
  这篇基础论文介绍了少样本学习，展示了大型语言模型如何在提示中仅提供少量示例即可执行新任务，而无需显式微调。

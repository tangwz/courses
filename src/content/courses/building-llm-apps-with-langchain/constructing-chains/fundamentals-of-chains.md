---
course: "building-llm-apps-with-langchain"
chapter: "constructing-chains"
lesson: "fundamentals-of-chains"
sourceId: 7417
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-3-constructing-chains/fundamentals-of-chains"
title: "链的基本原理"
description: "介绍LangChain中链的原理，以及它们如何实现顺序和逻辑的应用流程。"
order: 1
plots: []
sourceHash: "d23cd9eb4c156300603cefc71208adb5be56fe02b1bc1164cdce568cf0881b4c"
sourceCorrections: []
---

尽管对语言模型的独立调用很有用，但大多数实际应用都涉及一系列操作。用户的查询可能首先需要被优化，然后传递给模型，最后，模型的输出可能需要被解析并用作另一项任务的输入。链提供了一种结构，可以将这些操作连接成一个统一的整体。

本质上，链是一个端到端流程，它接收输入并通过一系列组件处理数据以产生输出。可以将其视为函数组合。如果格式化提示词 (prompt)是一个函数 $f_{提示词}$，调用模型是另一个函数 $f_{模型}$，那么一个简单的链就代表了它们的组合：


$$
\text{输出} = f_{模型}(f_{提示词}(\text{输入}))
$$


这种结构使您的应用逻辑清晰且易于管理。您无需编写命令式代码来处理每个步骤，而是定义一个由LangChain执行的声明式序列。

### 基本链的构成

最基本的链包含三个部分，您在上一章中已经熟悉它们：提示词 (prompt)模板、模型和输出解析器。

1. **PromptTemplate (提示词模板)**: 接收初始输入变量（例如，一个字典），并将它们格式化为模型所需的提示词值（可以是字符串或消息列表）。
2. **Model (模型，即LLM或ChatModel)**: 接收格式化后的提示词并生成响应（一个文本字符串或一个消息对象）。
3. **OutputParser (输出解析器)**: 接收模型输出并将其转换为更有用的格式，例如JSON或Python对象。

此序列代表一个单一的、可重用的逻辑块。下图说明了这一流程，显示了数据在每个步骤中如何转换。

> 标准链通过提示词模板、语言模型和输出解析器来处理输入，以生成结构化数据。

### 链为何是一种不可或缺的结构

将应用逻辑组织成链，相比手动编排每次调用，提供了多项优势。

- **标准化**: 每个链都暴露出统一的接口。无论一个链执行单次LLM调用还是一个包含十个步骤的流程，您都以相同的方式与其交互。这通常涉及诸如用于单个输入的 `invoke()` 方法、用于流式输出的 `stream()` 方法以及用于高效处理多个输入的 `batch()` 方法。这种一致性简化了构建和测试。
- **模块化和可组合性**: 链是自包含且可重用的。一个用于总结文章的链可以是一个更大链中的单一组件，该链首先获取文章，然后总结，最后翻译摘要。这种模块化对于构建精密应用而非创建难以管理的代码来说非常重要。
- **可观察性**: 当您运行一个链时，LangChain可以追踪整个执行流程。这使得调试更容易，因为您可以看到序列中每个步骤的精确输入和输出。当我们将介绍LangSmith时，会对此进行详细说明。

在LangChain中构建链的标准方式是使用LangChain表达式语言 (LCEL)，它使用管道 (`|`) 运算符来连接组件。这种语法使得数据流直观且易读。一个由提示词 (prompt)、模型和解析器组成的简单链将是这样的：

```python
chain = prompt | model | parser
```

这行代码定义了一个完整、可执行的流程。输入首先“通过管道”进入提示词，生成的提示词“通过管道”进入模型，模型的输出“通过管道”进入解析器。在后续部分中，我们将使用这种组合语法来构建我们的第一个链，并学习更高级的顺序模式。

## 参考资料

- [LangChain Chains Overview](https://python.langchain.com/docs/conceptual_guide/) — LangChain Developers (2024)
  Publisher: LangChain
  LangChain框架中链的定义、目的和基本组件的官方说明。
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/) — LangChain Developers (2024)
  LangChain表达式语言的官方指南，用于声明式地构建和组合应用逻辑。
- [Building LLM-Powered Applications: From Prompt Engineering to Production](https://www.oreilly.com/library/view/building-llm-powered-applications/9781098670857/) — Pradeeban Kathiravelu, Jamshed Wadia (2024)
  Publisher: O'Reilly Media
  开发基于LLM应用的实用指南，涵盖编排多个模型调用和操作的架构考量。
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou (2022)
  Journal: arXiv preprint arXiv:2201.11903; DOI: [10.48550/arXiv.2201.11903](https://doi.org/10.48550/arXiv.2201.11903)
  提出向大型语言模型提供中间推理步骤的概念，强调顺序处理对于提高复杂任务中模型性能的益处。

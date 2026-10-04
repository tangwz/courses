---
course: "prompt-engineering-llm-application-development"
chapter: "advanced-prompting-strategies"
lesson: "zero-shot-prompting"
sourceId: 2973
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-2-advanced-prompting-strategies/zero-shot-prompting"
title: "零样本提示"
description: "理解并应用LLM在没有预先示例的情况下执行任务的零样本提示。"
order: 1
plots: []
sourceHash: "2a1077f0af074511890ea3985cadb886cc6a8014013c8ceebf48016d28f77f91"
sourceCorrections: []
---

有效引导大型语言模型（LLM）需要精密的策略。这些方法改进了基本提示技术，包括构建基本请求和调整温度等参数 (parameter)。**零样本提示**是一种直接且强大的精密策略。

零样本提示指仅根据描述或指令要求LLM执行任务，而无需在提示本身中提供任务完成的任何示例。你只需说明希望模型做什么，并依赖其广泛的预训练 (pre-training)知识来理解并完成任务。

可以将其想象成给一位经验丰富、之前处理过无数类似任务的同事下达指令。你无需向他们演示如何将邮件分类为垃圾邮件或非垃圾邮件；你只需要求他们“分类此邮件：”并提供邮件内容即可。

### 零样本提示的运作方式

零样本提示的有效性直接源于大型语言模型的特性。这些模型通过包含来自互联网、书籍、代码等海量文本的数据集进行训练。在训练过程中，它们不仅学习语言结构，还学习了与翻译、摘要、问答、分类甚至简单推理 (inference)等常见任务相关的模式。

当你提供一个零样本提示时，你实际上是以一种激活模型已有能力的方式来制定你的请求。如果任务描述类似于模型在训练中遇到的模式，它通常可以进行泛化并成功执行任务，即使它在当前提示中没有看到你的特定任务通过示例进行说明。现代经过指令微调 (fine-tuning)的模型尤其善于遵循零样本指令。

### 零样本提示的示例

让我们看几个不同任务的示例：

**1. 情感分类：**

```text
将以下客户评论的情感分类为积极、消极或中性。

评论: "The setup process was incredibly confusing, but the support team was very helpful in resolving the issue."

情感:
```

在这里，模型预期能理解情感分类的理念，并将其应用于提供的文本，输出指定的标签之一。

**2. 文本摘要：**

```text
将以下文章的要点总结为三个要点：

文章: [在此处插入一段长段落或文章文本]

摘要:
*
```

该指令明确定义了任务（摘要）和期望的输出格式（三个要点）。

**3. 信息提取：**

```text
从下面的句子中提取人名和他们工作的公司。将输出格式化为 JSON。

句子: "After joining the project, Sarah Lee from Innovate Solutions quickly identified the main bottlenecks."

输出:
```

此提示要求提取特定实体并指定输出格式（JSON），依赖模型解析文本和组织数据的能力。

**4. 简单翻译：**

```text
将以下英文文本翻译成法文：

文本: "Hello, how are you?"

法文:
```

翻译是LLM 常规训练的任务，使其非常适合零样本提示。

### 零样本提示的优点

- **简单易行：** 这是最直接的提示方法。你只需制定清晰的指令。
- **无需示例数据：** 你无需整理或格式化示例，这在数据稀缺时可能耗时或很困难。
- **基准性能：** 它提供了一种快速衡量LLM在任务上能力的方法，而无需大量提示工程 (prompt engineering)工作。

### 局限性与考量

零样本提示虽然强大，但并非总是足够的，特别是对于更复杂或专业的任务：

- **性能：** 对于高度具体、新颖或需要训练数据中未充分体现的知识的任务，零样本性能可能低于使用示例的方法（如接下来讨论的少样本提示）。
- **指令清晰度：** 输出的质量很大程度上取决于指令措辞的清晰度和明确性。模糊的提示会导致更不可预测的结果。
- **模型依赖性：** 不同模型之间的零样本能力可能显著不同。更大、更强大或经过指令微调 (fine-tuning)的模型通常在零样本任务上表现更好。

### 何时使用

零样本提示是许多应用的`出色的起点`。如果你需要对文本进行分类、翻译语言、摘要内容或执行其他常见NLP任务，请务必先尝试零样本提示。如果性能符合你的要求，你就节省了大量精力。如果不能，它可作为基准，表明可能需要更高级的技术，例如在提示中提供示例。

接下来，我们将研究少样本提示，它涉及向模型提供示例以提高其在特定任务上的性能。然而，理解并有效使用零样本提示是利用LLM固有能力的基本技能。

## 参考资料

- [Finetuned Language Models are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) — Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le (2022)
  Journal: ICLR; DOI: [10.48550/arXiv.2109.01652](https://doi.org/10.48550/arXiv.2109.01652)
  这篇论文介绍了“FLAN”，表明对语言模型进行多样化指令的微调显著提高了其对未见任务的零样本泛化能力，直接支持了关于指令调优模型的论点。

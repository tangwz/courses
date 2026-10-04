---
course: "prompt-engineering-llm-application-development"
chapter: "foundations-prompt-engineering"
lesson: "components-of-a-prompt"
sourceId: 2960
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-1-foundations-prompt-engineering/components-of-a-prompt"
title: "提示词的构成要素"
description: "分析提示词的结构：指令、上下文、输入数据、输出指示。"
order: 3
plots: []
sourceHash: "ad5988b262b74d659093240d94e5ea36c893217891d95ab5812f44acec510cd2"
sourceCorrections: []
---

提示词 (prompt)工程在与大型语言模型交互中具有主要作用。分析常见提示词的结构有助于理解其基本要素。虽然提示词的复杂程度差异很大，但它们通常包含几个不同的要素，共同作用以指导LLM。了解这些构成要素有助于更系统地构建提示词，并在模型行为不如预期时进行问题排查。

不要将提示词仅仅看作一个问题，而应将其视为你希望LLM执行任务的小型规范。主要构成要素通常包含：

1. **指令：** 指定LLM应执行的任务。
2. **上下文 (context)：** 提供LLM应考虑的背景资料或限制。
3. **输入数据：** LLM需要处理的特定信息。
4. **输出指示：** 定义回复所需的格式或结构。

值得注意的是，并非每个提示词都需要所有这些构成要素。简单的提示词可能只包含指令和输入数据。然而，对于更复杂的任务，明确定义每个相关的构成要素可以提高清晰度并更好地控制LLM的输出。

### 指令

指令是提示词 (prompt)中的“动作动词”。它告诉LLM*你希望它做什么*。清晰、直接的指令通常比模糊的更有效。

- **示例（简单）：** “总结以下文本：”
- **示例（更具体）：** “将以下英文句子翻译成法文：”
- **示例（任务导向）：** “编写Python代码以读取CSV文件并打印前5行。”

指令为LLM的生成过程设定了主要目标。

### 上下文 (context)

上下文提供LLM在执行指令时应使用的背景、限制或补充信息。这可以包含：

- LLM可能不自带的信息（例如，特定项目的细节、不在其训练数据中的近期事件）。
- 对输出的限制（例如，“假设你是一名乐于助人的助教”，“总结应不超过两句话”）。
- 相关领域知识。
- **示例：**

  ```
  背景信息：用户正在询问我们上周发布的新产品“量子小部件”。它具有自对准机制，并有三种颜色：红色、蓝色和绿色。

  指令：起草一封简短友好的电子邮件回复，回答用户关于颜色选项的问题。

  输入数据：用户问题：“量子小部件有黑色吗？”
  ```

  在这里，上下文提供了LLM需要用来形成准确回复的产品细节。

### 输入数据

这是你希望LLM基于指令和上下文 (context)进行处理或转换的特定信息。它可以是文本、代码、问题或任何与任务相关的数据。

- **示例（总结）：** 输入数据将是你希望总结的长篇文章。
- **示例（翻译）：** 输入数据是源语言中的句子。
- **示例（代码生成）：** 输入数据可能是对所需功能或逻辑的描述。

在提示词 (prompt)结构中清晰地分离输入数据通常有助于模型专注于正确的信息。

### 输出指示

输出指示表明输出应如何格式化或结构化。这有助于确保LLM的回复可以直接用于你的应用程序。它可以是简单的提示，也可以是明确的格式描述。

- **示例（简单提示）：** 以“总结：”或“法文翻译：”结束提示词 (prompt)，可以促使模型以适当的方式开始回复。
- **示例（格式规范）：** “以JSON格式提供答案，键为'original\_sentence'和'translated\_sentence'。”
- **示例（代码结构）：** “将结果输出为Python字符串列表。”

在将LLM回复以编程方式集成到其他系统时，指定输出格式变得尤为重要。像请求JSON或Markdown这样的技术将在第2章中更详细地介绍。

### 构成要素的组合

这些构成要素共同作用以形成一个完整的提示词 (prompt)。它们的顺序和措辞可以显著影响结果。请看这个针对分类任务组合所有要素的例子：

```
指令：根据以下客户评论的情绪进行分类。
上下文：情绪类别为：积极、消极、中立。仅回复类别名称。
输入数据：评论：“设置很容易，但电池续航时间相当短。”
输出指示：情绪：
```

这个结构化提示词清晰地定义了任务（分类情绪），提供了必要的上下文 (context)（类别），确定了要处理的数据（评论），并指示了所需输出的开始。

> 常见的提示词结构将指令、上下文、输入数据和输出指示组合起来，以指导LLM。

尝试如何措辞和组合这些构成要素是提示词工程的核心。在您完成本章的动手练习并学习更高级的技术时，请留意调整每个构成要素如何影响LLM的回复。

## 参考资料

- [Language Models are Few-Shot Learners](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0fc60e972337d0793b8f1e6879e-Paper.pdf) — Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Anna Padilla, Skyler Gray, Alec Radford, Mark Chen, Raymond Teztloff, Sam McCandlish, Boris Dayma, Svetlana Sutskever, Sam Altman, Dario Amodei (2020)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 33; Pages: 1877-1901; DOI: [10.48550/arXiv.2005.14165](https://doi.org/10.48550/arXiv.2005.14165)
  本文介绍了GPT-3并展示了上下文学习的效用，为提示工程奠定了基础。
- [Prompt engineering](https://platform.openai.com/docs/guides/prompt-engineering) — OpenAI (2024)
  Publisher: OpenAI
  来自领先LLM开发者的官方指南，为设计高效提示提供实用建议和最佳实践，包括其组成部分。
- [Pre-train, Prompt, and Predict: A Systematic Survey of Prompt Engineering](https://arxiv.org/pdf/2107.13586.pdf) — Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhenglian Jiang, Fanchao Qi, Adam Romo, Kenneth Johnson (2023)
  Journal: ACM Computing Surveys; DOI: [10.1145/3602931](https://doi.org/10.1145/3602931)
  一项全面的调查，分类和分析了各种提示工程技术，包括有效提示设计的组成部分及其在LLM交互中的作用。

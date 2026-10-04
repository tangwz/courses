---
course: "getting-started-rag"
chapter: "rag-generation-augmentation"
lesson: "context-injection-methods"
sourceId: 3654
sourceUrl: "https://apxml.com/zh/courses/getting-started-rag/chapter-4-rag-generation-augmentation/context-injection-methods"
title: "上下文注入方法"
description: "展示将获取到的文本段落插入到LLM提示词中的不同方式。"
order: 3
plots: []
sourceHash: "49b9b78ff58a3c443168847af4e354b442c76bdcbcf76f80ced7819a6d303144"
sourceCorrections: []
---

上下文 (context)注入是将相关文本段落（即“上下文”）整合到将发送给大型语言模型（LLM）的提示词 (prompt)中。注入此上下文的方式显著影响LLM有效发挥其作用的能力。常见的上下文注入方法在此讨论。

### 直接拼接

最直接的做法是简单地将获取到的上下文 (context)直接附加到原始用户问题的前面或后面。通常会使用分隔符或引导短语。

**示例结构：**

```
上下文：
[获取到的段落 1]
[获取到的段落 2]
...

根据以上上下文，回答以下问题：[用户问题]
```

或者，问题可能在前：

```
问题：[用户问题]

使用以下信息回答问题：
[获取到的段落 1]
[获取到的段落 2]
...
```

- **优点：** 实现简单。对提示词 (prompt)工程要求不高。
- **缺点：** 对顺序和格式敏感。如果获取到的段落很多，原始问题可能会被“淹没”，或者重要上下文离问题太远，导致LLM无法有效集中处理。这种方法也更难控制LLM如何使用上下文而非其内部知识。

### 模板化注入

一种更有条理且通常更受欢迎的方法是使用提示词 (prompt)模板。这些是预定义的字符串，带有问题和上下文 (context)的占位符。Python的f-strings或专门的模板库（如Jinja2，常用于LangChain等框架中）可以使这项工作易于管理。

**示例模板（Python f-string）：**

```python
# 假设 'retrieved_docs' 是字符串列表
# 'user_query' 是原始问题
context_string = "\n".join(retrieved_docs)

prompt_template = f"""
你是一个助手，负责根据提供的上下文回答问题。
不要使用以下上下文之外的任何信息。

上下文：
{context_string}

问题：{user_query}

答案：
"""

# 'prompt_template' 现在包含为LLM完全格式化好的提示词
```

- **优点：** 对提示词结构有更强的控制力。允许向LLM提供关于如何使用上下文的清晰指令（例如，“仅*根据*提供的上下文回答”）。使提示词更易于管理、版本控制和调试。
- **缺点：** 需要比直接拼接更多的设置。设计有效的模板可能需要一些尝试。

### 结构化输入格式

一些LLM或交互框架可能支持更结构化的输入格式，可能接受问题和上下文 (context)作为对象内的独立参数 (parameter)或字段。

**示例（API调用）：**

```python
response = llm_api.generate(
  query="什么是RAG？",
  context_documents=[
    "RAG 是检索增强生成（Retrieve-Augmented Generation）的缩写...",
    "它结合了检索与生成..."
  ],
  instructions="仅使用提供的文档回答问题。"
)
```

- **优点：** 可能非常清晰和明确。可能允许底层模型架构更有效地处理上下文分离。
- **缺点：** 对于标准开源或广泛可用的商业LLM API来说不太常见，这些API通常期望单个字符串提示词 (prompt)。支持程度在不同模型和平台之间差异很大。

### 提示词 (prompt)中的位置

将上下文 (context)放在模板中的位置也很重要。常见模式包括：

1. **上下文优先：** 将所有上下文放在问题和指令之前。
2. **问题优先：** 将问题放在最前面，然后是指令，最后是上下文。
3. **交错放置（高级）：** 在更复杂的场景中，上下文可能与指令或推理 (inference)过程的部分内容交错放置。这在基本的RAG中不太常见。

最佳位置取决于所使用的具体LLM和任务的性质。一些模型表现出近因偏差，更关注提示词中后面出现的信息。通常需要进行实验。

下图展示了模板化注入的流程：

> 用户问题和获取到的上下文段落被插入到提示词模板中指定的占位符内。生成的格式化提示词随后发送给LLM。

选择合适的注入方法需要在实现上的简易性、对LLM行为的控制需求以及优化其使用提供信息的方式之间取得平衡。模板化为大多数RAG应用提供了灵活性和控制力的良好结合。在构建RAG系统时，思考这些不同的注入策略如何影响最终的生成输出，尤其是在处理不同数量的获取上下文时。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍了基础的检索增强生成（RAG）架构，强调了将检索到的知识整合到语言模型提示中的必要性及方法。
- [Prompt Engineering Guide](https://www.promptingguide.ai/) — Prompt Engineering Guide Contributors (2024)
  一个全面的在线资源，提供设计有效大型语言模型提示的策略和技术，包括如何用外部上下文构建提示。
- [LangChain Documentation - Prompt Templates](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG8y_mQ75b6xci3Hq3T5MxFojfBrr2fm1rEyih93kH61J130v4kjyQCLVjmEFhsXrmmRMNizx6nKwwtXn2h-W030CNs9VSuApy0tc97VvQwsBYmrMz1nXrvCB82U5eis0Wu5RGH8yB6UiQbd3nM1a6NWXxyitOEHg==) — LangChain (2024)
  LangChain框架中提示模板使用的官方文档，这是一种管理和将上下文注入LLM提示的实用方法。
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang (2023)
  Journal: Transactions of the Association for Computational Linguistics (TACL); DOI: [10.48550/arXiv.2307.03172](https://doi.org/10.48550/arXiv.2307.03172)
  这项研究调查了信息在长上下文中的放置如何影响语言模型性能，特别是讨论了与上下文注入相关的“近因偏见”等现象。

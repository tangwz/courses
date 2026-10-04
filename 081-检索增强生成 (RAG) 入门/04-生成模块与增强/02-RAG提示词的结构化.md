# RAG提示词的结构化

来源：[原文](https://apxml.com/zh/courses/getting-started-rag/chapter-4-rag-generation-augmentation/structuring-rag-prompts)

[返回章节目录](README.md) · [返回课程目录](../README.md)

检索增强生成（RAG）系统的效用不仅取决于检索到正确的信息；它很大程度上取决于信息*如何*呈现给大型语言模型（LLM）。提示词 (prompt)是检索到的上下文 (context)与生成过程之间的桥梁。编写好这个提示词对于指导LLM生成准确、相关且基于上下文的回答必不可少。

可以将提示词视为您提供给LLM的使用手册。它需要清楚地说明任务，呈现相关证据（即检索到的上下文），并指明如何使用这些证据来满足用户最初的请求。一个结构不佳的提示词可能会使LLM偏离方向，即使检索步骤很成功。它可能忽略上下文，误解用户查询，或未能有效整合信息。

### 基本提示词 (prompt)模板

构建RAG提示词的常见起点是将检索到的上下文 (context)和原始用户查询组合成一个单一的LLM输入。一个基本模板可能如下所示：

```text
根据以下上下文：

[CONTEXT_CHUNK_1]
[CONTEXT_CHUNK_2]
...
[CONTEXT_CHUNK_N]

回答以下问题：[USER_QUERY]
```

在这里，`[CONTEXT_CHUNK_1]` 到 `[CONTEXT_CHUNK_N]` 是从您的知识源检索到的实际文本段落的占位符，而 `[USER_QUERY]` 则是用户原始的问题或指令。

### 上下文 (context)放置与指令

上下文相对于查询的放置位置很重要。虽然上面的模板将上下文放在前面，您也可以将其放在查询之后。一些LLM可能表现出近期偏见，对提示词 (prompt)后面出现的信息给予更多关注。通常需要通过实验来确定您的特定LLM和任务的最佳放置位置。

比单独放置更重要的是提供给LLM的*指令*。明确的指令有助于约束模型的行为，并鼓励它依赖所提供的信息。请考虑以下变体：

- **严格依据上下文：** “*只*使用以下提供的上下文来回答问题。如果答案未在上下文中找到，请说明您无法根据所提供的信息回答。”

  ```text
  上下文：
  [CONTEXT]

  问题：[USER_QUERY]

  严格根据所提供的上下文回答。如果信息不存在，请回复“我无法根据所提供的上下文回答。”
  ```
- **整合指令：** “整合以下文档中的信息来回答用户的问题。”

  ```text
  用户问题：[USER_QUERY]

  相关信息：
  文档1：[CONTEXT_CHUNK_1]
  文档2：[CONTEXT_CHUNK_2]

  结合文档中的相关信息，提供一个全面的回答。
  ```

这些指令指导LLM*如何*使用上下文，减少了产生幻觉 (hallucination)（编造信息）或单纯依赖其内部可能过时的知识的可能性。

### 格式化多个上下文 (context)片段

当您的检索器返回多个相关文本片段时，您需要在提示词 (prompt)中以清晰的方式呈现它们。简单地将它们连接起来可能会使LLM感到困惑。更好的方法包括：

1. **分隔符：** 在片段之间使用不同的分隔符，例如双换行符、水平分隔线（使用`* * *`等字符）或特定标签（如`[CONTEXT] ... [/CONTEXT]`）。
2. **编号/标签：** 为每个片段分配标识符。如果您以后希望LLM引用来源，这会很有帮助。

   ```text
   使用以下上下文片段来回答问题：

   上下文1：
   [CONTEXT_CHUNK_1]

   上下文2：
   [CONTEXT_CHUNK_2]

   问题：[USER_QUERY]
   ```
3. **包含元数据：** 如第3章所述，片段通常附带元数据（例如，源文档名称、页码）。在提示词中，将相关元数据与片段文本一同包含可以有益，尤其对于最终回答中的来源归属。

   ```text
   来自'report_v2.pdf'第5页的上下文：
   [CONTEXT_CHUNK_1]

   来自'website_faq.html'的上下文：
   [CONTEXT_CHUNK_2]

   问题：[USER_QUERY]
   使用提供的上下文回答问题。
   ```

### 处理不充分或不相关的上下文 (context)

检索步骤并非总是完美的。有时，它可能返回并非真正相关的片段，或者完全找不到任何相关信息。您的提示词 (prompt)结构应该预料到这一点。通过指导LLM在上下文无用时如何表现（如前面“严格依据上下文”示例所示），您可以鼓励其给出更诚实可靠的回答，而不是基于不良证据强行提供答案。

### RAG提示词 (prompt)示例

我们来看几个场景：

**场景1：简单问答**

- **用户查询：** "What is the maximum context window size for the `Llama-3-8B-Instruct` model?"
- **检索到的上下文 (context)（片段1）：** "The Llama 3 family includes models with 8B and 70B parameters. Both initial instruction-tuned versions support context lengths of 8,192 tokens."
- **检索到的上下文（片段2）：** "When choosing a model, consider the trade-off between parameter count and computational requirements. Larger models often perform better but require more resources. Context window limitations also affect suitability for tasks involving long documents."
- **提示词：**

  ```text
  根据以下上下文：

  上下文1：
  The Llama 3 family includes models with 8B and 70B parameters. Both initial instruction-tuned versions support context lengths of 8,192 tokens.

  上下文2：
  When choosing a model, consider the trade-off between parameter count and computational requirements. Larger models often perform better but require more resources. Context window limitations also affect suitability for tasks involving long documents.

  回答以下问题：What is the maximum context window size for the `Llama-3-8B-Instruct` model? 只使用提供的上下文。
  ```

**场景2：未找到相关上下文的查询**

- **用户查询：** "What is the airspeed velocity of an unladen swallow?"
- **检索到的上下文：** （假设检索器在索引文档中没有找到关于燕子飞行速度的任何相关片段）。
- **提示词：**

  ```text
  根据以下上下文：

  [未找到相关上下文]

  回答以下问题：What is the airspeed velocity of an unladen swallow? 只使用提供的上下文。如果上下文不包含答案，请说明。
  ```
- **预期LLM输出：** "I cannot answer based on the provided context."

### 迭代与改进

RAG的提示词 (prompt)工程很少是一次性过程。理想的结构很大程度上取决于所使用的特定LLM、您的数据特性以及您预期用户查询的复杂性。从基本结构开始，使用代表性查询和检索到的上下文 (context)进行测试，分析LLM的输出，并迭代地改进提示词。措辞、格式或指令清晰度上的微小变化有时可以给生成回答的质量和可靠性带来很大的改善。

通过精心构建提示词，您创建了一个清晰的沟通渠道，使LLM能够有效运用检索到的信息，并生成基于您特定知识库的回答。这种结构化方法对于发挥RAG的能力作用很大。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401.pdf) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Narsimha Chilkuri, Max Bartolo, Jeff Lu, Minqi Jiang, Harun Šaríć, Fabio Soares, Yury Kastryulin, Leon Bottou, Sebastian Riedel, Pasquale Minervini (2020)
  Journal: NeurIPS; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍了原始的检索增强生成（RAG）框架，详细说明了如何利用检索到的文档来增强语言模型的生成过程。
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/pdf/2307.03172.pdf) — Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Percy Liang, Matei Zaharia (2023)
  Journal: NeurIPS (Preprint on arXiv); DOI: [10.48550/arXiv.2307.03172](https://doi.org/10.48550/arXiv.2307.03172)
  研究了上下文信息位置对大型语言模型性能的影响，并讨论了长文本中近因偏误等现象。
- [Prompt engineering](https://platform.openai.com/docs/guides/prompt-engineering) — OpenAI (2024)
  Publisher: OpenAI
  为大型语言模型设计有效提示提供了实用指南，包括指令结构和外部信息整合。
- [A Survey on Retrieval-Augmented Generation for Large Language Models](https://doi.org/10.48550/arXiv.2312.10997) — Yunfan Gao, Yun Xiong, Xinyang Feng, Zhangyang Wang, Xunlei Wu, Jie Zhou, Wenqi Wang, Peng Zhang, Song Guo, Cheng-Zhong Xu (2024)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2312.10997](https://doi.org/10.48550/arXiv.2312.10997)
  全面回顾了RAG系统，讨论了生成组件中各种提示策略和上下文整合方法。

---

[上一节](01-%E7%94%9F%E6%88%90%E5%99%A8%EF%BC%88LLM%EF%BC%89%E5%9C%A8RAG%E4%B8%AD%E7%9A%84%E4%BD%9C%E7%94%A8.md) · [下一节](03-%E4%B8%8A%E4%B8%8B%E6%96%87%E6%B3%A8%E5%85%A5%E6%96%B9%E6%B3%95.md)

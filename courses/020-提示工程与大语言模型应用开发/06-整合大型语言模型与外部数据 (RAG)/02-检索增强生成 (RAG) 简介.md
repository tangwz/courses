# 检索增强生成 (RAG) 简介

来源：[原文](https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-6-integrating-llms-external-data-rag/introduction-retrieval-augmented-generation-rag)

[返回章节目录](README.md) · [返回课程目录](../README.md)

大型语言模型（LLM）尽管能力出众，但在运行时有一个明显的局限：它们的知识是静态的，固定在训练数据收集时的状态。它们本身无法获取其训练*之后*产生的信息，例如最新新闻、内部文档更新或私有知识库中的数据。这使得它们无法回答有关时事的问题，也无法基于专有信息提供有效回应。

检索增强生成（RAG）为此问题提供了一个直接的解决方法。它是一种技术，或者更准确地说，是一种架构模式，通过在生成过程中动态引入从外部来源获取的信息来提升LLM的回复质量。RAG并非仅依赖模型内部化（且可能过时）的知识，而是让LLM在生成答案*之前*查阅相关的外部文档。

可以将其比作开卷考试与闭卷考试。标准的LLM在闭卷环境下运行，仅根据其训练期间“记忆”的内容回答。RAG实际上赋予LLM查阅与所提问题相关的特定参考资料（即您的外部数据源）的能力，使其能够形成更具信息量且上下文 (context)更准确的回复。

RAG的核心流程包含三个主要步骤：

1. **检索：** 给定用户查询或提示，系统首先在一个外部知识源（如文档集合、数据库或网页）中查找与该查询相关的信息。
2. **增强：** 上一步获取的相关信息随后与原始用户查询结合，通常是通过将其作为附加上下文插入到将发送给LLM的提示中。
3. **生成：** LLM接收这个增强后的提示，其中现在包含原始查询和补充上下文。它使用这些组合信息生成最终回复，该回复以获取的数据为依据，或至少从中获得信息。

以下是RAG工作流程的简化视图：

> 检索增强生成过程的概述，展示了外部数据如何为最终LLM回复提供信息。

与在新数据上微调 (fine-tuning)LLM等其他方法相比，此方法具有多项优势：

- **获取最新信息：** RAG系统只需更新外部数据源即可整合最新信息，无需对基础LLM进行昂贵的重新训练。
- **减少幻觉 (hallucination)：** 通过将LLM的回复建立在具体的、获取的事实之上，RAG可以显著减少模型“幻觉”或生成看似合理但错误信息的倾向。
- **可验证性：** 由于回复基于获取的文档，用户通常可以追溯到原始资料，从而提高信任度并支持事实核查。
- **领域特定性：** 它使得通用LLM能够通过按需提供必要上下文，有效回答关于高度特定或专有领域的问题。

微调调整模型的内部参数 (parameter)，而RAG则在推理 (inference)时修改*提供给*模型的输入。这使得RAG成为构建需要LLM与特定、动态知识库交互的应用的灵活且高效的模式。

接下来的章节将详细研究该模式的各个组成部分，内容包括如何准备数据、执行高效检索以及如何有效地将获取的上下文与LLM提示结合。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 33; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍检索增强生成（RAG）架构并展示其有效性的基础性论文。
- [A Survey on Retrieval-Augmented Generation](https://arxiv.org/abs/2312.10997) — Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang (2023)
  Journal: arXiv preprint arXiv:2312.10997 [cs.CL]; DOI: [10.48550/arXiv.2312.10997](https://doi.org/10.48550/arXiv.2312.10997)
  对检索增强生成进行全面回顾，涵盖其背景、进展和各种方法。
- [What is Retrieval Augmented Generation (RAG)?](https://cloud.google.com/learn/what-is-retrieval-augmented-generation) — Google Cloud (2024)
  Publisher: Google Cloud
  来自权威行业来源的易懂概述，解释RAG概念、其优势和核心工作流程。

---

[上一节](01-%E6%A0%87%E5%87%86%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E7%9F%A5%E8%AF%86%E7%9A%84%E5%B1%80%E9%99%90.md) · [下一节](03-%E6%96%87%E6%A1%A3%E5%8A%A0%E8%BD%BD%E4%B8%8E%E6%8B%86%E5%88%86.md)

# 生成器（LLM）在RAG中的作用

来源：[原文](https://apxml.com/zh/courses/getting-started-rag/chapter-4-rag-generation-augmentation/generator-llm-role)

[返回章节目录](README.md) · [返回课程目录](../README.md)

检索增强生成（RAG）包含两个主要阶段。首先，检索组件响应用户查询，从知识库中识别并获取相关信息。接着，生成阶段启动，大型语言模型（LLM）整合检索到的信息并形成最终答案。

可以把RAG流程看作有两个主要引擎。第一个，检索器，负责找到原始材料（相关文本段落）。第二个，生成器，是处理这些材料、将其与原始请求结合并构建成品（即响应）的引擎。

### 整合器：LLM在RAG中如何工作

RAG系统中的生成器组件通常是一个预训练 (pre-training)的大型语言模型。这可以是任何有能力的通用模型，例如GPT系列、Llama、Mistral或通过API访问或本地部署的其他模型。它在RAG架构中的主要作用是**信息整合和连贯响应的生成**。

与标准LLM应用不同，标准应用中模型仅依赖其内部已有的知识（在其训练阶段学习到的），RAG系统中的LLM运行方式不同。它不仅接收用户的原始查询，还接收由第一阶段检索到的上下文 (context)片段。

它的主要职责是：

1. **理解上下文：** LLM必须首先理解检索器识别出的提供的上下文段落。
2. **将上下文与查询关联：** 它需要理解这些段落如何与用户查询中的具体问题或指令相关联。
3. **整合信息：** LLM不仅仅是复制粘贴检索到的文本。其优势在于能够整合可能来自多个片段的信息，将其与通用语言理解结合，并生成一段*新的*、连贯的文本，直接回应查询。
4. **生成自然语言：** 最终输出必须是流畅、结构良好的自然语言响应，适合用户。

请看这个流程图：

> 生成器LLM同时接收原始用户查询和检索到的上下文作为输入，并生成最终的响应。

本质上，检索到的上下文充当有针对性的、即时可用的知识源，指导LLM的生成过程。这使得RAG系统能够生成以下特点的答案：

- **更准确：** 基于特定的、检索到的信息，而不是仅仅依赖于LLM可能过时或泛化的内部知识。
- **更具体：** 根据提供的上下文进行调整，从而减少通用性响应。
- **更不容易产生幻觉 (hallucination)：** 通过提供相关的事实片段，RAG过程限制了LLM，减少了它编造信息的可能性。

因此，LLM组件充当智能整合器。它利用其强大的语言能力，并利用检索器提供的特定、相关数据来引导它们。该阶段的有效性很大程度上取决于检索到的上下文如何很好地整合到呈现给LLM的提示中，本章后续部分将讨论这个话题。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍了RAG框架，阐述了检索和生成组件之间的协同作用。
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen (2023)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2303.18223](https://doi.org/10.48550/arXiv.2303.18223)
  提供对大型语言模型的全面综述，涵盖其架构、功能和实际考量，有助于理解RAG的背景。

---

[上一节](../03-%E5%87%86%E5%A4%87%E6%A3%80%E7%B4%A2%E6%89%80%E9%9C%80%E6%95%B0%E6%8D%AE/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%96%87%E6%A1%A3%E5%88%86%E5%9D%97.md) · [下一节](02-RAG%E6%8F%90%E7%A4%BA%E8%AF%8D%E7%9A%84%E7%BB%93%E6%9E%84%E5%8C%96.md)

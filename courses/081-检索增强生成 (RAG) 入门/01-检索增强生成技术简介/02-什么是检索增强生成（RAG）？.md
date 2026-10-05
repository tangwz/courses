# 什么是检索增强生成（RAG）？

来源：[原文](https://apxml.com/zh/courses/getting-started-rag/chapter-1-introduction-to-rag/what-is-rag)

[返回章节目录](README.md) · [返回课程目录](../README.md)

标准的大型语言模型（LLMs），尽管它们在文本生成方面表现出色，但也面临一些问题。它们的知识是静态的，反映了其训练数据的内容，这意味着它们可能不了解最新事件或特定的私有信息。它们还可能"产生幻觉 (hallucination)"，生成听起来合理但实际错误或无意义的陈述。

检索增强生成（RAG）提供了一种直接的方法来减轻这些问题。其主要原理是，RAG是一种技术，通过整合从外部知识来源检索到的信息，*在*文本生成步骤发生*之前*，提高大型语言模型生成回复的质量和相关性。

可以将其设想为让大型语言模型在回答问题前，可以参考资料。RAG过程不单纯依赖于训练期间编码在其参数 (parameter)中虽广博但可能过时或不完整的信息，而是遵循两个主要阶段：

1. **检索：** 当用户提交查询时，RAG系统首先利用该查询在预定义的知识库中进行查找。该知识库可以是文档集合、数据库、网页或其他与预期查询相关的文本数据源。此阶段的目的是查找与用户查询最相关的文本片段或文档。此组成部分通常称为**检索器**。
2. **增强生成：** 第一步中检索到的相关信息随后与原始用户查询结合。此结合的文本形成了一个丰富或*增强的*提示。此增强提示随后输入到大型语言模型（即**生成器**）中。大型语言模型同时利用原始查询和所提供的背景信息来生成最终回复。

> 检索增强生成系统的基本运作流程。用户查询启动在知识源中的查找，检索到的背景信息增强大型语言模型生成器的查询，随后由其生成最终回复。

通过在提示中直接提供相关、及时且真实可信的背景信息，RAG有助于大型语言模型做到以下几点：

- **生成更准确、更符合事实的回复：** 大型语言模型可以根据所提供的文本来生成回复，减少产生幻觉的可能性。
- **获取最新信息：** 如果知识源保持更新，大型语言模型可以准确回复关于最新进展的问题。
- **应用特定领域知识：** RAG使得大型语言模型能够有效回答外部知识库中包含的专业主题问题，即便这些主题在其原始训练数据中体现不足。

总的来说，RAG动态地为大型语言模型提供处理特定查询所需的有针对性的信息，使生成过程更具信息量且更可靠。后续章节将详细分解检索器和生成器组件，说明如何为知识源准备数据，并指导您构建一个基本的RAG流程。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b49323023730724689b52a0a6d1947b-Paper.pdf) — Patrick Lewis, Yuxiang Wu, Punit Singh Koura, Sebastian Riedel, Edward Grefenstette, Ludovic Denoyer, and Mike Lewis (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 33; Publisher: NeurIPS; Volume: 33; Pages: 17659-17672; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍RAG框架的原创论文，展示了其在知识密集型任务中结合检索与生成的有效性。
- [A Survey on Retrieval-Augmented Generation](https://arxiv.org/pdf/2312.10997.pdf) — Yunfan Gao, Yun Xiong, Xinyu Gao, Kang Zhang, Jiajun Zhang, HUI SUN, and Haizhou Wang (2023)
  Journal: arXiv preprint arXiv:2312.10997
  对RAG的全面综述，涵盖其基本组成部分、各种架构、应用及当前研究方向。
- [Retrieval Augmented Generation: Building the Next Generation of LLM Applications](https://huggingface.co/blog/retrieval-augmented-generation) — Lewis Tunstall, Omar Espejel, and Philipp Schmid (2023)
  Publisher: Hugging Face Blog
  对RAG及其组成部分和实际优势的易懂解释，适合一般的技术读者。

---

[上一节](01-%E6%A0%87%E5%87%86%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md) · [下一节](03-RAG%E7%B3%BB%E7%BB%9F%E7%9A%84%E6%A0%B8%E5%BF%83%E6%9E%B6%E6%9E%84.md)

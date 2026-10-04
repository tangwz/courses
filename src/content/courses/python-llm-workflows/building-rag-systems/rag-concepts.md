---
course: "python-llm-workflows"
chapter: "building-rag-systems"
lesson: "rag-concepts"
sourceId: 5182
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-7-building-rag-systems/rag-concepts"
title: "检索增强生成原理"
description: "了解检索增强生成的原理和优势。"
order: 1
plots: []
sourceHash: "42e60c5adaf745c221f337616f39428f9a30ef8b4ede25d880714ee9682a0e24"
sourceCorrections: []
---

大型语言模型（LLM）能力强大，但其知识通常停留在上次训练运行时的状态。这意味着它们常常不了解时事、具体的公司内部数据或私人文档中的详细信息。由于计算成本和时间原因，持续地重新训练这些庞大的模型以纳入新信息通常不切实际。检索增强生成（RAG）为这个问题提供了一个巧妙且高效的解决办法。RAG不是将所有外部信息直接编码到LLM的参数 (parameter)中，而是在模型需要时——特别是在用户提出问题（推理 (inference)时）时——动态地为其提供相关的外部信息。

RAG背后的核心思想是将信息检索系统的优势与LLM的文本生成能力结合起来。当收到查询时，RAG系统不会立即要求LLM仅凭其内部记忆生成答案。首先，它会搜索外部知识源（如文档数据库、网页或内部维基）以查找与查询相关的信息。然后，这些检索到的信息被用来“增强”原始查询，创建一个新的、上下文 (context)丰富的提示。这个增强后的提示随后被提供给LLM，引导它生成一个基于所提供的外部事实的答案。

在LLM应用中使用RAG带来以下几项重要优势：

- **获取外部和及时信息：** RAG使LLM能够回答其训练数据之外的问题，例如最新进展、专有信息或您文档中保存的特定领域知识。
- **减少幻觉 (hallucination)：** LLM有时会生成听起来正确但事实不准确或毫无意义的回答——这通常被称为“幻觉”。通过在提示中直接提供相关的、事实性的上下文，RAG大大缓解了这个问题，确保LLM的输出基于可核实的信息。
- **提高具体性和相关性：** 检索到的上下文帮助LLM生成直接回应用户查询具体情况的答案，与LLM一般训练得出的通用答案相比，这能提供更具体和更相关的回答。
- **增加可信度：** 由于系统从特定来源检索信息，它通常可以在生成的答案旁边提供引用或参考文献。这使用户能够追溯信息的来源并确认其准确性，从而提升对系统的信任。
- **知识更新便捷：** RAG系统使用的外部知识库可以独立于LLM进行更新。添加新文档或修改现有文档不需要重新训练LLM，这使得保持系统知识的时效性更加简单和经济。

RAG系统在运行时的一般流程可以分为以下几个阶段：

1. **检索：** 当用户提交查询时，系统会在一个专门准备的知识库（通常是为语义搜索优化的向量 (vector)存储）中查找与查询含义最相关的文本片段或文档。
2. **增强：** 选出最相关的检索文本片段，并将其与用户的原始查询结合。这会创建一个增强后的提示。例如，该提示可能会这样指示LLM：“根据以下信息：[检索到的文本片段]，回答这个问题：[原始用户查询]”。
3. **生成：** 这个现在富含相关上下文的增强提示被发送给LLM。LLM结合所提供的上下文和其内部知识，为用户生成最终回应。

这里是展示运行时流程的图示：

> 该图展示了检索增强生成流程中的主要阶段：检索相关信息，用这些信息增强用户查询，然后将其提供给LLM进行生成。

整个过程依赖于有效的数据索引、语义搜索（常使用向量嵌入 (embedding)和向量存储）以及增强提示的构建方法。我们将了解LangChain和LlamaIndex等库，它们提供了工具和抽象，以简化这些RAG流程的构建。在接下来的章节中，我们将了解这些组件如何协同工作，并特别关注向量嵌入和数据库在实现高效检索中的作用。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2020); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍检索增强生成（RAG）架构的奠基性论文，概述其设计及其在知识密集型任务中的优势。

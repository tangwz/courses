---
course: "getting-started-with-llm-toolkit"
chapter: "data-preparation-for-retrieval"
lesson: "rationale-for-text-chunking"
sourceId: 7765
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-4-data-preparation-for-retrieval/rationale-for-text-chunking"
title: "文本分块的原理"
description: "了解为什么将大型文档分割成更小的数据块对于有效检索是必要的。"
order: 3
plots: []
sourceHash: "203b2cb75eebf5c5833bf0ae8d0bb8b196edfd4dca517b5fd568cad026399cb4"
sourceCorrections: []
---

将文档分解为易于管理的小块是为检索准备数据的重要一步。这个被称为分块的过程，是构建高效 RAG 系统的基础。其原因在于所有大型语言模型的一个主要制约：它们有限的上下文 (context)窗口。

可以将上下文窗口看作是模型的工作记忆。它是模型一次性可以处理的信息总量，包括你的指令、问题以及任何提供的文档。这个限制不是以页或字为单位，而是以*令牌*（词语的片段）为单位。一个拥有 8,000 令牌上下文窗口的模型无法在单次请求中处理一个 20,000 令牌的文档。尝试这样做会导致错误，即使能够处理，处理非常大的上下文通常也是缓慢且昂贵的。

这一制约给我们带来了一个难题。如果我们不能将整个文档提供给模型，如何能确保它拥有正确的信息来回答问题？这就是分块变得必不可少的地方。通过将大型文档 $D$ 分割成一系列更小、已索引的数据块 $\{c_1, c_2, ..., c_n\}$，我们将问题从“阅读整本书”转变为“找到最相关的页面”。

### 数据块大小的难题

RAG 系统的有效性很大程度上取决于其数据块的质量。这产生了一个“金发姑娘”问题，即数据块的大小必须恰到好处。

- **如果数据块过大：** 它们可能包含太多与用户查询不相关的信息。这种噪音会冲淡重要事实，使大型语言模型更难找到确切答案。想象一下，你询问一个具体的配料，却得到一本完整的食谱。信息虽然存在，但却被掩盖了。
- **如果数据块过小：** 它们可能没有足够的周边内容来传达完整意义。单独的一句话如果没有前后文可能会有歧义。这可能导致大型语言模型误解信息，或因必要细节分散在不同数据块中而无法回答问题。

目标是创建足够小以实现精确检索，但又足够大以保持语义完整并为大型语言模型提供充分上下文 (context)的数据块。

> RAG 工作流程。大型文档首先被分割成更小的数据块。当用户提出查询时，检索系统会找到最相关的数据块，然后将其作为上下文传递给大型语言模型以生成答案。

### 一种初步的分块方法

在原理明确之后，我们来看看一个基本实现。最简单的文本分块方法是根据固定字符数进行分割。虽然这种方法不考虑句子或段落边界，但它清楚地展示了分割的核心原则。

工具包提供了 `simple_chunker` 用于此目的。让我们取一个示例文本并将其分割成 100 个字符的数据块。

```python
from kerb.chunk import simple_chunker

text = """
Artificial intelligence (AI) is transforming technology.
Machine learning enables computers to learn from data without explicit programming.
Natural language processing allows machines to understand and generate human language.
""".strip()

# 将文本分割成大约100个字符的数据块
chunks = simple_chunker(text, chunk_size=100)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} ({len(chunk)} chars): '{chunk}'")
```

运行此代码会产生以下输出：

```text
Artificial intelligence (AI) is transforming technology.
Machine learning enables compute'
Chunk 2 (100 chars): 'rs to learn from data without explicit programming.
Natural language processing allows machines to unde'
Chunk 3 (62 chars): 'rstand and generate human language.'
```

正如您所看到的，这种方法直接但粗糙。它在“computers”和“understand”等词语中间分割文本。数据块边界处的上下文 (context)丢失会损害检索质量。一个关于“理解语言”的查询可能无法有效匹配第三个数据块，因为“understand”这个词不完整。

这种基本方法凸显了对更智能的数据块分割策略的需求，我们将在下一节中进行研究。通过尊重文本的自然结构，例如句子和段落，我们可以创建更有意义的数据块，从而保留上下文并带来更好的 RAG 性能。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401.pdf) — Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems; Publisher: Advances in Neural Information Processing Systems; Volume: 33; Pages: 9459–9474; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍了检索增强生成（RAG）及其基本架构，包括知识源的准备。
- [Understanding tokens and context windows](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them) — OpenAI (2025)
  Publisher: OpenAI
  清晰解释了令牌的工作原理、上下文窗口的概念及其在大型语言模型中的重要性。
- [A Survey on Retrieval-Augmented Generation](https://arxiv.org/pdf/2401.12192.pdf) — Yunfan Shao, Zhicheng Dou, Jiantao Ji, Xiaoxue Li, et al. (2024)
  Journal: arXiv preprint arXiv:2401.12192; Publisher: arXiv; DOI: [10.48550/arXiv.2401.12192](https://doi.org/10.48550/arXiv.2401.12192)
  一份最新的调查，全面回顾了检索增强生成，其中包含专门讨论文本分块等数据准备技术的章节。
- [Text Splitters Overview](https://python.langchain.com/docs/modules/data_connection/document_transformers/) — LangChain ()
  Publisher: LangChain
  介绍了RAG系统中使用的各种文本分割策略，展示了除简单固定大小分块之外，旨在保留上下文的方法。

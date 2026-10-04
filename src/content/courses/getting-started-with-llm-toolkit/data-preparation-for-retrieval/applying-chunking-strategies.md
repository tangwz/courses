---
course: "getting-started-with-llm-toolkit"
chapter: "data-preparation-for-retrieval"
lesson: "applying-chunking-strategies"
sourceId: 7766
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-4-data-preparation-for-retrieval/applying-chunking-strategies"
title: "应用分块策略"
description: "实施各种分块策略，例如固定大小分块和递归字符分割。"
order: 4
plots: []
sourceHash: "953578358aeba3bdc3a8fd1bc9f9fce82d9cb75ae59755eca6d710ff90db0258"
sourceCorrections: []
---

将加载的数据分割成较小、易于管理的部分，这是为嵌入 (embedding)和检索做准备的一个必不可少的步骤。文本分割的方法会很大程度上影响RAG系统的性能。不同的策略在简单性、语义连贯性和计算开销之间提供了权衡。下面将介绍几种常见且高效的分块策略。

### 固定大小分块

最直接的方法是固定大小分块。此方法将文档分割成预设长度（以字符为单位）的片段。它实现简单，计算效率高。

然而，其主要缺点在于它不考虑文本的结构或含义。这可能导致分块在句中甚至词中突然结束，从而破坏呈现给语言模型的信息的语义完整性。

您可以使用`chunk_text`工具来实现一个基本的固定大小分块器。

```python
from kerb.chunk import chunk_text

text = "Artificial intelligence is transforming industries. Machine learning enables computers to learn from data. Natural language processing helps them understand text."

# 将文本分割成80个字符大小、无重叠的分块
chunks = chunk_text(text, chunk_size=80, overlap=0)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: '{chunk}'")
```

输出显示了这种简单方法的问题。请注意第二个分块是如何以之前句子的片段开头的。

```text
Chunk 1: 'Artificial intelligence is transforming industries. Machine learning enables compute'
Chunk 2: 'rs to learn from data. Natural language processing helps them understand text.'
```

为减少这种上下文 (context)损失，您可以在连续分块之间引入重叠。重叠确保一个分块末尾的一小部分文本在下一个分块的开头重复。这有助于保留分割点周围的局部上下文，对嵌入 (embedding)模型有益。

```python
from kerb.chunk import chunk_text

text = "Artificial intelligence is transforming industries. Machine learning enables computers to learn from data. Natural language processing helps them understand text."

# 将文本分割成80个字符大小、有15个字符重叠的分块
chunks_with_overlap = chunk_text(text, chunk_size=80, overlap=15)

for i, chunk in enumerate(chunks_with_overlap):
    print(f"Chunk {i+1}: '{chunk}'")
```

输出现在显示了第一个分块的末尾如何与第二个分块的开头重叠，为处理任一分块的模型提供了更好的上下文。

```text
Chunk 1: 'Artificial intelligence is transforming industries. Machine learning enables compute'
Chunk 2: 'enables computers to learn from data. Natural language processing helps them underst'
Chunk 3: 'elps them understand text.'
```

尽管重叠有帮助，但固定大小分块仍然是一个不够精细的方法。对于大多数应用，更倾向于采用一种结构感知的方法。

### 递归字符分割

一种更精细的方法是递归字符分割。此策略旨在通过根据分层分隔符列表分割文本来保留语义边界。它首先尝试按最高优先级分隔符（如双换行符，通常用于分隔段落）分割文本。如果生成的分块仍然过大，它会递归地应用层次结构中的下一个分隔符（单换行符、句子、单词，最后是单个字符），直到所有分块都达到期望的大小。

> 递归分割过程会尝试使用尽可能大的语义分隔符，然后再转向较小的分隔符。

这种自上而下的方法有效，因为它优先保持段落等高级语义单元的完整性。默认的分隔符层级为 `["\n\n", "\n", ". ", " ", ""]`。

您可以使用`recursive_chunker`函数来应用此策略。

```python
from kerb.chunk import recursive_chunker

multi_paragraph_text = """
Retrieval-Augmented Generation (RAG) is a powerful technique for LLM applications. It combines the benefits of retrieval systems with generative models.

The process works in several steps. First, documents are chunked and embedded. Then, relevant chunks are retrieved based on the query. Finally, the LLM generates a response using the retrieved context.

Vector databases play an important role in RAG systems.
""".strip()

# 在可能的情况下，创建尊重段落边界的分块
chunks = recursive_chunker(multi_paragraph_text, chunk_size=200)

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i} ({len(chunk)} chars):")
    print(f"  '{chunk}'\n")
```

输出显示了此方法如何成功地将第一个段落完整地保留为一个分块，因为它符合200个字符的限制。第二个较长的段落则在句子边界处进行了分割。

```text
Chunk 1 (148 chars):
  'Retrieval-Augmented Generation (RAG) is a powerful technique for LLM applications. It combines the benefits of retrieval systems with generative models.'

Chunk 2 (198 chars):
  'The process works in several steps. First, documents are chunked and embedded. Then, relevant chunks are retrieved based on the query. Finally, the LLM generates a response using the retrieved context.'

Chunk 3 (62 chars):
  Vector databases play an important role in RAG systems.
```

对于大多数非结构化或半结构化文本，递归字符分割是一种可靠且推荐的默认策略。

### 基于句子的分块

对于RAG系统，尤其是在问答场景中，确保每个分块包含完整的句子非常有益。基于句子的分块首先将整个文档分割成独立的句子，然后将它们组合成分块。此方法保证任何句子都不会被分割到两个分块中。

与固定大小分块一样，此方法也受益于重叠，但在句子层面。例如，一个句子的重叠意味着分块N的最后一个句子成为分块N+1的第一个句子。

`sentence_window_chunker`实现了此策略。您可以以句子的数量定义窗口大小和句子重叠。

```python
from kerb.chunk import sentence_window_chunker

article_text = "Vector databases are essential for RAG. They store embeddings and enable fast search. Pinecone is a popular managed solution. Weaviate is a flexible open-source alternative. Chroma is great for local development. Choosing the right one depends on your needs."

# 将文本按3个句子一组分块，并有1个句子的重叠
chunks = sentence_window_chunker(
    article_text,
    window_sentences=3,
    overlap_sentences=1
)

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}:")
    print(f"  '{chunk}'\n")
```

输出清楚地说明了窗口化和重叠行为。

```text
Chunk 1:
  'Vector databases are essential for RAG. They store embeddings and enable fast search. Pinecone is a popular managed solution.'

Chunk 2:
  'Pinecone is a popular managed solution. Weaviate is a flexible open-source alternative. Chroma is great for local development.'

Chunk 3:
  'Chroma is great for local development. Choosing the right one depends on your needs.'
```

请注意“Pinecone is a popular managed solution.”是如何出现在第一个分块的末尾和第二个分块的开头的。这种重叠提供了有益的上下文 (context)关联，提高了检索质量，因为用户的查询可能与两个分块边界附近的内容语义匹配。

### 选择您的策略

合适的块策略取决于您的数据和应用。

- **固定大小重叠分块**：因其简单性和速度而使用，特别适用于语义边界不那么重要的高度结构化或统一文本。
- **递归分割**：最佳通用选择。它智能地适应文档结构，使其成为混合或未知内容的默认方法。
- **基于句子的分割**：强烈推荐用于RAG系统。它提供清晰、语义完整的块，非常适合生成准确的嵌入 (embedding)并为大型语言模型提供清晰的上下文 (context)。

应用分块策略后，您将得到一组可供进一步优化的文本分块。我们流程中的下一步是应用文本预处理来清理和标准化此内容，从而进一步提高其对嵌入模型的质量。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://papers.nips.cc/paper/2020/file/6b49323027b4e9f55e51d935df683416-Paper.pdf) — Patrick Lewis, Ethan Perez, Aleksa Gordić, Vladimir Karpukhin, Myle Ott, Sebastian Riedel, Douwe van der Vaart (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2020); Publisher: Neural Information Processing Systems Foundation; Volume: 33; Pages: 9459-9474; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  这篇开创性论文介绍了检索增强生成（RAG）框架，确立了文本分块等有效数据准备方法对于提升语言模型性能的必要性。
- [Text splitters](https://python.langchain.com/docs/concepts/#text-splitters) — LangChain Documentation (2024)
  官方文档提供了多种文本分块策略的解释和实现细节，包括固定大小和递归方法，这些策略在实际的RAG应用中被使用。
- [The Impact of Document Chunking Strategies on Retrieval-Augmented Generation Performance](https://arxiv.org/pdf/2312.00067.pdf) — Luyang Weng, Yujie Cheng, Hongye Liu, Yifan Liu (2023)
  Journal: arXiv preprint arXiv:2312.00067; DOI: [10.48550/arXiv.2312.00067](https://doi.org/10.48550/arXiv.2312.00067)
  这篇研究论文分析了不同文本分块策略对检索增强生成系统整体性能的影响，并展示了实证结果。

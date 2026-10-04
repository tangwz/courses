---
course: "getting-started-with-llm-toolkit"
chapter: "embeddings-and-semantic-search"
lesson: "generating-embeddings"
sourceId: 7769
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-5-embeddings-and-semantic-search/generating-embeddings"
title: "生成嵌入"
description: "使用嵌入模块为您的文本块生成向量嵌入。"
order: 2
plots: []
sourceHash: "a4b942735ffe5e39e1e3c8d87d85edbfe6b461157a32f664e0d834c0405e4e89"
sourceCorrections: []
---

为了让计算机能够根据含义理解和比较文档内容，会使用文本嵌入 (embedding)。这些嵌入是文本的数值表示，即向量 (vector)。它们能捕捉文本的语义核心，使系统能够找到与用户查询主题相关的文档片段，即使两者没有共享直接关键词。

Kerb 的 `embedding` 模块提供了一个统一接口，用于使用各种模型生成这些嵌入，从用于测试的本地、无依赖选项到来自 OpenAI 等服务商的高性能模型。

### 创建您的第一个嵌入 (embedding)

生成嵌入最直接的方式是使用 `embed` 函数。它接收一个文本字符串，并返回一个向量 (vector)，该向量以浮点数列表的形式呈现。

让我们为一个简单的句子生成嵌入，并检查其属性：

```python
from kerb.embedding import embed, embedding_dimension, vector_magnitude

text = "Machine learning transforms data into insights"
embedding = embed(text)

print(f"Text: '{text}'")
print(f"Embedding dimension: {embedding_dimension(embedding)}")
print(f"Vector magnitude: {vector_magnitude(embedding):.6f}")
print(f"First 5 values: {[round(v, 4) for v in embedding[:5]]}")
```

这将产生类似于以下内容的输出：

```
Text: 'Machine learning transforms data into insights'
Embedding dimension: 384
Vector magnitude: 1.000000
First 5 values: [0.034, -0.0121, 0.0589, 0.0076, -0.045]
```

默认情况下，`embed` 函数使用本地的、基于哈希的方法，不需要外部依赖或 API 调用。这对于原型设计和测试很有用，因为它快速且确定。然而，它不会产生语义上有意义的向量。对于真正的语义搜索，您需要使用更完善的模型。

### 选择嵌入 (embedding)模型

`embed` 函数通过其 `model` 参数 (parameter)支持多种模型。您可以使用 `EmbeddingModel` 枚举来指定模型，这提供了一种便捷、类型安全的方式，从知名本地模型和基于 API 的模型中选择。

#### 使用 Sentence Transformers 的本地模型

对于许多应用，在本地运行嵌入模型是一个很好的选择。它兼顾了高质量嵌入、隐私保护和无 API 成本。该工具包与 `sentence-transformers` 库集成以支持此功能。

要使用 Sentence Transformers 模型，您必须首先安装必要的依赖项：

```shell
pip install sentence-transformers
```

安装后，您可以指定 `EmbeddingModel.ALL_MINILM_L6_V2` 等模型，这是一个流行的、性能均衡的选择。

```python
from kerb.embedding import embed, EmbeddingModel

# 此代码需要 'pip install sentence-transformers'
text = "Natural language processing enables AI to understand text"
st_embedding = embed(
    text,
    model=EmbeddingModel.ALL_MINILM_L6_V2
)

print(f"Sentence Transformers 模型生成了一个具有 {len(st_embedding)} 维度的嵌入。")
```

#### 来自 OpenAI 的基于 API 的模型

要获取最高质量的嵌入，您可以使用来自 OpenAI 等提供商的基于 API 的模型。这需要 API 密钥并按使用量收费，但通常能为语义检索提供最佳表现。

首先，请确保已安装 OpenAI 库：

```shell
pip install openai
```

您还需要将 OpenAI API 密钥设置为环境变量 (`OPENAI_API_KEY`)。然后，您可以选择 `EmbeddingModel.TEXT_EMBEDDING_3_SMALL` 等 OpenAI 模型。

```python
from kerb.embedding import embed, EmbeddingModel

# 此代码需要 'pip install openai' 和一个 API 密钥
text = "Natural language processing enables AI to understand text"
openai_embedding = embed(
    text,
    model=EmbeddingModel.TEXT_EMBEDDING_3_SMALL
)

print(f"OpenAI 模型生成了一个具有 {len(openai_embedding)} 维度的嵌入。")
```

不同的模型产生不同维度的嵌入。例如，`ALL_MINILM_L6_V2` 创建一个 384 维向量 (vector)，而 OpenAI 的 `TEXT_EMBEDDING_3_SMALL` 创建一个 1536 维向量。虽然较大的向量能捕捉更多细节，但它们也需要更多的存储和计算资源。

### 批量处理文档

在典型的 RAG 系统中，您需要嵌入 (embedding)数百或数千个文档片段。循环调用 `embed` 处理每个片段效率低下，尤其是在使用 GPU 加速的本地模型或进行 API 调用时。

`embed_batch` 函数专为此情景设计。它在一次优化调用中处理文本列表，显著提高性能。

```python
from kerb.embedding import embed_batch, EmbeddingModel

# 来自上一章的文档片段列表
document_chunks = [
    "Python is a high-level programming language",
    "Machine learning models learn patterns from data",
    "Natural language processing helps computers understand text",
    "Deep neural networks have multiple layers",
    "Data science combines statistics and programming"
]

# 一次性为所有片段生成嵌入
# 注意：此操作需要 'pip install sentence-transformers'
chunk_embeddings = embed_batch(
    document_chunks,
    model=EmbeddingModel.ALL_MINILM_L6_V2
)

print(f"生成了 {len(chunk_embeddings)} 个嵌入。")
print(f"每个嵌入有 {len(chunk_embeddings[0])} 个维度。")
```

使用 `embed_batch` 是为 RAG 系统准备知识库的标准做法。它确保您的整个文本片段语料库能高效地转换为向量 (vector)集合，准备好进行存储和检索。有了这些向量，您现在就可以进行数学比较以查找语义相关信息，我们将在下一节介绍。

## 参考资料

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/) — Nils Reimers and Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Publisher: Association for Computational Linguistics; Pages: 3982-3992; DOI: [10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)
  介绍了Sentence-BERT架构，用于生成具有语义意义的句子嵌入，是`sentence-transformers`库的基础。
- [Embeddings](https://platform.openai.com/docs/api-reference/embeddings) — OpenAI (2024)
  Publisher: OpenAI
  OpenAI嵌入模型的官方指南，包括`text-embedding-3-small`，及其使用实践。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Pages: Chapter 5: Embeddings
  一本广受推崇的自然语言处理教材，涵盖了向量语义和词/句子嵌入等基础概念。请参阅相关章节（例如，第6章）。

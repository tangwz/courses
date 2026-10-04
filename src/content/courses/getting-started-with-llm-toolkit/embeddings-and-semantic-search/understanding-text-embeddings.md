---
course: "getting-started-with-llm-toolkit"
chapter: "embeddings-and-semantic-search"
lesson: "understanding-text-embeddings"
sourceId: 7768
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-5-embeddings-and-semantic-search/understanding-text-embeddings"
title: "理解文本嵌入"
description: "获得关于文本嵌入是什么的解释，以及它们如何捕捉语义。"
order: 1
plots: ["plots/7768-0.json"]
sourceHash: "336c11be3930ad66b61eee5254cf75f48e17f4b59d9a68f29dbf05c80172ec29"
sourceCorrections: []
---

为了处理大量信息，文档通常被分成易于处理的单元。然而，这些单元最初只是原始文本。计算机要找到相关信息，需要一种方法来明白用户查询“如何让计算机思考”与描述“人工智能”的文本单元是有关联的。简单的关键词匹配将无法通过这项测试。文本嵌入 (embedding)在此发挥作用。

文本嵌入是文本的数值表示，通常以向量 (vector)形式存在，即浮点数列表。这些向量的重要之处在于它们捕捉了原始文本的语义和语境。嵌入模型是一种专门的神经网络 (neural network)，它处理一段文本并输出一个固定大小的密集向量 (dense vector)。

可以把它想象成给每个词或句子在高维“意义空间”中赋予一个独特的坐标。在这个空间中，意义相近的文本彼此靠近，而意义不同的文本则相距较远。例如，“car”、“automobile”和“vehicle”的向量会聚集在一起，而“banana”的向量则会处于该空间中一个完全不同的区域。

使用 Kerb 工具包，生成嵌入是一个简单的函数调用。`embed` 函数接收一个字符串文本并返回其向量表示。

```python
from kerb.embedding import embed, embedding_dimension

text = "Machine learning transforms data into insights"
embedding_vector = embed(text)

print(f"文本: '{text}'")
print(f"嵌入维度: {embedding_dimension(embedding_vector)}")
print(f"前5个值: {[round(v, 4) for v in embedding_vector[:5]]}")
```

运行此代码表明文本被转换成特定大小的向量。在此例中，默认模型生成一个384维向量。这意味着我们使用此模型处理的每段文本都将表示为384维空间中的一个点。

```text
文本: '机器学习将数据转化为价值'
嵌入维度: 384
前5个值: [0.0347, -0.0121, 0.0563, -0.0205, 0.0488]
```

不同的嵌入模型生成不同大小的向量，模型的选择会影响语义搜索的效果。例如，OpenAI 的 `text-embedding-3-small` 模型生成1536维向量，而像 `all-MiniLM-L6-v2` 这样的 Sentence Transformer 模型则生成384维向量。

### 意义向量 (vector)空间

可视化384维空间是不可能的，但我们可以将其概念投射到二维空间，以便把握其中的关联。在这个空间中，词语和短语根据它们从大量文本数据中学习到的关联进行组织。



![向量空间的二维视图](plots/7768-0.json)



> 相关术语群集在向量空间中聚集，从而可以对它们的含义进行数学比较。

在这个简化的空间中，与皇室相关的术语彼此靠近，而表示水果的术语则位于一个独立的群集中。这种空间布局使我们能够进行语义搜索。当用户查询“royal family”时，其向量会落在“皇室”群集中，我们就可以通过数学方法找到其最近的邻居，如“国王”和“王后”。

### 向量 (vector)的特性与归一化 (normalization)

嵌入 (embedding)向量主要有两个特性：它的**大小**（或长度）和**方向**。方向指向意义空间中的一个特定位置，而大小有时可以表示重要性或强度，尽管这因模型而异。

为了比较相似度，我们通常更关注两个向量之间的夹角，而非其大小。如果两个向量指向同一方向，那么它们对应的文本在语义上是相同的，无论其长度如何。为了使这种比较更简便和稳定，嵌入通常会进行**归一化**。归一化会缩放向量，使其大小变为1，从而成为一个“单位向量”。这个过程确保了当我们稍后计算相似度时，我们只比较向量的方向。

您可以使用 `embedding` 模块中提供的辅助函数来查看向量的特性。

```python
from kerb.embedding import embed, vector_magnitude, normalize_vector

vec = embed("嵌入将文本转换为数值向量")
print(f"原始向量大小: {vector_magnitude(vec):.6f}")

normalized_vec = normalize_vector(vec)
print(f"归一化向量大小: {vector_magnitude(normalized_vec):.6f}")
```

输出结果证实，归一化后，向量的大小为 1.0。

```text
原始向量大小: 1.000000
归一化向量大小: 1.000000
```

Kerb 的 `embed()` 和 `embed_batch()` 函数默认返回归一化向量，因此您通常无需手动执行此步骤。然而，理解归一化对于把握我们接下来要介绍的相似度指标如何作用于这些向量是很有帮助的。

文本块现已转换为有意义的数值向量，我们已准备好实现检索系统的核心逻辑：即衡量查询与文档之间的相似度。

## 参考资料

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/) — Nils Reimers and Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Publisher: Association for Computational Linguistics; Pages: 3982–3992; DOI: [10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)
  介绍了Sentence-BERT架构，用于优化语义相似性任务的先进句子嵌入。
- [OpenAI Embeddings API Reference](https://platform.openai.com/docs/api-reference/embeddings) — OpenAI (2024)
  Publisher: OpenAI
  提供了OpenAI文本嵌入模型的详细信息和使用说明，包括`text-embedding-3-small`。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Pearson
  对自然语言处理进行了全面介绍，包含文本嵌入和向量空间模型的基础概念（第4版草稿）。
- [Embedding Representations](https://developers.google.com/machine-learning/crash-course/embeddings) — Google Developers (2023)
  Publisher: Google
  提供了嵌入（embeddings）的概念性介绍，解释了它们如何表示含义及其在机器学习中的作用。

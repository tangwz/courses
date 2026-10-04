---
course: "getting-started-with-llm-toolkit"
chapter: "embeddings-and-semantic-search"
lesson: "choosing-embedding-model"
sourceId: 7772
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-5-embeddings-and-semantic-search/choosing-embedding-model"
title: "嵌入模型的选择"
description: "了解不同嵌入模型之间的权衡，以及如何根据您的使用场景进行选择。"
order: 5
plots: ["plots/7772-0.json"]
sourceHash: "47d6ada42d67868c908168aa0fdbefc9ff9ac3932960a0e6b2a5dff73a4a99a6"
sourceCorrections: []
---

选择合适的嵌入 (embedding)模型是构建任何检索系统的基本决定。这个选择直接影响您的应用程序的检索质量、运行成本、速度和基础设施要求。没有一个“最佳”模型；相反，最佳选择取决于针对您的特定应用场景而进行的一系列权衡。该工具包提供了到多种模型的统一接口，这些模型可分为三类。

### 嵌入 (embedding)模型的分类

通过 `embedding` 模块可用的模型分为三个不同层次，每个都有不同作用。

1. **本地基于哈希的模型：** 这些模型通过 `EmbeddingModel.LOCAL` 访问，不依赖机器学习 (machine learning)。它们通过对输入文本进行哈希处理来生成确定性向量 (vector)。它们不需要外部依赖或下载，速度极快，这使它们非常适合于测试和原型开发，在这些场景中不需要真正的语义理解。
2. **本地机器学习模型（Sentence Transformers）：** 这些是像 `all-MiniLM-L6-v2` 这样的开源模型，您可以在自己的基础设施上下载和运行。它们在质量和成本之间取得良好平衡（因为它们免费使用，只需支付计算成本）。对于需要数据隐私、离线能力或无需API相关网络延迟即可获得可预测性能的应用，它们是很好的选择。
3. **云端API模型（OpenAI）：** 这些是先进的专有模型，例如 `text-embedding-3-large`，通过API访问。它们通常提供可用的最高质量嵌入，但带有按用量计费的定价和对互联网连接的依赖。它们适用于以顶尖性能为首要考量的生产应用。

### 选择模型时的权衡

您的决定应由四个主要因素决定：质量、成本、速度和基础设施。

#### 质量

嵌入 (embedding)质量是指模型的输出向量 (vector)能多大程度地把握文本的语义。高质量嵌入会将相似概念在向量空间中靠近放置，这直接转化为更相关的搜索结果。像大规模文本嵌入基准（MTEB）排行榜这样的公共基准是比较不同模型在各项工作上表现的良好参考。通常，像OpenAI的 `text-embedding-3-large` 这样更大、更现代的模型，质量排名最高。

#### 成本与基础设施

成本在模型类型之间显著不同。

- **云端API模型**有直接的、按token计费的成本。虽然易于上手，但高用量会使成本累积。
- **本地机器学习 (machine learning)模型**免费使用，但需要您管理计算基础设施。这可能涉及使用一台带有GPU的机器以获得可接受的性能，这会带来其自身的额外成本。
- **本地基于哈希的模型**完全免费，在标准Python环境中没有基础设施要求。

#### 速度（延迟）

延迟是生成嵌入所需的时间。对于实时应用，例如需要即时进行RAG查询的聊天机器人，低延迟举足轻重。

- **本地模型**（基于哈希和机器学习）通常非常快，因为它们在您的硬件上运行，没有网络开销。
- **API模型**会引入网络延迟，尽管提供商会优化速度。批处理请求，如 `embed_batch` 所示，是提升API模型吞吐量 (throughput)的有效方法。

#### 维度

每个嵌入模型都生成固定大小或维度的向量。例如：

- `all-MiniLM-L6-v2`：384 维度
- `text-embedding-3-small`：1536 维度
- `text-embedding-3-large`：3072 维度

更高的维度可以包含更详细的信息，通常与更高的质量相关。然而，它伴随着权衡：更大的向量需要更多存储空间，并可能增加向量数据库中相似性搜索的计算成本和时间。

下图展示了一些常用模型的嵌入质量与成本之间的关系。



![嵌入模型的权衡：质量与成本](plots/7772-0.json)



> 此图根据公共基准的相对质量分数和处理一百万token的成本绘制。本地模型的成本为零。

### 实用建议与示例

以下是根据您的应用程序需求选择和使用模型的方法。

#### 针对测试与原型开发

当您构建单元测试或开发应用程序的初始结构时，语义准确性通常不如速度和简洁性那么要紧。默认的本地哈希模型非常适合此用途。它无需设置且是确定性的。

```python
from kerb.embedding import embed, EmbeddingModel

# 无需依赖或API密钥
test_vector = embed(
    "这是一个单元测试。", 
    model=EmbeddingModel.LOCAL
)

print(f"生成了一个 {len(test_vector)} 维向量用于测试。")
```

#### 针对经济高效或注重隐私的应用

如果您正在构建一个预算是主要顾虑的应用程序，或者由于隐私原因数据不能发送给第三方API，Sentence Transformers 是最佳选择。它们完全在您自己的基础设施上运行。作为好的起点，`all-MiniLM-L6-v2` 在速度和质量之间提供了良好平衡。

要使用 Sentence Transformer 模型，您首先需要安装所需的库：
`pip install sentence-transformers`

```python
from kerb.embedding import embed, EmbeddingModel

# 这将在首次使用时下载模型并在本地运行
minilm_vector = embed(
    "一个用于经济高效RAG系统的文档。",
    model=EmbeddingModel.ALL_MINILM_L6_V2
)

print(f"MiniLM向量维度：{len(minilm_vector)}")
```

#### 针对高性能生产系统

当实现尽可能高的检索质量是主要目标时，云端API模型是推荐选项。OpenAI的模型直接集成，并提供先进的性能。`text-embedding-3-small` 模型是一个很好的、经济高效的起点，而 `text-embedding-3-large` 为要求更高的工作提供最佳质量。

要使用 OpenAI 模型，您需要 `openai` 库和一个 API 密钥。
`pip install openai`

```python
import os
from kerb.embedding import embed, EmbeddingModel

# 假设 OPENAI_API_KEY 已设置为环境变量
# 或者您可以通过 api_key 参数直接传递
openai_vector = embed(
    "一个用于高性能生产RAG系统的查询。",
    model=EmbeddingModel.TEXT_EMBEDDING_3_SMALL,
    api_key=os.getenv("OPENAI_API_KEY") 
)

print(f"OpenAI向量维度：{len(openai_vector)}")
```

### 决策框架

为帮助您决策，请遵循这个简单的决策过程。

> 一个决策流程图，根据项目需求指导您选择嵌入 (embedding)模型。

最终，选择模型的最佳方法是进行实验。将此建议作为起点，但请考虑对您实际数据的小样本进行自己的评估，以查看哪个模型能为您的特定领域和查询模式提供最佳结果。

## 参考资料

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) — Nils Reimers and Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Pages: 3972-3982; DOI: [10.48550/arXiv.1908.10084](https://doi.org/10.48550/arXiv.1908.10084)
  解释本地机器学习嵌入模型架构和训练方法的奠基性论文。
- [Sentence Transformers Documentation](https://www.sbert.net/) — UKPLab, Hugging Face, and the Sentence Transformers community (2024)
  Publisher: Hugging Face
  Sentence Transformer模型使用的全面实践指南和文档，涵盖各种模型及其应用。

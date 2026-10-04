# 实践：构建一个简单的RAG应用

来源：[原文](https://apxml.com/zh/courses/python-llm-workflows/chapter-7-building-rag-systems/practice-simple-rag-application)

[返回章节目录](README.md) · [返回课程目录](../README.md)

构建一个基本的检索增强生成 (RAG)（Retrieval-Augmented Generation, RAG）应用，需使用Python和LlamaIndex，它将大型语言模型（LLM）与外部数据源相结合。构建RAG应用的过程包括摄取少量文本数据，使用嵌入 (embedding)对其进行索引，然后使用大型语言模型进行查询，先获取相关背景信息。向量 (vector)存储在此过程中促进了大型语言模型与数据的连接。

### 目标

创建一个简单的问答应用，使用RAG根据一组给定的文本文档来回答问题。

### 准备工作

开始之前，请确保已安装所需的库。你主要需要LlamaIndex和一个大型语言模型提供商库（如`openai`）。你还需要一个用于向量 (vector)存储组件的库；我们在这里将使用FAISS，它需要`faiss-cpu`包（如果你有兼容的GPU并安装了CUDA，则需要`faiss-gpu`）。

```bash
pip install llama-index openai faiss-cpu python-dotenv
```

请记住安全地设置你的API密钥，例如，使用环境变量和`python-dotenv`库，如第2章所述。本例中，我们假设你的OpenAI API密钥可通过名为`OPENAI_API_KEY`的环境变量访问。

### 第一步：设置和导入

首先导入LlamaIndex所需的组件并配置你的环境。

```python
import os
import logging
import sys
from dotenv import load_dotenv

# 加载环境变量（特别是OPENAI_API_KEY）
load_dotenv()

# 可选：配置日志以提高可见性
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Document
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding # 或使用其他嵌入模型
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI # 或使用其他大型语言模型

import faiss # 向量存储库

# 检查API密钥是否可用
if os.getenv("OPENAI_API_KEY") is None:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

print("设置完成。库已导入，API密钥已加载。")
```

此代码导入了LlamaIndex的核心类、FAISS向量 (vector)存储集成、OpenAI嵌入 (embedding)和大型语言模型类，并检查所需的API密钥。

### 第二步：准备示例数据

对于一个简单的RAG系统，我们需要一些可供查询的数据。我们不从文件加载数据，而是直接将一些文本片段定义为LlamaIndex的`Document`对象。这使得示例更易于独立运行。

```python
# 创建示例文档对象
text1 = """
生成式预训练Transformer 3 (GPT-3) 是一种自回归语言模型，
于2020年发布，使用深度学习生成类人文本。
给定一个初始文本作为提示，它会生成继续该提示的文本。
"""

text2 = """
检索增强生成（RAG）通过整合外部知识来增强大型语言模型（LLMs）。
在生成回答之前，RAG模型从预定义的知识源，如文档集合或数据库中获取相关信息。
然后，这种获取到的背景信息被用来指导和奠定生成过程，
从而产生更准确和真实的回答。
"""

text3 = """
向量嵌入将文本（或其他数据类型）表示为高维空间中的数值向量。
相似的思想或文本被映射到该空间中相近的点。
这使得语义搜索更为高效，查询能够根据含义而不是仅仅基于关键词匹配来寻找文档。
这些嵌入对于RAG系统中的检索步骤很重要。
"""

documents = [
    Document(text=text1, doc_id="doc_gpt3"),
    Document(text=text2, doc_id="doc_rag"),
    Document(text=text3, doc_id="doc_embeddings")
]

print(f"已创建 {len(documents)} 个示例文档。")
```

我们创建了三段与大型语言模型、RAG和嵌入 (embedding)相关的不同文本，并将每段文本包装成一个`Document`对象。分配`doc_id`是追踪来源的良好做法。

### 第三步：初始化嵌入 (embedding)模型和大型语言模型

我们需要指定使用哪个嵌入模型将文本转换为向量 (vector)，以及使用哪个大型语言模型来生成最终回答。LlamaIndex集成了各种提供商；这里我们使用OpenAI。

```python
# 初始化嵌入模型
embed_model = OpenAIEmbedding()

# 初始化大型语言模型
llm = OpenAI(model="gpt-3.5-turbo") # 或者选择其他模型，如gpt-4

print("已初始化OpenAI嵌入模型和大型语言模型。")
```

### 第四步：设置向量 (vector)存储

现在，让我们创建一个我们选择的向量存储FAISS的实例。我们定义向量的维度，这取决于所使用的嵌入 (embedding)模型（OpenAI的`text-embedding-ada-002`，作为`OpenAIEmbedding`的默认模型，生成1536维向量）。

```python
# OpenAI ada-002向量的维度
d = 1536
faiss_index = faiss.IndexFlatL2(d) # 使用L2距离进行相似度计算

# 实例化FaissVectorStore
vector_store = FaissVectorStore(faiss_index=faiss_index)

print("FAISS向量存储已初始化。")
```

我们创建一个基本的FAISS索引（`IndexFlatL2`），它适用于可以进行详尽搜索的小型数据集。`IndexFlatL2`计算查询向量和所有已索引向量之间的L2（欧几里得）距离，以找到最近的邻居。

### 第五步：创建索引

文档、嵌入 (embedding)模型和向量 (vector)存储都准备就绪后，我们就可以创建索引了。LlamaIndex处理文档分块（如果需要）、为每个块生成嵌入以及将它们存储到向量存储中的过程。我们还将定义一个存储上下文 (context)以链接向量存储。

```python
# 定义一个使用我们FAISS向量存储的存储上下文
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 定义一个文本分割器（可选但推荐）
# 这有助于在需要时分解更大的文档
node_parser = SentenceSplitter(chunk_size=100, chunk_overlap=20)

# 构建索引
# 此过程包括：
# 1. 将文档解析为节点（块）
# 2. 使用embed_model为每个节点生成嵌入
# 3. 将节点及其嵌入存储在vector_store中
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
    node_parser=node_parser, # 使用已定义的解析器
    llm=llm # 关联大型语言模型以进行潜在的索引时操作
)

print("索引已创建，数据已嵌入到FAISS中。")

# 可选：将索引保存到磁盘以备后用
# index.storage_context.persist("./my_rag_index")
# print("索引已保存到磁盘。")

# 可选：如果索引存在，则从磁盘加载
# try:
#     storage_context = StorageContext.from_defaults(
#         vector_store=vector_store, persist_dir="./my_rag_index"
#     )
#     index = load_index_from_storage(storage_context, embed_model=embed_model, llm=llm)
#     print("索引已从磁盘加载。")
# except FileNotFoundError:
#     print("磁盘上未找到索引，正在创建新索引。")
#     # （如上所示构建索引的代码）
#     index.storage_context.persist("./my_rag_index")
```

这里，`VectorStoreIndex.from_documents`是核心函数。它接收我们的`Document`对象列表，通过`embed_model`协调嵌入的生成，使用`node_parser`将文本分割成可管理的块（节点），并将结果存储在`storage_context`中定义的`vector_store`中。在索引时关联大型语言模型可能用于某些高级索引策略，但对于这个基本设置并非严格必需。我们还展示了用于持久化和重新加载索引的注释代码，这对于索引耗时较长的大型数据集很有用。

### 第六步：创建查询引擎

为了与已索引的数据交互，LlamaIndex提供了查询引擎。一个基本的查询引擎会根据查询从索引中获取相关背景信息，然后将查询和背景信息传递给大型语言模型进行合成。

```python
# 从索引创建查询引擎
# similarity_top_k=2 表示获取最相似的2个节点
query_engine = index.as_query_engine(similarity_top_k=2, llm=llm)

print("查询引擎已创建。")
```

`as_query_engine()`是索引对象上的一个便捷方法。我们指定`similarity_top_k=2`以获取向量 (vector)存储中与每个查询最相关的两个文本块（节点）。大型语言模型实例再次传递，用于最终的答案生成步骤。

### 第七步：查询数据

最后，让我们提出一个与已索引文档相关的问题。

```python
# 定义一个查询
query_text = "RAG如何改进大型语言模型的回答？"

# 执行查询
response = query_engine.query(query_text)

# 打印回答
print("\n查询：", query_text)
print("\n回答：")
print(response) # 大型语言模型合成的回答

# 可选：查看获取到的源节点
# print("\n源节点：")
# for node in response.source_nodes:
#     print(f"  得分: {node.score:.4f}")
#     print(f"  内容: {node.get_content().strip()}")
#     print("-" * 20)
```

`query_engine.query()`方法执行RAG过程：

1. 它接收`query_text`。
2. 使用`embed_model`为查询生成嵌入 (embedding)。
3. 在`vector_store` (FAISS) 中搜索与查询嵌入最接近的`similarity_top_k`个节点。
4. 构建一个包含原始`query_text`和获取到的节点内容的新提示。
5. 将这个增强的提示发送给大型语言模型。
6. 返回大型语言模型生成的回答。

### 预期结果

当你运行整个脚本时，你将看到类似这样的输出（大型语言模型的具体回答措辞可能略有不同）：

```
设置完成。库已导入，API密钥已加载。
已创建 3 个示例文档。
已初始化OpenAI嵌入模型和大型语言模型。
FAISS向量存储已初始化。
索引已创建，数据已嵌入到FAISS中。
查询引擎已创建。

查询：RAG如何改进大型语言模型的回答？

回答：
检索增强生成（RAG）通过整合外部知识来增强大型语言模型（LLMs）。在生成回答之前，RAG模型从预定义的知识源，如文档集合或数据库中获取相关信息。然后，这种获取到的背景信息被用来指导和奠定生成过程，从而产生更准确和真实的回答。
```

如果你取消注释打印源节点的代码，你将看到从文档中获取到的具体文本块（可能是`doc_rag`的内容，也可能根据相似度得分包含`doc_embeddings`），大型语言模型使用这些文本块来组织其回答。

### 总结

在此实践环节中，你成功构建了一个使用Python、LlamaIndex、OpenAI和FAISS的基本RAG流程。你摄取了文本数据，创建了向量 (vector)嵌入 (embedding)，将它们存储在向量数据库中，并使用查询引擎获取相关背景信息，然后从大型语言模型生成一个基于信息的回答。这体现了RAG的核心流程：先获取相关信息，然后根据这些信息生成回答。你可以通过更改数据源、嵌入模型、向量存储或大型语言模型来调整此模式，以构建更复杂的RAG应用。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: NeurIPS 2020; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  本文介绍了RAG框架，结合了预训练参数化和非参数化记忆，以实现更好的开放域问答。
- [LlamaIndex Documentation](https://docs.llamaindex.ai/en/stable/) — LlamaIndex (2024)
  LlamaIndex的官方文档，提供了构建集成外部数据（包括RAG系统）的LLM应用的指南和API参考。
- [OpenAI API Documentation](https://platform.openai.com/docs) — OpenAI (2024)
  Publisher: OpenAI
  OpenAI语言模型和嵌入服务的官方文档，提供了将它们集成到RAG应用所需的信息。

---

[上一节](06-%E8%AF%84%E4%BC%B0%20RAG%20%E6%80%A7%E8%83%BD%E6%8C%87%E6%A0%87.md) · [下一节](../08-Python%20%E4%B8%AD%E7%9A%84%E6%8F%90%E7%A4%BA%E5%B7%A5%E7%A8%8B%E6%8A%80%E6%9C%AF/01-%E6%9C%89%E6%95%88%E6%8F%90%E7%A4%BA%E7%9A%84%E5%8E%9F%E5%88%99.md)
